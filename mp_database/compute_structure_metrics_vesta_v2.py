#!/usr/bin/env python3
"""Compute explicitly versioned VESTA-v2 structure metrics.

This is the discovery-v2 structure-metrics implementation.  It must not be
used to regenerate the unprefixed ``dim/st1/st2/st3`` columns consumed by the
historical ExtraTrees model: the archived historical implementation used
CrystalNN.  This entry point emits only ``vesta__*`` metric names and writes
the bonding-rule identity and cutoff-table hash beside every output.

Definitions
-----------
dim
    Larsen dimensionality of the periodic M-X graph.
st1 / Xcn
    Mean number of VESTA-bonded X neighbors over M centres with CN >= 3.
st2 / Xsh
    Mean number of shared X atoms over connected periodic polyhedron-pair
    orbits.
st3 / Pcn
    Mean periodic polyhedron-graph degree over all CN >= 3 M centres,
    including isolated polyhedra.

All periodic image vectors are retained.  There is no CrystalNN, Gorai,
distance-shell, or supercell fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure

try:
    from src.discovery.bonding_vesta import (
        PeriodicMXGraph,
        build_vesta_mx_graph,
        vesta_rule_provenance,
    )
except ModuleNotFoundError:  # direct execution from src/screening
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "discovery"))
    from bonding_vesta import (  # type: ignore[no-redef]
        PeriodicMXGraph,
        build_vesta_mx_graph,
        vesta_rule_provenance,
    )


MIN_NEIGHBORS_FOR_POLYHEDRON = 3
SUPPORTED_SUFFIXES = (".cif", ".poscar", ".vasp")
NO_MX_BOND_ERROR = (
    "VESTA-2019 pair is defined but identifies no M-X bond; "
    "zero bonds cannot be interpreted as a physical 0D network"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dimension(structure: Structure) -> str:
    """Compatibility helper backed only by the VESTA periodic graph."""

    graph = build_vesta_mx_graph(structure)
    if not graph.bonds:
        raise ValueError(NO_MX_BOND_ERROR)
    return f"{graph.dimensionality()}D"


def serialize_polyhedra(graph: PeriodicMXGraph) -> list[dict[str, Any]]:
    grouped = graph.bonds_by_center()
    metrics = graph.polyhedron_metrics(
        minimum_cn=MIN_NEIGHBORS_FOR_POLYHEDRON
    )
    degree_map = {int(key): int(value) for key, value in metrics["degrees"].items()}
    out: list[dict[str, Any]] = []
    for pid, center_index in enumerate(metrics["eligible_center_indices"]):
        bonds = grouped[center_index]
        out.append(
            {
                "id": pid,
                "center_index": int(center_index),
                "center_symbol": graph.center_symbol,
                "coordination_number": len(bonds),
                "neighbors": [
                    {
                        "halogen_index": int(bond.halogen_index),
                        "image": list(bond.image),
                        "distance_A": float(bond.distance_A),
                    }
                    for bond in bonds
                ],
                "degree": degree_map.get(center_index, 0),
            }
        )
    return out


def compute_metrics(structure: Structure) -> tuple[PeriodicMXGraph, dict[str, Any]]:
    graph = build_vesta_mx_graph(structure)
    if not graph.bonds:
        raise ValueError(NO_MX_BOND_ERROR)
    metrics = graph.polyhedron_metrics(
        minimum_cn=MIN_NEIGHBORS_FOR_POLYHEDRON
    )
    return graph, metrics


def process_file(filepath: str, input_root: str, output_root: str) -> dict[str, Any]:
    path = Path(filepath)
    root = Path(input_root)
    out_root = Path(output_root)
    rel = path.relative_to(root)
    try:
        structure = Structure.from_file(path)
        graph, metrics = compute_metrics(structure)
        polyhedra = serialize_polyhedra(graph)
        category = "polyhedra" if polyhedra else "no_polyhedra"
        outdir = out_root / category / rel.parent
        outdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, outdir / path.name)
        payload = {
            "schema_version": 3,
            "structure_name": path.stem,
            "source_cif_sha256": sha256_file(path),
            "bonding": vesta_rule_provenance(verify=True),
            "center_symbol": graph.center_symbol,
            "halogen_symbol": graph.halogen_symbol,
            "vesta__mx_cutoff_A": graph.cutoff_A,
            "vesta__dim": int(metrics["dim"]),
            "vesta__Xcn": metrics["st1"],
            "vesta__Xsh": metrics["st2"],
            "vesta__Pcn": metrics["st3"],
            "n_polyhedra": metrics["n_polyhedra"],
            "shared_pair_counts": metrics["shared_pair_counts"],
            "polyhedra": polyhedra,
        }
        with (outdir / f"{path.name}.vesta_v2.poly.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(
            f"[OK] {rel} dim={metrics['dim']}D "
            f"Xcn={metrics['st1']:.3f} Xsh={metrics['st2']:.3f} "
            f"Pcn={metrics['st3']:.3f}"
        )
        return {
            "filename": str(rel),
            "vesta__dim": int(metrics["dim"]),
            "vesta__Xcn": metrics["st1"],
            "vesta__Xsh": metrics["st2"],
            "vesta__Pcn": metrics["st3"],
            "vesta__mx_cutoff_A": graph.cutoff_A,
            "feature_status": "ok",
            "feature_error": "",
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {rel}: {message}")
        return {
            "filename": str(rel),
            "vesta__dim": np.nan,
            "vesta__Xcn": np.nan,
            "vesta__Xsh": np.nan,
            "vesta__Pcn": np.nan,
            "vesta__mx_cutoff_A": np.nan,
            "feature_status": "error",
            "feature_error": message,
        }


def process_all_files(
    input_root: str, output_root: str, workers: int | None = None
) -> list[dict[str, Any]]:
    root = Path(input_root)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"no supported structure files below {root}")
    n_workers = min(workers or cpu_count(), len(files))
    args = [(str(path), str(root), str(output_root)) for path in files]
    if n_workers == 1:
        return [process_file(*values) for values in args]
    chunksize = max(1, len(files) // (8 * n_workers))
    with Pool(processes=n_workers) as pool:
        return pool.starmap(process_file, args, chunksize=chunksize)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="directory containing structures")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--supercell",
        default=None,
        help="deprecated compatibility argument; periodic image edges make expansion unnecessary",
    )
    args = parser.parse_args()

    if args.supercell:
        print(
            "[INFO] --supercell is ignored: discovery v2 retains exact periodic "
            "image edges in the input cell."
        )
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    bonding = vesta_rule_provenance(verify=True)
    rows = process_all_files(args.input, args.output, args.workers)
    frame = pd.DataFrame(rows).sort_values("filename")
    metrics_path = out_root / "structure_metrics_vesta_v2.csv"
    frame.to_csv(metrics_path, index=False)
    provenance = {
        "schema_version": 3,
        "script_sha256": sha256_file(Path(__file__)),
        "input_root": str(Path(args.input).resolve()),
        "n_structures": len(frame),
        "n_errors": int(frame["feature_status"].ne("ok").sum()),
        "minimum_cn_for_polyhedron": MIN_NEIGHBORS_FOR_POLYHEDRON,
        "bonding": bonding,
        "supercell_expansion": None,
        "metric_namespace": "vesta__",
        "legacy_extratrees_compatible": False,
        "output_csv": metrics_path.name,
    }
    with (out_root / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False)
    print(f"[OK] wrote {metrics_path} ({len(frame)} structures)")


if __name__ == "__main__":
    main()
