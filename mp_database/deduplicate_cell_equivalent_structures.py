#!/usr/bin/env python3
"""Deduplicate selected MP structures by strict cell representation equivalence."""
from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.io.cif import CifParser

from deduplicate_candidate_structures import _representative_key, sha256_file


class CellChoiceMatcher(StructureMatcher):
    """Allow changes of cell handedness without equating physical mirror images."""

    def _get_lattices(self, target_lattice, s, supercell_size=1):
        for lattice, matrix in super()._get_lattices(target_lattice, s, supercell_size):
            # A proper rigid rotation preserves the determinant of the cell.
            # The integer basis matrix itself may have either determinant sign.
            if np.linalg.det(lattice.matrix) * np.linalg.det(target_lattice.matrix) > 0:
                yield lattice, matrix


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def normalize(structure, config):
    # Do not idealize the lattice, round fractional sites, or rescale volume.
    primitive = structure.get_primitive_structure(
        tolerance=config["primitive_tolerance_angstrom"]
    )
    return primitive.get_reduced_structure(reduction_algo="niggli")


def compare(a, b, config, evidence=False):
    va, vb = a.volume / len(a), b.volume / len(b)
    volume_difference = abs(va - vb) / max(va, vb)
    info = {"equivalent": False, "volume_per_atom_relative_difference": volume_difference}
    if volume_difference > config["max_volume_per_atom_relative_difference"]:
        return {**info, "reason": "volume_per_atom_differs"}
    if max(len(a), len(b)) % min(len(a), len(b)):
        return {**info, "reason": "incompatible_primitive_site_counts"}
    if not config["proper_rotations_only"]:
        raise ValueError("Cell-choice equivalence must not merge physical mirror images")
    matcher = CellChoiceMatcher(**config["structure_matcher"])
    if not matcher.fit(a, b, symmetric=True, skip_structure_reduction=True):
        return {**info, "reason": "lattice_or_atomic_positions_differ"}
    # Record an explicit lattice/site transformation after strict primitive reduction.
    reference, other = (a, b) if len(a) >= len(b) else (b, a)
    transformation = matcher.get_transformation(reference, other)
    if transformation is None:
        raise RuntimeError("Fit passed but no cell transformation was returned")
    matrix, translation, mapping = transformation
    transformed = other.copy()
    transformed.make_supercell(matrix)
    transformed.translate_sites(range(len(transformed)), translation)
    if len(mapping) != len(reference) or any(index is None for index in mapping):
        raise RuntimeError("Cell equivalence requires a complete bijection of atomic sites")
    for i, j in enumerate(mapping):
        if reference[i].species != transformed[j].species:
            raise RuntimeError("Cell transformation changed atomic species or occupancy")
    aligned_lattice = Lattice.from_parameters(*(
        (np.array(reference.lattice.parameters) + np.array(transformed.lattice.parameters)) / 2
    ))
    delta = reference.frac_coords - transformed.frac_coords[mapping]
    delta -= np.round(delta)
    residual = delta @ aligned_lattice.matrix
    residual -= residual.mean(axis=0)
    distances = np.linalg.norm(residual, axis=1)
    maximum = float(distances.max())
    info.update(max_site_displacement_angstrom=maximum,
                rms_site_displacement_angstrom=float(np.sqrt(np.mean(distances ** 2))))
    if maximum > config["max_site_displacement_angstrom"]:
        return {**info, "reason": "absolute_site_tolerance_exceeded"}
    info.update(equivalent=True, reason="cell_equivalent")
    if evidence:
        info.update(
            transformation_reference="a" if reference is a else "b",
            supercell_matrix=np.asarray(matrix, dtype=int).tolist(),
            fractional_translation=np.asarray(translation).tolist(),
            site_mapping=[int(index) for index in mapping],
        )
    return info


def load_structure(row, cif_dir, config):
    path = cif_dir / row["cif_file"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parser = CifParser(path, frac_tolerance=0, site_tolerance=1e-8)
        parsed = parser.parse_structures(primitive=False)
        if len(parsed) != 1:
            raise ValueError(f"Expected one structure in {path}")
        original = parsed[0]
        primitive = normalize(original, config)
    formula = Composition(row["cif_file"].split("_mp-", 1)[0]).reduced_formula
    if original.composition.reduced_composition != Composition(formula).reduced_composition:
        raise ValueError(f"CIF composition disagrees with filename: {path}")
    if len(primitive) != len(original):
        check = compare(original, primitive, config)
        if not check["equivalent"]:
            raise RuntimeError(f"Primitive reduction failed strict equivalence validation: {path}: {check}")
    details = {
        "cif_file": row["cif_file"], "formula": formula,
        "structure_chemsys": "-".join(sorted(element.symbol for element in original.composition.elements)),
        "original_nsites": len(original), "primitive_nsites": len(primitive),
        "cell_multiplicity": len(original) // len(primitive),
        "volume_per_atom": original.volume / len(original),
        "warnings": " | ".join(sorted({str(w.message) for w in caught})),
        "sha256": sha256_file(path),
    }
    return details, primitive


def compare_task(i, j, a, b, config):
    return i, j, compare(a, b, config, evidence=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/materials_project_cell_equivalence_v1.json"))
    parser.add_argument("--scope", choices=["experimental", "all"], default="experimental",
                        help="all includes theoretical entries and CIFs with missing MP metadata")
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    if args.outdir is None:
        label = "experimental" if args.scope == "experimental" else "all"
        args.outdir = Path(f"results/screening/materials_project_{label}_cell_equivalence_v1")
    if args.outdir.exists():
        raise FileExistsError("Use a new output directory")
    config = json.loads(args.config.read_text())
    cif_dir = Path("data/candidates/materials_project/cif")
    metadata_path = cif_dir.parent / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    if not metadata["cif_file"].is_unique or metadata["cif_file"].isna().any():
        raise ValueError("Input CIF identities must be present and unique")
    if args.scope == "experimental":
        selected = metadata.loc[metadata["theoretical"].eq(False)].copy()
        if not selected["mp_metadata_status"].eq("ok").all():
            raise ValueError("Experimental inputs require valid MP metadata")
    else:
        selected = metadata.copy()
        if set(selected["cif_file"]) != {path.name for path in cif_dir.glob("*.cif")}:
            raise ValueError("All-inventory selection must cover the original CIF directory exactly")
    selected = selected.sort_values("cif_file").reset_index(drop=True)
    input_hashes = {str(path): sha256_file(path) for path in [metadata_path, args.config, Path(__file__)]}
    print(f"Parsing and strictly reducing {len(selected)} CIFs (scope={args.scope})", flush=True)
    loaded = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(load_structure)(row, cif_dir, config) for row in selected.to_dict("records")
    )
    structures = [structure for _, structure in loaded]
    inventory = pd.DataFrame([details for details, _ in loaded])
    inventory = selected.merge(inventory, on="cif_file", validate="one_to_one", sort=False)
    groups = list(inventory.groupby("formula", sort=True).groups.items())
    pairs = [(int(i), int(j)) for _, indices in groups for i, j in itertools.combinations(indices, 2)]
    quick_results, candidates = [], []
    for i, j in pairs:
        a, b = structures[i], structures[j]
        va, vb = a.volume / len(a), b.volume / len(b)
        dv = abs(va-vb) / max(va, vb)
        if dv > config["max_volume_per_atom_relative_difference"]:
            quick_results.append((i, j, {"equivalent": False, "reason": "volume_per_atom_differs",
                                        "volume_per_atom_relative_difference": dv}))
        elif max(len(a), len(b)) % min(len(a), len(b)):
            quick_results.append((i, j, {"equivalent": False, "reason": "incompatible_primitive_site_counts",
                                        "volume_per_atom_relative_difference": dv}))
        else:
            candidates.append((i, j))
    print(f"Same-formula pairs: {len(pairs)}; candidate cell comparisons: {len(candidates)}", flush=True)
    matched = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(compare_task)(i, j, structures[i], structures[j], config) for i, j in candidates
    )
    pair_results = sorted(quick_results + matched, key=lambda item: (item[0], item[1]))
    edges = {(i, j) for i, j, result in pair_results if result["equivalent"]}
    def linked(i, j):
        return i == j or tuple(sorted((i, j))) in edges
    families, assignment_rows = [], []
    for formula, indices in groups:
        formula_groups = []
        ordered = sorted(indices, key=lambda i: _representative_key(inventory.loc[i].to_dict()))
        for i in ordered:
            for group in formula_groups:
                if all(linked(i, j) for j in group):
                    group.append(i)
                    break
            else:
                formula_groups.append([i])
        for number, group in enumerate(formula_groups, 1):
            representative = min(group, key=lambda i: _representative_key(inventory.loc[i].to_dict()))
            rep = inventory.loc[representative]
            family_id = f"{formula}::cell-{number:04d}"
            for i in group:
                assignment_rows.append({**inventory.loc[i].to_dict(), "structure_family_id": family_id,
                    "structure_family_size": len(group), "is_structure_representative": i == representative,
                    "representative_cif_file": rep["cif_file"], "representative_material_id": rep["material_id"]})
            families.append({"structure_family_id": family_id, "formula": formula,
                "structure_family_size": len(group), "representative_cif_file": rep["cif_file"],
                "member_cif_files": ";".join(sorted(inventory.loc[i, "cif_file"] for i in group))})
    assignments = pd.DataFrame(assignment_rows).sort_values("cif_file")
    representatives = assignments.loc[assignments["is_structure_representative"]]
    args.outdir.mkdir(parents=True, exist_ok=False)
    assignments.to_csv(args.outdir / "structure_dedup_assignments.csv", index=False)
    representatives.to_csv(args.outdir / "representative_inventory.csv", index=False)
    pd.DataFrame(families).to_csv(args.outdir / "structure_families.csv", index=False)
    assignments.loc[~assignments["is_structure_representative"]].to_csv(args.outdir / "removed_duplicate_entries.csv", index=False)
    inventory.to_csv(args.outdir / "input_structure_inventory.csv", index=False)
    evidence = [{"cif_a": inventory.loc[i, "cif_file"], "cif_b": inventory.loc[j, "cif_file"], **result}
                for i, j, result in pair_results if result["equivalent"]]
    write_json(args.outdir / "equivalent_pair_transformations.json", evidence)
    audit = [{"cif_a": inventory.loc[i, "cif_file"], "cif_b": inventory.loc[j, "cif_file"],
              **{key: value for key, value in result.items() if not isinstance(value, list)}}
             for i, j, result in pair_results]
    pd.DataFrame(audit).to_csv(args.outdir / "pair_audit.csv", index=False)
    counts = []
    for h in ["F", "Cl", "Br", "I"]:
        before = int(assignments["structure_chemsys"].map(lambda s: h in s.split("-")).sum())
        after = int(representatives["structure_chemsys"].map(lambda s: h in s.split("-")).sum())
        counts.append({"halogen": h, "input_entries": before, "representatives": after, "removed_duplicates": before-after})
    pd.DataFrame(counts).to_csv(args.outdir / "counts_by_halogen.csv", index=False)
    archive_path = args.outdir / "representative_cifs.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for name in representatives["cif_file"]:
            archive.write(cif_dir / name, arcname=name)
    with ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(representatives) and archive.testzip() is None
    for path, digest in input_hashes.items():
        assert sha256_file(Path(path)) == digest
    for row in inventory.to_dict("records"):
        assert sha256_file(cif_dir / row["cif_file"]) == row["sha256"]
    assert len(assignments) == len(selected) and assignments["cif_file"].is_unique
    assert set(assignments["cif_file"]) == set(selected["cif_file"])
    if args.scope == "experimental":
        assert representatives["theoretical"].eq(False).all()
    index_by_cif = {row["cif_file"]: i for i, row in inventory.iterrows()}
    for _, group in assignments.groupby("structure_family_id"):
        ids = [index_by_cif[cif] for cif in group["cif_file"]]
        assert all(linked(i, j) for i, j in itertools.combinations(ids, 2))
        assert int(group["is_structure_representative"].sum()) == 1
    assigned_family = assignments.set_index("cif_file")["structure_family_id"]
    unmerged_edges = [item for item in evidence if assigned_family[item["cif_a"]] != assigned_family[item["cif_b"]]]
    write_json(args.outdir / "nontransitive_boundary_pairs.json", unmerged_edges)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "method": config,
        "scope": args.scope,
        "input_experimental_status": {
            "experimentally_observed": int(selected["theoretical"].eq(False).sum()),
            "theoretical": int(selected["theoretical"].eq(True).sum()),
            "unknown": int(selected["theoretical"].isna().sum()),
        },
        "input_entries": len(assignments), "representatives": len(representatives),
        "removed_duplicates": len(assignments)-len(representatives),
        "unique_formulas": inventory["formula"].nunique(), "same_formula_pairs": len(pairs),
        "candidate_pairs": len(candidates), "equivalent_pairs": len(edges),
        "nontransitive_boundary_pairs": len(unmerged_edges),
        "pair_reasons": dict(Counter(result["reason"] for _, _, result in pair_results)),
        "family_size_distribution": dict(Counter(family["structure_family_size"] for family in families)),
        "counts_by_halogen": counts, "input_hashes": input_hashes,
        "software": {name: importlib.metadata.version(name) for name in ["pymatgen", "numpy", "pandas", "joblib"]},
        "validation": "pass: complete selected input coverage, complete-link families, primitive reductions, species bijections, unchanged source hashes, ZIP CRC",
    }
    write_json(args.outdir / "deduplication_summary.json", summary)
    write_json(args.outdir / "output_hashes.json", {p.name: sha256_file(p) for p in sorted(args.outdir.iterdir())})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
