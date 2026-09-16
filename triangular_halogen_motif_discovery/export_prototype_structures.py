#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export selected multi-prototype source structures for physical inspection.

The ML representation is anonymized M/X, but interpretation must return to the
original chemistry. This script reads ``multiprototype_selected_prototypes.csv`` and
exports each prototype's *original* parent crystal as a POSCAR-format file with a
`.vasp` suffix. A manifest records the prototype-center site in the original structure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pymatgen.io.vasp import Poscar

from deep_study import load_dataset


def safe(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    out = Path(args.results_dir)
    src = out / "multiprototype_selected_prototypes.csv"
    if not src.exists():
        raise SystemExit("multiprototype_selected_prototypes.csv missing")

    df, scaled, raw = load_dataset(Path(args.cif_root), Path(args.metadata))
    protos = pd.read_csv(src)
    vdir = out / "prototype_structures_vasp"
    vdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in protos.iterrows():
        mpid = str(r.source_material_id)
        hit = df.index[df.material_id.astype(str) == mpid].tolist()
        if not hit:
            continue
        i = int(hit[0])
        st = raw[i]
        site_idx = int(r.source_site)
        if not (0 <= site_idx < len(st)):
            continue
        site = st[site_idx]
        fn = (
            f"{safe(r.config)}_P{int(r.prototype_no)}_"
            f"{safe(r.source_formula)}_{safe(mpid)}_site{site_idx}.vasp"
        )
        Poscar(st).write_file(vdir / fn)
        rows.append({
            "config": r.config,
            "prototype_no": int(r.prototype_no),
            "source_formula": r.source_formula,
            "source_material_id": mpid,
            "source_site": site_idx,
            "original_center_element": site.specie.symbol,
            "center_frac_a": float(site.frac_coords[0]),
            "center_frac_b": float(site.frac_coords[1]),
            "center_frac_c": float(site.frac_coords[2]),
            "n_atoms": len(st),
            "vasp_file": f"prototype_structures_vasp/{fn}",
        })

    pd.DataFrame(rows).to_csv(out / "prototype_structure_manifest.csv", index=False)
    readme = [
        "# Prototype structures\n\n",
        "These `.vasp` files are the original chemical parent crystals from which the positive-only facility-location SOAP prototypes were selected. The prototype center is identified in `prototype_structure_manifest.csv` by the original zero-based site index and fractional coordinates.\n\n",
        "The full parent structure is exported intentionally: SOAP discovery used the full periodic M-X framework, and physical interpretation should inspect the surrounding connectivity rather than an artificially cut finite cluster.\n",
    ]
    (vdir / "README.md").write_text("".join(readme), encoding="utf-8")
    print(f"Exported {len(rows)} prototype parent structures to {vdir}")


if __name__ == "__main__":
    main()
