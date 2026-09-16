#!/usr/bin/env python3
"""Select a one-row-per-CIF table onto a structure-deduplicated representative pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_FAMILY_COLUMNS = [
    "structure_family_id",
    "structure_family_size",
    "topology_guard_bucket",
    "topology_signature",
    "family_contains_train_cif",
    "family_contains_train_formula",
    "family_contains_train_group_key",
    "n_train_cifs_in_family",
    "representative_is_train_cif",
    "seen_train_structure_family",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representatives", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()

    representatives = pd.read_csv(args.representatives, low_memory=False)
    table = pd.read_csv(args.input, low_memory=False)
    for label, frame in [("representatives", representatives), ("input", table)]:
        if "cif_file" not in frame:
            raise ValueError(f"{label} table lacks cif_file")
        if frame["cif_file"].duplicated().any():
            raise ValueError(f"{label} cif_file values must be unique")
    if "is_structure_representative" in representatives and not representatives[
        "is_structure_representative"
    ].astype(bool).all():
        raise ValueError("representatives table contains a non-representative row")
    missing = sorted(
        set(representatives["cif_file"].astype(str))
        - set(table["cif_file"].astype(str))
    )
    if missing:
        raise ValueError(f"input table lacks {len(missing)} representatives: {missing[:5]}")

    carry = [column for column in DEFAULT_FAMILY_COLUMNS if column in representatives]
    overlap = [column for column in carry if column in table]
    if overlap:
        table = table.drop(columns=overlap)
    selected = representatives[["cif_file", "formula", *carry]].merge(
        table,
        on="cif_file",
        how="left",
        validate="one_to_one",
        suffixes=("_representative", ""),
        sort=False,
    )
    if "formula_representative" in selected:
        loaded = selected["formula"].astype(str)
        canonical = selected.pop("formula_representative").astype(str)
        mismatched = loaded.ne(canonical)
        if mismatched.any():
            examples = selected.loc[mismatched, "cif_file"].astype(str).head().tolist()
            raise ValueError(
                "input formula is not canonical for representative rows; "
                f"examples={examples}"
            )
        selected["formula"] = canonical
    elif "formula" not in selected:
        raise RuntimeError("canonical representative formula was lost during merge")

    preferred = [
        column
        for column in ["cif_file", "material_id", "formula", *carry]
        if column in selected
    ]
    selected = selected[[*preferred, *[c for c in selected if c not in preferred]]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)

    audit_path = args.audit or args.output.with_suffix(".summary.json")
    audit = {
        "schema_version": 1,
        "scope": "exact CIF-identity subset to structure representatives",
        "inputs": {
            str(args.representatives): sha256_file(args.representatives),
            str(args.input): sha256_file(args.input),
        },
        "counts": {
            "input_rows": int(len(table)),
            "representative_rows": int(len(representatives)),
            "output_rows": int(len(selected)),
            "family_columns_carried": int(len(carry)),
        },
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
