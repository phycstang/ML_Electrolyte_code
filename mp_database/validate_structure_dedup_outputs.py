#!/usr/bin/env python3
"""Validate structure-deduplication and representative-screening artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_inf(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.inf
    return number if math.isfinite(number) else math.inf


def deprecated_rank(value: Any) -> int:
    if pd.isna(value):
        return 1
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no"}:
            return 0
        if normalized in {"true", "1", "yes"}:
            return 2
        return 1
    return 2 if bool(value) else 0


def representative_key(row: pd.Series) -> tuple[int, float, float, str]:
    return (
        deprecated_rank(row.get("deprecated")),
        finite_or_inf(row.get("energy_above_hull")),
        finite_or_inf(row.get("formation_energy_per_atom")),
        str(row["cif_file"]),
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--families", required=True, type=Path)
    parser.add_argument("--representatives", required=True, type=Path)
    parser.add_argument("--cif-dir", required=True, type=Path)
    parser.add_argument("--model-input", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--risk-flags", required=True, type=Path)
    parser.add_argument("--audited-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cif-hashes-output", required=True, type=Path)
    args = parser.parse_args()

    assignments = pd.read_csv(args.assignments, low_memory=False)
    families = pd.read_csv(args.families, low_memory=False)
    representatives = pd.read_csv(args.representatives, low_memory=False)
    model_input = pd.read_csv(args.model_input, low_memory=False)
    scores = pd.read_csv(args.scores, low_memory=False)
    risk = pd.read_csv(args.risk_flags, low_memory=False)
    audited = pd.read_csv(args.audited_scores, low_memory=False)

    require(assignments["cif_file"].is_unique, "assignment CIFs are not unique")
    require(families["structure_family_id"].is_unique, "family IDs are not unique")
    require(representatives["cif_file"].is_unique, "representative CIFs are not unique")
    require(
        int(assignments["is_structure_representative"].astype(bool).sum())
        == len(representatives),
        "assignment representative count differs from representative table",
    )
    require(
        assignments["structure_family_id"].nunique() == len(families),
        "assignment family count differs from family table",
    )
    require(
        int(families["structure_family_size"].sum()) == len(assignments),
        "family sizes do not sum to assignment rows",
    )
    require(
        not assignments["structure_match_error"].fillna("").astype(str).ne("").any(),
        "parse or matcher errors are present",
    )

    policy_errors: list[str] = []
    for family_id, group in assignments.groupby("structure_family_id", sort=False):
        expected = min(
            (row for _, row in group.iterrows()), key=representative_key
        )["cif_file"]
        selected = group.loc[
            group["is_structure_representative"].astype(bool), "cif_file"
        ].tolist()
        if selected != [expected]:
            policy_errors.append(str(family_id))
    require(not policy_errors, f"representative policy failures: {policy_errors[:5]}")

    representative_cifs = set(representatives["cif_file"].astype(str))
    for label, frame in [
        ("model input", model_input),
        ("scores", scores),
        ("risk flags", risk),
        ("audited scores", audited),
    ]:
        require(frame["cif_file"].is_unique, f"{label} CIFs are not unique")
        require(
            set(frame["cif_file"].astype(str)) == representative_cifs,
            f"{label} CIF set differs from representative pool",
        )
    require(
        assignments["formula_corrected_from_loaded"].astype(bool).sum() == 2,
        "expected exactly two spreadsheet formula repairs",
    )
    require(
        int(audited["passes_element_rules"].astype(bool).sum())
        == int(
            (
                ~risk[["risk_radioactive", "risk_toxic", "risk_precious"]]
                .astype(bool)
                .any(axis=1)
            ).sum()
        ),
        "risk pass counts disagree",
    )
    sorted_scores = audited.sort_values(
        ["score", "material_id"], ascending=[False, True], kind="mergesort"
    )
    require(
        np.array_equal(
            sorted_scores["cif_file"].astype(str).to_numpy(),
            audited["cif_file"].astype(str).to_numpy(),
        ),
        "audited scores are not deterministically ranked",
    )
    require(
        np.array_equal(audited["rank_all"].to_numpy(), np.arange(1, len(audited) + 1)),
        "rank_all is not consecutive",
    )

    cif_hash_rows = []
    for cif_file in sorted(assignments["cif_file"].astype(str)):
        path = args.cif_dir / cif_file
        require(path.is_file(), f"missing CIF: {cif_file}")
        cif_hash_rows.append({"cif_file": cif_file, "sha256": sha256_file(path)})
    cif_hashes = pd.DataFrame(cif_hash_rows)
    args.cif_hashes_output.parent.mkdir(parents=True, exist_ok=True)
    cif_hashes.to_csv(args.cif_hashes_output, index=False)

    counts = {
        "input_structure_entries": int(len(assignments)),
        "representative_structures": int(len(representatives)),
        "removed_duplicate_entries": int(len(assignments) - len(representatives)),
        "duplicate_families": int((families["structure_family_size"] > 1).sum()),
        "parse_or_match_errors": 0,
        "radioactive_representatives": int(risk["risk_radioactive"].astype(bool).sum()),
        "toxic_representatives": int(risk["risk_toxic"].astype(bool).sum()),
        "precious_representatives": int(risk["risk_precious"].astype(bool).sum()),
        "risk_union_representatives": int(
            risk[["risk_radioactive", "risk_toxic", "risk_precious"]]
            .astype(bool)
            .any(axis=1)
            .sum()
        ),
        "passes_element_rules": int(audited["passes_element_rules"].astype(bool).sum()),
        "representative_is_training_cif": int(audited["seen_train_cif"].astype(bool).sum()),
        "families_containing_training_cif": int(
            audited["seen_train_structure_family"].astype(bool).sum()
        ),
        "element_eligible_unseen_training_family": int(
            audited["element_eligible_and_unseen_train_structure_family"]
            .astype(bool)
            .sum()
        ),
    }
    report = {
        "schema_version": 1,
        "status": "pass",
        "counts": counts,
        "input_hashes": {
            str(path): sha256_file(path)
            for path in [
                args.assignments,
                args.families,
                args.representatives,
                args.model_input,
                args.scores,
                args.risk_flags,
                args.audited_scores,
            ]
        },
        "cif_content_manifest": {
            "path": str(args.cif_hashes_output),
            "rows": int(len(cif_hashes)),
            "sha256": sha256_file(args.cif_hashes_output),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
