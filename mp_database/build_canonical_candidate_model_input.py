#!/usr/bin/env python3
"""Repair canonical candidate identities and rebuild corrupted model rows.

The historical Materials Project CSV passed through spreadsheet software, which
converted ``FeBr2`` and ``FeBr3`` to date-like strings.  Their composition
features were consequently left almost entirely missing.  This utility keeps
all unaffected rows byte-for-value equivalent, derives the canonical formula
from the CIF filename, and replaces every model feature in an affected row from
an independently generated table with the same canonical
``formula/dim/st1/st2/st3`` key.

No prediction or acceptance-target information is read by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition


STRUCTURE_COLUMNS = ["dim", "st1", "st2", "st3"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_formula_from_cif(value: Any) -> str:
    name = Path(str(value)).name
    if "_mp-" not in name:
        raise ValueError(f"CIF filename lacks '_mp-' delimiter: {name}")
    return str(Composition(name.split("_mp-", 1)[0]).reduced_formula)


def model_columns(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = payload.get("numeric")
    if not isinstance(columns, list) or not columns:
        raise ValueError("columns JSON must contain a non-empty numeric list")
    if len(columns) != len(set(columns)):
        raise ValueError("model feature names must be unique")
    return [str(value) for value in columns]


def normalized_key(frame: pd.DataFrame) -> pd.Series:
    values = frame[["formula", *STRUCTURE_COLUMNS]].copy()
    values["formula"] = values["formula"].map(
        lambda value: str(Composition(str(value)).reduced_formula)
    )
    for column in STRUCTURE_COLUMNS:
        numeric = pd.to_numeric(values[column], errors="coerce").round(8)
        values[column] = numeric.map(
            lambda value: "NA" if pd.isna(value) else format(float(value), ".8g")
        )
    return values.astype(str).agg("|".join, axis=1)


def equal_with_nan(left: pd.Series, right: pd.Series) -> bool:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.array_equal(a, b, equal_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--model-input", required=True, type=Path)
    parser.add_argument("--repair-source", required=True, type=Path)
    parser.add_argument("--columns", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()

    features = model_columns(args.columns)
    inventory = pd.read_csv(args.inventory, low_memory=False)
    historical = pd.read_csv(args.model_input, low_memory=False)
    repair = pd.read_csv(args.repair_source, low_memory=False)

    for label, frame, required in [
        (
            "inventory",
            inventory,
            {"cif_file", "formula", *STRUCTURE_COLUMNS},
        ),
        (
            "model input",
            historical,
            {"cif_file", "material_id", "formula", *features},
        ),
        (
            "repair source",
            repair,
            {"formula", *STRUCTURE_COLUMNS, *features},
        ),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} lacks required columns: {missing[:12]}")
    if inventory["cif_file"].duplicated().any():
        raise ValueError("inventory cif_file values must be unique")
    if historical["cif_file"].duplicated().any():
        raise ValueError("model-input cif_file values must be unique")
    if set(inventory["cif_file"].astype(str)) != set(
        historical["cif_file"].astype(str)
    ):
        raise ValueError("inventory and model input cover different CIF sets")

    inventory = inventory.copy()
    inventory["canonical_formula"] = inventory["cif_file"].map(
        canonical_formula_from_cif
    )
    historical = inventory[["cif_file", "canonical_formula", *STRUCTURE_COLUMNS]].merge(
        historical,
        on="cif_file",
        how="left",
        validate="one_to_one",
        suffixes=("_inventory", ""),
        sort=False,
    )
    for column in STRUCTURE_COLUMNS:
        left = pd.to_numeric(
            historical[f"{column}_inventory"], errors="coerce"
        ).to_numpy(dtype=float)
        right = pd.to_numeric(historical[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-8, equal_nan=True):
            raise ValueError(f"inventory/model-input mismatch in {column}")

    historical["formula_as_loaded"] = historical["formula"].astype(str)
    historical["formula"] = historical["canonical_formula"]
    affected = historical["formula_as_loaded"].ne(historical["formula"])
    repair = repair.copy()
    repair["formula"] = repair["formula"].map(
        lambda value: str(Composition(str(value)).reduced_formula)
    )
    repair["_repair_key"] = normalized_key(repair)
    historical["_repair_key"] = normalized_key(historical)

    repair_records: list[dict[str, Any]] = []
    for index in historical.index[affected]:
        key = str(historical.at[index, "_repair_key"])
        candidates = repair.loc[repair["_repair_key"].eq(key), features]
        if candidates.empty:
            raise ValueError(f"no repair-source row for canonical key {key}")
        distinct = candidates.drop_duplicates()
        if len(distinct) != 1:
            raise ValueError(
                f"repair source has {len(distinct)} feature vectors for key {key}"
            )
        replacement = distinct.iloc[0]
        before = historical.loc[index, features].copy()
        historical.loc[index, features] = replacement.to_numpy()
        changed = 0
        for feature in features:
            a = before[feature]
            b = replacement[feature]
            same = (pd.isna(a) and pd.isna(b)) or (
                not pd.isna(a)
                and not pd.isna(b)
                and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=0.0)
            )
            changed += int(not same)
        repair_records.append(
            {
                "cif_file": str(historical.at[index, "cif_file"]),
                "formula_as_loaded": str(
                    historical.at[index, "formula_as_loaded"]
                ),
                "canonical_formula": str(historical.at[index, "formula"]),
                "repair_key": key,
                "non_null_features_before": int(before.notna().sum()),
                "non_null_features_after": int(replacement.notna().sum()),
                "feature_values_changed": changed,
            }
        )

    unaffected = historical.index[~affected]
    original_unaffected = pd.read_csv(args.model_input, low_memory=False).set_index(
        "cif_file"
    ).loc[historical.loc[unaffected, "cif_file"], features]
    rebuilt_unaffected = historical.loc[unaffected, features]
    rebuilt_unaffected.index = historical.loc[unaffected, "cif_file"].astype(str)
    for feature in features:
        if not equal_with_nan(original_unaffected[feature], rebuilt_unaffected[feature]):
            raise RuntimeError(f"unaffected values changed in feature {feature}")

    identity = ["cif_file", "material_id", "formula"]
    output = historical[[*identity, *features]].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    audit = {
        "schema_version": 1,
        "scope": "canonical identity repair before representative-pool scoring",
        "inputs": {
            str(args.inventory): sha256_file(args.inventory),
            str(args.model_input): sha256_file(args.model_input),
            str(args.repair_source): sha256_file(args.repair_source),
            str(args.columns): sha256_file(args.columns),
        },
        "counts": {
            "rows": int(len(output)),
            "model_features": int(len(features)),
            "identity_repairs": int(affected.sum()),
            "unaffected_rows_preserved": int((~affected).sum()),
        },
        "repairs": repair_records,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.audit.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
