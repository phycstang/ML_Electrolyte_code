#!/usr/bin/env python3
"""Align an engineered candidate table with a trained model's feature schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = ("formula", "cif_file", "material_id")


def read_feature_names(columns_json: Path) -> list[str]:
    payload = json.loads(columns_json.read_text(encoding="utf-8"))
    for key in ("numeric", "feature_columns", "columns_numeric"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return [str(name) for name in value]
    raise ValueError(
        f"{columns_json} does not contain numeric, feature_columns, or columns_numeric"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and order candidate columns to match a saved model"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--columns", required=True, type=Path,
                        help="columns.json written by train_extratrees.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="optional row-aligned CSV supplying cif_file/material_id lost during featurization",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="create absent model columns as NaN; the saved pipeline will median-impute them",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    if args.metadata is not None:
        metadata = pd.read_csv(args.metadata, low_memory=False)
        if len(metadata) != len(frame):
            raise ValueError(
                f"metadata rows ({len(metadata)}) do not match feature rows ({len(frame)})"
            )
        if "formula" in metadata and "formula" in frame:
            left = metadata["formula"].astype(str).str.strip().reset_index(drop=True)
            right = frame["formula"].astype(str).str.strip().reset_index(drop=True)
            mismatch = left.ne(right)
            if mismatch.any():
                first = int(np.flatnonzero(mismatch.to_numpy())[0])
                raise ValueError(
                    f"metadata/features formula mismatch at row {first}: "
                    f"{left.iloc[first]!r} != {right.iloc[first]!r}"
                )
        for name in ("cif_file", "material_id"):
            if name not in frame and name in metadata:
                frame[name] = metadata[name].to_numpy()
        if "material_id" not in frame and "cif_file" in frame:
            frame["material_id"] = frame["cif_file"].astype(str).str.extract(
                r"(mp-\d+)", expand=False
            )
    features = read_feature_names(args.columns)
    missing = [name for name in features if name not in frame.columns]
    if missing and not args.allow_missing:
        preview = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise KeyError(f"{len(missing)} required features are absent: {preview}{suffix}")

    for name in missing:
        frame[name] = np.nan

    identities = [name for name in IDENTITY_COLUMNS if name in frame.columns]
    ordered = identities + [name for name in features if name not in identities]
    output = frame.loc[:, ordered]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(
        f"rows={len(output)} model_features={len(features)} "
        f"metadata_columns={len(identities)} missing_filled={len(missing)}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
