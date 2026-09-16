#!/usr/bin/env python3
"""Flag or remove candidate halides containing configured risk-element groups."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


HALOGENS = {"F", "Cl", "Br", "I"}
ELEMENT_GROUPS = {
    "radioactive": {"Ac", "Np", "Pa", "Pm", "Pu", "Tc", "Th", "U"},
    "toxic": {"As", "Be", "Cd", "Hg", "Pb", "Tl"},
    "precious": {"Ag", "Au", "Ir", "Os", "Pd", "Pt", "Rh", "Ru"},
}
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")


def formula_from_row(row: pd.Series, formula_col: str, cif_col: str) -> str:
    if cif_col in row.index and pd.notna(row[cif_col]):
        name = Path(str(row[cif_col])).name
        if "_mp-" in name:
            return name.split("_mp-", 1)[0]
    if formula_col in row.index and pd.notna(row[formula_col]):
        return str(row[formula_col])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter binary-halide candidates by risk-element group"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--formula-col", default="formula")
    parser.add_argument("--cif-col", default="cif_file")
    parser.add_argument(
        "--exclude",
        default="radioactive,toxic,precious",
        help="comma-separated groups: radioactive,toxic,precious",
    )
    parser.add_argument(
        "--flag-only",
        action="store_true",
        help="retain every row and append risk columns instead of filtering",
    )
    args = parser.parse_args()

    selected = [name.strip() for name in args.exclude.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(ELEMENT_GROUPS))
    if unknown:
        raise ValueError(f"unknown element groups: {unknown}")

    frame = pd.read_csv(args.input, low_memory=False)
    formulas = frame.apply(
        formula_from_row, axis=1, formula_col=args.formula_col, cif_col=args.cif_col
    )
    non_halogen = formulas.map(
        lambda value: sorted(set(ELEMENT_RE.findall(value)) - HALOGENS)
    )

    for group, elements in ELEMENT_GROUPS.items():
        frame[f"risk_{group}"] = non_halogen.map(
            lambda found, allowed=elements: bool(set(found) & allowed)
        )
    frame["risk_elements"] = non_halogen.map(
        lambda found: ";".join(
            sorted(
                element
                for element in found
                if any(element in ELEMENT_GROUPS[group] for group in selected)
            )
        )
    )

    remove = frame[[f"risk_{group}" for group in selected]].any(axis=1)
    output = frame if args.flag_only else frame.loc[~remove].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    counts = {group: int(frame[f"risk_{group}"].sum()) for group in selected}
    print(f"input={len(frame)} removed={int(remove.sum())} output={len(output)}")
    print("group_counts=" + ", ".join(f"{key}:{value}" for key, value in counts.items()))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
