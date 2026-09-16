#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leave-two-positive-out (LTPO) robustness audit for the multi-prototype model.

This is a *post-selection audit*, not another model-selection stage.
The primary K is already frozen by LOPO in ``multiprototype_validation.py``.
LTPO asks a harder question: with only five of the seven known positives available,
how well do K=1..4 positive-only facility-location prototypes recover two unseen
known positives simultaneously?

Important safeguards
--------------------
- Blind formulas are never accessed here.
- Background formulas only calibrate percentiles; they are not negative training data.
- All 21 held-out pairs are enumerated deterministically.
- K is NOT re-selected from LTPO results.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deep_study import build_soap, load_dataset, local_soap_all
from formula_clean_final import formula_clean_split
from multiprototype_validation import (
    CONFIGS, KS, facility_select, bg_formula_distribution,
    structure_score, percentile,
)


def run_ltpo(df, structs, local_by_cfg, known, bg, out: Path):
    rows = []
    formulas = list(known)
    pairs = list(itertools.combinations(formulas, 2))

    for cfg_name, local in local_by_cfg.items():
        for pair_no, (f1, f2) in enumerate(pairs, 1):
            held = {f1: known[f1], f2: known[f2]}
            train = {f: i for f, i in known.items() if f not in held}
            for k in KS:
                protos, meta, hist = facility_select(local, train, k)
                bgs = bg_formula_distribution(local, protos, df, bg)
                pcts = []
                for f, i in held.items():
                    s = structure_score(local, i, protos)
                    p = percentile(s, bgs)
                    pcts.append(p)
                    rows.append({
                        "config": cfg_name,
                        "pair_no": pair_no,
                        "k": k,
                        "heldout_pair": f"{f1};{f2}",
                        "heldout_formula": f,
                        "heldout_material_id": str(df.loc[i, "material_id"]),
                        "heldout_score": s,
                        "heldout_formula_background_percentile": p,
                        "pair_min_percentile": np.nan,
                        "pair_mean_percentile": np.nan,
                        "prototype_source_formulas": ";".join(m[0] for m in meta),
                        "prototype_source_sites": ";".join(str(m[2]) for m in meta),
                        "train_min_coverage": hist[-1]["min_train_coverage"],
                        "train_mean_coverage": hist[-1]["mean_train_coverage"],
                    })
                for r in rows[-2:]:
                    r["pair_min_percentile"] = float(np.min(pcts))
                    r["pair_mean_percentile"] = float(np.mean(pcts))

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "multiprototype_ltpo_all_pairs.csv", index=False)

    by_cfg = rdf.groupby(["config", "k"], as_index=False).agg(
        ltpo_mean_pct=("heldout_formula_background_percentile", "mean"),
        ltpo_median_pct=("heldout_formula_background_percentile", "median"),
        ltpo_min_pct=("heldout_formula_background_percentile", "min"),
        ltpo_top20_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        ltpo_top10_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
        pair_min_mean=("pair_min_percentile", "mean"),
        pair_min_median=("pair_min_percentile", "median"),
        train_min_coverage_mean=("train_min_coverage", "mean"),
    )
    by_cfg.to_csv(out / "multiprototype_ltpo_by_config.csv", index=False)

    pooled = rdf.groupby("k", as_index=False).agg(
        ltpo_pooled_mean_pct=("heldout_formula_background_percentile", "mean"),
        ltpo_pooled_median_pct=("heldout_formula_background_percentile", "median"),
        ltpo_pooled_min_pct=("heldout_formula_background_percentile", "min"),
        ltpo_pooled_top20_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        ltpo_pooled_top10_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
        pair_min_mean=("pair_min_percentile", "mean"),
        pair_min_median=("pair_min_percentile", "median"),
    ).sort_values("k")
    pooled.to_csv(out / "multiprototype_ltpo_pooled.csv", index=False)
    return rdf, by_cfg, pooled


def write_report(out: Path, by_cfg: pd.DataFrame, pooled: pd.DataFrame, frozen_k: int):
    frozen = pooled[pooled.k == frozen_k]
    k1 = pooled[pooled.k == 1]
    lines = [
        "# Leave-two-positive-out robustness audit\n\n",
        "LTPO is deliberately **not** used to re-select K. The primary K was frozen by LOPO before this audit. Each of the 21 pairs of known positives is hidden, prototypes are learned from the remaining five positives, and both held-out materials are ranked against the formula-balanced unlabeled background.\n\n",
        "## By SOAP representation and K\n\n",
        by_cfg.to_markdown(index=False, floatfmt=".3f"),
        "\n\n## Pooled across both SOAP representations\n\n",
        pooled.to_markdown(index=False, floatfmt=".3f"),
        "\n\n",
        f"Frozen LOPO-selected K = **{frozen_k}**. LTPO is an audit only.\n\n",
    ]
    if not frozen.empty and not k1.empty:
        delta = float(frozen.iloc[0].ltpo_pooled_mean_pct - k1.iloc[0].ltpo_pooled_mean_pct)
        lines.append(f"Mean LTPO percentile difference K={frozen_k} minus K=1: **{delta:+.3f}** percentile points.\n\n")
    lines.append(
        "## Interpretation rule\n\n"
        "If the LOPO-selected multi-prototype model remains clearly better under LTPO, the evidence for multiple structural pathways strengthens. If its advantage collapses, changes sign, or only training coverage improves while held-out ranks do not, K>1 should be treated as an underdetermined fit to seven positives rather than a validated predictive structural dictionary.\n"
    )
    text = "".join(lines)
    (out / "LEAVE_TWO_OUT_VALIDATION.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "MULTIPROTOTYPE_SUMMARY.json"
    if not summary_path.exists():
        raise SystemExit("MULTIPROTOTYPE_SUMMARY.json missing: run multiprototype_validation.py first")
    frozen_k = int(json.loads(summary_path.read_text(encoding="utf-8"))["selected_k"])

    df, structs, _ = load_dataset(Path(a.cif_root), Path(a.metadata))
    known, blind, bg, reserved = formula_clean_split(df)
    local_by_cfg = {}
    for cfg in CONFIGS:
        print("LTPO SOAP", cfg["name"])
        local_by_cfg[cfg["name"]] = local_soap_all(build_soap(cfg), structs)

    rdf, by_cfg, pooled = run_ltpo(df, structs, local_by_cfg, known, bg, out)
    write_report(out, by_cfg, pooled, frozen_k)


if __name__ == "__main__":
    main()
