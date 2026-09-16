#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistical summary for the formula-clean motif study.

The goal is not to manufacture significance from n=7 positives. This script makes
small-sample limitations explicit and reports exact tests against the formula-clean
experimental background.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from deep_study import KNOWN_ID_BY_FORMULA, BLIND_ID_BY_FORMULA


def fisher_row(a, b, c, d, label):
    res = fisher_exact([[a,b],[c,d]], alternative="greater")
    return {
        "test": label, "positive_success": a, "positive_fail": b,
        "background_success": c, "background_fail": d,
        "odds_ratio": float(res.statistic), "p_one_sided": float(res.pvalue),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    a = ap.parse_args(); d = Path(a.results_dir)

    topo = pd.read_csv(d / "triangular_layer_screen_formula_clean.csv")
    selected = pd.read_csv(d / "triangular_layer_known_blind_formula_clean.csv")
    rob = pd.read_csv(d / "soap_robustness_formula_clean.csv")

    reserved = set(KNOWN_ID_BY_FORMULA) | set(BLIND_ID_BY_FORMULA)
    bg = topo[~topo.formula.isin(reserved)].copy()
    known = selected[selected.group == "known"].copy()
    blind = selected[selected.group == "blind"].copy()

    bg_strict = int(np.sum(bg.frac_exact6 >= 1.0 - 1e-12))
    bg_fallback = int(np.sum(bg.fallback_sixfold.astype(str).str.lower().eq("true")))
    k_strict = int(np.sum(known.frac_exact6 >= 1.0 - 1e-12))
    k_fallback = int(np.sum(known.fallback_sixfold.astype(str).str.lower().eq("true")))

    rows = [
        fisher_row(k_strict, len(known)-k_strict, bg_strict, len(bg)-bg_strict,
                   "known positives: strict exact-6 vs background"),
        fisher_row(k_fallback, len(known)-k_fallback, bg_fallback, len(bg)-bg_fallback,
                   "known positives: fallback sixfold vs background"),
    ]

    # The three core blind discoveries are a retrospective validation set, not a new
    # independent experiment. The Fisher row is reported descriptively and must not be
    # presented as prospective significance.
    core = blind[blind.target_formula.isin(["InI3","AlBr3","ZnCl2"])]
    core_strict = int(np.sum(core.frac_exact6 >= 1.0 - 1e-12))
    rows.append(fisher_row(core_strict, len(core)-core_strict, bg_strict, len(bg)-bg_strict,
                           "core blind 3: strict exact-6 vs background (retrospective)"))
    tests = pd.DataFrame(rows)
    tests.to_csv(d / "topology_fisher_exact.csv", index=False)

    # SOAP robustness consensus for blind formulas.
    blind_cols = [c for c in rob.columns if c.endswith("_pct")]
    b_rows = []
    for col in blind_cols:
        formula = col[:-4]
        vals = rob[col].astype(float).values
        b_rows.append({
            "formula": formula,
            "median_percentile_all6": float(np.median(vals)),
            "min_percentile_all6": float(np.min(vals)),
            "max_percentile_all6": float(np.max(vals)),
            "n_configs_top20": int(np.sum(vals >= 80.0)),
            "n_configs_top10": int(np.sum(vals >= 90.0)),
        })
    bcons = pd.DataFrame(b_rows)
    bcons.to_csv(d / "soap_blind_robustness_consensus.csv", index=False)

    # Descriptor-selection stability itself is a result, but configs are correlated;
    # do not assign a binomial p-value.
    source_summary = (rob.groupby(["origin_formula", "origin_center"]).size()
                        .reset_index(name="n_of_6_configs")
                        .sort_values("n_of_6_configs", ascending=False))
    source_summary.to_csv(d / "soap_motif_source_stability.csv", index=False)

    lines = ["# Statistical validation\n\n"]
    lines.append("## Exact topology tests\n\n")
    lines.append(tests.to_markdown(index=False, floatfmt=".4g"))
    lines.append("\n\nWith only seven known positives and a high background prevalence of exact-6/sixfold order, these tests are expected to have low power. A non-significant p-value is scientifically important: it prevents treating triangular order alone as a positive-specific descriptor.\n\n")
    lines.append("## Blind SOAP robustness across the six pre-declared SOAP parameterizations\n\n")
    lines.append(bcons.to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n\n## Source-motif stability\n\n")
    lines.append(source_summary.to_markdown(index=False))
    lines.append("\n\nThe SOAP parameterizations are correlated representations of the same structures, so `n_of_6_configs` is a robustness count, not an independent-trial significance test.\n")
    (d / "STATISTICAL_VALIDATION.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))

if __name__ == "__main__":
    main()
