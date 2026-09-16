#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formula-clean full-MX REMatch-SOAP validation.

Purpose
-------
The recurrent-motif analysis asks whether a *single* local environment recurs across
positives. REMatch provides an independent structure-level test: it softly matches
the complete sets of local environments between two crystals, without global mean
pooling and without deleting metal atoms.

Protocol
--------
- complete anonymous periodic M/X structures from `deep_study.load_dataset`
- all known/blind formula polymorphs excluded from the background pool
- one pre-specified structure per known/blind formula
- two pre-declared SOAP scales (short and baseline)
- REMatch entropy regularization alpha = 0.1, 1.0, 10.0
- no triangular/psi6/exact-6 information is used
- report both mean similarity to the positive set and nearest-positive similarity
  rather than selecting one after seeing results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize

from deep_study import build_soap, load_dataset, local_soap_all
from formula_clean_final import formula_clean_split

SOAP_SUBSET = [
    {"name": "short", "r_cut": 2.8, "n_max": 4, "l_max": 4, "sigma": 0.25},
    {"name": "baseline", "r_cut": 3.2, "n_max": 6, "l_max": 6, "sigma": 0.30},
]
ALPHAS = [0.1, 1.0, 10.0]


def rematch_cross(env_all, env_refs, alpha):
    """Return all-structure x reference REMatch kernel."""
    # local SOAP rows are already L2 normalized by local_soap_all, but normalize again
    # defensively so the linear local kernel is cosine-like and bounded numerically.
    xa = [normalize(np.asarray(x), norm="l2", axis=1) for x in env_all]
    yr = [normalize(np.asarray(x), norm="l2", axis=1) for x in env_refs]
    ker = REMatchKernel(metric="linear", alpha=float(alpha), threshold=1e-6)
    K = ker.create(xa, yr)
    return np.asarray(K, dtype=float)


def pct(score, bg):
    return float(100.0 * np.mean(np.asarray(bg) <= score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    df, structs, _ = load_dataset(Path(a.cif_root), Path(a.metadata))
    known, blind, bg, reserved = formula_clean_split(df)
    known_items = list(known.items())
    known_indices = [i for _, i in known_items]

    summary_rows = []
    lopo_rows = []
    blind_rows = []
    candidate_rows = []

    for cfg in SOAP_SUBSET:
        print("REMatch SOAP", cfg["name"])
        soap = build_soap(cfg)
        env_all = local_soap_all(soap, structs)
        env_refs = [env_all[i] for i in known_indices]

        for alpha in ALPHAS:
            print("  alpha", alpha)
            K = rematch_cross(env_all, env_refs, alpha)

            # LOPO: each known positive is scored against the remaining six refs.
            lopo_pcts_mean, lopo_pcts_max = [], []
            for col, (formula, i) in enumerate(known_items):
                keep = [j for j in range(len(known_items)) if j != col]
                s_mean = float(np.mean(K[i, keep]))
                s_max = float(np.max(K[i, keep]))
                bg_mean = np.mean(K[np.asarray(bg)[:, None], np.asarray(keep)[None, :]], axis=1)
                bg_max = np.max(K[np.asarray(bg)[:, None], np.asarray(keep)[None, :]], axis=1)
                pmean, pmax = pct(s_mean, bg_mean), pct(s_max, bg_max)
                lopo_pcts_mean.append(pmean); lopo_pcts_max.append(pmax)
                lopo_rows.append({
                    "soap_config": cfg["name"], "alpha": alpha, "formula": formula,
                    "material_id": df.loc[i, "material_id"],
                    "mean_similarity": s_mean, "mean_percentile": pmean,
                    "nearest_similarity": s_max, "nearest_percentile": pmax,
                })

            # Blind assessment uses all seven known positives.
            bg_mean_all = np.mean(K[bg, :], axis=1)
            bg_max_all = np.max(K[bg, :], axis=1)
            for formula, i in blind.items():
                smean = float(np.mean(K[i, :]))
                smax = float(np.max(K[i, :]))
                blind_rows.append({
                    "soap_config": cfg["name"], "alpha": alpha, "formula": formula,
                    "material_id": df.loc[i, "material_id"],
                    "mean_similarity": smean, "mean_percentile": pct(smean, bg_mean_all),
                    "nearest_similarity": smax, "nearest_percentile": pct(smax, bg_max_all),
                })

            # Prospective candidates: one best polymorph per non-reserved formula.
            for i in bg:
                candidate_rows.append({
                    "soap_config": cfg["name"], "alpha": alpha,
                    "material_id": df.loc[i, "material_id"], "formula": df.loc[i, "formula_norm"],
                    "mean_similarity": float(np.mean(K[i, :])),
                    "mean_percentile": pct(float(np.mean(K[i, :])), bg_mean_all),
                    "nearest_similarity": float(np.max(K[i, :])),
                    "nearest_percentile": pct(float(np.max(K[i, :])), bg_max_all),
                })

            summary_rows.append({
                "soap_config": cfg["name"], "alpha": alpha,
                "lopo_mean_percentile_mean_agg": float(np.mean(lopo_pcts_mean)),
                "lopo_min_percentile_mean_agg": float(np.min(lopo_pcts_mean)),
                "lopo_mean_percentile_nearest_agg": float(np.mean(lopo_pcts_max)),
                "lopo_min_percentile_nearest_agg": float(np.min(lopo_pcts_max)),
            })

    sdf = pd.DataFrame(summary_rows)
    ldf = pd.DataFrame(lopo_rows)
    bdf = pd.DataFrame(blind_rows)
    cdf = pd.DataFrame(candidate_rows)
    sdf.to_csv(out / "rematch_summary_formula_clean.csv", index=False)
    ldf.to_csv(out / "rematch_lopo_formula_clean.csv", index=False)
    bdf.to_csv(out / "rematch_blind_formula_clean.csv", index=False)

    # Consensus across the six pre-declared REMatch settings. First reduce polymorphs
    # within each setting by formula, then aggregate percentiles across settings.
    best_per_setting = (cdf.sort_values(["soap_config", "alpha", "mean_percentile"], ascending=[True, True, False])
                          .drop_duplicates(["soap_config", "alpha", "formula"]))
    cons = best_per_setting.groupby("formula").agg(
        median_mean_percentile=("mean_percentile", "median"),
        min_mean_percentile=("mean_percentile", "min"),
        median_nearest_percentile=("nearest_percentile", "median"),
        min_nearest_percentile=("nearest_percentile", "min"),
        settings=("mean_percentile", "count"),
    ).reset_index().sort_values(["median_mean_percentile", "min_mean_percentile"], ascending=False)
    cons.to_csv(out / "rematch_candidate_consensus_formula_clean.csv", index=False)
    cons.head(50).to_csv(out / "rematch_top50_consensus_formula_clean.csv", index=False)

    # Blind consensus table.
    bcons = bdf.groupby("formula").agg(
        median_mean_percentile=("mean_percentile", "median"),
        min_mean_percentile=("mean_percentile", "min"),
        median_nearest_percentile=("nearest_percentile", "median"),
        min_nearest_percentile=("nearest_percentile", "min"),
        top20_mean_count=("mean_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        top10_mean_count=("mean_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
    ).reset_index()
    bcons.to_csv(out / "rematch_blind_consensus_formula_clean.csv", index=False)

    lines = ["# Full-MX REMatch SOAP validation\n\n"]
    lines.append("REMatch compares the **complete sets of local M/X environments** between crystals. No X-only filtering or triangular descriptors are used. All known/blind formula polymorphs are removed from the background pool.\n\n")
    lines.append("## LOPO summary\n\n")
    lines.append(sdf.to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n\n## Blind consensus across all 6 REMatch settings\n\n")
    lines.append(bcons.to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n\n## Top 20 prospective formulas by REMatch mean-similarity consensus\n\n")
    lines.append(cons.head(20).to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n\nInterpretation: mean aggregation tests similarity to the positive set as a whole; nearest aggregation allows multiple structural mechanisms. Neither is selected post-hoc as the sole metric.\n")
    (out / "REMATCH_RESULTS.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))

if __name__ == "__main__":
    main()
