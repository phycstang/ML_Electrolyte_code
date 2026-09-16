#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthesize all structural-motif evidence into paper-facing figures and conclusions.

This script does not train or tune a model. It only combines already-frozen outputs
from local SOAP, REMatch, topology, LOPO/LTPO, and multi-prototype audits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLIND_ORDER = ["InI3", "AlBr3", "ZnCl2", "SnCl2"]


def read(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing required result: {path}")
    return pd.read_csv(path)


def fig_generalization(out: Path, lopo: pd.DataFrame, ltpo: pd.DataFrame):
    a = lopo.sort_values("k")
    b = ltpo.sort_values("k")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(a.k, a.pooled_mean_pct, marker="o", label="LOPO mean percentile")
    ax.plot(b.k, b.ltpo_pooled_mean_pct, marker="o", label="LTPO mean percentile")
    ax.plot(a.k, a.pooled_min_pct, marker="s", linestyle="--", label="LOPO minimum")
    ax.plot(b.k, b.ltpo_pooled_min_pct, marker="s", linestyle="--", label="LTPO minimum")
    ax.set_xlabel("Number of positive-only prototypes K")
    ax.set_ylabel("Formula-balanced background percentile")
    ax.set_xticks(sorted(a.k.unique()))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.set_title("Generalization versus structural-family complexity")
    fig.tight_layout()
    fig.savefig(out / "figure_generalization_vs_k.png", dpi=220)
    fig.savefig(out / "figure_generalization_vs_k.pdf")
    plt.close(fig)


def blind_comparison_table(out: Path):
    single = read(out / "local_fixed_known_blind_consensus.csv")
    multi = read(out / "multiprototype_fixed_known_blind_consensus.csv")
    rem = read(out / "rematch_blind_consensus_formula_clean.csv")
    s = single[single.group == "blind"][["formula", "fixed_median_pct", "fixed_min_pct"]].copy()
    m = multi[multi.group == "blind"][["formula", "multiproto_median_pct", "multiproto_min_pct"]].copy()
    r = rem[["formula", "median_mean_percentile", "median_nearest_percentile"]].copy()
    x = s.merge(m, on="formula", how="outer").merge(r, on="formula", how="outer")
    x["formula"] = pd.Categorical(x.formula, categories=BLIND_ORDER, ordered=True)
    x = x.sort_values("formula").reset_index(drop=True)
    x.to_csv(out / "blind_crossmethod_comparison.csv", index=False)

    vals = x[["fixed_median_pct", "multiproto_median_pct", "median_nearest_percentile"]].to_numpy(float)
    labels = ["Single local motif", "K-selected multi-prototype", "REMatch nearest-positive"]
    pos = np.arange(len(x)); w = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for j, lab in enumerate(labels):
        ax.bar(pos + (j - 1) * w, vals[:, j], width=w, label=lab)
    ax.set_xticks(pos, x.formula.astype(str))
    ax.set_ylim(0, 105)
    ax.set_ylabel("Median formula-background percentile")
    ax.set_title("Blind structures expose method-dependent structural signals")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure_blind_crossmethod.png", dpi=220)
    fig.savefig(out / "figure_blind_crossmethod.pdf")
    plt.close(fig)
    return x


def topology_figure(out: Path):
    t = read(out / "triangular_layer_screen_formula_clean.csv")
    known = t[t.group == "known"]
    # The background rows are all non-reserved structures in this table.
    bg = t[t.group == "background"] if "background" in set(t.group.astype(str)) else pd.DataFrame()
    if bg.empty:
        summary = json.loads((out / "FINAL_SUMMARY.json").read_text(encoding="utf-8"))
        prev = summary.get("topology_background_prevalence", {})
        bg_strict = float(prev.get("strict_exact6_all", prev.get("frac_exact6_ge_1", np.nan)))
        bg_fallback = float(prev.get("fallback_sixfold", np.nan))
    else:
        bg_strict = float(np.mean(bg.strict_exact6_all.astype(bool)))
        bg_fallback = float(np.mean(bg.fallback_sixfold.astype(bool)))
    vals = [
        float(np.mean(known.strict_exact6_all.astype(bool))), bg_strict,
        float(np.mean(known.fallback_sixfold.astype(bool))), bg_fallback,
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.bar([0, 1, 3, 4], vals)
    ax.set_xticks([0, 1, 3, 4], ["Known\nstrict", "Background\nstrict", "Known\nfallback", "Background\nfallback"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of structures")
    ax.set_title("Sixfold topology is common in the background")
    fig.tight_layout()
    fig.savefig(out / "figure_topology_prevalence.png", dpi=220)
    fig.savefig(out / "figure_topology_prevalence.pdf")
    plt.close(fig)


def candidate_crossview(out: Path):
    local = read(out / "local_formula_balanced_consensus.csv")
    rem = read(out / "rematch_candidate_consensus_formula_clean.csv")
    multi = read(out / "multiprototype_formula_consensus.csv")
    x = local.merge(rem, on="formula", how="left").merge(
        multi[["formula", "multiproto_median_pct", "multiproto_min_pct"]], on="formula", how="left"
    )
    x["local_minus_rematch_mean"] = x.local_median_pct - x.median_mean_percentile
    x["local_minus_multi"] = x.local_median_pct - x.multiproto_median_pct
    x.to_csv(out / "candidate_crossview_evidence.csv", index=False)

    z = x[~x.reserved_formula.astype(bool)].dropna(subset=["local_median_pct", "median_mean_percentile"])
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.scatter(z.local_median_pct, z.median_mean_percentile, s=20, alpha=0.65)
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1)
    ax.set_xlim(0, 102); ax.set_ylim(0, 102)
    ax.set_xlabel("Local-motif consensus percentile")
    ax.set_ylabel("Whole-crystal REMatch mean percentile")
    ax.set_title("Local and global structural evidence are not equivalent")
    # Annotate the largest disagreements only; avoid an unreadable label cloud.
    lab = z.reindex(z.local_minus_rematch_mean.abs().sort_values(ascending=False).head(10).index)
    for _, r in lab.iterrows():
        ax.annotate(str(r.formula), (r.local_median_pct, r.median_mean_percentile), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figure_candidate_local_vs_global.png", dpi=220)
    fig.savefig(out / "figure_candidate_local_vs_global.pdf")
    plt.close(fig)
    return x


def conclusions(out: Path, blind: pd.DataFrame, cross: pd.DataFrame):
    lopo = read(out / "multiprototype_lopo_pooled_k_selection.csv").sort_values("k")
    ltpo = read(out / "multiprototype_ltpo_pooled.csv").sort_values("k")
    stats = json.loads((out / "baseline_random7_formula_null_summary.json").read_text(encoding="utf-8"))
    msummary = json.loads((out / "MULTIPROTOTYPE_SUMMARY.json").read_text(encoding="utf-8"))
    k = int(msummary["selected_k"])
    l1 = lopo[lopo.k == 1].iloc[0]
    lk = lopo[lopo.k == k].iloc[0]
    t1 = ltpo[ltpo.k == 1].iloc[0]
    tk = ltpo[ltpo.k == k].iloc[0]

    exact = read(out / "exact6_statistical_tests.csv") if (out / "exact6_statistical_tests.csv").exists() else None
    if exact is not None and len(exact):
        exact_p = float(exact.iloc[0].p_one_sided)
    else:
        exact_p = np.nan

    lines = ["# Research synthesis: what the static-parent structures actually support\n\n"]
    lines.append("## Evidence hierarchy\n\n")
    lines.append(
        "**1. A single universal triangular-halogen descriptor is not supported.** The independent exact-6 screen has high background prevalence, and the known-positive enrichment is not statistically significant. Triangular/sixfold order should therefore remain a post-hoc structural interpretation, not the main classifier or causal claim.\n\n"
    )
    lines.append(
        "**2. A GaCl3-like local structural signal is real enough to be interesting, but not yet a universal positive-family model.** It retrospectively ranks InI3, AlBr3 and ZnCl2 highly under most local SOAP settings, yet single-motif LOPO across the seven known positives is weak and the random-seven discovery-score test is only borderline.\n\n"
    )
    lines.append(
        f"The 1000-run random-seven null gives empirical p = **{stats['empirical_p_discovery_score']:.4f}** for the discovery score and p = **{stats['empirical_p_commonality_min']:.4f}** for minimum commonality.\n\n"
    )
    lines.append(
        f"**3. Multiple prototypes improve internal positive-set generalization, but complexity must survive harder holdout tests.** LOPO mean percentile changes from **{l1.pooled_mean_pct:.1f}** at K=1 to **{lk.pooled_mean_pct:.1f}** at the frozen K={k}. Under LTPO the corresponding values are **{t1.ltpo_pooled_mean_pct:.1f}** and **{tk.ltpo_pooled_mean_pct:.1f}**.\n\n"
    )
    if tk.ltpo_pooled_mean_pct <= t1.ltpo_pooled_mean_pct + 2.0:
        lines.append("The LTPO advantage is small or absent, so the multi-prototype dictionary should be treated as underdetermined rather than as a validated predictive model.\n\n")
    else:
        lines.append("The multi-prototype advantage persists under LTPO, supporting structural heterogeneity; however, seven positives remain too few to interpret K as a uniquely determined number of physical mechanisms.\n\n")
    lines.append(
        "**4. Whole-crystal similarity and local-motif similarity answer different questions.** REMatch does not reproduce the local-motif blind recovery, showing that the discriminating signal is not simply global crystal similarity.\n\n"
    )
    lines.append("## Blind cross-method comparison\n\n")
    lines.append(blind.to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append(
        "The crucial negative result is that fitting a richer structural family to cover the seven known positives does **not** preserve the strong InI3/ZnCl2 retrospective recovery of the single GaCl3-like motif. Therefore the current static-CIF evidence does not support one structural model that simultaneously explains every known and hidden positive.\n\n"
    )
    lines.append("## Claim boundary for a paper\n\n")
    lines.append(
        "A defensible claim is: **binary-halide parent structures contain recurring local motifs and multiple structural pathways that can enrich candidate discovery, but static parent geometry alone is insufficient to define a universal viscoelastic-precursor rule.** The next discriminating layer should describe *reconstructability under reaction* rather than add more complexity to static-CIF fitting.\n\n"
    )
    lines.append("## Recommended next research stage\n\n")
    lines.append(
        "Freeze the current static-structure analysis as an interpretable structural prior. For a small, diverse candidate set, calculate reaction-aware quantities: O/X substitution energy, mixed-anion metastability, substitution degeneracy, coordination/connectivity changes after relaxation, structural RMSD, elastic/shear softness, and low-frequency modes. The scientifically useful score should eventually combine **structural-family membership + reconstructability**, not replace one with the other.\n"
    )
    text = "".join(lines)
    (out / "RESEARCH_CONCLUSIONS.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    a = ap.parse_args(); out = Path(a.results_dir)

    lopo = read(out / "multiprototype_lopo_pooled_k_selection.csv")
    ltpo = read(out / "multiprototype_ltpo_pooled.csv")
    fig_generalization(out, lopo, ltpo)
    blind = blind_comparison_table(out)
    topology_figure(out)
    cross = candidate_crossview(out)
    conclusions(out, blind, cross)


if __name__ == "__main__":
    main()
