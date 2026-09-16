#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formula-balanced validation of the discovered local structural motif.

Key safeguards
--------------
- Discovery still uses full periodic M-X structures anonymized only to M/X.
- No triangular/psi6/exact-6/X-centered feature is used for motif selection.
- Candidate ranking is formula-level: a formula may use its best experimental
  polymorph, and percentiles are compared with one best score per background formula.
- Known/blind validation NEVER uses best-polymorph selection: only the pre-specified
  MP structure for each reserved formula is scored.
- LOPO hides one known positive at a time.
- A random-seven empirical null uses one deterministic representative per background
  formula to test whether the positive group's recurrent-motif statistic is unusual.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from deep_study import (
    KNOWN_ID_BY_FORMULA, BLIND_ID_BY_FORMULA, SOAP_CONFIGS,
    build_soap, discover_recurrent_motif, load_dataset, local_soap_all,
    posthoc_geometry, mp_num,
)
from formula_clean_final import formula_clean_split


def formula_groups(df: pd.DataFrame, indices: List[int]) -> Dict[str, List[int]]:
    g: Dict[str, List[int]] = {}
    for i in indices:
        g.setdefault(str(df.loc[i, "formula_norm"]), []).append(int(i))
    return g


def sscore(local: List[np.ndarray], i: int, proto: np.ndarray) -> float:
    return float(np.max(local[i] @ proto))


def percentile(score: float, bg: np.ndarray) -> float:
    return float(100.0 * np.mean(bg <= score))


def bg_formula_scores(local, proto, df, bg):
    return np.asarray([
        max(sscore(local, i, proto) for i in inds)
        for inds in formula_groups(df, bg).values()
    ], float)


def robust_formula_ranking(df, structs, known, blind, bg, reserved, out: Path):
    all_groups = formula_groups(df, list(range(len(df))))
    candidate_rows, fixed_rows, source_rows = [], [], []
    cache = {}

    for cfg in SOAP_CONFIGS:
        print("LOCAL ROBUSTNESS", cfg["name"])
        soap = build_soap(cfg)
        local = local_soap_all(soap, structs)
        if cfg["name"] in {"short", "baseline"}:
            cache[cfg["name"]] = local

        best = discover_recurrent_motif(local, structs, known, bg)[0]
        proto = best["prototype"]
        bgs = bg_formula_scores(local, proto, df, bg)
        oi, os = int(best["origin_index"]), int(best["site_index"])
        geo = posthoc_geometry(structs[oi], os)
        source_rows.append({
            "config": cfg["name"], "origin_formula": best["origin_formula"],
            "origin_material_id": str(df.loc[oi, "material_id"]), "origin_site": os,
            "origin_center_posthoc": "X" if structs[oi][os].specie.symbol == "He" else "M",
            "commonality_min": best["commonality_min"], "commonality_mean": best["commonality_mean"],
            "structure_background_prev95": best["background_prev95"],
            "formula_background_prev95": float(np.mean(bgs >= 0.95)),
            "posthoc_best_m": geo.get("best_m"), "posthoc_psi6": geo.get("psi6"),
        })

        # Prospective ranking: best available experimental polymorph per formula.
        for f, inds in all_groups.items():
            s = max(sscore(local, i, proto) for i in inds)
            candidate_rows.append({
                "config": cfg["name"], "formula": f, "best_structure_score": s,
                "formula_background_percentile": percentile(s, bgs),
                "reserved_formula": f in reserved,
            })

        # Validation: exactly one pre-specified structure, never best-polymorph selection.
        for group, mapping in (("known", known), ("blind", blind)):
            for f, i in mapping.items():
                s = sscore(local, i, proto)
                fixed_rows.append({
                    "config": cfg["name"], "group": group, "formula": f,
                    "material_id": str(df.loc[i, "material_id"]), "score": s,
                    "formula_background_percentile": percentile(s, bgs),
                })

    cdf = pd.DataFrame(candidate_rows)
    fdf = pd.DataFrame(fixed_rows)
    sdf = pd.DataFrame(source_rows)
    cdf.to_csv(out / "local_formula_balanced_scores_all_configs.csv", index=False)
    fdf.to_csv(out / "local_fixed_known_blind_all_configs.csv", index=False)
    sdf.to_csv(out / "local_formula_balanced_motif_sources.csv", index=False)

    agg = cdf.groupby("formula", as_index=False).agg(
        local_median_pct=("formula_background_percentile", "median"),
        local_min_pct=("formula_background_percentile", "min"),
        local_max_pct=("formula_background_percentile", "max"),
        local_mean_pct=("formula_background_percentile", "mean"),
        local_top20_count=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        local_top10_count=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
        n_configs=("config", "count"), reserved_formula=("reserved_formula", "max"),
    ).sort_values(
        ["local_top20_count", "local_top10_count", "local_median_pct", "local_min_pct"],
        ascending=False,
    )
    fixed = fdf.groupby(["group", "formula", "material_id"], as_index=False).agg(
        fixed_median_pct=("formula_background_percentile", "median"),
        fixed_min_pct=("formula_background_percentile", "min"),
        fixed_max_pct=("formula_background_percentile", "max"),
        fixed_top20_count=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        fixed_top10_count=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
        n_configs=("config", "count"),
    )
    agg.to_csv(out / "local_formula_balanced_consensus.csv", index=False)
    fixed.to_csv(out / "local_fixed_known_blind_consensus.csv", index=False)
    agg[~agg.reserved_formula].head(50).to_csv(out / "local_formula_balanced_top50_unseen.csv", index=False)
    return agg, fixed, cache, sdf


def run_lopo(df, structs, cache, known, blind, bg, out: Path):
    rows = []
    for cfg_name in ("short", "baseline"):
        local = cache[cfg_name]
        for held_f, held_i in known.items():
            train = {f: i for f, i in known.items() if f != held_f}
            best = discover_recurrent_motif(local, structs, train, bg)[0]
            proto = best["prototype"]
            bgs = bg_formula_scores(local, proto, df, bg)
            oi, os = int(best["origin_index"]), int(best["site_index"])
            row = {
                "config": cfg_name, "heldout_formula": held_f,
                "heldout_material_id": str(df.loc[held_i, "material_id"]),
                "heldout_formula_background_percentile": percentile(sscore(local, held_i, proto), bgs),
                "motif_origin_formula": best["origin_formula"],
                "motif_origin_center_posthoc": "X" if structs[oi][os].specie.symbol == "He" else "M",
                "train_commonality_min": best["commonality_min"],
                "train_commonality_mean": best["commonality_mean"],
                "formula_background_prev95": float(np.mean(bgs >= 0.95)),
            }
            for f, i in blind.items():
                row[f"blind_{f}_pct"] = percentile(sscore(local, i, proto), bgs)
            rows.append(row)
    ldf = pd.DataFrame(rows)
    ldf.to_csv(out / "local_lopo_formula_clean.csv", index=False)
    summary = ldf.groupby("config", as_index=False).agg(
        lopo_mean_pct=("heldout_formula_background_percentile", "mean"),
        lopo_median_pct=("heldout_formula_background_percentile", "median"),
        lopo_min_pct=("heldout_formula_background_percentile", "min"),
        lopo_top20_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        lopo_top10_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
    )
    summary.to_csv(out / "local_lopo_summary_formula_clean.csv", index=False)
    return ldf, summary


def set_similarity(A, B):
    K = A @ B.T
    return float(0.5 * (np.mean(np.max(K, axis=1)) + np.mean(np.max(K, axis=0))))


def positive_similarity(local, known, out: Path):
    fs = list(known); n = len(fs); S = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            S[a, b] = S[b, a] = set_similarity(local[known[fs[a]]], local[known[fs[b]]])
    pd.DataFrame(S, index=fs, columns=fs).to_csv(out / "known_positive_local_set_similarity.csv")
    D = np.clip(1 - S, 0, 2); np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    pd.DataFrame(Z, columns=["left", "right", "distance", "cluster_size"]).to_csv(
        out / "known_positive_local_set_linkage.csv", index=False
    )


def motif_family(df, structs, local, known, blind, bg, out: Path, n=12):
    motifs = discover_recurrent_motif(local, structs, known, bg, top_common_candidates=max(24, n))
    mrows, srows = [], []
    for rank, m in enumerate(motifs[:n], 1):
        proto = m["prototype"]; bgs = bg_formula_scores(local, proto, df, bg)
        oi, os = int(m["origin_index"]), int(m["site_index"])
        geo = posthoc_geometry(structs[oi], os)
        mrows.append({
            "motif_rank": rank, "origin_formula": m["origin_formula"],
            "origin_material_id": str(df.loc[oi, "material_id"]), "origin_site": os,
            "origin_center_posthoc": "X" if structs[oi][os].specie.symbol == "He" else "M",
            "commonality_min": m["commonality_min"], "commonality_mean": m["commonality_mean"],
            "formula_background_prev95": float(np.mean(bgs >= 0.95)),
            "posthoc_best_m": geo.get("best_m"), "posthoc_best_psi": geo.get("best_psi"),
            "posthoc_psi6": geo.get("psi6"), "posthoc_planarity": geo.get("planarity_ratio"),
        })
        for group, mapping in (("known", known), ("blind", blind)):
            for f, i in mapping.items():
                s = sscore(local, i, proto)
                srows.append({"motif_rank": rank, "group": group, "formula": f,
                              "score": s, "formula_background_percentile": percentile(s, bgs)})
    mdf, sdf = pd.DataFrame(mrows), pd.DataFrame(srows)
    mdf.to_csv(out / "baseline_recurrent_motif_family.csv", index=False)
    sdf.to_csv(out / "baseline_recurrent_motif_family_known_blind_scores.csv", index=False)
    return mdf


def permutation_null(df, structs, local, known, bg, out: Path, n_perm=100, seed=20260916):
    """Formula-balanced null: one deterministic representative per background formula."""
    groups = formula_groups(df, bg)
    reps = {
        f: min(inds, key=lambda i: (mp_num(df.loc[i, "material_id"]), str(df.loc[i, "cif_file"])))
        for f, inds in groups.items()
    }
    rep_bg = list(reps.values())
    observed = discover_recurrent_motif(local, structs, known, rep_bg, top_common_candidates=6)[0]
    obs_score, obs_common = float(observed["discovery_score"]), float(observed["commonality_min"])
    formulas = sorted(reps); rng = np.random.default_rng(seed); rows = []
    for t in range(int(n_perm)):
        fs = rng.choice(formulas, size=len(known), replace=False).tolist(); fsset = set(fs)
        fake = {f: reps[f] for f in fs}; null_bg = [reps[f] for f in formulas if f not in fsset]
        best = discover_recurrent_motif(local, structs, fake, null_bg, top_common_candidates=6)[0]
        rows.append({"perm": t, "best_discovery_score": float(best["discovery_score"]),
                     "best_commonality_min": float(best["commonality_min"]),
                     "best_background_prev95": float(best["background_prev95"]),
                     "origin_formula": best["origin_formula"]})
        if (t + 1) % 20 == 0: print("random-seven", t + 1, "/", n_perm)
    pdf = pd.DataFrame(rows); pdf.to_csv(out / "baseline_random7_formula_null.csv", index=False)
    result = {
        "n_permutations": int(n_perm), "seed": int(seed),
        "observed_discovery_score": obs_score, "observed_commonality_min": obs_common,
        "null_discovery_score_mean": float(pdf.best_discovery_score.mean()),
        "null_discovery_score_95pct": float(pdf.best_discovery_score.quantile(.95)),
        "null_commonality_min_mean": float(pdf.best_commonality_min.mean()),
        "null_commonality_min_95pct": float(pdf.best_commonality_min.quantile(.95)),
        "empirical_p_discovery_score": float((1 + np.sum(pdf.best_discovery_score >= obs_score)) / (len(pdf) + 1)),
        "empirical_p_commonality_min": float((1 + np.sum(pdf.best_commonality_min >= obs_common)) / (len(pdf) + 1)),
    }
    (out / "baseline_random7_formula_null_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def multiview(local_agg, out: Path):
    x = local_agg.copy()
    rp = out / "rematch_candidate_consensus_formula_clean.csv"
    tp = out / "triangular_layer_screen_formula_clean.csv"
    if rp.exists(): x = x.merge(pd.read_csv(rp), on="formula", how="left")
    if tp.exists():
        t = pd.read_csv(tp).groupby("formula", as_index=False).agg(
            topology_max_frac_exact6=("frac_exact6", "max"),
            topology_any_fallback=("fallback_sixfold", "max"))
        x = x.merge(t, on="formula", how="left")
    x = x.sort_values(["local_top20_count", "local_top10_count", "local_median_pct", "local_min_pct"], ascending=False)
    x.to_csv(out / "multiview_candidate_evidence.csv", index=False)
    x[~x.reserved_formula].head(50).to_csv(out / "multiview_top50_local_primary.csv", index=False)
    return x


def report(out, lopo, lsum, perm, mdf, fixed, mv):
    blind = fixed[fixed.group == "blind"].copy()
    top = mv[~mv.reserved_formula].head(20).copy()
    topcols = [c for c in ["formula","local_median_pct","local_min_pct","local_top20_count","local_top10_count",
                           "median_mean_percentile","median_nearest_percentile","topology_max_frac_exact6"] if c in top]
    lines = ["# Local motif validation and formula-balanced ranking\n\n",
             "Candidate ranking is formula-best, but all known/blind validation below uses only the pre-specified MP structure. REMatch and topology are independent views and are not collapsed into a tuned score.\n\n",
             "## LOPO local-motif recovery\n\n", lsum.to_markdown(index=False, floatfmt=".3f"), "\n\n"]
    cols = ["config","heldout_formula","heldout_formula_background_percentile","motif_origin_formula","motif_origin_center_posthoc","train_commonality_min"]
    lines += [lopo[cols].to_markdown(index=False, floatfmt=".3f"), "\n\n## Random-seven empirical null\n\n```json\n",
              json.dumps(perm, indent=2), "\n```\n\n## Fixed-structure blind robustness\n\n",
              blind[["formula","material_id","fixed_median_pct","fixed_min_pct","fixed_max_pct","fixed_top20_count","fixed_top10_count"]].to_markdown(index=False, floatfmt=".3f"),
              "\n\n## Recurrent baseline motif family\n\n"]
    mcols = ["motif_rank","origin_formula","origin_center_posthoc","commonality_min","formula_background_prev95","posthoc_best_m","posthoc_psi6"]
    lines += [mdf[mcols].to_markdown(index=False, floatfmt=".3f"),
              "\n\n## Top prospective formulas — local evidence primary\n\n",
              top[topcols].to_markdown(index=False, floatfmt=".3f"),
              "\n\n## Interpretation\n\nA claim of a common local structural family should be supported by LOPO and the empirical null. Exact-6 is retained only as post-hoc interpretation because its background prevalence is high and its Fisher tests are non-significant. Whole-crystal REMatch remains a complementary control: disagreement with local SOAP is evidence that the signal is localized rather than a reason to force a single global metric.\n"]
    text = "".join(lines); (out / "LOCAL_MOTIF_VALIDATION.md").write_text(text, encoding="utf-8"); print(text)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cif-root", required=True); ap.add_argument("--metadata", required=True); ap.add_argument("--outdir", required=True); ap.add_argument("--n-perm", type=int, default=100)
    a = ap.parse_args(); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df, structs, _ = load_dataset(Path(a.cif_root), Path(a.metadata)); known, blind, bg, reserved = formula_clean_split(df)
    print("LOCAL VALIDATION:", len(df), "structures /", len(formula_groups(df, bg)), "background formulas")
    agg, fixed, cache, _ = robust_formula_ranking(df, structs, known, blind, bg, reserved, out)
    lopo, lsum = run_lopo(df, structs, cache, known, blind, bg, out)
    positive_similarity(cache["baseline"], known, out)
    mdf = motif_family(df, structs, cache["baseline"], known, blind, bg, out)
    perm = permutation_null(df, structs, cache["baseline"], known, bg, out, n_perm=a.n_perm)
    mv = multiview(agg, out); report(out, lopo, lsum, perm, mdf, fixed, mv)


if __name__ == "__main__": main()
