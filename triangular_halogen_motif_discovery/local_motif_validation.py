#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rigorous validation of the data-driven local structural motif.

This module addresses four remaining methodological questions:

1. LOPO generalization: if one known positive is hidden, can a motif discovered from
   the other six recover it against the formula-clean background?
2. Formula-balance: candidate percentiles are evaluated against one best score per
   *formula*, not one score per structure, so formulas with many polymorphs do not
   obtain an unfair advantage.
3. Motif-family structure: retain several SOAP-diverse recurrent motifs instead of
   pretending that one hard cluster/prototype is the full explanation.
4. Empirical null: compare the known-positive recurrent-motif statistic with random
   groups of seven background formulas using a pre-declared baseline SOAP setting.

No triangular, psi6, exact-6, X-centered, or chemistry label is used for discovery.
Those quantities remain post-hoc interpretation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from deep_study import (
    KNOWN_ID_BY_FORMULA,
    BLIND_ID_BY_FORMULA,
    SOAP_CONFIGS,
    build_soap,
    discover_recurrent_motif,
    load_dataset,
    local_soap_all,
    posthoc_geometry,
    select_idx,
)
from formula_clean_final import formula_clean_split


def formula_groups(df: pd.DataFrame, indices: List[int]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i in indices:
        out.setdefault(str(df.loc[i, "formula_norm"]), []).append(int(i))
    return out


def structure_score(local: List[np.ndarray], i: int, proto: np.ndarray) -> float:
    return float(np.max(local[i] @ proto))


def formula_best_scores(
    local: List[np.ndarray], proto: np.ndarray, df: pd.DataFrame, indices: List[int]
) -> Dict[str, float]:
    groups = formula_groups(df, indices)
    return {
        f: max(structure_score(local, i, proto) for i in inds)
        for f, inds in groups.items()
    }


def pct(score: float, background: np.ndarray) -> float:
    return float(100.0 * np.mean(background <= score))


def selected_structure_scores(
    local: List[np.ndarray], proto: np.ndarray, mapping: Dict[str, int]
) -> Dict[str, float]:
    return {f: structure_score(local, i, proto) for f, i in mapping.items()}


def formula_balanced_robust_ranking(
    df: pd.DataFrame,
    structs,
    known: Dict[str, int],
    blind: Dict[str, int],
    bg: List[int],
    reserved_formulas: set,
    out: Path,
):
    """Six SOAP configurations; percentile background is one best score per formula."""
    bg_formula_groups = formula_groups(df, bg)
    all_formula_groups = formula_groups(df, list(range(len(df))))
    rows = []
    source_rows = []
    local_cache = {}

    for cfg in SOAP_CONFIGS:
        print("FORMULA-BALANCED LOCAL SOAP", cfg["name"])
        soap = build_soap(cfg)
        local = local_soap_all(soap, structs)
        if cfg["name"] in {"short", "baseline"}:
            local_cache[cfg["name"]] = local

        motifs = discover_recurrent_motif(local, structs, known, bg)
        best = motifs[0]
        proto = best["prototype"]
        bg_formula_scores = np.asarray([
            max(structure_score(local, i, proto) for i in inds)
            for inds in bg_formula_groups.values()
        ], float)

        oi, os = int(best["origin_index"]), int(best["site_index"])
        geo = posthoc_geometry(structs[oi], os)
        source_rows.append({
            "config": cfg["name"],
            "origin_formula": best["origin_formula"],
            "origin_material_id": str(df.loc[oi, "material_id"]),
            "origin_site": os,
            "origin_center_posthoc": "X" if structs[oi][os].specie.symbol == "He" else "M",
            "commonality_min": best["commonality_min"],
            "commonality_mean": best["commonality_mean"],
            "structure_background_prev95": best["background_prev95"],
            "formula_background_prev95": float(np.mean(bg_formula_scores >= 0.95)),
            "discovery_score_structure_bg": best["discovery_score"],
            "posthoc_best_m": geo.get("best_m"),
            "posthoc_psi6": geo.get("psi6"),
        })

        for formula, inds in all_formula_groups.items():
            score = max(structure_score(local, i, proto) for i in inds)
            rows.append({
                "config": cfg["name"],
                "formula": formula,
                "best_structure_score": score,
                "formula_background_percentile": pct(score, bg_formula_scores),
                "reserved_formula": formula in reserved_formulas,
            })

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "local_formula_balanced_scores_all_configs.csv", index=False)
    pd.DataFrame(source_rows).to_csv(out / "local_formula_balanced_motif_sources.csv", index=False)

    agg = rdf.groupby("formula", as_index=False).agg(
        local_median_pct=("formula_background_percentile", "median"),
        local_min_pct=("formula_background_percentile", "min"),
        local_max_pct=("formula_background_percentile", "max"),
        local_mean_pct=("formula_background_percentile", "mean"),
        local_top20_count=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80.0))),
        local_top10_count=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90.0))),
        n_configs=("config", "count"),
        reserved_formula=("reserved_formula", "max"),
    )
    agg = agg.sort_values(
        ["local_top20_count", "local_top10_count", "local_median_pct", "local_min_pct"],
        ascending=False,
    )
    agg.to_csv(out / "local_formula_balanced_consensus.csv", index=False)
    agg[~agg.reserved_formula].head(50).to_csv(
        out / "local_formula_balanced_top50_unseen.csv", index=False
    )
    return agg, local_cache, pd.DataFrame(source_rows)


def run_lopo(
    df: pd.DataFrame,
    structs,
    local_cache: Dict[str, List[np.ndarray]],
    known: Dict[str, int],
    blind: Dict[str, int],
    bg: List[int],
    out: Path,
):
    """Leave one known positive out; recover it with a motif learned from the other six."""
    bg_groups = formula_groups(df, bg)
    rows = []
    cfg_names = [x for x in ("short", "baseline") if x in local_cache]

    for cfg_name in cfg_names:
        local = local_cache[cfg_name]
        for held_formula, held_idx in known.items():
            train = {f: i for f, i in known.items() if f != held_formula}
            motifs = discover_recurrent_motif(local, structs, train, bg)
            best = motifs[0]
            proto = best["prototype"]
            bg_formula_scores = np.asarray([
                max(structure_score(local, i, proto) for i in inds)
                for inds in bg_groups.values()
            ], float)
            held_score = structure_score(local, held_idx, proto)
            oi, os = int(best["origin_index"]), int(best["site_index"])
            rows.append({
                "config": cfg_name,
                "heldout_formula": held_formula,
                "heldout_material_id": str(df.loc[held_idx, "material_id"]),
                "heldout_score": held_score,
                "heldout_formula_background_percentile": pct(held_score, bg_formula_scores),
                "motif_origin_formula": best["origin_formula"],
                "motif_origin_center_posthoc": "X" if structs[oi][os].specie.symbol == "He" else "M",
                "train_commonality_min": best["commonality_min"],
                "train_commonality_mean": best["commonality_mean"],
                "formula_background_prev95": float(np.mean(bg_formula_scores >= 0.95)),
                **{
                    f"blind_{f}_pct": pct(structure_score(local, i, proto), bg_formula_scores)
                    for f, i in blind.items()
                },
            })

    ldf = pd.DataFrame(rows)
    ldf.to_csv(out / "local_lopo_formula_clean.csv", index=False)
    summary = ldf.groupby("config", as_index=False).agg(
        lopo_mean_pct=("heldout_formula_background_percentile", "mean"),
        lopo_median_pct=("heldout_formula_background_percentile", "median"),
        lopo_min_pct=("heldout_formula_background_percentile", "min"),
        lopo_top20_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80.0))),
        lopo_top10_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90.0))),
    )
    summary.to_csv(out / "local_lopo_summary_formula_clean.csv", index=False)
    return ldf, summary


def chamfer_set_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric soft set coverage; unlike max-site similarity, every local site contributes."""
    K = A @ B.T
    return float(0.5 * (np.mean(np.max(K, axis=1)) + np.mean(np.max(K, axis=0))))


def positive_similarity_analysis(
    local: List[np.ndarray], known: Dict[str, int], out: Path
):
    formulas = list(known)
    n = len(formulas)
    S = np.eye(n, dtype=float)
    for a in range(n):
        for b in range(a + 1, n):
            s = chamfer_set_similarity(local[known[formulas[a]]], local[known[formulas[b]]])
            S[a, b] = S[b, a] = s
    pd.DataFrame(S, index=formulas, columns=formulas).to_csv(
        out / "known_positive_local_set_similarity.csv"
    )
    D = np.clip(1.0 - S, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    zdf = pd.DataFrame(Z, columns=["left", "right", "distance", "cluster_size"])
    zdf.to_csv(out / "known_positive_local_set_linkage.csv", index=False)
    return S, zdf


def recurrent_motif_family(
    df: pd.DataFrame,
    structs,
    local: List[np.ndarray],
    known: Dict[str, int],
    blind: Dict[str, int],
    bg: List[int],
    out: Path,
    n_motifs: int = 12,
):
    """Keep multiple SOAP-diverse recurrent motifs from the baseline representation."""
    motifs = discover_recurrent_motif(local, structs, known, bg, top_common_candidates=max(24, n_motifs))
    bg_groups = formula_groups(df, bg)
    motif_rows, score_rows = [], []
    for rank, m in enumerate(motifs[:n_motifs], start=1):
        proto = m["prototype"]
        bg_formula_scores = np.asarray([
            max(structure_score(local, i, proto) for i in inds)
            for inds in bg_groups.values()
        ], float)
        oi, os = int(m["origin_index"]), int(m["site_index"])
        geo = posthoc_geometry(structs[oi], os)
        motif_rows.append({
            "motif_rank": rank,
            "origin_formula": m["origin_formula"],
            "origin_material_id": str(df.loc[oi, "material_id"]),
            "origin_site": os,
            "origin_center_posthoc": "X" if structs[oi][os].specie.symbol == "He" else "M",
            "commonality_min": m["commonality_min"],
            "commonality_mean": m["commonality_mean"],
            "structure_background_prev95": m["background_prev95"],
            "formula_background_prev95": float(np.mean(bg_formula_scores >= 0.95)),
            "discovery_score_structure_bg": m["discovery_score"],
            "posthoc_best_m": geo.get("best_m"),
            "posthoc_best_psi": geo.get("best_psi"),
            "posthoc_psi6": geo.get("psi6"),
            "posthoc_planarity": geo.get("planarity_ratio"),
            "posthoc_radial_cv": geo.get("radial_cv"),
        })
        for group, mapping in (("known", known), ("blind", blind)):
            for f, i in mapping.items():
                s = structure_score(local, i, proto)
                score_rows.append({
                    "motif_rank": rank,
                    "group": group,
                    "formula": f,
                    "score": s,
                    "formula_background_percentile": pct(s, bg_formula_scores),
                })
    mdf = pd.DataFrame(motif_rows)
    sdf = pd.DataFrame(score_rows)
    mdf.to_csv(out / "baseline_recurrent_motif_family.csv", index=False)
    sdf.to_csv(out / "baseline_recurrent_motif_family_known_blind_scores.csv", index=False)
    return mdf, sdf


def permutation_null(
    df: pd.DataFrame,
    structs,
    local: List[np.ndarray],
    known: Dict[str, int],
    bg: List[int],
    out: Path,
    n_perm: int = 200,
    seed: int = 20260916,
):
    """Empirical null for the same baseline recurrent-motif discovery statistic."""
    observed = discover_recurrent_motif(local, structs, known, bg, top_common_candidates=12)[0]
    obs_score = float(observed["discovery_score"])
    obs_common = float(observed["commonality_min"])

    bg_groups = formula_groups(df, bg)
    bg_formulas = sorted(bg_groups)
    # Deterministic one-structure representative for each random formula.
    reps = {f: int(sorted(inds, key=lambda i: str(df.loc[i, "material_id"]))[0]) for f, inds in bg_groups.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(int(n_perm)):
        fs = rng.choice(bg_formulas, size=len(known), replace=False).tolist()
        fake = {f: reps[f] for f in fs}
        fs_set = set(fs)
        null_bg = [i for i in bg if str(df.loc[i, "formula_norm"]) not in fs_set]
        best = discover_recurrent_motif(local, structs, fake, null_bg, top_common_candidates=8)[0]
        rows.append({
            "perm": t,
            "best_discovery_score": float(best["discovery_score"]),
            "best_commonality_min": float(best["commonality_min"]),
            "best_background_prev95": float(best["background_prev95"]),
            "origin_formula": best["origin_formula"],
        })
        if (t + 1) % 25 == 0:
            print("permutation", t + 1, "/", n_perm)
    pdf = pd.DataFrame(rows)
    pdf.to_csv(out / "baseline_random7_formula_null.csv", index=False)
    p_score = float((1 + np.sum(pdf.best_discovery_score >= obs_score)) / (len(pdf) + 1))
    p_common = float((1 + np.sum(pdf.best_commonality_min >= obs_common)) / (len(pdf) + 1))
    summary = {
        "n_permutations": int(n_perm),
        "seed": int(seed),
        "observed_discovery_score": obs_score,
        "observed_commonality_min": obs_common,
        "null_discovery_score_mean": float(pdf.best_discovery_score.mean()),
        "null_discovery_score_95pct": float(pdf.best_discovery_score.quantile(0.95)),
        "null_commonality_min_mean": float(pdf.best_commonality_min.mean()),
        "null_commonality_min_95pct": float(pdf.best_commonality_min.quantile(0.95)),
        "empirical_p_discovery_score": p_score,
        "empirical_p_commonality_min": p_common,
    }
    (out / "baseline_random7_formula_null_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def multiview_table(local_consensus: pd.DataFrame, results_dir: Path):
    x = local_consensus.copy()
    rematch_path = results_dir / "rematch_candidate_consensus_formula_clean.csv"
    topo_path = results_dir / "triangular_layer_screen_formula_clean.csv"
    if rematch_path.exists():
        r = pd.read_csv(rematch_path)
        x = x.merge(r, on="formula", how="left")
    if topo_path.exists():
        t = pd.read_csv(topo_path)
        ta = t.groupby("formula", as_index=False).agg(
            topology_max_frac_exact6=("frac_exact6", "max"),
            topology_any_fallback=("fallback_sixfold", "max"),
        )
        x = x.merge(ta, on="formula", how="left")
    x = x.sort_values(
        ["local_top20_count", "local_top10_count", "local_median_pct", "local_min_pct"],
        ascending=False,
    )
    x.to_csv(results_dir / "multiview_candidate_evidence.csv", index=False)
    x[~x.reserved_formula].head(50).to_csv(
        results_dir / "multiview_top50_local_primary.csv", index=False
    )
    return x


def write_report(
    out: Path,
    lopo: pd.DataFrame,
    lopo_summary: pd.DataFrame,
    perm: dict,
    motif_family: pd.DataFrame,
    local_consensus: pd.DataFrame,
    multiview: pd.DataFrame,
):
    blind_local = local_consensus[local_consensus.formula.isin(BLIND_ID_BY_FORMULA)].copy()
    top = multiview[~multiview.reserved_formula].head(20).copy()
    cols_top = [
        c for c in [
            "formula", "local_median_pct", "local_min_pct", "local_top20_count", "local_top10_count",
            "median_mean_percentile", "median_nearest_percentile", "topology_max_frac_exact6"
        ] if c in top.columns
    ]
    lines = ["# Local motif validation and formula-balanced ranking\n\n"]
    lines.append("This analysis treats the local SOAP motif as the primary discovery object and uses REMatch/topology only as independent views. Formula percentiles are balanced at one best score per formula, removing polymorph-count bias.\n\n")
    lines.append("## 1. Leave-one-positive-out local-motif recovery\n\n")
    lines.append(lopo_summary.to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lcols = ["config","heldout_formula","heldout_formula_background_percentile","motif_origin_formula","motif_origin_center_posthoc","train_commonality_min"]
    lines.append(lopo[lcols].to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append("## 2. Empirical random-seven null\n\n")
    lines.append("```json\n" + json.dumps(perm, indent=2) + "\n```\n\n")
    lines.append("## 3. Formula-balanced blind robustness across six SOAP configurations\n\n")
    bcols = ["formula","local_median_pct","local_min_pct","local_max_pct","local_top20_count","local_top10_count"]
    lines.append(blind_local[bcols].to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append("## 4. Baseline recurrent motif family\n\n")
    mcols = ["motif_rank","origin_formula","origin_center_posthoc","commonality_min","formula_background_prev95","posthoc_best_m","posthoc_psi6"]
    lines.append(motif_family[mcols].to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append("## 5. Top prospective formulas (local evidence primary; other views shown, not collapsed into a tuned score)\n\n")
    lines.append(top[cols_top].to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append("## Interpretation rule\n\n")
    lines.append("A result is described as a **local structural-family signal** only if it is stable under LOPO and/or the empirical random-seven null. Exact-6 triangular order is never promoted to a causal or necessary criterion solely because it is present in positives; its high background prevalence and non-significant Fisher tests remain explicit controls. Whole-crystal REMatch is treated as a complementary global-similarity control, not as a replacement for the local motif.\n")
    (out / "LOCAL_MOTIF_VALIDATION.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-perm", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    df, structs, raw = load_dataset(Path(a.cif_root), Path(a.metadata))
    known, blind, bg, reserved = formula_clean_split(df)
    print("LOCAL VALIDATION:", len(df), "structures;", len(set(df.loc[bg, 'formula_norm'])), "background formulas")

    local_consensus, cache, sources = formula_balanced_robust_ranking(
        df, structs, known, blind, bg, reserved, out
    )
    lopo, lopo_summary = run_lopo(df, structs, cache, known, blind, bg, out)
    positive_similarity_analysis(cache["baseline"], known, out)
    motif_family, motif_scores = recurrent_motif_family(
        df, structs, cache["baseline"], known, blind, bg, out
    )
    perm = permutation_null(
        df, structs, cache["baseline"], known, bg, out, n_perm=a.n_perm
    )
    multiview = multiview_table(local_consensus, out)
    write_report(out, lopo, lopo_summary, perm, motif_family, local_consensus, multiview)


if __name__ == "__main__":
    main()
