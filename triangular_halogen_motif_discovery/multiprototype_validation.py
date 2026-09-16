#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Positive-only multi-prototype structural-family validation.

Motivation
----------
The single recurrent local motif is not strongly LOPO-stable across all seven known
positive precursors. This script tests a more physical hypothesis: several local
structural pathways may collectively describe the positive set.

Method
------
For each SOAP representation (short and baseline), candidate prototypes are ALL local
environments from the training positive structures. A positive material is a client,
and its similarity to a candidate prototype is the best site-to-prototype SOAP
similarity in that material. We greedily maximize the standard facility-location
objective

    F(P) = sum_i max_{p in P} s(i,p)

where each positive formula i has equal weight. No background/unlabeled structure,
triangular descriptor, center type, chemistry identity, or blind formula participates
in prototype selection.

K is selected strictly from leave-one-positive-out (LOPO) performance over K=1..4,
using the mean formula-balanced background percentile across BOTH SOAP settings and
all seven held-out known positives; ties prefer the smaller K. Only after K is frozen
are the four blind formulas evaluated using their pre-specified MP structures.

Background formulas are used only to convert similarity into an interpretable rank
percentile, not as negative training labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from deep_study import build_soap, load_dataset, local_soap_all, posthoc_geometry
from formula_clean_final import formula_clean_split

CONFIGS = [
    {"name": "short", "r_cut": 2.8, "n_max": 4, "l_max": 4, "sigma": 0.25},
    {"name": "baseline", "r_cut": 3.2, "n_max": 6, "l_max": 6, "sigma": 0.30},
]
KS = (1, 2, 3, 4)


def formula_groups(df: pd.DataFrame, indices: List[int]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i in indices:
        out.setdefault(str(df.loc[i, "formula_norm"]), []).append(int(i))
    return out


def structure_score(local: List[np.ndarray], i: int, protos: np.ndarray) -> float:
    """Best local-environment similarity to any selected prototype."""
    return float(np.max(local[i] @ protos.T))


def bg_formula_distribution(local, protos, df, bg) -> np.ndarray:
    groups = formula_groups(df, bg)
    return np.asarray([
        max(structure_score(local, i, protos) for i in inds)
        for inds in groups.values()
    ], dtype=float)


def percentile(score: float, bg_scores: np.ndarray) -> float:
    return float(100.0 * np.mean(bg_scores <= score))


def candidate_pool(local, train: Dict[str, int]):
    vectors, meta = [], []
    for formula, i in train.items():
        for site in range(len(local[i])):
            vectors.append(local[i][site])
            meta.append((formula, int(i), int(site)))
    return np.asarray(vectors, dtype=np.float32), meta


def facility_select(local, train: Dict[str, int], k: int):
    """Greedy facility-location over positive *materials* with equal material weight."""
    C, meta = candidate_pool(local, train)
    formulas = list(train)
    # S[i,c] = best match of candidate c to any site in positive material i.
    S = np.empty((len(formulas), len(C)), dtype=np.float32)
    for a, f in enumerate(formulas):
        S[a] = np.max(local[train[f]] @ C.T, axis=0)

    covered = np.zeros(len(formulas), dtype=np.float32)
    selected: List[int] = []
    available = np.ones(len(C), dtype=bool)
    history = []
    for step in range(min(int(k), len(C))):
        idxs = np.where(available)[0]
        # Standard facility-location marginal gain.
        cand_cov = np.maximum(covered[:, None], S[:, idxs])
        gains = np.sum(cand_cov, axis=0) - np.sum(covered)
        # Deterministic tie breaking by candidate index.
        best_local = int(np.argmax(gains))
        c = int(idxs[best_local])
        selected.append(c); available[c] = False
        covered = np.maximum(covered, S[:, c])
        history.append({
            "step": step + 1,
            "candidate_index": c,
            "source_formula": meta[c][0],
            "source_structure_index": meta[c][1],
            "source_site": meta[c][2],
            "marginal_gain": float(gains[best_local]),
            "facility_value": float(np.sum(covered)),
            "min_train_coverage": float(np.min(covered)),
            "mean_train_coverage": float(np.mean(covered)),
        })
    return C[selected], [meta[c] for c in selected], history


def run_lopo(df, structs, local_by_cfg, known, bg, out: Path):
    rows = []
    for cfg_name, local in local_by_cfg.items():
        for held_f, held_i in known.items():
            train = {f: i for f, i in known.items() if f != held_f}
            for k in KS:
                protos, meta, hist = facility_select(local, train, k)
                bgs = bg_formula_distribution(local, protos, df, bg)
                held_score = structure_score(local, held_i, protos)
                rows.append({
                    "config": cfg_name,
                    "k": k,
                    "heldout_formula": held_f,
                    "heldout_material_id": str(df.loc[held_i, "material_id"]),
                    "heldout_score": held_score,
                    "heldout_formula_background_percentile": percentile(held_score, bgs),
                    "prototype_source_formulas": ";".join(m[0] for m in meta),
                    "prototype_source_sites": ";".join(str(m[2]) for m in meta),
                    "train_min_coverage": hist[-1]["min_train_coverage"],
                    "train_mean_coverage": hist[-1]["mean_train_coverage"],
                    "background_median_score": float(np.median(bgs)),
                    "background_p95_score": float(np.quantile(bgs, 0.95)),
                })
    ldf = pd.DataFrame(rows)
    ldf.to_csv(out / "multiprototype_lopo.csv", index=False)

    summary = ldf.groupby(["config", "k"], as_index=False).agg(
        lopo_mean_pct=("heldout_formula_background_percentile", "mean"),
        lopo_median_pct=("heldout_formula_background_percentile", "median"),
        lopo_min_pct=("heldout_formula_background_percentile", "min"),
        lopo_top20_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        lopo_top10_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
        train_min_coverage_mean=("train_min_coverage", "mean"),
        train_mean_coverage_mean=("train_mean_coverage", "mean"),
    )
    summary.to_csv(out / "multiprototype_lopo_by_config.csv", index=False)

    pooled = ldf.groupby("k", as_index=False).agg(
        pooled_mean_pct=("heldout_formula_background_percentile", "mean"),
        pooled_median_pct=("heldout_formula_background_percentile", "median"),
        pooled_min_pct=("heldout_formula_background_percentile", "min"),
        pooled_top20_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        pooled_top10_count=("heldout_formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
    )
    # Pre-declared selection: maximize pooled mean LOPO percentile; ties -> smaller K.
    pooled = pooled.sort_values(["pooled_mean_pct", "k"], ascending=[False, True]).reset_index(drop=True)
    selected_k = int(pooled.iloc[0]["k"])
    pooled.to_csv(out / "multiprototype_lopo_pooled_k_selection.csv", index=False)
    return ldf, summary, pooled, selected_k


def train_all_and_evaluate(df, structs, local_by_cfg, known, blind, bg, reserved, selected_k: int, out: Path):
    proto_rows, fixed_rows, rank_rows = [], [], []
    all_groups = formula_groups(df, list(range(len(df))))

    for cfg_name, local in local_by_cfg.items():
        protos, meta, hist = facility_select(local, known, selected_k)
        bgs = bg_formula_distribution(local, protos, df, bg)

        for pno, ((src_f, src_i, src_site), vec) in enumerate(zip(meta, protos), 1):
            geo = posthoc_geometry(structs[src_i], src_site)
            proto_rows.append({
                "config": cfg_name, "selected_k": selected_k, "prototype_no": pno,
                "source_formula": src_f, "source_material_id": str(df.loc[src_i, "material_id"]),
                "source_site": src_site,
                "source_center_posthoc": "X" if structs[src_i][src_site].specie.symbol == "He" else "M",
                "posthoc_best_m": geo.get("best_m"), "posthoc_best_psi": geo.get("best_psi"),
                "posthoc_psi6": geo.get("psi6"), "posthoc_planarity": geo.get("planarity_ratio"),
                "posthoc_radial_cv": geo.get("radial_cv"),
                "facility_step_value": hist[pno-1]["facility_value"],
                "train_min_coverage_after_step": hist[pno-1]["min_train_coverage"],
                "train_mean_coverage_after_step": hist[pno-1]["mean_train_coverage"],
            })

        for group, mapping in (("known", known), ("blind", blind)):
            for f, i in mapping.items():
                s = structure_score(local, i, protos)
                fixed_rows.append({
                    "config": cfg_name, "selected_k": selected_k, "group": group,
                    "formula": f, "material_id": str(df.loc[i, "material_id"]),
                    "score": s, "formula_background_percentile": percentile(s, bgs),
                })

        # Prospective candidate ranking may use best available experimental polymorph.
        for f, inds in all_groups.items():
            s = max(structure_score(local, i, protos) for i in inds)
            rank_rows.append({
                "config": cfg_name, "selected_k": selected_k, "formula": f,
                "best_polymorph_score": s, "formula_background_percentile": percentile(s, bgs),
                "reserved_formula": f in reserved,
            })

    pdf = pd.DataFrame(proto_rows); fdf = pd.DataFrame(fixed_rows); rdf = pd.DataFrame(rank_rows)
    pdf.to_csv(out / "multiprototype_selected_prototypes.csv", index=False)
    fdf.to_csv(out / "multiprototype_fixed_known_blind.csv", index=False)
    rdf.to_csv(out / "multiprototype_formula_scores.csv", index=False)

    fixed_cons = fdf.groupby(["group", "formula", "material_id"], as_index=False).agg(
        multiproto_median_pct=("formula_background_percentile", "median"),
        multiproto_min_pct=("formula_background_percentile", "min"),
        multiproto_max_pct=("formula_background_percentile", "max"),
        configs_top20=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        configs_top10=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
    )
    fixed_cons.to_csv(out / "multiprototype_fixed_known_blind_consensus.csv", index=False)

    rank_cons = rdf.groupby("formula", as_index=False).agg(
        multiproto_median_pct=("formula_background_percentile", "median"),
        multiproto_min_pct=("formula_background_percentile", "min"),
        multiproto_max_pct=("formula_background_percentile", "max"),
        configs_top20=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 80))),
        configs_top10=("formula_background_percentile", lambda x: int(np.sum(np.asarray(x) >= 90))),
        reserved_formula=("reserved_formula", "max"),
    ).sort_values(["configs_top20", "configs_top10", "multiproto_median_pct", "multiproto_min_pct"], ascending=False)
    rank_cons.to_csv(out / "multiprototype_formula_consensus.csv", index=False)
    rank_cons[~rank_cons.reserved_formula].head(50).to_csv(out / "multiprototype_top50_unseen.csv", index=False)
    return pdf, fixed_cons, rank_cons


def write_report(out: Path, lopo_summary, pooled, selected_k, protos, fixed, ranking):
    blind = fixed[fixed.group == "blind"]
    known = fixed[fixed.group == "known"]
    top = ranking[~ranking.reserved_formula].head(20)
    lines = ["# Positive-only multi-prototype structural-family validation\n\n"]
    lines.append("Prototype selection uses only known-positive local SOAP environments and the facility-location representativeness objective. Background formulas are used only for percentile calibration. Blind structures are touched only after K is frozen by LOPO.\n\n")
    lines.append("## 1. LOPO performance by K and SOAP setting\n\n")
    lines.append(lopo_summary.to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append("## 2. Pooled K selection across all 14 LOPO folds\n\n")
    lines.append(pooled.to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append(f"Selected K = **{selected_k}** by maximum pooled mean LOPO percentile (tie-break: smaller K).\n\n")
    lines.append("## 3. Frozen prototypes trained on all seven known positives\n\n")
    pcols=["config","prototype_no","source_formula","source_material_id","source_site","source_center_posthoc","posthoc_best_m","posthoc_psi6","train_min_coverage_after_step","train_mean_coverage_after_step"]
    lines.append(protos[pcols].to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append("## 4. Fixed-structure known and blind evaluation\n\n### Known\n\n")
    kcols=["formula","material_id","multiproto_median_pct","multiproto_min_pct","multiproto_max_pct","configs_top20","configs_top10"]
    lines.append(known[kcols].to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n### Blind\n\n")
    lines.append(blind[kcols].to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append("## 5. Top prospective formulas under the frozen multi-prototype family\n\n")
    rcols=["formula","multiproto_median_pct","multiproto_min_pct","configs_top20","configs_top10"]
    lines.append(top[rcols].to_markdown(index=False, floatfmt=".3f")); lines.append("\n\n")
    lines.append("## Interpretation\n\nIf K>1 materially improves LOPO relative to K=1, the data support a **small family of local structural pathways** rather than one universal local motif. The selected prototypes should then be decoded physically (center type, coordination/connectivity, triangularity only post-hoc) and linked to reaction/reconstructability features in the next stage. If K=1 remains optimal or gains are weak, the evidence for a multi-pathway structural dictionary is also weak and should not be overstated.\n")
    text="".join(lines); (out / "MULTIPROTOTYPE_VALIDATION.md").write_text(text, encoding="utf-8"); print(text)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cif-root", required=True); ap.add_argument("--metadata", required=True); ap.add_argument("--outdir", required=True)
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df, structs, _ = load_dataset(Path(a.cif_root), Path(a.metadata)); known, blind, bg, reserved = formula_clean_split(df)
    local_by_cfg={}
    for cfg in CONFIGS:
        print("MULTIPROTOTYPE SOAP", cfg["name"])
        local_by_cfg[cfg["name"]] = local_soap_all(build_soap(cfg), structs)
    ldf, lsum, pooled, selected_k = run_lopo(df, structs, local_by_cfg, known, bg, out)
    protos, fixed, ranking = train_all_and_evaluate(df, structs, local_by_cfg, known, blind, bg, reserved, selected_k, out)
    summary={"selected_k": selected_k, "selection_rule": "max pooled mean LOPO percentile across short+baseline and 7 heldouts; tie smaller K"}
    (out/"MULTIPROTOTYPE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out, lsum, pooled, selected_k, protos, fixed, ranking)


if __name__ == "__main__": main()
