#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final formula-clean analysis for triangular-halogen motif discovery.

This script fixes a subtle leakage risk in the preliminary runs: excluding only one
blind MP-ID is insufficient when the same blind formula has other experimental
polymorphs. Here ALL structures whose reduced formula belongs to any known or blind
formula are excluded from the background/fitting pool. One pre-specified structure
per known/blind formula is retained only for positive recurrence or final blind
assessment, respectively.

The script produces the paper-facing results. Earlier outputs remain in the repo as
an explicit audit trail and must be treated as preliminary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from deep_study import (
    KNOWN_ID_BY_FORMULA, BLIND_ID_BY_FORMULA, SOAP_CONFIGS,
    build_soap, discover_recurrent_motif, load_dataset, local_soap_all,
    posthoc_geometry, select_idx, percentile, unsupervised_dictionary,
)
from posthoc_validation import (
    SHORT_CFG, local_soap_raw, get_pair_slice, best_block_match,
    triangular_layer_screen,
)


def formula_clean_split(df: pd.DataFrame):
    known = {f: select_idx(df, f, mid) for f, mid in KNOWN_ID_BY_FORMULA.items()}
    blind = {f: select_idx(df, f, mid) for f, mid in BLIND_ID_BY_FORMULA.items()}
    known = {f: i for f, i in known.items() if i is not None}
    blind = {f: i for f, i in blind.items() if i is not None}
    reserved_formulas = set(KNOWN_ID_BY_FORMULA) | set(BLIND_ID_BY_FORMULA)
    bg = [i for i in range(len(df)) if str(df.loc[i, "formula_norm"]) not in reserved_formulas]
    return known, blind, bg, reserved_formulas


def best_site(local, i, proto):
    vals = local[i] @ proto
    j = int(np.argmax(vals))
    return j, float(vals[j])


def recurrent_robustness(df, structs, known, blind, bg, out: Path):
    rows, matched_rows = [], []
    baseline = None
    short_pack = None
    for cfg in SOAP_CONFIGS:
        print("FORMULA-CLEAN SOAP", cfg["name"])
        soap = build_soap(cfg)
        local = local_soap_all(soap, structs)
        rec = discover_recurrent_motif(local, structs, known, bg)
        best = rec[0]
        proto = best["prototype"]
        bgs = np.asarray([float(np.max(local[i] @ proto)) for i in bg], float)
        oi, os = int(best["origin_index"]), int(best["site_index"])
        geo = posthoc_geometry(structs[oi], os)
        blind_vals = {}
        for f, i in blind.items():
            sidx, score = best_site(local, i, proto)
            blind_vals[f] = {"site": sidx, "score": score, "percentile": percentile(score, bgs)}
        rows.append({
            "config": cfg["name"], "r_cut": cfg["r_cut"], "n_max": cfg["n_max"],
            "l_max": cfg["l_max"], "sigma": cfg["sigma"],
            "origin_formula": best["origin_formula"], "origin_material_id": df.loc[oi, "material_id"],
            "origin_site": os, "origin_center": "X" if structs[oi][os].specie.symbol == "He" else "M",
            "commonality_min": best["commonality_min"], "commonality_mean": best["commonality_mean"],
            "background_prev95": best["background_prev95"], "background_prev98": best["background_prev98"],
            "discovery_score": best["discovery_score"], "posthoc_best_m": geo.get("best_m"),
            "posthoc_psi6": geo.get("psi6"), "posthoc_planarity": geo.get("planarity_ratio"),
            "posthoc_radial_cv": geo.get("radial_cv"),
            **{f"{f}_pct": blind_vals[f]["percentile"] for f in blind_vals},
        })
        for group, mapping in (("known", known), ("blind", blind)):
            for f, i in mapping.items():
                sidx, score = best_site(local, i, proto)
                matched_rows.append({
                    "config": cfg["name"], "group": group, "formula": f,
                    "material_id": df.loc[i, "material_id"], "matched_site": sidx,
                    "matched_center": "X" if structs[i][sidx].specie.symbol == "He" else "M",
                    "soap_similarity": score, **posthoc_geometry(structs[i], sidx),
                })
        if cfg["name"] == "baseline":
            baseline = (local, best)
        if cfg["name"] == "short":
            short_pack = (soap, local, best)
    pd.DataFrame(rows).to_csv(out / "soap_robustness_formula_clean.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(out / "matched_site_geometry_formula_clean.csv", index=False)
    return rows, baseline, short_pack


def run_unsupervised(df, structs, known, blind, bg, local, out: Path):
    results, models = unsupervised_dictionary(local, structs, known, blind, bg)
    flat = []
    for r in results:
        x = {k: v for k, v in r.items() if k != "geometry_posthoc"}
        g = r.get("geometry_posthoc", {})
        x.update({f"geo_{k}": v for k, v in g.items() if not isinstance(v, (dict, list))})
        flat.append(x)
    udf = pd.DataFrame(flat)
    udf.to_csv(out / "unsupervised_cluster_enrichment_formula_clean.csv", index=False)
    eligible = [r for r in results if r["positive_coverage"] >= 5]
    eligible.sort(key=lambda r: (r["fdr_q"], -r["positive_coverage"], -r["enrichment"]))
    return eligible


def run_channel_ablation(df, structs, known, blind, bg, short_pack, out: Path):
    soap, local_norm, motif = short_pack
    # Recompute raw local SOAP for exact species-pair blocks with same frozen config.
    raw = local_soap_raw(soap, structs)
    oi, os = int(motif["origin_index"]), int(motif["site_index"])
    proto_raw = raw[oi][os]
    rows = []
    for name, slc in {
        "MM": get_pair_slice(soap, 1, 1),
        "MX": get_pair_slice(soap, 1, 2),
        "XX": get_pair_slice(soap, 2, 2),
    }.items():
        bg_scores = np.asarray([best_block_match(raw, i, proto_raw, slc)[1] for i in bg], float)
        for group, mapping in (("known", known), ("blind", blind)):
            for f, i in mapping.items():
                site, score = best_block_match(raw, i, proto_raw, slc)
                rows.append({
                    "block": name, "group": group, "formula": f, "material_id": df.loc[i, "material_id"],
                    "matched_site": site, "score": score,
                    "percentile_vs_background": float(100*np.mean(bg_scores <= score)),
                })
    pd.DataFrame(rows).to_csv(out / "soap_species_channel_ablation_formula_clean.csv", index=False)
    return rows


def run_topology(df, raw_structs, known, blind, bg, out: Path):
    rows = []
    for i, st in enumerate(raw_structs):
        rows.append({"material_id": df.loc[i, "material_id"], "formula": df.loc[i, "formula_norm"],
                     **triangular_layer_screen(st)})
        if (i+1) % 100 == 0:
            print("formula-clean topology", i+1, "/", len(raw_structs))
    tdf = pd.DataFrame(rows)
    tdf.to_csv(out / "triangular_layer_screen_formula_clean.csv", index=False)
    selected = []
    for group, mapping in (("known", known), ("blind", blind)):
        for f, i in mapping.items():
            r = dict(rows[i]); r.update(group=group, target_formula=f)
            selected.append(r)
    sdf = pd.DataFrame(selected)
    sdf.to_csv(out / "triangular_layer_known_blind_formula_clean.csv", index=False)
    b = tdf.iloc[bg]
    stats = {
        "strict_exact6_all": float(np.mean(b.frac_exact6 >= 1-1e-12)),
        "frac_exact6_ge_0.8": float(np.mean(b.frac_exact6 >= 0.8)),
        "frac_exact6_ge_0.5": float(np.mean(b.frac_exact6 >= 0.5)),
        "fallback_sixfold": float(np.mean(b.fallback_sixfold.astype(bool))),
    }
    return sdf, stats, tdf


def formula_ranking(df, baseline, bg, reserved_formulas, out: Path):
    local, motif = baseline
    proto = motif["prototype"]
    scores = np.asarray([float(np.max(local[i] @ proto)) for i in range(len(df))])
    bg_scores = scores[bg]
    x = df.copy()
    x["motif_score"] = scores
    x["motif_percentile_vs_formula_clean_background"] = [float(100*np.mean(bg_scores <= s)) for s in scores]
    x["reserved_formula"] = x.formula_norm.isin(reserved_formulas)
    # Formula-level ranking uses best polymorph, but ALL reserved formulas are withheld from prospective candidates.
    fr = x.sort_values("motif_score", ascending=False).drop_duplicates("formula_norm")
    fr.to_csv(out / "formula_motif_ranking_formula_clean.csv", index=False)
    fr[~fr.reserved_formula].head(50).to_csv(out / "top50_unseen_candidates_formula_clean.csv", index=False)
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-root", required=True); ap.add_argument("--metadata", required=True); ap.add_argument("--outdir", required=True)
    a = ap.parse_args(); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df, structs, raw = load_dataset(Path(a.cif_root), Path(a.metadata))
    known, blind, bg, reserved = formula_clean_split(df)
    print("FORMULA-CLEAN split:", len(df), "structures;", len(bg), "background structures")
    print("Reserved formula polymorph counts:", df[df.formula_norm.isin(reserved)].groupby('formula_norm').size().to_dict())

    rob, baseline, short_pack = recurrent_robustness(df, structs, known, blind, bg, out)
    eligible = run_unsupervised(df, structs, known, blind, bg, baseline[0], out)
    abl = run_channel_ablation(df, structs, known, blind, bg, short_pack, out)
    selected_topo, topo_stats, topo_all = run_topology(df, raw, known, blind, bg, out)
    fr = formula_ranking(df, baseline, bg, reserved, out)

    abl_df = pd.DataFrame(abl)
    abl_summary = abl_df.groupby(["block","group"]).agg(
        mean_score=("score","mean"), mean_percentile=("percentile_vs_background","mean"),
        min_percentile=("percentile_vs_background","min")
    ).reset_index()
    robdf = pd.DataFrame(rob)

    payload = {
        "n_structures": len(df), "n_formulas": int(df.formula_norm.nunique()), "n_formula_clean_background_structures": len(bg),
        "reserved_polymorph_counts": df[df.formula_norm.isin(reserved)].groupby('formula_norm').size().to_dict(),
        "topology_background": topo_stats,
        "n_unsupervised_clusters_covering_ge5_positives": len(eligible),
        "robustness": rob,
    }
    (out / "FINAL_SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Formula-clean final results\n\n"]
    lines.append("This is the paper-facing rerun. **All polymorphs of every known and blind formula are excluded from the background/fitting pool.** One pre-specified MP structure per formula is used only for the intended known/blind role.\n\n")
    lines.append("## SOAP recurrent-motif robustness\n\n")
    cols=["config","origin_formula","origin_center","commonality_min","background_prev95","posthoc_best_m","posthoc_psi6","InI3_pct","AlBr3_pct","ZnCl2_pct","SnCl2_pct"]
    lines.append(robdf[cols].to_markdown(index=False,floatfmt=".3f")); lines.append("\n\n")
    lines.append("## Species-channel ablation of the frozen short-range motif\n\n")
    lines.append(abl_summary.to_markdown(index=False,floatfmt=".3f")); lines.append("\n\n")
    lines.append("## Independent X-layer exact-6 screen\n\n")
    tcols=["group","target_formula","material_id","frac_exact6","mean_psi6_exact6","mean_psi4_exact6","strict_exact6_all","fallback_sixfold","h","k","l"]
    lines.append(selected_topo[tcols].to_markdown(index=False,floatfmt=".3f")); lines.append("\n\n")
    lines.append("Formula-clean background topology prevalence: `"+json.dumps(topo_stats)+"`.\n\n")
    lines.append("## Fully unsupervised hard-cluster test\n\n")
    if eligible:
        z=[]
        for r in eligible[:10]:
            z.append({"k":r["k"],"cluster":r["cluster"],"positive_coverage":r["positive_coverage"],"background_prevalence":r["background_prevalence"],"enrichment":r["enrichment"],"fdr_q":r["fdr_q"],"center":r.get("center_type")})
        lines.append(pd.DataFrame(z).to_markdown(index=False,floatfmt=".4g"))
    else:
        lines.append("No hard local-SOAP cluster covered >=5/7 known positives. This argues for a continuous/fuzzy motif family rather than one discrete prototype cluster.")
    lines.append("\n\n## Interpretation\n\n")
    lines.append("A triangular-halogen network is **not** treated as a proven universal necessary condition. The recurrent SOAP family is robustly X-centered for most parameterizations, while the independent exact-6 screen tests extended triangular order separately. The final claim must reflect both results and the substantial background prevalence of exact-6 order.\n")
    (out / "FINAL_RESULTS.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))

if __name__ == '__main__':
    main()
