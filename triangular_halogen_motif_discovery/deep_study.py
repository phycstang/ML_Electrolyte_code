#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep study of data-driven structural motifs in experimental binary metal halides.

Design principles
-----------------
1. Discovery uses the complete periodic M-X structure. No atoms are removed.
2. Chemical identities are coarse-grained only to two generic species: metal=M,
   halogen=X. This suppresses trivial chemistry matching while preserving M-X,
   X-X and M-M geometry.
3. Blind formulas (InI3, AlBr3, ZnCl2, SnCl2) are excluded from all fitting and
   motif selection. They are evaluated only after the discovery model is frozen.
4. No triangular, sixfold, psi6, exact-6 or X-centered criterion is used to select
   motifs. Geometry is decoded only after discovery.
5. Two complementary discovery routes are run:
   A) positive-only recurrent local SOAP motif + background rarity;
   B) fully unsupervised local-SOAP motif dictionary (PCA + MiniBatchKMeans),
      followed by positive-enrichment/Fisher tests.
6. SOAP hyperparameter robustness is tested with a small pre-declared grid.

Outputs are written as CSV/JSON/Markdown and are intended to be committed as
paper-audit-ready research records.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import fisher_exact
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

HALOGENS = {"F", "Cl", "Br", "I"}
KNOWN_ID_BY_FORMULA = {
    "AlCl3": "mp-25470",
    "FeCl3": "mp-23204",
    "GaF3": "mp-588",
    "InBr3": "mp-570219",
    "TaCl5": "mp-29831",
    "ZrCl4": "mp-569175",
    "GaCl3": "mp-30952",
}
BLIND_ID_BY_FORMULA = {
    "InI3": "mp-567789",
    "AlBr3": "mp-23288",
    "ZnCl2": "mp-22909",
    "SnCl2": "mp-29179",
}

SOAP_CONFIGS = [
    {"name": "short",        "r_cut": 2.8, "n_max": 4, "l_max": 4, "sigma": 0.25},
    {"name": "baseline",     "r_cut": 3.2, "n_max": 6, "l_max": 6, "sigma": 0.30},
    {"name": "medium",       "r_cut": 3.6, "n_max": 6, "l_max": 6, "sigma": 0.30},
    {"name": "wide",         "r_cut": 4.0, "n_max": 6, "l_max": 6, "sigma": 0.35},
    {"name": "radial_rich",  "r_cut": 3.2, "n_max": 8, "l_max": 6, "sigma": 0.30},
    {"name": "angular_rich", "r_cut": 3.2, "n_max": 6, "l_max": 8, "sigma": 0.30},
]


def is_binary_metal_halide(st: Structure) -> bool:
    els = list(st.composition.elements)
    if len(els) != 2:
        return False
    syms = {e.symbol for e in els}
    hs = syms & HALOGENS
    if len(hs) != 1:
        return False
    other = next(s for s in syms if s not in HALOGENS)
    try:
        return bool(Element(other).is_metal)
    except Exception:
        return False


def median_nn(st: Structure, radius: float = 10.0) -> float:
    vals = []
    for site in st:
        ds = [float(n.nn_distance) for n in st.get_neighbors(site, radius) if float(n.nn_distance) > 1e-7]
        if ds:
            vals.append(min(ds))
    if not vals:
        raise ValueError("No nearest-neighbour distances found")
    return float(np.median(vals))


def anonymize_and_scale(st: Structure) -> Tuple[Structure, float]:
    """Keep all atoms; map metal->H (M), halogen->He (X), scale by all-atom median NN."""
    d0 = median_nn(st)
    species = ["He" if s.specie.symbol in HALOGENS else "H" for s in st]
    out = Structure(Lattice(np.asarray(st.lattice.matrix, float) / d0), species, st.frac_coords)
    return out, d0


def mp_num(x: str) -> int:
    try:
        return int(str(x).split("-")[-1])
    except Exception:
        return 10**12


def select_idx(df: pd.DataFrame, formula: str, mpid: str | None) -> int | None:
    sub = df[df.formula_norm == formula]
    if sub.empty:
        return None
    if mpid:
        hit = sub[sub.material_id.astype(str) == mpid]
        if not hit.empty:
            return int(hit.index[0])
    return int(sorted(sub.index, key=lambda i: (mp_num(df.loc[i, "material_id"]), str(df.loc[i, "cif_file"])))[0])


def load_dataset(cif_root: Path, metadata: Path):
    meta = pd.read_csv(metadata)
    meta["cif_file"] = meta.cif_file.astype(str)
    mask = (
        meta.experimentally_observed.astype(str).str.lower().eq("yes")
        & meta.is_structure_representative.astype(str).str.lower().eq("true")
    )
    meta = meta[mask].copy()
    files = {p.name: p for p in cif_root.rglob("*.cif")}

    rows, scaled_structs, raw_structs = [], [], []
    for _, r in meta.iterrows():
        p = files.get(Path(str(r.cif_file)).name)
        if p is None:
            continue
        try:
            raw = Structure.from_file(str(p))
            if not is_binary_metal_halide(raw):
                continue
            st, d0 = anonymize_and_scale(raw)
        except Exception:
            continue
        rows.append({
            "material_id": str(r.material_id),
            "cif_file": p.name,
            "formula_norm": raw.composition.reduced_formula,
            "n_atoms": len(raw),
            "d0_A": d0,
        })
        scaled_structs.append(st)
        raw_structs.append(raw)
    df = pd.DataFrame(rows).reset_index(drop=True)
    return df, scaled_structs, raw_structs


def build_soap(cfg: dict) -> SOAP:
    kw = dict(
        species=[1, 2], periodic=True, r_cut=cfg["r_cut"], n_max=cfg["n_max"],
        l_max=cfg["l_max"], sigma=cfg["sigma"], average="off", sparse=False,
    )
    try:
        return SOAP(**kw)
    except TypeError:
        return SOAP(cfg["r_cut"], cfg["n_max"], cfg["l_max"], cfg["sigma"],
                    species=[1, 2], periodic=True, average="off", sparse=False)


def local_soap_all(soap: SOAP, structs: List[Structure]) -> List[np.ndarray]:
    ad = AseAtomsAdaptor()
    out = []
    for i, st in enumerate(structs):
        at = ad.get_atoms(st)
        at.set_pbc([True, True, True])
        x = np.asarray(soap.create(at), dtype=np.float32)
        x = normalize(x, norm="l2", axis=1).astype(np.float32)
        out.append(x)
        if (i + 1) % 100 == 0:
            print(f"  SOAP {i+1}/{len(structs)}")
    return out


def posthoc_geometry(st: Structure, center_idx: int) -> dict:
    """Decode six nearest X neighbours after motif discovery; never used as input."""
    site = st[center_idx]
    neigh = [n for n in st.get_neighbors(site, 4.5) if n.specie.symbol == "He" and n.nn_distance > 1e-6]
    neigh = sorted(neigh, key=lambda n: n.nn_distance)[:6]
    if len(neigh) < 6:
        return {"nX_available": len(neigh)}
    c = np.asarray(site.coords, float)
    V = np.asarray([np.asarray(n.coords, float) - c for n in neigh])
    d = np.linalg.norm(V, axis=1)
    C = V.T @ V / len(V)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    e1, e2 = evecs[:, 2], evecs[:, 1]
    ang = np.arctan2(V @ e2, V @ e1)
    psis = {m: float(abs(np.mean(np.exp(1j * m * ang)))) for m in range(2, 11)}
    best_m = max(psis, key=psis.get)
    return {
        "nX_available": 6,
        "six_X_distances_scaled": [float(v) for v in d],
        "radial_cv": float(np.std(d) / max(np.mean(d), 1e-12)),
        "planarity_ratio": float(evals[0] / max(np.sum(evals), 1e-12)),
        "best_m": int(best_m),
        "best_psi": float(psis[best_m]),
        "psi6": float(psis[6]),
        **{f"psi{m}": float(psis[m]) for m in range(2, 11)},
    }


def best_site_match(local: List[np.ndarray], structure_idx: int, prototype: np.ndarray) -> Tuple[int, float]:
    vals = local[structure_idx] @ prototype
    j = int(np.argmax(vals))
    return j, float(vals[j])


def discover_recurrent_motif(
    local: List[np.ndarray], structs: List[Structure], known: Dict[str, int], bg: List[int],
    top_common_candidates: int = 24,
):
    """Find motif common to all positives, then penalize motifs common in background.

    Geometry/center type are not used for selection. Rarity is computed only after the
    top recurrent motifs have been identified from positives.
    """
    candidates = []
    for formula, i in known.items():
        for site_idx, v in enumerate(local[i]):
            per = {}
            sims = []
            for f2, j in known.items():
                if j == i:
                    continue
                s = float(np.max(local[j] @ v))
                per[f2] = s
                sims.append(s)
            candidates.append({
                "origin_formula": formula,
                "origin_index": i,
                "site_index": site_idx,
                "commonality_min": float(min(sims)),
                "commonality_mean": float(np.mean(sims)),
                "matches": per,
            })
    candidates.sort(key=lambda x: (x["commonality_min"], x["commonality_mean"]), reverse=True)

    # SOAP-diverse top recurrent candidates.
    diverse, vecs = [], []
    for c in candidates:
        v = local[c["origin_index"]][c["site_index"]]
        if vecs and max(float(np.dot(v, u)) for u in vecs) >= 0.985:
            continue
        diverse.append(c)
        vecs.append(v)
        if len(diverse) >= top_common_candidates:
            break

    scored = []
    for c, v in zip(diverse, vecs):
        bg_scores = np.asarray([float(np.max(local[i] @ v)) for i in bg])
        prev95 = float(np.mean(bg_scores >= 0.95))
        prev98 = float(np.mean(bg_scores >= 0.98))
        # Pre-declared balance: recurrence across every positive × rarity in background.
        dscore = float(c["commonality_min"] * (1.0 - prev95))
        scored.append({**c, "background_prev95": prev95, "background_prev98": prev98,
                       "discovery_score": dscore, "prototype": v})
    scored.sort(key=lambda x: (x["discovery_score"], x["commonality_min"]), reverse=True)
    return scored


def percentile(score: float, background: np.ndarray) -> float:
    return float(100.0 * np.mean(background <= score))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, float)
    out[order] = q
    return out


def unsupervised_dictionary(
    local: List[np.ndarray], structs: List[Structure], known: Dict[str, int], blind: Dict[str, int],
    bg: List[int], ks=(32, 64, 96), pca_dim=32,
):
    """Cluster all non-blind local environments without labels; overlay labels afterward."""
    fit_struct_idx = sorted(set(known.values()) | set(bg))
    pieces, owner, site_no = [], [], []
    for i in fit_struct_idx:
        pieces.append(local[i])
        owner.extend([i] * len(local[i]))
        site_no.extend(range(len(local[i])))
    X = np.vstack(pieces).astype(np.float32)
    owner = np.asarray(owner, int)
    site_no = np.asarray(site_no, int)

    ncomp = min(pca_dim, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=0)
    Z = pca.fit_transform(X)

    known_set = set(known.values())
    results = []
    models = {}
    for k in ks:
        km = MiniBatchKMeans(n_clusters=int(k), random_state=0, n_init=10, batch_size=4096)
        labels = km.fit_predict(Z)
        models[k] = (km, pca)
        rows = []
        for c in range(k):
            struct_presence = set(owner[labels == c].tolist())
            pos_n = sum(i in struct_presence for i in known_set)
            bg_n = sum(i in struct_presence for i in bg)
            pos_abs = len(known_set) - pos_n
            bg_abs = len(bg) - bg_n
            odds, p = fisher_exact([[pos_n, pos_abs], [bg_n, bg_abs]], alternative="greater")
            pos_prev = pos_n / max(len(known_set), 1)
            bg_prev = bg_n / max(len(bg), 1)
            enrichment = (pos_prev + 1e-6) / (bg_prev + 1e-6)

            members = np.where(labels == c)[0]
            if len(members):
                center = km.cluster_centers_[c]
                d2 = np.sum((Z[members] - center) ** 2, axis=1)
                med = members[int(np.argmin(d2))]
                med_i, med_site = int(owner[med]), int(site_no[med])
            else:
                med_i, med_site = -1, -1
            rows.append({
                "k": int(k), "cluster": int(c), "positive_coverage": int(pos_n),
                "positive_prevalence": float(pos_prev), "background_prevalence": float(bg_prev),
                "enrichment": float(enrichment), "fisher_p": float(p),
                "medoid_structure_idx": med_i, "medoid_site_idx": med_site,
            })
        q = benjamini_hochberg(np.asarray([r["fisher_p"] for r in rows], float))
        for r, qq in zip(rows, q):
            r["fdr_q"] = float(qq)
            if r["medoid_structure_idx"] >= 0:
                i, s = r["medoid_structure_idx"], r["medoid_site_idx"]
                r["medoid_formula"] = str(structs[i].composition.reduced_formula).replace("H", "M").replace("He", "X")
                r["center_type"] = "X" if structs[i][s].specie.symbol == "He" else "M"
                r["geometry_posthoc"] = posthoc_geometry(structs[i], s)
        results.extend(rows)
    return results, models


def hidden_cluster_presence(local, blind, km, pca, cluster_id):
    out = {}
    for f, i in blind.items():
        z = pca.transform(local[i])
        lab = km.predict(z)
        frac = float(np.mean(lab == cluster_id))
        out[f] = {"presence": bool(np.any(lab == cluster_id)), "site_fraction": frac}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df, structs, raw_structs = load_dataset(Path(args.cif_root), Path(args.metadata))
    known = {f: select_idx(df, f, mid) for f, mid in KNOWN_ID_BY_FORMULA.items()}
    blind = {f: select_idx(df, f, mid) for f, mid in BLIND_ID_BY_FORMULA.items()}
    known = {f: i for f, i in known.items() if i is not None}
    blind = {f: i for f, i in blind.items() if i is not None}
    known_set, blind_set = set(known.values()), set(blind.values())
    bg = [i for i in range(len(df)) if i not in known_set and i not in blind_set]

    print("DATASET", len(df), "structures", df.formula_norm.nunique(), "formulas")
    print("KNOWN", {f: df.loc[i, "material_id"] for f, i in known.items()})
    print("BLIND", {f: df.loc[i, "material_id"] for f, i in blind.items()})

    robustness_rows = []
    matched_rows = []
    baseline_local = None
    baseline_best = None

    for cfg in SOAP_CONFIGS:
        print("\nSOAP CONFIG", cfg)
        soap = build_soap(cfg)
        local = local_soap_all(soap, structs)
        recurrent = discover_recurrent_motif(local, structs, known, bg)
        best = recurrent[0]
        v = best["prototype"]
        bg_scores = np.asarray([float(np.max(local[i] @ v)) for i in bg])

        origin_i, origin_s = int(best["origin_index"]), int(best["site_index"])
        g = posthoc_geometry(structs[origin_i], origin_s)
        blind_res = {}
        for f, i in blind.items():
            site, score = best_site_match(local, i, v)
            blind_res[f] = {"score": score, "percentile": percentile(score, bg_scores), "site": site}

        robustness_rows.append({
            "config": cfg["name"], "r_cut": cfg["r_cut"], "n_max": cfg["n_max"],
            "l_max": cfg["l_max"], "sigma": cfg["sigma"],
            "origin_formula": best["origin_formula"], "origin_material_id": df.loc[origin_i, "material_id"],
            "origin_site": origin_s,
            "origin_center": "X" if structs[origin_i][origin_s].specie.symbol == "He" else "M",
            "commonality_min": best["commonality_min"], "commonality_mean": best["commonality_mean"],
            "background_prev95": best["background_prev95"], "background_prev98": best["background_prev98"],
            "discovery_score": best["discovery_score"],
            "posthoc_best_m": g.get("best_m"), "posthoc_psi6": g.get("psi6"),
            "posthoc_planarity": g.get("planarity_ratio"), "posthoc_radial_cv": g.get("radial_cv"),
            **{f"{f}_pct": blind_res[f]["percentile"] for f in blind_res},
        })

        # Decode the exact best-matching site in every known and blind structure.
        for group, mapping in (("known", known), ("blind", blind)):
            for f, i in mapping.items():
                site, score = best_site_match(local, i, v)
                geo = posthoc_geometry(structs[i], site)
                matched_rows.append({
                    "config": cfg["name"], "group": group, "formula": f,
                    "material_id": df.loc[i, "material_id"], "matched_site": site,
                    "matched_center": "X" if structs[i][site].specie.symbol == "He" else "M",
                    "soap_similarity": score,
                    **geo,
                })

        if cfg["name"] == "baseline":
            baseline_local = local
            baseline_best = best

    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(out / "soap_robustness.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(out / "matched_site_geometry.csv", index=False)

    # Fully unsupervised dictionary using the frozen baseline SOAP representation.
    assert baseline_local is not None and baseline_best is not None
    print("\nUNSUPERVISED MOTIF DICTIONARY")
    unsup, models = unsupervised_dictionary(baseline_local, structs, known, blind, bg)
    unsup_df_rows = []
    for r in unsup:
        rr = {k: v for k, v in r.items() if k != "geometry_posthoc"}
        gg = r.get("geometry_posthoc", {})
        rr.update({f"geo_{k}": v for k, v in gg.items() if not isinstance(v, (list, dict))})
        unsup_df_rows.append(rr)
    unsup_df = pd.DataFrame(unsup_df_rows)
    unsup_df.to_csv(out / "unsupervised_cluster_enrichment.csv", index=False)

    # Select the strongest clusters with broad positive coverage. No geometry used.
    eligible = [r for r in unsup if r["positive_coverage"] >= 5]
    eligible.sort(key=lambda r: (r["fdr_q"], -r["positive_coverage"], -r["enrichment"]))
    top_unsup = eligible[:12]
    unsup_hidden = []
    for r in top_unsup:
        km, pca = models[r["k"]]
        h = hidden_cluster_presence(baseline_local, blind, km, pca, r["cluster"])
        for f, info in h.items():
            unsup_hidden.append({
                "k": r["k"], "cluster": r["cluster"], "rank_key_q": r["fdr_q"],
                "positive_coverage": r["positive_coverage"], "background_prevalence": r["background_prevalence"],
                "enrichment": r["enrichment"], "center_type": r.get("center_type"),
                "formula": f, **info,
            })
    pd.DataFrame(unsup_hidden).to_csv(out / "unsupervised_hidden_validation.csv", index=False)

    # Formula-level candidate ranking from the frozen baseline recurrent motif.
    proto = baseline_best["prototype"]
    scores = np.asarray([float(np.max(baseline_local[i] @ proto)) for i in range(len(df))])
    bg_scores = scores[bg]
    cand = df.copy()
    cand["motif_score"] = scores
    cand["motif_percentile_vs_background"] = [percentile(float(s), bg_scores) for s in scores]
    cand["is_known"] = cand.index.isin(known_set)
    cand["is_blind"] = cand.index.isin(blind_set)
    # one representative row per formula = highest motif score
    formula_rank = cand.sort_values("motif_score", ascending=False).drop_duplicates("formula_norm")
    formula_rank.to_csv(out / "formula_motif_ranking_all.csv", index=False)
    formula_rank[~formula_rank.is_known & ~formula_rank.is_blind].head(50).to_csv(out / "top50_unseen_candidates.csv", index=False)

    # Compact JSON report.
    payload = {
        "dataset": {"n_structures": len(df), "n_formulas": int(df.formula_norm.nunique())},
        "known": {f: str(df.loc[i, "material_id"]) for f, i in known.items()},
        "blind": {f: str(df.loc[i, "material_id"]) for f, i in blind.items()},
        "robustness": robustness_rows,
        "baseline_recurrent_motif": {
            k: v for k, v in baseline_best.items() if k != "prototype"
        },
        "top_unsupervised_clusters": top_unsup,
    }
    (out / "deep_study.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable research summary.
    lines = []
    lines.append("# Deep triangular-halogen motif study\n\n")
    lines.append(f"Dataset: **{len(df)} experimental representative structures / {df.formula_norm.nunique()} formulas**. ")
    lines.append("All structures retain the complete M-X framework and are anonymized to generic M/X species. ")
    lines.append("Blind formulas are excluded from all fitting and motif selection.\n\n")
    lines.append("## 1. SOAP robustness of the recurrent-motif discovery\n\n")
    show_cols = ["config", "origin_formula", "origin_center", "commonality_min", "background_prev95",
                 "posthoc_best_m", "posthoc_psi6", "InI3_pct", "AlBr3_pct", "ZnCl2_pct", "SnCl2_pct"]
    lines.append(robustness[show_cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n\n")
    lines.append("## 2. Fully unsupervised local-SOAP dictionary\n\n")
    if top_unsup:
        trows = []
        for r in top_unsup:
            g = r.get("geometry_posthoc", {})
            trows.append({
                "k": r["k"], "cluster": r["cluster"], "positive coverage": r["positive_coverage"],
                "background prevalence": r["background_prevalence"], "enrichment": r["enrichment"],
                "FDR q": r["fdr_q"], "center": r.get("center_type"), "posthoc best_m": g.get("best_m"),
                "posthoc psi6": g.get("psi6"),
            })
        lines.append(pd.DataFrame(trows).to_markdown(index=False, floatfmt=".4g"))
    else:
        lines.append("No cluster covered >=5 known positives.\n")
    lines.append("\n\n## 3. Interpretation guardrail\n\n")
    lines.append("The discovery stage never uses triangularity, psi6, exact-6, planarity, or X-only filtering. ")
    lines.append("Sixfold/triangular geometry is evaluated only after a motif or cluster has been selected by recurrent similarity or unsupervised enrichment. ")
    lines.append("Therefore a high post-hoc psi6 is evidence for interpretation, not an input that forced the discovery.\n\n")
    lines.append("## 4. Files\n\n")
    lines.append("- `soap_robustness.csv`: pre-declared SOAP parameter scan.\n")
    lines.append("- `matched_site_geometry.csv`: best matching site and post-hoc geometry for every known/blind formula.\n")
    lines.append("- `unsupervised_cluster_enrichment.csv`: label-free local SOAP clusters + Fisher/FDR enrichment.\n")
    lines.append("- `unsupervised_hidden_validation.csv`: blind-formula presence in enriched clusters.\n")
    lines.append("- `formula_motif_ranking_all.csv`: all formula-level motif rankings.\n")
    lines.append("- `top50_unseen_candidates.csv`: top 50 candidates excluding known/blind formulas.\n")
    lines.append("- `deep_study.json`: machine-readable full summary.\n")
    (out / "RESULTS.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


if __name__ == "__main__":
    main()
