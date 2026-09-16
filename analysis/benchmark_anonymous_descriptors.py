#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark element-agnostic structural descriptors for binary metal halides.

Goal
----
Discover which representation best captures a shared structural motif among the
known viscoelastic-precursor positives without hard-coding triangular order.

Discovery representation keeps *all* atoms but anonymizes chemistry:
  any metal -> M (encoded as H)
  halogen F/Cl/Br/I -> X (encoded as He)

Descriptors compared:
  1) 2-body species-resolved pair-distance histogram (MM/MX/XX)
  2) 3-body species-resolved angle histogram around M/X centers
  3) concatenated 2+3-body histogram
  4) SOAP(mean+std over all sites), if DScribe is available

All structures are globally rescaled by the median nearest-neighbour distance d0
computed from the complete M-X structure. No X-only preprocessing is used.

Evaluation is label-light and fixed:
  * leave-one-positive-out (LOPO) retrieval percentile for seven known positives
  * pairwise compactness of the seven positives
  * blind retrieval percentiles for InI3, AlBr3, ZnCl2, SnCl2

The four blind formulas are excluded from fitting any unsupervised scaling.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.preprocessing import StandardScaler, normalize

HALOGENS = {"F", "Cl", "Br", "I"}

KNOWN_ID_BY_FORMULA = {
    "AlCl3": "mp-25470",
    "FeCl3": "mp-23204",
    "GaF3": "mp-588",
    "InBr3": "mp-570219",
    "TaCl5": "mp-29831",
    "ZrCl4": "mp-569175",
    # GaCl3 is selected deterministically from experimental representative rows.
    "GaCl3": None,
}

BLIND_ID_BY_FORMULA = {
    "InI3": "mp-567789",
    "AlBr3": "mp-23288",
    "ZnCl2": "mp-22909",
    # Prefer the monoclinic SnCl2 used in the project; fall back by formula.
    "SnCl2": "mp-29179",
}


def canonical_formula(st: Structure) -> str:
    return st.composition.reduced_formula


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


def median_nearest_neighbor(st: Structure, search_r: float = 10.0) -> float:
    vals = []
    for i, site in enumerate(st):
        neigh = st.get_neighbors(site, search_r)
        ds = [float(n.nn_distance) for n in neigh if float(n.nn_distance) > 1e-6]
        if ds:
            vals.append(min(ds))
    if not vals:
        raise ValueError("no nearest-neighbour distances found")
    return float(np.median(vals))


def anonymous_scaled_structure(st: Structure) -> Structure:
    d0 = median_nearest_neighbor(st)
    species = []
    for site in st:
        s = site.specie.symbol
        species.append("He" if s in HALOGENS else "H")  # X / M labels only
    lat = Lattice(np.asarray(st.lattice.matrix, float) / d0)
    return Structure(lat, species, st.frac_coords, coords_are_cartesian=False)


def site_type(site) -> str:
    return "X" if site.specie.symbol == "He" else "M"


def pair_hist(st: Structure, rmax: float = 3.2, nbins: int = 72) -> np.ndarray:
    channels = {"MM": np.zeros(nbins), "MX": np.zeros(nbins), "XX": np.zeros(nbins)}
    edges = np.linspace(0.0, rmax, nbins + 1)
    # Each periodic pair appears from both endpoints; divide by 2 at the end.
    for i, site in enumerate(st):
        ti = site_type(site)
        for n in st.get_neighbors(site, rmax):
            tj = site_type(n)
            key = "".join(sorted((ti, tj)))
            if key == "XM":
                key = "MX"
            r = float(n.nn_distance)
            b = np.searchsorted(edges, r, side="right") - 1
            if 0 <= b < nbins:
                channels[key][b] += math.exp(-r / 2.5)
    vec = np.concatenate([channels[k] for k in ("MM", "MX", "XX")]) / max(1, 2 * len(st))
    return vec.astype(np.float32)


def angle_hist(st: Structure, rmax: float = 2.8, nbins: int = 72, max_neigh: int = 18) -> np.ndarray:
    # center type x unordered neighbour pair type
    names = [
        "M_MM", "M_MX", "M_XX",
        "X_MM", "X_MX", "X_XX",
    ]
    ch = {k: np.zeros(nbins) for k in names}
    edges = np.linspace(-1.0, 1.0, nbins + 1)

    for i, site in enumerate(st):
        ctype = site_type(site)
        neigh = sorted(st.get_neighbors(site, rmax), key=lambda n: n.nn_distance)[:max_neigh]
        if len(neigh) < 2:
            continue
        c = np.asarray(site.coords, float)
        for a, b in combinations(neigh, 2):
            va = np.asarray(a.coords, float) - c
            vb = np.asarray(b.coords, float) - c
            ra = float(np.linalg.norm(va)); rb = float(np.linalg.norm(vb))
            if ra < 1e-8 or rb < 1e-8:
                continue
            cosang = float(np.clip(np.dot(va, vb) / (ra * rb), -1.0, 1.0))
            ta, tb = site_type(a), site_type(b)
            pair = "".join(sorted((ta, tb)))
            if pair == "XM":
                pair = "MX"
            key = f"{ctype}_{pair}"
            ib = np.searchsorted(edges, cosang, side="right") - 1
            ib = min(max(ib, 0), nbins - 1)
            ch[key][ib] += math.exp(-(ra + rb) / 4.0)

    vec = np.concatenate([ch[k] for k in names]) / max(1, len(st))
    return vec.astype(np.float32)


def soap_feature(st: Structure) -> np.ndarray | None:
    try:
        from dscribe.descriptors import SOAP
    except Exception:
        return None
    at = AseAtomsAdaptor().get_atoms(st)
    at.set_pbc([True, True, True])
    try:
        soap = SOAP(
            species=[1, 2], periodic=True, r_cut=3.2, n_max=6, l_max=6,
            sigma=0.30, average="off", sparse=False,
        )
        F = np.asarray(soap.create(at), dtype=np.float32)
    except TypeError:
        # Compatibility with older DScribe signature.
        soap = SOAP(3.2, 6, 6, 0.30, species=[1, 2], periodic=True,
                    average="off", sparse=False)
        F = np.asarray(soap.create(at), dtype=np.float32)
    return np.concatenate([F.mean(axis=0), F.std(axis=0)]).astype(np.float32)


def l2_rows(X: np.ndarray) -> np.ndarray:
    return normalize(X, norm="l2", axis=1)


def cosine_scores(Xn: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    c = centroid / max(np.linalg.norm(centroid), 1e-12)
    return Xn @ c


def percentile_against_background(score: float, bg_scores: np.ndarray) -> float:
    return float(100.0 * np.mean(bg_scores <= score))


def choose_formula_row(df: pd.DataFrame, formula: str, preferred_id: str | None) -> int | None:
    sub = df[df["formula_norm"] == formula]
    if sub.empty:
        return None
    if preferred_id is not None:
        hit = sub[sub["material_id"].astype(str) == preferred_id]
        if not hit.empty:
            return int(hit.index[0])
    # deterministic fallback: smallest MP numeric id, then filename
    def mpnum(x):
        try:
            return int(str(x).split("-")[-1])
        except Exception:
            return 10**12
    idx = sorted(sub.index, key=lambda ii: (mpnum(sub.loc[ii, "material_id"]), str(sub.loc[ii, "cif_file"])))[0]
    return int(idx)


def evaluate_descriptor(name: str, X: np.ndarray, df: pd.DataFrame, known_idx: list[int], blind_idx: dict[str, int], fit_idx: list[int]) -> dict:
    # Unsupervised scaling is fit without blind structures. For histogram features,
    # z-scoring equalizes bins/channels; row L2 then supports cosine retrieval.
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X[fit_idx])
    Xs = scaler.transform(X)
    Xn = l2_rows(Xs)

    known_set = set(known_idx)
    blind_set = set(blind_idx.values())
    bg_idx = [i for i in fit_idx if i not in known_set and i not in blind_set]

    # Pairwise known-positive compactness.
    sims = []
    for a, b in combinations(known_idx, 2):
        sims.append(float(np.dot(Xn[a], Xn[b])))

    lopo = {}
    for i in known_idx:
        train = [j for j in known_idx if j != i]
        centroid = Xn[train].mean(axis=0)
        scores = cosine_scores(Xn, centroid)
        pct = percentile_against_background(float(scores[i]), scores[bg_idx])
        lopo[df.loc[i, "formula_norm"]] = {
            "material_id": str(df.loc[i, "material_id"]),
            "score": float(scores[i]),
            "percentile": pct,
        }

    centroid_all = Xn[known_idx].mean(axis=0)
    scores_all = cosine_scores(Xn, centroid_all)
    blind = {}
    for formula, i in blind_idx.items():
        blind[formula] = {
            "material_id": str(df.loc[i, "material_id"]),
            "score": float(scores_all[i]),
            "percentile": percentile_against_background(float(scores_all[i]), scores_all[bg_idx]),
        }

    return {
        "descriptor": name,
        "n_features": int(X.shape[1]),
        "positive_pairwise_cosine_mean": float(np.mean(sims)),
        "positive_pairwise_cosine_min": float(np.min(sims)),
        "lopo_mean_percentile": float(np.mean([x["percentile"] for x in lopo.values()])),
        "lopo_min_percentile": float(np.min([x["percentile"] for x in lopo.values()])),
        "lopo": lopo,
        "blind": blind,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cif_root = Path(args.cif_root)
    meta = pd.read_csv(args.metadata)
    meta["cif_file"] = meta["cif_file"].astype(str)

    # Restrict to experimentally observed representative structures first.
    m = meta["experimentally_observed"].astype(str).str.lower().eq("yes")
    m &= meta["is_structure_representative"].astype(str).str.lower().eq("true")
    meta = meta[m].copy()

    paths = {p.name: p for p in cif_root.rglob("*.cif")}
    rows = []
    structs = []
    for _, r in meta.iterrows():
        fn = str(r["cif_file"])
        p = paths.get(Path(fn).name)
        if p is None:
            continue
        try:
            st0 = Structure.from_file(str(p))
            if not is_binary_metal_halide(st0):
                continue
            st = anonymous_scaled_structure(st0)
        except Exception as exc:
            continue
        rows.append({
            "material_id": str(r["material_id"]),
            "cif_file": Path(fn).name,
            "formula_norm": canonical_formula(st0),
            "n_atoms": len(st0),
        })
        structs.append(st)

    df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"DATASET experimental representative binary metal halides: {len(df)}")
    print(f"Unique formulas: {df.formula_norm.nunique()}")

    known_idx = []
    known_selected = {}
    for formula, mpid in KNOWN_ID_BY_FORMULA.items():
        i = choose_formula_row(df, formula, mpid)
        if i is not None:
            known_idx.append(i); known_selected[formula] = {"idx": i, "material_id": str(df.loc[i, "material_id"])}
    blind_idx = {}
    for formula, mpid in BLIND_ID_BY_FORMULA.items():
        i = choose_formula_row(df, formula, mpid)
        if i is not None:
            blind_idx[formula] = i

    print("KNOWN selected:", known_selected)
    print("BLIND selected:", {f: str(df.loc[i, 'material_id']) for f, i in blind_idx.items()})
    if len(known_idx) < 6:
        raise RuntimeError(f"Too few known positives found: {len(known_idx)}")

    print("Computing anonymous full-MX 2-body descriptors...")
    X2 = np.vstack([pair_hist(s) for s in structs])
    print("Computing anonymous full-MX 3-body descriptors...")
    X3 = np.vstack([angle_hist(s) for s in structs])
    X23 = np.hstack([X2, X3])

    feats = {"2body_pair": X2, "3body_angle": X3, "2+3body": X23}

    print("Computing SOAP descriptors...")
    soap_rows = []
    soap_ok = True
    for k, s in enumerate(structs):
        f = soap_feature(s)
        if f is None:
            soap_ok = False; break
        soap_rows.append(f)
        if (k + 1) % 100 == 0:
            print(f" SOAP {k+1}/{len(structs)}")
    if soap_ok and soap_rows:
        feats["SOAP_mean_std"] = np.vstack(soap_rows)

    blind_set = set(blind_idx.values())
    fit_idx = [i for i in range(len(df)) if i not in blind_set]

    results = []
    for name, X in feats.items():
        print(f"Evaluating {name}: shape={X.shape}")
        res = evaluate_descriptor(name, X, df, known_idx, blind_idx, fit_idx)
        results.append(res)

    # ranking: LOPO mean primary, LOPO minimum secondary.
    results.sort(key=lambda r: (r["lopo_mean_percentile"], r["lopo_min_percentile"]), reverse=True)

    with open(outdir / "descriptor_benchmark.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_structures": len(df),
            "n_formulas": int(df.formula_norm.nunique()),
            "known_selected": known_selected,
            "blind_selected": {f: {"idx": i, "material_id": str(df.loc[i, "material_id"])} for f, i in blind_idx.items()},
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    df.to_csv(outdir / "dataset_used.csv", index=False)

    lines = []
    lines.append("# Anonymous M/X descriptor benchmark\n")
    lines.append(f"Dataset: {len(df)} experimental representative binary metal-halide structures; {df.formula_norm.nunique()} formulas.\n")
    lines.append("Blind formulas were excluded from unsupervised scaler fitting.\n")
    lines.append("## Summary\n")
    lines.append("| rank | descriptor | features | positive pairwise mean | positive pairwise min | LOPO mean percentile | LOPO min percentile |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|\n")
    for rank, r in enumerate(results, 1):
        lines.append(f"| {rank} | {r['descriptor']} | {r['n_features']} | {r['positive_pairwise_cosine_mean']:.3f} | {r['positive_pairwise_cosine_min']:.3f} | {r['lopo_mean_percentile']:.1f} | {r['lopo_min_percentile']:.1f} |\n")
    lines.append("\n## LOPO details\n")
    for r in results:
        lines.append(f"### {r['descriptor']}\n")
        lines.append("| formula | mp-id | percentile | score |\n|---|---|---:|---:|\n")
        for f, x in r["lopo"].items():
            lines.append(f"| {f} | {x['material_id']} | {x['percentile']:.1f} | {x['score']:.4f} |\n")
        lines.append("\nBlind retrieval:\n\n")
        lines.append("| formula | mp-id | percentile | score |\n|---|---|---:|---:|\n")
        for f, x in r["blind"].items():
            lines.append(f"| {f} | {x['material_id']} | {x['percentile']:.1f} | {x['score']:.4f} |\n")
        lines.append("\n")
    (outdir / "summary.md").write_text("".join(lines), encoding="utf-8")

    print("\n" + "".join(lines))


if __name__ == "__main__":
    main()
