#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc validation for the data-driven motif study.

This script deliberately runs *after* the full-MX motif discovery. It provides two
independent interpretation tests:

1) SOAP species-pair channel ablation (M-M, M-X, X-X) to determine which part of
   the complete environment drives similarity to the discovered motif.
2) A periodic, low-index-plane search on the halogen sublattice to test whether an
   extended exact-6 quasi-planar X network actually exists. This topology screen is
   not used in motif discovery or model selection.

The topology screen follows the spirit of the prior exact-6 criterion: 3x3x3 periodic
images, candidate crystallographic plane normals, in-plane direction-cosine tolerance
|n.u_hat| <= 0.12, and a 5 A in-plane X-X neighbourhood. The best normal maximizes the
fraction of X sites having exactly six in-plane X neighbours.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.preprocessing import normalize

from deep_study import (
    BLIND_ID_BY_FORMULA, KNOWN_ID_BY_FORMULA, HALOGENS,
    build_soap, discover_recurrent_motif, load_dataset, posthoc_geometry,
    select_idx,
)

SHORT_CFG = {"name": "short", "r_cut": 2.8, "n_max": 4, "l_max": 4, "sigma": 0.25}


def local_soap_raw(soap, structs):
    ad = AseAtomsAdaptor()
    out = []
    for i, st in enumerate(structs):
        at = ad.get_atoms(st); at.set_pbc([True, True, True])
        x = np.asarray(soap.create(at), dtype=np.float32)
        out.append(x)
        if (i + 1) % 100 == 0:
            print(" raw SOAP", i + 1, "/", len(structs))
    return out


def normalized_rows(x):
    return normalize(np.asarray(x), norm="l2", axis=1).astype(np.float32)


def get_pair_slice(soap, a, b):
    """DScribe SOAP exposes a species-pair power-spectrum block location."""
    for pair in ((a, b), (b, a)):
        try:
            return soap.get_location(pair)
        except Exception:
            pass
    raise RuntimeError(f"Could not get SOAP block for {(a,b)}")


def best_block_match(raw_list, structure_idx, proto_raw, slc):
    p = proto_raw[slc].reshape(1, -1)
    p = normalized_rows(p)[0]
    x = raw_list[structure_idx][:, slc]
    x = normalized_rows(x)
    vals = x @ p
    j = int(np.argmax(vals))
    return j, float(vals[j])


def candidate_hkl(max_index=3):
    out = []
    seen = set()
    for h, k, l in itertools.product(range(-max_index, max_index + 1), repeat=3):
        if (h, k, l) == (0, 0, 0):
            continue
        g = np.gcd.reduce([abs(h), abs(k), abs(l)])
        if g == 0:
            continue
        h0, k0, l0 = h // g, k // g, l // g
        # canonical sign
        first = next(v for v in (h0, k0, l0) if v != 0)
        if first < 0:
            h0, k0, l0 = -h0, -k0, -l0
        t = (h0, k0, l0)
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def psi_m_from_vectors(vectors, normal, m):
    n = normal / np.linalg.norm(normal)
    # choose a stable in-plane basis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    a = np.arctan2(vectors @ e2, vectors @ e1)
    return float(abs(np.mean(np.exp(1j * m * a))))


def triangular_layer_screen(raw: Structure, rcut=5.0, direction_cos_tol=0.12, hkl_max=3):
    """Search low-index crystallographic planes for an extended exact-6 X network."""
    x_indices = [i for i, s in enumerate(raw) if s.specie.symbol in HALOGENS]
    if not x_indices:
        return {"n_X": 0}

    frac = np.asarray([raw[i].frac_coords for i in x_indices], float)
    cart = np.asarray([raw[i].coords for i in x_indices], float)
    lattice = np.asarray(raw.lattice.matrix, float)
    recip = np.asarray(raw.lattice.reciprocal_lattice.matrix, float)

    # 3x3x3 periodic images of X sites.
    trans = np.asarray(list(itertools.product([-1, 0, 1], repeat=3)), float)
    image_cart = []
    image_origin = []
    for t in trans:
        shift = t @ lattice
        for local_j, c in enumerate(cart):
            image_cart.append(c + shift)
            image_origin.append(local_j)
    image_cart = np.asarray(image_cart, float)
    image_origin = np.asarray(image_origin, int)

    best = None
    for hkl in candidate_hkl(hkl_max):
        n = hkl[0] * recip[0] + hkl[1] * recip[1] + hkl[2] * recip[2]
        nn = np.linalg.norm(n)
        if nn < 1e-10:
            continue
        n = n / nn
        exact = 0
        psi6_vals = []
        psi4_vals = []
        counts = []
        for local_i, c in enumerate(cart):
            V = image_cart - c
            d = np.linalg.norm(V, axis=1)
            mask = (d > 1e-6) & (d <= rcut)
            if not np.any(mask):
                counts.append(0); continue
            W = V[mask]
            dd = d[mask]
            u = W / dd[:, None]
            planar = np.abs(u @ n) <= direction_cos_tol
            Wp = W[planar]
            dp = dd[planar]
            order = np.argsort(dp)
            Wp = Wp[order]
            counts.append(len(Wp))
            if len(Wp) == 6:
                exact += 1
                psi6_vals.append(psi_m_from_vectors(Wp, n, 6))
                psi4_vals.append(psi_m_from_vectors(Wp, n, 4))
        frac_exact = exact / len(cart)
        row = {
            "h": hkl[0], "k": hkl[1], "l": hkl[2],
            "frac_exact6": float(frac_exact),
            "mean_inplane_count": float(np.mean(counts)),
            "median_inplane_count": float(np.median(counts)),
            "mean_psi6_exact6": float(np.mean(psi6_vals)) if psi6_vals else 0.0,
            "mean_psi4_exact6": float(np.mean(psi4_vals)) if psi4_vals else 0.0,
        }
        # Lexicographic: exact-6 fraction first, then sixfold orientation.
        key = (row["frac_exact6"], row["mean_psi6_exact6"] - row["mean_psi4_exact6"], row["mean_psi6_exact6"])
        if best is None or key > best[0]:
            best = (key, row)
    if best is None:
        return {"n_X": len(cart)}
    ans = {"n_X": len(cart), **best[1]}
    ans["psi6_minus_psi4"] = ans["mean_psi6_exact6"] - ans["mean_psi4_exact6"]
    ans["psi6_over_psi4"] = ans["mean_psi6_exact6"] / max(ans["mean_psi4_exact6"], 1e-12)
    ans["strict_exact6_all"] = bool(ans["frac_exact6"] >= 1.0 - 1e-12)
    ans["fallback_sixfold"] = bool(
        ans["mean_psi6_exact6"] >= 0.60
        and ans["psi6_minus_psi4"] >= 0.18
        and ans["psi6_over_psi4"] >= 1.35
    )
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    df, structs, raw_structs = load_dataset(Path(args.cif_root), Path(args.metadata))
    known = {f: select_idx(df, f, mid) for f, mid in KNOWN_ID_BY_FORMULA.items()}
    blind = {f: select_idx(df, f, mid) for f, mid in BLIND_ID_BY_FORMULA.items()}
    known = {f: i for f, i in known.items() if i is not None}
    blind = {f: i for f, i in blind.items() if i is not None}
    known_set, blind_set = set(known.values()), set(blind.values())
    bg = [i for i in range(len(df)) if i not in known_set and i not in blind_set]

    # ---- Full-SOAP discovery frozen first ----
    soap = build_soap(SHORT_CFG)
    raw_local = local_soap_raw(soap, structs)
    full_local = [normalized_rows(x) for x in raw_local]
    recurrent = discover_recurrent_motif(full_local, structs, known, bg)
    motif = recurrent[0]
    oi, os = int(motif["origin_index"]), int(motif["site_index"])
    proto_full = full_local[oi][os]
    proto_raw = raw_local[oi][os]

    # ---- Species-pair block ablation ----
    blocks = {
        "MM": get_pair_slice(soap, 1, 1),
        "MX": get_pair_slice(soap, 1, 2),
        "XX": get_pair_slice(soap, 2, 2),
    }
    ablation_rows = []
    all_groups = [("known", known), ("blind", blind)]
    for block, slc in blocks.items():
        bg_scores = []
        for i in bg:
            _, s = best_block_match(raw_local, i, proto_raw, slc)
            bg_scores.append(s)
        bg_scores = np.asarray(bg_scores, float)
        for group, mapping in all_groups:
            for f, i in mapping.items():
                site, score = best_block_match(raw_local, i, proto_raw, slc)
                ablation_rows.append({
                    "block": block, "group": group, "formula": f,
                    "material_id": df.loc[i, "material_id"], "matched_site": site,
                    "score": score, "percentile_vs_background": float(100*np.mean(bg_scores <= score)),
                })
    pd.DataFrame(ablation_rows).to_csv(out / "soap_species_channel_ablation.csv", index=False)

    # ---- Independent periodic triangular-layer topology screen ----
    topo_rows = []
    for i, raw in enumerate(raw_structs):
        r = triangular_layer_screen(raw)
        topo_rows.append({
            "material_id": df.loc[i, "material_id"], "formula": df.loc[i, "formula_norm"],
            "is_known": i in known_set, "is_blind": i in blind_set, **r,
        })
        if (i + 1) % 100 == 0:
            print(" topology", i + 1, "/", len(raw_structs))
    topo = pd.DataFrame(topo_rows)
    topo.to_csv(out / "triangular_layer_screen.csv", index=False)

    selected_rows = []
    for group, mapping in (("known", known), ("blind", blind)):
        for f, i in mapping.items():
            row = topo.iloc[i].to_dict(); row["group"] = group; row["target_formula"] = f
            selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(out / "triangular_layer_known_blind.csv", index=False)

    # Background prevalence for useful thresholds, reported rather than tuned.
    bgt = topo[~topo.is_known & ~topo.is_blind]
    stats = {
        "motif_origin": {
            "formula": motif["origin_formula"], "material_id": str(df.loc[oi, "material_id"]),
            "site": os, "center_type": "X" if structs[oi][os].specie.symbol == "He" else "M",
            "commonality_min": motif["commonality_min"], "background_prev95": motif["background_prev95"],
            "geometry": posthoc_geometry(structs[oi], os),
        },
        "topology_background": {
            "frac_exact6_ge_1": float(np.mean(bgt.frac_exact6 >= 1.0 - 1e-12)),
            "frac_exact6_ge_0.8": float(np.mean(bgt.frac_exact6 >= 0.8)),
            "frac_exact6_ge_0.5": float(np.mean(bgt.frac_exact6 >= 0.5)),
            "fallback_sixfold": float(np.mean(bgt.fallback_sixfold.astype(bool))),
        },
    }
    (out / "posthoc_validation.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Post-hoc motif validation\n\n"]
    lines.append("## SOAP species-channel ablation\n\n")
    abl = pd.DataFrame(ablation_rows)
    summ = abl.groupby(["block", "group"]).agg(mean_score=("score","mean"), mean_percentile=("percentile_vs_background","mean"), min_percentile=("percentile_vs_background","min")).reset_index()
    lines.append(summ.to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n\n## Independent halogen-layer topology screen\n\n")
    cols = ["group","target_formula","material_id","frac_exact6","mean_psi6_exact6","mean_psi4_exact6","psi6_minus_psi4","psi6_over_psi4","strict_exact6_all","fallback_sixfold","h","k","l"]
    lines.append(selected[cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n\nBackground prevalence: `" + json.dumps(stats["topology_background"]) + "`\n")
    lines.append("\nThis topology screen is post-hoc and independent of SOAP motif selection. It is a first-pass crystallographic-plane implementation, not yet a substitute for a fully validated layer-connectivity/exact-6 production detector.\n")
    (out / "POSTHOC_VALIDATION.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


if __name__ == "__main__":
    main()
