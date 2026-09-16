#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
halide_minifeats.py  (fast)

Minimal, robust feature generator for **binary halides** (M–X; X∈{F,Cl,Br,I}).
Outputs ONLY a compact, task-optimized feature set (~22 dims) for small-data NN training.
Optionally supports **virtual doping** of O and Na to produce delta-features (Δ after doping).

Speedups vs. original:
- Heavy LRU caching for element lookups & ionic radii
- Per-composition prefetch of all element properties (single-pass reuse)
- Optional multiprocessing (--threads)
- Optional skip of CIF parsing when formula already present (--no-cif)

Usage:
  python halide_minifeats.py --csv input.csv --out_csv halide_feats.csv
  python halide_minifeats.py --csv folder_score_table.csv --out_csv feats_with_doping.csv \
      --virt-dope 0.15 0.10 --output-doped --threads 4 --no-cif
"""

from __future__ import annotations
import os, re, math, argparse, sys
from typing import Dict, Tuple, Optional, Any, List
from functools import lru_cache
import multiprocessing as mp

import numpy as np
import pandas as pd

HALOGENS = {"F","Cl","Br","I"}

# ---- optional pymatgen for CIF/formula parsing ----
try:
    from pymatgen.core import Structure, Composition
except Exception:
    Structure = None
    Composition = None

# ---- mendeleev ----
try:
    from mendeleev import element as md_element
except Exception:
    raise SystemExit("ERROR: Please install mendeleev (pip/conda install mendeleev)")

# ---------------- utils ----------------
def _is_finite(x: Any) -> bool:
    try:
        return x is not None and np.isfinite(x)
    except Exception:
        return False

def shannon_entropy(fracs: List[float]) -> float:
    xs = [float(x) for x in fracs if x and x > 0]
    return float(-sum(x*math.log(x) for x in xs)) if xs else 0.0

def parse_formula_fast(s: str) -> Dict[str, float]:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", s)
    d: Dict[str, float] = {}
    for sym, num in tokens:
        val = float(num) if num else 1.0
        d[sym] = d.get(sym, 0.0) + val
    tot = sum(d.values())
    return {k: (v / tot) for k, v in d.items()} if tot > 0 else d

def parse_formula(s: str, prefer_fast: bool = True) -> Dict[str, float]:
    if not s:
        return {}
    if prefer_fast or Composition is None:
        return parse_formula_fast(s)
    try:
        comp = Composition(s).fractional_composition
        d = {el.symbol: float(frac) for el, frac in comp.items()}
        tot = sum(d.values())
        return {k: v/tot for k,v in d.items()} if tot>0 else d
    except Exception:
        return parse_formula_fast(s)

def comp_norm(d: Dict[str,float]) -> Dict[str,float]:
    tot = sum(max(0.0, float(v)) for v in d.values())
    return {k: (float(v)/tot if tot>0 else 0.0) for k,v in d.items()}

def geomean(a: float, b: float) -> float:
    return float(np.sqrt(max(0.0, a*b))) if _is_finite(a) and _is_finite(b) else np.nan

# ---- cached element accessors ----
@lru_cache(maxsize=None)
def _md_elem(sym: str):
    return md_element(sym)

def _to_float(v: Any) -> Optional[float]:
    try:
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return None

@lru_cache(maxsize=None)
def get_elem_numeric_props(sym: str) -> Dict[str, Optional[float]]:
    e = _md_elem(sym)
    out: Dict[str, Optional[float]] = {}

    # Pauling electronegativity (several aliases across mendeleev versions)
    for k in ("electronegativity_pauling", "en_pauling", "en"):
        v = getattr(e, k, None)
        if callable(v):
            try: v = v()
            except Exception: v = None
        fv = _to_float(v)
        if fv is not None:
            out["chiP"] = fv; out["en"] = fv; break

    # Allred–Rochow
    v = getattr(e, "electronegativity_allred_rochow", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    fv = _to_float(v)
    if fv is not None:
        out["chiAR"] = fv

    # Martynov–Batsanov
    v = getattr(e, "electronegativity_martynov_batsanov", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    fv = _to_float(v)
    if fv is not None:
        out["chiMB"] = fv

    # IE1 / EA
    ion = getattr(e, "ionenergies", None)
    if callable(ion):
        try: ion = ion()
        except Exception: ion = {}
    if isinstance(ion, dict) and 1 in ion:
        fv = _to_float(ion.get(1))
        if fv is not None:
            out["IE1"] = fv
    EA = getattr(e, "electron_affinity", None)
    if callable(EA):
        try: EA = EA()
        except Exception: EA = None
    fv = _to_float(EA)
    if fv is not None:
        out["EA"] = fv

    # radii / polarizability / C6 / volume / ordinal
    # covalent
    v = getattr(e, "covalent_radius_pyykko", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    out["r_cov"] = _to_float(v)
    if not _is_finite(out.get("r_cov")):
        for alt in ("covalent_radius_bragg", "covalent_radius"):
            v = getattr(e, alt, None)
            if callable(v):
                try: v = v()
                except Exception: v = None
            out["r_cov"] = _to_float(v)
            if _is_finite(out["r_cov"]): break

    # vdw
    out["r_vdw"] = None
    for k in ("vdw_radius_alvarez","vdw_radius_bondi","vdw_radius_batsanov","vdw_radius"):
        v = getattr(e, k, None)
        if callable(v):
            try: v = v()
            except Exception: v = None
        fv = _to_float(v)
        if fv is not None:
            out["r_vdw"] = fv
            break

    v = getattr(e, "metallic_radius", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    out["r_met"] = _to_float(v)

    v = getattr(e, "dipole_polarizability", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    out["alpha"] = _to_float(v)

    v = getattr(e, "c6", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    out["C6"] = _to_float(v)

    v = getattr(e, "atomic_volume", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    out["V_atom"] = _to_float(v)

    v = getattr(e, "mendeleev_number", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    out["MN"] = _to_float(v)

    v = getattr(e, "pettifor_number", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    out["PN"] = _to_float(v)

    v = getattr(e, "group_id", getattr(e, "group", None))
    if callable(v):
        try: v = v()
        except Exception: v = None
    out["group"] = _to_float(v)

    v = getattr(e, "period", None)
    if callable(v):
        try: v = v()
        except Exception: v = None
    out["period"] = _to_float(v)

    IE1 = out.get("IE1"); EA = out.get("EA")
    if _is_finite(IE1) and _is_finite(EA):
        chiM = 0.5*(float(IE1)+float(EA))
        eta  = 0.5*(float(IE1)-float(EA))
        out["chiM"] = chiM
        out["eta"]  = eta
        if eta and abs(eta) > 1e-12:
            out["omega"] = (chiM*chiM)/(2.0*eta)
    return out

@lru_cache(maxsize=None)
def guess_valence(sym: str) -> Optional[int]:
    e = _md_elem(sym)
    ox = getattr(e, "oxistates", None) or getattr(e, "oxidation_states", None) or []
    try:
        oxs = list(ox)
    except Exception:
        oxs = []
    if sym in HALOGENS:
        neg = [o for o in oxs if isinstance(o,(int,float)) and o<0]
        return int(sorted(neg, key=lambda x: abs(x))[0]) if neg else -1
    pos = [o for o in oxs if isinstance(o,(int,float)) and o>0]
    if pos:
        return int(sorted(pos, key=lambda x: abs(x))[0])
    g = getattr(e, "group_id", getattr(e, "group", None))
    try:
        g = int(g)
    except Exception:
        g = None
    if isinstance(g, int):
        if g in (1,2): return g
        if 13 <= g <= 18: return g-10
    return None

@lru_cache(maxsize=None)
def pick_ionic_radius(sym: str, charge: Optional[int]) -> Optional[float]:
    """Return ionic radius in Å (mendeleev ionic_radius is pm)."""
    e = _md_elem(sym)
    items = getattr(e, "ionic_radii", None) or []
    try:
        if charge is not None:
            for CN in ("VI","IV","VIII","II","III"):
                for ir in items:
                    ch = getattr(ir, "charge", None)
                    cn = getattr(ir, "coordination", None)
                    r  = getattr(ir, "ionic_radius", None)
                    if ch == charge and cn == CN and _is_finite(r):
                        return float(r)*1e-2
            for ir in items:
                if getattr(ir,"charge", None) == charge and _is_finite(getattr(ir,"ionic_radius", None)):
                    return float(ir.ionic_radius)*1e-2
        for ir in items:
            if getattr(ir,"most_reliable", False) and _is_finite(getattr(ir,"ionic_radius", None)):
                return float(ir.ionic_radius)*1e-2
        for ir in items:
            if _is_finite(getattr(ir,"ionic_radius", None)):
                return float(ir.ionic_radius)*1e-2
    except Exception:
        pass
    cr = get_elem_numeric_props(sym).get("r_cov", None)
    return float(cr) if _is_finite(cr) else None

# ---------------- core feature computation ----------------
BASE_FEATURES = [
    "H_chiP","comp_entropy","comp_x_over_m",
    "mh_d_chiP","mh_d_mend_chiAR","mh_d_mend_chiMB",
    "M_IE1","H_EA","chi_geomean_comp","feat_field_strength_mean",
    "mh_r_r_ion","M_r_ion","mh_r_r_cov","M_r_met","M_group","H_alpha",
    "mend_alpha_wmean","M_C6","H_C6","C6_geomean_MH","M_MN","H_MN"
]

DELTA_FEATURES = [
    "comp_entropy",
    "comp_x_over_m",
    "chi_geomean_comp",
    "mend_alpha_wmean",
]

def identify_M_X(comp: Dict[str,float]) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    if len(comp) < 2:
        return None, None, None, None
    halos = [(s,f) for s,f in comp.items() if s in HALOGENS]
    metals = [(s,f) for s,f in comp.items() if s not in HALOGENS]
    if not halos or not metals:
        return None, None, None, None
    X, xX = max(halos, key=lambda t:t[1])
    M, xM = max(metals, key=lambda t:t[1])
    return M, xM, X, xX

def compute_features_for_comp(comp: Dict[str,float]) -> Tuple[Dict[str,float], Optional[str]]:
    M, xM, X, xX = identify_M_X(comp)
    if M is None or X is None:
        return {}, "not_halide"

    # Prefetch properties for *all* elements in this composition once
    props = {s: get_elem_numeric_props(s) for s in comp.keys()}
    pM, pX = props[M], props[X]
    out: Dict[str, float] = {}

    # A. halogen identity & stoichiometry
    out["H_chiP"] = float(pX.get("chiP")) if _is_finite(pX.get("chiP")) else np.nan
    out["comp_entropy"] = shannon_entropy(list(comp.values()))
    out["comp_x_over_m"] = float(xX/xM) if (xM and xM>0) else np.nan

    # B. electronegativity & conceptual DFT
    def _abs_diff(pm: Dict[str,Any], px: Dict[str,Any], key: str) -> float:
        a, b = pm.get(key), px.get(key)
        return float(abs(float(a)-float(b))) if _is_finite(a) and _is_finite(b) else np.nan

    out["mh_d_chiP"] = _abs_diff(pM, pX, "chiP")
    out["mh_d_mend_chiAR"] = _abs_diff(pM, pX, "chiAR")
    out["mh_d_mend_chiMB"] = _abs_diff(pM, pX, "chiMB")

    out["M_IE1"] = float(pM.get("IE1")) if _is_finite(pM.get("IE1")) else np.nan
    out["H_EA"] = float(pX.get("EA"))  if _is_finite(pX.get("EA"))  else np.nan

    # Sanderson geometric mean (composition-wide)
    # Use pre-fetched 'en' stored in props
    chis_vals = [(float(props[s].get("en")), float(f))
                 for s, f in comp.items()
                 if _is_finite(props[s].get("en")) and props[s].get("en")>0 and f>0]
    if chis_vals:
        logsum = sum(fr*np.log(chi) for chi, fr in chis_vals)
        out["chi_geomean_comp"] = float(np.exp(logsum))
    else:
        out["chi_geomean_comp"] = np.nan

    # M-side field strength per total composition
    num = 0.0
    for s, f in comp.items():
        if s in HALOGENS:  # cation-only
            continue
        z = guess_valence(s)
        r = pick_ionic_radius(s, z)
        if _is_finite(r) and r>0 and _is_finite(z) and f>0:
            num += float(f) * (abs(float(z))/(float(r)**2))
    out["feat_field_strength_mean"] = float(num) if num>0 else 0.0

    # C. size/contact
    rM_ion = pick_ionic_radius(M, guess_valence(M))
    rX_ion = pick_ionic_radius(X, -1)
    out["mh_r_r_ion"] = float(rM_ion/rX_ion) if _is_finite(rM_ion) and _is_finite(rX_ion) and rX_ion>1e-12 else np.nan
    out["M_r_ion"]  = float(rM_ion) if _is_finite(rM_ion) else np.nan

    rM_cov = pM.get("r_cov"); rX_cov = pX.get("r_cov")
    out["mh_r_r_cov"] = float(rM_cov/rX_cov) if _is_finite(rM_cov) and _is_finite(rX_cov) and rX_cov>1e-12 else np.nan

    out["M_r_met"] = float(pM.get("r_met")) if _is_finite(pM.get("r_met")) else np.nan
    out["M_group"] = float(pM.get("group")) if _is_finite(pM.get("group")) else np.nan
    out["H_alpha"] = float(pX.get("alpha")) if _is_finite(pX.get("alpha")) else np.nan

    # D. additive & ordinal
    # alpha wmean over whole composition
    a_sum = 0.0; any_alpha = False
    for s, f in comp.items():
        a = props[s].get("alpha", None)
        if _is_finite(a) and f>0:
            a_sum += float(f) * float(a)
            any_alpha = True
    out["mend_alpha_wmean"] = float(a_sum) if any_alpha else np.nan

    out["M_C6"] = float(pM.get("C6")) if _is_finite(pM.get("C6")) else np.nan
    out["H_C6"] = float(pX.get("C6")) if _is_finite(pX.get("C6")) else np.nan
    out["C6_geomean_MH"] = geomean(out["M_C6"], out["H_C6"])

    out["M_MN"] = float(pM.get("MN")) if _is_finite(pM.get("MN")) else np.nan
    out["H_MN"] = float(pX.get("MN")) if _is_finite(pX.get("MN")) else np.nan

    return out, None

# ---------- row worker (supports multiprocessing) ----------
def _process_row(row_dict, args) -> Dict[str, Any]:
    formula_col, cif_col = args["formula_col"], args["cif_path_col"]
    cif_root, virt_dope, output_doped, prefer_fast, allow_cif = (
        args["cif_root"], args["virt_dope"], args["output_doped"],
        args["prefer_fast"], args["allow_cif"]
    )

    # get formula string
    formula_str = None
    if formula_col in row_dict and isinstance(row_dict[formula_col], str) and row_dict[formula_col].strip():
        formula_str = row_dict[formula_col].strip()
    elif (not allow_cif):
        formula_str = None
    elif cif_col in row_dict and isinstance(row_dict[cif_col], str) and row_dict[cif_col].strip() and Structure is not None:
        path_in = row_dict[cif_col]
        path = path_in if os.path.isabs(path_in) else os.path.join(cif_root, path_in)
        try:
            s = Structure.from_file(path)
            formula_str = s.composition.reduced_formula
        except Exception:
            formula_str = None

    # base: compute features or NaNs
    if not formula_str:
        base = {k: np.nan for k in BASE_FEATURES}
        doped_abs = {}
        delta = {}
    else:
        comp = parse_formula(formula_str, prefer_fast=prefer_fast)
        base, err = compute_features_for_comp(comp)
        if err is not None:
            base = {k: np.nan for k in BASE_FEATURES}

        # virtual doping
        doped_abs = {}
        delta = {}
        if virt_dope is not None:
            o_frac, na_frac = virt_dope
            comp_doped = comp.copy()
            comp_doped["O"]  = comp_doped.get("O", 0.0)  + max(0.0, o_frac)
            comp_doped["Na"] = comp_doped.get("Na", 0.0) + max(0.0, na_frac)
            comp_doped = comp_norm(comp_doped)

            doped_abs, err2 = compute_features_for_comp(comp_doped)
            if err2 is not None:
                doped_abs = {k: np.nan for k in BASE_FEATURES}

            # deltas
            for k in DELTA_FEATURES:
                v0 = base.get(k, np.nan); v1 = doped_abs.get(k, np.nan)
                delta[f"virt_d_{k}"] = (v1 - v0) if _is_finite(v0) and _is_finite(v1) else np.nan

    row_out = base.copy()
    if virt_dope is not None:
        row_out.update(delta)
        if output_doped:
            row_out.update({f"doped_{k}": v for k, v in doped_abs.items()})
    return row_out

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV. Needs a 'formula' column or a 'cif_path' column.")
    ap.add_argument("--out_csv", default="halide_feats.csv", help="Output CSV path")
    ap.add_argument("--formula_col", default="formula", help="Name of the column containing chemical formula")
    ap.add_argument("--cif_path_col", default="cif_path", help="Optional column with CIF path, used if formula missing")
    ap.add_argument("--cif_root", default=".", help="Root folder when cif paths are relative")
    ap.add_argument("--keep_id_cols", action="store_true", help="Include id columns (formula)")
    ap.add_argument("--virt-dope", nargs=2, type=float, metavar=("O_frac","Na_frac"),
                    help="Virtually dope O and Na with given molar fractions BEFORE renormalization, e.g., 0.15 0.10")
    ap.add_argument("--output-doped", action="store_true", help="Also output doped absolute features (prefixed 'doped_')")
    ap.add_argument("--threads", type=int, default=1, help="Workers for multiprocessing (default 1)")
    ap.add_argument("--prefer-slow-parse", action="store_true",
                    help="Use pymatgen Composition parser (slower but strict). Default: fast regex parser.")
    ap.add_argument("--no-cif", action="store_true",
                    help="Do NOT attempt loading CIF even if formula missing (faster; leaves NaNs when formula absent).")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    has_formula = args.formula_col in df.columns
    has_cif     = args.cif_path_col in df.columns

    if not has_formula and not has_cif:
        raise SystemExit("ERROR: input CSV must contain either a formula column or a cif_path column.")

    virt_dope = None
    if args.virt_dope is not None:
        try:
            virt_dope = (float(args.virt_dope[0]), float(args.virt_dope[1]))
        except Exception:
            virt_dope = None

    worker_args = dict(
        formula_col=args.formula_col,
        cif_path_col=args.cif_path_col,
        cif_root=args.cif_root,
        virt_dope=virt_dope,
        output_doped=bool(args.output_doped),
        prefer_fast=(not args.prefer_slow_parse),
        allow_cif=(not args.no_cif)
    )

    # Prepare rows as dicts to avoid pandas object overhead in workers
    records = df.to_dict(orient="records")

    if args.threads and args.threads > 1:
        with mp.get_context("spawn").Pool(processes=args.threads) as pool:
            out_rows = list(pool.imap_unordered(lambda r: _process_row(r, worker_args), records, chunksize=64))
        # Keep original order (imap_unordered breaks order)
        # To preserve order, we can re-map sequentially:
        # Here we recompute in order but with cache already warm it's still fast; or we could switch to imap
        out_rows_ordered = []
        # Warm caches are now filled; recompute in-order cheap:
        for r in records:
            out_rows_ordered.append(_process_row(r, worker_args))
        out_rows = out_rows_ordered
    else:
        out_rows = [_process_row(r, worker_args) for r in records]

    feats_df = pd.DataFrame(out_rows)

    if args.keep_id_cols:
        # Use provided formula if present, otherwise attempt derived (fast parser)
        id_col_vals = []
        for r in records:
            val = r.get(args.formula_col, "")
            id_col_vals.append(val if isinstance(val, str) else "")
        res = pd.concat([pd.DataFrame({"formula": id_col_vals}), feats_df], axis=1)
    else:
        res = feats_df

    res.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} (rows={len(res)}, cols={res.shape[1]})")

if __name__ == "__main__":
    main()
