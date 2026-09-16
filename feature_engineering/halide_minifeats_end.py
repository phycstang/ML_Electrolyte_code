#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
halide_minifeats.py  (robust χM/η/ω; actinide-ready; keep id column)

- Atomic-only compact features for binary halides (M–X; X∈{F,Cl,Br,I}).
- χM / η / ω 计算顺序：
  (1) mendeleev: electronegativity_mulliken / hardness / electrophilicity
  (2) 若缺，用 IE1 & EA 计算
  (3) 若 EA 缺：U 覆盖 EA=0.309 eV；其它金属/锕系 EA≈0；或用 χ_P 反推 (IE1+EA)
- C6：先取 mendeleev 的 c6（hartree/bohr^6），缺时用 London 近似 (3/4)*alpha^2*I（a.u.）
- metallic radius fallback: r_met → r_cov → r_ion
- f-block (lanthanoid/actinoid) : M_group = 3.5（连续编码）
- 输出会保留你原 CSV 中的 `--id-col`（默认 'id'），以及可选的 formula 列。

Usage:
  python halide_minifeats.py --csv input.csv --out_csv feats.csv --id-col id --keep_id_cols
"""

from __future__ import annotations
import os, re, math, argparse
from typing import Dict, Tuple, Optional, Any, List
from functools import lru_cache
import multiprocessing as mp

import numpy as np
import pandas as pd

HALOGENS = {"F","Cl","Br","I"}
AU_EV = 27.211386245988  # eV per Hartree

# ---- optional parsers ----
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

# ---- utils ----
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

def _c6_london_from_alpha_IE1(alpha_au: Optional[float], IE1_eV: Optional[float]) -> Optional[float]:
    if not (_is_finite(alpha_au) and _is_finite(IE1_eV)):
        return None
    IE1_Ha = float(IE1_eV) / AU_EV
    if IE1_Ha <= 0:
        return None
    return 0.75 * (float(alpha_au) ** 2) * IE1_Ha  # (3/4) * alpha^2 * I

# ---- overrides for sparse actinide EA ----
# U: modern experimental EA ~ 0.309 eV (Ciborowski et al., JCP 2021)
ACTINIDE_EA_OVERRIDES = {"U": 0.309}

# ---- cached access ----
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

_d_pat = re.compile(r"(\d)([spdfg])(\d+)")
def _count_d_electrons(conf: str) -> Optional[int]:
    if not conf:
        return None
    try:
        total = 0
        for _, orb, occ in _d_pat.findall(conf):
            if orb == 'd':
                total += int(occ)
        return int(total)
    except Exception:
        return None

@lru_cache(maxsize=None)
def get_elem_numeric_props(sym: str) -> Dict[str, Optional[float]]:
    e = _md_elem(sym)
    out: Dict[str, Optional[float]] = {}

    # ---- electronegativity family ----
    for k in ("electronegativity_pauling", "en_pauling", "en"):
        v = getattr(e, k, None); v = v() if callable(v) else v
        fv = _to_float(v)
        if fv is not None:
            out["chiP"] = fv; out["en"] = fv; break

    v = getattr(e, "electronegativity_allred_rochow", None); v = v() if callable(v) else v
    out["chiAR"] = _to_float(v)
    v = getattr(e, "electronegativity_martynov_batsanov", None); v = v() if callable(v) else v
    out["chiMB"] = _to_float(v)

    # ---- IE1 / EA ----
    ion = getattr(e, "ionenergies", None); ion = ion() if callable(ion) else ion
    if isinstance(ion, dict) and 1 in ion:
        out["IE1"] = _to_float(ion.get(1))
    v = getattr(e, "electron_affinity", None); v = v() if callable(v) else v
    EA_val = _to_float(v)

    # override sparse EA for actinides when missing
    if EA_val is None and sym in ACTINIDE_EA_OVERRIDES:
        EA_val = float(ACTINIDE_EA_OVERRIDES[sym])
    out["EA"] = EA_val

    # ---- Mulliken χ / η / ω ----
    # Prefer library (if present in your mendeleev version)
    v = getattr(e, "electronegativity_mulliken", None); v = v() if callable(v) else v
    out["chiM"] = _to_float(v)
    v = getattr(e, "hardness", None); v = v() if callable(v) else v
    out["eta"] = _to_float(v)
    v = getattr(e, "electrophilicity", None); v = v() if callable(v) else v
    out["omega"] = _to_float(v)

    IE1, EA = out.get("IE1"), out.get("EA")

    # (a) compute from IE1 & EA if any missing
    if _is_finite(IE1) and _is_finite(EA):
        if not _is_finite(out.get("chiM")): out["chiM"] = 0.5*(float(IE1)+float(EA))
        if not _is_finite(out.get("eta")):  out["eta"]  = 0.5*(float(IE1)-float(EA))
        if not _is_finite(out.get("omega")) and _is_finite(out.get("eta")) and abs(out["eta"])>1e-12:
            out["omega"] = (out["chiM"]*out["chiM"])/(2.0*out["eta"])

    # (b) if EA still missing: use Pauling mapping to infer (IE1+EA)
    if not _is_finite(out.get("chiM")):
        chiP = out.get("chiP")
        if _is_finite(chiP):
            S = (float(chiP) - 0.17) / 0.187  # (IE1+EA) in eV
            out["chiM"] = 0.5 * S
            if not _is_finite(out.get("eta")):
                if _is_finite(IE1) and _is_finite(EA):
                    out["eta"] = 0.5*(float(IE1)-float(EA))
                elif _is_finite(IE1):
                    out["eta"] = 0.5*(float(IE1) - (S - float(IE1)))  # = IE1 - S/2
                elif _is_finite(EA):
                    out["eta"] = 0.5*((S - float(EA)) - float(EA))    # = S/2 - EA
                else:
                    out["eta"] = 0.5 * S  # conservative: EA≈0
            if not _is_finite(out.get("omega")) and _is_finite(out.get("eta")) and abs(out["eta"])>1e-12:
                out["omega"] = (out["chiM"]*out["chiM"])/(2.0*out["eta"])

    # ---- size / polarizability / dispersion ----
    v = getattr(e, "covalent_radius_pyykko", None); v = v() if callable(v) else v
    out["r_cov"] = _to_float(v)
    if not _is_finite(out["r_cov"] if "r_cov" in out else None):
        for alt in ("covalent_radius_bragg", "covalent_radius"):
            v = getattr(e, alt, None); v = v() if callable(v) else v
            out["r_cov"] = _to_float(v)
            if _is_finite(out["r_cov"]): break

    out["r_vdw"] = None
    for k in ("vdw_radius_alvarez","vdw_radius_bondi","vdw_radius_batsanov","vdw_radius"):
        v = getattr(e, k, None); v = v() if callable(v) else v
        fv = _to_float(v)
        if fv is not None:
            out["r_vdw"] = fv; break

    v = getattr(e, "metallic_radius", None); v = v() if callable(v) else v
    out["r_met"] = _to_float(v)

    v = getattr(e, "dipole_polarizability", None); v = v() if callable(v) else v
    out["alpha"] = _to_float(v)  # bohr^3

    v = getattr(e, "c6", None); v = v() if callable(v) else v
    out["C6"] = _to_float(v)  # hartree/bohr^6
    if not _is_finite(out.get("C6")):
        c6_est = _c6_london_from_alpha_IE1(out.get("alpha"), out.get("IE1"))
        if _is_finite(c6_est): out["C6"] = c6_est

    v = getattr(e, "atomic_volume", None); v = v() if callable(v) else v
    out["V_atom"] = _to_float(v)

    for k, outk in (("mendeleev_number","MN"), ("pettifor_number","PN")):
        v = getattr(e, k, None); v = v() if callable(v) else v
        out[outk] = _to_float(v)

    g = getattr(e, "group_id", getattr(e, "group", None)); g = g() if callable(g) else g
    out["group"] = _to_float(g)
    p = getattr(e, "period", None); p = p() if callable(p) else p
    out["period"] = _to_float(p)

    series = getattr(e, "series", None); series = series() if callable(series) else series
    out["series"] = series if isinstance(series, str) else None

    SERIES_TO_CODE = {
        "alkali metal": 1, "alkaline earth metal": 2, "transition metal": 3,
        "lanthanoid": 4, "actinoid": 5, "metalloid": 6, "post-transition metal": 7,
        "nonmetal": 8, "halogen": 9, "noble gas": 10, "unknown": 0
    }
    out["series_code"] = float(SERIES_TO_CODE.get(series, 0)) if series else np.nan

    conf = getattr(e, "electronic_configuration", None); conf = conf() if callable(conf) else conf
    out["n_d_electrons"] = _count_d_electrons(conf) if conf else None
    return out

@lru_cache(maxsize=None)
def guess_valence(sym: str) -> Optional[int]:
    e = _md_elem(sym)
    ox = getattr(e, "oxistates", None) or getattr(e, "oxidation_states", None) or []
    try: oxs = list(ox)
    except Exception: oxs = []
    if sym in HALOGENS:
        neg = [o for o in oxs if isinstance(o,(int,float)) and o<0]
        return int(sorted(neg, key=lambda x: abs(x))[0]) if neg else -1
    pos = [o for o in oxs if isinstance(o,(int,float)) and o>0]
    if pos: return int(sorted(pos, key=lambda x: abs(x))[0])
    g = getattr(e, "group_id", getattr(e, "group", None))
    try: g = int(g)
    except Exception: g = None
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

# ---- feature lists ----
BASE_FEATURES = [
    "H_chiP","comp_entropy","comp_x_over_m",
    "mh_d_chiP","mh_d_mend_chiAR","mh_d_mend_chiMB",
    "M_IE1","H_EA","chi_geomean_comp","feat_field_strength_mean",
    "mh_r_r_ion","M_r_ion","mh_r_r_cov","M_r_met","M_group","H_alpha",
    "mend_alpha_wmean","M_C6","H_C6","C6_geomean_MH","M_MN","H_MN"
]
EXTRA_FEATURES = [
    "M_chiM","M_eta","M_omega",
    "M_Z2_over_r2",
    "M_PN","H_PN","M_period","H_period",
    "M_n_d_electrons",
    "alpha_ratio","alpha_vol_norm",
    "M_series_code","H_series_code"
]
DELTA_FEATURES = [
    "comp_entropy","comp_x_over_m","chi_geomean_comp","mend_alpha_wmean",
    "mh_d_chiP","feat_field_strength_mean","C6_geomean_MH","mh_r_r_ion",
]
DOPED_ABSOLUTE_KEYS = BASE_FEATURES + EXTRA_FEATURES

# ---- helpers ----
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

    props = {s: get_elem_numeric_props(s) for s in comp.keys()}
    pM, pX = props[M], props[X]
    out: Dict[str, float] = {}

    # A. halogen identity & stoichiometry
    out["H_chiP"] = float(pX.get("chiP")) if _is_finite(pX.get("chiP")) else np.nan
    out["comp_entropy"] = shannon_entropy(list(comp.values()))
    out["comp_x_over_m"] = float(xX/xM) if (xM and xM>0) else np.nan

    # B. EN contrasts
    def _abs_diff(pm: Dict[str,Any], px: Dict[str,Any], key: str) -> float:
        a, b = pm.get(key), px.get(key)
        return float(abs(float(a)-float(b))) if _is_finite(a) and _is_finite(b) else np.nan
    out["mh_d_chiP"] = _abs_diff(pM, pX, "chiP")
    out["mh_d_mend_chiAR"] = _abs_diff(pM, pX, "chiAR")
    out["mh_d_mend_chiMB"] = _abs_diff(pM, pX, "chiMB")

    out["M_IE1"] = float(pM.get("IE1")) if _is_finite(pM.get("IE1")) else np.nan
    out["H_EA"]  = float(pX.get("EA"))  if _is_finite(pX.get("EA"))  else np.nan

    # C. Sanderson geomean (composition-wide)
    chis_vals = [(float(props[s].get("en")), float(f))
                 for s, f in comp.items()
                 if _is_finite(props[s].get("en")) and props[s].get("en")>0 and f>0]
    out["chi_geomean_comp"] = float(np.exp(sum(fr*np.log(chi) for chi, fr in chis_vals))) if chis_vals else np.nan

    # D. field strength (cation-only): sum f * |z|/r^2
    num = 0.0
    for s, f in comp.items():
        if s in HALOGENS:  # cations only
            continue
        z = guess_valence(s)
        r = pick_ionic_radius(s, z)
        if _is_finite(r) and r>0 and _is_finite(z) and f>0:
            num += float(f) * (abs(float(z))/(float(r)**2))
    out["feat_field_strength_mean"] = float(num) if num>0 else 0.0

    # E. size/contact
    rM_ion = pick_ionic_radius(M, guess_valence(M))
    rX_ion = pick_ionic_radius(X, -1)
    out["mh_r_r_ion"] = float(rM_ion/rX_ion) if _is_finite(rM_ion) and _is_finite(rX_ion) and rX_ion>1e-12 else np.nan
    out["M_r_ion"]  = float(rM_ion) if _is_finite(rM_ion) else np.nan

    rM_cov = pM.get("r_cov"); rX_cov = pX.get("r_cov")
    out["mh_r_r_cov"] = float(rM_cov/rX_cov) if _is_finite(rM_cov) and _is_finite(rX_cov) and rX_cov>1e-12 else np.nan

    # metallic radius: r_met -> r_cov -> r_ion
    r_met = pM.get("r_met")
    if not _is_finite(r_met):
        r_met = pM.get("r_cov") if _is_finite(pM.get("r_cov")) else (rM_ion if _is_finite(rM_ion) else np.nan)
    out["M_r_met"] = float(r_met) if _is_finite(r_met) else np.nan

    # f-block group = 3.5; else original group
    seriesM = pM.get("series")
    g_raw = pM.get("group")
    out["M_group"] = 3.5 if isinstance(seriesM, str) and seriesM in ("lanthanoid", "actinoid") \
                     else (float(g_raw) if _is_finite(g_raw) else np.nan)

    out["H_alpha"] = float(pX.get("alpha")) if _is_finite(pX.get("alpha")) else np.nan

    # F. additive & ordinal
    a_sum = 0.0; any_alpha = False
    for s, f in comp.items():
        a = props[s].get("alpha", None)
        if _is_finite(a) and f>0:
            a_sum += float(f) * float(a); any_alpha = True
    out["mend_alpha_wmean"] = float(a_sum) if any_alpha else np.nan

    out["M_C6"] = float(pM.get("C6")) if _is_finite(pM.get("C6")) else np.nan
    out["H_C6"] = float(pX.get("C6")) if _is_finite(pX.get("C6")) else np.nan
    out["C6_geomean_MH"] = geomean(out["M_C6"], out["H_C6"])

    out["M_MN"] = float(pM.get("MN")) if _is_finite(pM.get("MN")) else np.nan
    out["H_MN"] = float(pX.get("MN")) if _is_finite(pX.get("MN")) else np.nan

    # ------- EXTRA -------
    out["M_chiM"] = float(pM.get("chiM")) if _is_finite(pM.get("chiM")) else np.nan
    out["M_eta"]  = float(pM.get("eta"))  if _is_finite(pM.get("eta"))  else np.nan
    out["M_omega"] = float(pM.get("omega")) if _is_finite(pM.get("omega")) else np.nan

    zM = guess_valence(M); rM = rM_ion
    out["M_Z2_over_r2"] = float((float(zM)**2)/(float(rM)**2)) if _is_finite(zM) and _is_finite(rM) and rM>1e-12 else np.nan

    out["M_PN"] = float(pM.get("PN")) if _is_finite(pM.get("PN")) else np.nan
    out["H_PN"] = float(pX.get("PN")) if _is_finite(pX.get("PN")) else np.nan
    out["M_period"] = float(pM.get("period")) if _is_finite(pM.get("period")) else np.nan
    out["H_period"] = float(pX.get("period")) if _is_finite(pX.get("period")) else np.nan

    nde = pM.get("n_d_electrons")
    out["M_n_d_electrons"] = float(nde) if _is_finite(nde) else np.nan

    aM = pM.get("alpha"); aX = pX.get("alpha"); rcovM = pM.get("r_cov")
    out["alpha_ratio"] = float(aM/aX) if _is_finite(aM) and _is_finite(aX) and aX>1e-12 else np.nan
    out["alpha_vol_norm"] = float(aM/(rcovM**3)) if _is_finite(aM) and _is_finite(rcovM) and rcovM>1e-12 else np.nan

    out["M_series_code"] = float(pM.get("series_code")) if _is_finite(pM.get("series_code")) else np.nan
    out["H_series_code"] = float(pX.get("series_code")) if _is_finite(pX.get("series_code")) else np.nan

    return out, None

# ---- worker ----
def _process_row_pack(pack) -> Tuple[int, Dict[str, Any]]:
    idx, row_dict, args = pack
    formula_col, cif_col = args["formula_col"], args["cif_path_col"]
    cif_root, virt_dope, output_doped, prefer_fast, allow_cif = (
        args["cif_root"], args["virt_dope"], args["output_doped"],
        args["prefer_fast"], args["allow_cif"]
    )

    # get formula
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

    # compute
    if not formula_str:
        base = {k: np.nan for k in BASE_FEATURES + EXTRA_FEATURES}
        doped_abs, delta = {}, {}
    else:
        comp = parse_formula(formula_str, prefer_fast=prefer_fast)
        base, err = compute_features_for_comp(comp)
        if err is not None:
            base = {k: np.nan for k in BASE_FEATURES + EXTRA_FEATURES}

        doped_abs, delta = {}, {}
        if virt_dope is not None:
            o_frac, na_frac = virt_dope
            comp_doped = comp.copy()
            comp_doped["O"]  = comp_doped.get("O", 0.0)  + max(0.0, o_frac)
            comp_doped["Na"] = comp_doped.get("Na", 0.0) + max(0.0, na_frac)
            comp_doped = comp_norm(comp_doped)

            doped_abs, err2 = compute_features_for_comp(comp_doped)
            if err2 is not None:
                doped_abs = {k: np.nan for k in BASE_FEATURES + EXTRA_FEATURES}

            for k in DELTA_FEATURES:
                v0 = base.get(k, np.nan); v1 = doped_abs.get(k, np.nan)
                delta[f"virt_d_{k}"] = (v1 - v0) if _is_finite(v0) and _is_finite(v1) else np.nan

    row_out = {**{k: np.nan for k in BASE_FEATURES + EXTRA_FEATURES}, **base}
    if virt_dope is not None:
        row_out.update(delta)
        if output_doped:
            row_out.update({f"doped_{k}": doped_abs.get(k, np.nan) for k in DOPED_ABSOLUTE_KEYS})
    return idx, row_out

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV. Needs a 'formula' column or a 'cif_path' column.")
    ap.add_argument("--out_csv", default="halide_feats.csv", help="Output CSV path")
    ap.add_argument("--formula_col", default="formula", help="Column containing chemical formula")
    ap.add_argument("--cif_path_col", default="cif_path", help="Optional CIF path column")
    ap.add_argument("--cif_root", default=".", help="Root folder when cif paths are relative")
    ap.add_argument("--keep_id_cols", action="store_true", help="Also include 'formula' in output")
    ap.add_argument("--id-col", default="id", help="Name of ID column to pass through if present (default: id)")
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

    records_packed = [(i, r, worker_args) for i, r in enumerate(df.to_dict(orient="records"))]
    if args.threads and args.threads > 1:
        with mp.get_context("spawn").Pool(processes=args.threads) as pool:
            results = list(pool.imap(_process_row_pack, records_packed, chunksize=64))
    else:
        results = [_process_row_pack(t) for t in records_packed]

    results_sorted = [None]*len(results)
    for idx, row_out in results:
        results_sorted[idx] = row_out
    feats_df = pd.DataFrame(results_sorted)

    # assemble output: id column (if present) + (optional) formula + features
    pieces = []
    if args.id_col in df.columns:
        pieces.append(df[[args.id_col]])
    if args.keep_id_cols and args.formula_col in df.columns:
        pieces.append(pd.DataFrame({args.formula_col: df[args.formula_col]}))
    pieces.append(feats_df)
    res = pd.concat(pieces, axis=1) if pieces else feats_df

    res.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} (rows={len(res)}, cols={res.shape[1]})")

if __name__ == "__main__":
    main()
