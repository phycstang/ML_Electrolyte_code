#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_features_valence_IE_radius_FS.py

在 make_features_valence_IE_radius.py 的基础上，新增：
- 阳离子场强 Cation Field Strength (CFS) = Z / r^2
  * 输出：T0_M__field_strength__ionic_Ainv2、__crystal_Ainv2、__primary_Ainv2
其它逻辑保持一致：
- 金属 M：电离能 sum(IE1..IE_valence)；半径按 (valence, CN) 精确选择；
- 卤素 H：不导出 IE 与 r_ion 两类；
- 保留 elements 数值列（M/H 两套）、元素比值、pairwise（对共有列）；
- 可选 T1/T2。
"""

import argparse, re, os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from mendeleev.fetch import (
    fetch_table,
    fetch_ionization_energies,
    fetch_ionic_radii,
)
from mendeleev import element as get_element

try:
    from pymatgen.core.composition import Composition
    from pymatgen.core import Structure
    PMG = True
except Exception:
    PMG = False
    Structure = None

HALOGENS = {"F","Cl","Br","I","At","Ts"}
CN_MAP = {4: "IV", 6: "VI"}

EXCLUDE_EXACT = {
    "abundance_crust","abundance_sea","relative_supply_risk","price_per_kg",
    "production_concentration","reserve_distribution","recycling_rate",
}
EXCLUDE_PATTERNS = [r"abund", r"supply", r"price", r"reserve", r"recycl", r"prod(_|)conc"]

def normalize_symbol(s):
    if not isinstance(s, str): return None
    s = re.sub(r"[^A-Za-z]", "", s.strip())
    if not s: return None
    return s[0].upper() + s[1:].lower()

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def col_wanted(col: str) -> bool:
    if col in EXCLUDE_EXACT: return False
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, col, flags=re.IGNORECASE):
            return False
    return True

def infer_from_formula(formula: str) -> Tuple[str,str,float,float]:
    M = H = None
    nM = nH = np.nan
    if not (PMG and isinstance(formula, str) and formula.strip()):
        return M,H,nM,nH
    try:
        comp = Composition(formula).get_el_amt_dict()
        for el, amt in comp.items():
            if el in HALOGENS and H is None:
                H, nH = el, float(amt)
            if el not in HALOGENS and M is None:
                M, nM = el, float(amt)
        return M,H,float(nM),float(nH)
    except Exception:
        return None,None,np.nan,np.nan

def infer_MH_stoich_and_cn(row, m_col, x_col) -> Tuple[str,str,float,float,str]:
    M = normalize_symbol(row.get(m_col)) if m_col else None
    H = normalize_symbol(row.get(x_col)) if x_col else None
    nM = nH = np.nan
    if (not M or not H):
        M2,H2,nM,nH = infer_from_formula(row.get("formula",""))
        M = M or M2; H = H or H2
    cn = None
    if "st1" in row:
        try:
            s = int(row.get("st1"))
            cn = CN_MAP.get(s, None)
        except Exception:
            cn = None
    if not np.isfinite(nM): nM = 1.0
    if not np.isfinite(nH): nH = 1.0
    return M,H,float(nM),float(nH),cn

def build_elements_numeric_cols(elements_df: pd.DataFrame) -> List[str]:
    cols = []
    for c in elements_df.columns:
        if c in ("name","symbol","econf","description","cas","discoverers",
                 "discovery_location","discovery_year","name_origin","uses",
                 "uses_description","cpk_color"):
            continue
        if not col_wanted(c): continue
        if is_numeric_series(elements_df[c]): cols.append(c)
    return cols

def melt_ir(df, kind):
    id_cols = ["atomic_number","charge"]
    cn_cols = [c for c in df.columns if c not in id_cols]
    return df.melt(id_vars=id_cols, value_vars=cn_cols, var_name="CN", value_name=kind)

def t1_features(struct: "Structure", M: str, H: str) -> Dict[str,float]:
    out = {}
    try:
        cutoff = 3.5
        cn, dists = [], []
        symbols = [str(s.specie) for s in struct]
        for i, si in enumerate(symbols):
            if si != M: continue
            local = []
            for j, sj in enumerate(symbols):
                if i==j or sj!=H: continue
                d = float(struct[i].distance(struct[j]))
                if d <= cutoff: local.append(d)
            if local:
                cn.append(len(local)); dists += local
        if cn:
            out["T1_env__M_CN_Hal_mean"] = float(np.mean(cn))
            out["T1_env__M_CN_Hal_std"]  = float(np.std(cn))
        if dists:
            out["T1_env__MHal_bond_length_A_mean"] = float(np.mean(dists))
            out["T1_env__MHal_bond_length_A_std"]  = float(np.std(dists))
    except Exception:
        pass
    return out

def t2_features(struct: "Structure", M: str, H: str) -> Dict[str,float]:
    out = {}
    try:
        cutoff = 4.0
        cn, dists = [], []
        symbols = [str(s.specie) for s in struct]
        for i, si in enumerate(symbols):
            if si != H: continue
            local = []
            for j, sj in enumerate(symbols):
                if i==j or sj!=M: continue
                d = float(struct[i].distance(struct[j]))
                if d <= cutoff: local.append(d)
            if local:
                cn.append(len(local)); dists += local
        if cn:
            out["T2_env__H_CN_M_mean"] = float(np.mean(cn))
            out["T2_env__H_CN_M_std"]  = float(np.std(cn))
        if dists:
            out["T2_env__HalM_bond_length_A_mean"] = float(np.mean(dists))
            out["T2_env__HalM_bond_length_A_std"]  = float(np.std(dists))
    except Exception:
        pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m-col", default=None)
    ap.add_argument("--x-col", default=None)
    ap.add_argument("--enable-t1", type=int, default=1)
    ap.add_argument("--enable-t2", type=int, default=1)
    ap.add_argument("--with_abs_delta", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    elements_df = fetch_table("elements")
    elem_cols = build_elements_numeric_cols(elements_df)
    sym2row = {row["symbol"]: row for _, row in elements_df.iterrows() if isinstance(row.get("symbol",None), str)}

    ies = fetch_ionization_energies(degree=list(range(1, 11))).reset_index()
    ir_ionic  = fetch_ionic_radii(radius="ionic_radius").reset_index()
    ir_cryst  = fetch_ionic_radii(radius="crystal_radius").reset_index()
    ir_long = melt_ir(ir_ionic, "r_ionic").merge(
        melt_ir(ir_cryst, "r_crystal"), on=["atomic_number","charge","CN"], how="outer"
    )
    for col in ("r_ionic","r_crystal"):
        vals = pd.to_numeric(ir_long[col], errors="coerce")
        ir_long[col] = np.where(vals>3.0, vals/100.0, vals)

    df_in = pd.read_csv(args.csv)
    base_cols = [c for c in ("cif_path","formula","name","id","score","dim","st1","st2","st3") if c in df_in.columns]

    out_rows = []
    for _, row in df_in.iterrows():
        rec: Dict[str, Any] = {c: row.get(c, None) for c in base_cols}

        M,H,nM,nH,cn = infer_MH_stoich_and_cn(row, args.m_col, args.x_col)
        if not (M and H):
            rec["T0_flags__missing_M_or_H__bool"] = 1.0
            out_rows.append(rec); continue

        rec["T0_comp__H_over_M"] = float(nH/nM) if nM>0 else np.nan
        rec["T0_comp__M_over_H"] = float(nM/nH) if nH>0 else np.nan

        tot = (nM if np.isfinite(nM) else 1.0) + (nH if np.isfinite(nH) else 1.0)
        wM = (nM/tot) if tot>0 else 0.5
        wH = 1.0 - wM

        for side, sym in (("M",M),("H",H)):
            e_row = sym2row.get(sym)
            if e_row is None: continue
            for c in elem_cols:
                v = e_row.get(c, None)
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    rec[f"T0_{side}__{c}"] = fv

        # 名义价态
        valence = None
        try:
            ratio = nH / nM if (np.isfinite(nH) and np.isfinite(nM) and nM>0) else np.nan
            if np.isfinite(ratio):
                valence = int(round(ratio))
        except Exception:
            valence = None

        # IE 累计（按价态）
        rec["T0_M__IE_sum_valence"] = np.nan
        try:
            if valence is not None and valence > 0:
                Z_M = int(sym2row[M]["atomic_number"])
                row_ie = ies.loc[ies["atomic_number"]==Z_M]
                if not row_ie.empty:
                    vals = []
                    for k in range(1, min(valence, 10)+1):
                        v = float(row_ie.iloc[0].get(f"IE{k}", np.nan))
                        if np.isfinite(v): vals.append(v)
                    rec["T0_M__IE_sum_valence"] = float(sum(vals)) if vals else np.nan
        except Exception:
            pass

        # 半径（按价态 & CN）
        rec["T0_M__r_ion_sel__ionic_A"]   = np.nan
        rec["T0_M__r_ion_sel__crystal_A"] = np.nan
        rec["T0_M__r_ion_sel__charge"]    = valence if valence is not None else np.nan
        rec["T0_M__r_ion_sel__CN"]        = cn if cn is not None else None
        try:
            if (valence is not None) and (cn is not None):
                Z_M = int(sym2row[M]["atomic_number"])
                sub = ir_long[(ir_long["atomic_number"]==Z_M) &
                              (ir_long["charge"]==valence) &
                              (ir_long["CN"]==cn)]
                if not sub.empty:
                    r_i = pd.to_numeric(sub.iloc[0]["r_ionic"], errors="coerce")
                    r_c = pd.to_numeric(sub.iloc[0]["r_crystal"], errors="coerce")
                    rec["T0_M__r_ion_sel__ionic_A"]   = float(r_i) if np.isfinite(r_i) else np.nan
                    rec["T0_M__r_ion_sel__crystal_A"] = float(r_c) if np.isfinite(r_c) else np.nan
        except Exception:
            pass

        # === NEW: 阳离子场强 Z/r^2 ===
        try:
            Zc = float(valence) if valence is not None else np.nan
            r_i = float(rec.get("T0_M__r_ion_sel__ionic_A",   np.nan))
            r_c = float(rec.get("T0_M__r_ion_sel__crystal_A", np.nan))
            rec["T0_M__field_strength__ionic_Ainv2"]   = (Zc / (r_i**2)) if (np.isfinite(Zc) and np.isfinite(r_i) and r_i>0) else np.nan
            rec["T0_M__field_strength__crystal_Ainv2"] = (Zc / (r_c**2)) if (np.isfinite(Zc) and np.isfinite(r_c) and r_c>0) else np.nan
            rec["T0_M__field_strength__primary_Ainv2"] = (
                rec["T0_M__field_strength__ionic_Ainv2"]
                if np.isfinite(rec["T0_M__field_strength__ionic_Ainv2"])
                else rec["T0_M__field_strength__crystal_Ainv2"]
            )
        except Exception:
            rec["T0_M__field_strength__ionic_Ainv2"]   = np.nan
            rec["T0_M__field_strength__crystal_Ainv2"] = np.nan
            rec["T0_M__field_strength__primary_Ainv2"] = np.nan

        # Pairwise（仅对共有的元素表物理量）
        s = pd.Series(rec)
        m_cols = [c for c in s.index if c.startswith("T0_M__")]
        h_cols = [c for c in s.index if c.startswith("T0_H__")]
        M_map = {c[len("T0_M__"):]: c for c in m_cols}
        H_map = {c[len("T0_H__"):]: c for c in h_cols}
        common = sorted(set(M_map) & set(H_map))
        for suf in common:
            vM = s.get(M_map[suf]); vH = s.get(H_map[suf])
            if pd.isna(vM) or pd.isna(vH):
                rec[f"T0_mh__{suf}__diff"]  = np.nan
                rec[f"T0_mh__{suf}__ratio"] = np.nan
                rec[f"T0_mh__{suf}__wmean"] = np.nan
            else:
                try:
                    vM = float(vM); vH = float(vH)
                except Exception:
                    rec[f"T0_mh__{suf}__diff"]  = np.nan
                    rec[f"T0_mh__{suf}__ratio"] = np.nan
                    rec[f"T0_mh__{suf}__wmean"] = np.nan
                    continue
                d = abs(vM-vH) if args.with_abs_delta else (vM-vH)
                r = (vM/vH) if abs(vH)>1e-12 else np.nan
                wm = wM*vM + wH*vH
                rec[f"T0_mh__{suf}__diff"]  = d
                rec[f"T0_mh__{suf}__ratio"] = r
                rec[f"T0_mh__{suf}__wmean"] = wm

        cif = row.get("cif_path", None)
        if isinstance(cif, str) and os.path.isfile(cif) and Structure is not None:
            try:
                struct = Structure.from_file(cif)
                if args.enable_t1: rec.update(t1_features(struct, M, H))
                if args.enable_t2: rec.update(t2_features(struct, M, H))
            except Exception:
                rec["T1_flags__failed_read_cif__bool"] = 1.0

        out_rows.append(rec)

    out = pd.DataFrame(out_rows)

    if args.verbose:
        numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        nz = sorted(((c, float(out[c].notna().mean())) for c in numeric_cols), key=lambda x: x[1])
        print("[non-null ratio] worst 25:")
        for c, r0 in nz[:25]:
            print(f"  {c:60s} : {r0:6.3f}")
        print(f"[done] rows={len(out)}  cols={out.shape[1]}")

    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
