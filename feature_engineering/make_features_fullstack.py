#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_features_fullstack.py

功能总览
- T0: 纯元素属性（两个元素各一整套，来自 mendeleev）
    * Elements 表：全部“物理意义明确”的数值列（去掉资源/丰度/价格/供应风险等）
    * Ionization Energies：IE1..IEn（逐元素全阶展开）
    * Ionic Radii：所有 (charge, coordination) 组合的 ionic_radius & crystal_radius（统一换算为 Å）
    * Phase transitions：melting_point, boiling_point, fusion_heat, evaporation_heat（已在 Elements 表中）
- Pairwise: 对每个 T0 数值列进行 M/H 两元素的 pairwise 组合：diff、ratio、wmean（按化学计量）
    * 并输出元素比值：H_over_M、M_over_H
- T1/T2: 轻量稳健版的结构特征（需 cif_path；开关 --enable-t1/--enable-t2）
- 其余：从输入 CSV 透传基础列（cif_path, formula, name, id, score, dim, st1, st2, st3 如存在）

用法
  python make_features_fullstack.py \
    --csv data_clean_dedup.csv \
    --out features_fullstack.csv \
    --m-col M --x-col H \
    --enable-t1 1 --enable-t2 1 \
    --with_abs_delta 1 -v
"""

import argparse, re, os, math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

# ---- mendeleev ----
from mendeleev.fetch import fetch_table
from mendeleev import element as get_element

# 可选：用于从 formula 推断 M/H 与化学计量
try:
    from pymatgen.core.composition import Composition
    from pymatgen.core import Structure
    PMG = True
except Exception:
    PMG = False
    Structure = None

HALOGENS = {"F","Cl","Br","I","At","Ts"}

# 过滤“非物理”类列（资源/丰度/价格/供应风险等）
EXCLUDE_EXACT = {
    "abundance_crust","abundance_sea","relative_supply_risk","price_per_kg",
    "production_concentration","reserve_distribution","recycling_rate",
}
EXCLUDE_PATTERNS = [r"abund", r"supply", r"price", r"reserve", r"recycl", r"prod(_|)conc"]

# ---------------- 工具 ----------------
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
    # 返回 M, H, nM, nH；简单规则：第一个卤素=H，第一个非卤素=M
    M = H = None
    nM = nH = np.nan
    if not (PMG and formula and isinstance(formula, str)):
        return M, H, nM, nH
    try:
        comp = Composition(formula)
        d = comp.get_el_amt_dict()
        for el, amt in d.items():
            if el in HALOGENS and H is None:
                H, nH = el, float(amt)
            if el not in HALOGENS and M is None:
                M, nM = el, float(amt)
        return M, H, nM, nH
    except Exception:
        return None, None, np.nan, np.nan

def infer_MH_and_stoich(r: pd.Series, m_col: str|None, x_col: str|None) -> Tuple[str,str,float,float]:
    M = normalize_symbol(r.get(m_col)) if m_col else None
    H = normalize_symbol(r.get(x_col)) if x_col else None
    nM = nH = np.nan
    if (not M or not H):
        M2,H2,nM,nH = infer_from_formula(r.get("formula",""))
        M = M or M2; H = H or H2
    if not np.isfinite(nM): nM = 1.0
    if not np.isfinite(nH): nH = 1.0
    return M,H,float(nM),float(nH)

# ------------- mendeleev：统一列空间 -------------
def build_elements_numeric_cols(elements_df: pd.DataFrame) -> List[str]:
    cols = []
    for c in elements_df.columns:
        if c in ("name","symbol","econf","description","cas","discoverers",
                 "discovery_location","discovery_year","name_origin","uses",
                 "uses_description","cpk_color"):
            continue
        if not col_wanted(c):
            continue
        if is_numeric_series(elements_df[c]):
            cols.append(c)
    return cols

def collect_all_ie_indices() -> List[int]:
    indices = set()
    for Z in range(1,119):
        try:
            e = get_element(Z)
            ie = getattr(e, "ionenergies", None) or {}
            for k in ie.keys():
                indices.add(int(k))
        except Exception:
            continue
    return sorted(indices)

def collect_all_ionic_radii_keys():
    keys = set()
    for Z in range(1,119):
        try:
            e = get_element(Z)
            items = getattr(e, "ionic_radii", None) or []
            for ir in items:
                ch = getattr(ir, "charge", None)
                cn = getattr(ir, "coordination", None)
                try:
                    ch_int = int(ch) if ch is not None else None
                except Exception:
                    continue
                cn_str = str(cn) if cn is not None else "NA"
                keys.add((ch_int, cn_str))
        except Exception:
            continue
    return sorted(keys, key=lambda x: (x[0] if x[0] is not None else -999, str(x[1])))

# ------------- 单元素抓取 -------------
def grab_elements_numeric(e_row: pd.Series, cols: List[str], prefix: str) -> Dict[str,float]:
    out = {}
    for c in cols:
        v = e_row.get(c, None)
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out[f"{prefix}__{c}"] = fv
    return out

def grab_ionization_energies(e_obj, ie_indices: List[int], prefix: str) -> Dict[str, float]:
    out = {}
    ie = getattr(e_obj, "ionenergies", None) or {}
    for n in ie_indices:
        v = ie.get(n, None)
        try:
            fv = float(v)
        except Exception:
            fv = np.nan
        out[f"{prefix}__IE{n}"] = fv
    return out

def grab_all_ionic_radii(e_obj, ir_keys: List[Tuple[int, str]], prefix: str) -> Dict[str, float]:
    out = {}
    items = getattr(e_obj, "ionic_radii", None) or []
    table = {}
    for ir in items:
        ch = getattr(ir, "charge", None)
        cn = getattr(ir, "coordination", None)
        ionic = getattr(ir, "ionic_radius", None)
        crystal = getattr(ir, "crystal_radius", None)
        try:
            ch_int = int(ch) if ch is not None else None
        except Exception:
            continue
        cn_str = str(cn) if cn is not None else "NA"
        table[(ch_int, cn_str)] = (ionic, crystal)

    def _to_ang(v):
        try:
            fv = float(v)
        except Exception:
            return np.nan
        if not np.isfinite(fv): return np.nan
        # 经验：>3 视作 pm -> Å（常见 40–200 pm）；否则认为是 Å
        return fv/100.0 if fv>3.0 else fv

    for (ch, cn) in ir_keys:
        ionic, crystal = table.get((ch, cn), (np.nan, np.nan))
        ch_lab = f"{'+%d'%ch if (ch is not None and ch>=0) else ('%d'%ch if ch is not None else 'NA')}"
        lab = f"{prefix}__r_ion__{ch_lab}__CN_{cn}"
        out[f"{lab}__ionic"]   = _to_ang(ionic)
        out[f"{lab}__crystal"] = _to_ang(crystal)
    return out

# ------------- Pairwise -------------
def pairwise_ops(df_row: pd.Series, key_prefix_M: str, key_prefix_H: str,
                 wM: float, wH: float, abs_delta: bool) -> Dict[str,float]:
    """
    对同名 T0 数值列（去掉前缀后名相同）做 diff/ratio/wmean。
    """
    out = {}
    # 找出 M/H 的可配对列名集合
    m_cols = [c for c in df_row.index if c.startswith(key_prefix_M)]
    h_cols = [c for c in df_row.index if c.startswith(key_prefix_H)]
    M_set = {c[len(key_prefix_M):]: c for c in m_cols}   # 后缀 -> 全名
    H_set = {c[len(key_prefix_H):]: c for c in h_cols}
    common_suffix = sorted(set(M_set.keys()) & set(H_set.keys()))
    for suf in common_suffix:
        cM = M_set[suf]; cH = H_set[suf]
        vM = df_row.get(cM); vH = df_row.get(cH)
        if pd.isna(vM) or pd.isna(vH):
            out[f"T0_mh__{suf}__diff"]  = np.nan
            out[f"T0_mh__{suf}__ratio"] = np.nan
            out[f"T0_mh__{suf}__wmean"] = np.nan
            continue
        try:
            vM = float(vM); vH = float(vH)
        except Exception:
            out[f"T0_mh__{suf}__diff"]  = np.nan
            out[f"T0_mh__{suf}__ratio"] = np.nan
            out[f"T0_mh__{suf}__wmean"] = np.nan
            continue
        d = abs(vM - vH) if abs_delta else (vM - vH)
        r = (vM / vH) if abs(vH) > 1e-12 else np.nan
        wm = wM*vM + wH*vH
        out[f"T0_mh__{suf}__diff"]  = d
        out[f"T0_mh__{suf}__ratio"] = r
        out[f"T0_mh__{suf}__wmean"] = wm
    return out

# ------------- T1/T2（轻量稳健版）-------------
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

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m-col", default=None, help="金属列名（可选）")
    ap.add_argument("--x-col", default=None, help="卤素列名（可选）")
    ap.add_argument("--enable-t1", type=int, default=1)
    ap.add_argument("--enable-t2", type=int, default=1)
    ap.add_argument("--with_abs_delta", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    elements_df = fetch_table("elements")
    sym2row = {row["symbol"]: row for _, row in elements_df.iterrows() if isinstance(row.get("symbol",None), str)}

    elem_cols = build_elements_numeric_cols(elements_df)
    ie_indices = collect_all_ie_indices()
    ir_keys = collect_all_ionic_radii_keys()

    if args.verbose:
        print(f"[info] numeric element cols: {len(elem_cols)}")
        print(f"[info] IE count            : {len(ie_indices)}  (max n={max(ie_indices) if ie_indices else 0})")
        print(f"[info] ionic radii keys    : {len(ir_keys)}")

    df = pd.read_csv(args.csv)
    base_cols = [c for c in ("cif_path","formula","name","id","score","dim","st1","st2","st3") if c in df.columns]

    out_rows = []
    for _, r in df.iterrows():
        row_out: Dict[str, Any] = {c: r.get(c, None) for c in base_cols}

        M,H,nM,nH = infer_MH_and_stoich(r, args.m_col, args.x_col)
        if not (M and H):
            row_out["T0_flags__missing_M_or_H__bool"] = 1.0
            out_rows.append(row_out); continue

        # 元素比值（化学计量）
        row_out["T0_comp__H_over_M"] = float(nH/nM) if nM>0 else np.nan
        row_out["T0_comp__M_over_H"] = float(nM/nH) if nH>0 else np.nan

        # 权重
        tot = (nM if np.isfinite(nM) else 1.0) + (nH if np.isfinite(nH) else 1.0)
        wM = (nM/tot) if tot>0 else 0.5
        wH = 1.0 - wM

        # --- 抓 M/H 的 T0 原始列 ---
        try: eM = get_element(M)
        except Exception: eM = None
        try: eH = get_element(H)
        except Exception: eH = None
        eM_row = sym2row.get(M); eH_row = sym2row.get(H)

        if eM_row is not None:
            row_out.update(grab_elements_numeric(eM_row, elem_cols, "T0_M"))
        if eM is not None:
            row_out.update(grab_ionization_energies(eM, ie_indices, "T0_M"))
            row_out.update(grab_all_ionic_radii(eM, ir_keys, "T0_M"))

        if eH_row is not None:
            row_out.update(grab_elements_numeric(eH_row, elem_cols, "T0_H"))
        if eH is not None:
            row_out.update(grab_ionization_energies(eH, ie_indices, "T0_H"))
            row_out.update(grab_all_ionic_radii(eH, ir_keys, "T0_H"))

        # --- Pairwise：对所有可配对的 T0 数值列做 diff/ratio/wmean ---
        row_out.update(
            pairwise_ops(
                pd.Series(row_out),
                key_prefix_M="T0_M__",
                key_prefix_H="T0_H__",
                wM=wM, wH=wH, abs_delta=bool(args.with_abs_delta)
            )
        )

        # --- T1/T2（可选；需 cif_path） ---
        cif = r.get("cif_path", None)
        if isinstance(cif, str) and os.path.isfile(cif) and Structure is not None:
            try:
                struct = Structure.from_file(cif)
                if args.enable_t1: row_out.update(t1_features(struct, M, H))
                if args.enable_t2: row_out.update(t2_features(struct, M, H))
            except Exception:
                row_out["T1_flags__failed_read_cif__bool"] = 1.0

        out_rows.append(row_out)

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
