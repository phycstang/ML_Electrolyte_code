#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_features_valence_IE_radius_FS.py  (T0-only, dedup EN, ordered columns, +A/C)

更新要点：
- A: 新增两张表的聚合特征：
  * screeningconstants -> 每元素筛常数聚合：sconst_mean/min/max 及按 l(0/1/2/3) 的均值列
  * phasetransitions   -> 每元素相变统计：pt_count / 温度与压力的 min/max
- C: FORCE_KEEP_NUMERIC 扩充：atomic_weight, atomic_weight_uncertainty, atomic_volume,
     c6, c6_gb, proton_affinity, gas_basicity, group, period
- 其它：保持原有 EN/IE/离子半径与 M/H/mh 排序逻辑；不改 B（离子半径的 coordination/spin 细化）
"""

import argparse, re
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from mendeleev.fetch import fetch_table, fetch_ionization_energies, fetch_ionic_radii
from mendeleev import element as get_element

HALOGENS = {"F","Cl","Br","I","At","Ts"}
CN_MAP = {4: "IV", 6: "VI"}

# —— 排除经济/供应风险列 —— #
EXCLUDE_EXACT = {
    "abundance_crust","abundance_sea","relative_supply_risk","price_per_kg",
    "production_concentration","reserve_distribution","recycling_rate",
    "substitutability_index","production","reserves","reserve_base",
}
EXCLUDE_PATTERNS = [r"abund", r"supply", r"price", r"reserve", r"recycl", r"prod(_|)conc", r"substitut"]

# —— 强制保留的“物理/化学”数值列（若存在则保留） —— #
FORCE_KEEP_NUMERIC = {
    "electron_affinity","electrophilicity","hardness","softness",
    "dipole_polarizability","dipole_polarizability_unc",
    "covalent_radius_bragg","covalent_radius_cordero",
    "covalent_radius_pyykko","covalent_radius_pyykko_double","covalent_radius_pyykko_triple",
    "metallic_radius","metallic_radius_c12",
    "vdw_radius_bondi","vdw_radius_batsanov","vdw_radius_alvarez",
    "vdw_radius_rowland","vdw_radius_truhlar","vdw_radius_uff","vdw_radius_mm3","vdw_radius_dreiding",
    "mendeleev_number","pettifor_number","glawe_number",
    "melting_point","boiling_point","triple_point_temperature","triple_point_pressure",
    "critical_temperature","critical_pressure","fusion_heat","evaporation_heat",
    "specific_heat_capacity","molar_heat_capacity","thermal_conductivity",
    "miedema_electron_density","miedema_molar_volume",
    # ---- C: 新增强保留 ----
    "atomic_weight","atomic_weight_uncertainty","atomic_volume",
    "c6","c6_gb","proton_affinity","gas_basicity","group","period",
}

# —— 统一 API 的电负性标度 —— #
EN_SCALES_STORED = [
    "allen","pauling","miedema","mullay","gunnarsson-lundqvist",
    "robles-bartolotti","ghosh",
]
EN_SCALES_DERIVED = [
    "allred-rochow","cottrell-sutton","gordy","li-xue",
    "martynov-batsanov","mulliken","nagle","sanderson",
]
EN_ALL_SCALES = EN_SCALES_STORED + EN_SCALES_DERIVED

# —— 工具函数 —— #
def normalize_symbol(s):
    if not isinstance(s, str): return None
    s = re.sub(r"[^A-Za-z]", "", s.strip())
    return (s[0].upper() + s[1:].lower()) if s else None

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def col_wanted(col: str) -> bool:
    # 排除经济/供应风险 + 排除电负性类（避免与统一 API 重复）
    if col in EXCLUDE_EXACT: 
        return False
    if re.match(r"^(en_|electronegativity_)", col, flags=re.IGNORECASE):
        return False
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, col, flags=re.IGNORECASE):
            return False
    return True

def infer_from_formula(formula: str) -> Tuple[str,str,float,float]:
    try:
        tokens = re.findall(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", str(formula))
        if not tokens: return None, None, np.nan, np.nan
        M = H = None; nM = nH = np.nan
        for el, num in tokens:
            cnt = float(num) if num else 1.0
            if el in HALOGENS and H is None: H, nH = el, cnt
            if el not in HALOGENS and M is None: M, nM = el, cnt
        return M, H, nM, nH
    except Exception:
        return None, None, np.nan, np.nan

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
            cn = CN_MAP.get(int(row.get("st1")), None)
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
        if not col_wanted(c) and c not in FORCE_KEEP_NUMERIC:
            continue
        # 尝试数值化（防止 dtype 被误判为 object）
        s = pd.to_numeric(elements_df[c], errors="coerce")
        if pd.api.types.is_numeric_dtype(s):
            cols.append(c)
    # 强保留兜底
    for keep in FORCE_KEEP_NUMERIC:
        if keep in elements_df.columns and keep not in cols:
            s = pd.to_numeric(elements_df[keep], errors="coerce")
            if pd.api.types.is_numeric_dtype(s):
                cols.append(keep)
    return sorted(set(cols))

def melt_ir(df, kind):
    id_cols = ["atomic_number","charge"]
    cn_cols = [c for c in df.columns if c not in id_cols]
    return df.melt(id_vars=id_cols, value_vars=cn_cols, var_name="CN", value_name=kind)

def collect_all_electronegativities(sym: str, valence: int | None, cn_tag: str | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        el = get_element(sym)
    except Exception:
        return out
    for scale in EN_ALL_SCALES:
        try:
            if scale == "li-xue":
                v = el.electronegativity(scale, charge=valence) if valence else None
                if isinstance(v, dict) and cn_tag:
                    v = v.get(cn_tag, None)
            else:
                v = el.electronegativity(scale)
            if v is None: 
                continue
            fv = float(v)
            if np.isfinite(fv):
                out[f"electronegativity__{scale.replace('-','_')}"] = fv
        except Exception:
            continue
    return out
def reorder_columns(df: pd.DataFrame) -> List[str]:
    """
    列顺序：
      [base_ids]
      -> [M-only 属性: 仅存在 M__{suf}、不存在 H__{suf}]
      -> [H-only 属性: 仅存在 H__{suf}、不存在 M__{suf}]
      -> [共有属性组: M__ -> H__ -> mh__{ratio,diff,wmean}]
      -> [rest]
    """
    import re
    cols = list(df.columns)

    # 1) 基础信息
    base_ids = [c for c in ("formula","name","id","score","dim","st1","st2","st3") if c in df.columns]

    M_pref, H_pref, MH_pref = "M__", "H__", "mh__"

    def suf_M(c): return c[len(M_pref):]
    def suf_H(c): return c[len(H_pref):]
    def suf_mh(c):
        m = re.match(rf"{MH_pref}(.+)__(ratio|diff|wmean)$", c)
        return (m.group(1), m.group(2)) if m else (None, None)

    M_cols  = [c for c in cols if c.startswith(M_pref)]
    H_cols  = [c for c in cols if c.startswith(H_pref)]
    mh_cols = [c for c in cols if c.startswith(MH_pref)]

    M_map = {suf_M(c): c for c in M_cols}
    H_map = {suf_H(c): c for c in H_cols}
    mh_map = {}
    for c in mh_cols:
        s, kind = suf_mh(c)
        if s:
            mh_map.setdefault(s, {})[kind] = c

    # 2) M-only / H-only 后缀集合
    m_only_suf = sorted(set(M_map) - set(H_map))
    h_only_suf = sorted(set(H_map) - set(M_map))

    m_only = [M_map[s] for s in m_only_suf]
    h_only = [H_map[s] for s in h_only_suf]

    # 3) 共有属性组（跳过已归入 M-only/H-only 的）
    shared_suf = sorted(set(M_map) | set(H_map) | set(mh_map))
    grouped = []
    for s in shared_suf:
        if s in m_only_suf or s in h_only_suf:
            continue
        if s in M_map: grouped.append(M_map[s])
        if s in H_map: grouped.append(H_map[s])
        if s in mh_map and "ratio" in mh_map[s]: grouped.append(mh_map[s]["ratio"])
        if s in mh_map and "diff"  in mh_map[s]: grouped.append(mh_map[s]["diff"])
        if s in mh_map and "wmean" in mh_map[s]: grouped.append(mh_map[s]["wmean"])

    # 4) 拼接
    ordered = base_ids + m_only + h_only + grouped
    rest = [c for c in cols if c not in ordered]
    return ordered + rest
def write_common_suffix_names(df: pd.DataFrame, out_csv_path: str) -> None:
    """
    从 df 中找出既有 M__{suf} 又有 H__{suf} 的后缀 {suf}，
    仅输出后缀本名到 out_csv_path（列名为 'name'）。
    """
    M_pref, H_pref = "M__", "H__"
    # 收集后缀集合
    m_suf = {c[len(M_pref):] for c in df.columns if c.startswith(M_pref)}
    h_suf = {c[len(H_pref):] for c in df.columns if c.startswith(H_pref)}
    common = sorted(m_suf & h_suf)
    pd.DataFrame({"name": common}).to_csv(out_csv_path, index=False)


# ====== 新增：A 的聚合处理 ====== #
def build_sconst_features() -> Dict[int, Dict[str, float]]:
    """
    读取 screeningconstants 表，并对每个 atomic_number 产生聚合特征：
    - sconst_mean/min/max
    - 若存在 'l' 列：sconst_l0_mean, sconst_l1_mean, sconst_l2_mean, sconst_l3_mean
    返回： {Z: {feat_name: value, ...}, ...}
    """
    try:
        sc = fetch_table("screeningconstants")  # 文档：Data > Screening Constants
    except Exception:
        return {}
    # 容错：不同版本字段名
    candidates = [c for c in sc.columns if c.lower() in ("sconst","sigma","sigma_nlm")]
    if candidates:
        s_col = candidates[0]
    else:
        # 没有主值就放弃
        return {}

    # 数值化
    sc[s_col] = pd.to_numeric(sc[s_col], errors="coerce")
    sc = sc.dropna(subset=[s_col, "atomic_number"])

    out: Dict[int, Dict[str, float]] = {}
    grp = sc.groupby("atomic_number", dropna=True)
    agg = grp[s_col].agg(["mean","min","max"]).rename(columns={"mean":"sconst_mean","min":"sconst_min","max":"sconst_max"})
    # 合并 l 分壳层（如果存在）
    by_l = None
    if "l" in sc.columns:
        try:
            tmp = sc.dropna(subset=["l"]).copy()
            tmp["l"] = pd.to_numeric(tmp["l"], errors="coerce")
            by_l = tmp.groupby(["atomic_number","l"])[s_col].mean().unstack("l")
        except Exception:
            by_l = None

    feat_df = agg.copy()
    if by_l is not None:
        for L in by_l.columns:
            feat_df[f"sconst_l{int(L)}_mean"] = by_l[L]

    for Z, row in feat_df.reset_index().to_dict(orient="records"):
        z = int(Z)
        d = {k: float(v) for k, v in row.items() if k != "atomic_number" and pd.notna(v)}
        out[z] = d
    return out

def build_pt_features() -> Dict[int, Dict[str, float]]:
    """
    读取 phasetransitions 表，并对每个 atomic_number 统计：
    - pt_count
    - pt_min_temp, pt_max_temp
    - pt_min_pressure, pt_max_pressure
    """
    try:
        pt = fetch_table("phasetransitions")  # 文档：Bulk data access 列表
    except Exception:
        return {}
    cols = {c.lower(): c for c in pt.columns}
    # 容错：温度/压力列名（常见 temperature/pressure）
    tcol = cols.get("temperature") or cols.get("temp") or None
    pcol = cols.get("pressure") or cols.get("press") or None

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    if tcol: pt[tcol] = to_num(pt[tcol])
    if pcol: pt[pcol] = to_num(pt[pcol])

    out: Dict[int, Dict[str, float]] = {}
    grp = pt.groupby("atomic_number", dropna=True)
    for Z, g in grp:
        d: Dict[str, float] = {}
        d["pt_count"] = float(len(g))
        if tcol:
            tt = g[tcol].dropna()
            if len(tt):
                d["pt_min_temp"] = float(tt.min())
                d["pt_max_temp"] = float(tt.max())
        if pcol:
            pp = g[pcol].dropna()
            if len(pp):
                d["pt_min_pressure"] = float(pp.min())
                d["pt_max_pressure"] = float(pp.max())
        out[int(Z)] = d
    return out
# ====== 新增：A 的聚合处理（END） ====== #

# —— 主程序 —— #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m-col", default=None)
    ap.add_argument("--x-col", default=None)
    ap.add_argument("--with_abs_delta", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    elements_df = fetch_table("elements")  # elements 字段详见官方 Data 页面
    elem_cols = build_elements_numeric_cols(elements_df)
    sym2row = {row["symbol"]: row for _, row in elements_df.iterrows() if isinstance(row.get("symbol",None), str)}

    # A: 预构建 sconst / phase transitions 的映射（Z -> 特征字典）
    z2sconst = build_sconst_features()       # 需要 Data: Screening Constants
    z2pt     = build_pt_features()           # 需要 Data: Phase Transitions

    ies = fetch_ionization_energies(degree=list(range(1, 11))).reset_index()
    ir_ionic  = fetch_ionic_radii(radius="ionic_radius").reset_index()
    ir_cryst  = fetch_ionic_radii(radius="crystal_radius").reset_index()
    ir_long = melt_ir(ir_ionic, "r_ionic").merge(
        melt_ir(ir_cryst, "r_crystal"), on=["atomic_number","charge","CN"], how="outer"
    )
    for col in ("r_ionic","r_crystal"):
        vals = pd.to_numeric(ir_long[col], errors="coerce")
        ir_long[col] = np.where(vals>3.0, vals/100.0, vals)  # pm→Å

    df_in = pd.read_csv(args.csv)
    base_cols = [c for c in ("formula","name","id","score","dim","st1","st2","st3") if c in df_in.columns]

    out_rows = []
    for _, row in df_in.iterrows():
        rec: Dict[str, Any] = {c: row.get(c, None) for c in base_cols}

        M,H,nM,nH,cn = infer_MH_stoich_and_cn(row, args.m_col, args.x_col)
        if not (M and H):
            rec["flags__missing_M_or_H__bool"] = 1.0
            out_rows.append(rec); 
            continue

        # 组成配比
        rec["comp__H_over_M"] = float(nH/nM) if nM>0 else np.nan
        rec["comp__M_over_H"] = float(nM/nH) if nH>0 else np.nan
        tot = (nM if np.isfinite(nM) else 1.0) + (nH if np.isfinite(nH) else 1.0)
        wM = (nM/tot) if tot>0 else 0.5
        wH = 1.0 - wM

        # 元素物理字段（M/H 两套；已去除电负性类）
        for side, sym in (("M",M),("H",H)):
            e_row = sym2row.get(sym); 
            if e_row is None: 
                continue
            for c in elem_cols:
                v = e_row.get(c, None)
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    rec[f"{side}__{c}"] = fv

            # A: 并入 sconst/phase transitions 聚合（按原子序 Z）
            try:
                Z = int(e_row.get("atomic_number"))
            except Exception:
                Z = None
            if Z is not None:
                if Z in z2sconst:
                    for k, v in z2sconst[Z].items():
                        rec[f"{side}__{k}"] = v
                if Z in z2pt:
                    for k, v in z2pt[Z].items():
                        rec[f"{side}__{k}"] = v

        # 名义价态（由配比近似）
        valence = None
        try:
            ratio = nH / nM if (np.isfinite(nH) and np.isfinite(nM) and nM>0) else np.nan
            if np.isfinite(ratio): valence = int(round(ratio))
        except Exception:
            valence = None
        cn_tag = cn  # 'IV'/'VI' 或 None

        # —— 统一 API 抓电负性（避免重复） —— #
        en_M = collect_all_electronegativities(M, valence, cn_tag)
        en_H = collect_all_electronegativities(H, valence, cn_tag)
        for k, v in en_M.items(): rec[f"M__{k}"] = v
        for k, v in en_H.items(): rec[f"H__{k}"] = v

        # IE 累计（仅 M）
        rec["M__IE_sum_valence"] = np.nan
        try:
            if valence and valence > 0 and M in sym2row:
                Z_M = int(sym2row[M]["atomic_number"])
                row_ie = ies.loc[ies["atomic_number"]==Z_M]
                if not row_ie.empty:
                    vals = []
                    for k in range(1, min(valence, 10)+1):
                        v = float(row_ie.iloc[0].get(f"IE{k}", np.nan))
                        if np.isfinite(v): vals.append(v)
                    rec["M__IE_sum_valence"] = float(sum(vals)) if vals else np.nan
        except Exception:
            pass

        # 半径选择（仅 M；按价态 & CN）—— 按你要求，不做 B 的细化改动
        rec["M__r_ion_sel__ionic_A"]   = np.nan
        rec["M__r_ion_sel__crystal_A"] = np.nan
        rec["M__r_ion_sel__charge"]    = valence if valence is not None else np.nan
        rec["M__r_ion_sel__CN"]        = cn if cn is not None else None
        try:
            if (valence is not None) and (cn is not None) and M in sym2row:
                Z_M = int(sym2row[M]["atomic_number"])
                sub = ir_long[(ir_long["atomic_number"]==Z_M) &
                              (ir_long["charge"]==valence) &
                              (ir_long["CN"]==cn)]
                if not sub.empty:
                    r_i = pd.to_numeric(sub.iloc[0]["r_ionic"], errors="coerce")
                    r_c = pd.to_numeric(sub.iloc[0]["r_crystal"], errors="coerce")
                    rec["M__r_ion_sel__ionic_A"]   = float(r_i) if np.isfinite(r_i) else np.nan
                    rec["M__r_ion_sel__crystal_A"] = float(r_c) if np.isfinite(r_c) else np.nan
        except Exception:
            pass

        # CFS = Z / r^2（三列）
        try:
            Zc = float(valence) if valence is not None else np.nan
            r_i = float(rec.get("M__r_ion_sel__ionic_A",   np.nan))
            r_c = float(rec.get("M__r_ion_sel__crystal_A", np.nan))
            rec["M__field_strength__ionic_Ainv2"]   = (Zc / (r_i**2)) if (np.isfinite(Zc) and np.isfinite(r_i) and r_i>0) else np.nan
            rec["M__field_strength__crystal_Ainv2"] = (Zc / (r_c**2)) if (np.isfinite(Zc) and np.isfinite(r_c) and r_c>0) else np.nan
            rec["M__field_strength__primary_Ainv2"] = (
                rec["M__field_strength__ionic_Ainv2"]
                if np.isfinite(rec["M__field_strength__ionic_Ainv2"])
                else rec["M__field_strength__crystal_Ainv2"]
            )
        except Exception:
            rec["M__field_strength__ionic_Ainv2"]   = np.nan
            rec["M__field_strength__crystal_Ainv2"] = np.nan
            rec["M__field_strength__primary_Ainv2"] = np.nan

        # Pairwise（仅对 M/H 共有键）
        s = pd.Series(rec)
        m_cols = [c for c in s.index if c.startswith("M__")]
        h_cols = [c for c in s.index if c.startswith("H__")]
        M_map = {c[len("M__"):]: c for c in m_cols}
        H_map = {c[len("H__"):]: c for c in h_cols}
        common = sorted(set(M_map) & set(H_map))
        for suf in common:
            vM = s.get(M_map[suf]); vH = s.get(H_map[suf])
            if pd.isna(vM) or pd.isna(vH):
                rec[f"mh__{suf}__diff"]  = np.nan
                rec[f"mh__{suf}__ratio"] = np.nan
                rec[f"mh__{suf}__wmean"] = np.nan
            else:
                try:
                    vM = float(vM); vH = float(vH)
                except Exception:
                    rec[f"mh__{suf}__diff"]  = np.nan
                    rec[f"mh__{suf}__ratio"] = np.nan
                    rec[f"mh__{suf}__wmean"] = np.nan
                    continue
                d  = abs(vM-vH) if args.with_abs_delta else (vM-vH)
                r  = (vM/vH) if abs(vH)>1e-12 else np.nan
                wm = wM*vM + wH*vH
                rec[f"mh__{suf}__diff"]  = d
                rec[f"mh__{suf}__ratio"] = r
                rec[f"mh__{suf}__wmean"] = wm

        out_rows.append(rec)

    out = pd.DataFrame(out_rows)

    # —— 关键：按 M_, H_, ratio_, diff_, wmean_ 的组内顺序重排 —— #
    new_order = reorder_columns(out)
    out = out.reindex(columns=new_order)

    if args.verbose:
        numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        nz = sorted(((c, float(out[c].notna().mean())) for c in numeric_cols), key=lambda x: x[1])
        print("[non-null ratio] worst 25:")
        for c, r0 in nz[:25]:
            print(f"  {c:60s} : {r0:6.3f}")
        print(f"[done] rows={len(out)}  cols={out.shape[1]}")

    # 导出共有后缀名清单 name.csv（与 --out 同目录）
    from pathlib import Path
    out_path = Path(args.out)
    name_csv = out_path.with_name("name.csv")
    write_common_suffix_names(out, str(name_csv))


    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
