#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_features_T0_full.py

T0 = 纯元素属性（两个元素各一整套），来源 mendeleev：
  - Elements 表：所有“物理意义明确的数值型列”（剔除资源/丰度/价格/供应风险等）
  - Ionization Energies：IE1..IEn（遍历 element.ionenergies 的全部阶次，统一列名）
  - Ionic Radii：对每个元素，将全部 (charge, coordination) 组合的
      - ionic_radius 与 crystal_radius
    展平成专用列（不做“最可信”筛选），列名含价态与配位。
  - Phase transitions：melting_point, boiling_point, fusion_heat, evaporation_heat 等
    已包含在 Elements 表的数值列里。

输出：T0_M__<col> 与 T0_H__<col>
推断 M/H：优先 --m-col/--x-col；否则从 formula（首个卤素为 H，首个非卤素为 M）。
"""

import argparse, re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

# ---------- mendeleev ----------
from mendeleev.fetch import fetch_table
from mendeleev import element as get_element

# 可选：从 formula 推断 M/H
try:
    from pymatgen.core.composition import Composition
    PMG = True
except Exception:
    PMG = False

HALOGENS = {"F", "Cl", "Br", "I", "At", "Ts"}

# 剔除“资源/丰度/价格/供应风险”类列（非你要的“物理量”）
EXCLUDE_EXACT = {
    "abundance_crust", "abundance_sea", "relative_supply_risk", "price_per_kg",
    "production_concentration", "reserve_distribution", "recycling_rate",
}
EXCLUDE_PATTERNS = [
    r"abund", r"supply", r"price", r"reserve", r"recycl", r"prod(_|)conc"
]

# --------------------------------- 工具 ---------------------------------
def normalize_symbol(s):
    if not isinstance(s, str): return None
    s = re.sub(r"[^A-Za-z]", "", s.strip())
    if not s: return None
    return s[0].upper() + s[1:].lower()

def infer_M_H(formula, m_raw, x_raw) -> Tuple[str, str]:
    M = normalize_symbol(m_raw) if m_raw else None
    H = normalize_symbol(x_raw) if x_raw else None
    if (not M or not H) and PMG and isinstance(formula, str) and formula.strip():
        try:
            comp = Composition(formula).get_el_amt_dict()
            if not H:
                for k in comp.keys():
                    if k in HALOGENS: H = k; break
            if not M:
                for k in comp.keys():
                    if k not in HALOGENS: M = k; break
        except Exception:
            pass
    return M, H

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def col_wanted(col: str) -> bool:
    if col in EXCLUDE_EXACT:
        return False
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, col, flags=re.IGNORECASE):
            return False
    return True

# -------------------------- 预扫描：统一列空间 --------------------------
def build_elements_numeric_cols(elements_df: pd.DataFrame) -> List[str]:
    # 保留数值型列 + 过滤资源/丰度/价格类
    cols = []
    for c in elements_df.columns:
        if c in ("name", "symbol", "econf", "description", "cas", "discoverers",
                 "discovery_location", "discovery_year", "name_origin", "uses",
                 "uses_description", "cpk_color"):
            continue
        if not col_wanted(c):
            continue
        if is_numeric_series(elements_df[c]):
            cols.append(c)
    return cols

def collect_all_ie_indices() -> List[int]:
    # 遍历全部元素，合并 ionenergies 的键（阶次），统一化列名 IE{n}
    indices = set()
    for Z in range(1, 119):
        try:
            e = get_element(Z)
            ie = getattr(e, "ionenergies", None) or {}
            for k in ie.keys():
                try:
                    indices.add(int(k))
                except Exception:
                    pass
        except Exception:
            continue
    return sorted(indices)

def collect_all_ionic_radii_keys():
    """
    遍历全部元素的 ionic_radii，收集所有 (charge, coordination) 组合；
    对每个组合建立两类列：__ionic 和 __crystal。
    coordination 是字符串（如 'VI','IV','VIII'），列名统一为 CN_原样。
    """
    keys = set()
    for Z in range(1, 119):
        try:
            e = get_element(Z)
            items = getattr(e, "ionic_radii", None) or []
            for ir in items:
                ch = getattr(ir, "charge", None)
                cn = getattr(ir, "coordination", None)
                # 规范化
                try:
                    ch_int = int(ch) if ch is not None else None
                except Exception:
                    continue
                cn_str = str(cn) if cn is not None else "NA"
                keys.add((ch_int, cn_str))
        except Exception:
            continue
    # 返回排序稳定的列表（先 charge，再 coordination）
    return sorted(keys, key=lambda x: (x[0] if x[0] is not None else 0, str(x[1])))

# -------------------------- 单元素取值函数 --------------------------
def grab_elements_numeric(e_row: pd.Series, cols: List[str], prefix: str) -> Dict[str, float]:
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
    """
    为该元素填充所有 (charge,coordination) 的 ionic_radius/crystal_radius。
    列名示例：
      T0_M__r_ion__+3__CN_VI__ionic
      T0_M__r_ion__+3__CN_VI__crystal
    单位：Å（mendeleev 中半径一般以 pm 存储；若为 pm 则换算）
    实测 mendeleev.ionic_radii 的属性通常已是 pm；我们统一：若值>3.0 认为是 pm -> Å。
    """
    out = {}
    items = getattr(e_obj, "ionic_radii", None) or []
    # 建索引：(charge, coordination) -> (ionic_radius, crystal_radius)
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
        if not np.isfinite(fv):
            return np.nan
        # 粗略单位判断：>3 视为 pm（常见 40~200 pm），换算为 Å
        return fv/100.0 if fv > 3.0 else fv

    for (ch, cn) in ir_keys:
        ionic, crystal = table.get((ch, cn), (np.nan, np.nan))
        lab = f"{prefix}__r_ion__{'+%d'%ch if ch is not None and ch>=0 else '%d'% (ch if ch is not None else 0)}__CN_{cn}"
        out[f"{lab}__ionic"]  = _to_ang(ionic)
        out[f"{lab}__crystal"] = _to_ang(crystal)
    return out

# --------------------------------- 主流程 ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m-col", default=None, help="金属列名（可选）")
    ap.add_argument("--x-col", default=None, help="卤素列名（可选）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # 读取 mendeleev 基表
    elements_df = fetch_table("elements")
    # 备份 symbol 用于行定位
    sym2row = {row["symbol"]: row for _, row in elements_df.iterrows() if isinstance(row.get("symbol", None), str)}

    # 统一列空间
    elem_cols = build_elements_numeric_cols(elements_df)
    ie_indices = collect_all_ie_indices()
    ir_keys = collect_all_ionic_radii_keys()

    if args.verbose:
        print(f"[info] Elements numeric cols: {len(elem_cols)}")
        print(f"[info] IE indices found     : {ie_indices[:10]} ... total {len(ie_indices)}")
        print(f"[info] Ionic radii keys     : first 10 -> {ir_keys[:10]} (total {len(ir_keys)})")

    # 输入数据
    df = pd.read_csv(args.csv)
    base_cols = [c for c in ("cif_path", "formula", "name", "id", "score", "dim", "st1", "st2", "st3") if c in df.columns]

    out_rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        row_out: Dict[str, Any] = {c: r.get(c, None) for c in base_cols}

        formula = r.get("formula", None)
        m_raw = r.get(args.m_col, None) if args.m_col else None
        x_raw = r.get(args.x_col, None) if args.x_col else None
        M, H = infer_M_H(formula, m_raw, x_raw)

        if not (M and H):
            row_out["T0_flags__missing_M_or_H__bool"] = 1.0
            out_rows.append(row_out)
            continue

        # --- M 集 ---
        try:
            eM = get_element(M)
            eM_row = sym2row.get(M, None)
        except Exception:
            eM = None; eM_row = None
        if eM_row is not None:
            row_out.update(grab_elements_numeric(eM_row, elem_cols, "T0_M"))
        if eM is not None:
            row_out.update(grab_ionization_energies(eM, ie_indices, "T0_M"))
            row_out.update(grab_all_ionic_radii(eM, ir_keys, "T0_M"))

        # --- H 集 ---
        try:
            eH = get_element(H)
            eH_row = sym2row.get(H, None)
        except Exception:
            eH = None; eH_row = None
        if eH_row is not None:
            row_out.update(grab_elements_numeric(eH_row, elem_cols, "T0_H"))
        if eH is not None:
            row_out.update(grab_ionization_energies(eH, ie_indices, "T0_H"))
            row_out.update(grab_all_ionic_radii(eH, ir_keys, "T0_H"))

        out_rows.append(row_out)

    out = pd.DataFrame(out_rows)

    # 非空率预览（便于发现整列空）
    if args.verbose:
        numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        nz = sorted(((c, float(out[c].notna().mean())) for c in numeric_cols), key=lambda x: x[1])
        print("[non-null ratio] worst 25:")
        for c, r0 in nz[:25]:
            print(f"  {c:50s} : {r0:6.3f}")
        print(f"[done] rows={len(out)}, cols={out.shape[1]}")

    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
