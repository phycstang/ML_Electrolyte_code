#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make structural CSV (formula, dim, st1, st2, st3) from (metal, halogen) + KNOWN_RATIO_STRUCTS.

Usage:
  python make_struct_csv.py \
      --out generated_structs.csv \
      --halogens F Cl Br I
  # 或手动指定金属
  python make_struct_csv.py --metals Al Mg Ti Fe Zn Ca Na K Li --out gen.csv
"""

import os
import argparse
from typing import List, Tuple, Dict

# 写 CSV 不强依赖 pandas；若可用则用 pandas；否则用 csv 内置库
try:
    import pandas as pd
except Exception:
    pd = None
import csv

# ---- 可选依赖：mendeleev / pymatgen ----
try:
    from mendeleev import element as md_element
except Exception:
    md_element = None

# ---- 已知比例–结构（与你给出的完全一致）----
KNOWN_RATIO_STRUCTS: Dict[float, List[Tuple[int, int, int, int]]] = {
    1.0: [(2, 4, 1, 6), (3, 4, 1, 12), (3, 6, 2, 12)],
    1.333333: [(3, 6, 2, 8)],
    1.5: [(3, 6, 3, 1)],
    2.0: [(1, 4, 2, 2), (2, 4, 1, 4), (2, 6, 2, 6), (2, 6, 3, 2),
          (3, 4, 1, 4), (3, 6, 1, 8), (3, 6, 2, 6)],
    2.285714: [(3, 6, 2, 5)],
    2.666667: [(2, 6, 2, 4)],
    3.0: [(0, 4, 2, 1), (1, 6, 3, 2), (2, 6, 2, 3),
          (2, 6, 2, 6), (3, 6, 1, 6), (3, 6, 2, 2)],
    3.6: [(3, 6, 2, 5)],
    4.0: [(1, 6, 2, 2), (2, 4, 2, 2), (2, 6, 2, 4), (2, 6, 2, 6), (3, 6, 2, 6)],
    5.0: [(0, 6, 2, 1)],
    6.0: [(3, 6, 2, 6)],
}

INT_RATIOS = sorted(int(k) for k in KNOWN_RATIO_STRUCTS.keys() if float(k).is_integer())

# ---- 兜底：常见价态白名单（与原脚本一致）----
COMMON_VALENCES = {
    "Li":[1], "Na":[1], "K":[1], "Rb":[1], "Cs":[1],
    "Be":[2], "Mg":[2], "Ca":[2], "Sr":[2], "Ba":[2],
    "Al":[3], "Ga":[3], "In":[3], "Tl":[1,3],
    "Sn":[2,4], "Pb":[2,4],
    "Sc":[3], "Y":[3], "La":[3],
    "Ti":[4,3], "Zr":[4], "Hf":[4],
    "V":[5,4,3], "Nb":[5], "Ta":[5],
    "Cr":[3,2,6], "Mo":[6], "W":[6],
    "Mn":[2,4,7], "Re":[7],
    "Fe":[2,3], "Co":[2,3], "Ni":[2],
    "Cu":[1,2], "Ag":[1], "Au":[1,3],
    "Zn":[2], "Cd":[2], "Hg":[2,1],
    "B":[3], "Si":[4], "Ge":[4], "As":[3,5], "Sb":[3,5], "Bi":[3,5],
}

# 第二阶段候选白名单（自动兜底用）
DEFAULT_CANDIDATE_METALS = [
    "Al","Mg","Ti","Fe","Zn","Ca","Na","K","Li","Cu","Ni","Co","Mn",
    "Zr","Sr","Ba","Cr","V","Sc","Y","La","Sn","Pb","Ga","In","Tl"
]

def auto_all_metals() -> List[str]:
    """自动收集金属：mendeleev 标注的金属 ∪ COMMON_VALENCES 键集合（去重保序）"""
    metals: List[str] = []
    if md_element is not None:
        for Z in range(1, 119):
            try:
                el = md_element(Z)
                if el is not None and bool(getattr(el, "metal", False)):
                    metals.append(el.symbol)
            except Exception:
                continue
    metals.extend(list(COMMON_VALENCES.keys()))
    # 去重保序
    seen, out = set(), []
    for s in metals:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def get_positive_oxidations(symbol: str) -> List[int]:
    """
    正价态：mendeleev(oxidation_states/oxistates) → pymatgen → COMMON_VALENCES（只保留>0的整数）
    """
    vals: List[int] = []
    # 1) mendeleev
    try:
        if md_element is not None:
            el = md_element(symbol)
            for attr in ("oxidation_states", "oxistates"):
                arr = getattr(el, attr, None)
                if arr:
                    for o in arr:
                        try:
                            oi = int(o)
                            if oi > 0:
                                vals.append(oi)
                        except Exception:
                            pass
            vals = sorted(set(vals))
            if vals:
                return vals
    except Exception:
        pass
    # 2) pymatgen
    try:
        from pymatgen.core.periodic_table import Element as PmgElement
        pmg_el = PmgElement(symbol)
        pmg_vals = []
        for oi in getattr(pmg_el, "common_oxidation_states", []) or []:
            if oi > 0:
                pmg_vals.append(int(oi))
        pmg_vals = sorted(set(pmg_vals))
        if pmg_vals:
            return pmg_vals
    except Exception:
        pass
    # 3) 兜底
    return COMMON_VALENCES.get(symbol, [])

def formula_str(metal: str, halogen: str, x: int, y: int) -> str:
    """生成最常见的简单化学式 M_x X_y（x 或 y 为1时省略“1”）"""
    if x == 1 and y == 1:
        return f"{metal}{halogen}"
    elif x == 1:
        return f"{metal}{halogen}{y}"
    elif y == 1:
        return f"{metal}{x}{halogen}"
    else:
        return f"{metal}{x}{halogen}{y}"

def build_rows(metals: List[str], halogens: List[str]) -> List[Tuple[str, int, int, int, int]]:
    """
    返回若干行：(formula, dim, st1, st2, st3)
    规则：对每个金属，取得正价态，保留与 KNOWN_RATIO_STRUCTS 的整数键交集；
         对每个保留价态 v，取 KNOWN_RATIO_STRUCTS[v] 的所有 (dim, st1, st2, st3)；
         化学式用 M_1 X_v。
    """
    out: List[Tuple[str, int, int, int, int]] = []
    kept_any = False

    for M in metals:
        v_all = get_positive_oxidations(M)
        v_keep = sorted(set(v for v in v_all if v in INT_RATIOS))
        if not v_keep:
            continue
        kept_any = True
        for X in halogens:
            for v in v_keep:
                for (dim, st1, st2, st3) in KNOWN_RATIO_STRUCTS.get(float(v), []):
                    # x_count 固定为 1，y_count = v（与原脚本一致）
                    f = formula_str(M, X, 1, v)
                    out.append((f, dim, st1, st2, st3))

    if not kept_any:
        # 若完全匹配不到，则用默认白名单再过一遍（与原脚本思路一致）
        for M in DEFAULT_CANDIDATE_METALS:
            v_all = get_positive_oxidations(M)
            v_keep = sorted(set(v for v in v_all if v in INT_RATIOS))
            for X in halogens:
                for v in v_keep:
                    for (dim, st1, st2, st3) in KNOWN_RATIO_STRUCTS.get(float(v), []):
                        f = formula_str(M, X, 1, v)
                        out.append((f, dim, st1, st2, st3))
    return out

def write_csv(rows: List[Tuple[str, int, int, int, int]], out_path: str):
    headers = ["formula", "dim", "st1", "st2", "st3"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if pd is not None:
        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in rows:
                w.writerow(r)

def main():
    ap = argparse.ArgumentParser(description="生成仅含 formula, dim, st1, st2, st3 的结构 CSV")
    ap.add_argument("--metals", nargs="*", default=None,
                    help="金属列表；默认自动收集金属（mendeleev金属 ∪ 兜底白名单）")
    ap.add_argument("--halogens", nargs="*", default=["F", "Cl", "Br", "I"],
                    help="卤素列表，默认：F Cl Br I")
    ap.add_argument("--out", type=str, default="generated_structs.csv",
                    help="输出 CSV 路径（默认：generated_structs.csv）")
    args = ap.parse_args()

    metals = args.metals if args.metals else auto_all_metals()
    rows = build_rows(metals, args.halogens)

    if not rows:
        print("[Warn] 没有匹配到任何 (金属, 卤素, 价态→结构) 组合。请尝试手动指定 --metals")
        return

    write_csv(rows, args.out)
    print(f"[Done] rows={len(rows)} → {args.out}")

if __name__ == "__main__":
    main()
