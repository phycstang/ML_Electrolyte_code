#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import tempfile
import subprocess
from typing import List, Tuple, Dict

import pandas as pd

# ---- 可选依赖：mendeleev / pymatgen ----
try:
    from mendeleev import element as md_element
except Exception:
    md_element = None

# ---- 已知比例–结构（来自你的 feats.csv 提取，完整保留）----
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

# ---- 兜底：常见价态白名单 ----
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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def auto_all_metals() -> List[str]:
    """自动收集金属：mendeleev 标注的金属 ∪ COMMON_VALENCES 键集合"""
    metals = []
    if md_element is not None:
        for Z in range(1, 119):
            try:
                el = md_element(Z)
                if el is not None and bool(getattr(el, "metal", False)):
                    metals.append(el.symbol)
            except Exception:
                continue
    # 并上白名单里的元素，保证 Na/K/Ca 等一定包含
    metals.extend(list(COMMON_VALENCES.keys()))
    # 去重保序
    seen, out = set(), []
    for s in metals:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def get_positive_oxidations(symbol: str) -> List[int]:
    """正价态：mendeleev(oxidation_states/oxistates) → pymatgen → COMMON_VALENCES"""
    vals = []
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

def run_halide_minifeats(formulas: List[str], halide_minifeats_py: str, out_csv: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        in_path = f.name
        pd.DataFrame({"formula": formulas}).to_csv(in_path, index=False)
    cmd = ["python", halide_minifeats_py, "--csv", in_path, "--out_csv", out_csv]
    subprocess.run(cmd, check=True)
    df = pd.read_csv(out_csv)
    try:
        os.remove(in_path)
    except Exception:
        pass
    return df

def formula_str(metal: str, halogen: str, x: int, y: int) -> str:
    if x == 1 and y == 1:
        return f"{metal}{halogen}"
    elif x == 1:
        return f"{metal}{halogen}{y}"
    elif y == 1:
        return f"{metal}{x}{halogen}"
    else:
        return f"{metal}{x}{halogen}{y}"

def filter_useful_metals(candidates: List[str]) -> tuple[list[tuple[str, list[int]]], pd.DataFrame]:
    """返回：[(metal, kept_valences), ...], audit_df"""
    useful, audit = [], []
    for M in candidates:
        v_all = get_positive_oxidations(M)
        v_keep = sorted(set(v for v in v_all if v in INT_RATIOS))
        audit.append({
            "metal": M,
            "all_valences": "|".join(map(str, v_all)) if v_all else "",
            "kept_valences": "|".join(map(str, v_keep)) if v_keep else ""
        })
        if v_keep:
            useful.append((M, v_keep))
    return useful, pd.DataFrame(audit)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", nargs="*", default=None,
                    help="金属列表；默认自动筛选所有‘有用金属’（含兜底白名单）")
    ap.add_argument("--halogens", nargs="*", default=["F","Cl","Br","I"])
    ap.add_argument("--halide-minifeats", type=str, default="halide_minifeats.py")
    ap.add_argument("--outdir", type=str, default="generated_runs/dataset_fixed_auto")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 阶段一：从“自动金属全集”筛选
    all_metals = args.metals if args.metals else auto_all_metals()
    useful, audit_df = filter_useful_metals(all_metals)

    # 如果一个都没有，阶段二：启用默认候选白名单再筛一次
    fell_back = False
    if not useful and args.metals is None:
        fell_back = True
        fallback_list = DEFAULT_CANDIDATE_METALS
        useful, audit_df_fb = filter_useful_metals(fallback_list)
        # 合并审计（去重）
        audit_df = pd.concat([audit_df, audit_df_fb]).drop_duplicates(subset=["metal"]).reset_index(drop=True)

    # 写审计表
    audit_path = os.path.join(args.outdir, "matched_ratios.csv")
    audit_df.to_csv(audit_path, index=False, encoding="utf-8")

    if not useful:
        print("[Warn] 自动筛选未找到可用金属（价态∩{1..6}为空）。")
        print("→ 可手动指定：--metals Al Mg Ti Fe Zn Ca Na K Li")
        return

    # 保存使用到的金属清单
    with open(os.path.join(args.outdir, "metals_used.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([m for m,_ in useful]))

    if fell_back:
        print(f"[Info] 已启用候选白名单兜底，金属数={len(useful)}。详见 {audit_path}")

    # 生成组合：只用整数 v 对应的 KNOWN_RATIO_STRUCTS[v]
    combos = []
    for M, v_keep in useful:
        for X in args.halogens:
            for v in v_keep:
                for (dim, st1, st2, st3) in KNOWN_RATIO_STRUCTS.get(float(v), []):
                    combos.append({
                        "metal": M, "halogen": X, "ox": v,
                        "x_count": 1, "y_count": v, "ratio_key": float(v),
                        "dim": dim, "st1": st1, "st2": st2, "st3": st3,
                    })

    if not combos:
        print("[Warn] 没有匹配到任何 (金属, 价态, 比例→结构) 组合。")
        print("→ 请检查 matched_ratios.csv 中 kept_valences 是否为空；或改用 --metals 手动指定。")
        return

    # 生成特征并合并
    formulas = [formula_str(c["metal"], c["halogen"], c["x_count"], c["y_count"]) for c in combos]
    feats_path = os.path.join(args.outdir, "raw_minifeats.csv")
    df_feats = run_halide_minifeats(formulas, args.halide_minifeats, feats_path)

    df_meta = pd.DataFrame(combos).reset_index(drop=True)
    df_meta["formula"] = formulas
    if "formula" not in df_feats.columns:
        df_feats = df_feats.copy(); df_feats["formula"] = formulas
    df_out = pd.merge(df_meta, df_feats, on="formula", how="inner")

    df_out.insert(0, "cif_file", [
        f"{m}_{x}{h}{y}" for m,h,x,y in
        zip(df_out["metal"], df_out["halogen"], df_out["x_count"], df_out["y_count"])
    ])

    out_csv = os.path.join(args.outdir, "generated_dataset.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[Done] Metals={len(useful)}, Candidates={len(df_out)}. Saved to {out_csv}")

if __name__ == "__main__":
    main()
