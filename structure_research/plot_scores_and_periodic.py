#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制两张图：
1) 分数分布：一个分数一个柱子，y 轴对数，并在柱顶标注数量
2) 元素周期表热图：原子分数加权的元素平均分；支持可选配色、清晰的标签和边框

依赖：pandas、numpy、matplotlib
"""

import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------- 工具函数 ---------------------------
EL_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

def parse_formula(s: str):
    if not isinstance(s, str) or not s.strip():
        return {}
    out = {}
    for el, num in EL_RE.findall(s):
        n = float(num) if num else 1.0
        out[el] = out.get(el, 0.0) + n
    return out

def detect_columns(df):
    score_candidates = ["score", "id_score", "target"]
    formula_candidates = ["formula", "composition", "Formula", "FORMULA"]
    score_col = next((c for c in score_candidates if c in df.columns), None)
    formula_col = next((c for c in formula_candidates if c in df.columns), None)
    if not score_col:
        raise ValueError(f"未找到分数列，尝试列名：{score_candidates}")
    if not formula_col:
        raise ValueError(f"未找到化学式列，尝试列名：{formula_candidates}")
    return score_col, formula_col

def compute_elem_avg(df, score_col, formula_col, weighting="atom_fraction"):
    wsum, fsum = {}, {}
    for _, r in df.iterrows():
        try:
            s = float(r[score_col])
        except Exception:
            continue
        comp = parse_formula(str(r[formula_col]))
        if not comp: 
            continue
        if weighting == "atom_fraction":
            total = sum(comp.values())
            if total <= 0: 
                continue
            for el, cnt in comp.items():
                w = cnt / total
                wsum[el] = wsum.get(el, 0.0) + s * w
                fsum[el] = fsum.get(el, 0.0) + w
        elif weighting == "equal":
            k = len(comp)
            if k == 0: 
                continue
            w = 1.0 / k
            for el in comp:
                wsum[el] = wsum.get(el, 0.0) + s * w
                fsum[el] = fsum.get(el, 0.0) + w
        else:
            raise ValueError("weighting 仅支持 atom_fraction / equal")
    avg = {el: wsum[el]/fsum[el] for el in wsum if fsum.get(el, 0) > 0}
    return avg, fsum

def build_periodic_grid(elem_avg):
    main = [
        ['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
        ['Li','Be','','','','','','','','','','','B','C','N','O','F','Ne'],
        ['Na','Mg','','','','','','','','','','','Al','Si','P','S','Cl','Ar'],
        ['K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr'],
        ['Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe'],
        ['Cs','Ba','La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn'],
        ['Fr','Ra','Ac','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og'],
    ]
    lan, act = (
        ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'],
        ['Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'],
    )
    R, C = 9, 18
    grid = np.full((R, C), np.nan, float)
    sym  = [['' for _ in range(C)] for __ in range(R)]
    for r,row in enumerate(main):
        for c,s in enumerate(row):
            if s:
                sym[r][c] = s
                grid[r,c] = elem_avg.get(s, np.nan)
    for i,s in enumerate(lan):
        c = 2+i
        if c < C:
            sym[7][c] = s
            grid[7][c] = elem_avg.get(s, np.nan)
    for i,s in enumerate(act):
        c = 2+i
        if c < C:
            sym[8][c] = s
            grid[8][c] = elem_avg.get(s, np.nan)
    return grid, sym

# --------------------------- 绘图 ---------------------------
def plot_score_bar_log(df, score_col, out_png, out_svg=None):
    def key(x):
        try:
            v = float(x)
        except:
            return None
        return int(round(v)) if abs(v-round(v))<1e-9 else round(v,1)
    vc = df[score_col].apply(key).dropna().value_counts().sort_index()
    xs, ys = list(vc.index), vc.values

    fig, ax = plt.subplots(figsize=(12,6))
    bars = ax.bar(xs, ys, width=0.8, edgecolor="white", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Score distribution (one bar per score)")
    ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    # x 轴刻度：若范围大显示每 5 分
    if all(isinstance(x,(int,np.integer)) for x in xs) and (max(xs)-min(xs) > 25):
        ax.set_xticks(range(int(min(xs)), int(max(xs))+1, 5))
    else:
        ax.set_xticks(xs)
    for b,y in zip(bars, ys):
        if y>0:
            ax.text(b.get_x()+b.get_width()/2, y*1.05, str(y), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=280)
    if out_svg: plt.savefig(out_svg)
    plt.close()

def plot_periodic_heatmap(elem_avg, out_png, cmap_name="cividis", hide_values=False, out_svg=None):
    grid, sym = build_periodic_grid(elem_avg)
    # 配色：cividis 更均匀、色盲友好；缺失值为浅灰
    cmap = mpl.cm.get_cmap(cmap_name).copy()
    cmap.set_bad("#f2f2f2")
    vmin = float(np.nanmin(grid)) if np.isfinite(grid).any() else 0.0
    vmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(16,7.5))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, interpolation="none", cmap=cmap, norm=norm)

    # 单元格白色边框
    R,C = masked.shape
    for r in range(R):
        for c in range(C):
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False,
                                       edgecolor="white", linewidth=0.9, alpha=0.9))
    # 文本颜色自适应对比度
    def tcolor(v):
        if not isinstance(v,(float,np.floating)) or np.isnan(v): return "#666666"
        x = (v - vmin) / (vmax - vmin + 1e-12)
        return "#111111" if x>0.55 else "#ffffff"

    for r in range(R):
        for c in range(C):
            s = sym[r][c]
            if not s: 
                continue
            v = grid[r][c]
            col = tcolor(v)
            ax.text(c, r-0.06, s, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=col)
            if (not hide_values) and isinstance(v,float) and not np.isnan(v):
                ax.text(c, r+0.32, f"{v:.1f}", ha="center", va="center",
                        fontsize=8.5, color=col)

    ax.set_xticks(range(C)); ax.set_yticks(range(R))
    ax.set_xticklabels([]);  ax.set_yticklabels([])
    ax.set_xlim([-0.5, C-0.5]); ax.set_ylim([R-0.5, -0.5])
    ax.set_title(f"Element score heatmap (periodic layout)\n"
                 f"(atom-fraction weighted; cmap={cmap_name})", pad=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average score")

    plt.tight_layout()
    plt.savefig(out_png, dpi=320)
    if out_svg: plt.savefig(out_svg)
    plt.close()

# --------------------------- 主程序 ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--formula-col", default=None)
    ap.add_argument("--weighting", choices=["atom_fraction","equal"], default="atom_fraction")
    ap.add_argument("--cmap", default="cividis",
                    choices=["cividis","viridis","plasma","magma","inferno","turbo"])
    ap.add_argument("--hide-values", action="store_true", help="热图只显示元素符号，不显示数值")
    ap.add_argument("--svg", action="store_true", help="同时导出 SVG")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    s_col, f_col = detect_columns(df)
    if args.score_col:   s_col = args.score_col
    if args.formula_col: f_col = args.formula_col

    # 清理分数
    df["_s_"] = pd.to_numeric(df[s_col], errors="coerce")
    df = df.dropna(subset=["_s_"]).copy()
    df[s_col] = df["_s_"]; df.drop(columns=["_s_"], inplace=True)

    # 1) 分数分布
    hist_png = os.path.join(args.outdir, "score_bar_log.png")
    hist_svg = os.path.join(args.outdir, "score_bar_log.svg") if args.svg else None
    plot_score_bar_log(df, s_col, hist_png, hist_svg)

    # 2) 元素平均分与热图
    elem_avg, wsum = compute_elem_avg(df, s_col, f_col, weighting=args.weighting)
    heat_png = os.path.join(args.outdir, f"element_score_periodic_{args.cmap}.png")
    heat_svg = os.path.join(args.outdir, f"element_score_periodic_{args.cmap}.svg") if args.svg else None
    plot_periodic_heatmap(elem_avg, heat_png, cmap_name=args.cmap, hide_values=args.hide_values, out_svg=heat_svg)

    # 保存元素平均分表
    elem_df = (pd.DataFrame({"element": list(elem_avg.keys()),
                             "avg_score": list(elem_avg.values()),
                             "weight_sum": [wsum[e] for e in elem_avg.keys()]})
               .sort_values("avg_score", ascending=False)
               .reset_index(drop=True))
    elem_df.to_csv(os.path.join(args.outdir, "element_average_scores.csv"), index=False)

if __name__ == "__main__":
    main()
