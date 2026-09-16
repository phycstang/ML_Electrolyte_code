#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 用 pymatgen 的 periodic_table_heatmap 画底图，
# 再叠加：卤素红框 + 出现过的金属加圆点。

import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatgen.util.plotting import periodic_table_heatmap
from pymatgen.core import Element

CSV = "structure_table.csv"
OUT = "periodic_table_heatmap.png"

# 1) 读取 CSV（列名容错）
df = pd.read_csv(CSV)
def pick(df, cands):
    for c in df.columns:
        if c.lower() in [x.lower() for x in cands]:
            return c
    return None

col_formula = pick(df, ["formula","composition","reduced_formula","pretty_formula","name","cif_file"])
col_score   = pick(df, ["score","label","target","y"])
if not col_score:
    raise SystemExit("需要分数列：score/label/target/y 其一")
if not col_formula:
    raise SystemExit("需要化学式或文件名列：formula/name/cif_file 等其一")

# 2) 解析每行化学式，按“含该元素的样本平均分”聚合
elem_vals = {}
pat = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
for _, r in df.iterrows():
    raw = str(r[col_formula])
    token = os.path.splitext(os.path.basename(raw))[0]
    token = "".join(ch for ch in token if ch.isalnum())  # 去掉 mp-xxx 等
    parts = pat.findall(token)
    if not parts:
        continue
    for sym, _ in parts:
        try:
            _ = Element(sym)  # 保证是合法元素
        except Exception:
            continue
        elem_vals.setdefault(sym, []).append(float(r[col_score]))

elem_avg = {e: float(np.mean(v)) for e, v in elem_vals.items()}

# 3) 画底图（pymatgen 内置）
fig = periodic_table_heatmap(
    elemental_data=elem_avg,
    cbar_label="Average score per element",
    show_plot=False,           # 我们自己控制保存
    cmap="viridis",            # 你也可以换别的
    pymatviz=False             # 强制用 matplotlib 输出
)  # 返回的是 matplotlib Figure

ax = fig.axes[0]

# 4) 叠加卤素红框 + 出现过的金属打点
HALOGENS = {"F","Cl","Br","I"}

# 为了拿到每个元素在图中的格子位置，遍历全部元素，用其 period/group 算坐标
# pymatgen 的 Element 提供 period(=row), group(=column) 等属性
# 约定与内置底图相同：x = group-1, y = (max_row - period)
max_row = 9
for Z in range(1, 119):
    try:
        el = Element.from_Z(Z)
    except Exception:
        continue
    p, g = el.row, el.group
    if g is None or p is None:  # 未分组的超重元素跳过
        continue
    x, y = g-1, max_row - p

    # 卤素红框
    if el.symbol in HALOGENS:
        rect = plt.Rectangle((x, y), 1, 1, fill=False, edgecolor="red", linewidth=2.0)
        ax.add_patch(rect)

    # 数据里出现过的“金属”打点
    # Element 有 is_metal / is_halogen / is_noble_gas 等属性
    if el.symbol in elem_avg and el.is_metal:
        ax.plot(x+0.5, y+0.5, "o", markersize=5)

fig.suptitle("Periodic Table — colored by element-wise mean score\nRed border: F/Cl/Br/I; Dot: metals present", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=220)
print(f"saved -> {OUT}")
