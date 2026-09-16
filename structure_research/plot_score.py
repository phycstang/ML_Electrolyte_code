#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("extra_features.csv")

# 统计每个分数的数量
counts = df["score"].value_counts().sort_index()

# 设置绘图风格
sns.set_theme(style="whitegrid", font_scale=1.3)

plt.figure(figsize=(10, 6))

# 配色
color = sns.color_palette("Blues")[4]

# 画柱状图
bars = plt.bar(counts.index, counts.values,
               color=color, edgecolor="black", alpha=0.8)

# y 轴对数
plt.yscale("log")

# 添加数量标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval,
             f"{int(yval)}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color="black")

# 美化坐标轴
plt.title("Score Distribution (log scale)", fontsize=18, weight="bold")
plt.xlabel("Score", fontsize=15)
plt.ylabel("Count (log scale)", fontsize=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# 加细网格线
plt.grid(axis="y", which="both", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("score_distribution_log_beautiful.png", dpi=400)