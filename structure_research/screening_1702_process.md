# 1702 个 CIF 的来源与元素筛选统计

本文档记录 `/data/home/tmy/ML_Electrolyte3/new_material/material_P/cif` 中 1702 个 CIF 的来源条件，以及后续按“放射性 / 有毒 / 贵金属”三类元素进行筛除的统计过程。

## 1. 1702 个 CIF 的来源

这些 CIF 由 `mpd.py` 从 Materials Project 下载得到，核心条件如下：

- 目标卤素元素为 `F, Cl, Br, I`。
- 查询二元卤化物体系，即材料中元素数为 2。
- 材料必须包含目标卤素元素。
- 另一个元素不能也是卤素，除非显式允许卤素-卤素二元体系。
- 不要求另一个元素必须是金属，因此数据中也包含部分非金属卤化物。

下载后的结构通过 `pymatgen` 写成 CIF 文件，文件名形式为：

```text
Formula_mp-id.cif
```

当前数量可用下面命令确认：

```bash
find cif -name '*.cif' | wc -l
```

结果：

```text
1702
```

## 2. 元素筛选的统计对象

由于每个结构本身都是卤化物，所有样本都会含有 `F/Cl/Br/I` 中的一个卤素。因此这里的“放射性 / 有毒 / 贵金属”筛选只统计**非卤素元素**，不把卤素元素计入有毒元素筛除。

例如：

```text
CdI2_mp-570019.cif -> 非卤素元素为 Cd
UCl4_mp-23235.cif  -> 非卤素元素为 U
AgBr_mp-23231.cif  -> 非卤素元素为 Ag
```

统计时优先从 `cif_file` 文件名解析化学式，而不是直接使用 `summary.csv` 中的 `formula` 列。原因是 `summary.csv` 里有少量公式可能被表格软件或 CSV 解析误写成日期格式，例如 `FeBr2`、`FeBr3` 曾显示为类似 `2-Feb`、`3-Feb`。

## 3. 当前采用的三类元素集合

### 3.1 放射性元素

```text
Ac, Np, Pa, Pm, Pu, Tc, Th, U
```

### 3.2 有毒元素

当前采用较严格的重毒元素口径：

```text
Be, As, Cd, Hg, Pb, Tl
```

说明：有毒元素的定义不是唯一标准。如果把 `Sb, Se, Te, Cr, Co, Ni, Bi` 等也纳入“有毒 / 需谨慎元素”，筛除数量会增加。

### 3.3 贵金属

当前采用银、金和铂族金属：

```text
Ag, Au, Ru, Rh, Pd, Os, Ir, Pt
```

## 4. 复算代码

在 `material_P` 目录下运行：

```python
import pandas as pd
import re

HALOGENS = {"F", "Cl", "Br", "I"}

RADIOACTIVE = {"Ac", "Np", "Pa", "Pm", "Pu", "Tc", "Th", "U"}
TOXIC = {"Be", "As", "Cd", "Hg", "Pb", "Tl"}
PRECIOUS = {"Ag", "Au", "Ru", "Rh", "Pd", "Os", "Ir", "Pt"}

pat = re.compile(r"([A-Z][a-z]?)([0-9.]*)")

df = pd.read_csv("summary.csv")

def non_halogen_element(cif_file):
    formula = str(cif_file).split("_mp-")[0]
    elems = [el for el, _ in pat.findall(formula)]
    non_halogens = [el for el in elems if el not in HALOGENS]
    if len(non_halogens) != 1:
        raise ValueError((cif_file, formula, elems, non_halogens))
    return non_halogens[0]

df["M"] = df["cif_file"].map(non_halogen_element)

radio_mask = df["M"].isin(RADIOACTIVE)
toxic_mask = df["M"].isin(TOXIC)
precious_mask = df["M"].isin(PRECIOUS)

remove_mask = radio_mask | toxic_mask | precious_mask

print("total:", len(df))
print("radioactive:", int(radio_mask.sum()))
print("toxic:", int(toxic_mask.sum()))
print("precious:", int(precious_mask.sum()))
print("removed_union:", int(remove_mask.sum()))
print("remaining:", int((~remove_mask).sum()))
```

## 5. 统计结果

| 类别 | 元素集合 | 数量 |
|---|---|---:|
| 总数 | 全部 CIF | 1702 |
| 放射性 | Ac, Np, Pa, Pm, Pu, Tc, Th, U | 106 |
| 有毒 | Be, As, Cd, Hg, Pb, Tl | 245 |
| 贵金属 | Ag, Au, Ru, Rh, Pd, Os, Ir, Pt | 140 |
| 三类合计去除 | 三类并集 | 491 |
| 筛除后剩余 | 不含上述三类 | 1211 |

这三类在当前数据集中没有重叠，因此：

```text
106 + 245 + 140 = 491
1702 - 491 = 1211
```

## 6. 分元素计数

### 6.1 放射性

| 元素 | 数量 |
|---|---:|
| Ac | 10 |
| Np | 12 |
| Pa | 6 |
| Pm | 9 |
| Pu | 13 |
| Tc | 11 |
| Th | 16 |
| U | 29 |

合计：

```text
106
```

### 6.2 有毒

| 元素 | 数量 |
|---|---:|
| As | 6 |
| Be | 19 |
| Cd | 135 |
| Hg | 25 |
| Pb | 30 |
| Tl | 30 |

合计：

```text
245
```

### 6.3 贵金属

| 元素 | 数量 |
|---|---:|
| Ag | 33 |
| Au | 19 |
| Ir | 12 |
| Os | 6 |
| Pd | 18 |
| Pt | 19 |
| Rh | 17 |
| Ru | 16 |

合计：

```text
140
```

## 7. 与 1702 -> 1593 的关系

当前目录中没有找到明确把 `1702` 处理成 `1593` 的脚本或输出文件。

当前可复现的处理包括：

- `cif/*.cif`：1702 个。
- `merged_stats4/summary.csv`：1702 行。
- `mpdata.csv`：1702 行。
- `merged_stats4/polyhedra/*.cif`：1517 个。
- `merged_stats4/no_polyhedra/*.cif`：185 个。

也就是说，当前能确认的结构分类步骤是：

```text
1702 -> 1517 polyhedra + 185 no_polyhedra
```

按本文档三类元素筛除，则是：

```text
1702 -> 1211
```

如果需要得到 `1593`，需要进一步确认当时使用的筛选元素集合或对应的中间输出文件。
