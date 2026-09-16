# 1702 个 CIF 的来源、晶体去重与元素筛选统计

本文档记录当前项目中 1702 个 Materials Project CIF 的来源条件、晶体结构去重，以及按“放射性 / 有毒 / 贵金属”三类元素进行筛除的统计过程。原始 1702 条清单不覆盖；当前后续分析以 1575 个结构族代表为准。

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

### 5.1 原始 1702 个结构条目（历史口径）

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

### 5.2 晶体去重后的 1575 个代表（当前口径）

| 类别 | 数量 |
|---|---:|
| 原始结构条目 | 1702 |
| 晶体结构族代表 | 1575 |
| 合并的重复条目 | 127 |
| 多成员结构族 | 101 |
| 放射性代表 | 101 |
| 严格有毒代表 | 239 |
| 贵金属代表 | 127 |
| 三类并集排除 | 467 |
| 通过元素规则 | 1108 |

三类元素集合互不重叠，因此：

```text
101 + 239 + 127 = 467
1575 - 467 = 1108
```

去重使用“同一约化化学式 + 相同历史 `dim/st1/st2/st3` 八位签名 + StructureMatcher 容差等价”规则。真实多晶型和历史结构特征不同的结构继续保留。完整映射、代表清单和验证报告分别见：

- [`structure_dedup_assignments.csv`](../build/screening/materials_project_structure_dedup_v1/structure_dedup_assignments.csv)；
- [`representative_inventory.csv`](../build/screening/materials_project_structure_dedup_v1/representative_inventory.csv)；
- [`validation_report.json`](../results/screening/materials_project_structure_dedup_v1/validation_report.json)。

因此，1575 是当前模型安全的保守口径，不是脱离规则的唯一“纯几何”数量。去掉四结构量保护、固定 CIF 字典序时得到 1547 个族；由于 StructureMatcher 在容差边界存在方向与顺序敏感性，顺序扰动范围为 1544–1554。完整审计见 [`TOPOLOGY_GUARD_SENSITIVITY.md`](../results/screening/materials_project_structure_dedup_v1/TOPOLOGY_GUARD_SENSITIVITY.md)。

## 6. 分元素计数

### 6.1 放射性

| 元素 | 原始 1702 | 去重代表 1575 |
|---|---:|---:|
| Ac | 10 | 10 |
| Np | 12 | 9 |
| Pa | 6 | 6 |
| Pm | 9 | 9 |
| Pu | 13 | 12 |
| Tc | 11 | 10 |
| Th | 16 | 16 |
| U | 29 | 29 |

合计：

```text
原始 106；去重后 101。
```

### 6.2 有毒

| 元素 | 原始 1702 | 去重代表 1575 |
|---|---:|---:|
| As | 6 | 6 |
| Be | 19 | 17 |
| Cd | 135 | 133 |
| Hg | 25 | 24 |
| Pb | 30 | 29 |
| Tl | 30 | 30 |

合计：

```text
原始 245；去重后 239。
```

### 6.3 贵金属

| 元素 | 原始 1702 | 去重代表 1575 |
|---|---:|---:|
| Ag | 33 | 28 |
| Au | 19 | 19 |
| Ir | 12 | 11 |
| Os | 6 | 6 |
| Pd | 18 | 15 |
| Pt | 19 | 17 |
| Rh | 17 | 17 |
| Ru | 16 | 14 |

合计：

```text
原始 140；去重后 127。
```

## 7. 晶体结构去重结果及与“1593”的关系

历史文档曾出现 `1702 → 1593`，但没有找到其脚本、判据或输出，不能把 1593 当作可复现结果。当前首次完成并保存的可审计结构去重结果是：

```text
1702 个 MP 结构条目
→ 1575 个容差等价结构族代表
→ 1108 个通过三类元素规则的代表
```

StructureMatcher 固定参数为 `ltol=0.20`、`stol=0.30`、`angle_tol=5°`、`primitive_cell=True`、`scale=True`、`attempt_supercell=True`、`allow_subset=False`。代表按“非 deprecated → 最低凸包能 → 最低形成能 → CIF 名”确定，绝不按预测分数选择。1702 行完整族映射保留，因此 127 条被合并记录仍可追溯。

历史上另外可复算、但不属于晶体去重的结构分类包括：

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

原始条目直接按三类元素筛除是：

```text
1702 -> 1211
```

它与当前“先晶体去重、再按代表结构筛元素”的 `1575 → 1108` 口径不同，二者不能混写。

## 8. 实验观察记录的独立结构去重（历史模型口径）

按项目保存的 MP 元数据（2026-09-02 获取），原始 1702 条中有 933 条 `theoretical=False`，对应 Experimentally Observed: Yes；另有 764 条为 True、5 条元数据缺失。

对这 933 条实验记录单独运行上述 v1 保守结构去重规则，得到 **903 个代表结构，合并 30 条重复记录**，涉及 28 个多成员结构族，CIF 解析及匹配错误为 0。代表均从实验子集内选择。该结果包含 580 种约化化学式。

| 卤素 | 去重前 | 去重后 | 合并条目 |
|---|---:|---:|---:|
| F | 219 | 207 | 12 |
| Cl | 233 | 228 | 5 |
| Br | 180 | 172 | 8 |
| I | 301 | 296 | 5 |
| 合计 | 933 | 903 | 30 |

本结果仍保留 `dim/st1/st2/st3` 八位签名限制，不是纯几何去重口径。方法、代表清单、完整映射、CIF 压缩包及复现命令见[实验观察结构去重结果](../results/screening/materials_project_experimental_structure_dedup_v1/README.md)。原有 1702 条库存与 1575 个全库存代表结果保留。

## 9. 仅因晶胞选择不同而重复（用户明确后的口径）

用户进一步明确，只希望合并同一周期性结构因原胞、常规胞、超胞或基矢/原点选择不同而出现的重复。第 8 节的 903 个结果使用宽容差及等体积缩放，不满足这一严格范围。

重新检查时去掉历史结构特征限制，显式以 `1e-4 Å` 容差约化原胞，关闭体积缩放，以长度相对容差 `1e-5`、角度容差 `0.001°`、最大原子位移 `1e-4 Å` 检查晶胞变换及原子一一对应。共检查 7314 个同组成结构对，**未发现纯晶胞表示重复，933 条均保留**。这是严格数值判据下的结果，不表示 933 种不同晶型；独立弛豫造成的小幅几何差异没有自动合并。

完整参数、变换不变性验证、旧结果差异原因及输出见[晶胞表示等价检查](../results/screening/materials_project_experimental_cell_equivalence_v1/README.md)。

## 10. 全部 1702 条的严格晶胞等价检查

将第 9 节的相同判据应用于全部 1702 个 CIF（包含 5 条 MP 元数据缺失记录），得到 **1 个重复组，涉及 2 条记录，合并 1 条后保留 1701 条**。

唯一重复组为 `VF5_mp-1160849.cif` 与 `VF5_mp-2041039.cif`，两文件内容逐字节相同。代表保留 `mp-1160849`。除这对完全相同文件外，未找到额外的纯晶胞表示重复；此组不属于 933 条实验观察子集。

全库存共核查 9479 个同组成结构对，方法、重复清单、代表 CIF 压缩包及完整映射见[全部 1702 条的晶胞等价去重结果](../results/screening/materials_project_all_cell_equivalence_v1/README.md)。该 1701 口径与第 5–7 节的历史模型容差去重 1575 不同。
