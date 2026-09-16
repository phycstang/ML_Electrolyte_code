
## 1702 个 CIF 结构包与实验记录/去重表（2026-09-16 新增）

- `MP_1702_cif_files.tar.gz` — 1702 个 MP 含卤素二元化合物的 CIF 结构文件打包（解压后为 `cif_files/` 目录，文件名格式 `化学式_mp编号.cif`）
- `MP_1702_实验记录与结构去重表.csv` — 逐条目表格，包含：
  - `experimentally_observed`：Yes/No（来自 MP 元数据 `theoretical` 字段，False=Experimentally Observed: Yes，共 933 条 Yes；另有 5 条元数据缺失记为 unknown）
  - `is_structure_representative` / `representative_material_id`：该结构是否为去重后代表，以及所在结构族的代表 ID
  - `structure_family_id` / `structure_family_size` / `n_duplicates_in_family` / `duplicate_material_ids`：所在结构族、族大小、族内其它重复结构的 MP 编号
  - 去重结果：1702 → **1575** 个代表结构；127 个条目为族内重复
- `MP_结构重复组清单.md` — 101 个多成员结构族（即互相重复的 MP 结构分组）的清单

去重方法：VESTA 成键拓扑签名 + 严格结构匹配，详见源项目 `ML_Electrolyte_end/build/screening/materials_project_structure_dedup_v1/`。
