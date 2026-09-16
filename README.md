# ML_Electrolyte_code

从 nano 集群 `/data/home/tmy/` 下的 ML_Electrolyte、ML_Electrolyte1、ML_Electrolyte2、ML_Electrolyte3、ML_Electrolyte_end、MLElectrolyte_Clustering 六个项目目录中整理出的程序代码，只保留与 **特征制作**、**结构研究**、**MP (Materials Project) 数据库下载** 和 **结构去重** 相关的脚本。机器学习训练/建模程序、数据文件、日志和结果未包含在内。

## 目录结构

### `feature_engineering/` — 特征制作与清洗

来自各项目目录的特征工程脚本：

- `make_features*.py` — 特征生成：全栈特征（fullstack）、价电子/电离能/半径特征（valence_IE_radius）、T0/T1/T2 组分特征（mx_features_T0T1T2）、超参考版本（max/plus）等
- `make_features_25_10_10.py` — 2025-10-10 版本特征制作
- `halide_minifeats*.py` — 卤化物小特征集（含 v2、end 不同版本）
- `clean_data.py` / `clean_and_report.py` / `clean_features.py` / `feature_clean.py` / `clean_csv.py` — 数据与特征清洗、缺失值填充与报告
- `preprocess_features.py` / `preprocess_and_analyze_features*.py` / `apply_scaler.py` — 特征预处理、标准化与分析
- `feature_pool_builder*.py` — 特征池构建
- `feature_selection_full_pipeline.py` / `feature_study_pipeline.py` / `feature_optimize.py` / `pure_filter_features.py` — 特征筛选/优化流水线
- `generate_composition_features*.py` — 组分特征生成
- `generate_dataset_deterministic.py` — 数据集确定性生成

来源：ML_Electrolyte（make_features、feature_clean、feature_pool_builder、feature_study_pipeline）、ML_Electrolyte1（make_features）、ML_Electrolyte2（make_features）、ML_Electrolyte3（make_feat、clean_data、clean_feature、make_feature25_10_10）、ML_Electrolyte_end（src/features、experiments/features、paper/figure1_source）

### `structure_research/` — 结构研究

- `feature_study.py` / `plot_analysis.py` / `study_features.py` / `viz_features.py` — 特征研究、特征可视化
- `analysis_extended.py` / `plot_scores_and_periodic.py` / `plot_score.py` — 数据分析与打分分布/元素周期表绘图
- `ptable_pymatviz_svg.py` — 用 pymatviz 画元素周期表 SVG
- `mpd.py` / `t23.py` / `new.py` / `make.py` / `make_struct_csv.py` — 三角晶格 (triangular lattice, t23) 结构生成、随机结构 CSV 制作
- `polyhedra_classifier1.py` / `structure_dimension_classify1.py` — 多面体分类、结构维度（低维/三维）分类
- `polyhedron_geometry.py` / `periodic_polyhedron_union.py` / `compute_experimental_polyhedron_volume.py` — 多面体几何与周期性多面体并集体积计算
- `bonding_vesta.py` / `extract_features.py` / `run_discovery.py` — VESTA 风格成键判定、七特征 (seven_feature_v1) 原型发现流程
- `screening_1702_process.md` — 1702 个候选物筛选手记
- `merge.sh` — 多面体统计合并脚本
- `seven_feature_v1/` — 七特征提取与检索子包（common、dataset、extract、model、pipeline、radii、retrieval、validate）
- `triangular_lattice/` — 三角晶格参考实现（t23_reference.py）

来源：ML_Electrolyte（feature_study、study_features、analysis、pymatviz、picture）、ML_Electrolyte1（new.py）、ML_Electrolyte3（new_material、data_analyze）、ML_Electrolyte_end（src/seven_feature_v1、src/analysis、src/discovery、experiments/triangular_lattice）

### `mp_database/` — MP 数据库下载与结构去重

- `download_materials_project.py` / `fetch_mp_metadata.py` — 从 Materials Project 下载候选结构/元数据
- `filter_candidate_elements*.py` / `generate_synthetic_candidates.py` — 按元素过滤候选物、合成候选物生成
- `compute_structure_metrics*.py` — 结构度量计算（含 vesta_v2 与 legacy CrystalNN 版本）
- `deduplicate_candidate_structures.py` / `deduplicate_cell_equivalent_structures.py` / `deduplicate_experimental_structures.py` — 候选物结构去重、晶胞等价去重、实验结构去重
- `deduplicate_training_data*.py` / `deduplicate_dataset.py` — 训练数据去重
- `deduplicate_mp_phonopy.py` / `phonopy_dedup_matching.py` / `standardize_mp_phonopy.py` / `validate_structure_dedup_outputs.py` — MP 与 phonopy 结构对齐/标准化/去重及输出校验
- `build_canonical_candidate_model_input.py` / `prepare_model_input.py` / `subset_candidate_table_to_representatives.py` — 规范化候选表与模型输入准备
- `materials_project_screening.md` — MP 筛选流程文档

来源：ML_Electrolyte（deduplicate_dataset）、ML_Electrolyte_end（src/screening、src/data、src/analysis、docs、paper/figure1_source）

### `clustering_structure/` — 基于结构描述符（SOAP/MBTR）的聚类研究

- `halide_only_soap.py` / `halide_only_mbtr.py` — 纯卤化物体系的 SOAP / MBTR 聚类
- `halide_sp.py` — SOAP + 结构描述符组合聚类
- `REMatch_soap.py` — REMatch 核 SOAP 聚类
- `plot_umap_clusters.py` / `plot_clusters.py` — UMAP / 聚类结果绘图
- `cluster_*.py` / `search_soap_mbtr.py` — ML_Electrolyte_end 中整理后的聚类与描述符超参搜索版本
- `semi_supervised/` — 半监督聚类（Semi-supervised.py）
- `python.sh` / `halide-gpu-cuda118.yml` — 运行环境脚本与 conda 环境定义

来源：MLElectrolyte_Clustering、ML_Electrolyte_end（src/clustering）

## 说明

- 各脚本保持了原始文件名；重命名（加后缀 `_v2`、`_end`、`_paper` 等）仅用于区分同一脚本的不同版本
- 环境依赖：Python 3 + pymatgen、matminer、dscribe、umap-learn、scikit-learn 等；具体见 `clustering_structure/halide-gpu-cuda118.yml`
