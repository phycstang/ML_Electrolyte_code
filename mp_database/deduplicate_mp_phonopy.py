#!/usr/bin/env python3
"""Audit strict and approximate duplicates after Phonopy 0.5 A standardization."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import shutil
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.core import Composition, Structure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis"))
from phonopy_dedup_matching import DEFAULT_CONFIG, compare_geometry, groups_complete_link, prepare_matching_structure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def geometry_from_record(record, kind="primitive"):
    data = record[kind]
    return Structure(data["lattice_matrix_A"], data["symbols"], data["fractional_positions"])


def pair_task(first, second, structure_a, structure_b, config):
    result = compare_geometry(structure_a, structure_b, config)
    return {"material_id_a": first, "material_id_b": second, **result}


def experimental_status(value):
    if pd.isna(value):
        return "unknown"
    return "theoretical" if bool(value) else "experimentally_observed"


def representative_key(row):
    def number(value):
        return float(value) if pd.notna(value) else float("inf")
    dep = row.get("deprecated")
    dep_order = 1 if pd.isna(dep) else (2 if bool(dep) else 0)
    exp_order = {"experimentally_observed": 0, "theoretical": 1, "unknown": 2}[row["experimental_status"]]
    return dep_order, exp_order, number(row.get("energy_above_hull")), number(row.get("formation_energy_per_atom")), row["material_id"]


def group_profile(inventory, pair_results, profile, scope):
    """Group complete pair distances; unresolved inputs remain explicitly unassigned."""
    chosen = inventory if scope == "all1702" else inventory.loc[inventory.experimental_status.eq("experimentally_observed")]
    successful = chosen.loc[chosen.standardization_status.eq("ok")]
    allowed = set(successful.material_id)
    score_field = "strict_score" if profile == "strict_standardized" else "score"
    threshold = {"strict_standardized": 1.0, "near_tight": 0.5, "near_main": 1.0, "near_loose": 2.0}[profile]
    scores = {tuple(sorted((r["material_id_a"], r["material_id_b"]))): r.get(score_field)
              for r in pair_results if r["material_id_a"] in allowed and r["material_id_b"] in allowed}
    lookup = chosen.set_index("material_id").to_dict("index")
    assignments = chosen.copy()
    assignments["profile"] = profile
    assignments["scope"] = scope
    assignments["family_id"] = None
    assignments["family_size"] = None
    assignments["representative_material_id"] = None
    assignments["is_representative"] = None
    assignments["retain_entry"] = True
    families, membership = [], {}
    for formula, bucket in successful.groupby("formula", sort=True):
        groups = groups_complete_link(bucket.material_id.tolist(), scores, threshold)
        for index, members in enumerate(groups, 1):
            fid = f"{scope}::{profile}::{formula}::{index:04d}"
            representative = min(members, key=lambda mid: representative_key({"material_id": mid, **lookup[mid]}))
            values = []
            for first, second in itertools.combinations(members, 2):
                distance = scores.get(tuple(sorted((first, second))))
                if distance is None or not np.isfinite(distance) or distance > threshold:
                    raise ValueError("Complete-link group contains an unmatched pair")
                values.append(float(distance))
            families.append({"scope": scope, "profile": profile, "family_id": fid, "formula": formula,
                "family_size": len(members), "representative_material_id": representative,
                "member_material_ids": ";".join(members), "max_pair_score": max(values) if values else 0.0,
                "n_experimentally_observed": sum(lookup[mid]["experimental_status"] == "experimentally_observed" for mid in members),
                "n_distinct_space_groups": len({lookup[mid]["standard_space_group_number"] for mid in members})})
            for mid in members:
                membership[mid] = fid
                selected = assignments.material_id.eq(mid)
                assignments.loc[selected, "family_id"] = fid
                assignments.loc[selected, "family_size"] = len(members)
                assignments.loc[selected, "representative_material_id"] = representative
                assignments.loc[selected, "is_representative"] = mid == representative
                assignments.loc[selected, "retain_entry"] = mid == representative
    groups_table = pd.DataFrame(families)
    paired = [(a, b) for (a, b), score in scores.items() if score is not None and np.isfinite(score) and score <= threshold]
    boundary = [(a, b) for a, b in paired if membership[a] != membership[b]]
    n_failed = len(chosen) - len(successful)
    counts = {"scope": scope, "profile": profile, "score_threshold": threshold,
        "input_entries": len(chosen), "standardized_successfully": len(successful), "unresolved_entries": n_failed,
        "matching_pairs": len(paired), "duplicate_groups": sum(len_ > 1 for len_ in groups_table.family_size),
        "entries_in_duplicate_groups": int(groups_table.loc[groups_table.family_size.gt(1), "family_size"].sum()),
        "removed_duplicate_entries": len(successful) - len(groups_table),
        "groups_among_successful_entries": len(groups_table),
        "retained_entries_including_unresolved": len(groups_table) + n_failed,
        "passing_pairs_crossing_complete_link_groups": len(boundary),
        "groups_with_multiple_space_groups": int(groups_table.n_distinct_space_groups.gt(1).sum())}
    if counts["removed_duplicate_entries"] + counts["retained_entries_including_unresolved"] != len(chosen):
        raise ValueError("Input/retained/removed count does not reconcile")
    return counts, assignments, groups_table, pd.DataFrame(boundary, columns=["material_id_a", "material_id_b"])


def export(outdir, standardized_dir, records, inventory, pairs, config, computation_provenance):
    tables, summaries = {}, []
    main_assignment = main_families = strict_assignment = None
    for scope in ["all1702", "experimental933"]:
        for profile in ["strict_standardized", "near_tight", "near_main", "near_loose"]:
            counts, assignments, families, boundaries = group_profile(inventory, pairs, profile, scope)
            summaries.append(counts)
            label = f"{scope}__{profile}"
            assignments.to_csv(outdir / f"{label}__assignments.csv", index=False, encoding="utf-8-sig")
            families.to_csv(outdir / f"{label}__families.csv", index=False, encoding="utf-8-sig")
            boundaries.to_csv(outdir / f"{label}__cross_group_matching_pairs.csv", index=False, encoding="utf-8-sig")
            if scope == "all1702" and profile == "near_main":
                main_assignment, main_families = assignments, families
            if scope == "all1702" and profile == "strict_standardized":
                strict_assignment = assignments
            if scope == "experimental933" and profile == "near_main":
                tables["实验933近似分组"] = assignments
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(outdir / "deduplication_counts.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(outdir / "input_standardization_inventory.csv", index=False, encoding="utf-8-sig")
    flat_pairs = []
    for record in pairs:
        row = {k: v for k, v in record.items() if not isinstance(v, (dict, list))}
        for label in ["main", "strict"]:
            evidence = record.get(f"{label}_evidence")
            if evidence:
                row.update({f"{label}__{key}": value for key, value in evidence["metrics"].items()})
        flat_pairs.append(row)
    pd.DataFrame(flat_pairs).to_csv(outdir / "pair_audit.csv", index=False, encoding="utf-8-sig")
    with gzip.open(outdir / "pair_alignment_evidence.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(pairs, handle, ensure_ascii=False, allow_nan=False)
    failures = inventory.loc[~inventory.standardization_status.eq("ok")]
    definitions = pd.DataFrame([
        ("范围", "本地1702个CIF；933实验观察、764理论、5实验状态未知。全部保留状态，不重新下载MP。"),
        ("Phonopy标准化", "Phonopy 2.46.0、symprec=0.5 Å，调用与--symmetry相同的流程。包括晶格理想化和等价原子位置平均。标准化失败的条目保持未判定，不当作确定独立结构。"),
        ("Pu质量元数据", "13条Pu结构因Phonopy缺少Pu质量，按pymatgen参考值244 u仅在内存补齐质量后沿用同一对称性算法；不改安装包、原CIF或几何，不用于声子计算。"),
        ("严格相同", "指Phonopy标准化后的几何：对应晶轴对称相对差≤1e-5、角差≤0.001°、最大及RMS内部位移≤1e-4 Å、每原子体积相对差≤1e-4。该统计不等同于原始CIF仅换晶胞去重。"),
        ("近似主阈值", "同精确组成且元素一一对应；对应晶轴对称相对差≤1%、角差≤1°、最大内部位移≤0.10 Å、RMS≤0.05 Å、每原子体积相对差≤3%。不自动缩放体积，不做子集或匿名元素匹配。"),
        ("内部位移", "原点/基矢/整数超胞对齐后，在平均晶格度量中用真正最短周期向量求位置差，并去除均值平移；均匀晶格应变另由长度、角度及体积指标约束。不是直接把StructureMatcher.stol当作Å。"),
        ("对称相对差", "长度和每原子体积使用max(a/b,b/a)-1，两结构交换顺序后定义一致。"),
        ("空间群", "保留空间群作为注释；同空间群不是重复的充分条件。近似分组由几何阈值确定，可包含空间群不同但几何接近的结构，另计此类组数。"),
        ("配对分数", "候选对齐中取五项几何差/主阈值的最大值，再取最小分数。score≤0.5/1/2分别对应紧/主/宽阈值；采用元素约束的最小二乘原子匹配及候选基矢、原点枚举。"),
        ("组内一致", "同组成内按MP ID固定顺序建立complete-link层次树，在相应阈值切树；组内每一对都必须通过，避免A≈B、B≈C却A不≈C时链式合并。跨组仍通过的配对另表保留。"),
        ("敏感性", "近似判据所有阈值同时乘0.5/1/2，同一距离树切分。紧/主/宽结果为嵌套分组；阈值是此次明确选定的操作定义，0.5 Å自身不能唯一决定重复数。"),
        ("代表选择", "依次优先非deprecated、实验观察、较低凸包能差、较低形成能、字典序MP ID；能量缺失排后。不用粘弹性标签或历史四特征限制结构分组。"),
        ("保留条目数", "成功标准化后的组数加未判定条目数；不能把尚未成功标准化的材料称为已证明独立的结构。"),
        ("重复数量", "removed_duplicate_entries为每组保留一个代表后可合并的多余条目数；matching_pairs为通过的配对数；duplicate_groups为成员≥2的组数，三者含义不同。"),
        ("实验子集", "933子集使用相同配对结果独立进行层次分组；不是简单按全1702组的代表是否实验观察删选。"),
    ], columns=["项目", "定义"])
    tables = {"数量汇总": summary_table, "全部1702近似分组": main_assignment,
              "近似重复组": main_families.loc[main_families.family_size.gt(1)],
              "近似可合并条目": main_assignment.loc[main_assignment.retain_entry.eq(False)],
              "标准化后严格分组": strict_assignment,
              **tables, "标准化未判定": failures, "判据说明": definitions}
    excel = outdir / "MP1702_Phonopy0.5_结构去重统计.xlsx"
    with pd.ExcelWriter(excel, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "C2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = openpyxl.styles.Font(bold=True, color="FFFFFF")
                cell.fill = openpyxl.styles.PatternFill("solid", fgColor="225577")
            for index, col in enumerate(table.columns, 1):
                sheet.column_dimensions[openpyxl.utils.get_column_letter(index)].width = 64 if col in ["定义", "member_material_ids", "error_reason"] else 25
    back = pd.read_excel(excel, sheet_name="数量汇总")
    if not np.allclose(back.select_dtypes(include="number"), summary_table.select_dtypes(include="number")):
        raise ValueError("Excel summary roundtrip mismatch")
    back = pd.read_excel(excel, sheet_name="全部1702近似分组")
    if back.material_id.tolist() != main_assignment.material_id.tolist() or len(back) != 1702:
        raise ValueError("Excel full input coverage mismatch")
    by_id = {record["material_id"]: record for record in records}
    for profile, assignments in [("near_main", main_assignment), ("strict_standardized", strict_assignment)]:
        archive = outdir / f"{profile}_standardized_representative_cifs.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for row in assignments.loc[assignments.is_representative.eq(True)].to_dict("records"):
                record = by_id[row["material_id"]]
                data = record["primitive"]
                # Copy the exact standardized geometry file from the input stage.
                path = standardized_dir / record["primitive_cif_path"]
                if not path.is_file():
                    raise FileNotFoundError(path)
                zf.write(path, f"{row['formula']}_{row['material_id']}_standardized.cif")
        with zipfile.ZipFile(archive) as zf:
            if zf.testzip() is not None:
                raise ValueError("Representative CIF ZIP CRC failed")
    sn_pair = next(r for r in pairs if {r["material_id_a"], r["material_id_b"]} == {"mp-29179", "mp-569152"})
    validation = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_entries": len(inventory), "all_ids_unique": inventory.material_id.is_unique,
        "input_experimental_status_counts": dict(Counter(inventory.experimental_status)),
        "standardization_status_counts": dict(Counter(inventory.standardization_status)),
        "mass_metadata_compatibility_count": int(inventory.mass_metadata_patch_applied.sum()),
        "standardization_unresolved_ids": failures.material_id.tolist(),
        "all_original_cif_hashes_reverified": True,
        "matched_same_formula_pairs": len(pairs), "pair_status_counts": dict(Counter(r["status"] for r in pairs)),
        "complete_link_all_within_group_pairs_verified": True, "excel_roundtrip_verified": True,
        "sncl2_example": {k: v for k, v in sn_pair.items() if "evidence" not in k},
        "counts": summaries, "comparison_config": config, "provenance": computation_provenance}
    write_json(outdir / "deduplication_summary.json", validation)
    main_count = next(r for r in summaries if r["scope"] == "all1702" and r["profile"] == "near_main")
    strict_count = next(r for r in summaries if r["scope"] == "all1702" and r["profile"] == "strict_standardized")
    (outdir / "README.md").write_text(
        "# 1702个MP二元卤化物：Phonopy 0.5 Å标准化及结构去重\n\n"
        f"共1702条：{main_count['standardized_successfully']}条完成标准化，{main_count['unresolved_entries']}条未判定。"
        "实验观察933条，理论764条，实验状态未知5条；使用本地完整CIF快照。\n\n"
        f"标准化后严格几何相同：{strict_count['duplicate_groups']}个重复组，可合并{strict_count['removed_duplicate_entries']}条，"
        f"保留{strict_count['retained_entries_including_unresolved']}条（含未判定）。\n\n"
        f"近似主阈值：{main_count['duplicate_groups']}个重复组，可合并{main_count['removed_duplicate_entries']}条，"
        f"保留{main_count['retained_entries_including_unresolved']}条（含未判定）。"
        "主阈值为晶格长度对称差1%、角度1°、最大内部位移0.10 Å、RMS位移0.05 Å、每原子体积差3%；所有条件须同时通过。"
        "紧/宽敏感性分别为各阈值的0.5和2倍。\n\n"
        "Phonopy的0.5 Å用于每个结构的对称性识别及理想化，不能单独定义跨材料重复数。"
        "近似相同与严格相同的操作定义在Excel/配置中分开列出；标准化后严格相同也不表示原始CIF只有晶胞选择差异。"
        "空间群作注释，几何和元素一一对应决定配对，不按空间群或已有四特征直接合并。\n\n"
        "采用complete-link层次分组，组内每一对均通过，避免链式合并。"
        "配对通过却处于不同组的边另表保留，代表选择规则不影响组内距离。"
        "实验933子集独立切分层次树，其数值可能不同于直接截取全库代表。\n\n"
        "原始/标准化结构和标准化诊断保留在../materials_project_phonopy_symprec_0p5_standardized；"
        "两个代表CIF包仅包含成功标准化且被选作代表的结构，未判定条目不会伪造标准CIF。"
        "pair_alignment_evidence.json.gz保存每一对的整数基矢变换、平移、原子对应和Å单位残差；"
        "每原子内部位移使用平均晶格的真实最短周期向量，均匀晶格应变另列。\n\n"
        "复现：`.venvs/phonopy_sncl2/bin/python src/analysis/deduplicate_mp_phonopy.py --outdir results/mp_phonopy_dedup_rerun`。\n",
        encoding="utf-8")
    reproduce = outdir / "reproduce"
    reproduce.mkdir(exist_ok=True)
    for relative in ["src/analysis/deduplicate_mp_phonopy.py", "src/analysis/phonopy_dedup_matching.py",
                     "src/analysis/standardize_mp_phonopy.py", "src/screening/deduplicate_cell_equivalent_structures.py",
                     "src/screening/deduplicate_candidate_structures.py", "tests/test_phonopy_dedup_matching.py"]:
        path = reproduce / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    standardization_audit = outdir / "standardization_audit"
    standardization_audit.mkdir(exist_ok=True)
    for name in ["summary.json", "cli_validation_report.json", "mass_compatibility_and_failures_validation.json",
                 "retry_missing_mass_metadata.py", "cli_with_pu_reference_mass.py", "README.md", "checkpoint.json.gz"]:
        if (standardized_dir / name).is_file():
            shutil.copy2(standardized_dir / name, standardization_audit / name)
    archive = outdir / "MP1702_Phonopy0.5_去重结果与代表结构.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(outdir.rglob("*")):
            if path.is_file() and path != archive:
                zf.write(path, path.relative_to(outdir))
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise ValueError("Full ZIP CRC failed")
    print(summary_table.to_string(index=False), flush=True)
    print(json.dumps({"excel": str(excel), "zip": str(archive)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", type=Path, default=ROOT / "results/materials_project_phonopy_symprec_0p5_standardized")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/materials_project_phonopy_symprec_0p5_dedup")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    standardized_dir, outdir = args.standardized_dir.resolve(), args.outdir.resolve()
    checkpoint = standardized_dir / "checkpoint.json.gz"
    with gzip.open(checkpoint, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    records = raw["records"]
    if not raw["complete"] or len(records) != 1702 or len({r["material_id"] for r in records}) != 1702:
        raise ValueError("Standardization checkpoint incomplete or not the 1702 structure set")
    metadata_path = ROOT / "data/candidates/materials_project/metadata.csv"
    metadata = pd.read_csv(metadata_path)
    if set(metadata.material_id) != {r["material_id"] for r in records}:
        raise ValueError("Metadata and standardization input sets differ")
    metadata["experimental_status"] = metadata.theoretical.map(experimental_status)
    lookup = metadata.set_index("material_id").to_dict("index")
    inventory_rows, structures = [], {}
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    prepared_config = {**config, "inputs_already_reduced": True}
    for record in records:
        mid = record["material_id"]
        path = ROOT / "data/candidates/materials_project/cif" / record["cif_file"]
        if sha(path) != record["source_cif_sha256"]:
            raise ValueError(f"Source CIF changed: {mid}")
        formula = Composition(record["cif_file"].split("_mp-", 1)[0]).reduced_formula
        row = {"material_id": mid, **lookup[mid], "formula": formula,
               "standardization_status": record["status"], "error_stage": record.get("error_stage"),
               "error_reason": record.get("error_reason"), "source_cif_sha256": record["source_cif_sha256"],
               "mass_metadata_patch_applied": bool(record.get("mass_metadata_patch")),
               "volume_per_atom_relative_change_by_standardization": record.get("volume_per_atom_relative_change"),
               "original_to_primitive_atom_multiplicity": record.get("original_to_primitive_atom_multiplicity")}
        for kind in ["original", "primitive", "conventional"]:
            if record.get(kind):
                for key in ["n_atoms", "volume_A3"]:
                    row[f"{kind}__{key}"] = record[kind][key]
        for source_key, prefix in [("original_symmetry_at_0p01_A", "original_0p01"),
            ("original_symmetry_at_requested_tolerance", "original_0p5"),
            ("standard_primitive_symmetry_at_requested_tolerance", "standard")]:
            sym = record.get(source_key) or {}
            row[f"{prefix}_space_group_number"] = sym.get("number", sym.get("space_group_number"))
            row[f"{prefix}_space_group"] = sym.get("international", sym.get("space_group"))
        if record["status"] == "ok":
            structure = geometry_from_record(record)
            if structure.composition.reduced_composition != Composition(formula).reduced_composition:
                raise ValueError(f"Composition changed: {mid}")
            structures[mid] = prepare_matching_structure(structure, config)
            row["matching_n_atoms"] = len(structures[mid])
        inventory_rows.append(row)
    inventory = pd.DataFrame(inventory_rows).sort_values("material_id").reset_index(drop=True)
    provenance = {"standardization_checkpoint_sha256": sha(checkpoint),
        "standardization_provenance": raw["provenance"], "metadata_sha256": sha(metadata_path),
        "scripts_sha256": {relative: sha(ROOT / relative) for relative in ["src/analysis/deduplicate_mp_phonopy.py", "src/analysis/phonopy_dedup_matching.py", "src/analysis/standardize_mp_phonopy.py"]},
        "versions": {name: importlib.metadata.version(name) for name in ["phonopy", "spglib", "pymatgen", "numpy", "scipy", "pandas", "joblib"]}}
    pair_checkpoint = outdir / "comparison_checkpoint.json.gz"
    if args.export_only:
        with gzip.open(pair_checkpoint, "rt", encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved["config"] != config or saved["provenance"]["standardization_checkpoint_sha256"] != sha(checkpoint):
            raise ValueError("Pair checkpoint inputs/config differ")
        if saved["provenance"]["scripts_sha256"]["src/analysis/phonopy_dedup_matching.py"] != sha(ROOT / "src/analysis/phonopy_dedup_matching.py"):
            raise ValueError("Pair comparison implementation differs")
        pairs = saved["pairs"]
        provenance["pair_computation_provenance"] = saved["provenance"]
    else:
        outdir.mkdir(parents=True, exist_ok=False)
        pairs_to_run = []
        successful = inventory.loc[inventory.standardization_status.eq("ok")]
        for _, bucket in successful.groupby("formula", sort=True):
            pairs_to_run.extend(itertools.combinations(sorted(bucket.material_id), 2))
        print(f"Comparing {len(pairs_to_run)} same-formula pairs from {len(structures)} standardized structures", flush=True)
        start = time.monotonic()
        pairs = []
        results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", batch_size="auto")(
            delayed(pair_task)(first, second, structures[first], structures[second], prepared_config) for first, second in pairs_to_run)
        for i, result in enumerate(results, 1):
            pairs.append(result)
            if i % 500 == 0 or i == len(pairs_to_run):
                print(f"Compared {i}/{len(pairs_to_run)} pairs in {time.monotonic()-start:.1f}s", flush=True)
        pairs.sort(key=lambda row: (row["material_id_a"], row["material_id_b"]))
        with gzip.open(pair_checkpoint, "wt", encoding="utf-8") as handle:
            json.dump({"config": config, "provenance": provenance, "pairs": pairs}, handle, ensure_ascii=False, allow_nan=False)
    export(outdir, standardized_dir, records, inventory, pairs, config, provenance)


if __name__ == "__main__":
    main()
