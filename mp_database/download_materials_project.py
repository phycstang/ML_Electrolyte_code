#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpd.py — Download ONLY binary materials that contain exactly ONE halogen (F/Cl/Br/I)
排除卤素-卤素（如 F-Br、Cl-I）

依赖:  pip install mp-api pymatgen
用法:
  export MP_API_KEY='YOUR_KEY'
  python mpd.py
  # 可选参数见 --help
"""
from __future__ import annotations

import os
import time
import json
import csv
import argparse
import pathlib
from typing import List, Dict, Any, Tuple

from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element as PMGElement

HALOGENS_DEFAULT = ["F", "Cl", "Br", "I"]
HALOGENS_SET = set(HALOGENS_DEFAULT)

# ---------- 工具函数 ----------
def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def elem_list_to_symbols(elems: List[Any]) -> List[str]:
    """将 SummaryDoc.elements（可能是 Element 或 str）统一成符号字符串列表."""
    return [(e.symbol if isinstance(e, PMGElement) else str(e)) for e in elems]

def fetch_by_chemsys(mpr: MPRester, hx: str, fields: List[str]):
    """首选：chemsys='X-*'（服务端严格二元+含该卤素）"""
    return mpr.materials.summary.search(chemsys=f"{hx}-*", fields=fields)

def fetch_by_elements_numel(mpr: MPRester, hx: str, fields: List[str]):
    """兜底A：elements + num_elements (或 nelements 兼容)"""
    try:
        return mpr.materials.summary.search(elements=[hx], num_elements=2, fields=fields)
    except TypeError:
        return mpr.materials.summary.search(elements=[hx], nelements=2, fields=fields)

def client_filter_binary_single_halogen(
    docs, hx: str, forbid_halogen_pair: bool = True, verbose_limit: int = 8
) -> Tuple[List[Any], Dict[str, int], List[Tuple[str, str, List[str]]]]:
    """
    客户端强制过滤：二元、含 hx，且（可选）另一元素非卤素。
    返回: (保留列表, 丢弃原因计数, 样例若干).
    """
    kept, reasons = [], {"not_binary": 0, "no_hx": 0, "halogen_pair": 0}
    samples: List[Tuple[str, str, List[str]]] = []
    for d in docs:
        syms = elem_list_to_symbols(getattr(d, "elements", []))
        if len(syms) != 2:
            reasons["not_binary"] += 1
            if len(samples) < verbose_limit:
                samples.append(("not_binary", str(getattr(d, "material_id", "?")), syms))
            continue
        if hx not in syms:
            reasons["no_hx"] += 1
            if len(samples) < verbose_limit:
                samples.append(("no_hx", str(getattr(d, "material_id", "?")), syms))
            continue
        partner = syms[0] if syms[1] == hx else syms[1]
        if forbid_halogen_pair and partner in HALOGENS_SET:
            reasons["halogen_pair"] += 1
            if len(samples) < verbose_limit:
                samples.append(("halogen_pair", str(getattr(d, "material_id", "?")), syms))
            continue
        kept.append(d)
    return kept, reasons, samples

def safe_write_cif(struct: Structure, fp: pathlib.Path, retries: int = 3, delay: float = 0.1):
    for k in range(retries):
        try:
            struct.to(fmt="cif", filename=str(fp))
            return
        except Exception as e:
            if k == retries - 1:
                raise
            time.sleep(delay)

# ---------- 主逻辑 ----------
def main():
    ap = argparse.ArgumentParser(description="Download binary single-halogen materials from Materials Project.")
    ap.add_argument("--outdir", default="halide_structures_binary_onehalogen", help="输出目录")
    ap.add_argument("--halogens", default="F,Cl,Br,I", help="卤素列表, 逗号分隔（默认: F,Cl,Br,I）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    ap.add_argument("--max-per-halogen", type=int, default=None, help="每个卤素最多下载多少条（调试用）")
    ap.add_argument("--allow-halogen-pair", action="store_true", help="允许卤素-卤素（二元的两元素都为卤素）")
    ap.add_argument("--api-key", default=None, help="可显式传入 API Key（一般不建议；推荐用环境变量 MP_API_KEY）")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("MP_API_KEY") or os.getenv("MAPI_KEY")
    if not api_key:
        raise SystemExit("未发现 API Key。请先 `export MP_API_KEY='你的Key'` ，或用 --api-key 传入。")

    halogens = [h.strip() for h in args.halogens.split(",") if h.strip()]
    out_root = pathlib.Path(args.outdir)
    ensure_dir(out_root)

    fields = [
        "material_id", "formula_pretty", "elements", "nelements", "structure",
        "energy_above_hull", "formation_energy_per_atom", "is_stable",
        "theoretical", "deprecated", "band_gap", "density", "volume", "nsites",
        "last_updated",
    ]

    summary = {}
    metadata_rows: List[Dict[str, Any]] = []
    with MPRester(api_key) as mpr:
        # 版本提示（排错友好）
        try:
            import mp_api
            print(f"[info] mp_api version: {mp_api.__version__}")
        except Exception:
            pass

        for hx in halogens:
            print(f"\n=== {hx}: chemsys='{hx}-*' ===")
            docs = fetch_by_chemsys(mpr, hx, fields)
            print(f"[chemsys] got {len(docs)} docs before client filtering.")

            if len(docs) == 0:
                print("[chemsys] zero → fallback elements+(num|ne)elements")
                docs = fetch_by_elements_numel(mpr, hx, fields)
                print(f"[fallback] got {len(docs)} docs before client filtering.")

            # 客户端强制过滤（统一用符号字符串判断）
            filtered, reasons, samples = client_filter_binary_single_halogen(
                docs, hx, forbid_halogen_pair=(not args.allow_halogen_pair)
            )
            print(f"[filter] kept {len(filtered)} | dropped: {reasons}"
                  + (f" | samples: {samples}" if samples else ""))

            out_dir = out_root / hx
            ensure_dir(out_dir)
            written = 0

            for d in filtered:
                if args.max_per_halogen and written >= args.max_per_halogen:
                    break
                mpid = d.material_id
                formula = d.formula_pretty.replace(" ", "")
                cif_path = out_dir / f"{formula}_{mpid}.cif"

                if args.dry_run:
                    print(f"[dry-run] would write: {cif_path.name}")
                    metadata_rows.append({
                        "material_id": str(mpid),
                        "cif_file": cif_path.name,
                        "formula": formula,
                        **{key: getattr(d, key, None) for key in fields if key not in {
                            "material_id", "formula_pretty", "elements", "nelements", "structure"
                        }},
                    })
                    written += 1
                    continue

                if cif_path.exists():
                    structure = getattr(d, "structure", None)
                    metadata_rows.append({
                        "material_id": str(mpid),
                        "cif_file": cif_path.name,
                        "formula": formula,
                        "is_ordered": bool(structure.is_ordered) if structure is not None else None,
                        **{key: getattr(d, key, None) for key in fields if key not in {
                            "material_id", "formula_pretty", "elements", "nelements", "structure"
                        }},
                    })
                    written += 1
                    continue

                structure = getattr(d, "structure", None)
                if structure is None:
                    # 极少数情况下结构不在 summary 里，兜底单 ID 接口（官方示例提供的便捷方法）
                    structure = mpr.get_structure_by_material_id(mpid)
                try:
                    safe_write_cif(structure, cif_path)
                    metadata_rows.append({
                        "material_id": str(mpid),
                        "cif_file": cif_path.name,
                        "formula": formula,
                        "is_ordered": bool(structure.is_ordered),
                        **{key: getattr(d, key, None) for key in fields if key not in {
                            "material_id", "formula_pretty", "elements", "nelements", "structure"
                        }},
                    })
                    written += 1
                except Exception as e:
                    print(f"[warn] write CIF failed for {mpid}: {e}")
                time.sleep(0.02)  # 轻微节流

            summary[hx] = {"kept": len(filtered), "written": written, "dropped": reasons}
            print(f"Done {hx}: {written} files -> {out_dir}")

    # 元数据
    meta = {
        "filters": {
            "binary_only": True,
            "single_halogen": True,
            "forbid_halogen_pair": not args.allow_halogen_pair,
        },
        "halogens": halogens,
        "summary": summary,
    }
    with open(out_root / "_download_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if metadata_rows:
        columns = [
            "material_id", "cif_file", "formula", "energy_above_hull",
            "formation_energy_per_atom", "is_stable", "theoretical", "deprecated",
            "band_gap", "density", "volume", "nsites", "is_ordered", "last_updated",
        ]
        with open(out_root / "metadata.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in metadata_rows:
                writer.writerow({key: str(value) if key == "last_updated" and value is not None else value
                                 for key, value in row.items()})

    print("\nAll done.")
    print("Output:", str(out_root.resolve()))
    print("Per halogen:", {k: v["written"] for k, v in summary.items()})

if __name__ == "__main__":
    main()
