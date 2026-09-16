#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polyhedra pipeline — FAST (sparse) version
------------------------------------------
- 一次抽取 polyhedra 贯穿全流程（避免重复 CrystalNN）
- 用 稀疏矩阵乘法 (SciPy) 或哈希回退，替代 O(P^2) 两两求交
- 无 NetworkX；度与连通分量用 scipy.sparse.csgraph（若可用）或轻量回退
- 三指标均为“平均值口径”：
    st1: 所有多面体的卤素配位数平均
    st2: 仅对共享>=1的多面体对，shared_count 的平均
    st3: 仅对度>=1的多面体，度的平均（与 st2 口径一致）
- JSON 中仍输出每个多面体的 degree（>=0），component_id 若 SciPy 可用则提供；否则为 None
"""

import os
import json
import shutil
import argparse
from multiprocessing import Pool, cpu_count
from itertools import combinations
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis import dimensionality as dimmod
from pymatgen.transformations.standard_transformations import SupercellTransformation

# ---------------- 参数配置 ----------------
MIN_NEIGHBORS_FOR_CLUSTER = 3
HALOGENS = {"F", "Cl", "Br", "I"}

# --------- 维度识别（CrystalNN → Larsen；fallback Gorai） ---------
def compute_dimension(structure: Structure) -> str:
    try:
        bonded = CrystalNN().get_bonded_structure(structure)
        dim_int = dimmod.get_dimensionality_larsen(bonded)  # 0/1/2/3
        return f"{dim_int}D"
    except Exception:
        try:
            dim_int = dimmod.get_dimensionality_gorai(structure)  # 1/2/3
            return f"{dim_int}D" if dim_int in (1, 2, 3) else "failed"
        except Exception:
            return "failed"

# ---------------- 扩胞 ----------------
def expand_structure(structure: Structure, scale_matrix=(3, 3, 3)) -> Structure:
    try:
        return SupercellTransformation(scale_matrix).apply_transformation(structure)
    except Exception:
        return structure

# --------------- 多面体识别（CrystalNN，仅卤素为配位体） ---------------
def extract_polyhedra(structure: Structure):
    cnn = CrystalNN()
    polyhedra = []
    pid = 0
    for i, site in enumerate(structure):
        sym = site.specie.symbol
        if sym in HALOGENS:
            continue  # 卤素不作中心
        try:
            nns = cnn.get_nn_info(structure, i)
            nb = [n['site_index'] for n in nns
                  if structure[n['site_index']].specie.symbol in HALOGENS]
            if len(nb) >= MIN_NEIGHBORS_FOR_CLUSTER:
                polyhedra.append({
                    "id": pid,
                    "center_index": i,
                    "neighbors": nb,                 # 允许重复，这里后面会 set 去重
                    "frac_coords": list(structure[i].frac_coords),
                    "degree": 0,                     # 运行后填充
                    "component_id": None             # 运行后填充（若可用）
                })
                pid += 1
        except Exception:
            continue
    return polyhedra

# --------- 稀疏矩阵法：从 polyhedra 构建 B（多面体×卤素）并计算指标 ---------
def compute_metrics_from_poly(structure_super: Structure, polyhedra):
    """
    返回：
      st1, st2, st3, degrees(list), comp_ids(list or None)
    - st1: mean CN over all polyhedra
    - st2: mean shared count over pairs with shared>=1
    - st3: mean node degree over ALL polyhedra (including degree==0 nodes)  ← 修复点
    """
    if not polyhedra:
        return 0.0, 0.0, 0.0, [], []

    # --- st1：平均配位数（全部多面体） ---
    st1 = float(np.mean([len(p["neighbors"]) for p in polyhedra]))

    # --- 构建 邻接指示（多面体×卤素） ---
    hal_map = {}
    rows, cols = [], []
    for p in polyhedra:
        pid = p["id"]
        for h in set(p["neighbors"]):   # set 去重
            if h not in hal_map:
                hal_map[h] = len(hal_map)
            rows.append(pid)
            cols.append(hal_map[h])

    n_poly = len(polyhedra)
    n_hal = len(hal_map)

    if n_hal == 0:
        degrees = [0]*n_poly
        comp_ids = list(range(n_poly))
        # st2=0.0, st3=平均度=0.0
        return st1, 0.0, 0.0, degrees, comp_ids

    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components

        data = np.ones(len(rows), dtype=np.int8)
        B = sp.csr_matrix((data, (rows, cols)), shape=(n_poly, n_hal))

        # 共享数矩阵 C = B * B^T
        C = (B @ B.T).tocsr()

        # 上三角（不含对角）非零的共享数
        shared_vals = sp.triu(C, k=1).data
        st2 = float(shared_vals.mean()) if shared_vals.size > 0 else 0.0

        # 度（按共享≥1 的邻接）：A = (C>0) - I 去掉自环后等价，但 sign() 后对角为1；我们不取对角
        A = C.sign()
        A.setdiag(0)            # 去掉对角，确保度数不被自环影响
        A.eliminate_zeros()

        deg = np.asarray(A.sum(axis=1)).ravel().astype(int)
        degrees = deg.tolist()

        # ✅ st3：全部节点（含度=0）平均
        st3 = float(deg.mean()) if deg.size > 0 else 0.0

        # 连通分量（基于 A）
        n_comp, labels = connected_components(A, directed=False, connection='weak', return_labels=True)
        comp_ids = labels.tolist()

        return st1, st2, st3, degrees, comp_ids

    except Exception:
        # 回退：哈希计数 + 轻量 BFS
        from collections import defaultdict, deque

        hal_to_polys = defaultdict(list)
        for p in polyhedra:
            pid = p["id"]
            for h in set(p["neighbors"]):
                hal_to_polys[h].append(pid)

        pair_counter = defaultdict(int)
        adj = defaultdict(set)

        for polys in hal_to_polys.values():
            polys.sort()
            for i in range(len(polys)):
                pi = polys[i]
                for j in range(i+1, len(polys)):
                    pj = polys[j]
                    pair_counter[(pi, pj)] += 1
                    adj[pi].add(pj)
                    adj[pj].add(pi)

        # st2
        st2 = float(np.mean(list(pair_counter.values()))) if pair_counter else 0.0

        # 度（全部节点；默认 0 度给 0）
        degrees = [len(adj[i]) for i in range(n_poly)]

        # ✅ st3：全部节点平均度
        st3 = float(np.mean(degrees)) if degrees else 0.0

        # 连通分量（无边节点各自成分量）
        comp_ids = [-1]*n_poly
        cid = 0
        for v in range(n_poly):
            if comp_ids[v] != -1:
                continue
            q = deque([v])
            comp_ids[v] = cid
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if comp_ids[w] == -1:
                        comp_ids[w] = cid
                        q.append(w)
            cid += 1

        return st1, st2, st3, degrees, comp_ids

# --------------- 分类（Level-1） ---------------
def classify_polyhedra(polyhedra):
    return "polyhedra" if polyhedra else "no_polyhedra"

# --------------- 单文件处理 ---------------
def process_file(filepath, input_root, output_root, supercell=(3,3,3)):
    try:
        st0 = Structure.from_file(filepath)
        dimension = compute_dimension(st0)

        s_super = expand_structure(st0, supercell)
        polyhedra = extract_polyhedra(s_super)

        # 三指标 + 度/分量（加速版）
        st1, st2, st3, degrees, comp_ids = compute_metrics_from_poly(s_super, polyhedra)

        # 回填 degree / component_id
        pid_index = {p["id"]: idx for idx, p in enumerate(polyhedra)}
        for pid, d in enumerate(degrees):
            if pid in pid_index:
                polyhedra[pid_index[pid]]["degree"] = int(d)
        if comp_ids:
            for pid, c in enumerate(comp_ids):
                if pid in pid_index:
                    polyhedra[pid_index[pid]]["component_id"] = int(c)

        category = classify_polyhedra(polyhedra)

        # 输出目录
        rel_path = os.path.relpath(filepath, input_root)
        outdir = os.path.join(output_root, category, os.path.dirname(rel_path))
        os.makedirs(outdir, exist_ok=True)

        # 复制 CIF & 写 JSON
        fn = os.path.basename(filepath)
        shutil.copy(filepath, os.path.join(outdir, fn))
        with open(os.path.join(outdir, fn + ".poly.json"), "w", encoding="utf-8") as f:
            json.dump({
                "structure_name": os.path.splitext(fn)[0],
                "dimension": dimension,
                "supercell": list(supercell),
                "st1": st1, "st2": st2, "st3": st3,
                "n_polyhedra": len(polyhedra),
                "polyhedra": polyhedra
            }, f, indent=2, ensure_ascii=False)

        print(f"[✓] {rel_path} → {category} | dim={dimension} "
              f"st1_avg={st1:.3f} st2_avg={st2:.3f} st3_avg={st3:.3f} | polys={len(polyhedra)}")
        return {"filename": rel_path, "dimension": dimension, "st1": st1, "st2": st2, "st3": st3}

    except Exception as e:
        print(f"[✗] Failed: {filepath} — {e}")
        return {"filename": os.path.relpath(filepath, input_root), "dimension": "failed",
                "st1": 0.0, "st2": 0.0, "st3": 0.0}

# --------------- 批量处理（合理并行 + chunksize） ---------------
def process_all_files(input_root, output_root, supercell):
    files = []
    for root, _, fs in os.walk(input_root):
        for f in fs:
            if f.lower().endswith((".cif", ".poscar", ".vasp", ".xyz")):
                files.append(os.path.join(root, f))

    if not files:
        print("[!] No structure files found.")
        return []

    procs = min(cpu_count(), len(files))
    chunksize = max(1, len(files) // (8 * max(procs, 1)))
    print(f"[*] Found {len(files)} files. Using {procs} workers, chunksize={chunksize}.")

    args = [(p, input_root, output_root, supercell) for p in files]
    results = []
    with Pool(processes=procs) as pool:
        for rec in pool.starmap(process_file, args, chunksize=chunksize):
            results.append(rec)
    return results

# --------------- 主函数 ---------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast polyhedra metrics (sparse-based)")
    parser.add_argument("--input", type=str, required=True, help="Input directory with structure files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--supercell", type=str, default="3,3,3", help="Supercell, e.g., 3,3,3")
    args = parser.parse_args()

    sc = tuple(int(x) for x in args.supercell.split(","))
    os.makedirs(args.output, exist_ok=True)

    rows = process_all_files(args.input, args.output, sc)
    if rows:
        df = pd.DataFrame(rows, columns=["filename", "dimension", "st1", "st2", "st3"]).sort_values("filename")
        df.to_csv(os.path.join(args.output, "summary.csv"), index=False)
        print(f"[✓] summary.csv written to {args.output}")
