#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REMatch-SOAP pipeline (HAL-only, DScribe 1.x/2.x compatible)
CIF -> [per-atom SOAP on halogen-only -> row L2 normalize]
    -> REMatchKernel (global similarity K, float64)
    -> kernel-induced distance D (float64, symmetric, zero diag, C-contiguous)
    -> UMAP(metric='precomputed') -> HDBSCAN(metric='precomputed')
    -> CSV/JSON outputs
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel

from umap import UMAP      # 预计算距离用 CPU 版最稳
import hdbscan
from sklearn.preprocessing import normalize

# ======================= CONFIG =======================
CONFIG: Dict[str, object] = {
    "cif_dir": "data/candidates/materials_project/cif",
    "outdir": "results/clustering/rematch_soap",

    # SOAP（兼容 1.x/2.x 键名；下方会映射）
    "soap": {
        "r_cut": 7.0,   # 旧版可用 rcut；脚本会自动兼容
        "n_max": 8,
        "l_max": 6,
        "sigma": 0.5,
        "periodic": True,
        "batch": 64
    },

    # REMatch 参数
    "rematch": {
        "metric": "linear",   # 'linear' 或 'rbf'
        "gamma": 1.0,         # 仅 'rbf' 有效
        "alpha": 1.0,         # 小→偏最佳匹配；大→偏平均
        "threshold": 1e-6
    },

    # UMAP（预计算距离）
    "umap": {
        "n_neighbors": 45,
        "min_dist": 0.03,
        "n_components": 2,
        "random_state": 42
    },

    # HDBSCAN（预计算距离；无软概率）
    "hdbscan": {
        "min_cluster_size": 80,
        "min_samples": 10,
        "cluster_selection_method": "leaf"
    },

    "seed": 42
}
# =====================================================


# ======================= IO & utils =======================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_cifs(cif_dir: Path) -> List[Path]:
    return sorted(cif_dir.rglob("*.cif"))


def load_structures(cif_paths: Iterable[Path]) -> Tuple[List[Structure], pd.DataFrame]:
    structs, rows = [], []
    for p in tqdm(cif_paths, desc="Loading CIFs"):
        try:
            st = Structure.from_file(str(p))
            structs.append(st)
            comp = "-".join(sorted({el.symbol for el in st.composition.elements}))
            rows.append({"file": p.name, "path": str(p), "composition": comp})
        except Exception as e:
            rows.append({"file": p.name, "path": str(p), "composition": None, "error": str(e)})
    return structs, pd.DataFrame(rows)


def to_ase_atoms_list(structs: List[Structure]):
    adaptor = AseAtomsAdaptor()
    atoms_list = []
    for st in structs:
        at = adaptor.get_atoms(st)
        at.set_pbc([True, True, True])
        atoms_list.append(at)
    return atoms_list


# ======================= HAL-only filter =======================
from ase.symbols import atomic_numbers
from ase import Atoms

HALOGENS = {"F", "Cl", "Br", "I"}
HALOGEN_Z = {atomic_numbers[s] for s in HALOGENS}

def build_halogen_only_atoms_list(atoms_list):
    """仅保留卤素位点（保留 cell/pbc），若无卤素则返回空 Atoms。"""
    hal_list = []
    for at in atoms_list:
        idx = [i for i, a in enumerate(at) if int(a.number) in HALOGEN_Z]
        if len(idx) == 0:
            hal_list.append(Atoms(cell=at.cell, pbc=at.pbc))
            continue
        pos = at.get_positions()[idx]
        nums = [int(at[i].number) for i in idx]
        sub = Atoms(numbers=nums, positions=pos, cell=at.cell, pbc=at.pbc)
        hal_list.append(sub)
    return hal_list


# ======================= SOAP (per-atom) =======================
def _make_soap_compat(halogen_Z_sorted, rcut, nmax, lmax, sigma, periodic):
    """DScribe 2.x 使用 r_cut/n_max/l_max；1.x 使用 rcut/nmax/lmax。"""
    try:
        return SOAP(
            species=halogen_Z_sorted,
            r_cut=float(rcut),
            n_max=int(nmax),
            l_max=int(lmax),
            sigma=float(sigma),
            periodic=bool(periodic),
            average="off",
            sparse=False,
        )
    except TypeError:
        return SOAP(
            float(rcut), int(nmax), int(lmax),
            sigma=float(sigma),
            species=halogen_Z_sorted,
            periodic=bool(periodic),
            average="off",
            sparse=False,
        )


def compute_soap_per_atom_list(
    hal_atoms_list,
    halogen_Z_sorted: List[int],
    rcut=7.0, nmax=8, lmax=6, sigma=0.5,
    batch=64, periodic=True
) -> List[np.ndarray]:
    """
    返回 env_list：长度 n_struct，
    env_list[i] = Fi ∈ R^{n_i × d}（第 i 个结构的逐原子 SOAP；n_i 为卤素数）。
    """
    soap = _make_soap_compat(halogen_Z_sorted, rcut, nmax, lmax, sigma, periodic)
    env_list = []
    for i in tqdm(range(0, len(hal_atoms_list), batch), desc="SOAP per-atom (HAL-only)"):
        chunk = hal_atoms_list[i:i+batch]
        feats = soap.create(chunk)  # list of (n_hal_i, d)
        env_list.extend(feats)
    d = soap.get_number_of_features()
    out = []
    for Fi in env_list:
        if Fi.size == 0:
            out.append(np.zeros((0, d), dtype=np.float32))
        else:
            out.append(np.asarray(Fi, dtype=np.float32))
    return out


# ======================= REMatch-Kernel & Distance =======================
def compute_rematch_kernel(env_list: List[np.ndarray], metric="linear", gamma=1.0, alpha=1.0, threshold=1e-6) -> np.ndarray:
    """
    逐行 L2 归一化 -> REMatch 全局相似度核 K (float64)。
    """
    env_norm = [normalize(F, norm="l2") if F.size else F for F in env_list]
    if metric == "rbf":
        re = REMatchKernel(metric="rbf", gamma=float(gamma), alpha=float(alpha), threshold=float(threshold))
    else:
        re = REMatchKernel(metric="linear", alpha=float(alpha), threshold=float(threshold))
    K = re.create(env_norm)
    return np.asarray(K, dtype=np.float64, order="C")


def kernel_to_distance(K: np.ndarray) -> np.ndarray:
    """
    核诱导距离: D_ij = sqrt(max(0, K_ii + K_jj - 2*K_ij))  -> float64 / 对称 / 零对角 / C 连续
    """
    diag = np.clip(np.diag(K), 0.0, None)
    D2 = np.maximum(0.0, diag[:, None] + diag[None, :] - 2.0 * K)
    D = np.sqrt(D2, dtype=np.float64)
    # 数值稳健处理
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return np.ascontiguousarray(D, dtype=np.float64)


# ======================= Prototypes =======================
def pick_prototypes(labels: np.ndarray, topk: int = 5):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"idx": np.arange(len(labels)), "label": labels})
    protos = {}
    for lab, sub in df[df.label >= 0].groupby("label"):
        if len(sub) <= topk:
            protos[int(lab)] = sub["idx"].tolist()
        else:
            protos[int(lab)] = rng.choice(sub["idx"].values, size=topk, replace=False).tolist()
    return protos


# ======================= MAIN =======================
def main():
    cfg = CONFIG
    np.random.seed(cfg["seed"])

    cif_dir = Path(cfg["cif_dir"])
    outdir = Path(cfg["outdir"])
    ensure_dir(outdir)

    cif_paths = list_cifs(cif_dir)
    if not cif_paths:
        raise SystemExit(f"[ERR] No CIF files under: {cif_dir}")

    # 读取结构
    structs, meta = load_structures(cif_paths)
    atoms_list = to_ase_atoms_list(structs)

    # 元素集合 -> 卤素集合
    species_Z_all = sorted({int(site.specie.Z) for st in structs for site in st.sites})
    from ase.data import chemical_symbols
    halogen_Z = sorted([z for z in species_Z_all if chemical_symbols[z] in {"F","Cl","Br","I"}])
    if len(halogen_Z) == 0:
        raise SystemExit("[ERR] No halogen species detected in structures.")

    # 卤素-仅 Atoms
    hal_atoms_list = build_halogen_only_atoms_list(atoms_list)

    # SOAP 参数（兼容键名）
    scfg = cfg["soap"]
    rcut = scfg.get("r_cut", scfg.get("rcut"))
    nmax = scfg.get("n_max", scfg.get("nmax"))
    lmax = scfg.get("l_max", scfg.get("lmax"))
    sigma = scfg["sigma"]

    # 逐原子 SOAP
    env_list = compute_soap_per_atom_list(
        hal_atoms_list, halogen_Z_sorted=halogen_Z,
        rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma,
        batch=scfg["batch"], periodic=scfg["periodic"]
    )

    # REMatch 核 (float64) 与距离矩阵 D (float64, symmetric, zero diag, C-contiguous)
    rcfg = cfg["rematch"]
    K = compute_rematch_kernel(
        env_list,
        metric=rcfg["metric"], gamma=rcfg.get("gamma", 1.0),
        alpha=rcfg["alpha"], threshold=rcfg["threshold"]
    )
    D = kernel_to_distance(K)

    # ===== UMAP（预计算距离：要求方阵距离）=====
    um = cfg["umap"]
    reducer = UMAP(
        metric="precomputed",
        n_neighbors=int(um["n_neighbors"]),
        min_dist=float(um["min_dist"]),
        n_components=int(um["n_components"]),
        random_state=int(um.get("random_state", 42))
    )
    Z = reducer.fit_transform(D)  # 需方阵距离；官方 API 如是规定。  # :contentReference[oaicite:4]{index=4}

    # ===== HDBSCAN（预计算距离；dtype 必须 float64）=====
    hcfg = cfg["hdbscan"]
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(hcfg["min_cluster_size"]),
        min_samples=int(hcfg["min_samples"]),
        cluster_selection_method=str(hcfg["cluster_selection_method"])
    ).fit(D)  # 历史问题：若为 float32 会抛 dtype mismatch。  # :contentReference[oaicite:5]{index=5}
    labels = clusterer.labels_

    # 代表样本（简单随机挑 topk）
    prototypes = pick_prototypes(labels, topk=5)

    # ===== 输出 =====
    outdir.mkdir(parents=True, exist_ok=True)

    # 嵌入
    emb_cols = [f"umap_{i}" for i in range(Z.shape[1])]
    pd.DataFrame(Z, columns=emb_cols).to_csv(outdir / "embedding_umap.csv", index=False)

    # 聚类结果
    meta_out = meta.copy()
    meta_out["cluster_id"] = labels
    meta_out["is_noise"] = (labels < 0).astype(int)
    meta_out.to_csv(outdir / "clusters.csv", index=False)

    # 原型
    with open(outdir / "prototypes.json", "w") as f:
        json.dump({str(k): v for k, v in prototypes.items()}, f, indent=2)

    # 参数与说明
    with open(outdir / "params_used.json", "w") as f:
        dump_cfg = {**cfg}
        dump_cfg["soap"]["species_used"] = halogen_Z
        dump_cfg["backend"] = {"umap": "umap-learn(cpu)", "hdbscan": "hdbscan(cpu)", "distance_metric": "precomputed"}
        dump_cfg["notes"] = {
            "rematch_distance": "D_ij = sqrt(max(0, K_ii + K_jj - 2*K_ij)) (float64)",
            "hdbscan_soft_probabilities": "not available with metric='precomputed'"
        }
        json.dump(dump_cfg, f, indent=2)

    print(f"[OK] Wrote -> {outdir}/embedding_umap.csv, clusters.csv, prototypes.json, params_used.json")


if __name__ == "__main__":
    main()
