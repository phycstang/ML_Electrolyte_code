#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAL-only SOAP unsupervised pipeline (DScribe 1.x/2.x compatible)
CIF -> [SOAP (HAL-only, average='off' -> mean/std/median pooling)]
    -> Standardize -> (GPU) UMAP -> (GPU) HDBSCAN -> CSV/JSON

- 仅在卤素子体系（F/Cl/Br/I）上计算 SOAP；邻域与密度均只包含卤素。
- DScribe 新旧版本参数名自动兼容：r_cut/n_max/l_max ↔ rcut/nmax/lmax。
- UMAP + HDBSCAN（有 cuML 则用 GPU，否则回退 CPU）。
- 输出：clusters.csv, embedding_umap.csv, prototypes.json, params_used.json
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

from dscribe.descriptors import SOAP  # 仅导入，实际构造走 _make_soap_compat
from sklearn.preprocessing import StandardScaler

# ===== Try GPU backends (cuML). Fallback to CPU if not available. =====
USE_CUML = False
try:
    import cupy as cp
    from cuml.manifold import UMAP as CUML_UMAP
    from cuml.cluster import HDBSCAN as CUML_HDBSCAN
    USE_CUML = True
except Exception:
    from umap import UMAP as SKUMAP         # umap-learn
    import hdbscan as SKHDBSCAN             # python-hdbscan

# ======================= CONFIG =======================
CONFIG: Dict[str, object] = {
    "use_gpu": True,  # True 且环境有 cuML 时走 GPU
    "cif_dir": "data/candidates/materials_project/cif",
    "outdir": "results/clustering/soap_halide",

    # SOAP（HAL-only）——支持新旧键名；脚本里会做映射
    "soap": {
        "r_cut": 7.0,      # DScribe 2.x 命名（旧版请用 rcut；两者都可）
        "n_max": 8,
        "l_max": 6,
        # 旧版写法（如需）： "rcut": 7.0, "nmax": 8, "lmax": 6,
        "sigma": 0.5,
        "periodic": True,
        "pooling": ["mean", "std", "median"],
        "batch": 64
    },

    # UMAP（用于嵌入与可视化）
    "umap": {
        "n_neighbors": 45,
        "min_dist": 0.03,      # 越小越“团簇化”，利于密度聚类
        "n_components": 2,
        "metric": "euclidean",
        "random_state": 42
    },

    # HDBSCAN（密度聚类）
    "hdbscan": {
        "min_cluster_size": 80,
        "min_samples": 10,
        "cluster_selection_method": "leaf"  # 可试 "leaf"
    },

    "standardize": True,
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
        at.set_pbc([True, True, True])  # 明确周期性
        atoms_list.append(at)
    return atoms_list


# ======================= HAL-only SOAP =======================

from ase.symbols import atomic_numbers
from ase import Atoms

HALOGENS = {"F", "Cl", "Br", "I"}
HALOGEN_Z = {atomic_numbers[s] for s in HALOGENS}

def build_halogen_only_atoms_list(atoms_list):
    """将每个 Atoms 裁剪为仅含卤素的子体系（保留 cell/pbc）。"""
    hal_list = []
    for at in atoms_list:
        idx = [i for i, a in enumerate(at) if int(a.number) in HALOGEN_Z]
        if len(idx) == 0:
            # 保险起见：返回空结构（后续会转为零向量）
            hal_list.append(Atoms(cell=at.cell, pbc=at.pbc))
            continue
        pos = at.get_positions()[idx]
        nums = [int(at[i].number) for i in idx]
        sub = Atoms(numbers=nums, positions=pos, cell=at.cell, pbc=at.pbc)
        hal_list.append(sub)
    return hal_list


# --- DScribe SOAP 构造的版本兼容封装 ---
def _make_soap_compat(halogen_Z_sorted, rcut, nmax, lmax, sigma, periodic):
    """
    DScribe 2.x: SOAP(species=..., r_cut=..., n_max=..., l_max=..., sigma=..., periodic=..., average='off', sparse=False)
    DScribe 0.x–1.x: SOAP(rcut, nmax, lmax, sigma=..., species=..., periodic=..., average='off', sparse=False)
    """
    try:
        # 新版优先（≥2.x）
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
        # 旧版回退（≤1.x）
        return SOAP(
            float(rcut), int(nmax), int(lmax),
            sigma=float(sigma),
            species=halogen_Z_sorted,
            periodic=bool(periodic),
            average="off",
            sparse=False,
        )


def compute_soap_halogen_only(
    hal_atoms_list,
    halogen_Z_sorted: List[int],
    rcut=7.0, nmax=8, lmax=6, sigma=0.5,
    pooling=("mean","std","median"),
    batch=64, periodic=True
) -> np.ndarray:
    """在卤素子体系上计算 SOAP；返回 [n_struct, len(pooling)*d] 的矩阵。"""
    soap = _make_soap_compat(
        halogen_Z_sorted=halogen_Z_sorted,
        rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma, periodic=periodic
    )
    out = []
    for i in tqdm(range(0, len(hal_atoms_list), batch), desc="SOAP (HAL-only)"):
        chunk = hal_atoms_list[i:i+batch]
        feats_list = soap.create(chunk)  # list of (n_hal_i, d)
        d = soap.get_number_of_features()
        for Fi in feats_list:
            if Fi.size == 0:
                out.append(np.zeros(d * len(pooling), dtype=np.float32))
                continue
            pools = []
            if "mean" in pooling:   pools.append(Fi.mean(axis=0))
            if "std"  in pooling:   pools.append(Fi.std(axis=0))
            if "median" in pooling: pools.append(np.median(Fi, axis=0))
            out.append(np.concatenate(pools, axis=0))
    return np.vstack(out).astype(np.float32, copy=False)


# ======================= UMAP/HDBSCAN =======================

def embed_umap_gpu(X: np.ndarray, umap_cfg: Dict[str, object]) -> Tuple[np.ndarray, object]:
    Xg = cp.asarray(X)
    reducer = CUML_UMAP(
        n_neighbors=int(umap_cfg["n_neighbors"]),
        min_dist=float(umap_cfg["min_dist"]),
        n_components=int(umap_cfg["n_components"]),
        metric=str(umap_cfg["metric"]),
        random_state=int(umap_cfg.get("random_state", 42))
    )
    Zg = reducer.fit_transform(Xg)
    Z = cp.asnumpy(Zg)
    return Z, reducer


def cluster_hdbscan_gpu(Z: np.ndarray, hdb_cfg: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, object]:
    Zg = cp.asarray(Z)
    clusterer = CUML_HDBSCAN(
        min_cluster_size=int(hdb_cfg["min_cluster_size"]),
        min_samples=int(hdb_cfg["min_samples"]),
        cluster_selection_method=str(hdb_cfg["cluster_selection_method"]),
        prediction_data=True
    )
    labels_g = clusterer.fit_predict(Zg)
    labels = cp.asnumpy(labels_g)
    probs = cp.asnumpy(clusterer.probabilities_)
    return labels, probs, clusterer


def embed_umap_cpu(X: np.ndarray, umap_cfg: Dict[str, object]) -> Tuple[np.ndarray, object]:
    reducer = SKUMAP(
        n_neighbors=int(umap_cfg["n_neighbors"]),
        min_dist=float(umap_cfg["min_dist"]),
        n_components=int(umap_cfg["n_components"]),
        metric=str(umap_cfg["metric"]),
        random_state=int(umap_cfg.get("random_state", 42))
    )
    Z = reducer.fit_transform(X)
    return Z, reducer


def cluster_hdbscan_cpu(Z: np.ndarray, hdb_cfg: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, object]:
    clusterer = SKHDBSCAN.HDBSCAN(
        min_cluster_size=int(hdb_cfg["min_cluster_size"]),
        min_samples=int(hdb_cfg["min_samples"]),
        cluster_selection_method=str(hdb_cfg["cluster_selection_method"]),
        prediction_data=True   # 软聚类/概率需要
    )
    labels = clusterer.fit_predict(Z)
    probs = clusterer.probabilities_
    return labels, probs, clusterer


def pick_prototypes(labels: np.ndarray, probs: np.ndarray, topk: int = 5):
    df = pd.DataFrame({"idx": np.arange(len(labels)), "label": labels, "prob": probs})
    protos = {}
    for lab, sub in df[df.label >= 0].groupby("label"):
        top = sub.sort_values("prob", ascending=False).head(topk)["idx"].tolist()
        protos[int(lab)] = top
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

    structs, meta = load_structures(cif_paths)
    atoms_list = to_ase_atoms_list(structs)

    # 元素集合（全局统计），再取卤素集合
    species_Z_all = sorted({int(site.specie.Z) for st in structs for site in st.sites})
    halogen_Z = sorted([z for z in species_Z_all if z in HALOGEN_Z])
    if len(halogen_Z) == 0:
        raise SystemExit("[ERR] No halogen species detected in structures.")

    # 兼容读取 SOAP 参数（新旧键名）
    scfg = cfg["soap"]
    rcut = scfg.get("r_cut", scfg.get("rcut"))
    nmax = scfg.get("n_max", scfg.get("nmax"))
    lmax = scfg.get("l_max", scfg.get("lmax"))
    sigma = scfg["sigma"]

    # ===== SOAP (HAL-only) =====
    hal_atoms_list = build_halogen_only_atoms_list(atoms_list)
    X_soap = compute_soap_halogen_only(
        hal_atoms_list, halogen_Z_sorted=halogen_Z,
        rcut=rcut, nmax=nmax, lmax=lmax, sigma=sigma,
        pooling=tuple(scfg["pooling"]), batch=scfg["batch"], periodic=scfg["periodic"]
    )

    X = X_soap.astype(np.float32, copy=False)

    # ===== 预处理 =====
    if cfg["standardize"]:
        X = StandardScaler().fit_transform(X).astype(np.float32, copy=False)

    # ===== UMAP + HDBSCAN =====
    use_gpu = bool(cfg["use_gpu"]) and USE_CUML
    if use_gpu:
        Z, umap_model = embed_umap_gpu(X, cfg["umap"])
        labels, probs, clusterer = cluster_hdbscan_gpu(Z, cfg["hdbscan"])
        backend = "gpu(cuML)"
    else:
        Z, umap_model = embed_umap_cpu(X, cfg["umap"])
        labels, probs, clusterer = cluster_hdbscan_cpu(Z, cfg["hdbscan"])
        backend = "cpu(umap-learn+hdbscan)"

    prototypes = pick_prototypes(labels, probs, topk=5)

    # ===== 输出 =====
    emb_cols = [f"umap_{i}" for i in range(Z.shape[1])]
    pd.DataFrame(Z, columns=emb_cols).to_csv(outdir / "embedding_umap.csv", index=False)

    meta_out = meta.copy()
    meta_out["cluster_id"] = labels
    meta_out["soft_prob"] = probs
    meta_out["is_noise"] = (labels < 0).astype(int)
    meta_out.to_csv(outdir / "clusters.csv", index=False)

    with open(outdir / "prototypes.json", "w") as f:
        json.dump({str(k): v for k, v in prototypes.items()}, f, indent=2)

    with open(outdir / "params_used.json", "w") as f:
        dump_cfg = {**cfg, "backend_used": backend}
        dump_cfg["features_used"] = ["SOAP_HAL_only(pool: mean/std/median)"]
        dump_cfg["soap"]["species_used"] = halogen_Z
        json.dump(dump_cfg, f, indent=2)

    print(f"[OK] Backend={backend}. Wrote -> {outdir}/clusters.csv, embedding_umap.csv, prototypes.json, params_used.json")


if __name__ == "__main__":
    main()
