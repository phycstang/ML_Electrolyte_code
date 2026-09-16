#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAL-only MBTR unsupervised pipeline (GPU fallback ready)
CIF -> [MBTR (HAL-only, sparse) -> safe L2 -> TruncatedSVD]
    -> (GPU) UMAP -> (GPU) HDBSCAN -> CSV/JSON

- 仅对卤素子体系（F/Cl/Br/I）计算 MBTR(k=1,2,3)。
- 稀疏特征逐块 L2 标准化 -> TruncatedSVD 降维。
- UMAP + HDBSCAN（cuML 有则用 GPU，否则 CPU 回退）。
- 输出：clusters.csv, embedding_umap.csv, params_used.json, prototypes.json

Refs:
- DScribe MBTR (periodic, weighting/geometry/grid) docs
- UMAP parameters (n_neighbors/min_dist) docs
- HDBSCAN soft clustering & prediction_data usage
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

from dscribe.descriptors import MBTR
from scipy import sparse

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD as SKTruncatedSVD

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

# Optional: pydata/sparse 支持（不是必须）
try:
    import sparse as spx
except Exception:
    spx = None


# ======================= CONFIG =======================
CONFIG: Dict[str, object] = {
    "use_gpu": True,  # True 且环境有 cuML 时走 GPU
    "cif_dir": "data/candidates/materials_project/cif",
    "outdir": "results/clustering/mbtr_halide",

    # MBTR（仅卤素）
    "mbtr": {
        "k_terms": [1, 2, 3],
        "periodic": True,
        # 单块降维目标维数（HAL-only）
        "svd_dim": 256,
        # 网格/权重参数可按需调整（此处选用常见设置）
        "grid_k1": {"sigma": 0.10, "pad": 2.0},
        "grid_k2": {"min": 0.0, "max": 1.0, "sigma": 0.02, "n": 200},
        "grid_k3": {"min": -1.0, "max": 1.0, "sigma": 0.05, "n": 100},
        "weight_k2": {"function": "exp", "scale": 1.0, "threshold": 1e-3},
        "weight_k3": {"function": "exp", "scale": 1.0, "threshold": 1e-3},
    },

    # UMAP（用于嵌入与可视化）
    "umap": {
        "n_neighbors": 45,
        "min_dist": 0.03,      # 越小越“团簇化”，利于聚类可分性
        "n_components": 2,
        "metric": "euclidean",
        "random_state": 42
    },

    # HDBSCAN（密度聚类）
    "hdbscan": {
        "min_cluster_size": 80,
        "min_samples": 10,
        "cluster_selection_method": "eom"  # "leaf" 也可尝试
    },

    "standardize": True,  # 对 SVD 后的连续特征做标准化
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


# ---- pydata.sparse/DScribe 输出 -> SciPy CSR（避免 densify） ----
def _to_scipy_csr(X):
    import numpy as _np
    from scipy import sparse as _sps

    if _sps.issparse(X):
        return X.tocsr()

    is_pydata = False
    if spx is not None:
        try:
            is_pydata = (not _sps.issparse(X)) and (
                (hasattr(X, "coords") and hasattr(X, "data") and hasattr(X, "shape")) or
                (getattr(X, "__class__", None) and getattr(X.__class__, "__module__", "").startswith("sparse"))
            )
        except Exception:
            is_pydata = False

    if is_pydata:
        coo = spx.as_coo(X)
        if len(coo.shape) == 1:
            idx = coo.coords[0].astype(_np.int64, copy=False)
            data = coo.data
            n = int(coo.shape[0])
            return _sps.coo_matrix((data, (_np.zeros_like(idx), idx)), shape=(1, n)).tocsr()
        if len(coo.shape) == 2:
            r = coo.coords[0].astype(_np.int64, copy=False)
            c = coo.coords[1].astype(_np.int64, copy=False)
            return _sps.coo_matrix((coo.data, (r, c)), shape=(int(coo.shape[0]), int(coo.shape[1]))).tocsr()
        flat = spx.as_coo(coo.reshape((-1,)))
        idx = flat.coords[0].astype(_np.int64, copy=False)
        data = flat.data
        n = int(flat.shape[0])
        return _sps.coo_matrix((data, (_np.zeros_like(idx), idx)), shape=(1, n)).tocsr()

    arr = _np.asarray(X)
    if arr.ndim == 1:
        return _sps.csr_matrix(arr.reshape(1, -1))
    return _sps.csr_matrix(arr)


def _safe_l2_normalize_csr(X: sparse.csr_matrix, eps: float = 1e-12) -> sparse.csr_matrix:
    row_norm = np.sqrt(X.power(2).sum(axis=1)).A1
    scale = 1.0 / np.maximum(row_norm, eps)
    scale[row_norm == 0] = 0.0
    return sparse.diags(scale) @ X


# ======================= HAL-only MBTR =======================

from ase.symbols import atomic_numbers
from ase import Atoms

HALOGENS = {"F", "Cl", "Br", "I"}
HALOGEN_Z = {atomic_numbers[s] for s in HALOGENS}


def _filter_atoms_by_Z_keep_cell(at, allowed_Z):
    """只保留 Z∈allowed_Z 的原子，保留 cell 与 pbc；若为空返回 None。"""
    if not allowed_Z:
        return None
    idx = [i for i, a in enumerate(at) if int(a.number) in allowed_Z]
    if len(idx) == 0:
        return None
    pos = at.get_positions()[idx]
    nums = [int(at[i].number) for i in idx]
    sub = Atoms(numbers=nums, positions=pos, cell=at.cell, pbc=at.pbc)
    return sub


def _build_mbtr_for_k(species_list, k, cfg) -> MBTR:
    """根据 k=1/2/3 返回 MBTR 实例（sparse, no normalization）。"""
    if k == 1:
        # k=1: atomic number 分布（网格长度随元素范围自适应）
        zmin, zmax = float(min(species_list)), float(max(species_list))
        pad = float(cfg["grid_k1"]["pad"])
        return MBTR(
            species=species_list, periodic=cfg["periodic"],
            geometry={"function": "atomic_number"},
            grid={"min": zmin - pad, "max": zmax + pad, "sigma": float(cfg["grid_k1"]["sigma"]),
                  "n": int((zmax - zmin) + 2*pad + 5)},
            weighting={"function": "unity"},
            sparse=True, dtype="float32", normalization="none"
        )
    elif k == 2:
        return MBTR(
            species=species_list, periodic=cfg["periodic"],
            geometry={"function": "inverse_distance"},
            grid={"min": float(cfg["grid_k2"]["min"]), "max": float(cfg["grid_k2"]["max"]),
                  "sigma": float(cfg["grid_k2"]["sigma"]), "n": int(cfg["grid_k2"]["n"])},
            weighting=cfg["weight_k2"],
            sparse=True, dtype="float32", normalization="none"
        )
    elif k == 3:
        return MBTR(
            species=species_list, periodic=cfg["periodic"],
            geometry={"function": "cosine"},
            grid={"min": float(cfg["grid_k3"]["min"]), "max": float(cfg["grid_k3"]["max"]),
                  "sigma": float(cfg["grid_k3"]["sigma"]), "n": int(cfg["grid_k3"]["n"])},
            weighting=cfg["weight_k3"],
            sparse=True, dtype="float32", normalization="none"
        )
    else:
        raise ValueError(f"Unsupported MBTR k={k}; only 1,2,3.")


def compute_mbtr_hal_only(
    atoms_list,
    halogen_Z_sorted: List[int],
    k_terms=(1, 2, 3),
    periodic=True,
    svd_dim: int = 256,
    mbtr_cfg: Dict[str, object] = None
) -> np.ndarray:
    """
    在卤素子体系上计算 MBTR(k=1/2/3)，各 k 拼接 -> L2 标准化 -> TruncatedSVD(svd_dim)。
    返回 [n_struct, svd_dim] 的 float32 矩阵。
    """
    mats = []
    for k in sorted(set(int(k) for k in k_terms)):
        mbtr_k = _build_mbtr_for_k(halogen_Z_sorted, k, mbtr_cfg)
        nfeat = mbtr_k.get_number_of_features()
        rows = []

        for at in tqdm(atoms_list, desc=f"MBTR{k} (HAL-only)"):
            at_sub = _filter_atoms_by_Z_keep_cell(at, set(halogen_Z_sorted))
            if at_sub is None or len(at_sub) == 0:
                Xi = sparse.csr_matrix((1, nfeat), dtype=np.float32)  # 空 -> 零行
            else:
                Xi = _to_scipy_csr(mbtr_k.create(at_sub))
            rows.append(Xi)

        Xk = sparse.vstack(rows, format="csr")
        Xk = _safe_l2_normalize_csr(Xk, eps=1e-12)
        mats.append(Xk)

    X_sparse = sparse.hstack(mats, format="csr")

    svd = SKTruncatedSVD(n_components=int(svd_dim), random_state=0)
    X_block = svd.fit_transform(X_sparse).astype(np.float32, copy=False)
    return X_block


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
    probs = cp.asnumpy(clusterer.probabilities_)  # cuML 暴露 probabilities_
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
        prediction_data=True   # 开启软聚类概率/后验预测支持
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

    # ===== MBTR (HAL-only) =====
    mbtr_cfg = cfg["mbtr"]
    X_mbtr = compute_mbtr_hal_only(
        atoms_list,
        halogen_Z_sorted=halogen_Z,
        k_terms=tuple(mbtr_cfg["k_terms"]),
        periodic=bool(mbtr_cfg["periodic"]),
        svd_dim=int(mbtr_cfg["svd_dim"]),
        mbtr_cfg=mbtr_cfg
    )

    X = X_mbtr.astype(np.float32, copy=False)

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
    meta_out = meta.copy()
    meta_out["cluster_id"] = labels
    meta_out["soft_prob"] = probs
    meta_out["is_noise"] = (labels < 0).astype(int)
    meta_out.to_csv(outdir / "clusters.csv", index=False)

    emb_cols = [f"umap_{i}" for i in range(Z.shape[1])]
    pd.DataFrame(Z, columns=emb_cols).to_csv(outdir / "embedding_umap.csv", index=False)

    with open(outdir / "prototypes.json", "w") as f:
        json.dump({str(k): v for k, v in prototypes.items()}, f, indent=2)

    with open(outdir / "params_used.json", "w") as f:
        dump_cfg = {**cfg, "backend_used": backend}
        dump_cfg["mbtr"]["species_blocks"] = {"HAL": halogen_Z}
        dump_cfg["features_used"] = ["MBTR_HAL_only"]
        json.dump(dump_cfg, f, indent=2)

    print(f"[OK] Backend={backend}. Wrote -> {outdir}/clusters.csv, embedding_umap.csv, prototypes.json, params_used.json")


if __name__ == "__main__":
    main()
