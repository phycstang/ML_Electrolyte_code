#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-enabled unsupervised pipeline (HAL/MET aware):
CIF -> [SOAP(PBC, 2-channel HAL|MET) + MBTR(HAL | MET | ALL, sparse) -> safe L2 + per-block TruncatedSVD]
   -> (GPU) UMAP -> (GPU) HDBSCAN -> CSV/JSON

- SOAP: average="off"，在池化阶段按“中心原子是否卤素”分通道统计（mean/std/median 可选）。
- MBTR: 分别对 HAL-only / MET-only / ALL-mix 三个 species 集合计算，再各自 SVD 并拼接。
- 其余流程/配置与原版一致；自动 GPU/CPU 回退。

作者：改进版（卤素/非卤素通道化）
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

from dscribe.descriptors import SOAP, MBTR

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD as SKTruncatedSVD

from scipy import sparse

# ===== Try GPU backends (cuML). Fallback to CPU if not available. =====
USE_CUML = False
try:
    import cupy as cp
    from cuml.manifold import UMAP as CUML_UMAP
    from cuml.cluster import HDBSCAN as CUML_HDBSCAN
    from cuml.decomposition import TruncatedSVD as CUML_TruncatedSVD  # 未使用，但保留以便扩展
    USE_CUML = True
except Exception:
    from umap import UMAP as SKUMAP
    import hdbscan as SKHDBSCAN

# ---- Robust: pydata/sparse or SciPy -> SciPy CSR (no densify) ----
try:
    import sparse as spx  # pydata/sparse
except Exception:
    spx = None


def _to_scipy_csr(X):
    """Convert DScribe/pydata.sparse/NumPy outputs to SciPy CSR without densifying."""
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


# ======================= CONFIG =======================
CONFIG: Dict[str, object] = {
    "use_gpu": True,  # set False to force CPU
    "cif_dir": "data/candidates/materials_project/cif",
    "outdir": "results/clustering/halide_channels",

    # SOAP：双通道池化（卤素中心 | 非卤素中心）
    "soap": {
        "rcut": 7.0, "nmax": 8, "lmax": 6, "sigma": 0.5,
        "pooling": ["mean", "std"],  # 可加入 "median"
        "batch": 64,
        "periodic": True,
    },

    # MBTR：HAL-only / MET-only / ALL-mix 分别 SVD 后拼接
    "mbtr": {
        "k_terms": [1, 2, 3],
        "periodic": True,
        "svd_dims": [128, 128, 256]  # 对应 [HAL, MET, ALL] 的降维
    },

    "umap": {
        "n_neighbors": 45,
        "min_dist": 0.03,
        "n_components": 2,
        "metric": "euclidean",
        "random_state": 42
    },

    "hdbscan": {
        "min_cluster_size": 80,
        "min_samples": 10,
        "cluster_selection_method": "eom"
    },

    "standardize": True,
    "seed": 42
}
# =====================================================


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


def pool_atomic_matrix(M: np.ndarray, modes=("mean", "std", "median")) -> np.ndarray:
    pools = []
    if "mean" in modes:
        pools.append(M.mean(axis=0))
    if "std" in modes:
        pools.append(M.std(axis=0))
    if "median" in modes:
        pools.append(np.median(M, axis=0))
    return np.concatenate(pools, axis=0) if pools else M.mean(axis=0)


# ======================= SOAP (2-channel HAL | MET) =======================

def compute_soap_features(
    atoms_list,
    species_Z: List[int],
    rcut=5.0, nmax=8, lmax=6, sigma=0.5,
    pooling=("mean","std"),
    batch=64, periodic=True
) -> np.ndarray:
    """
    返回形状：[n_struct, 2 * len(pooling) * d] 的特征：
      [HAL-center pooling...,  MET-center pooling...]
    HAL = {F, Cl, Br, I}；MET = 其余元素。
    """
    from ase.symbols import atomic_numbers
    halogens = {"F", "Cl", "Br", "I"}
    halogen_Z = {atomic_numbers[s] for s in halogens}

    soap = SOAP(
        rcut, nmax, lmax, sigma,
        species=species_Z,   # 覆盖结构内全部元素
        periodic=periodic,
        sparse=False,
        average="off"        # 返回每原子的描述子矩阵 (n_atoms_i, d)
    )

    out = []
    for i in tqdm(range(0, len(atoms_list), batch), desc="SOAP (CPU, 2-channel HAL|MET)"):
        chunk = atoms_list[i:i+batch]
        feats_list = soap.create(chunk)  # list of (n_atoms_i, d)
        for at, Fi in zip(chunk, feats_list):
            Zs = np.array([a.number for a in at], dtype=np.int32)
            mask_hal = np.isin(Zs, list(halogen_Z))
            mask_met = ~mask_hal
            d = Fi.shape[1]

            # HAL-center
            if mask_hal.any():
                Fi_hal = Fi[mask_hal]
                pools_hal = []
                if "mean" in pooling:   pools_hal.append(Fi_hal.mean(axis=0))
                if "std"  in pooling:   pools_hal.append(Fi_hal.std(axis=0))
                if "median" in pooling: pools_hal.append(np.median(Fi_hal, axis=0))
                vec_hal = np.concatenate(pools_hal, axis=0) if pools_hal else np.zeros(d)
            else:
                vec_hal = np.zeros(d * len(pooling), dtype=np.float64)

            # MET-center
            if mask_met.any():
                Fi_met = Fi[mask_met]
                pools_met = []
                if "mean" in pooling:   pools_met.append(Fi_met.mean(axis=0))
                if "std"  in pooling:   pools_met.append(Fi_met.std(axis=0))
                if "median" in pooling: pools_met.append(np.median(Fi_met, axis=0))
                vec_met = np.concatenate(pools_met, axis=0) if pools_met else np.zeros(d)
            else:
                vec_met = np.zeros(d * len(pooling), dtype=np.float64)

            out.append(np.concatenate([vec_hal, vec_met], axis=0))

    X_soap = np.vstack(out).astype(np.float32, copy=False)
    return X_soap


# ======================= MBTR (HAL | MET | ALL blocks) =======================

def _safe_l2_normalize_csr(X: sparse.csr_matrix, eps: float = 1e-12) -> sparse.csr_matrix:
    row_norm = np.sqrt(X.power(2).sum(axis=1)).A1
    scale = 1.0 / np.maximum(row_norm, eps)
    scale[row_norm == 0] = 0.0
    return sparse.diags(scale) @ X


def _build_mbtr_for_k(species_list, k, periodic):
    if k == 1:
        return MBTR(
            species=species_list, periodic=periodic,
            geometry={"function": "atomic_number"},
            grid={"min": float(min(species_list)) - 0.5,
                  "max": float(max(species_list)) + 0.5,
                  "sigma": 0.1, "n": int(max(species_list)-min(species_list)+5)},
            weighting={"function": "unity"},
            sparse=True, dtype="float32", normalization="none"
        )
    elif k == 2:
        return MBTR(
            species=species_list, periodic=periodic,
            geometry={"function": "inverse_distance"},
            grid={"min": 0.0, "max": 1.0, "sigma": 0.02, "n": 200},
            weighting={"function": "exp", "scale": 1.0, "threshold": 1e-3},
            sparse=True, dtype="float32", normalization="none"
        )
    elif k == 3:
        return MBTR(
            species=species_list, periodic=periodic,
            geometry={"function": "cosine"},
            grid={"min": -1.0, "max": 1.0, "sigma": 0.05, "n": 100},
            weighting={"function": "exp", "scale": 1.0, "threshold": 1e-3},
            sparse=True, dtype="float32", normalization="none"
        )
    else:
        raise ValueError(f"Unsupported MBTR k={k}; only 1,2,3.")

# 放在 MBTR 代码附近（import 后、compute_mbtr_features_multi 之前）
from ase import Atoms

def _filter_atoms_by_Z_keep_cell(at, allowed_Z):
    """从一个 ASE Atoms 中只保留 Z ∈ allowed_Z 的原子，保留 cell 与 pbc。
    若过滤后为空，返回 None。
    """
    if not allowed_Z:
        return None
    idx = [i for i, a in enumerate(at) if int(a.number) in allowed_Z]
    if len(idx) == 0:
        return None
    pos = at.get_positions()[idx]
    nums = [int(at[i].number) for i in idx]
    sub = Atoms(numbers=nums, positions=pos, cell=at.cell, pbc=at.pbc)
    return sub

def compute_mbtr_features_multi(
    atoms_list,
    species_sets: List[List[int]],
    k_terms=(1, 2, 3),
    periodic=True,
    svd_dims: List[int] = None
) -> np.ndarray:
    """
    对多个 species 集合分别计算 MBTR(+SVD) 并拼接。
    关键修复：对每个结构先按 species 过滤子体系（保留 cell/pbc），
              再调用 MBTR.create()，避免 DScribe 的 species 校验报错。
    若某结构在该分支下无原子 -> 返回全零行（CSR）。
    """
    if svd_dims is None:
        svd_dims = [256] * len(species_sets)
    assert len(species_sets) == len(svd_dims)

    blocks = []
    for sp_list, dim in zip(species_sets, svd_dims):
        # 若该分支 species 为空，直接跳过（也可返回全零块）
        if len(sp_list) == 0:
            continue

        mats = []
        for k in sorted(set(int(k) for k in k_terms)):
            mbtr_k = _build_mbtr_for_k(sp_list, k, periodic)
            nfeat = mbtr_k.get_number_of_features()
            rows = []

            # 逐结构：先过滤出只包含 sp_list 的子体系
            for at in tqdm(atoms_list, desc=f"MBTR{k} [species len={len(sp_list)}]"):
                at_sub = _filter_atoms_by_Z_keep_cell(at, set(sp_list))
                if at_sub is None or len(at_sub) == 0:
                    # 该结构在此分支下无有效原子：返回 1×nfeat 的全零 CSR
                    Xi = sparse.csr_matrix((1, nfeat), dtype=np.float32)
                else:
                    Xi = _to_scipy_csr(mbtr_k.create(at_sub))
                rows.append(Xi)

            Xk = sparse.vstack(rows, format="csr")
            Xk = _safe_l2_normalize_csr(Xk, eps=1e-12)
            mats.append(Xk)

        # 拼接 k=1,2,3
        X_sparse = sparse.hstack(mats, format="csr")

        # SVD 降维到该分支指定维度
        svd = SKTruncatedSVD(n_components=dim, random_state=0)
        X_block = svd.fit_transform(X_sparse).astype(np.float32, copy=False)
        blocks.append(X_block)

    # 所有分支拼接
    return np.hstack(blocks).astype(np.float32, copy=False)


# ======================= UMAP/HDBSCAN (GPU or CPU) =======================

def embed_umap_gpu(X: np.ndarray, umap_cfg: Dict[str, object]) -> Tuple[np.ndarray, object]:
    Xg = cp.asarray(X)
    reducer = CUML_UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        n_components=umap_cfg["n_components"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg.get("random_state", 42)
    )
    Zg = reducer.fit_transform(Xg)
    Z = cp.asnumpy(Zg)
    return Z, reducer


def cluster_hdbscan_gpu(Z: np.ndarray, hdb_cfg: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, object]:
    Zg = cp.asarray(Z)
    clusterer = CUML_HDBSCAN(
        min_cluster_size=hdb_cfg["min_cluster_size"],
        min_samples=hdb_cfg["min_samples"],
        cluster_selection_method=hdb_cfg["cluster_selection_method"],
        prediction_data=True
    )
    labels_g = clusterer.fit_predict(Zg)
    labels = cp.asnumpy(labels_g)
    probs = cp.asnumpy(clusterer.probabilities_)
    return labels, probs, clusterer


def embed_umap_cpu(X: np.ndarray, umap_cfg: Dict[str, object]) -> Tuple[np.ndarray, object]:
    reducer = SKUMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        n_components=umap_cfg["n_components"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg.get("random_state", 42)
    )
    Z = reducer.fit_transform(X)
    return Z, reducer


def cluster_hdbscan_cpu(Z: np.ndarray, hdb_cfg: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, object]:
    clusterer = SKHDBSCAN.HDBSCAN(
        min_cluster_size=hdb_cfg["min_cluster_size"],
        min_samples=hdb_cfg["min_samples"],
        cluster_selection_method=hdb_cfg["cluster_selection_method"],
        prediction_data=True
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

    # 元素集合
    species_Z = sorted({int(site.specie.Z) for st in structs for site in st.sites})
    from ase.symbols import atomic_numbers
    halogens = {"F", "Cl", "Br", "I"}
    halogen_Z = sorted({atomic_numbers[s] for s in halogens})
    metal_Z   = sorted(set(species_Z) - set(halogen_Z))

    # ===== SOAP (2-channel) =====
    soap_cfg = cfg["soap"]
    X_soap = compute_soap_features(
        atoms_list, species_Z,
        rcut=soap_cfg["rcut"], nmax=soap_cfg["nmax"], lmax=soap_cfg["lmax"], sigma=soap_cfg["sigma"],
        pooling=tuple(soap_cfg["pooling"]), batch=soap_cfg["batch"], periodic=soap_cfg["periodic"]
    )

    # ===== MBTR (HAL | MET | ALL) =====
    mbtr_cfg = cfg["mbtr"]
    species_sets = [halogen_Z, metal_Z, species_Z]
    svd_dims = list(mbtr_cfg.get("svd_dims", [128, 128, 256]))
    X_mbtr = compute_mbtr_features_multi(
        atoms_list,
        species_sets=species_sets,
        k_terms=tuple(mbtr_cfg["k_terms"]),
        periodic=mbtr_cfg["periodic"],
        svd_dims=svd_dims
    )

    # ===== 拼接与预处理 =====
    X = np.hstack([X_soap, X_mbtr]).astype(np.float32, copy=False)

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
        backend = "cpu(sklearn+hdbscan)"

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
        dump_cfg["mbtr"]["species_blocks"] = {"HAL": halogen_Z, "MET": metal_Z, "ALL": species_Z}
        json.dump(dump_cfg, f, indent=2)

    print(f"[OK] Backend={backend}. Wrote -> {outdir}/clusters.csv, embedding_umap.csv, prototypes.json, params_used.json")


if __name__ == "__main__":
    main()
