#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-enabled semi-supervised pipeline:
CIF -> [SOAP(PBC) + MBTR(PBC, sparse) -> safe L2 + TruncatedSVD]
   -> (GPU/CPU) UMAP (semi-supervised) -> (GPU/CPU) HDBSCAN
   -> CSV/JSON (clusters, embedding, prototypes, seed-neighbors)

- 特征：DScribe 的 SOAP/MBTR（CPU 计算），MBTR 用 CSR + 安全 L2 + SVD 降维。
- 嵌入：UMAP 支持半监督（y 标签；target_metric/target_weight），优先 cuML，必要时回退到 umap-learn。  [umap supervised, API]  # :contentReference[oaicite:1]{index=1}
- 聚类：HDBSCAN（软聚类概率，支持 approximate_predict 上线新样本），优先 cuML，必要时回退到 hdbscan。  # :contentReference[oaicite:2]{index=2}
- GPU：RAPIDS cuML UMAP/HDBSCAN 后端（sklearn 风格），不可用时自动回退。  # :contentReference[oaicite:3]{index=3}
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
from ase import Atoms

from dscribe.descriptors import SOAP, MBTR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD as SKTruncatedSVD
from scipy import sparse

# ======== Try GPU backends (cuML); fallback to CPU (umap-learn + hdbscan) ========
USE_CUML = False
try:
    import cupy as cp
    from cuml.manifold import UMAP as CUML_UMAP
    from cuml.cluster import HDBSCAN as CUML_HDBSCAN
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
    "cif_dir": "/data/home/tmy/ML_Electrolyte3/new_material/material_P/cif",
    "outdir": "out/run_semi",

    # ---- 半监督：指定 5 个种子材料（按 file/path/composition 三选一） ----
    "seeds": {
        "by": "file",  # "file" | "path" | "composition"
        "items": ["AlCl3_mp-25469.cif", "TaCl5_mp-29831.cif", "ZrCl4_mp-569175.cif", "HfCl4_mp-29422.cif", "ZnCl2_mp-22889.cif"]
    },
    "semi": {
        "enable": True,
        "target_metric": "categorical",  # umap supervised
        "target_weight": 0.5             # 0.2~0.7 可网格搜索  # :contentReference[oaicite:4]{index=4}
    },
    "retrieval": {         # 基于嵌入的相似检索（以5个种子的质心为查询）
        "topk": 100,
        "use_centroid": True
    },

    "soap": {
        "rcut": 5.5, "nmax": 8, "lmax": 6, "sigma": 0.5,
        "pooling": ["mean", "std"],
        "batch": 64,
        "periodic": True,
    },

    # MBTR with sparse output + safe L2 + SVD reduction (dim set below)
    "mbtr": {
        "k_terms": [1, 2, 3],
        "periodic": True,
        "svd_dim": 256
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
        "cluster_selection_method": "eom",
        "prediction_data": True  # 便于 approximate_predict  # :contentReference[oaicite:5]{index=5}
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
    if "mean" in modes:   pools.append(M.mean(axis=0))
    if "std" in modes:    pools.append(M.std(axis=0))
    if "median" in modes: pools.append(np.median(M, axis=0))
    return np.concatenate(pools, axis=0) if pools else M.mean(axis=0)


# ======================= SOAP =======================
def compute_soap_features(
    atoms_list,
    species_Z: List[int],
    rcut=5.0, nmax=8, lmax=6, sigma=0.5,
    pooling=("mean","std"),
    batch=64, periodic=True
) -> np.ndarray:
    soap = SOAP(
        rcut, nmax, lmax, sigma,
        species=species_Z,
        periodic=periodic,
        sparse=False,
        average="off"
    )
    out = []
    for i in tqdm(range(0, len(atoms_list), batch), desc="SOAP (CPU)"):
        chunk = atoms_list[i:i+batch]
        feats = soap.create(chunk)  # list of (n_atoms_i, d)
        for Fi in feats:
            out.append(pool_atomic_matrix(Fi, modes=pooling))
    X_soap = np.vstack(out).astype(np.float32, copy=False)
    return X_soap


# ======================= MBTR (sparse, safe L2, SVD) =======================
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


def compute_mbtr_features(
    atoms_list, species_Z: List[int], k_terms=(1,2,3), periodic=True, svd_dim=256
) -> np.ndarray:
    species_list = list(species_Z)
    mats = []
    for k in sorted(set(int(k) for k in k_terms)):
        mbtr_k = _build_mbtr_for_k(species_list, k, periodic)
        rows = [_to_scipy_csr(mbtr_k.create(at)) for at in tqdm(atoms_list, desc=f"MBTR{k} (CPU, sparse)")]
        Xk = sparse.vstack(rows, format="csr")
        Xk = _safe_l2_normalize_csr(Xk, eps=1e-12)
        mats.append(Xk)
    X_sparse = sparse.hstack(mats, format="csr")
    svd = SKTruncatedSVD(n_components=svd_dim, random_state=0)
    X_mbtr = svd.fit_transform(X_sparse).astype(np.float32, copy=False)
    return X_mbtr


# ======================= 半监督标签 & 相似检索 =======================
def build_semi_supervised_labels(meta: pd.DataFrame, seeds_cfg: Dict[str, object]) -> np.ndarray:
    """5 个种子 -> 标签 1；其它 -> -1（未标注）"""
    key = seeds_cfg.get("by", "file")
    items = set(seeds_cfg.get("items", []))
    col = {"file": "file", "path": "path", "composition": "composition"}[key]
    y = np.full(len(meta), fill_value=-1, dtype=np.int32)
    mask = meta[col].isin(items)
    y[mask.values] = 1
    return y


def topk_neighbors_from_seeds(Z: np.ndarray, meta: pd.DataFrame, seeds_cfg: Dict[str, object], K: int = 50):
    key = seeds_cfg.get("by", "file")
    items = set(seeds_cfg.get("items", []))
    col = {"file": "file", "path": "path", "composition": "composition"}[key]
    seed_idx = np.flatnonzero(meta[col].isin(items).values)
    if len(seed_idx) == 0:
        return pd.DataFrame(columns=list(meta.columns)+["dist_to_seed_centroid"])
    centroid = Z[seed_idx].mean(axis=0, keepdims=True)
    d = np.linalg.norm(Z - centroid, axis=1)
    order = np.argsort(d)
    top = order[:int(K)]
    out = meta.iloc[top].copy()
    out["dist_to_seed_centroid"] = d[top]
    return out.sort_values("dist_to_seed_centroid")


# ======================= UMAP/HDBSCAN (GPU/CPU; 半监督优先) =======================
def embed_umap_gpu_semi(X: np.ndarray, y: np.ndarray, umap_cfg: Dict[str, object], semi_cfg: Dict[str, object]):
    # 注：不同版本 cuML 对 supervised 选项支持程度不同；若报错在外层捕获并回退到 CPU。  # :contentReference[oaicite:6]{index=6}
    reducer = CUML_UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        n_components=umap_cfg["n_components"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg.get("random_state", 42),
        target_metric=semi_cfg.get("target_metric", "categorical"),
        target_weight=float(semi_cfg.get("target_weight", 0.5))
    )
    Xg, yg = cp.asarray(X), cp.asarray(y)
    Zg = reducer.fit_transform(Xg, yg)
    return cp.asnumpy(Zg), reducer


def embed_umap_cpu_semi(X: np.ndarray, y: np.ndarray, umap_cfg: Dict[str, object], semi_cfg: Dict[str, object]):
    reducer = SKUMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        n_components=umap_cfg["n_components"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg.get("random_state", 42),
        target_metric=semi_cfg.get("target_metric", "categorical"),
        target_weight=float(semi_cfg.get("target_weight", 0.5))
    )
    Z = reducer.fit_transform(X, y=y)  # umap-learn 半监督入口  # :contentReference[oaicite:7]{index=7}
    return Z, reducer


def embed_umap_gpu(X: np.ndarray, umap_cfg: Dict[str, object]):
    Xg = cp.asarray(X)
    reducer = CUML_UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        n_components=umap_cfg["n_components"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg.get("random_state", 42)
    )
    Zg = reducer.fit_transform(Xg)
    return cp.asnumpy(Zg), reducer


def embed_umap_cpu(X: np.ndarray, umap_cfg: Dict[str, object]):
    reducer = SKUMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        n_components=umap_cfg["n_components"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg.get("random_state", 42)
    )
    Z = reducer.fit_transform(X)
    return Z, reducer


def cluster_hdbscan_gpu(Z: np.ndarray, hdb_cfg: Dict[str, object]):
    Zg = cp.asarray(Z)
    clusterer = CUML_HDBSCAN(
        min_cluster_size=hdb_cfg["min_cluster_size"],
        min_samples=hdb_cfg["min_samples"],
        cluster_selection_method=hdb_cfg["cluster_selection_method"],
        prediction_data=hdb_cfg.get("prediction_data", True)
    )
    labels_g = clusterer.fit_predict(Zg)
    labels = cp.asnumpy(labels_g)
    # cuML HDBSCAN 提供软概率（版本依赖）；如不可用，此处会抛错或返回属性缺失
    try:
        probs = cp.asnumpy(clusterer.probabilities_)
    except Exception:
        probs = np.ones_like(labels, dtype=np.float32)
    return labels, probs, clusterer


def cluster_hdbscan_cpu(Z: np.ndarray, hdb_cfg: Dict[str, object]):
    clusterer = SKHDBSCAN.HDBSCAN(
        min_cluster_size=hdb_cfg["min_cluster_size"],
        min_samples=hdb_cfg["min_samples"],
        cluster_selection_method=hdb_cfg["cluster_selection_method"],
        prediction_data=hdb_cfg.get("prediction_data", True)
    )
    labels = clusterer.fit_predict(Z)
    probs = clusterer.probabilities_  # 软聚类概率  # :contentReference[oaicite:8]{index=8}
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
    species_Z = sorted({int(site.specie.Z) for st in structs for site in st.sites})

    # ---- Features (CPU) ----
    soap_cfg = cfg["soap"]
    X_soap = compute_soap_features(
        atoms_list, species_Z,
        rcut=soap_cfg["rcut"], nmax=soap_cfg["nmax"], lmax=soap_cfg["lmax"], sigma=soap_cfg["sigma"],
        pooling=tuple(soap_cfg["pooling"]), batch=soap_cfg["batch"], periodic=soap_cfg["periodic"]
    )
    mbtr_cfg = cfg["mbtr"]
    X_mbtr = compute_mbtr_features(
        atoms_list, species_Z,
        k_terms=tuple(mbtr_cfg["k_terms"]), periodic=mbtr_cfg["periodic"],
        svd_dim=mbtr_cfg["svd_dim"]
    )
    X = np.hstack([X_soap, X_mbtr]).astype(np.float32, copy=False)

    if cfg["standardize"]:
        X = StandardScaler().fit_transform(X).astype(np.float32, copy=False)

    # ---- Build semi-supervised labels y ----
    if cfg.get("semi", {}).get("enable", False):
        y = build_semi_supervised_labels(meta, cfg["seeds"])
    else:
        y = np.full(len(meta), -1, dtype=np.int32)

    # ---- UMAP (semi first) ----
    use_gpu = bool(cfg["use_gpu"]) and USE_CUML
    semi_cfg = cfg.get("semi", {"enable": False})

    if semi_cfg.get("enable", False):
        try:
            if use_gpu:
                Z, umap_model = embed_umap_gpu_semi(X, y, cfg["umap"], semi_cfg)
                backend_umap = "gpu(cuML, semi)"
            else:
                Z, umap_model = embed_umap_cpu_semi(X, y, cfg["umap"], semi_cfg)
                backend_umap = "cpu(umap-learn, semi)"
        except Exception as e:
            # cuML 若不支持 supervised UMAP，则回退到 CPU umap-learn 半监督
            Z, umap_model = embed_umap_cpu_semi(X, y, cfg["umap"], semi_cfg)
            backend_umap = f"cpu(umap-learn, semi; gpu-fallback due to {type(e).__name__})"
    else:
        if use_gpu:
            Z, umap_model = embed_umap_gpu(X, cfg["umap"])
            backend_umap = "gpu(cuML)"
        else:
            Z, umap_model = embed_umap_cpu(X, cfg["umap"])
            backend_umap = "cpu(umap-learn)"

    # ---- HDBSCAN ----
    if use_gpu and backend_umap.startswith("gpu"):
        labels, probs, clusterer = cluster_hdbscan_gpu(Z, cfg["hdbscan"])
        backend_cluster = "gpu(cuML)"
    else:
        labels, probs, clusterer = cluster_hdbscan_cpu(Z, cfg["hdbscan"])
        backend_cluster = "cpu(hdbscan)"

    prototypes = pick_prototypes(labels, probs, topk=5)

    # ---- Retrieval: Top-K neighbors from seed centroid ----
    topk = int(cfg.get("retrieval", {}).get("topk", 100))
    nn_df = topk_neighbors_from_seeds(Z, meta, cfg["seeds"], K=topk)

    # ---- Outputs ----
    ensure_dir(outdir)
    meta_out = meta.copy()
    meta_out["cluster_id"] = labels
    meta_out["soft_prob"] = probs
    meta_out["is_noise"] = (labels < 0).astype(int)
    meta_out.to_csv(outdir / "clusters.csv", index=False)

    emb_cols = [f"umap_{i}" for i in range(Z.shape[1])]
    pd.DataFrame(Z, columns=emb_cols).to_csv(outdir / "embedding_umap.csv", index=False)

    nn_df.to_csv(outdir / "neighbors_from_seeds.csv", index=False)

    with open(outdir / "prototypes.json", "w") as f:
        json.dump({str(k): v for k, v in prototypes.items()}, f, indent=2)

    with open(outdir / "params_used.json", "w") as f:
        dump_cfg = {**cfg, "backend_used": {"umap": backend_umap, "cluster": backend_cluster}}
        json.dump(dump_cfg, f, indent=2)

    print(f"[OK] UMAP={backend_umap} | HDBSCAN={backend_cluster}")
    print(f"[OK] Wrote -> {outdir}/clusters.csv, embedding_umap.csv, neighbors_from_seeds.csv, prototypes.json, params_used.json")


if __name__ == "__main__":
    main()
