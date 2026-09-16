#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-enabled unsupervised pipeline:
CIF -> [SOAP(PBC) + MBTR(PBC, sparse) -> safe L2 + TruncatedSVD] -> (GPU) UMAP -> (GPU) HDBSCAN -> CSV/JSON
+ Grid search w/ Trustworthiness (UMAP) + DBCV (HDBSCAN) for automatic model selection.

- Features (SOAP/MBTR) computed with DScribe (CPU).
- Dimensionality reduction & clustering with RAPIDS cuML (GPU); auto-fallback to CPU if cuML is unavailable.

Refs (conceptual):
- UMAP parameters & effects (umap-learn docs).
- HDBSCAN parameters, eom vs leaf, validity_index (DBCV).
- Trustworthiness metric in scikit-learn / RAPIDS cuML.
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
_CUML_IMPORT_ERR = None
try:
    import cupy as cp
    from cuml.manifold import UMAP as CUML_UMAP
    from cuml.cluster import HDBSCAN as CUML_HDBSCAN
    USE_CUML = True
except Exception as _e:
    _CUML_IMPORT_ERR = repr(_e)
    # CPU fallbacks
    from umap import UMAP as SKUMAP
    import hdbscan as SKHDBSCAN

# ---- Robust: pydata/sparse or SciPy -> SciPy CSR (no densify) ----
try:
    import sparse as spx  # pydata/sparse
except Exception:
    spx = None

def _to_scipy_csr(X):
    """Convert DScribe/pydata.sparse/NumPy outputs to SciPy CSR without densifying.
    Handles:
      - SciPy sparse matrices (any format) -> CSR
      - pydata/sparse 1D/2D/>2D -> CSR (1xN for 1D)
      - Dense numpy arrays -> CSR (row vector if 1D)
    """
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
        coo = spx.as_coo(X)  # no densify
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
    "outdir": "results/clustering/search_soap_mbtr",

    "soap": {
        "rcut": 6.0, "nmax": 8, "lmax": 6, "sigma": 0.5,
        "pooling": ["mean", "std"],
        "batch": 64,
        "periodic": True,
    },

    # MBTR with sparse output + safe L2 + SVD reduction (dim set below)
    "mbtr": {
        "k_terms": [1, 2, 3],
        "periodic": True,
        "svd_dim": 256  # reduce sparse MBTR to 256 dims (then concat with SOAP)
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
    "seed": 42,

    # ====== Parameter search config ======
    "param_search": {
        "enable": True,           # set False to skip grid search and run single pass
        "umap_n_components": [2], # for clustering stability try [5,10,15] too
        "umap_n_neighbors": [15, 30, 45, 60, 90],
        "umap_min_dist": [0.03, 0.1, 0.2, 0.3],
        "umap_metric": ["euclidean", "cosine"],

        "hdb_min_cluster_size": [30, 50, 80, 120],
        "hdb_min_samples_mode": ["equal", "half", "sqrt"],  # =S, floor(S/2), floor(sqrt(S))
        "hdb_method": ["eom", "leaf"],

        "trust_k": 15,            # neighbors for trustworthiness
        "max_combos": None,       # e.g. 120 to cap time; None = full grid
        "random_state": 42
    }
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


def pool_atomic_matrix(M: np.ndarray, modes=("mean", "std")) -> np.ndarray:
    pools = []
    if "mean" in modes:
        pools.append(M.mean(axis=0))
    if "std" in modes:
        pools.append(M.std(axis=0))
    if "median" in modes:
        pools.append(np.median(M, axis=0))
    return np.concatenate(pools, axis=0)


def compute_soap_features(
    atoms_list,
    species_Z: List[int],
    rcut=5.0, nmax=8, lmax=6, sigma=0.5,
    pooling=("mean","std"),
    batch=64, periodic=True
) -> np.ndarray:
    # DScribe SOAP: use positionals for rcut/nmax/lmax/sigma; average="off"
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


# ---------- MBTR (sparse, safe L2, SVD) ----------

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


# ===== UMAP/HDBSCAN: GPU & CPU variants =====

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


# ===== Metrics: Trustworthiness (CPU/GPU) & DBCV =====
def compute_trustworthiness(X: np.ndarray, Z: np.ndarray, k: int = 15, metric: str = "euclidean", use_gpu: bool = False) -> float:
    """
    Prefer GPU metric if available; fall back to sklearn.manifold.trustworthiness.
    Trustworthiness ∈ [0,1], higher is better.
    """
    if use_gpu:
        try:
            # RAPIDS >= 25.x
            try:
                from cuml.metrics import trustworthiness as _tw_func  # function module
                return float(_tw_func(X, Z, n_neighbors=k))
            except Exception:
                # some versions expose trustworthiness_score
                try:
                    from cuml.metrics import trustworthiness_score as _tw_score
                    import cupy as cp
                    Xg = cp.asarray(X)
                    Zg = cp.asarray(Z)
                    return float(_tw_score(Xg, Zg, n_neighbors=k))
                except Exception:
                    pass
        except Exception:
            pass

    # CPU fallback (supports metric selection)
    from sklearn.manifold import trustworthiness as sk_trust
    return float(sk_trust(X, Z, n_neighbors=k, metric=metric))

def compute_dbcv(Z: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """
    Density-Based Clustering Validation (DBCV), via hdbscan.validity.validity_index
    返回 [-1,1]，越大越好；全噪声/单簇等极端情形返回 -1.
    """
    try:
        from hdbscan.validity import validity_index
        val = float(validity_index(Z, labels, metric=metric))
    except Exception:
        val = -1.0
    return val


# ===== Grid search: UMAP × HDBSCAN with Trustworthiness & DBCV =====
def _expand_min_samples(s: int, mode: str) -> int:
    if mode == "equal":
        return int(s)
    if mode == "half":
        return max(1, int(s // 2))
    if mode == "sqrt":
        return max(1, int(np.sqrt(max(1, s))))
    raise ValueError(mode)

def run_param_search(X: np.ndarray, cfg: Dict[str, object], use_gpu: bool, outdir: Path) -> Dict[str, object]:
    """
    外层枚举 UMAP 组合 -> 计算一次 Z + trust
    内层枚举 HDBSCAN 组合 -> 计算 labels/probs + DBCV
    排序：DBCV 降序，其次 Trust 降序
    """
    ps = cfg["param_search"]
    rng = np.random.default_rng(ps.get("random_state", 42))

    umap_grid = []
    for nc in ps["umap_n_components"]:
        for nn in ps["umap_n_neighbors"]:
            for md in ps["umap_min_dist"]:
                for met in ps["umap_metric"]:
                    umap_grid.append((nc, nn, md, met))

    if ps.get("max_combos"):
        # 简单限速：随机抽样 UMAP 组合
        umap_grid = [umap_grid[i] for i in rng.choice(len(umap_grid), size=min(ps["max_combos"], len(umap_grid)), replace=False)]

    hdb_grid = []
    for s in ps["hdb_min_cluster_size"]:
        for ms_mode in ps["hdb_min_samples_mode"]:
            for sel in ps["hdb_method"]:
                hdb_grid.append((s, ms_mode, sel))

    rows, best = [], None  # (dbcv, trust, params, labels, probs, Z)
    print(f"[INFO] UMAP combos: {len(umap_grid)}  ×  HDBSCAN combos: {len(hdb_grid)}")

    for (nc, nn, md, met) in tqdm(umap_grid, desc="UMAP stage"):
        # —— UMAP 配置（CPU 搜索时，取消 random_state 以启用多核）
        um_cfg = dict(
            n_neighbors=nn, min_dist=md, n_components=nc, metric=met,
            random_state=(cfg["umap"].get("random_state", 42) if use_gpu else None)
        )
        # 可选：搜索阶段降低迭代数，加快收敛
        um_cfg["n_epochs"] = 150

        # 1) UMAP
        try:
            if use_gpu:
                Z, _ = embed_umap_gpu(X, um_cfg)
            else:
                Z, _ = embed_umap_cpu(X, um_cfg)
        except Exception as e:
            rows.append({"umap_nc": nc, "umap_nn": nn, "umap_min_dist": md, "metric": met,
                         "hdb_s": None, "hdb_ms_mode": None, "hdb_sel": None,
                         "trust": np.nan, "dbcv": np.nan, "n_clusters": -1, "noise_ratio": 1.0,
                         "error": f"UMAP:{e}"})
            continue

        # 2) Trustworthiness（一次 UMAP 仅算一次）
        trust = compute_trustworthiness(X, Z, k=ps.get("trust_k", 15), metric=met, use_gpu=use_gpu)

        # 3) 扫 HDBSCAN
        for (s, ms_mode, sel) in tqdm(hdb_grid, desc="HDBSCAN stage", leave=False):
            ms = _expand_min_samples(s, ms_mode)
            h_cfg = {"min_cluster_size": s, "min_samples": ms, "cluster_selection_method": sel}
            try:
                if use_gpu:
                    labels, probs, _ = cluster_hdbscan_gpu(Z, h_cfg)
                else:
                    labels, probs, _ = cluster_hdbscan_cpu(Z, h_cfg)
            except Exception as e:
                rows.append({"umap_nc": nc, "umap_nn": nn, "umap_min_dist": md, "metric": met,
                             "hdb_s": s, "hdb_ms_mode": ms_mode, "hdb_sel": sel,
                             "trust": trust, "dbcv": np.nan, "n_clusters": -1, "noise_ratio": 1.0,
                             "error": f"HDBSCAN:{e}"})
                continue

            n_noise = int((labels < 0).sum())
            n_clusters = int(len(np.unique(labels[labels >= 0])))
            noise_ratio = float(n_noise / len(labels)) if len(labels) else 1.0

            dbcv = compute_dbcv(Z, labels, metric=met)

            rows.append({
                "umap_nc": nc, "umap_nn": nn, "umap_min_dist": md, "metric": met,
                "hdb_s": s, "hdb_ms_mode": ms_mode, "hdb_sel": sel,
                "trust": trust, "dbcv": dbcv, "n_clusters": n_clusters, "noise_ratio": noise_ratio,
                "error": ""
            })

            if best is None or (dbcv, trust) > (best[0], best[1]):
                best = (dbcv, trust, (nc, nn, md, met, s, ms_mode, sel), labels, probs, Z)

    df = pd.DataFrame(rows)
    df.sort_values(["dbcv", "trust"], ascending=False, inplace=True)
    df.to_csv(outdir / "grid_scores.csv", index=False)

    if best is None:
        if not use_gpu and _CUML_IMPORT_ERR:
            raise SystemExit(f"[ERR] Param search failed and GPU UMAP unavailable. cuML import error: {_CUML_IMPORT_ERR}")
        raise SystemExit("[ERR] Param search failed: no valid combos.")

    (best_dbcv, best_trust, params, labels, probs, Z) = best
    (nc, nn, md, met, s, ms_mode, sel) = params
    best_combo = {
        "umap": {"n_components": nc, "n_neighbors": nn, "min_dist": md, "metric": met},
        "hdbscan": {"min_cluster_size": s, "min_samples": _expand_min_samples(s, ms_mode), "cluster_selection_method": sel},
        "scores": {"dbcv": best_dbcv, "trust": best_trust},
        "gpu_backend": bool(use_gpu)
    }
    with open(outdir / "best_combo.json", "w") as f:
        json.dump(best_combo, f, indent=2)

    return {"best": best_combo, "labels": labels, "probs": probs, "Z": Z}

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

    # ===== Features (CPU) =====
    soap_cfg = cfg["soap"]
    X_soap = compute_soap_features(
        atoms_list, species_Z,
        rcut=soap_cfg["rcut"], nmax=soap_cfg["nmax"], lmax=soap_cfg["lmax"], sigma=soap_cfg["sigma"],
        pooling=tuple(soap_cfg["pooling"]), batch=soap_cfg["batch"], periodic=soap_cfg["periodic"]
    )

    mbtr_cfg = cfg["mbtr"]
    X_mbtr = compute_mbtr_features(
        atoms_list, species_Z,
        k_terms=tuple(mbtr_cfg["k_terms"]), periodic=mbtr_cfg["periodic"], svd_dim=mbtr_cfg["svd_dim"]
    )

    X = np.hstack([X_soap, X_mbtr]).astype(np.float32, copy=False)

    if cfg["standardize"]:
        X = StandardScaler().fit_transform(X).astype(np.float32, copy=False)

    # ===== Embedding + Clustering（自动网格搜索 or 单次跑） =====
    use_gpu = bool(cfg["use_gpu"]) and USE_CUML
    ps_cfg = cfg.get("param_search", {"enable": False})

    if ps_cfg.get("enable", False):
        print("[INFO] Running parameter search (UMAP × HDBSCAN) ...")
        search_res = run_param_search(X, cfg, use_gpu, outdir)
        Z = search_res["Z"]
        labels = search_res["labels"]
        probs = search_res["probs"]
        backend = "gpu(cuML)" if use_gpu else "cpu(sklearn+hdbscan)"
        with open(outdir / "params_used.json", "w") as f:
            json.dump({**cfg, "backend_used": backend, "best_combo": search_res["best"]}, f, indent=2)
    else:
        if use_gpu:
            Z, umap_model = embed_umap_gpu(X, cfg["umap"])
            labels, probs, clusterer = cluster_hdbscan_gpu(Z, cfg["hdbscan"])
            backend = "gpu(cuML)"
        else:
            Z, umap_model = embed_umap_cpu(X, cfg["umap"])
            labels, probs, clusterer = cluster_hdbscan_cpu(Z, cfg["hdbscan"])
            backend = "cpu(sklearn+hdbscan)"
        with open(outdir / "params_used.json", "w") as f:
            json.dump({**cfg, "backend_used": backend}, f, indent=2)

    prototypes = pick_prototypes(labels, probs, topk=5)

    # ===== Outputs =====
    meta_out = meta.copy()
    meta_out["cluster_id"] = labels
    meta_out["soft_prob"] = probs
    meta_out["is_noise"] = (labels < 0).astype(int)
    meta_out.to_csv(outdir / "clusters.csv", index=False)

    emb_cols = [f"umap_{i}" for i in range(Z.shape[1])]
    pd.DataFrame(Z, columns=emb_cols).to_csv(outdir / "embedding_umap.csv", index=False)

    with open(outdir / "prototypes.json", "w") as f:
        json.dump({str(k): v for k, v in prototypes.items()}, f, indent=2)

    print(f"[OK] Backend={backend}. Wrote -> {outdir}/clusters.csv, embedding_umap.csv, prototypes.json, params_used.json")


if __name__ == "__main__":
    main()
