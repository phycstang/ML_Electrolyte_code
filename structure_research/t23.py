#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===== Global warnings & threads =====
import os, warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen\.core\.structure")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

# ===== Imports =====
import re, glob, sys, json
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter

from joblib import Parallel, delayed
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from scipy.spatial.distance import pdist
except Exception:
    pdist = None

from tqdm import tqdm
try:
    from tqdm_joblib import tqdm_joblib
    HAVE_TQDM_JOBLIB = True
except Exception:
    HAVE_TQDM_JOBLIB = False

# ===== Config =====
CFG = {
    # ---- 全局判定阈值 ----
    "frac_gate": 1.0,          # 通过比例门槛（可设 0.95~1.0）
    "psi6_gate": 0.90,         # 六方序下限
    "psi4_max": 0.40,          # 方格序上限
    "rad_rstd_gate": 0.50,     # 面内半径 CV 上限

    # ---- 邻域规模/扩包 ----
    "neighbor_cutoff": 6.5,    # 仅作保护；core-6 不再用它截断
    "kmax": 64,                # 候选近邻上限（用于法向生成）
    "supercell": (3, 3, 3),    # 扩包稳定中心统计

    # ---- pairplane 种子（法向生成路径）----
    "pair_kmax": 36,
    "pair_min_angle_deg": 12.0,
    "seed_max_variants": 60,
    "seed_min_normal_sep_deg": 5.0,

    # ---- 法向聚类 ----
    "angle_tol_deg": 5.0,
    "min_normal_cluster_count": 4,

    # ---- 一维分层（n 轴投影，稳健版参数）----
    "z_eps_min": 0.06,
    "z_eps_max": 0.90,
    "zflat_det": 1.80,
    "min_layer_size": 7,
    "merge_gain": 1.8,
    "reassign_margin_factor": 1.4,

    # 分层稳健性细化（峰检测/迭代/MAD）
    "peak_prominence_frac": 0.08,  # 峰显著性（相对最高峰占比）
    "peak_min_sep_factor": 1.0,    # 峰最小间隔 = factor * z_eps_min
    "kde_bw_factor": 1.0,          # 平滑带宽倍率
    "max_iter": 12,                # 迭代轮数
    "outlier_sigma": 2.5,          # 层内 |Δz| > outlier_sigma·MAD*sigma → 离群

    # ---- 空位占据（不影响主判定）----
    "vo_score_margin": 0.05,
    "vo_min_quality": 0.35,

    # ---- 诊断 ----
    "per_center_report": False,
    "per_center_report_dir": "tri_results_3d/per_center",
}

# ===== 常量 =====
HALOGENS = {"F","Cl","Br","I"}
_LAST_BEST_NORMAL = None
_LAST_NORMALS_INFO = {}

# ===== 基础工具 =====
def extract_mp_id(filename: str) -> str:
    m = re.search(r'mp-\d+', filename)
    return m.group(0) if m else ''

def compute_formula(s0: Structure, mode: str = "reduced") -> str:
    comp = s0.composition
    if mode == "reduced":
        f = comp.reduced_formula
    elif mode == "iupac":
        f, _ = comp.get_reduced_formula_and_factor(iupac_ordering=True)
    elif mode == "anonymized":
        f = comp.anonymized_formula
    else:
        f = comp.reduced_formula
    _rep = re.compile(r'^((?:[A-Z][a-z]?\d*)+)\1+$')
    m = _rep.match(f)
    if m: f = m.group(1)
    return f

def _as_frac_str(numer: int, denom: int) -> str:
    if denom <= 0: return "0"
    try: return str(Fraction(int(numer), int(denom)))
    except Exception: return f"{int(numer)}/{int(denom)}"

# ===== 选择中心超胞索引 =====
def _center_cell_indices_by_lattice(atoms, supercell, tol=1e-6):
    nx, ny, nz = supercell
    center_origin = np.array([nx//2, ny//2, nz//2], dtype=int)
    frac = atoms.get_scaled_positions(wrap=False)
    frac = np.mod(frac, 1.0)
    ijk = np.floor(frac * np.array(supercell, float) + tol).astype(int)
    ijk = np.minimum(ijk, np.array(supercell, int) - 1)
    center_mask = np.all(ijk == center_origin, axis=1)
    return np.where(center_mask)[0].tolist()

def get_center_unitcell_halogen_indices(atoms, supercell, halogen_symbols=HALOGENS, tol=1e-6):
    center_idx = _center_cell_indices_by_lattice(atoms, supercell, tol=tol)
    halogen_mask = np.array([a.symbol in halogen_symbols for a in atoms])
    mask = np.zeros(len(atoms), dtype=bool); mask[center_idx] = True
    return np.where(mask & halogen_mask)[0].tolist()

def get_center_unitcell_nonhalogen_indices(atoms, supercell, halogen_symbols=HALOGENS, tol=1e-6):
    center_idx = _center_cell_indices_by_lattice(atoms, supercell, tol=tol)
    halogen_mask = np.array([a.symbol in halogen_symbols for a in atoms])
    mask = np.zeros(len(atoms), dtype=bool); mask[center_idx] = True
    return np.where(mask & (~halogen_mask))[0].tolist()

# ===== 向量/法向工具 =====
def _unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    if n < 1e-12: return np.array([0.0,0.0,1.0], float)
    return v / n

def _cluster_and_average_normals(normals, angle_tol_deg=2.0, min_count=1):
    if not normals: return []
    N = [_unit(n) if n[2]>=0 else -_unit(n) for n in normals]
    ct = np.cos(np.deg2rad(angle_tol_deg))
    clusters, centers = [], []
    for n in N:
        placed = False
        for i, c in enumerate(centers):
            if abs(float(np.dot(n, c))) >= ct:
                clusters[i].append(n)
                v = _unit(np.sum(clusters[i], axis=0))
                centers[i] = v if v[2]>=0 else -v
                placed = True; break
        if not placed:
            clusters.append([n]); centers.append(n)
    out = []
    for vecs in clusters:
        if len(vecs) >= min_count:
            v = _unit(np.sum(vecs, axis=0))
            out.append(v if v[2]>=0 else -v)
    return out

# ===== 面内基与序参量 =====
def _plane_basis_from_normal(n):
    n = _unit(n)
    ex = _unit(np.cross(n, [1.0,0.0,0.0]))
    if np.linalg.norm(ex) < 1e-8:
        ex = _unit(np.cross(n, [0.0,1.0,0.0]))
    ey = _unit(np.cross(n, ex))
    return ex, ey

def _psi2_on_vectors(V2, k):
    V2 = np.asarray(V2, float)
    r = np.linalg.norm(V2, axis=1)
    mask = r > 1e-10
    if not np.any(mask): return 0.0
    ang = np.arctan2(V2[mask,1], V2[mask,0])
    return float(np.abs(np.mean(np.exp(1j*k*ang))))

def psi_k_order(center, neigh_pos, normal, k: int):
    if len(neigh_pos) == 0: return 0.0
    n = _unit(normal)
    R = neigh_pos - center
    z = R @ n
    Rp = R - np.outer(z, n)
    ex, ey = _plane_basis_from_normal(n)
    x = Rp @ ex; y = Rp @ ey
    theta = np.arctan2(y, x)
    return float(np.abs(np.mean(np.exp(1j * k * theta))))

# ===== 空位占据质量分 =====
def _rstd(x):
    x = np.asarray(x, float)
    mu = float(np.mean(x)) if x.size else 0.0
    if mu <= 1e-12: return 1.0
    return float(np.std(x)/mu)

def _pair_opposites_score(vecs):
    V = np.asarray(vecs, float)
    nrm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    U = V / nrm
    used = np.zeros(len(U), dtype=bool)
    pairs = []
    for _ in range(3):
        best = None; best_val = +1.0
        for i in range(len(U)):
            if used[i]: continue
            for j in range(i+1, len(U)):
                if used[j]: continue
                val = float(np.dot(U[i], U[j]))
                if val < best_val:
                    best_val = val; best = (i,j)
        if best is None: break
        i,j = best
        used[i]=used[j]=True
        pairs.append((i,j))
    if len(pairs) < 3: return 0.0
    s = [np.linalg.norm(U[i]+U[j])/2.0 for i,j in pairs]
    s = float(np.mean(s))
    return float(max(0.0, 1.0 - s))

def _tetra_quality(center, neigh4):
    P = np.asarray(neigh4, float); c = np.asarray(center, float)
    r = np.linalg.norm(P - c, axis=1)
    rscore = 1.0 - min(1.0, _rstd(r)*4.0)
    if pdist is not None and P.shape[0] == 4:
        d = pdist(P)
        dscore = 1.0 - min(1.0, _rstd(d)*4.0)
    else:
        dscore = rscore
    U = P - c; U /= (np.linalg.norm(U, axis=1, keepdims=True)+1e-12)
    cos_vals = [float(np.dot(U[i], U[j])) for i, j in combinations(range(4), 2)]
    ang_err = np.mean(np.abs(np.array(cos_vals) + 1.0/3.0))
    ascore = 1.0 - min(1.0, ang_err/0.15)
    return float(max(0.0, min(1.0, 0.35*rscore + 0.25*dscore + 0.40*ascore)))

def _octa_quality(center, neigh6):
    P = np.asarray(neigh6, float); c = np.asarray(center, float)
    V = P - c; r = np.linalg.norm(V, axis=1)
    rscore = 1.0 - min(1.0, _rstd(r)*4.0)
    opp = _pair_opposites_score(V)
    idx = np.argsort(r)
    pair_lens = np.array([r[idx[5]]+r[idx[0]], r[idx[4]]+r[idx[1]], r[idx[3]]+r[idx[2]]]) / 2.0
    lscore = 1.0 - min(1.0, _rstd(pair_lens)*6.0)
    return float(max(0.0, min(1.0, 0.4*rscore + 0.4*opp + 0.2*lscore)))

def count_void_occupancy_by_geometry(atoms, center_nonhalogen_idx, halogen_indices,
                                     score_margin=None, min_quality=None):
    if score_margin is None: score_margin = CFG["vo_score_margin"]
    if min_quality is None: min_quality = CFG["vo_min_quality"]
    if len(center_nonhalogen_idx) == 0 or len(halogen_indices) == 0:
        return 0, 0, 0
    Hpos = np.array([atoms[i].position for i in halogen_indices])
    tetra = octa = other = 0
    for idx in center_nonhalogen_idx:
        pos = atoms[idx].position
        d = np.linalg.norm(Hpos - pos, axis=1)
        order = np.argsort(d)
        neigh4_idx = order[:4] if len(order) >= 4 else None
        neigh6_idx = order[:6] if len(order) >= 6 else None
        q_tet = _tetra_quality(pos, Hpos[neigh4_idx]) if (neigh4_idx is not None and len(neigh4_idx)==4) else -1.0
        q_oct = _octa_quality(pos, Hpos[neigh6_idx]) if (neigh6_idx is not None and len(neigh6_idx)==6) else -1.0
        if (q_tet >= q_oct + score_margin) and (q_tet >= min_quality): tetra += 1
        elif (q_oct >= q_tet + score_margin) and (q_oct >= min_quality): octa += 1
        else:
            if max(q_tet, q_oct) >= min_quality:
                if q_tet >= q_oct: tetra += 1
                else: octa += 1
            else:
                other += 1
    return tetra, octa, other

# ===== pairplane：从近邻对生成法向 =====
def _seed_normals_from_neighbor_pairs_simple(
    cpos, Hpos, *,
    cutoff, kmax, pair_kmax=18, pair_min_angle_deg=12.0,
    max_variants=4, min_sep_deg=7.0,
):
    D = np.linalg.norm(Hpos - cpos, axis=1)
    order = np.argsort(D)
    order = np.array([j for j in order if D[j] > 1e-9], dtype=int)
    if order.size < 2:
        return []
    j = order[:min(len(order), int(kmax))]
    if cutoff and cutoff > 0 and len(j) >= 2:
        keep = np.where(D[j] <= float(cutoff))[0]
        if keep.size >= 2: j = j[keep]
    if len(j) > int(pair_kmax): j = j[:int(pair_kmax)]

    neigh = Hpos[j]
    V = neigh - cpos
    r = np.linalg.norm(V, axis=1)
    U = V / (r[:, None] + 1e-12)

    cos_max = np.cos(np.deg2rad(float(pair_min_angle_deg)))
    normals_out = []
    for a in range(len(U)-1):
        ua = U[a]
        for b in range(a+1, len(U)):
            ub = U[b]
            if abs(float(np.dot(ua, ub))) >= cos_max:
                continue
            n = np.cross(ua, ub)
            n = n / (np.linalg.norm(n) + 1e-12)
            if n[2] < 0: n = -n
            # 去重（最小夹角）
            duplicate = False
            for n0 in normals_out:
                c = float(np.clip(np.dot(n0, n), -1.0, 1.0))
                ang = np.degrees(np.arccos(c))
                if ang < float(min_sep_deg):
                    duplicate = True; break
            if duplicate: continue
            normals_out.append(n)
            if len(normals_out) >= int(max_variants):
                return normals_out
    return normals_out

def _planes_from_all_centers_pairplane(Hpos, halogen_indices, centers_hal):
    normals = []
    idx_in_hal = {at_idx: row for row, at_idx in enumerate(halogen_indices)}
    per_center_counts = []
    for center_idx in centers_hal:
        hrow = idx_in_hal.get(center_idx, None)
        if hrow is None:
            per_center_counts.append(0)
            continue
        cpos = Hpos[hrow]
        normals_center = _seed_normals_from_neighbor_pairs_simple(
            cpos, Hpos,
            cutoff=CFG["neighbor_cutoff"], kmax=CFG["kmax"],
            pair_kmax=CFG.get("pair_kmax", 18),
            pair_min_angle_deg=CFG.get("pair_min_angle_deg", 12.0),
            max_variants=CFG.get("seed_max_variants", 4),
            min_sep_deg=CFG.get("seed_min_normal_sep_deg", 7.0),
        )
        per_center_counts.append(len(normals_center))
        normals.extend(normals_center)
    _LAST_NORMALS_INFO.clear()
    _LAST_NORMALS_INFO.update({
        "total_raw_normals": int(len(normals)),
        "per_center_raw_counts": per_center_counts[:2000],
    })
    return normals

# ======= 一维分层（n 轴投影）：稳健版（峰检测 + 迭代 median/MAD + 合并） =======
def project_and_cluster_along_n(
    X, n, *,
    z_eps_min=0.06, z_eps_max=0.90,
    min_layer_size=7,
    merge_gain=1.8,
    peak_prominence_frac=None,
    peak_min_sep_factor=None,
    kde_bw_factor=None,
    max_iter=None,
    outlier_sigma=None,
):
    """返回: clusters(list[np.ndarray]), stats(dict), labels(np.ndarray)
       stats = { 'z':z, 'order':order, 'eps0':eps0, 'meds':np.array, 'sigmas':np.array }
    """
    # 缺省取 CFG
    if peak_prominence_frac is None: peak_prominence_frac = CFG.get("peak_prominence_frac", 0.08)
    if peak_min_sep_factor is None:  peak_min_sep_factor  = CFG.get("peak_min_sep_factor", 1.0)
    if kde_bw_factor is None:        kde_bw_factor        = CFG.get("kde_bw_factor", 1.0)
    if max_iter is None:             max_iter             = CFG.get("max_iter", 12)
    if outlier_sigma is None:        outlier_sigma        = CFG.get("outlier_sigma", 2.5)

    def _mad_sigma(arr):
        if arr.size == 0:
            return 0.0, 0.0
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return med, 1.4826 * mad

    n = _unit(n)
    X = np.asarray(X, float)
    N = len(X)
    if N == 0:
        return [], {'z':np.array([]), 'order':np.array([],int), 'eps0':0.0,
                    'meds':np.array([]), 'sigmas':np.array([])}, np.array([], int)

    # 1) 投影 & 预估 eps0
    z = X @ n
    order = np.argsort(z)
    z_sorted = z[order]
    gaps = np.diff(z_sorted)
    if gaps.size > 0:
        k = max(1, int(0.7 * gaps.size))
        small = np.sort(np.abs(gaps))[:k]
        eps0 = float(np.clip(1.1 * np.quantile(small, 0.75), z_eps_min, z_eps_max))
    else:
        eps0 = float(z_eps_max)

    # 2) 平滑直方图找层峰
    q25, q75 = np.quantile(z, [0.25, 0.75])
    iqr = max(1e-9, float(q75 - q25))
    bin_w = 2.0 * iqr / (N ** (1.0/3.0))
    bin_w = float(np.clip(bin_w, z_eps_min*0.5, z_eps_max))
    z_min, z_max = float(np.min(z)), float(np.max(z))
    M = max(25, int(np.ceil((z_max - z_min) / (bin_w*0.75))))
    grid = np.linspace(z_min - bin_w, z_max + bin_w, M)
    hist, _ = np.histogram(z, bins=M, range=(z_min - bin_w, z_max + bin_w))
    hist = hist.astype(float)

    std = float(np.std(z)) if N>1 else 0.0
    bw = 1.06 * (std + 1e-12) * (N ** (-1/5)) * kde_bw_factor
    bw = float(np.clip(bw, z_eps_min*0.5, z_eps_max))
    k_range = int(max(3, np.ceil(4.0 * bw / ( (grid[1]-grid[0]) + 1e-12 ))))
    ks = np.arange(-k_range, k_range+1)
    gk = np.exp(-0.5 * (ks / max(1.0, (bw / (grid[1]-grid[0]))))**2)
    gk /= np.sum(gk)
    smooth = np.convolve(hist, gk, mode="same")

    peak_mask = np.zeros_like(smooth, dtype=bool)
    for i in range(1, len(smooth)-1):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]:
            peak_mask[i] = True
    peak_vals = smooth[peak_mask]
    peak_idx  = np.where(peak_mask)[0]

    if peak_vals.size == 0:
        # 回退为一层
        med0, sig0 = _mad_sigma(z)
        labels0 = np.zeros(N, dtype=int)
        return [np.arange(N, dtype=int)], {'z':z, 'order':order, 'eps0':eps0,
                                           'meds':np.array([med0]), 'sigmas':np.array([sig0])}, labels0

    vmax = float(np.max(peak_vals))
    keep = peak_vals >= (peak_prominence_frac * vmax)
    peak_idx = peak_idx[keep]
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(smooth))])

    p_ord = np.argsort(smooth[peak_idx])[::-1]
    cand = list(peak_idx[p_ord])
    selected = []
    min_sep = max(CFG["z_eps_min"] * peak_min_sep_factor, bin_w)
    for i in cand:
        zi = grid[i]
        ok = True
        for j in selected:
            if abs(zi - grid[j]) < min_sep:
                ok = False; break
        if ok:
            selected.append(i)
    selected = np.array(sorted(selected), dtype=int)
    if selected.size == 0:
        selected = np.array([int(np.argmax(smooth))])

    centers = grid[selected].astype(float)
    labels = np.full(N, -1, dtype=int)

    # 3) 迭代分配 / 更新 / 剔离群
    cap0 = max(2.0*bw, eps0, CFG["z_eps_min"])
    for _ in range(int(max_iter)):
        for i in range(N):
            j = int(np.argmin(np.abs(centers - z[i])))
            labels[i] = j if abs(z[i] - centers[j]) <= cap0 else -1

        new_centers = []
        sigmas = []
        for j in range(len(centers)):
            idxs = np.where(labels == j)[0]
            if idxs.size < min_layer_size: continue
            med, sigma = _mad_sigma(z[idxs])
            keep = np.abs(z[idxs] - med) <= max(CFG["z_eps_min"], outlier_sigma * max(1e-9, sigma))
            idxs = idxs[keep]
            if idxs.size < min_layer_size: continue
            med, sigma = _mad_sigma(z[idxs])
            new_centers.append(med); sigmas.append(sigma)

        if len(new_centers) == 0:
            med0, sig0 = _mad_sigma(z)
            labels0 = np.zeros(N, dtype=int)
            return [np.arange(N, dtype=int)], {'z':z, 'order':order, 'eps0':eps0,
                                               'meds':np.array([med0]), 'sigmas':np.array([sig0])}, labels0

        new_centers = np.array(new_centers, float)
        sigmas = np.array(sigmas, float)
        if centers.size == new_centers.size and np.allclose(centers, new_centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers

    # 严格分配（≤ max(eps0, 3σ)）
    labels = np.full(N, -1, dtype=int)
    tmp_sigs = []
    for j in range(len(centers)):
        idxs = np.where(np.argmin(abs(centers - z[:,None]), axis=1) == j)[0]
        _, sj = _mad_sigma(z[idxs]) if idxs.size>0 else (centers[j], CFG["z_eps_min"])
        tmp_sigs.append(sj)
    sigmas = np.array(tmp_sigs, float)

    for i in range(N):
        j = int(np.argmin(np.abs(centers - z[i])))
        thr = max(eps0, 3.0 * max(1e-9, sigmas[j]))
        if abs(z[i] - centers[j]) <= thr:
            labels[i] = j

    # 4) 形成层簇并排序
    clusters = []
    meds = []
    sigm = []
    for j in range(len(centers)):
        idxs = np.where(labels == j)[0]
        if idxs.size >= min_layer_size:
            m = float(np.median(z[idxs]))
            s = 1.4826 * float(np.median(np.abs(z[idxs] - m)))
            clusters.append(np.array(idxs, dtype=int))
            meds.append(m); sigm.append(s)
    if len(clusters) == 0:
        med0, sig0 = _mad_sigma(z)
        labels0 = np.zeros(N, dtype=int)
        return [np.arange(N, dtype=int)], {'z':z, 'order':order, 'eps0':eps0,
                                           'meds':np.array([med0]), 'sigmas':np.array([sig0])}, labels0

    meds = np.array(meds, float)
    sigm = np.array(sigm, float)
    srt = np.argsort(meds)
    clusters = [clusters[i] for i in srt]
    meds = meds[srt]; sigm = sigm[srt]

    # 5) 近层合并（与旧版准则等价）
    def _mad_sigma_by_indices(idxs):
        zz = z[idxs]
        med = float(np.median(zz))
        mad = float(np.median(np.abs(zz - med)))
        return med, 1.4826 * mad

    merged, merged_m, merged_s = [], [], []
    i = 0
    while i < len(clusters):
        cur = list(clusters[i])
        cur_m, cur_s = meds[i], sigm[i]
        j = i + 1
        while j < len(clusters):
            thr = max(merge_gain*eps0, 2.0*cur_s, 2.0*sigm[j])
            if abs(meds[j] - cur_m) <= thr:
                cur.extend(list(clusters[j]))
                cur_m, cur_s = _mad_sigma_by_indices(cur)
                j += 1
            else:
                break
        merged.append(np.array(sorted(cur), dtype=int))
        merged_m.append(cur_m); merged_s.append(cur_s)
        i = j

    clusters = [c for c in merged if c.size >= int(min_layer_size)]
    meds = np.array([merged_m[k] for k in range(len(merged)) if merged[k].size >= int(min_layer_size)], float)
    sigm = np.array([merged_s[k] for k in range(len(merged)) if merged[k].size >= int(min_layer_size)], float)

    labels0 = np.full(N, -1, dtype=int)
    for cid, c in enumerate(clusters):
        labels0[c] = cid

    stats = { 'z': z, 'order': order, 'eps0': float(eps0), 'meds': meds, 'sigmas': sigm }
    return clusters, stats, labels0

# ======= 基于分层的方向评估（面内“最近 6”） =======
def evaluate_direction_metrics(
    n, Hpos, halogen_indices, centers_hal_indices,
    *, neighbor_cutoff=None,
    zflat_det=None, z_eps_min=None, z_eps_max=None,
    z_local_radius=None, z_local_min_pts=None,
    pre_Dmat=None, pre_neighbors=None,
):
    if neighbor_cutoff is None: neighbor_cutoff = CFG["neighbor_cutoff"]
    if zflat_det is None:       zflat_det       = CFG.get("zflat_det", 1.80)
    if z_eps_min is None:       z_eps_min       = CFG.get("z_eps_min", 0.06)
    if z_eps_max is None:       z_eps_max       = CFG.get("z_eps_max", 0.90)

    min_layer_size = int(CFG.get("min_layer_size", 7))
    merge_gain     = float(CFG.get("merge_gain", 1.8))
    reassign_fac   = float(CFG.get("reassign_margin_factor", 1.4))

    n = _unit(n)
    X = np.asarray(Hpos, float)

    # 1) 投影并分层（稳健版）
    layers, stats, labels0 = project_and_cluster_along_n(
        X, n,
        z_eps_min=z_eps_min, z_eps_max=z_eps_max,
        min_layer_size=min_layer_size,
        merge_gain=merge_gain,
    )

    # 2) 选“合格层”
    good_layers = []
    for cid in range(len(layers)):
        if stats['sigmas'][cid] <= float(zflat_det):
            good_layers.append(cid)
    good_layers = set(good_layers)

    # 3) 初步标签（好层保留）
    labels = np.full(len(X), -1, dtype=int)
    for cid, c in enumerate(layers):
        if cid in good_layers:
            labels[c] = cid

    # 4) 软重分配
    if good_layers:
        z = stats['z']; meds = stats['meds']
        good_ids = sorted(list(good_layers))
        thr = float(max(zflat_det, stats['eps0'])) * reassign_fac
        for i in range(len(X)):
            if labels[i] >= 0: continue
            j = int(np.argmin(np.abs(meds[good_ids] - z[i])))
            gid = good_ids[j]
            if abs(z[i] - meds[gid]) <= thr:
                labels[i] = gid

    # 5) 逐中心（Δz 异常剔除 + 直接取最近 6）
    idx_in_hal = {halogen_indices[i]: i for i in range(len(halogen_indices))}
    psi6_list, psi4_list, zrms_list, rstd_list, coord_list = [], [], [], [], []

    ex, ey = _plane_basis_from_normal(n)

    valid_cnt = 0; coord6_cnt = 0
    for idx_c in centers_hal_indices:
        c_in = idx_in_hal.get(idx_c, None)
        if c_in is None:
            psi6_list.append(0.0); psi4_list.append(1.0); zrms_list.append(1e9); rstd_list.append(1.0); coord_list.append(0)
            continue
        cid = labels[c_in]
        if cid < 0:
            psi6_list.append(0.0); psi4_list.append(1.0); zrms_list.append(1e9); rstd_list.append(1.0); coord_list.append(0)
            continue

        layer_inds = (layers[cid] if cid < len(layers) else [])
        if len(layer_inds) < 7:
            psi6_list.append(0.0); psi4_list.append(1.0); zrms_list.append(1e9); rstd_list.append(1.0); coord_list.append(0)
            continue

        # Δz 异常剔除（混层弱污染）
        z_med = float(stats['meds'][cid])
        z_sig = float(stats['sigmas'][cid])
        z_thr = max(1e-6, CFG.get("outlier_sigma", 2.5) * z_sig)

        cand = []
        for ii in layer_inds:
            if ii == c_in: continue
            dz = abs(stats['z'][ii] - z_med)
            if dz <= z_thr:
                cand.append(ii)

        if len(cand) < 6:
            psi6_list.append(0.0); psi4_list.append(1.0); zrms_list.append(1e9); rstd_list.append(1.0); coord_list.append(0)
            continue

        valid_cnt += 1
        cpos = X[c_in]
        R = X[cand] - cpos
        z_comp = (R @ n); Rp = R - np.outer(z_comp, n)

        x = Rp @ ex; y = Rp @ ey
        V2 = np.stack([x,y], axis=1)
        r2 = np.einsum('ij,ij->i', V2, V2)
        ord_r = np.argsort(r2)

        if len(ord_r) < 6:
            psi6_list.append(0.0); psi4_list.append(1.0); zrms_list.append(1e9); rstd_list.append(1.0); coord_list.append(0)
            continue

        pick6 = ord_r[:6]
        V2_core = V2[pick6]

        psi6 = _psi2_on_vectors(V2_core, 6)
        psi4 = _psi2_on_vectors(V2_core, 4)
        zrms = float(stats['sigmas'][cid])
        d6 = np.linalg.norm(V2_core, axis=1)
        rstd = float(np.std(d6) / (np.mean(d6) + 1e-12))

        coord = 6
        coord6_cnt += 1

        psi6_list.append(psi6); psi4_list.append(psi4)
        zrms_list.append(zrms); rstd_list.append(rstd); coord_list.append(coord)

    n_centers = len(centers_hal_indices)
    frac6 = coord6_cnt / max(1, n_centers)
    valid_frac = valid_cnt / max(1, n_centers)
    return psi6_list, psi4_list, zrms_list, rstd_list, coord_list, frac6, valid_frac

# ===== 主检测器 =====
def detect_halogen_triangular_layered(s0: Structure, log_buffer=None):
    def log(msg):
        if log_buffer is not None:
            log_buffer.append(str(msg))

    atoms_once = AseAtomsAdaptor.get_atoms(s0)
    atoms_super = atoms_once * tuple(CFG["supercell"])
    halogen_indices = [i for i, a in enumerate(atoms_super) if a.symbol in HALOGENS]
    centers_hal = get_center_unitcell_halogen_indices(atoms_super, CFG["supercell"], HALOGENS)
    if len(centers_hal) == 0 or len(halogen_indices) == 0:
        log("[detect] 无中心或全局无卤素，直接失败")
        return False, 0.0, 0.0, 0.0, 0.0

    Hpos = np.array([atoms_super[i].position for i in halogen_indices])

    normals_raw = _planes_from_all_centers_pairplane(Hpos, halogen_indices, centers_hal)
    normals_K = _cluster_and_average_normals(
        normals_raw, angle_tol_deg=CFG["angle_tol_deg"], min_count=CFG["min_normal_cluster_count"]
    )
    _LAST_NORMALS_INFO["clustered_count"] = int(len(normals_K))
    if len(normals_K) == 0:
        log("[detect] 法向生成/聚类为空")
        return False, 0.0, 0.0, 0.0, 0.0

    log(f"[detect] raw_normals={_LAST_NORMALS_INFO.get('total_raw_normals',0)}, "
        f"clustered={len(normals_K)}, HposN={len(Hpos)}, centers={len(centers_hal)}")

    best_frac = -1.0; best_avg_coord = 0.0; best_avg_rstd = 0.0; best_n = None
    best_diag = None

    for idx, n_try in enumerate(normals_K):
        psi6_list, psi4_list, zr_list, rstd_list, coord_list, _, _ = evaluate_direction_metrics(
            n_try, Hpos, halogen_indices, centers_hal,
            neighbor_cutoff=CFG["neighbor_cutoff"],
            zflat_det=CFG.get("zflat_det", 1.80),
            z_eps_min=CFG["z_eps_min"], z_eps_max=CFG["z_eps_max"],
        )

        reasons = []
        for ps6, ps4, zr, rs, c in zip(psi6_list, psi4_list, zr_list, rstd_list, coord_list):
            r = []
            if c != 6:                         r.append("coord!=6")
            if ps6 < CFG["psi6_gate"]:         r.append(f"psi6<{CFG['psi6_gate']:.2f}")
            if ps4 > CFG["psi4_max"]:          r.append(f"psi4>{CFG['psi4_max']:.2f}")
            if rs  > CFG["rad_rstd_gate"]:     r.append(f"rad_rstd>{CFG['rad_rstd_gate']:.2f}")
            reasons.append("OK" if not r else "|".join(r))

        passes = [1 if (x=="OK") else 0 for x in reasons]
        n_centers = len(centers_hal)
        frac_pass = (np.sum(passes) / max(1, n_centers)) if n_centers>0 else 0.0

        avg_coord = float(np.mean(coord_list)) if len(coord_list) else 0.0
        avg_rstd  = float(np.mean(rstd_list)) if len(rstd_list) else 0.0
        log(f"[detect] dir#{idx+1}/{len(normals_K)} "
            f"frac_pass={frac_pass:.3f} avg_coord={avg_coord:.2f} avg_rstd={avg_rstd:.3f}")

        cnt = Counter(r for r in reasons if r != "OK")
        top = ", ".join(f"{k}:{v}" for k,v in cnt.most_common(10))
        log(f"[detect] dir#{idx+1} reasons_top: {top if top else 'all OK for passes'}")

        _df_dir = pd.DataFrame(dict(
            psi6=np.asarray(psi6_list, float),
            psi4=np.asarray(psi4_list, float),
            z_rms=np.asarray(zr_list, float),
            radial_rstd=np.asarray(rstd_list, float),
            coord=np.asarray(coord_list, int),
            pass_=np.asarray(passes, int),
            reason=np.asarray(reasons, object),
        ))
        log(f"[detect] dir#{idx+1} n_vec={list(np.round(_unit(n_try),8))}")
        log("[Per-Dir Per-Center Table]")
        log(_df_dir.to_csv(index=False))

        if frac_pass > best_frac:
            best_frac = float(frac_pass)
            best_avg_coord = avg_coord
            best_avg_rstd  = avg_rstd
            best_n = n_try
            best_diag = dict(
                psi6_list=psi6_list, psi4_list=psi4_list,
                z_rms_list=zr_list, radial_rstd_list=rstd_list,
                coord_list=coord_list, passes=passes, reasons=reasons
            )

    if best_n is not None:
        globals()['_LAST_BEST_NORMAL'] = best_n

    tri_ok = (best_frac >= float(CFG.get("frac_gate", 1.0)))
    _LAST_NORMALS_INFO["best_frac_pass"] = float(best_frac)
    _LAST_NORMALS_INFO["best_avg_coord"] = float(best_avg_coord)
    _LAST_NORMALS_INFO["best_avg_rstd"]  = float(best_avg_rstd)
    _LAST_NORMALS_INFO["best_dir"] = [float(x) for x in (best_n if best_n is not None else [0,0,1])]
    _LAST_NORMALS_INFO["n_centers"] = int(len(centers_hal))

    if log_buffer is not None and best_diag is not None:
        _df_best = pd.DataFrame(dict(
            psi6=best_diag["psi6_list"],
            psi4=best_diag["psi4_list"],
            z_rms=best_diag["z_rms_list"],
            radial_rstd=best_diag["radial_rstd_list"],
            coord=best_diag["coord_list"],
            pass_=best_diag["passes"],
            reason=best_diag["reasons"],
        ))
        log("\n[Best Direction Per-Center (reasons)]")
        log(_df_best.to_csv(index=False))

    return tri_ok, float(best_frac), float(best_avg_coord), float(best_frac), float(best_avg_rstd)

# ===== 单文件处理 =====
def process_one_file(filepath, verbose=False, log_buffer=None):
    def vlog(step_msg):
        msg = str(step_msg)
        if verbose:
            if log_buffer is not None: log_buffer.append(msg)
            else: print(msg, flush=True)

    try:
        vlog(f"[1/8] 读取结构: {filepath}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            s0 = Structure.from_file(str(Path(filepath)))

        vlog("[2/8] 扩包并选择中心单元的卤素/非卤素")
        atoms_once = AseAtomsAdaptor.get_atoms(s0)
        atoms_super = atoms_once * tuple(CFG["supercell"])
        halogen_indices_as = [i for i, a in enumerate(atoms_super) if a.symbol in HALOGENS]
        center_hal_as = get_center_unitcell_halogen_indices(atoms_super, tuple(CFG["supercell"]), HALOGENS)
        center_non = get_center_unitcell_nonhalogen_indices(atoms_super, tuple(CFG["supercell"]), HALOGENS)
        vlog(f"[2/8] supercell={CFG['supercell']} | halogen_total={len(halogen_indices_as)} | centers_hal={len(center_hal_as)} | centers_nonhal={len(center_non)}")
        if len(halogen_indices_as) == 0 or len(center_hal_as) == 0:
            raise RuntimeError("中心单元内无卤素或全局无卤素，跳过")

        vlog("[3/8] 构建 Hpos 并生成 pairplane 种子法向")
        Hpos = np.array([atoms_super[i].position for i in halogen_indices_as])
        normals_raw = _planes_from_all_centers_pairplane(Hpos, halogen_indices_as, center_hal_as)
        normals_K = _cluster_and_average_normals(normals_raw,
                                                 angle_tol_deg=CFG["angle_tol_deg"],
                                                 min_count=CFG["min_normal_cluster_count"])
        vlog(f"[3/8] raw_normals={_LAST_NORMALS_INFO.get('total_raw_normals',0)}, clustered={len(normals_K)}")
        if len(normals_K)==0:
            raise RuntimeError("无法生成有效法向")

        vlog("[4/8] 单阶段全局评估（投影-分层-面内 core-6）")
        tri_ok, frac_pass, avg_coord, _avg_tri, avg_rstd = detect_halogen_triangular_layered(s0, log_buffer=log_buffer)
        vlog(f"[4/8] tri_ok={int(tri_ok)} frac_pass={frac_pass:.3f} avg_coord={avg_coord:.2f} avg_rstd={avg_rstd:.3f}")
        vlog(f"[4/8] normals_info={json.dumps(_LAST_NORMALS_INFO)}")
        if globals().get("_LAST_BEST_NORMAL", None) is not None:
            vlog(f"[4/8] best_normal={_LAST_BEST_NORMAL}")

        vlog("[5/8] 统计空位占据（四/八面体）")
        tetra_filled, octa_filled, _ = count_void_occupancy_by_geometry(
            atoms_super, center_non, halogen_indices_as,
            score_margin=CFG["vo_score_margin"], min_quality=CFG["vo_min_quality"]
        )
        vlog(f"[5/8] tetra_filled={tetra_filled}, octa_filled={octa_filled}")

        vlog("[6/8] 汇总材料基础信息")
        name = os.path.basename(filepath)
        mp_guess = extract_mp_id(name)
        chemical_formula = compute_formula(s0, mode="reduced")
        elems_once = atoms_once
        element_count = len(set([a.symbol for a in elems_once]))
        uniq = sorted(set([a.symbol for a in atoms_super]))
        halogen_elems = ",".join(sorted(set(uniq) & HALOGENS))
        vlog(f"[6/8] formula={chemical_formula}, mp={mp_guess}, halogens={halogen_elems}, elem_count={element_count}")

        n_center_hal = len(center_hal_as)
        n_octa_sites  = n_center_hal * 1
        n_tetra_sites = n_center_hal * 2
        octa_rate_str  = _as_frac_str(octa_filled,  n_octa_sites)
        tetra_rate_str = _as_frac_str(tetra_filled, n_tetra_sites)

        vlog("[8/8] 生成结果行")
        row = dict(
            filename=name,
            chemical_formula=chemical_formula,
            mp_id=mp_guess,
            halogen_elements=halogen_elems,
            element_count=element_count,
            is_triangular_layered=int(tri_ok),
            octahedral_occupancy_count=int(octa_filled),
            tetrahedral_occupancy_count=int(tetra_filled),
            octahedral_occupancy_rate=octa_rate_str,
            tetrahedral_occupancy_rate=tetra_rate_str,
            _diag_frac_pass=round(frac_pass,3),
            _diag_avg_core6_ltcut=round(avg_coord,2),
            _diag_avg_radial_rstd=round(avg_rstd,3),
        )
        return row

    except Exception as e:
        if verbose and log_buffer is not None:
            log_buffer.append(f"[ERROR] {e}")
        elif verbose:
            print(f"[ERROR] {e}", flush=True)
        return dict(
            filename=os.path.basename(filepath),
            chemical_formula="",
            mp_id=extract_mp_id(filepath),
            halogen_elements="",
            element_count="",
            is_triangular_layered=0,
            octahedral_occupancy_count="",
            tetrahedral_occupancy_count="",
            octahedral_occupancy_rate="",
            tetrahedral_occupancy_rate="",
            _diag_frac_pass="",
            _diag_avg_core6_ltcut="",
            _diag_avg_radial_rstd="",
            error=str(e),
        )

# ===== CLI =====
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Halogen triangular-layer detector via robust 1D layering + in-plane nearest-6 core."
    )
    ap.add_argument("input", help="Path to a CIF/POSCAR/VASP file or a directory")
    ap.add_argument("--sc", type=int, nargs=3, default=None, metavar=("A","B","C"),
                    help="Override supercell in CFG (e.g., --sc 3 3 3)")
    ap.add_argument("--outdir", type=str, default="tri_results_3d")
    ap.add_argument("--patterns", type=str, nargs="*", default=["**/*.cif","**/POSCAR*","**/*.vasp"])
    ap.add_argument("--n_jobs", type=int, default=52, help="并行进程数；设为 1 可配合 --verbose 顺序打印步骤")
    ap.add_argument("--verbose", action="store_true", help="逐步日志（建议在 --n_jobs 1 时使用）")
    ap.add_argument("--set", type=str, nargs="*", default=[],
                    help=("Override CFG by k=v pairs, e.g. "
                          "--set frac_gate=0.95 zflat_det=1.6 merge_gain=1.6 "
                          "peak_prominence_frac=0.10 outlier_sigma=2.2 reassign_margin_factor=1.2"))
    ap.add_argument("--dump-cfg", action="store_true", help="在日志文件开头输出当前 CFG 全量内容")

    args = ap.parse_args()
    in_path = Path(args.input)

    if args.sc is not None:
        CFG["supercell"] = tuple(int(x) for x in args.sc)

    def _coerce_val(v):
        for caster in (int, float):
            try:
                return caster(v)
            except Exception:
                pass
        if isinstance(v, str) and v.lower() in ("true","false"):
            return v.lower() == "true"
        return v
    for kv in args.__dict__.get("set", []) or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            CFG[k.strip()] = _coerce_val(v.strip())

    if in_path.is_file():
        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        log_file = out_dir / f"{stem}__full_log.txt"

        log_buffer = []
        if args.dump_cfg:
            log_buffer.append("[CFG]")
            log_buffer.append(json.dumps(CFG, ensure_ascii=False, indent=2))
            log_buffer.append("")

        row = process_one_file(str(in_path), verbose=True, log_buffer=log_buffer)

        out_csv = out_dir / "batch_summary.csv"
        newf = not out_csv.exists()
        df = pd.DataFrame([row])
        cols = ["filename","chemical_formula","mp_id","halogen_elements","element_count",
                "is_triangular_layered","octahedral_occupancy_count","tetrahedral_occupancy_count",
                "octahedral_occupancy_rate","tetrahedral_occupancy_rate",
                "_diag_frac_pass","_diag_avg_core6_ltcut","_diag_avg_radial_rstd"]
        if newf: df[cols].to_csv(out_csv, index=False)
        else:    df[cols].to_csv(out_csv, mode="a", header=False, index=False)

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in log_buffer))
            f.write("\n")
        print(f"[INFO] Full log saved to {log_file}")

        sys.exit(0 if int(row.get("is_triangular_layered",0))==1 else 2)

    # 目录批处理
    files = []
    for pat in args.patterns:
        files.extend(glob.glob(str(in_path / pat), recursive=True))
    files = sorted(set(files))
    print(f"Total files: {len(files)}")

    if args.n_jobs == 1:
        results = [process_one_file(f, verbose=args.verbose) for f in tqdm(files, desc="Processing", unit="file")]
    else:
        if HAVE_TQDM_JOBLIB:
            with tqdm_joblib(tqdm(total=len(files), desc="Processing", unit="file")):
                results = Parallel(n_jobs=args.n_jobs, prefer="processes", batch_size=64)(
                    delayed(process_one_file)(f, False) for f in files
                )
        else:
            print("[WARN] tqdm_joblib 未安装，降级为普通 tqdm（显示提交进度）")
            results = Parallel(n_jobs=args.n_jobs, prefer="processes", batch_size=64)(
                delayed(process_one_file)(f, False) for f in tqdm(files, desc="Processing", unit="file")
            )

    df = pd.DataFrame(results)
    cols = ["filename","chemical_formula","mp_id","halogen_elements","element_count",
            "is_triangular_layered","octahedral_occupancy_count","tetrahedral_occupancy_count",
            "octahedral_occupancy_rate","tetrahedral_occupancy_rate",
            "_diag_frac_pass","_diag_avg_core6_ltcut","_diag_avg_radial_rstd","error"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols[:-1] + (["error"] if "error" in df.columns else [])]

    out_csv = Path(args.outdir) / "batch_summary_single_stage.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in ["error"] if c in df.columns and df[c].eq("").all()], errors="ignore").to_csv(out_csv, index=False)
    print(f"\nAll done. Results saved in {out_csv}")

if __name__ == "__main__":
    main()
