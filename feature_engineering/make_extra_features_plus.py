#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_extra_features_plus.py

在原始脚本基础上加入：
1) 成分侧增强：卤素混合熵/占比、离子半径失配（M 与 X 分别 wmean/wstd/range_norm）、电荷平衡代理。
2) 几何侧增强：M–X–M 与 X–M–X 键角统计、卤素三角网（层状三角格）评分（平面起伏 RMS + 最近邻三角角度 std）。
3) 图侧增强：在 M–M 投影图上增加平均最短路、全局效率、度同配性、最大 k-core、最大团大小。
4) 对称性：空间群编号 + 晶系编号（1~7）。
5) Voronoi 与配位多样性：CN 熵与 Voronoi 面数统计。
6) “增强包”可选特征：
   - Matminer（Magpie 风格成分特征，mm_magpie_*）
   - DScribe SOAP（mean/std 池化，soap_*）
   - Zeo++ 孔隙占位钩子（zeopp_*；等你有可执行文件后再落地）

保持健壮：即使版本差异/可选依赖缺失，也会返回 NaN 或空字典而不报错。

用法示例：
  python make_extra_features_plus.py \
      --csv folder_score_table.csv --cif_root . \
      --out_csv extra_features.csv --nn crystalnn --graph \
      --n_jobs 8 --compact-stats --impute median --keep-raw

启用增强包（已安装依赖后）：
  python make_extra_features_plus.py \
      --csv folder_score_table.csv --cif_root . \
      --out_csv extra_features_plus.csv --nn crystalnn --graph \
      --mm-magpie --dscribe-soap --soap-rcut 5.0 --soap-nmax 8 --soap-lmax 6 --soap-sigma 0.4 \
      --n_jobs 8 --compact-stats --impute median
"""

from __future__ import annotations
import os, math, argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from functools import lru_cache

from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.dimensionality import get_dimensionality_larsen

# 可选：networkx
try:
    import networkx as nx
except Exception:
    nx = None

# mendeleev 元素属性
try:
    from mendeleev import element as md_element
except Exception:
    raise SystemExit("mendeleev is required. Install with: pip/conda install mendeleev")

# 可选增强包：Matminer
try:
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
    from pymatgen.core.composition import Composition as PmgComposition
    _HAVE_MATM = True
except Exception:
    _HAVE_MATM = False

# 可选增强包：DScribe SOAP
try:
    from dscribe.descriptors import SOAP
    from ase import Atoms  # noqa: F401
    from pymatgen.io.ase import AseAtomsAdaptor
    _HAVE_DSCR = True
except Exception:
    _HAVE_DSCR = False

HALOGENS = {"F", "Cl", "Br", "I"}
OXYGEN_EN = 3.44  # Pauling EN
COMPACT_STATS: bool = False
ARGS = None  # 在 main() 中设为全局，process_row 使用

def _is_finite(x) -> bool:
    try:
        return x is not None and np.isfinite(x)
    except Exception:
        return False

def wavg(values: List[float], weights: List[float]) -> float:
    vs, ws = [], []
    for v, w in zip(values, weights):
        if _is_finite(v) and (w is not None) and (w > 0):
            vs.append(float(v)); ws.append(float(w))
    return float(np.average(vs, weights=ws)) if vs else float("nan")

def stats_pack(values: List[float], prefix: str) -> Dict[str, float]:
    xs = [float(x) for x in values if _is_finite(x)]
    if COMPACT_STATS:
        keys = ["mean","std","min","max"]
    else:
        keys = ["mean","median","std","min","max","range"]
    if not xs:
        return {f"{prefix}_{k}": float("nan") for k in keys}
    out = {
        f"{prefix}_mean": float(np.mean(xs)),
        f"{prefix}_std":  float(np.std(xs)),
        f"{prefix}_min":  float(np.min(xs)),
        f"{prefix}_max":  float(np.max(xs)),
    }
    if not COMPACT_STATS:
        out[f"{prefix}_median"] = float(np.median(xs))
        out[f"{prefix}_range"]  = out[f"{prefix}_max"] - out[f"{prefix}_min"]
    return out

def shannon_entropy(fractions: List[float]) -> float:
    xs = [float(x) for x in fractions if x and x > 0]
    return float(-sum(x * math.log(x) for x in xs)) if xs else 0.0

def _maybe_call(x):
    if callable(x):
        try:
            return x()
        except Exception:
            return None
    return x

@lru_cache(maxsize=None)
def get_elem_props(sym: str) -> Dict[str, Optional[float]]:
    e = md_element(sym)
    en_candidates = [
        getattr(e, "en_pauling", None),
        getattr(e, "electronegativity_pauling", None),
        getattr(e, "en_allen", None),
        getattr(e, "electronegativity_allen", None),
    ]
    en = next((v for v in en_candidates if v is not None), None)
    ox = getattr(e, "oxistates", None) or getattr(e, "oxidation_states", None)
    ox = _maybe_call(ox) or []
    ir = getattr(e, "ionic_radii", None)
    ir = _maybe_call(ir) or []
    ionenergies = getattr(e, "ionenergies", None)
    ionenergies = _maybe_call(ionenergies) or {}
    try:
        ie1 = ionenergies.get(1)
    except Exception:
        ie1 = None
    return {
        "Z": getattr(e, "atomic_number", None),
        "group": getattr(e, "group_id", getattr(e, "group", None)),
        "period": getattr(e, "period", None),
        "X": en,
        "covalent_radius": getattr(e, "covalent_radius_pyykko",
                                   getattr(e, "covalent_radius_bragg", None)),
        "ea": getattr(e, "electron_affinity", None),
        "ie1": ie1,
        "polarizability": getattr(e, "dipole_polarizability", None),
        "atomic_volume": getattr(e, "atomic_volume", None),
        "common_oxi_states": ox,
        "ionic_radii": ir,
    }

def guess_valence(sym: str, props: Dict[str, Any]) -> Optional[int]:
    oxs_raw = props.get("common_oxi_states", []) or []
    oxs_raw = _maybe_call(oxs_raw) or []
    try:
        oxs = list(oxs_raw)
    except Exception:
        oxs = []
    if sym in HALOGENS:
        neg = [o for o in oxs if isinstance(o, (int, float)) and o < 0]
        if neg: return int(sorted(neg, key=lambda x: abs(x))[0])
        return -1
    else:
        pos = [o for o in oxs if isinstance(o, (int, float)) and o > 0]
        if pos: return int(sorted(pos, key=lambda x: abs(x))[0])
        g = props.get("group", None)
        if isinstance(g, int):
            if g in (1, 2): return g
            if 13 <= g <= 18: return g - 10
        return None

def pick_ionic_radius(sym: str, charge: Optional[int], props: Dict[str, Any]) -> Optional[float]:
    items = _maybe_call(props.get("ionic_radii", [])) or []
    try:
        if charge is not None:
            for CN in ("IV","VI","VIII","II","III"):
                for ir in items:
                    ch = getattr(ir, "charge", None)
                    cn = getattr(ir, "coordination", None)
                    rad = getattr(ir, "ionic_radius", None)
                    if ch == charge and cn == CN and _is_finite(rad):
                        return float(rad) * 1e-2  # pm -> Å
            for ir in items:
                if getattr(ir, "charge", None) == charge and _is_finite(getattr(ir, "ionic_radius", None)):
                    return float(ir.ionic_radius) * 1e-2
        for ir in items:
            if getattr(ir, "most_reliable", False) and _is_finite(getattr(ir, "ionic_radius", None)):
                return float(ir.ionic_radius) * 1e-2
        for ir in items:
            if _is_finite(getattr(ir, "ionic_radius", None)):
                return float(ir.ionic_radius) * 1e-2
    except Exception:
        pass
    cr = props.get("covalent_radius", None)
    return float(cr) if _is_finite(cr) else None

# --------------------------
# 成分侧基础 + 增强
# --------------------------
def composition_features(struct: Structure) -> Dict[str, float]:
    comp = struct.composition.fractional_composition
    elems = sorted([(el.symbol, float(frac)) for el, frac in comp.items()], key=lambda x: x[0])
    fracs = [f for _, f in elems]
    props_by_el = {s: get_elem_props(s) for s, _ in elems}

    def pool(prop_key: str) -> Tuple[List[float], List[float]]:
        vals = []
        for s, _ in elems:
            v = props_by_el[s].get(prop_key, None)
            vals.append(v if _is_finite(v) else np.nan)
        return vals, fracs

    features: Dict[str, float] = {
        "feat_n_species": float(len(elems)),
        "feat_entropy": shannon_entropy(fracs),
        "feat_frac_halogens": float(sum(frac for s, frac in elems if s in HALOGENS)),
    }

    for pk, name in [
        ("X", "en"),
        ("covalent_radius", "rcov"),
        ("polarizability", "alpha"),
        ("ea", "ea"),
        ("ie1", "ie1"),
        ("atomic_volume", "atomvol"),
    ]:
        vals, ws = pool(pk)
        features[f"feat_{name}_wmean"] = wavg(vals, ws)
        features.update(stats_pack(vals, f"feat_{name}"))

    # Δχ 与 O 偏好代理
    chi_M = [props_by_el[s].get("X") for s, _ in elems if s not in HALOGENS and _is_finite(props_by_el[s].get("X"))]
    chi_X = [props_by_el[s].get("X") for s, _ in elems if s in HALOGENS and _is_finite(props_by_el[s].get("X"))]
    if chi_M and chi_X:
        features["feat_dchi_MX_mean"] = float(np.mean([abs(m - x) for m in chi_M for x in chi_X]))
        features["feat_dchi_OminusX_M"] = float(np.mean([abs(OXYGEN_EN - m) for m in chi_M])) - \
                                          float(np.mean([abs(x - np.mean(chi_M)) for x in chi_X]))
    else:
        features["feat_dchi_MX_mean"] = float("nan")
        features["feat_dchi_OminusX_M"] = float("nan")

    # 场强 <|z|/r^2>
    num, den = 0.0, 0.0
    for s, frac in elems:
        props = props_by_el[s]
        val = guess_valence(s, props)
        rion = pick_ionic_radius(s, val, props)
        if s not in HALOGENS and (val is not None) and _is_finite(rion) and rion > 0:
            num += frac * (abs(val) / (rion ** 2))
            den += frac
    features["feat_field_strength_mean"] = (num / den) if den > 0 else float("nan")
    return features

def halogen_mixing_entropy(struct: Structure) -> Dict[str, float]:
    comp = struct.composition.fractional_composition
    hal_fracs = [float(frac) for el, frac in comp.items() if el.symbol in HALOGENS]
    s = shannon_entropy(hal_fracs) if hal_fracs else 0.0
    total_hal = sum(hal_fracs)
    total_m = 1.0 - total_hal
    ratio = float(total_hal / total_m) if total_m > 0 else float("nan")
    return {"feat_X_mixing_entropy": float(s),
            "feat_X_frac": float(total_hal),
            "feat_X_over_M": ratio}

def ionic_radius_mismatch(struct: Structure) -> Dict[str, float]:
    comp = struct.composition.fractional_composition
    elems = [(el.symbol, float(frac)) for el, frac in comp.items()]
    vals_M, vals_X, wM, wX = [], [], [], []
    for sym, f in elems:
        props = get_elem_props(sym)
        val = guess_valence(sym, props)
        rion = pick_ionic_radius(sym, val, props)
        if _is_finite(rion):
            if sym in HALOGENS:
                vals_X.append(rion); wX.append(f)
            else:
                vals_M.append(rion); wM.append(f)
    def _stats(vs: List[float], ws: List[float], prefix: str) -> Dict[str, float]:
        if not vs:
            return {f"{prefix}_wmean": float("nan"),
                    f"{prefix}_wstd": float("nan"),
                    f"{prefix}_range_norm": float("nan")}
        wmean = wavg(vs, ws)
        variance = np.average((np.array(vs)-wmean)**2, weights=ws)
        wstd = float(math.sqrt(variance))
        rrange = (max(vs)-min(vs))/wmean if wmean and _is_finite(wmean) else float("nan")
        return {f"{prefix}_wmean": wmean, f"{prefix}_wstd": wstd, f"{prefix}_range_norm": rrange}
    out: Dict[str, float] = {}
    out.update(_stats(vals_M, wM, "feat_rion_M"))
    out.update(_stats(vals_X, wX, "feat_rion_X"))
    return out

def charge_balance_proxy(struct: Structure) -> Dict[str, float]:
    comp = struct.composition.fractional_composition
    total = 0.0
    for el, frac in comp.items():
        sym = el.symbol
        props = get_elem_props(sym)
        val = guess_valence(sym, props)
        if val is not None:
            total += float(frac) * float(val)
    return {"feat_charge_balance_abs": float(abs(total))}

# --------------------------
# 结构侧：构网/距离/角度/三角网/图论/对称性/Voronoi
# --------------------------
def build_bonded(struct: Structure, method: str = "crystalnn") -> Tuple[StructureGraph, Any]:
    if method == "crystalnn":
        try:
            nn = CrystalNN()
            sg = nn.get_bonded_structure(struct)
            return sg, nn
        except Exception:
            pass
    nn = VoronoiNN(cutoff=10.0)
    sg = nn.get_bonded_structure(struct)
    return sg, nn

def _to_nx_graph(sg: StructureGraph):
    if nx is None: return None
    if hasattr(sg, "as_graph"):
        try: return sg.as_graph()
        except Exception: pass
    if hasattr(sg, "graph") and sg.graph is not None:
        try: return nx.Graph(sg.graph)
        except Exception:
            try: return nx.Graph(nx.MultiGraph(sg.graph))
            except Exception: return None
    return None

def _safe_getattr(obj: Any, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n): return getattr(obj, n)
    return default

def _neighbor_distance(struct: Structure, i: int, conn_site) -> Optional[float]:
    site_obj = _safe_getattr(conn_site, ["site", "to_site", "neighbor"], None)
    if site_obj is not None:
        try: return float(struct[i].distance(site_obj))
        except Exception: pass
    j = _safe_getattr(conn_site, ["index","j","to_index"], None)
    if j is not None:
        try: return float(struct.get_distance(i, int(j)))
        except Exception: pass
    w = _safe_getattr(conn_site, ["weight","nn_distance"], None)
    if _is_finite(w): return float(w)
    return None

def cn_and_bonds(struct: Structure, sg: StructureGraph) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cn_all, cn_cation, cn_hal = [], [], []
    bond_lengths_MX, bond_lengths_MX_norm = [], []
    symbols = [site.specie.symbol for site in struct.sites]
    covr = {sym: get_elem_props(sym).get("covalent_radius", None) for sym in set(symbols)}
    for i, site in enumerate(struct.sites):
        try:
            neighs = sg.get_connected_sites(i)
        except Exception:
            neighs = []
        cn_i = len(neighs)
        cn_all.append(cn_i)
        sym_i = site.specie.symbol
        (cn_hal if sym_i in HALOGENS else cn_cation).append(cn_i)
        for cs in neighs:
            js = _safe_getattr(cs, ["site","to_site","neighbor"], None)
            j_sym = js.specie.symbol if js is not None else None
            if j_sym is None: continue
            if (sym_i in HALOGENS) ^ (j_sym in HALOGENS):
                d = _neighbor_distance(struct, i, cs)
                if _is_finite(d):
                    bond_lengths_MX.append(float(d))
                    rc_sum = (covr.get(sym_i) or 0.0) + (covr.get(j_sym) or 0.0)
                    bond_lengths_MX_norm.append(float(d) / rc_sum if rc_sum and rc_sum > 1e-8 else float("nan"))
    out.update(stats_pack(cn_all, "feat_cn"))
    if cn_cation: out.update(stats_pack(cn_cation, "feat_cn_cation"))
    else:
        for k in (["mean","std","min","max"] if COMPACT_STATS else ["mean","median","std","min","max","range"]):
            out[f"feat_cn_cation_{k}"] = float("nan")
    if cn_hal: out.update(stats_pack(cn_hal, "feat_cn_halogen"))
    else:
        for k in (["mean","std","min","max"] if COMPACT_STATS else ["mean","median","std","min","max","range"]):
            out[f"feat_cn_halogen_{k}"] = float("nan")
    out.update(stats_pack(bond_lengths_MX, "feat_bond_MX"))
    out.update(stats_pack(bond_lengths_MX_norm, "feat_bond_MX_norm"))
    return out

def _angle_from_vectors(struct: Structure, va, vb) -> float:
    A = struct.lattice.matrix
    va = A.dot(np.array(va)); vb = A.dot(np.array(vb))
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    if na < 1e-8 or nb < 1e-8: return float("nan")
    cos = np.clip(np.dot(va, vb)/(na*nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def angle_stats_MXM_XMX(struct: Structure, sg: StructureGraph) -> Dict[str, float]:
    MXM, XMX = [], []
    symbols = [site.specie.symbol for site in struct.sites]
    for i in range(len(struct)):
        si = symbols[i]
        try:
            neighs_i = sg.get_connected_sites(i)
        except Exception:
            continue
        if si in HALOGENS:
            m_neighbors = []
            for cs in neighs_i:
                j = _safe_getattr(cs, ["index","j","to_index"], None)
                if j is not None and symbols[int(j)] not in HALOGENS:
                    m_neighbors.append(int(j))
            for a in range(len(m_neighbors)):
                for b in range(a+1, len(m_neighbors)):
                    j, k = m_neighbors[a], m_neighbors[b]
                    va = struct[j].frac_coords - struct[i].frac_coords
                    vb = struct[k].frac_coords - struct[i].frac_coords
                    ang = _angle_from_vectors(struct, va, vb)
                    if _is_finite(ang): MXM.append(ang)
        else:
            x_neighbors = []
            for cs in neighs_i:
                j = _safe_getattr(cs, ["index","j","to_index"], None)
                if j is not None and symbols[int(j)] in HALOGENS:
                    x_neighbors.append(int(j))
            for a in range(len(x_neighbors)):
                for b in range(a+1, len(x_neighbors)):
                    j, k = x_neighbors[a], x_neighbors[b]
                    va = struct[j].frac_coords - struct[i].frac_coords
                    vb = struct[k].frac_coords - struct[i].frac_coords
                    ang = _angle_from_vectors(struct, va, vb)
                    if _is_finite(ang): XMX.append(ang)
    out = {}
    out.update(stats_pack(MXM, "feat_angle_MXM"))
    out.update(stats_pack(XMX, "feat_angle_XMX"))
    return out

def halogen_trinet_score(struct: Structure, sg: StructureGraph) -> Dict[str, float]:
    X_idx = [i for i, site in enumerate(struct) if site.specie.symbol in HALOGENS]
    if len(X_idx) < 6:
        return {"feat_trinet_planarity_rms": float("nan"),
                "feat_trinet_triangle_angle_std": float("nan")}
    coords = np.array([struct[i].coords for i in X_idx])
    coords_c = coords - coords.mean(axis=0)
    U, S, Vt = np.linalg.svd(coords_c, full_matrices=False)
    normal = Vt[2, :]
    rms = float(np.sqrt(np.mean((coords_c.dot(normal))**2)))
    tri_angles = []
    symbols = [site.specie.symbol for site in struct.sites]
    for i in X_idx:
        try:
            neighs = sg.get_connected_sites(i)
        except Exception:
            continue
        xneis = []
        for cs in neighs:
            j = _safe_getattr(cs, ["index","j","to_index"], None)
            if j is None: continue
            if symbols[int(j)] in HALOGENS:
                d = _neighbor_distance(struct, i, cs)
                if _is_finite(d): xneis.append((int(j), float(d)))
        if len(xneis) >= 2:
            xneis.sort(key=lambda t: t[1])
            j, k = xneis[0][0], xneis[1][0]
            ang = _angle_from_vectors(struct,
                                      struct[j].frac_coords - struct[i].frac_coords,
                                      struct[k].frac_coords - struct[i].frac_coords)
            if _is_finite(ang): tri_angles.append(ang)
    return {"feat_trinet_planarity_rms": rms,
            "feat_trinet_triangle_angle_std": float(np.std(tri_angles)) if tri_angles else float("nan")}

def project_M_graph(struct: Structure, sg: StructureGraph):
    if nx is None: return None
    G = _to_nx_graph(sg)
    if G is None: return None
    sym = {i: struct.sites[i].specie.symbol for i in range(len(struct))}
    M_nodes = [i for i in G.nodes if sym[i] not in HALOGENS]
    X_nodes = [i for i in G.nodes if sym[i] in HALOGENS]
    GM = nx.Graph(); GM.add_nodes_from(M_nodes)
    for x in X_nodes:
        neigh_M = [n for n in G.neighbors(x) if sym[n] not in HALOGENS]
        for a in range(len(neigh_M)):
            for b in range(a+1, len(neigh_M)):
                i, j = neigh_M[a], neigh_M[b]
                if GM.has_edge(i, j):
                    GM[i][j]["w"] += 1
                else:
                    GM.add_edge(i, j, w=1)
    return GM

def graph_rich_features(struct: Structure, sg: StructureGraph) -> Dict[str, float]:
    out: Dict[str, float] = {}
    GM = project_M_graph(struct, sg)
    keys = ["deg_mean","deg_std","clust","spectrum_max","ncc","diam","cycle3","cycle4"]
    if GM is None or GM.number_of_nodes() == 0:
        for k in keys: out[f"feat_Mproj_{k}"] = float("nan")
        return out
    degs = [d for _, d in GM.degree()]
    out["feat_Mproj_deg_mean"] = float(np.mean(degs)) if degs else float("nan")
    out["feat_Mproj_deg_std"]  = float(np.std(degs)) if degs else float("nan")
    try:
        out["feat_Mproj_clust"] = float(nx.average_clustering(GM))
    except Exception:
        out["feat_Mproj_clust"] = float("nan")
    try:
        A = nx.to_numpy_array(GM, weight="w")
        out["feat_Mproj_spectrum_max"] = float(np.max(np.linalg.eigvalsh(A))) if A.size else float("nan")
    except Exception:
        out["feat_Mproj_spectrum_max"] = float("nan")
    try:
        ccs = list(nx.connected_components(GM))
        out["feat_Mproj_ncc"] = float(len(ccs))
        Gbig = GM.subgraph(max(ccs, key=len)).copy() if ccs else GM
        out["feat_Mproj_diam"] = float(nx.diameter(Gbig)) if Gbig.number_of_nodes() > 1 else 0.0
    except Exception:
        out["feat_Mproj_ncc"] = float("nan"); out["feat_Mproj_diam"] = float("nan")
    try:
        lens = [len(c) for c in nx.cycle_basis(GM)]
        out["feat_Mproj_cycle3"] = float(sum(1 for l in lens if l == 3))
        out["feat_Mproj_cycle4"] = float(sum(1 for l in lens if l == 4))
    except Exception:
        out["feat_Mproj_cycle3"] = float("nan"); out["feat_Mproj_cycle4"] = float("nan")
    return out

def graph_more_features(struct: Structure, sg: StructureGraph) -> Dict[str, float]:
    if nx is None:
        return {"feat_Mproj_asp": float("nan"), "feat_Mproj_eff": float("nan"),
                "feat_Mproj_assort": float("nan"), "feat_Mproj_kcore": float("nan"),
                "feat_Mproj_clique_max": float("nan")}
    GM = project_M_graph(struct, sg)
    if GM is None or GM.number_of_nodes() == 0:
        return {"feat_Mproj_asp": float("nan"), "feat_Mproj_eff": float("nan"),
                "feat_Mproj_assort": float("nan"), "feat_Mproj_kcore": float("nan"),
                "feat_Mproj_clique_max": float("nan")}
    out: Dict[str, float] = {}
    try:
        ccs = list(nx.connected_components(GM))
        Gbig = GM.subgraph(max(ccs, key=len)).copy() if ccs else GM
        out["feat_Mproj_asp"] = float(nx.average_shortest_path_length(Gbig)) if Gbig.number_of_edges() > 0 and Gbig.number_of_nodes() > 1 else 0.0
    except Exception:
        out["feat_Mproj_asp"] = float("nan")
    try:
        out["feat_Mproj_eff"] = float(nx.global_efficiency(GM))
    except Exception:
        out["feat_Mproj_eff"] = float("nan")
    try:
        out["feat_Mproj_assort"] = float(nx.degree_assortativity_coefficient(GM))
    except Exception:
        out["feat_Mproj_assort"] = float("nan")
    try:
        core = nx.core_number(GM)
        out["feat_Mproj_kcore"] = float(max(core.values()) if core else 0)
    except Exception:
        out["feat_Mproj_kcore"] = float("nan")
    try:
        out["feat_Mproj_clique_max"] = float(len(max(nx.find_cliques(GM), key=len)))
    except Exception:
        out["feat_Mproj_clique_max"] = float("nan")
    return out

def symmetry_features(struct: Structure) -> Dict[str, float]:
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except Exception:
        return {"feat_sg_number": float("nan"), "feat_crystal_system_id": float("nan")}
    try:
        sga = SpacegroupAnalyzer(struct, symprec=1e-2, angle_tolerance=5)
        sgnum = sga.get_space_group_number()
        system = sga.get_crystal_system()
        systems = ["triclinic","monoclinic","orthorhombic","tetragonal","trigonal","hexagonal","cubic"]
        cs_id = float(systems.index(system)+1) if system in systems else float("nan")
        return {"feat_sg_number": float(sgnum), "feat_crystal_system_id": cs_id}
    except Exception:
        return {"feat_sg_number": float("nan"), "feat_crystal_system_id": float("nan")}

def voronoi_and_cn_diversity(struct: Structure) -> Dict[str, float]:
    try:
        vnn = VoronoiNN(cutoff=10.0)
    except Exception:
        return {"feat_cn_entropy": float("nan"),
                "feat_voro_face_mean": float("nan"),
                "feat_voro_face_std": float("nan"),
                "feat_voro_face_min": float("nan"),
                "feat_voro_face_max": float("nan")}
    CNs, face_counts = [], []
    for i in range(len(struct)):
        try:
            ns = vnn.get_nn_info(struct, i)
        except Exception:
            continue
        CNs.append(len(ns)); face_counts.append(len(ns))
    if CNs:
        hist = {}
        for c in CNs: hist[c] = hist.get(c, 0) + 1
        fracs = [v/len(CNs) for v in hist.values()]
        H = shannon_entropy(fracs)
    else:
        H = float("nan")
    out = {"feat_cn_entropy": float(H)}
    out.update(stats_pack(face_counts, "feat_voro_face"))
    return out

def structural_features(struct: Structure, nn_method: str = "crystalnn", do_graph: bool = True) -> Dict[str, float]:
    feats: Dict[str, float] = {
        "feat_n_sites": float(len(struct)),
        "feat_density": float(getattr(struct, "density", float("nan"))),
        "feat_volume": float(struct.volume),
        "feat_volume_per_atom": float(struct.volume / len(struct)) if len(struct) > 0 else float("nan"),
    }
    try:
        sg_tmp, _ = build_bonded(struct, nn_method)
        feats["feat_dim_larsen"] = int(get_dimensionality_larsen(sg_tmp))
    except Exception:
        feats["feat_dim_larsen"] = float("nan"); sg_tmp = None
    if sg_tmp is None:
        sg_tmp, _ = build_bonded(struct, nn_method)

    feats.update(cn_and_bonds(struct, sg_tmp))
    if do_graph:
        feats.update(graph_rich_features(struct, sg_tmp))
        feats.update(graph_more_features(struct, sg_tmp))

    try:
        feats.update(angle_stats_MXM_XMX(struct, sg_tmp))
    except Exception:
        pass
    try:
        feats.update(halogen_trinet_score(struct, sg_tmp))
    except Exception:
        feats["feat_trinet_planarity_rms"] = float("nan")
        feats["feat_trinet_triangle_angle_std"] = float("nan")

    feats.update(symmetry_features(struct))
    feats.update(voronoi_and_cn_diversity(struct))
    return feats

# --------------------------
# 可选增强包：Matminer / DScribe / Zeo++ 占位
# --------------------------
def matminer_magpie_features(struct: Structure) -> Dict[str, float]:
    if not _HAVE_MATM:
        return {}
    try:
        comp = PmgComposition(struct.composition.reduced_composition.alphabetical_formula)
        feats: Dict[str, float] = {}
        ep = ElementProperty(features=[
            "Number","MendeleevNumber","AtomicWeight",
            "CovalentRadius","Electronegativity","ElectronAffinity",
            "FusionEnthalpy","ThermalConductivity","BoilingT",
        ], stats=["mean","avg_dev","max","min","range"])
        st = Stoichiometry(p_list=(0,2,3))
        vo = ValenceOrbital(props=["s","p","d","f"], stats=["sum","frac"])
        def _safe_append(prefix, keys, values):
            for k, v in zip(keys, values):
                feats[f"{prefix}_{k}"] = float(v) if _is_finite(v) else float("nan")
        ep_vals = ep.featurize(comp); _safe_append("mm_magpie_EP", ep.feature_labels(), ep_vals)
        st_vals = st.featurize(comp); _safe_append("mm_magpie_ST", st.feature_labels(), st_vals)
        vo_vals = vo.featurize(comp); _safe_append("mm_magpie_VO", vo.feature_labels(), vo_vals)
        return feats
    except Exception:
        return {}

def dscribe_soap_features(struct: Structure, rcut: float, nmax: int, lmax: int, sigma: float) -> Dict[str, float]:
    if not _HAVE_DSCR:
        return {}
    try:
        species = sorted({site.specie.symbol for site in struct.sites})
        atoms = AseAtomsAdaptor.get_atoms(struct)
        desc = SOAP(species=species, periodic=True, rcut=rcut, nmax=nmax, lmax=lmax,
                    sigma=sigma, sparse=False, average=False)
        X = desc.create(atoms)  # (n_atoms, n_feat)
        if X is None or X.size == 0:
            return {}
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
        out: Dict[str, float] = {}
        for i, v in enumerate(mu): out[f"soap_mean_{i}"] = float(v) if _is_finite(v) else float("nan")
        for i, v in enumerate(sd): out[f"soap_std_{i}"]  = float(v) if _is_finite(v) else float("nan")
        out["soap_mean_l2"] = float(np.linalg.norm(mu))
        out["soap_std_l2"]  = float(np.linalg.norm(sd))
        return out
    except Exception:
        return {}

def zeopp_porosity_stub(struct: Structure, zeopp_path: str, probe: float) -> Dict[str, float]:
    if not zeopp_path:
        return {"zeopp_P0": float("nan"), "zeopp_PF": float("nan"),
                "zeopp_Di": float("nan"), "zeopp_Df": float("nan")}
    # TODO: 你在集群上装好 zeo++ 后，在此处调用 subprocess 解析输出
    return {"zeopp_P0": float("nan"), "zeopp_PF": float("nan"),
            "zeopp_Di": float("nan"), "zeopp_Df": float("nan")}

# --------------------------
# 行处理与后处理
# --------------------------
def structural_and_optional_features(struct: Structure, nn: str, do_graph: bool) -> Dict[str, float]:
    out = structural_features(struct, nn_method=nn, do_graph=do_graph)

    # 可选增强包
    if getattr(ARGS, "mm_magpie", False):
        out.update(matminer_magpie_features(struct))
    if getattr(ARGS, "dscribe_soap", False):
        out.update(dscribe_soap_features(struct,
                rcut=getattr(ARGS, "soap_rcut", 5.0),
                nmax=getattr(ARGS, "soap_nmax", 8),
                lmax=getattr(ARGS, "soap_lmax", 6),
                sigma=getattr(ARGS, "soap_sigma", 0.4)))
    zp = getattr(ARGS, "zeopp_path", "")
    if zp is not None:
        out.update(zeopp_porosity_stub(struct, zp, probe=getattr(ARGS, "zeopp_probe", 1.2)))
    return out

def process_row(row_dict: Dict[str, Any], cif_root: str, nn: str, do_graph: bool) -> Dict[str, float]:
    path_in = str(row_dict["cif_path"])
    path = path_in if os.path.isabs(path_in) else os.path.join(cif_root, path_in)
    out: Dict[str, float] = {}
    try:
        s = Structure.from_file(path)
    except Exception as e:
        out["feat_parse_error"] = 1.0
        out["feat_parse_error_msg"] = str(e)
        return out
    # 成分侧（基础 + 增强）
    out.update(composition_features(s))
    out.update(halogen_mixing_entropy(s))
    out.update(ionic_radius_mismatch(s))
    out.update(charge_balance_proxy(s))
    # 结构侧（基础 + 图 + 角度 + 三角网 + 对称性 + Voronoi + 可选增强包）
    out.update(structural_and_optional_features(s, nn, do_graph))
    return out

def postprocess_features(df_in: pd.DataFrame, drop_constant: bool = True, const_thresh: int = 1) -> Tuple[pd.DataFrame, List[str]]:
    df = df_in.copy()
    dropped: List[str] = []
    if drop_constant:
        nunq = df.nunique(dropna=False)
        to_drop = [c for c, n in nunq.items() if n <= const_thresh]
        if to_drop:
            df = df.drop(columns=to_drop, errors="ignore"); dropped.extend(to_drop)
    return df, dropped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with at least cif_path column")
    ap.add_argument("--cif_root", default=".", help="Root dir for CIFs if cif_path is relative")
    ap.add_argument("--out_csv", default="extra_features.csv", help="Output CSV path")
    ap.add_argument("--nn", choices=["crystalnn", "voronoi"], default="crystalnn", help="NN method for bonding")
    ap.add_argument("--graph", action="store_true", help="Also compute networkx graph features")

    # 通用优化
    ap.add_argument("--compact-stats", action="store_true",
                    help="Only output mean/std/min/max for stats (omit median and range)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Save an additional .raw.csv file containing unprocessed features")
    ap.add_argument("--impute", choices=["none","median","zero"], default="none",
                    help="Impute missing numeric features and create NA indicator columns")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Number of parallel jobs for processing; >1 enables joblib parallelism")

    # 可选增强包开关
    ap.add_argument("--mm-magpie", action="store_true",
                    help="Enable matminer Magpie-style composition features (pooled).")
    ap.add_argument("--dscribe-soap", action="store_true",
                    help="Enable DScribe SOAP descriptor with mean/std pooling.")
    ap.add_argument("--soap-rcut", type=float, default=5.0, help="SOAP cutoff radius (Å).")
    ap.add_argument("--soap-nmax", type=int, default=8, help="SOAP radial basis nmax.")
    ap.add_argument("--soap-lmax", type=int, default=6, help="SOAP angular momentum lmax.")
    ap.add_argument("--soap-sigma", type=float, default=0.4, help="SOAP Gaussian smearing width.")
    ap.add_argument("--zeopp-path", type=str, default="",
                    help="Path to zeo++ binaries (optional; if set, enable porosity hook).")
    ap.add_argument("--zeopp-probe", type=float, default=1.2,
                    help="Probe radius for Zeo++ (Å), e.g., 1.2 for He.")

    args = ap.parse_args()

    # 设全局 ARGS / 统计开关
    global ARGS, COMPACT_STATS
    ARGS = args
    COMPACT_STATS = bool(args.compact_stats)

    # 读 CSV
    df = pd.read_csv(args.csv)
    rows_in = df.to_dict("records")

    # 并行处理
    rows: List[Dict[str, float]] = []
    if args.n_jobs and args.n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            def _worker(rdict: Dict[str, Any]) -> Dict[str, float]:
                return process_row(rdict, args.cif_root, args.nn, args.graph)
            rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
                delayed(_worker)(rd) for rd in rows_in
            )
        except Exception as e:
            print(f"[WARN] joblib parallelism failed, falling back to serial: {e}")
            rows = [process_row(r, args.cif_root, args.nn, args.graph) for r in rows_in]
    else:
        rows = [process_row(r, args.cif_root, args.nn, args.graph) for r in rows_in]

    # 合并与保存
    fdf = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), fdf.reset_index(drop=True)], axis=1)

    if args.keep_raw:
        raw_path = os.path.splitext(args.out_csv)[0] + ".raw.csv"
        out.to_csv(raw_path, index=False)

    out, dropped_cols = postprocess_features(out, drop_constant=True, const_thresh=1)
    if dropped_cols:
        print(f"[INFO] Dropped constant/near-constant columns: {len(dropped_cols)}")

    if args.impute != "none":
        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        if "score" in num_cols:  # 保护你的目标列
            num_cols.remove("score")
        for c in num_cols:
            if out[c].isna().any():
                out[f"{c}_isna"] = out[c].isna().astype(np.uint8)
        if args.impute == "median":
            med = out[num_cols].median(numeric_only=True)
            out[num_cols] = out[num_cols].fillna(med)
        elif args.impute == "zero":
            out[num_cols] = out[num_cols].fillna(0.0)

    out.to_csv(args.out_csv, index=False)
    new_cols = out.shape[1] - df.shape[1]
    print(f"[OK] Wrote {args.out_csv} with {new_cols} new feature columns (after cleaning)")

if __name__ == "__main__":
    main()
