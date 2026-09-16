#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_features_renamed.py  (full, physics-meaningful mix, extended mendeleev)

升级要点：
- 新命名：<domain>_<source>_<object>_<property>_<op>[_<stat>][_<norm>][_<method>][__<unit>]
- ID 列置前：id_cif_path, id_cif, id_score, id_dim, id_connect
- 并行稳健：标准库 concurrent.futures.ProcessPoolExecutor，限制子进程 BLAS/OMP 线程，失败自动回退串行

功能概览：
- 组成侧（mendeleev/Magpie + M/H 分组 + 差/比 + Sanderson/C6 几何均值等）
- 结构侧（配位、键、角、图 M 投影、对称、Voronoi、packing）
- Matminer（EP/ST/VO/EF/BC，RDF/ADF/CM/SCM/OFM/Ewald/BoB/XRD，可选自动枚举）
- DScribe（SOAP/ACSF/MBTR 均值/方差与 L2 范数）
"""

from __future__ import annotations
import os
import re
import math
import argparse
import inspect
import importlib
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.dimensionality import get_dimensionality_larsen

# ====== 并行（最常见方式）======
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================ 可选依赖 ================
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None  # type: ignore

try:
    from mendeleev import element as md_element
except Exception:
    raise SystemExit("ERROR: 请先安装 mendeleev：pip/conda install mendeleev")

_HAVE_MATM = True
try:
    from pymatgen.core.composition import Composition as PmgComposition
    from pymatgen.core.periodic_table import Element as PmgElement
    from matminer.featurizers.base import BaseFeaturizer
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, ValenceOrbital, ElementFraction, BandCenter,
    )
    from matminer.featurizers.structure import (
        SiteStatsFingerprint,
        OPSiteFingerprint,
        RadialDistributionFunction,
        AngularDistributionFunction,
        DensityFeatures,
        XRDPowderPattern,
        BagofBonds,
    )
    from matminer.featurizers.structure.sites import CoordinationNumber
    from matminer.featurizers.structure.matrix import (
        CoulombMatrix,
        SineCoulombMatrix,
        EwaldEnergy,
        OrbitalFieldMatrix,
    )
except Exception:
    _HAVE_MATM = False

_HAVE_MAGPIE = True
try:
    from matminer.utils.data import MagpieData
    _MAG = MagpieData(impute_nan=True)
except Exception:
    _HAVE_MAGPIE = False
    _MAG = None  # type: ignore

_HAVE_DSCR = True
try:
    from dscribe.descriptors import SOAP, ACSF, MBTR
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:
    _HAVE_DSCR = False

# ================ 工具函数与全局 ================
HALOGENS = {"F", "Cl", "Br", "I"}
OXYGEN_EN = 3.44  # Pauling
ARGS = None


def _is_finite(x: Any) -> bool:
    try:
        return x is not None and np.isfinite(x)
    except Exception:
        return False


def _maybe_call(x: Any) -> Any:
    if callable(x):
        try:
            return x()
        except Exception:
            return None
    return x


def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df, "columns") and getattr(df.columns, "duplicated", None) is not None:
        if df.columns.duplicated().any():
            return df.T.groupby(level=0).first().T
    return df


def wavg(values: List[float], weights: List[float]) -> float:
    vs, ws = [], []
    for v, w in zip(values, weights):
        if _is_finite(v) and w and w > 0:
            vs.append(float(v)); ws.append(float(w))
    return float(np.average(vs, weights=ws)) if vs else float("nan")


def stats_pack(values: List[float], prefix: str, compact: bool = False) -> Dict[str, float]:
    xs = [float(x) for x in values if _is_finite(x)]
    if not xs:
        out: Dict[str, float] = {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_median": np.nan,
        }
        if not compact:
            out.update({f"{prefix}_min": np.nan, f"{prefix}_max": np.nan, f"{prefix}_range": np.nan})
        return out
    out: Dict[str, float] = {
        f"{prefix}_mean": float(np.mean(xs)),
        f"{prefix}_std": float(np.std(xs)),
        f"{prefix}_median": float(np.median(xs)),
    }
    if not compact:
        mn, mx = float(np.min(xs)), float(np.max(xs))
        out[f"{prefix}_min"] = mn; out[f"{prefix}_max"] = mx; out[f"{prefix}_range"] = mx - mn
    return out


def shannon_entropy(fracs: List[float]) -> float:
    xs = [float(x) for x in fracs if x and x > 0]
    return float(-sum(x * math.log(x) for x in xs)) if xs else 0.0


# ================ mendeleev：动态数值属性 & 别名 ================
NUMERIC_EXCLUDES = {
    "is_radioactive","ionic_radii","oxistates","oxidation_states","is_noble_gas",
    "all_isotopes","stable_isotopes","unstable_isotopes","cpk_color","name","symbol",
    "ec","electronic_configuration","electron_configuration","tag","series","block","long_name",
}


def _to_float(v: Any) -> Optional[float]:
    try:
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return None


@lru_cache(maxsize=None)
def get_elem_numeric_props(sym: str) -> Dict[str, Optional[float]]:
    e = md_element(sym)
    out: Dict[str, Optional[float]] = {}
    for name in dir(e):
        if name.startswith("_") or name in NUMERIC_EXCLUDES:
            continue
        try:
            val = _maybe_call(getattr(e, name))
        except Exception:
            continue
        nv = _to_float(val)
        if nv is not None:
            out[name] = nv

    try:
        ion = _maybe_call(getattr(e, "ionenergies", None)) or {}
        if isinstance(ion, dict) and 1 in ion:
            nv = _to_float(ion.get(1))
            if nv is not None:
                out["ionization_energy_first"] = nv
    except Exception:
        pass

    def _get_en(scale_attr: Optional[str] = None, scale_call: Optional[str] = None):
        if scale_attr and scale_attr in out and _is_finite(out.get(scale_attr)):
            return float(out[scale_attr])
        if scale_call:
            try:
                val = e.electronegativity(scale_call)  # type: ignore
                return float(val) if _is_finite(val) else None
            except Exception:
                return None
        return None

    chiP = _get_en("electronegativity_pauling", "pauling")
    chiA = _get_en("electronegativity_allen", "allen")
    chiAR = _get_en("electronegativity_allred_rochow", "allred-rochow")
    chiMB = _get_en("electronegativity_martynov_batsanov", "martynov-batsanov")
    chiNag = _get_en("electronegativity_nagle", "nagle")
    chiMul = _get_en("electronegativity_mulliken", "mulliken")
    chiSand = _get_en("electronegativity_sanderson", "sanderson")
    chiGhosh = _get_en("electronegativity_ghosh", "ghosh")
    chiGordy = _get_en("electronegativity_gordy", "gordy")
    chiLX = _get_en("electronegativity_li_xue", "li-xue")
    chiCS = _get_en("electronegativity_cottrell_sutton", "cottrell-sutton")
    chiGL = _get_en("en_gunnarsson_lundqvist", None)
    chiMied = _get_en("en_miedema", None)
    chiMullay = _get_en("en_mullay", None)
    chiRB = _get_en("en_robles_bartolotti", None)

    for k, v in {
        "chiP": chiP, "chiA": chiA, "chiAR": chiAR, "chiMB": chiMB, "chiNag": chiNag,
        "chiMul": chiMul, "chiSand": chiSand, "chiGhosh": chiGhosh, "chiGordy": chiGordy,
        "chiLX": chiLX, "chiCS": chiCS, "chiGL": chiGL, "chiMied": chiMied, "chiMullay": chiMullay, "chiRB": chiRB,
    }.items():
        if _is_finite(v):
            out[k] = float(v)

    if _is_finite(chiP):
        out["en"] = float(chiP)
    elif _is_finite(chiA):
        out["en"] = float(chiA)

    cov_candidates = [
        out.get("covalent_radius_pyykko"),
        out.get("covalent_radius_cordero"),
        out.get("covalent_radius_bragg"),
    ]
    out["covalent_radius"] = next((v for v in cov_candidates if _is_finite(v)), out.get("covalent_radius", None))
    if _is_finite(out.get("covalent_radius_cordero")):
        out["r_cov_cordero"] = float(out["covalent_radius_cordero"])  # type: ignore[index]
    if _is_finite(out.get("covalent_radius_pyykko_double")):
        out["r_cov_pyykko_double"] = float(out["covalent_radius_pyykko_double"])  # type: ignore[index]
    if _is_finite(out.get("covalent_radius_pyykko_triple")):
        out["r_cov_pyykko_triple"] = float(out["covalent_radius_pyykko_triple"])  # type: ignore[index]

    vdw_candidates = [
        out.get("vdw_radius_alvarez"),
        out.get("vdw_radius_bondi"),
        out.get("vdw_radius_batsanov"),
        out.get("vdw_radius"),
    ]
    out["vdw_radius_best"] = next((v for v in vdw_candidates if _is_finite(v)), out.get("vdw_radius_best", None))
    if _is_finite(out.get("vdw_radius_alvarez")):
        out["r_vdw_alvarez"] = float(out["vdw_radius_alvarez"])  # type: ignore[index]
    if _is_finite(out.get("vdw_radius_bondi")):
        out["r_vdw_bondi"] = float(out["vdw_radius_bondi"])  # type: ignore[index]
    if _is_finite(out.get("vdw_radius_batsanov")):
        out["r_vdw_batsanov"] = float(out["vdw_radius_batsanov"])  # type: ignore[index]

    if _is_finite(out.get("metallic_radius_c12")):
        out["r_met_c12"] = float(out["metallic_radius_c12"])  # type: ignore[index]

    EA = out.get("electron_affinity")
    IE1 = out.get("ionization_energy_first")
    if _is_finite(EA) and _is_finite(IE1):
        chiM = 0.5 * (float(IE1) + float(EA))
        eta = 0.5 * (float(IE1) - float(EA))
        out["chiM"] = chiM; out["eta"] = eta
        if eta and abs(eta) > 1e-12:
            out["omega"] = (chiM * chiM) / (2.0 * eta)

    for src_key, alias in [
        ("electrophilicity", "omega_parr"), ("hardness", "eta_parr"), ("softness", "soft_parr"),
        ("proton_affinity", "PA"), ("miedema_electron_density", "miedema_rho_e"), ("miedema_molar_volume", "miedema_V_m"),
        ("glawe_number", "GN"), ("pettifor_number", "PN"),
        ("abundance_crust", "abund_crust"), ("abundance_sea", "abund_sea"),
        ("price_per_kg", "price_per_kg"), ("production_concentration", "prod_conc"),
        ("reserve_distribution", "resv_dist"), ("recycling_rate", "recycle_rate"),
        ("relative_supply_risk", "supply_risk"),
    ]:
        v = out.get(src_key, None)
        if _is_finite(v):
            out[alias] = float(v)

    return out


def get_elem_categorical_props(sym: str) -> Dict[str, str]:
    e = md_element(sym)
    gs_struct: Optional[str] = None
    for key in ["crystal_structure", "lattice_structure", "structure"]:
        v = _maybe_call(getattr(e, key, None))
        if isinstance(v, str) and v.strip():
            gs_struct = v.strip().lower(); break
    if not gs_struct:
        gs_struct = "unknown"
    phase = _maybe_call(getattr(e, "phase", None))
    phase = phase.strip().lower() if isinstance(phase, str) and phase.strip() else "unknown"
    aliases = {
        "body-centered cubic": "bcc", "face-centered cubic": "fcc", "hexagonal close packed": "hcp",
        "simple cubic": "sc", "cubic body-centered": "bcc", "cubic face-centered": "fcc",
    }
    norm = aliases.get(gs_struct, gs_struct)

    def _norm(s: str) -> str:
        s = s.lower()
        if "bcc" in s: return "bcc"
        if "fcc" in s: return "fcc"
        if "hcp" in s: return "hcp"
        if "diamond" in s: return "diamond"
        if s in ["sc", "simple cubic"]: return "sc"
        if "hexagon" in s: return "hexagonal"
        if "orthorhomb" in s: return "orthorhombic"
        if "tetragon" in s: return "tetragonal"
        if "monoclin" in s: return "monoclinic"
        if "triclin" in s: return "triclinic"
        if s == "cubic": return "cubic"
        return "unknown"

    try:
        gold = _maybe_call(getattr(e, "goldschmidt_class", None))
        gold = gold.strip().lower() if isinstance(gold, str) and gold.strip() else "unknown"
    except Exception:
        gold = "unknown"
    try:
        geo = _maybe_call(getattr(e, "geochemical_class", None))
        geo = geo.strip().lower() if isinstance(geo, str) and geo.strip() else "unknown"
    except Exception:
        geo = "unknown"
    return {
        "gs_structure": _norm(norm),
        "phase": phase,
        "goldschmidt_class": gold,
        "geochemical_class": geo,
    }


# ===== 白名单 =====
DIFF_KEYS: set[str] = {
    "chiP","chiA","chiAR","chiMB","chiNag","chiMul","chiSand","chiGhosh","chiGordy","chiLX","chiCS","chiGL","chiMied","chiMullay","chiRB",
    "IE1","EA","chiM","eta","omega_parr",
}
RATIO_KEYS: set[str] = {
    "r_cov","r_cov_cordero","r_cov_pyykko_double","r_cov_pyykko_triple",
    "r_vdw","r_vdw_alvarez","r_vdw_bondi","r_vdw_batsanov","r_met","r_met_c12","r_ion",
}
WMEAN_ONLY_KEYS: set[str] = {
    "alpha","C6","C6_gb","V_atom","MN","PN","GN","miedema_rho_e","miedema_V_m",
    "abund_crust","abund_sea","price_per_kg","prod_conc","resv_dist","recycle_rate","supply_risk",
}


def _mh_from_wmeans_guarded(m_w: Optional[float], h_w: Optional[float], key: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    m = float(m_w) if _is_finite(m_w) else np.nan
    h = float(h_w) if _is_finite(h_w) else np.nan
    if key in DIFF_KEYS and _is_finite(m) and _is_finite(h):
        out[f"mh_d_{key}"] = float(abs(m - h))
    if key in RATIO_KEYS and _is_finite(m) and _is_finite(h) and abs(h) > 1e-12:
        out[f"mh_r_{key}"] = float(m / h)
    return out


# ================ valence/state & ionic radius ================
def guess_valence(sym: str) -> Optional[int]:
    e = md_element(sym)
    ox = (
        _maybe_call(getattr(e, "oxistates", None))
        or _maybe_call(getattr(e, "oxidation_states", None))
        or []
    )
    try:
        oxs = list(ox)
    except Exception:
        oxs = []
    if sym in HALOGENS:
        neg = [o for o in oxs if isinstance(o, (int, float)) and o < 0]
        return int(sorted(neg, key=lambda x: abs(x))[0]) if neg else -1
    pos = [o for o in oxs if isinstance(o, (int, float)) and o > 0]
    if pos:
        return int(sorted(pos, key=lambda x: abs(x))[0])
    g = getattr(e, "group_id", getattr(e, "group", None))
    if isinstance(g, int):
        if g in (1, 2):
            return g
        if 13 <= g <= 18:
            return g - 10
    return None


def pick_ionic_radius(sym: str, charge: Optional[int]) -> Optional[float]:
    e = md_element(sym)
    items = _maybe_call(getattr(e, "ionic_radii", None)) or []
    try:
        if charge is not None:
            for CN in ("VI", "IV", "VIII", "II", "III"):
                for ir in items:
                    ch = getattr(ir, "charge", None)
                    cn = getattr(ir, "coordination", None)
                    r = getattr(ir, "ionic_radius", None)
                    if ch == charge and cn == CN and _is_finite(r):
                        return float(r) * 1e-2  # pm → Å
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
    cr = get_elem_numeric_props(sym).get("covalent_radius", None)
    return float(cr) if _is_finite(cr) else None


# ================ composition layer (mend_* + magpie_* + M/H/mh_*) ================
CORE_WMEAN_KEYS: List[Tuple[str, str]] = [
    ("en","chiP"),("chiP","chiP"),("chiA","chiA"),("chiM","chiM"),("eta","eta"),("omega","omega"),
    ("chiAR","chiAR"),("chiMB","chiMB"),("chiNag","chiNag"),("chiMul","chiMul"),("chiSand","chiSand"),
    ("chiGhosh","chiGhosh"),("chiGordy","chiGordy"),("chiLX","chiLX"),("chiCS","chiCS"),
    ("chiGL","chiGL"),("chiMied","chiMied"),("chiMullay","chiMullay"),("chiRB","chiRB"),
    ("electron_affinity","EA"),("ionization_energy_first","IE1"),
    ("dipole_polarizability","alpha"),("c6","C6"),("c6_gb","C6_gb"),
    ("covalent_radius","r_cov"),("metallic_radius","r_met"),("vdw_radius_best","r_vdw"),("atomic_volume","V_atom"),
    ("density","rho"),("melting_point","T_m"),("boiling_point","T_b"),("evaporation_heat","H_vap"),("fusion_heat","H_fus"),
    ("thermal_conductivity","kappa"),
    ("mendeleev_number","MN"),("pettifor_number","PN"),("glawe_number","GN"),("group_id","group"),("period","period"),
    ("electrophilicity","omega_parr"),("hardness","eta_parr"),("softness","soft_parr"),("proton_affinity","PA"),
    ("miedema_electron_density","miedema_rho_e"),("miedema_molar_volume","miedema_V_m"),
    ("abundance_crust","abund_crust"),("abundance_sea","abund_sea"),("price_per_kg","price_per_kg"),
    ("production_concentration","prod_conc"),("reserve_distribution","resv_dist"),
    ("recycling_rate","recycle_rate"),("relative_supply_risk","supply_risk"),
]


def mendeleev_MH_pack(elems: List[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    alias_map = {
        "en":"chiP","electronegativity_pauling":"chiP","en_pauling":"chiP",
        "electronegativity_allen":"chiA","en_allen":"chiA",
        "electronegativity_allred_rochow":"chiAR","electronegativity_martynov_batsanov":"chiMB",
        "electronegativity_nagle":"chiNag","electronegativity_mulliken":"chiMul",
        "electronegativity_sanderson":"chiSand","electronegativity_ghosh":"chiGhosh",
        "electronegativity_gordy":"chiGordy","electronegativity_li_xue":"chiLX",
        "electronegativity_cottrell_sutton":"chiCS","en_gunnarsson_lundqvist":"chiGL",
        "en_miedema":"chiMied","en_mullay":"chiMullay","en_robles_bartolotti":"chiRB",
        "covalent_radius":"r_cov","covalent_radius_bragg":"r_cov_bragg","covalent_radius_pyykko":"r_cov_pyykko",
        "covalent_radius_cordero":"r_cov_cordero","covalent_radius_pyykko_double":"r_cov_pyykko_double",
        "covalent_radius_pyykko_triple":"r_cov_pyykko_triple","vdw_radius_best":"r_vdw",
        "vdw_radius":"r_vdw_raw","vdw_radius_bondi":"r_vdw_bondi","vdw_radius_alvarez":"r_vdw_alvarez",
        "vdw_radius_batsanov":"r_vdw_batsanov","atomic_volume":"V_atom","density":"rho",
        "melting_point":"T_m","boiling_point":"T_b","evaporation_heat":"H_vap","fusion_heat":"H_fus","thermal_conductivity":"kappa",
        "electron_affinity":"EA","ionization_energy_first":"IE1","metallic_radius":"r_met","metallic_radius_c12":"r_met_c12",
        "electrophilicity":"omega_parr","hardness":"eta_parr","softness":"soft_parr","proton_affinity":"PA",
        "miedema_electron_density":"miedema_rho_e","miedema_molar_volume":"miedema_V_m",
        "abundance_crust":"abund_crust","abundance_sea":"abund_sea","price_per_kg":"price_per_kg",
        "production_concentration":"prod_conc","reserve_distribution":"resv_dist","recycling_rate":"recycle_rate",
        "relative_supply_risk":"supply_risk",
    }
    keys: set[str] = set()
    for sym, _ in elems:
        keys.update(get_elem_numeric_props(sym).keys())
    for raw in keys:
        alias = alias_map.get(raw, raw)
        m_vs: List[float] = []; m_ws: List[float] = []
        h_vs: List[float] = []; h_ws: List[float] = []
        for (sym, f) in elems:
            v = get_elem_numeric_props(sym).get(raw, None)
            if not _is_finite(v):
                continue
            if sym in HALOGENS:
                h_vs.append(float(v)); h_ws.append(float(f))
            else:
                m_vs.append(float(v)); m_ws.append(float(f))
        M = float(np.average(m_vs, weights=m_ws)) if m_vs else np.nan
        H = float(np.average(h_vs, weights=h_ws)) if h_vs else np.nan
        out[f"M_mend_{alias}"] = M
        out[f"H_mend_{alias}"] = H
        if alias in DIFF_KEYS and _is_finite(M) and _is_finite(H):
            out[f"mh_d_mend_{alias}"] = float(abs(M - H))
        if alias in RATIO_KEYS and _is_finite(M) and _is_finite(H) and abs(H) > 1e-12:
            out[f"mh_r_mend_{alias}"] = float(M / H)
    return out


def magpie_pack(elems: List[Tuple[str, float]], fracs: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not _HAVE_MAGPIE or not getattr(ARGS, "enable_magpie", False):
        return out
    props: List[str] = []
    for an in ["elemental_properties", "all_elemental_properties", "available_features"]:
        v = getattr(_MAG, an, None)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            props = list(v); break
    if not props:
        return out
    for prop in props:
        vals: List[float] = []
        for sym, _ in elems:
            try:
                val = _MAG.get_elemental_property(PmgElement(sym), prop)
            except Exception:
                val = np.nan
            vals.append(float(val) if _is_finite(val) else np.nan)
        out[f"magpie_{prop}_wmean"] = wavg(vals, fracs)
    return out


def magpie_MH_pack(elems: List[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not _HAVE_MAGPIE or not getattr(ARGS, "enable_magpie", False):
        return out
    props: List[str] = []
    for an in ["elemental_properties", "all_elemental_properties", "available_features"]:
        v = getattr(_MAG, an, None)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            props = list(v); break
    if not props:
        return out
    for prop in props:
        m_vs: List[float] = []; m_ws: List[float] = []
        h_vs: List[float] = []; h_ws: List[float] = []
        for (sym, f) in elems:
            try:
                val = _MAG.get_elemental_property(PmgElement(sym), prop)
            except Exception:
                val = np.nan
            if not _is_finite(val):
                continue
            if sym in HALOGENS:
                h_vs.append(float(val)); h_ws.append(float(f))
            else:
                m_vs.append(float(val)); m_ws.append(float(f))
        M = float(np.average(m_vs, weights=m_ws)) if m_vs else np.nan
        H = float(np.average(h_vs, weights=h_ws)) if h_vs else np.nan
        out[f"M_mag_{prop}"] = M
        out[f"H_mag_{prop}"] = H
        if prop in DIFF_KEYS and _is_finite(M) and _is_finite(H):
            out[f"mh_d_mag_{prop}"] = float(abs(M - H))
        if prop in RATIO_KEYS and _is_finite(M) and _is_finite(H) and abs(H) > 1e-12:
            out[f"mh_r_mag_{prop}"] = float(M / H)
    return out


def _grouped_wmean_only(elems: List[Tuple[str, float]], prop_key: str, group: str, out_name: str) -> Dict[str, float]:
    vs: List[float] = []; ws: List[float] = []
    for sym, frac in elems:
        is_hal = sym in HALOGENS
        if group == "M" and is_hal: continue
        if group == "H" and not is_hal: continue
        val = get_elem_numeric_props(sym).get(prop_key, None)
        vs.append(val if _is_finite(val) else np.nan)
        ws.append(frac)
    w = wavg(vs, ws) if any(_is_finite(v) for v in vs) else float("nan")
    return {out_name: float(w)}


def composition_features_all(struct: Structure) -> Dict[str, float]:
    comp = struct.composition.fractional_composition
    elems = sorted([(el.symbol, float(frac)) for el, frac in comp.items()], key=lambda x: x[0])
    fracs = [f for _, f in elems]
    out: Dict[str, float] = {
        "comp_entropy": shannon_entropy(fracs),
        "comp_x_frac": float(sum(f for s, f in elems if s in HALOGENS)),
    }
    for k_raw, k_alias in CORE_WMEAN_KEYS:
        mkey, hkey = f"M_{k_alias}", f"H_{k_alias}"
        out.update(_grouped_wmean_only(elems, k_raw, "M", out_name=mkey))
        out.update(_grouped_wmean_only(elems, k_raw, "H", out_name=hkey))
        out.update(_mh_from_wmeans_guarded(out.get(mkey), out.get(hkey), k_alias))

    key2vals: Dict[str, List[float]] = {}
    for sym, _f in elems:
        props = get_elem_numeric_props(sym)
        for k, v in props.items():
            key2vals.setdefault(k, []).append(float(v) if _is_finite(v) else np.nan)
    for k, vals in key2vals.items():
        alias = {
            "en": "chiP","electronegativity_pauling":"chiP","en_pauling":"chiP",
            "electronegativity_allen":"chiA","en_allen":"chiA",
            "covalent_radius":"r_cov","covalent_radius_bragg":"r_cov_bragg","covalent_radius_pyykko":"r_cov_pyykko",
            "vdw_radius_best":"r_vdw","vdw_radius":"r_vdw_raw","vdw_radius_bondi":"r_vdw_bondi",
            "vdw_radius_alvarez":"r_vdw_alvarez","vdw_radius_batsanov":"r_vdw_batsanov",
            "atomic_volume":"V_atom","density":"rho","melting_point":"T_m","boiling_point":"T_b",
            "evaporation_heat":"H_vap","fusion_heat":"H_fus","thermal_conductivity":"kappa",
            "electron_affinity":"EA","ionization_energy_first":"IE1","metallic_radius":"r_met",
        }.get(k, k)
        out[f"mend_{alias}_wmean"] = wavg(vals, fracs)
        out.update(stats_pack(vals, f"mend_{alias}", compact=True))

    out.update(magpie_pack(elems, fracs))
    out.update(mendeleev_MH_pack(elems))
    out.update(magpie_MH_pack(elems))

    chis_M: List[float] = []; chis_H: List[float] = []
    for sym, _ in elems:
        val = get_elem_numeric_props(sym).get("en", None)
        if _is_finite(val):
            (chis_H if sym in HALOGENS else chis_M).append(float(val))
    if chis_M and chis_H:
        out["feat_dchi_MH_mean"] = float(np.mean([abs(m - h) for m in chis_M for h in chis_H]))
        out["feat_dchi_OminusH_M"] = float(np.mean([abs(OXYGEN_EN - m) for m in chis_M])) - float(
            np.mean([abs(h - np.mean(chis_M)) for h in chis_H])
        )
    else:
        out["feat_dchi_MH_mean"] = np.nan
        out["feat_dchi_OminusH_M"] = np.nan

    hal_fracs = [f for s, f in elems if s in HALOGENS]
    out["comp_x_mix_entropy"] = shannon_entropy(hal_fracs) if hal_fracs else 0.0
    tot_h = sum(hal_fracs); tot_m = 1.0 - tot_h
    out["comp_x_over_m"] = float(tot_h / tot_m) if tot_m > 0 else np.nan

    def _rion(filter_halogen: bool, prefix: str):
        vals: List[float] = []; weights: List[float] = []
        for sym, frac in elems:
            if (sym in HALOGENS) != filter_halogen:
                continue
            z = guess_valence(sym)
            r = pick_ionic_radius(sym, z)
            if _is_finite(r):
                vals.append(float(r)); weights.append(float(frac))
        if vals:
            wmean = wavg(vals, weights)
            var = np.average((np.array(vals) - wmean) ** 2, weights=weights)
            out[f"{prefix}_wmean"] = float(wmean)
            out[f"{prefix}_wstd"] = float(math.sqrt(var))
        else:
            out[f"{prefix}_wmean"] = np.nan
            out[f"{prefix}_wstd"] = np.nan
    _rion(False, "feat_rion_M")
    _rion(True, "feat_rion_H")
    out["M_r_ion"] = out.get("feat_rion_M_wmean", np.nan)
    out["H_r_ion"] = out.get("feat_rion_H_wmean", np.nan)
    out.update(_mh_from_wmeans_guarded(out.get("M_r_ion"), out.get("H_r_ion"), "r_ion"))

    total_charge: float = 0.0
    num: float = 0.0; den: float = 0.0
    for sym, frac in elems:
        z = guess_valence(sym)
        if z is not None:
            total_charge += frac * float(z)
            r = pick_ionic_radius(sym, z)
            if (sym not in HALOGENS) and _is_finite(r) and r > 0:
                num += frac * (abs(z) / (r ** 2))
                den += frac
    out["feat_charge_balance_abs"] = float(abs(total_charge))
    out["feat_field_strength_mean"] = float(num / den) if den > 0 else np.nan

    def _cat_fracs(group: str, categories: List[str], key: str, prefix: str) -> None:
        total: float = 0.0
        counts: Dict[str, float] = {c: 0.0 for c in categories}
        counts["other"] = 0.0
        for sym, frac in elems:
            is_hal = sym in HALOGENS
            if group == "M" and is_hal: continue
            if group == "H" and not is_hal: continue
            cat = get_elem_categorical_props(sym).get(key, "unknown")
            total += frac
            counts[cat if cat in categories else "other"] += frac
        for c, v in counts.items():
            out[f"mend_{prefix}_{c}_frac"] = float(v / total) if total > 0 else np.nan

    _cat_fracs("M", ["bcc","fcc","hcp","diamond","sc","hexagonal"], "gs_structure", "gs_M")
    _cat_fracs("H", ["bcc","fcc","hcp","diamond","sc","hexagonal"], "gs_structure", "gs_H")
    _cat_fracs("M", ["solid","liquid","gas"], "phase", "phase_M")
    _cat_fracs("H", ["solid","liquid","gas"], "phase", "phase_H")
    _cat_fracs("M", ["alkali","alkaline earth","transition metal","metalloid","post-transition metal","lanthanoid","actinoid","nonmetal","halogen","noble gas"], "goldschmidt_class", "gold_M")
    _cat_fracs("H", ["alkali","alkaline earth","transition metal","metalloid","post-transition metal","lanthanoid","actinoid","nonmetal","halogen","noble gas"], "goldschmidt_class", "gold_H")
    _cat_fracs("M", ["atmophile","chalcophile","lithophile","siderophile"], "geochemical_class", "geo_M")
    _cat_fracs("H", ["atmophile","chalcophile","lithophile","siderophile"], "geochemical_class", "geo_H")

    if _is_finite(out.get("M_C6")) and _is_finite(out.get("H_C6")):
        out["C6_geomean_MH"] = float(np.sqrt(out["M_C6"] * out["H_C6"]))

    chis: List[Tuple[float, float]] = []
    for el, frac in comp.items():
        chi = get_elem_numeric_props(el.symbol).get("en", None)
        if _is_finite(chi):
            chis.append((float(chi), float(frac)))
    if chis:
        logsum = sum(fr * math.log(chi) for chi, fr in chis if chi > 0)
        out["chi_geomean_comp"] = float(np.exp(logsum))
    return out


# ================ 结构/图/对称/Voronoi ================
def build_bonded(struct: Structure, method: str = "crystalnn") -> Tuple[StructureGraph, Any]:
    if method == "crystalnn":
        try:
            nn = CrystalNN()
            return nn.get_bonded_structure(struct), nn
        except Exception:
            pass
    nn = VoronoiNN(cutoff=10.0)
    return nn.get_bonded_structure(struct), nn


def _to_nx_graph(sg: StructureGraph):
    if nx is None: return None
    if hasattr(sg, "as_graph"):
        try:
            return sg.as_graph()
        except Exception:
            pass
    if hasattr(sg, "graph") and sg.graph is not None:
        try:
            return nx.Graph(sg.graph)  # type: ignore[arg-type]
        except Exception:
            try:
                return nx.Graph(nx.MultiGraph(sg.graph))  # type: ignore[arg-type]
            except Exception:
                return None
    return None


def _safe_getattr(obj: Any, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _neighbor_distance(struct: Structure, i: int, conn_site) -> Optional[float]:
    site_obj = _safe_getattr(conn_site, ["site", "to_site", "neighbor"], None)
    if site_obj is not None:
        try:
            return float(struct[i].distance(site_obj))
        except Exception:
            pass
    j = _safe_getattr(conn_site, ["index", "j", "to_index"], None)
    if j is not None:
        try:
            return float(struct.get_distance(i, int(j)))
        except Exception:
            pass
    w = _safe_getattr(conn_site, ["weight", "nn_distance"], None)
    return float(w) if _is_finite(w) else None


def cn_bond_angle_graph_symm_voro(struct: Structure, sg: StructureGraph) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cn_all: List[int] = []; cn_M: List[int] = []; cn_H: List[int] = []
    bond_MH: List[float] = []; bond_MHn: List[float] = []
    bond_MM: List[float] = []; bond_MMn: List[float] = []
    bond_HH: List[float] = []; bond_HHn: List[float] = []
    symbols = [site.specie.symbol for site in struct.sites]
    covr = {sym: get_elem_numeric_props(sym).get("covalent_radius", None) for sym in set(symbols)}
    for i, site in enumerate(struct.sites):
        try:
            neighs = sg.get_connected_sites(i)
        except Exception:
            neighs = []
        cn = len(neighs)
        cn_all.append(cn)
        (cn_H if site.specie.symbol in HALOGENS else cn_M).append(cn)
        for cs in neighs:
            js = _safe_getattr(cs, ["site", "to_site", "neighbor"], None)
            j_sym = js.specie.symbol if js is not None else None
            if j_sym is None:
                continue
            d = _neighbor_distance(struct, i, cs)
            if not _is_finite(d):
                continue
            def _norm(a: str, b: str, dist: float) -> float:
                rc = (covr.get(a) or 0.0) + (covr.get(b) or 0.0)
                return float(dist) / rc if rc and rc > 1e-8 else np.nan
            si = site.specie.symbol
            if (si in HALOGENS) ^ (j_sym in HALOGENS):
                bond_MH.append(float(d)); bond_MHn.append(_norm(si, j_sym, d))
            elif (si in HALOGENS) and (j_sym in HALOGENS):
                bond_HH.append(float(d)); bond_HHn.append(_norm(si, j_sym, d))
            else:
                bond_MM.append(float(d)); bond_MMn.append(_norm(si, j_sym, d))
    out.update(stats_pack(cn_all, "feat_cn", compact=False))
    out.update(stats_pack(cn_M, "feat_cn_cation", compact=False))
    out.update(stats_pack(cn_H, "feat_cn_halogen", compact=False))
    out.update(stats_pack(bond_MH, "bond_MH", compact=False))
    out.update(stats_pack(bond_MHn, "bond_MH_norm", compact=False))
    out.update(stats_pack(bond_MM, "bond_MM", compact=False))
    out.update(stats_pack(bond_MMn, "bond_MM_norm", compact=False))
    out.update(stats_pack(bond_HH, "bond_HH", compact=False))
    out.update(stats_pack(bond_HHn, "bond_HH_norm", compact=False))

    def _angle(struct: Structure, va, vb) -> float:
        A = struct.lattice.matrix
        va_c, vb_c = A.dot(np.array(va)), A.dot(np.array(vb))
        na, nb = np.linalg.norm(va_c), np.linalg.norm(vb_c)
        if na < 1e-8 or nb < 1e-8: return np.nan
        cosang = np.clip(np.dot(va_c, vb_c) / (na * nb), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    MHM: List[float] = []; HMH: List[float] = []
    for i in range(len(struct)):
        si = symbols[i]
        try:
            ns = sg.get_connected_sites(i)
        except Exception:
            continue
        if si in HALOGENS:
            mnei = [
                int(_safe_getattr(cs, ["index", "j", "to_index"], None))
                for cs in ns
                if _safe_getattr(cs, ["index", "j", "to_index"], None) is not None
                and symbols[int(_safe_getattr(cs, ["index", "j", "to_index"], None))] not in HALOGENS
            ]
            for a in range(len(mnei)):
                for b in range(a + 1, len(mnei)):
                    j = mnei[a]; k = mnei[b]
                    va = struct[j].frac_coords - struct[i].frac_coords
                    vb = struct[k].frac_coords - struct[i].frac_coords
                    ang = _angle(struct, va, vb)
                    if _is_finite(ang): MHM.append(ang)
        else:
            hnei = [
                int(_safe_getattr(cs, ["index", "j", "to_index"], None))
                for cs in ns
                if _safe_getattr(cs, ["index", "j", "to_index"], None) is not None
                and symbols[int(_safe_getattr(cs, ["index", "j", "to_index"], None))] in HALOGENS
            ]
            for a in range(len(hnei)):
                for b in range(a + 1, len(hnei)):
                    j = hnei[a]; k = hnei[b]
                    va = struct[j].frac_coords - struct[i].frac_coords
                    vb = struct[k].frac_coords - struct[i].frac_coords
                    ang = _angle(struct, va, vb)
                    if _is_finite(ang): HMH.append(ang)
    out.update(stats_pack(MHM, "angle_MHM", compact=False))
    out.update(stats_pack(HMH, "angle_HMH", compact=False))

    if nx is not None:
        Gp = _to_nx_graph(sg)
        GM = None
        if Gp is not None:
            sym = {i: struct.sites[i].specie.symbol for i in range(len(struct))}
            M_nodes = [i for i in Gp.nodes if sym[i] not in HALOGENS]
            H_nodes = [i for i in Gp.nodes if sym[i] in HALOGENS]
            GM = nx.Graph(); GM.add_nodes_from(M_nodes)
            for h in H_nodes:
                neigh_M = [n for n in Gp.neighbors(h) if sym[n] not in HALOGENS]
                for a in range(len(neigh_M)):
                    for b in range(a + 1, len(neigh_M)):
                        i_node, j_node = neigh_M[a], neigh_M[b]
                        GM.add_edge(i_node, j_node, w=GM.get_edge_data(i_node, j_node, {}).get("w", 0) + 1)
        metrics: Dict[str, float] = {
            "graph_Mproj_deg_mean": np.nan, "graph_Mproj_deg_std": np.nan, "graph_Mproj_clust": np.nan,
            "graph_Mproj_spectrum_max": np.nan, "graph_Mproj_ncc": np.nan, "graph_Mproj_diam": np.nan,
            "graph_Mproj_cycle3": np.nan, "graph_Mproj_cycle4": np.nan, "graph_Mproj_asp": np.nan,
            "graph_Mproj_eff": np.nan, "graph_Mproj_assort": np.nan, "graph_Mproj_kcore": np.nan,
            "graph_Mproj_clique_max": np.nan, "graph_Mproj_algebraic_connectivity": np.nan,
        }
        if GM is not None and GM.number_of_nodes() > 0:
            degs = [d for _, d in GM.degree()]
            metrics["graph_Mproj_deg_mean"] = float(np.mean(degs)) if degs else np.nan
            metrics["graph_Mproj_deg_std"] = float(np.std(degs)) if degs else np.nan
            try: metrics["graph_Mproj_clust"] = float(nx.average_clustering(GM))
            except Exception: pass
            try:
                A = nx.to_numpy_array(GM, weight="w")
                metrics["graph_Mproj_spectrum_max"] = float(np.max(np.linalg.eigvalsh(A))) if A.size else np.nan
            except Exception: pass
            try:
                ccs = list(nx.connected_components(GM))
                metrics["graph_Mproj_ncc"] = float(len(ccs))
                Gbig = GM.subgraph(max(ccs, key=len)).copy() if ccs else GM
                metrics["graph_Mproj_diam"] = float(nx.diameter(Gbig)) if Gbig.number_of_nodes() > 1 else 0.0
            except Exception: pass
            try:
                lens = [len(c) for c in nx.cycle_basis(GM)]
                metrics["graph_Mproj_cycle3"] = float(sum(1 for l in lens if l == 3))
                metrics["graph_Mproj_cycle4"] = float(sum(1 for l in lens if l == 4))
            except Exception: pass
            try:
                ccs = list(nx.connected_components(GM))
                Gbig = GM.subgraph(max(ccs, key=len)).copy() if ccs else GM
                metrics["graph_Mproj_asp"] = float(nx.average_shortest_path_length(Gbig)) if Gbig.number_of_edges() > 0 and Gbig.number_of_nodes() > 1 else 0.0
            except Exception: pass
            try: metrics["graph_Mproj_eff"] = float(nx.global_efficiency(GM))
            except Exception: pass
            try: metrics["graph_Mproj_assort"] = float(nx.degree_assortativity_coefficient(GM))
            except Exception: pass
            try:
                core = nx.core_number(GM)
                metrics["graph_Mproj_kcore"] = float(max(core.values()) if core else 0)
            except Exception: pass
            try: metrics["graph_Mproj_clique_max"] = float(len(max(nx.find_cliques(GM), key=len)))
            except Exception: pass
            try:
                if GM.number_of_nodes() > 1:
                    L = nx.laplacian_matrix(GM, weight="w").astype(float).todense()
                    evals = np.linalg.eigvalsh(np.array(L))
                    metrics["graph_Mproj_algebraic_connectivity"] = float(sorted(evals)[1]) if evals.size > 1 else 0.0
                else:
                    metrics["graph_Mproj_algebraic_connectivity"] = 0.0
            except Exception: pass
        out.update(metrics)

    out.update(
        {"symm_sg_number": np.nan, "symm_crystal_system_id": np.nan, "symm_point_group_id": np.nan,
         "symm_n_wyckoff": np.nan, "symm_is_primitive": np.nan, "symm_n_atoms_primitive": np.nan}
    )
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(struct, symprec=1e-2, angle_tolerance=5)
        systems = ["triclinic","monoclinic","orthorhombic","tetragonal","trigonal","hexagonal","cubic"]
        out["symm_sg_number"] = float(sga.get_space_group_number())
        sys_name = sga.get_crystal_system()
        out["symm_crystal_system_id"] = float(systems.index(sys_name) + 1) if sys_name in systems else np.nan
        pg = sga.get_point_group_symbol()
        pg_map = {k: i + 1 for i, k in enumerate(sorted({
            "1","2","m","2/m","222","mm2","mmm","4","-4","4/m","422","4mm","-42m","4/mmm",
            "3","-3","32","3m","-3m","6","-6","6/m","622","6mm","-62m","6/mmm",
        })) }
        out["symm_point_group_id"] = float(pg_map.get(pg, np.nan))
        wy = sga.get_symmetry_dataset()
        if wy and "wyckoffs" in wy: out["symm_n_wyckoff"] = float(len(wy["wyckoffs"]))
        prim = sga.find_primitive()
        if prim:
            out["symm_is_primitive"] = 1.0; out["symm_n_atoms_primitive"] = float(len(prim))
        else:
            out["symm_is_primitive"] = 0.0
    except Exception:
        pass

    try:
        vnn = VoronoiNN(cutoff=10.0)
        CNs: List[int] = []; faces: List[int] = []
        for i in range(len(struct)):
            try:
                ns = vnn.get_nn_info(struct, i)
            except Exception:
                continue
            CNs.append(len(ns)); faces.append(len(ns))
        if CNs:
            counts = np.unique(CNs, return_counts=True)[1]
            out["feat_cn_entropy"] = shannon_entropy([v / len(CNs) for v in counts])
        else:
            out["feat_cn_entropy"] = np.nan
        out.update(stats_pack(faces, "voro_face", compact=False))
    except Exception:
        out["feat_cn_entropy"] = np.nan
        out.update({k: np.nan for k in ["voro_face_mean","voro_face_std","voro_face_median","voro_face_min","voro_face_max","voro_face_range"]})

    vols: List[float] = []
    for s in struct.sites:
        r = get_elem_numeric_props(s.specie.symbol).get("covalent_radius", None)
        if _is_finite(r) and r > 0:
            vols.append((4.0 / 3.0) * math.pi * (r ** 3))
    out["struct_packing_covrad"] = float(sum(vols) / struct.volume) if vols and struct.volume > 0 else np.nan
    return out


def structural_block(struct: Structure, nn_method: str = "crystalnn", do_graph: bool = True) -> Dict[str, float]:
    feats: Dict[str, float] = {
        "struct_n_sites": float(len(struct)),
        "struct_rho": float(getattr(struct, "density", np.nan)),
        "struct_V": float(struct.volume),
        "struct_V_per_atom": float(struct.volume / len(struct)) if len(struct) > 0 else np.nan,
    }
    try:
        sg_tmp, _ = build_bonded(struct, nn_method)
        feats["struct_dim_larsen"] = int(get_dimensionality_larsen(sg_tmp))
    except Exception:
        sg_tmp = None; feats["struct_dim_larsen"] = np.nan
    if sg_tmp is None:
        sg_tmp, _ = build_bonded(struct, nn_method)
    feats.update(cn_bond_angle_graph_symm_voro(struct, sg_tmp))
    return feats


# ================ Matminer（可选） ================
def matminer_composition_pack(struct: Structure) -> Dict[str, float]:
    if not _HAVE_MATM or not getattr(ARGS, "enable_matminer_comp", False):
        return {}
    out: Dict[str, float] = {}
    try:
        comp = PmgComposition(struct.composition.reduced_composition.alphabetical_formula)
        ep = ElementProperty(
            features=["Number","MendeleevNumber","AtomicWeight","MeltingT","BoilingT","CovalentRadius","Electronegativity","ElectronAffinity","FusionEnthalpy","ThermalConductivity"],
            stats=["mean","avg_dev","max","min","range","std"],
        )
        st = Stoichiometry(p_list=(0, 2, 3, 5))
        vo = ValenceOrbital(props=["s", "p", "d", "f"], stats=["sum", "frac"])
        ef = ElementFraction()
        bc = BandCenter()
        def add(prefix: str, labels: List[str], values: List[float]):
            for k, v in zip(labels, values):
                out[f"{prefix}_{k}"] = float(v) if _is_finite(v) else np.nan
        add("mm_EP", ep.feature_labels(), ep.featurize(comp))
        add("mm_ST", st.feature_labels(), st.featurize(comp))
        add("mm_VO", vo.feature_labels(), vo.featurize(comp))
        add("mm_EF", ef.feature_labels(), ef.featurize(comp))
        add("mm_BC", bc.feature_labels(), bc.featurize(comp))
    except Exception:
        pass
    return out


def matminer_structure_pack(struct: Structure) -> Dict[str, float]:
    if not _HAVE_MATM or not getattr(ARGS, "enable_structure_fp", False):
        return {}
    out: Dict[str, float] = {}
    try:
        dens = DensityFeatures(desired_features=("density","vpa","packing fraction"))
        vals = dens.featurize(struct)
        for k, v in zip(dens.feature_labels(), vals):
            out[f"mm_Density_{k.replace(' ', '_')}"] = float(v) if _is_finite(v) else np.nan
    except Exception:
        pass
    if getattr(ARGS, "enable_rdf", False):
        try:
            rdf = RadialDistributionFunction(cutoff=getattr(ARGS, "rdf_cutoff", 8.0), bin_size=getattr(ARGS, "rdf_bin", 0.2))
            vals = rdf.featurize(struct)
            for i, v in enumerate(vals):
                out[f"mm_RDF_bin_{i}"] = float(v) if _is_finite(v) else np.nan
        except Exception:
            pass
        try:
            adf = AngularDistributionFunction(cutoff=getattr(ARGS, "adf_cutoff", 6.0), n_bins=getattr(ARGS, "adf_bins", 30))
            vals = adf.featurize(struct)
            for i, v in enumerate(vals):
                out[f"mm_ADF_bin_{i}"] = float(v) if _is_finite(v) else np.nan
        except Exception:
            pass
    if getattr(ARGS, "enable_coulomb", False):
        for Cls, tag in [(CoulombMatrix, "CM"), (SineCoulombMatrix, "SCM"), (OrbitalFieldMatrix, "OFM")]:
            try:
                cm = Cls(flatten=True) if Cls is not OrbitalFieldMatrix else Cls(flatten=True, cation_anion=False)
                vals = cm.featurize(struct)
                for i, v in enumerate(vals):
                    out[f"mm_{tag}_{i}"] = float(v) if _is_finite(v) else np.nan
            except Exception:
                pass
        try:
            ew = EwaldEnergy()
            vals = ew.featurize(struct)
            for k, v in zip(ew.feature_labels(), vals):
                out[f"mm_Ewald_{k}"] = float(v) if _is_finite(v) else np.nan
        except Exception:
            pass
    if getattr(ARGS, "enable_opsf", False):
        try:
            op = OPSiteFingerprint()
            ssf = SiteStatsFingerprint(op, stats=("mean","std","min","max"))
            vals = ssf.featurize(struct)
            for k, v in zip(ssf.feature_labels(), vals):
                out[f"mm_OPSF_{k}"] = float(v) if _is_finite(v) else np.nan
            cnf = SiteStatsFingerprint(CoordinationNumber(), stats=("mean","std","min","max"))
            vals = cnf.featurize(struct)
            for k, v in zip(cnf.feature_labels(), vals):
                out[f"mm_CNStats_{k}"] = float(v) if _is_finite(v) else np.nan
        except Exception:
            pass
    try:
        bob = BagofBonds()
        vals = bob.featurize(struct)
        for i, v in enumerate(vals):
            out[f"mm_BoB_{i}"] = float(v) if _is_finite(v) else np.nan
    except Exception:
        pass
    if getattr(ARGS, "enable_xrd", False):
        try:
            xrd = XRDPowderPattern(two_theta_range=(10, 90))
            vals = xrd.featurize(struct)
            for k, v in zip(xrd.feature_labels(), vals):
                out[f"mm_XRD_{k}"] = float(v) if _is_finite(v) else np.nan
        except Exception:
            pass
    return out


def matminer_auto_pack(struct: Structure) -> Dict[str, float]:
    if not _HAVE_MATM or not getattr(ARGS, "enable_matminer_auto", False):
        return {}
    out: Dict[str, float] = {}
    targets = [
        "matminer.featurizers.composition",
        "matminer.featurizers.structure",
        "matminer.featurizers.structure.sites",
        "matminer.featurizers.structure.matrix",
    ]
    try:
        comp = PmgComposition(struct.composition.reduced_composition.alphabetical_formula)
    except Exception:
        comp = None
    for mod_name in targets:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            try:
                if not issubclass(obj, BaseFeaturizer):
                    continue
            except Exception:
                continue
            blacklist = {"BandFeaturizer", "DOSFeaturizer", "SiteFingerprint"}
            if name in blacklist:
                continue
            try:
                fea = obj()
            except Exception:
                continue
            did = False
            for kind, payload in [("Structure", struct), ("Composition", comp)]:
                if payload is None:
                    continue
                try:
                    vals = fea.featurize(payload)  # type: ignore[attr-defined]
                    labels = fea.feature_labels()  # type: ignore[attr-defined]
                    for k, v in zip(labels, vals):
                        out[f"mmauto_{name}_{k}"] = float(v) if _is_finite(v) else np.nan
                    did = True; break
                except Exception:
                    continue
            _ = did
    return out


# ================ DScribe（SOAP/ACSF/MBTR） ================
def dscribe_pack(struct: Structure) -> Dict[str, float]:
    if not _HAVE_DSCR:
        return {}
    out: Dict[str, float] = {}
    try:
        atoms = AseAtomsAdaptor.get_atoms(struct)
        species = sorted({site.specie.symbol for site in struct.sites})
    except Exception:
        return out
    if getattr(ARGS, "enable_dscribe_soap", False):
        try:
            desc = SOAP(
                species=species, periodic=True,
                rcut=getattr(ARGS, "soap_rcut", 5.0), nmax=getattr(ARGS, "soap_nmax", 8),
                lmax=getattr(ARGS, "soap_lmax", 6), sigma=getattr(ARGS, "soap_sigma", 0.4),
                sparse=False, average=False,
            )
            X = desc.create(atoms)
            mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
            for i, v in enumerate(mu): out[f"soap_mean_{i}"] = float(v) if _is_finite(v) else np.nan
            for i, v in enumerate(sd): out[f"soap_std_{i}"] = float(v) if _is_finite(v) else np.nan
            out["soap_mean_l2"] = float(np.linalg.norm(mu)); out["soap_std_l2"] = float(np.linalg.norm(sd))
        except Exception:
            pass
    if getattr(ARGS, "enable_dscribe_acsf", False):
        try:
            desc = ACSF(species=species, rcut=getattr(ARGS, "acsf_rcut", 6.0), sparse=False)
            X = desc.create(atoms)
            mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
            for i, v in enumerate(mu): out[f"acsf_mean_{i}"] = float(v) if _is_finite(v) else np.nan
            for i, v in enumerate(sd): out[f"acsf_std_{i}"] = float(v) if _is_finite(v) else np.nan
        except Exception:
            pass
    if getattr(ARGS, "enable_dscribe_mbtr", False):
        try:
            desc = MBTR(
                species=species, periodic=True,
                k1={"geometry": {"function": "atomic_number"}},
                k2={"geometry": {"function": "distance"}},
                k3={"geometry": {"function": "angle"}},
                grid={"min": 0, "max": 8, "n": 80, "sigma": 0.1},
                sparse=False, normalization="l2_each",
            )
            X = desc.create(atoms)
            mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
            for i, v in enumerate(mu): out[f"mbtr_mean_{i}"] = float(v) if _is_finite(v) else np.nan
            for i, v in enumerate(sd): out[f"mbtr_std_{i}"] = float(v) if _is_finite(v) else np.nan
        except Exception:
            pass
    return out


# ================ 行处理与主流程 ================
def process_row(row: Dict[str, Any], cif_root: str, nn: str, do_graph: bool) -> Dict[str, float]:
    path_in = str(row.get("id_cif_path", row.get("cif_path")))
    path = path_in if os.path.isabs(path_in) else os.path.join(cif_root, path_in)
    out: Dict[str, float] = {}
    try:
        s = Structure.from_file(path)
    except Exception as e:
        out["flag_parse_error"] = 1.0
        out["flag_parse_error_msg"] = str(e)
        return out
    out.update(composition_features_all(s))
    out.update(structural_block(s, nn_method=nn, do_graph=do_graph))
    out.update(matminer_composition_pack(s))
    out.update(matminer_structure_pack(s))
    out.update(matminer_auto_pack(s))
    out.update(dscribe_pack(s))
    return out


def drop_constant(df: pd.DataFrame, thresh_unique: int = 1) -> Tuple[pd.DataFrame, List[str]]:
    nunq = df.nunique(dropna=False)
    to_drop = [c for c, n in nunq.items() if n <= thresh_unique]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
    return df, to_drop


# ================ 新命名：信息保留 ================
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    新命名：<domain>_<source>_<object>_<property>_<op>[_<stat>][_<norm>][_<method>][__<unit>]
    """
    import re

    def unit_for(col: str) -> str:
        if re.search(r'(bond|_len_)', col): return '__A'
        if re.search(r'(_angle_|_deg)', col): return '__deg'
        if re.search(r'(_V_cell_|_V_per_atom_)', col):
            return '__A3' if '_V_cell_' in col else '__A3atom'
        if re.search(r'(_rho_)', col): return '__gcm3'
        if re.search(r'(_Ewald_|_BC_.*|^mm_.*_BandCenter_)', col): return '__eV'
        if re.search(r'(_field_strength_)', col): return '__1A2'
        return '__1'

    id_alias_map = {
        'cif_path': 'id_cif_path', 'cif_file': 'id_cif', 'score': 'id_score', 'dim': 'id_dim', 'connect': 'id_connect',
    }

    new = {}
    for c in df.columns:
        if c in id_alias_map:
            new[c] = id_alias_map[c]; continue
        nc = c

        if re.match(r'^(M|H)_(.+)$', c):
            g, prop = re.match(r'^(M|H)_(.+)$', c).groups()
            nc = f'comp_mend_{g}_{prop}_wmean{unit_for(prop)}'
        elif re.match(r'^mh_d_(.+)$', c):
            prop = re.match(r'^mh_d_(.+)$', c).group(1)
            nc = f'comp_mend_MH_{prop}_diff{unit_for(prop)}'
        elif re.match(r'^mh_r_(.+)$', c):
            prop = re.match(r'^mh_r_(.+)$', c).group(1)
            nc = f'comp_mend_M_{prop}_ratio{unit_for(prop)}'
        elif c.startswith('mend_'):
            m = re.match(r'^mend_([^_]+)_(wmean|mean|std|median)$', c)
            if m:
                prop, stat = m.groups()
                nc = f'comp_mend_all_{prop}_{stat}{unit_for(prop)}'
        elif c.startswith('magpie_'):
            m = re.match(r'^magpie_([^_]+)_wmean$', c)
            if m:
                prop = m.group(1)
                nc = f'comp_mag_all_{prop}_wmean{unit_for(prop)}'
        elif c in {'comp_entropy','comp_x_frac','comp_x_mix_entropy','comp_x_over_m','chi_geomean_comp',
                   'feat_field_strength_mean','feat_charge_balance_abs','C6_geomean_MH','M_C6','H_C6',
                   'M_r_ion','H_r_ion'}:
            if c == 'feat_field_strength_mean':
                nc = f'comp_calc_all_field_strength_mean__1A2'
            elif c == 'feat_charge_balance_abs':
                nc = f'comp_calc_all_charge_balance_abs__1'
            elif c == 'chi_geomean_comp':
                nc = f'comp_calc_all_chi_geomean_comp_raw__1'
            elif c in {'comp_entropy','comp_x_frac','comp_x_mix_entropy','comp_x_over_m'}:
                nc = f'comp_calc_all_{c.replace("comp_","")}_raw__1'
            elif c == 'C6_geomean_MH':
                nc = f'comp_calc_MH_C6_geomMean_raw__1'
            elif c in {'M_C6','H_C6'}:
                nc = f'comp_mend_{c[0]}_C6_wmean__1'
            elif c in {'M_r_ion','H_r_ion'}:
                nc = f'comp_calc_{c[0]}_r_ion_wmean__A'

        elif c.startswith('bond_'):
            m = re.match(r'^bond_(MM|MH|HH)_(norm_)?(mean|std|median|min|max|range)$', c)
            if m:
                pair, norm, stat = m.groups()
                norm_tag = 'norm_covsum' if norm else None
                nc = f'struct_calc_{pair}_bond_len_{stat}' + (f'_{norm_tag}' if norm_tag else '') + '__A'
        elif c.startswith('angle_'):
            m = re.match(r'^angle_(MHM|HMH)_(mean|std|median|min|max|range)$', c)
            if m:
                triple, stat = m.groups()
                nc = f'struct_calc_{triple}_angle_{stat}__deg'
        elif c.startswith('feat_cn_'):
            m = re.match(r'^feat_cn(_(cation|halogen))?_(mean|std|median|min|max|range)$', c)
            if m:
                sub, grp, stat = m.groups()
                obj = 'M' if grp=='cation' else ('H' if grp=='halogen' else 'all')
                nc = f'struct_calc_{obj}_cn_{stat}__1'
        elif c == 'struct_packing_covrad':
            nc = 'struct_calc_all_packing_covrad_raw__1'
        elif c in {'struct_n_sites','struct_rho','struct_V','struct_V_per_atom','struct_dim_larsen'}:
            table = {
                'struct_n_sites':'struct_calc_all_n_sites_raw__1',
                'struct_rho':'struct_calc_all_rho_raw__gcm3',
                'struct_V':'struct_calc_all_V_cell_raw__A3',
                'struct_V_per_atom':'struct_calc_all_V_per_atom_raw__A3atom',
                'struct_dim_larsen':'struct_calc_all_dim_Larsen_raw__1'
            }
            nc = table[c]
        elif c.startswith('graph_Mproj_'):
            name = c.replace('graph_Mproj_','')
            name = name.replace('clust','clust_coeff').replace('algebraic_connectivity','laplacian_Fiedler')
            nc = f'graph_calc_Mproj_{name}__1'
        elif c.startswith('symm_'):
            nc = f'symm_calc_all_{c.replace("symm_","")}_raw__1'
        elif c == 'feat_cn_entropy':
            nc = 'struct_calc_all_cn_entropy_raw__1'
        elif c.startswith('voro_face_'):
            stat = c.replace('voro_face_','')
            nc = f'voro_calc_all_faces_{stat}__1'
        elif c.startswith('mm_') or c.startswith('mmauto_'):
            nc = c + unit_for(c)
        elif c.startswith('soap_') or c.startswith('acsf_') or c.startswith('mbtr_'):
            fam = 'SOAP' if c.startswith('soap_') else ('ACSF' if c.startswith('acsf_') else 'MBTR')
            if re.match(r'^(soap|acsf|mbtr)_(mean|std)_(\d+)$', c):
                _, stat, idx = re.match(r'^(soap|acsf|mbtr)_(mean|std)_(\d+)$', c).groups()
                nc = f'local_ds_{fam}_{idx}_{stat}__1'
            elif c in {'soap_mean_l2','soap_std_l2'}:
                stat = 'mean' if 'mean' in c else 'std'
                nc = f'local_ds_SOAP_l2_{stat}__1'
            else:
                nc = f'local_ds_{fam}_{c.split("_")[-1]}_raw__1'

        new[c] = nc

    return df.rename(columns=new)


def reorder_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_order = [
        'id_cif_path', 'id_cif', 'id_score', 'id_dim', 'id_connect',
        'cif_path', 'cif_file', 'score', 'dim', 'connect',  # 兼容旧名
    ]
    id_cols = [c for c in id_order if c in df.columns]
    other = [c for c in df.columns if c not in id_cols]
    return df[id_cols + other]


# ================ 并行稳健化（标准库）===============
def _init_worker_env():
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(k, '1')


def _worker_row(r: Dict[str, Any], args_dict: Dict[str, Any]) -> Dict[str, float]:
    global ARGS
    class _Args: pass
    A = _Args()
    for k, v in args_dict.items():
        setattr(A, k, v)
    ARGS = A
    _init_worker_env()
    return process_row(r, A.cif_root, A.nn, A.graph)


def _worker_wrapper(payload: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, float]:
    r, args_dict = payload
    return _worker_row(r, args_dict)


# ================ 主程序 ================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="输入 CSV（至少包含 cif_path）")
    ap.add_argument("--cif_root", default=".", help="cif_path 为相对路径时的根目录")
    ap.add_argument("--out_csv", default="extra_features.csv", help="输出 CSV 文件名")
    ap.add_argument("--nn", choices=["crystalnn", "voronoi"], default="crystalnn")
    ap.add_argument("--graph", action="store_true", help="启用图论特征（需要 networkx）")
    # optional modules
    ap.add_argument("--enable-magpie", action="store_true", dest="enable_magpie")
    ap.add_argument("--enable-matminer-comp", action="store_true")
    ap.add_argument("--enable-structure-fp", action="store_true")
    ap.add_argument("--enable-rdf", action="store_true")
    ap.add_argument("--rdf-cutoff", type=float, default=8.0)
    ap.add_argument("--rdf-bin", type=float, default=0.2)
    ap.add_argument("--adf-cutoff", type=float, default=6.0)
    ap.add_argument("--adf-bins", type=int, default=30)
    ap.add_argument("--enable-coulomb", action="store_true")
    ap.add_argument("--enable-opsf", action="store_true")
    ap.add_argument("--enable-xrd", action="store_true")
    ap.add_argument("--enable-matminer-auto", action="store_true")
    ap.add_argument("--enable-dscribe-soap", action="store_true")
    ap.add_argument("--enable-dscribe-acsf", action="store_true")
    ap.add_argument("--enable-dscribe-mbtr", action="store_true")
    ap.add_argument("--soap-rcut", type=float, default=5.0)
    ap.add_argument("--soap-nmax", type=int, default=8)
    ap.add_argument("--soap-lmax", type=int, default=6)
    ap.add_argument("--soap-sigma", type=float, default=0.4)
    ap.add_argument("--acsf-rcut", type=float, default=6.0)
    # misc
    ap.add_argument("--keep-raw", action="store_true", help="另存重命名前的 .raw.csv")
    ap.add_argument("--impute", choices=["none", "median", "zero"], default="none")
    ap.add_argument("--n_jobs", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "cif_path" not in df.columns and "id_cif_path" not in df.columns:
        raise SystemExit("ERROR: 输入 CSV 必须包含 cif_path 列（或已命名为 id_cif_path）")

    # 标准化 ID 列名（仅 ID）
    df = df.rename(columns={
        'cif_path':'id_cif_path','cif_file':'id_cif','score':'id_score','dim':'id_dim','connect':'id_connect',
    })
    rows_in = df.to_dict("records")

    # 并行参数打包（纯 dict，可序列化）
    args_dict = dict(
        cif_root=args.cif_root, nn=args.nn, graph=args.graph,
        enable_magpie=getattr(args, "enable_magpie", False),
        enable_matminer_comp=getattr(args, "enable_matminer_comp", False),
        enable_structure_fp=getattr(args, "enable_structure_fp", False),
        enable_rdf=getattr(args, "enable_rdf", False),
        rdf_cutoff=getattr(args, "rdf_cutoff", 8.0), rdf_bin=getattr(args, "rdf_bin", 0.2),
        adf_cutoff=getattr(args, "adf_cutoff", 6.0), adf_bins=getattr(args, "adf_bins", 30),
        enable_coulomb=getattr(args, "enable_coulomb", False),
        enable_opsf=getattr(args, "enable_opsf", False),
        enable_xrd=getattr(args, "enable_xrd", False),
        enable_matminer_auto=getattr(args, "enable_matminer_auto", False),
        enable_dscribe_soap=getattr(args, "enable_dscribe_soap", False),
        enable_dscribe_acsf=getattr(args, "enable_dscribe_acsf", False),
        enable_dscribe_mbtr=getattr(args, "enable_dscribe_mbtr", False),
        soap_rcut=getattr(args, "soap_rcut", 5.0), soap_nmax=getattr(args, "soap_nmax", 8),
        soap_lmax=getattr(args, "soap_lmax", 6), soap_sigma=getattr(args, "soap_sigma", 0.4),
        acsf_rcut=getattr(args, "acsf_rcut", 6.0),
        keep_raw=getattr(args, "keep_raw", False),
        impute=args.impute,
        n_jobs=args.n_jobs,
    )

    # 并行执行（标准库，保序）
    if args.n_jobs and args.n_jobs > 1:
        try:
            _init_worker_env()
            payloads = [(rd, args_dict) for rd in rows_in]
            with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
                # map 按输入顺序返回
                rows = list(ex.map(_worker_wrapper, payloads, chunksize=1))
        except Exception as e:
            print(f"[WARN] 并行失败，改用串行：{e}")
            rows = [_worker_row(r, args_dict) for r in rows_in]
    else:
        rows = [_worker_row(r, args_dict) for r in rows_in]

    fdf = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), fdf.reset_index(drop=True)], axis=1)

    if args.keep_raw:
        raw_path = os.path.splitext(args.out_csv)[0] + ".raw.csv"
        out.to_csv(raw_path, index=False)

    # 新命名 + 折叠重复列
    out = rename_columns(out)
    out = _collapse_duplicate_columns(out)

    # 缺失标记与插补（除 id_score）
    if args.impute != "none":
        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        if "id_score" in num_cols:
            num_cols.remove("id_score")
        for c in list(num_cols):
            col = out[c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            if col.isna().any() and not c.endswith("_missing"):
                out[f"{c}_missing"] = col.isna().astype(np.uint8)
        if args.impute == "median":
            med = out[num_cols].median(numeric_only=True)
            out[num_cols] = out[num_cols].fillna(med)
        elif args.impute == "zero":
            out[num_cols] = out[num_cols].fillna(0.0)

    out, dropped = drop_constant(out, thresh_unique=1)
    if dropped:
        print(f"[INFO] Dropped constant columns: {len(dropped)}")

    # ID 列置前
    out = reorder_id_columns(out)

    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} (rows={len(out)}, cols={out.shape[1]})")


if __name__ == "__main__":
    main()
