#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_features.py — 修复版（元素符号标准化 + mendeleev 正确取值 + 组合特征）
- EN 多标度：优先 element.electronegativity('<scale>')，回退 electronegativity_* 属性
- 热学/反应性/临界/三相：字段别名链 + 单位归一
- 半径：别名链兜底，pm→Å；离子半径兼容旧/新结构，优先 most_reliable
- 电子组态：从 e.ec 解析
- 组合特征：*_M/_H + d_* / abs_d_* / wa_* / ratio_*
"""

from __future__ import annotations
import os, re, json, argparse, warnings
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

# ---- 依赖（容错） ----
try:
    from mendeleev import element as mendel_element
except Exception:
    mendel_element = None

try:
    from pymatgen.core.composition import Composition
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PMG = True
except Exception:
    Composition = None
    Structure = None
    SpacegroupAnalyzer = None
    PMG = False

PM_TO_ANG = 0.01

def _is_num(x): 
    try: return np.isfinite(float(x))
    except: return False

def _to_float(x):
    try:
        if x is None: return None
        f = float(x)
        return f if np.isfinite(f) else None
    except: return None

def _median(vals: List[float]) -> float:
    vs = [float(v) for v in vals if _is_num(v)]
    if not vs: return np.nan
    vs.sort(); n=len(vs)
    return vs[n//2] if n%2 else 0.5*(vs[n//2-1]+vs[n//2])

# ---------- 元素获取 + 符号标准化 ----------
_CACHE: Dict[str, Any] = {}
def normalize_symbol(s: str) -> Optional[str]:
    if not isinstance(s, str): return None
    s = s.strip()
    if not s: return None
    # 去尾部数字/杂字符，如 "Cl1" → "Cl"
    s = re.sub(r'[^A-Za-z]', '', s)
    if not s: return None
    # 统一大小写（H, He, Cl）
    s = s[0].upper() + (s[1:].lower() if len(s) > 1 else '')
    # 特例：如果超过2个字母，只保留前2（极少见脏数据）
    if len(s) > 2: s = s[:2]
    return s

def get_elem(sym: Optional[str]):
    if not sym or mendel_element is None: return None
    sym = normalize_symbol(sym)
    if not sym: return None
    if sym in _CACHE: return _CACHE[sym]
    e = None
    # 尝试 1：原符号
    try: e = mendel_element(sym)
    except: e = None
    # 尝试 2：全小写/全大写
    if e is None:
        for t in (sym.lower(), sym.upper()):
            try:
                e = mendel_element(t)
                if e is not None: break
            except: pass
    _CACHE[sym] = e
    return e

# ---------- 标度/半径/标量定义 ----------
EN_SCALES = [
    "pauling", "allen", "allred_rochow", "martynov_batsanov", "nagle",
    "mulliken", "sanderson", "ghosh", "gordy", "li_xue", "cottrell_sutton",
    "gunnarsson_lundqvist", "miedema", "mullay", "robles_bartolotti"
]

RADII_DEFS = {
    "r_cov": ["covalent_radius_bragg", "covalent_radius_cordero", "covalent_radius"],
    "r_cov_pyykko_s": ["covalent_radius_pyykko_single", "covalent_radius_pyykko"],
    "r_cov_pyykko_d": ["covalent_radius_pyykko_double"],
    "r_cov_pyykko_t": ["covalent_radius_pyykko_triple"],
    "r_vdw_bondi": ["vdw_radius_bondi", "vdw_radius"],
    "r_vdw_batsanov": ["vdw_radius_batsanov", "vdw_radius"],
    "r_metal_c12": ["metallic_radius_c12", "metallic_radius"],
}
def get_attr_chain(e, names: List[str]):
    for n in names:
        try: v = getattr(e, n, None)
        except: v = None
        fv = _to_float(v)
        if fv is not None: return fv
    return None

def get_en(e, scale: str):
    if e is None: return None
    # 方法优先（官方文档示例）
    method_token = {
        "pauling":"pauling","allen":"allen","mulliken":"mulliken","sanderson":"sanderson",
        "ghosh":"ghosh","gordy":"gordy","li_xue":"li-xue","cottrell_sutton":"cottrell-sutton",
        "nagle":"nagle","martynov_batsanov":"martynov-batsanov","allred_rochow":"allred-rochow",
    }.get(scale.lower())
    if method_token:
        try:
            # Use the built-in electronegativity method; for Li–Xue scale this token is 'li-xue'【511001520993768†L71-L111】
            v = e.electronegativity(method_token)
            f = float(v)
            if np.isfinite(f): return f
        except Exception:
            pass
    # 属性回退（文档列出 electronegativity_*；个别用 en_*）
    attr = {
        "pauling":"electronegativity_pauling",
        "allen":"electronegativity_allen",
        "mulliken":"electronegativity_mulliken",
        "sanderson":"electronegativity_sanderson",
        "nagle":"electronegativity_nagle",
        "martynov_batsanov":"electronegativity_martynov_batsanov",
        "gordy":"electronegativity_gordy",
        "li_xue":"electronegativity_li_xue",
        "cottrell_sutton":"electronegativity_cottrell_sutton",
        "ghosh":"electronegativity_ghosh",
        "allred_rochow":"electronegativity_allred_rochow",
        "miedema":"en_miedema","mullay":"en_mullay","robles_bartolotti":"en_robles_bartolotti",
        "gunnarsson_lundqvist":"en_gunnarsson_lundqvist",
    }.get(scale.lower())
    if attr:
        try:
            # fallback: access attribute directly; for Li–Xue scale this uses electronegativity_li_xue【511001520993768†L71-L111】
            v = getattr(e, attr, None)
            if v is not None:
                f = float(v)
                if np.isfinite(f): return f
        except Exception:
            pass
    return None

EXT_SCALARS = {
    "IE1": lambda e: _to_float(getattr(e, "ionenergies", {}).get(1)) if hasattr(e,"ionenergies") else _to_float(getattr(e,"ionization_energy",None)),
    "IE2": lambda e: _to_float(getattr(e, "ionenergies", {}).get(2)) if hasattr(e,"ionenergies") else None,
    "IE3": lambda e: _to_float(getattr(e, "ionenergies", {}).get(3)) if hasattr(e,"ionenergies") else None,
    "EA":  lambda e: _to_float(getattr(e,"electron_affinity",None)),
    "alpha": lambda e: (_to_float(getattr(e,"dipole_polarizability",None)) or _to_float(getattr(e,"polarizability",None))),
    "C6": lambda e: _to_float(getattr(e,"c6",None)),
    "MN": lambda e: _to_float(getattr(e,"mendeleev_number",None)),
    "group": lambda e: (_to_float(getattr(e,"group_id",None)) or _to_float(getattr(e,"group",None))),
    "period": lambda e: _to_float(getattr(e,"period",None)),
    "Z": lambda e: _to_float(getattr(e,"atomic_number",None)),
    "mass": lambda e: _to_float(getattr(e,"atomic_weight",None)),
    "atomic_volume": lambda e: _to_float(getattr(e,"atomic_volume",None)),
    "density": lambda e: _to_float(getattr(e,"density",None)),
}

EC_KEYS = ["ec","electron_configuration","electronic_configuration"]
EC_TOKEN_RE = re.compile(r'(\d+)([spdf])(\d+)', re.IGNORECASE)
def parse_ec_counts(e) -> Dict[str,float]:
    out = {"EC_tot_s":np.nan,"EC_tot_p":np.nan,"EC_tot_d":np.nan,"EC_tot_f":np.nan,
           "EC_valence_n":np.nan,"EC_val_s":np.nan,"EC_val_p":np.nan,"EC_val_d":np.nan,"EC_val_f":np.nan,"EC_val_total_e":np.nan}
    if e is None: return out
    cfg = None
    for k in EC_KEYS:
        try:
            v = getattr(e,k,None)
            if v: cfg = str(v); break
        except: pass
    if not cfg: return out
    cfg = re.sub(r'\[[^\]]+\]','',cfg)
    toks = EC_TOKEN_RE.findall(cfg)
    if not toks: return out
    totals = {"s":0,"p":0,"d":0,"f":0}; per_n = {}; maxn = 0
    for n,orb,occ in toks:
        n=int(n); orb=orb.lower(); occ=int(occ)
        totals[orb]+=occ
        per_n.setdefault(n,{"s":0,"p":0,"d":0,"f":0})[orb]+=occ
        if n>maxn: maxn=n
    out["EC_tot_s"],out["EC_tot_p"],out["EC_tot_d"],out["EC_tot_f"]=[float(totals[k]) for k in ("s","p","d","f")]
    if maxn>0:
        out["EC_valence_n"]=float(maxn)
        val = per_n.get(maxn,{"s":0,"p":0,"d":0,"f":0})
        out["EC_val_s"],out["EC_val_p"],out["EC_val_d"],out["EC_val_f"]=[float(val[k]) for k in ("s","p","d","f")]
        out["EC_val_total_e"]=float(sum(val.values()))
    return out

def get_mendeleev_extra_numeric(e) -> Dict[str, Optional[float]]:
    out = {k: np.nan for k in [
        "Tm_K","Tb_K","Cp_molar_J_molK","Cp_specific_J_gK","Hvap_kJ_mol","Hfus_kJ_mol",
        "k_W_mK","alpha_T_1e6K","hardness_eV","softness_1_per_eV","electrophilicity",
        "heat_of_formation_kJ_mol","gas_basicity_kJ_mol","proton_affinity_kJ_mol",
        "critical_T_K","critical_P_MPa","triple_T_K","triple_P_kPa",
        "miedema_phi_star","miedema_nws","miedema_Vm_cm3_mol",
        "lattice_constant_A","pettifor_number","glawe_number","work_function_eV",
    ]}
    if e is None: return out
    def _fv(names):
        for nm in names:
            try: v = getattr(e,nm,None)
            except: v = None
            if v is not None:
                try:
                    f=float(v)
                    if np.isfinite(f): return f
                except: pass
        return np.nan
    out["Tm_K"] = _fv(["melting_point","melting_point_k"])
    out["Tb_K"] = _fv(["boiling_point","boiling_point_k"])
    out["Cp_molar_J_molK"]  = _fv(["specific_heat_capacity_molar","cp_molar","molar_heat_capacity"])
    out["Cp_specific_J_gK"] = _fv(["specific_heat_capacity","cp_specific"])
    Hvap = _fv(["heat_of_vaporization","enthalpy_of_vaporization","Hvap"])
    if _is_num(Hvap): out["Hvap_kJ_mol"] = Hvap/1000.0 if Hvap>2e3 else Hvap
    Hfus = _fv(["heat_of_fusion","enthalpy_of_fusion","Hfus"])
    if _is_num(Hfus): out["Hfus_kJ_mol"] = Hfus/1000.0 if Hfus>2e3 else Hfus
    out["k_W_mK"] = _fv(["thermal_conductivity","k"])
    a = _fv(["thermal_expansion","alpha"])
    out["alpha_T_1e6K"] = a*1e6 if _is_num(a) and a<1e-2 else (a if _is_num(a) else np.nan)
    out["hardness_eV"]       = _fv(["hardness"])
    out["softness_1_per_eV"] = _fv(["softness"])
    out["electrophilicity"]  = _fv(["electrophilicity"])
    hof = _fv(["heat_of_formation","enthalpy_of_formation"])
    out["heat_of_formation_kJ_mol"] = hof if _is_num(hof) else np.nan
    out["gas_basicity_kJ_mol"]      = _fv(["gas_basicity"])
    out["proton_affinity_kJ_mol"]   = _fv(["proton_affinity"])
    # Critical and triple points: use official mendeleev attribute names
    # critical_temperature: critical temperature in K (stored)【692084507847046†L254-L270】
    out["critical_T_K"]   = _fv(["critical_temperature","critical_T_K"])  # fallback to old key if exists
    # critical_pressure: in MPa (stored)【692084507847046†L254-L270】
    out["critical_P_MPa"] = _fv(["critical_pressure","critical_pressure_mpa"])
    # triple_point_temperature: triple point temperature in K
    out["triple_T_K"]     = _fv(["triple_point_temperature","triple_T_K"])
    # triple_point_pressure: triple point pressure in kPa
    out["triple_P_kPa"]   = _fv(["triple_point_pressure","triple_P_kPa"])
    # Miedema parameters: use correct attribute names
    out["miedema_phi_star"]   = _fv(["miedema_phi_star","phi_star","miedema_phi"])
    # miedema_electron_density: electron density at Wigner-Seitz cell【692084507847046†L686-L701】
    out["miedema_nws"]        = _fv(["miedema_electron_density","electron_density_ws","n_ws"])
    # miedema_molar_volume: molar volume (cm^3/mol)【692084507847046†L686-L701】
    Vm = _fv(["miedema_molar_volume","molar_volume_miedema","miedema_Vm"])
    out["miedema_Vm_cm3_mol"] = Vm if _is_num(Vm) else np.nan
    out["lattice_constant_A"] = _fv(["lattice_constant","lattice_constant_A"])
    out["pettifor_number"]    = _fv(["pettifor_number"])
    out["glawe_number"]       = _fv(["glawe_number"])
    out["work_function_eV"]   = _fv(["work_function","work_function_eV"])
    return out

CATEGORICAL_BASES = ["series","geochemical_class","goldschmidt_class","lattice_structure"]
def get_mendeleev_extra_categoricals(e) -> Dict[str, Any]:
    out = {f"{b}_str": None for b in CATEGORICAL_BASES}
    if e is None: return out
    for b in CATEGORICAL_BASES:
        try: v = getattr(e, b, None)
        except: v = None
        out[f"{b}_str"] = (str(v) if v is not None else None)
    return out

def build_element_block(sym: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    e = get_elem(sym)
    if e is None: return out  # 若符号非法，整体为空（后续会打缺失标记）
    # EN
    for sc in EN_SCALES:
        out[f"EN_{sc}"] = get_en(e, sc)
    # Radii
    for rname, chain in RADII_DEFS.items():
        out[rname] = get_attr_chain(e, chain)
    # Scalars
    for pname, getter in EXT_SCALARS.items():
        out[pname] = getter(e)
    # Extras & EC & categorical
    out.update(get_mendeleev_extra_numeric(e))
    out.update(parse_ec_counts(e))
    out.update(get_mendeleev_extra_categoricals(e))
    return out

def _ionic_radius(sym: str, ox: int) -> Optional[float]:
    e = get_elem(sym)
    if e is None: return None
    # 旧：dict
    for name in ("ionic_radius","ionic_radius_pm"):
        try: d = getattr(e,name,None)
        except: d = None
        if isinstance(d, dict):
            val = _to_float(d.get(ox))
            if val is not None:
                return val*PM_TO_ANG if "pm" in name else val
    # 新：列表（Shannon）
    try:
        entries = getattr(e,"ionic_radii",None)
        if entries:
            cand=[]
            for r in entries:
                ch = getattr(r,"charge",None)
                if ch is None or int(ch)!=int(ox): continue
                val_pm = _to_float(getattr(r,"ionic_radius",None))
                mr = bool(getattr(r,"most_reliable",False))
                if val_pm is not None: cand.append((mr,val_pm))
            if cand:
                vals = [v for mr,v in cand if mr] or [v for mr,v in cand]
                return _median(vals)*PM_TO_ANG
    except: pass
    # 兜底：共价半径
    return get_attr_chain(e, RADII_DEFS["r_cov"])

def diffize(prefix: str, dM: Dict[str,Any], dH: Dict[str,Any],
            with_abs: bool, numeric_keys: List[str],
            wM: Optional[float], wH: Optional[float]) -> Dict[str,Any]:
    out: Dict[str,Any] = {}
    # 端点
    for k,v in dM.items(): out[f"{prefix}{k}_M"] = v
    for k,v in dH.items(): out[f"{prefix}{k}_H"] = v
    # 权重
    if not (_is_num(wM) and _is_num(wH) and (wM+wH)>0): wM,wH = 0.5,0.5
    # 组合
    for k in numeric_keys:
        vM, vH = dM.get(k, np.nan), dH.get(k, np.nan)
        d = (float(vM)-float(vH)) if (_is_num(vM) and _is_num(vH)) else np.nan
        out[f"{prefix}d_{k}"] = d
        if with_abs: out[f"{prefix}abs_d_{k}"] = abs(d) if _is_num(d) else np.nan
        out[f"{prefix}wa_{k}"] = (wM*float(vM)+wH*float(vH)) if (_is_num(vM) and _is_num(vH)) else np.nan
        out[f"{prefix}ratio_{k}"] = (float(vM)/float(vH)) if (_is_num(vM) and _is_num(vH) and float(vH)!=0.0) else np.nan
    return out

def t0_comp_features(M: str, H: str, nM: float, nH: float, row: Dict[str,Any]) -> Dict[str,Any]:
    out={}
    tot = (nM+nH) if (_is_num(nM) and _is_num(nH)) else np.nan
    xM = (nM/tot) if (_is_num(tot) and tot>0) else np.nan
    xH = (nH/tot) if (_is_num(tot) and tot>0) else np.nan
    s = (lambda x: -x*np.log(x) if (_is_num(x) and x>0) else 0.0)
    out["T0_comp__entropy"] = float(s(xM)+s(xH)) if (_is_num(xM) and _is_num(xH)) else np.nan
    def geo(a,b):
        return float(np.sqrt(a*b)) if (_is_num(a) and _is_num(b) and a>0 and b>0) else np.nan
    chiP_M = row.get("T0_elem__EN_pauling_M", np.nan)
    chiP_H = row.get("T0_elem__EN_pauling_H", np.nan)
    gm = geo(chiP_M, chiP_H)
    if not _is_num(gm):
        enM=[row.get(f"T0_elem__EN_{sc}_M",np.nan) for sc in EN_SCALES]
        enH=[row.get(f"T0_elem__EN_{sc}_H",np.nan) for sc in EN_SCALES]
        m = float(np.nanmean([v for v in enM if _is_num(v)])) if any(_is_num(v) for v in enM) else np.nan
        h = float(np.nanmean([v for v in enH if _is_num(v)])) if any(_is_num(v) for v in enH) else np.nan
        gm = geo(m,h)
    out["T0_comp__chi_geomean"] = gm if _is_num(gm) else np.nan
    C6M,C6H = row.get("T0_elem__C6_M",np.nan), row.get("T0_elem__C6_H",np.nan)
    out["T0_comp__C6_geomean"] = geo(C6M,C6H)
    # 摩尔质量（可选）
    out["T0_comp__molar_mass"] = np.nan
    try:
        if PMG:
            comp = Composition({M:nM, H:nH})
            out["T0_comp__molar_mass"] = float(comp.weight)
    except: pass
    return out

def _mean_std(x):
    v=[float(t) for t in x if _is_num(t)]
    if not v: return (np.nan,np.nan)
    a=np.array(v); return (float(a.mean()), float(a.std(ddof=0)))

def build_t1_features(struct, M: str, Hal: str) -> Dict[str,Any]:
    out={}
    try:
        L=struct.lattice
        out["T1_struct__volume_A3"] = float(L.volume)
        out["T1_struct__volume_per_atom_A3"] = float(L.volume/len(struct))
        out["T1_struct__density_g_cm3"] = float(getattr(struct,"density",np.nan))
        out["T1_struct__a_A"]=float(L.a); out["T1_struct__b_A"]=float(L.b); out["T1_struct__c_A"]=float(L.c)
        out["T1_struct__alpha_deg"]=float(L.alpha); out["T1_struct__beta_deg"]=float(L.beta); out["T1_struct__gamma_deg"]=float(L.gamma)
        if SpacegroupAnalyzer is not None:
            try:
                sga = SpacegroupAnalyzer(struct, symprec=1e-2)
                out["T1_struct__spacegroup_number"] = float(sga.get_space_group_number())
                out["T1_struct__spacegroup_symbol"] = str(sga.get_space_group_symbol())
            except: pass
    except: pass
    return out

def build_t2_features(struct, M: str, Hal: str) -> Dict[str,Any]:
    out={}
    try:
        eM,eH = get_elem(M), get_elem(Hal)
        rM = get_attr_chain(eM, RADII_DEFS["r_cov"]) or 1.2
        rH = get_attr_chain(eH, RADII_DEFS["r_cov"]) or 0.8
        cutoff = 1.25*(rM+rH)
        cn, dists = [], []
        for i,site in enumerate(struct):
            if str(site.specie)!=M: continue
            local=[]
            for j,site2 in enumerate(struct):
                if i==j or str(site2.specie)!=Hal: continue
                d=float(site.distance(struct[j]))
                if d<=cutoff: local.append(d)
            if local:
                cn.append(len(local)); dists += local
        out["T2_env__M_CN_Hal_mean"], out["T2_env__M_CN_Hal_std"] = _mean_std(cn)
        out["T2_env__MHal_bond_length_A_mean"], out["T2_env__MHal_bond_length_A_std"] = _mean_std(dists)
    except: pass
    return out

def infer_M_H(formula: str, m_col_val: Optional[str], h_col_val: Optional[str]) -> Tuple[Optional[str],Optional[str],float,float]:
    # 1) 若显式给了 M/X 列，优先用并标准化
    M = normalize_symbol(m_col_val) if m_col_val else None
    H = normalize_symbol(h_col_val) if h_col_val else None
    nM = nH = np.nan
    # 2) 如未提供或缺失，尝试用 formula 解析
    if (not M or not H) and PMG and isinstance(formula,str) and formula.strip():
        try:
            comp = Composition(formula)
            d = comp.get_el_amt_dict()
            # 识别卤素
            halogens = {"F","Cl","Br","I","At","Ts"}
            H = H or next((k for k in d.keys() if k in halogens), None)
            M = M or next((k for k in d.keys() if k not in halogens), None)
            nH = float(d.get(H,np.nan)); nM = float(d.get(M,np.nan))
        except: pass
    # 3) 仍没有，则只做符号标准化（尽力而为）
    return (M, H, nM, nH)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m-col", default=None, help="金属列名（可选）")
    ap.add_argument("--x-col", default=None, help="卤素列名（可选）")
    ap.add_argument("--with_abs_delta", type=int, default=1)
    ap.add_argument("--enable-t1", type=int, default=1)
    ap.add_argument("--enable-t2", type=int, default=1)
    ap.add_argument("-v","--verbose", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    base_cols = [c for c in ("cif_path","formula","name","id","score","dim","st1","st2","st3") if c in df.columns]

    rows=[]
    for _,r in df.iterrows():
        row={}
        for c in base_cols: row[c]=r.get(c,None)
        formula = r.get("formula", None)
        # 正确写法（argparse 会把 --m-col 映射成 args.m_col）：
        m_col = getattr(args, "m_col", None)
        x_col = getattr(args, "x_col", None)

        # 从当行记录 r（pandas.Series）安全取值；列不存在时返回 None
        m_raw = r.get(m_col, None) if m_col else None
        x_raw = r.get(x_col, None) if x_col else None
        M,H,nM,nH = infer_M_H(formula, m_raw, x_raw)

        if not (M and H):
            # 连符号都拿不到：仅打标记
            row["T0_comp__stoich_ratio_Hal_over_M"] = np.nan
            row["T0_flags__missing_formula__bool"] = 1.0
            rows.append(row); continue

        # 若计量未知，至少给 1:1，避免 wa_* 全空
        if not _is_num(nM): nM = 1.0
        if not _is_num(nH): nH = 1.0
        row["T0_comp__stoich_ratio_Hal_over_M"] = float(nH/nM) if (nM>0) else np.nan

        dM_all = build_element_block(M)
        dH_all = build_element_block(H)

        # 分离分类字符串键
        str_keys = [f"{b}_str" for b in CATEGORICAL_BASES]
        dM_str = {k:dM_all.pop(k) for k in list(dM_all.keys()) if k in str_keys}
        dH_str = {k:dH_all.pop(k) for k in list(dH_all.keys()) if k in str_keys}

        numeric_keys = sorted(set(dM_all.keys()) | set(dH_all.keys()))  # 取并集，避免某侧缺失导致整列空

        tot = nM+nH
        wM = (nM/tot) if (tot and _is_num(tot) and tot>0) else 0.5
        wH = 1.0 - wM

        row.update(diffize("T0_elem__", dM_all, dH_all, bool(args.with_abs_delta), numeric_keys, wM, wH))

        for k,v in dM_str.items(): row[f"T0_elem__{k}_M"]=v
        for k,v in dH_str.items(): row[f"T0_elem__{k}_H"]=v

        row.update(t0_comp_features(M,H,nM,nH,row))

        # T1/T2（若有结构）
        cif = r.get("cif_path", None)
        if (args.enable_t1 or args.enable_t2) and isinstance(cif,str) and os.path.isfile(cif) and (Structure is not None):
            try:
                struct = Structure.from_file(cif)
                if args.enable_t1: row.update(build_t1_features(struct,M,H))
                if args.enable_t2: row.update(build_t2_features(struct,M,H))
            except: row["T1_flags__failed_read_cif__bool"]=1.0

        rows.append(row)

    out = pd.DataFrame(rows)

    # 分类编码 & 差分
    for b in CATEGORICAL_BASES:
        mcol = f"T0_elem__{b}_str_M"; hcol = f"T0_elem__{b}_str_H"
        if mcol in out and hcol in out:
            cats = pd.Categorical(list(out[mcol].fillna("")) + list(out[hcol].fillna("")))
            mapping = {cat:i for i,cat in enumerate(cats.categories)}
            out[f"T0_elem__{b}_code_M"] = out[mcol].fillna("").map(mapping).astype(float)
            out[f"T0_elem__{b}_code_H"] = out[hcol].fillna("").map(mapping).astype(float)
            out[f"T0_elem__d_{b}_code"] = out[f"T0_elem__{b}_code_M"] - out[f"T0_elem__{b}_code_H"]
            if args.with_abs_delta:
                out[f"T0_elem__abs_d_{b}_code"] = out[f"T0_elem__d_{b}_code"].abs()

    # 数值缺失标志
    for c in out.columns:
        if out[c].dtype != "O":
            miss = out[c].isna().astype(float)
            if miss.sum()>0: out[f"{c}_missing"] = miss

    out.to_csv(args.out, index=False)
    if args.verbose:
        print(f"[OK] rows={len(out)} cols={out.shape[1]} saved -> {args.out}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
