#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mx_features_T0T1T2.py — Unified T0 + T1 + T2 features for single-metal single-halogen (MX) materials

- T0: composition-only (no ratio; selective deltas; family aggregation with robust z + median/IQR/MAD + coverage/sign)
- T1: structure-lite (lattice/volume/density/spacegroup and per-cell M/Hal counts), needs CIF path per row (column: 'cif_path')
- T2: local environment / graph descriptors (M–Hal bonds, CN, bond-angle stats), from structure only
- Clear names; avoid min/max unless absolutely needed (we still avoid min/max here)

CLI example:
  python mx_features_T0T1T2.py \
    --csv data_clean_dedup.csv \
    --out-stem mx_all \
    --out-format csv \
    --extra-dir extras_opt \
    --enable-t1 1 --enable-t2 1 \
    --no_validate 0

Requirements: numpy, pandas, pymatgen, mendeleev (for T0). T1/T2 only run when structure is available.
"""

from __future__ import annotations
import os, sys, json, argparse, glob, math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# ---------- Optional deps ----------
try:
    from mendeleev import element as mendel_element
except Exception:
    mendel_element = None

try:
    from pymatgen.core.composition import Composition
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except Exception:
    Composition = None
    Structure = None
    CrystalNN = None
    VoronoiNN = None
    SpacegroupAnalyzer = None

HALOGENS = {"F","Cl","Br","I"}
VERSION = "mx-t0t1t2-1.0"

# ========================= T0 (reuse MX-only logic; no ratio) =========================
class ElementStore:
    def __init__(self):
        self.tbl: Dict[str, Dict[str, float]] = {}
        self.available_props: set[str] = set()
        self._init_from_mendeleev()
        self.prop_center: Dict[str,float] = {}
        self.prop_scale: Dict[str,float] = {}

    def _init_from_mendeleev(self):
        if mendel_element is None:
            return
        for Z in range(1, 119):
            try:
                e = mendel_element(Z)
                sym = e.symbol
                rec = {
                    "chi_Pauling": e.en_pauling,
                    "chi_Allen": e.en_allen,
                    "chi_AllredRochow": e.en_allred_rochow,
                    "chi_MartynovBatsanov": getattr(e, "en_martynov_batsanov", None),
                    "r_cov_Cordero": e.covalent_radius,
                    "r_vdw_Bondi": e.vdw_radius,
                    "r_metallic_c12": e.atomic_radius,
                    "alpha": e.polarizability,
                    "C6": e.c6,
                    "IE1": e.ionenergies.get(1, np.nan) if getattr(e,"ionenergies",None) else np.nan,
                    "EA": e.electron_affinity,
                    "Z": e.atomic_number,
                    "mass": e.atomic_weight,
                    "MN": e.mendeleev_number,
                    "group": e.group_id,
                    "period": e.period,
                }
                self.tbl[sym] = {k: (np.nan if v is None else float(v)) for k,v in rec.items()}
            except Exception:
                continue
        self._refresh_props()

    def _refresh_props(self):
        props=set()
        for rec in self.tbl.values(): props.update(rec.keys())
        self.available_props = props

    def load_extra_dir(self, extra_dir: Optional[str]):
        if not extra_dir or not os.path.isdir(extra_dir): return
        for path in glob.glob(os.path.join(extra_dir,"*.csv")):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if "symbol" not in df.columns: continue
            cols = [c for c in df.columns if c!="symbol"]
            for _,row in df.iterrows():
                sym = str(row["symbol"]).strip()
                if sym not in self.tbl: self.tbl[sym] = {}
                for c in cols:
                    val = row[c]
                    if pd.isna(val): continue
                    try:
                        self.tbl[sym][c] = float(val)
                    except Exception:
                        pass
        self._refresh_props()

    def build_scale_calibration(self):
        self.prop_center = {}; self.prop_scale = {}
        for prop in self.available_props:
            vals = np.array([rec.get(prop, np.nan) for rec in self.tbl.values()], float)
            vals = vals[~np.isnan(vals)]
            if vals.size >= 10:
                med = float(np.median(vals)); mad = float(np.median(np.abs(vals-med))) or 1.0
            else:
                med, mad = 0.0, 1.0
            self.prop_center[prop] = med; self.prop_scale[prop] = mad

    def get(self, sym: str, prop: str) -> float:
        try:
            v = self.tbl.get(sym, {}).get(prop, np.nan)
            return np.nan if v is None else float(v)
        except Exception:
            return np.nan

FAMILIES = {
    "chi": {"keys":["chi_Pauling","chi_Allen","chi_AllredRochow","chi_MartynovBatsanov"], "keep_delta": True},
    "radius": {"keys":["r_cov_Cordero","r_vdw_Bondi","r_metallic_c12"], "keep_delta": False},
    "polar": {"keys":["alpha"], "keep_delta": False},
    "c6": {"keys":["C6"], "keep_delta": False},
    "IE": {"keys":["IE1"], "keep_delta": True},
    "EA": {"keys":["EA"], "keep_delta": True},
    "mulliken_chi": {"keys":["mulliken_chi"], "keep_delta": True},
    "hardness_eta": {"keys":["hardness_eta"], "keep_delta": True},
    "basic": {"keys":["MN","group","period","Z","mass"], "keep_delta": True},
}

def assign_extra_to_families(store: ElementStore):
    for k in sorted(store.available_props):
        if k.startswith("chi_") and k not in FAMILIES["chi"]["keys"]:
            FAMILIES["chi"]["keys"].append(k)
        if (k.startswith("r_cov_") or k.startswith("r_vdw_") or k.startswith("r_ion_") or k=="r_metallic_c12") and k not in FAMILIES["radius"]["keys"]:
            FAMILIES["radius"]["keys"].append(k)
        if (k.startswith("alpha") or k.startswith("pol_")) and k not in FAMILIES["polar"]["keys"] and k!="alpha":
            FAMILIES["polar"]["keys"].append(k)
        if (k.startswith("C6") or k.startswith("c6_")) and k not in FAMILIES["c6"]["keys"] and k!="C6":
            FAMILIES["c6"]["keys"].append(k)
        if k in {"IE2","IE3"} and k not in FAMILIES["IE"]["keys"]:
            FAMILIES["IE"]["keys"].append(k)

def parse_mx(formula: str):
    if Composition is None: return None, None, {}
    comp = Composition(formula)
    d = {el: float(amt) for el, amt in comp.get_el_amt_dict().items()}
    hal = [(el,amt) for el,amt in d.items() if el in HALOGENS]
    met = [(el,amt) for el,amt in d.items() if el not in HALOGENS]
    hal.sort(key=lambda x:-x[1]); met.sort(key=lambda x:-x[1])
    Hal = hal[0][0] if hal else None
    M = met[0][0] if met else None
    return M, Hal, d

def stoich_ratio_hal_over_m(d):
    nH = float(sum(v for k,v in d.items() if k in HALOGENS))
    nM = float(sum(v for k,v in d.items() if k not in HALOGENS))
    return np.nan if nM<=0 else nH/nM

def zify(x, c, s):
    if x is None or (isinstance(x,float) and np.isnan(x)): return np.nan
    return (float(x) - c) / (s if s!=0 else 1.0)

def aggregate_family(store: ElementStore, family: str, M: str, Hal: str) -> Dict[str,float]:
    keys = FAMILIES[family]["keys"]
    out = {}
    per_scale = {}
    for prop in keys:
        if prop in {"mulliken_chi","hardness_eta"}:
            IE1_M, EA_M = store.get(M,"IE1"), store.get(M,"EA")
            IE1_H, EA_H = store.get(Hal,"IE1"), store.get(Hal,"EA")
            if prop=="mulliken_chi":
                vM = np.nan if (np.isnan(IE1_M) or np.isnan(EA_M)) else 0.5*(IE1_M+EA_M)
                vH = np.nan if (np.isnan(IE1_H) or np.isnan(EA_H)) else 0.5*(IE1_H+EA_H)
            else:
                vM = np.nan if (np.isnan(IE1_M) or np.isnan(EA_M)) else (IE1_M-EA_M)
                vH = np.nan if (np.isnan(IE1_H) or np.isnan(EA_H)) else (IE1_H-EA_H)
        else:
            vM = store.get(M, prop); vH = store.get(Hal, prop)
        per_scale[prop] = {
            "M_value": vM,
            "Hal_value": vH,
            "delta_M_minus_Hal": (np.nan if (np.isnan(vM) or np.isnan(vH)) else float(vM - vH)),
        }

    med, iqr, mad, cov, signc = {}, {}, {}, {}, {}
    forms = ["M_value","Hal_value"] + (["delta_M_minus_Hal"] if FAMILIES[family]["keep_delta"] else [])
    for form in forms:
        zvals=[]; used=0
        for prop in keys:
            val = per_scale[prop][form]
            if np.isnan(val): continue
            c = store.prop_center.get(prop,0.0); s = store.prop_scale.get(prop,1.0)
            zv = zify(val, c, s)
            if np.isnan(zv): continue
            zvals.append(zv); used += 1
        if used==0:
            med[form]=np.nan; iqr[form]=np.nan; mad[form]=np.nan; cov[form]=0.0; signc[form]=np.nan
        else:
            arr = np.array(zvals,float)
            q25,q50,q75 = np.percentile(arr,[25,50,75])
            med[form]=float(q50)
            iqr[form]=float(q75-q25)
            mad[form]=float(np.median(np.abs(arr-q50)))
            cov[form]=used/len(keys)
            if "delta" in form and q50!=0:
                signc[form]=float(np.mean(np.sign(arr)==np.sign(q50)))
            else:
                signc[form]=np.nan

    for form in ["M_value","Hal_value"]:
        out[f"T0_{family}_family__{form}__median"] = med[form]
        out[f"T0_{family}_family__{form}__iqr"] = iqr[form]
        out[f"T0_{family}_family__{form}__mad"] = mad[form]
        out[f"T0_{family}_family__{form}__coverage"] = cov[form]
    if FAMILIES[family]["keep_delta"]:
        form = "delta_M_minus_Hal"
        out[f"T0_{family}_family__{form}__median"] = med[form]
        out[f"T0_{family}_family__{form}__iqr"] = iqr[form]
        out[f"T0_{family}_family__{form}__mad"] = mad[form]
        out[f"T0_{family}_family__{form}__coverage"] = cov[form]
        out[f"T0_{family}_family__{form}__sign_consistency"] = signc[form]
    return out

def ionic_character_from_pauling(store, M, Hal):
    pM = store.get(M,"chi_Pauling"); pH = store.get(Hal,"chi_Pauling")
    if np.isnan(pM) or np.isnan(pH): return np.nan
    d = float(pM - pH)
    return 1.0 - math.exp(-0.25*(d*d))

def reduced_mass(store, M, Hal):
    mM = store.get(M,"mass"); mH = store.get(Hal,"mass")
    if np.isnan(mM) or np.isnan(mH) or (mM+mH)==0: return np.nan
    return float((mM*mH)/(mM+mH))

def hsab_strength(store, M, Hal):
    IE1_M, EA_M = store.get(M,"IE1"), store.get(M,"EA")
    IE1_H, EA_H = store.get(Hal,"IE1"), store.get(Hal,"EA")
    if any(np.isnan(x) for x in [IE1_M,EA_M,IE1_H,EA_H]): return np.nan
    chiM = 0.5*(IE1_M+EA_M); chiH = 0.5*(IE1_H+EA_H)
    etaM = IE1_M - EA_M; etaH = IE1_H - EA_H
    den = abs(etaM) + abs(etaH)
    if den<=0: return np.nan
    return float(((chiM-chiH)**2)/den)

def kapustinskii_proxy(store, M, Hal, zM):
    rM = store.get(M,"r_cov_Cordero"); rH = store.get(Hal,"r_cov_Cordero")
    if np.isnan(rM) or np.isnan(rH):
        rM = store.get(M,"r_vdw_Bondi"); rH = store.get(Hal,"r_vdw_Bondi")
    if np.isnan(rM) or np.isnan(rH) or (rM+rH)<=0: return np.nan
    zH = 1.0
    return float(abs(zM)*abs(zH)/(rM+rH))

def build_t0_features(store: ElementStore, formula: str, validate_mx=True) -> Dict[str,Any]:
    out = {}
    if not formula:
        out["T0_flags__missing_formula__bool"]=1.0; return out
    if Composition is None:
        out["T0_flags__pymatgen_missing__bool"]=1.0; return out
    M, Hal, d = parse_mx(formula)
    out["T0_meta__metal_symbol"] = M
    out["T0_meta__halogen_symbol"] = Hal
    sto = stoich_ratio_hal_over_m(d)
    out["T0_comp__stoich_ratio_Hal_over_M"] = sto
    for hx in ["F","Cl","Br","I"]:
        out[f"T0_comp__Hal_is_{hx}"] = 1.0 if Hal==hx else 0.0
    if validate_mx:
        nH = sum(1 for k,v in d.items() if k in HALOGENS and v>0)
        nM = sum(1 for k,v in d.items() if k not in HALOGENS and v>0)
        if nH!=1 or nM!=1: out["T0_flags__not_single_MX__bool"]=1.0
    if not M or not Hal:
        out["T0_flags__failed_parse_M_or_Hal__bool"]=1.0; return out

    for fam in ["chi","radius","polar","c6","IE","EA","mulliken_chi","hardness_eta","basic"]:
        out.update(aggregate_family(store, fam, M, Hal))

    out["T0_chi_ionic_character__Pauling"] = ionic_character_from_pauling(store,M,Hal)
    out["T0_pair__reduced_mass_amu"] = reduced_mass(store,M,Hal)
    out["T0_hsab__interaction_strength"] = hsab_strength(store,M,Hal)
    out["T0_lattice_energy__kapustinskii_proxy"] = kapustinskii_proxy(store,M,Hal, sto)
    return out

# ========================= T1: structure-lite =========================
def build_t1_features(struct: Structure, M: str, Hal: str) -> Dict[str,Any]:
    out = {}
    try:
        vol = float(struct.lattice.volume)
        na = float(len(struct.sites))
        out["T1_struct__volume_A3"] = vol
        out["T1_struct__volume_per_atom_A3"] = (vol/na if na>0 else np.nan)
        out["T1_struct__density_g_cm3"] = float(struct.density) if hasattr(struct, "density") else np.nan
        L = struct.lattice
        out["T1_struct__a_A"] = float(L.a); out["T1_struct__b_A"] = float(L.b); out["T1_struct__c_A"] = float(L.c)
        out["T1_struct__alpha_deg"] = float(L.alpha); out["T1_struct__beta_deg"] = float(L.beta); out["T1_struct__gamma_deg"] = float(L.gamma)
        if SpacegroupAnalyzer is not None:
            try:
                sga = SpacegroupAnalyzer(struct, symprec=1e-2)
                out["T1_struct__spacegroup_number"] = float(sga.get_space_group_number())
                out["T1_struct__spacegroup_symbol"] = sga.get_space_group_symbol()
            except Exception:
                out["T1_struct__spacegroup_number"] = np.nan
                out["T1_struct__spacegroup_symbol"] = None
        # Per-cell counts for M and Hal
        m_cnt = sum(1 for site in struct if site.specie.symbol==M)
        h_cnt = sum(1 for site in struct if site.specie.symbol==Hal)
        out["T1_struct__count_M_per_cell"] = float(m_cnt)
        out["T1_struct__count_Hal_per_cell"] = float(h_cnt)
    except Exception:
        # leave NaNs
        pass
    return out

# ========================= T2: local environment / graph =========================
def _neighbors_analyzer():
    # Prefer CrystalNN (robust for crystals); fallback to VoronoiNN; else None
    if CrystalNN is not None:
        try:
            return CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        except Exception:
            pass
    if VoronoiNN is not None:
        try:
            return VoronoiNN()
        except Exception:
            pass
    return None

def build_t2_features(struct: Structure, M: str, Hal: str) -> Dict[str,Any]:
    out = {}
    try:
        nn = _neighbors_analyzer()
        if nn is None:
            out["T2_flags__nn_unavailable__bool"]=1.0; return out

        # Collect bonds of type M–Hal (undirected)
        m_indices = [i for i,s in enumerate(struct) if s.specie.symbol==M]
        h_indices = [i for i,s in enumerate(struct) if s.specie.symbol==Hal]
        if len(m_indices)==0 or len(h_indices)==0:
            out["T2_flags__missing_species__bool"]=1.0; return out

        dists = []
        cn_M_list = []
        angles_deg = []  # Hal-M-Hal

        for im in m_indices:
            neighs = nn.get_nn_info(struct, im)
            # count only Hal neighbors
            hal_neighs = [n for n in neighs if getattr(n["site"].specie, "symbol", None)==Hal]
            cn_M_list.append(float(len(hal_neighs)))
            for n in hal_neighs:
                d = float(n["site"].distance(struct[im]))
                dists.append(d)
            # angles around M with halide neighbors
            for i in range(len(hal_neighs)):
                for j in range(i+1, len(hal_neighs)):
                    v1 = hal_neighs[i]["site"].coords - struct[im].coords
                    v2 = hal_neighs[j]["site"].coords - struct[im].coords
                    # angle between v1 and v2
                    a = float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12), -1.0, 1.0))))
                    angles_deg.append(a)

        # Aggregate without min/max
        def _mean_std(xs):
            xs = np.array(xs, float)
            xs = xs[~np.isnan(xs)]
            if xs.size==0: return np.nan, np.nan
            return float(np.mean(xs)), float(np.std(xs))

        out["T2_env__M_CN_Hal_mean"] , out["T2_env__M_CN_Hal_std"]  = _mean_std(cn_M_list)
        out["T2_env__MHal_bond_length_A_mean"], out["T2_env__MHal_bond_length_A_std"] = _mean_std(dists)
        out["T2_env__Hal_M_Hal_angle_deg_mean"], out["T2_env__Hal_M_Hal_angle_deg_std"] = _mean_std(angles_deg)

        # Simple connectivity proxy: fraction of Hal neighbors among all neighbors for M
        frac_hal_neigh = []
        for im in m_indices:
            neighs = nn.get_nn_info(struct, im)
            if len(neighs)==0: continue
            frac = np.mean([1.0 if getattr(n["site"].specie,"symbol",None)==Hal else 0.0 for n in neighs])
            frac_hal_neigh.append(frac)
        out["T2_env__M_neighbor_fraction_Hal_mean"], out["T2_env__M_neighbor_fraction_Hal_std"] = _mean_std(frac_hal_neigh)

    except Exception:
        out["T2_flags__exception__bool"]=1.0
    return out

# ========================= Main driver =========================
def main():
    ap = argparse.ArgumentParser("mx_features_T0T1T2")
    ap.add_argument("--csv", required=True, help="Input CSV with 'formula' and optionally 'cif_path'")
    ap.add_argument("--out-stem", default="mx_all_features")
    ap.add_argument("--out-format", default="csv", choices=["csv","parquet"])
    ap.add_argument("--extra-dir", default=None)
    ap.add_argument("--enable-t1", type=int, default=1)
    ap.add_argument("--enable-t2", type=int, default=1)
    ap.add_argument("--no_validate", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "formula" not in df.columns:
        print("ERROR: need 'formula' column", file=sys.stderr); sys.exit(2)

    # T0 prep
    store = ElementStore()
    store.load_extra_dir(args.extra_dir)
    assign_extra_to_families(store)
    store.build_scale_calibration()

    results = []
    for rec in df.to_dict("records"):
        row = {}
        # T0
        row.update(build_t0_features(store, rec.get("formula",""), validate_mx=(args.no_validate==0)))

        # T1/T2 if structure exists and enabled
        struct = None
        cif_path = rec.get("cif_path", None)
        if (args.enable_t1 or args.enable_t2) and cif_path and isinstance(cif_path, str) and os.path.isfile(cif_path) and Structure is not None:
            try:
                struct = Structure.from_file(cif_path)
            except Exception:
                row["T1_flags__failed_read_cif__bool"] = 1.0

        M = row.get("T0_meta__metal_symbol", None)
        Hal = row.get("T0_meta__halogen_symbol", None)

        if struct is not None and M and Hal:
            if args.enable_t1:
                row.update(build_t1_features(struct, M, Hal))
            if args.enable_t2:
                row.update(build_t2_features(struct, M, Hal))
        else:
            if args.enable_t1 or args.enable_t2:
                row["T1T2_flags__structure_missing_or_symbols__bool"] = 1.0

        results.append(row)

    out = pd.DataFrame(results)
    id_cols = [c for c in ["name","id","formula","cif_path","score"] if c in df.columns]
    out = pd.concat([df[id_cols].reset_index(drop=True) if id_cols else pd.DataFrame(), out.reset_index(drop=True)], axis=1)

    # Write
    if args.out_format=="parquet":
        out_path = f"{args.out_stem}.parquet"; out.to_parquet(out_path, index=False)
    else:
        out_path = f"{args.out_stem}.csv"; out.to_csv(out_path, index=False)

    schema = {
        "version": VERSION,
        "notes": {
            "no_ratio": True,
            "delta_policy": "chi, mulliken_chi, IE, EA, hardness_eta, basics only",
            "aggregation": "per-scale robust z (median/MAD), then median/IQR/MAD + coverage/sign_consistency",
            "t1_enabled": bool(args.enable_t1),
            "t2_enabled": bool(args.enable_t2),
            "mx_validation": args.no_validate==0
        },
        "families": {k: {"n_keys": len(v["keys"]), "keep_delta": v["keep_delta"]} for k,v in FAMILIES.items()},
        "out": out_path
    }
    with open(f"{args.out_stem}.schema.json","w",encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {out_path} and {args.out_stem}.schema.json")

if __name__ == "__main__":
    main()
