#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discover common local motifs across known positives using full anonymous M-X SOAP.

No triangular, six-fold, X-only, or coordination constraint is used during discovery.
All atoms are retained and anonymized as M/X. Each structure is globally rescaled by
its complete-structure median nearest-neighbour distance. Local SOAP vectors are kept
site-by-site (no structure-level mean pooling).

For every local environment in the seven known positives, define a cross-positive
commonality score: its best SOAP cosine match in each *other* positive, summarized by
minimum and mean similarity. The highest-scoring environments are therefore motifs
that recur across all positives without specifying what geometry they should have.

Only after discovery, geometry of top environments is decoded (center type, six nearest
X neighbours, planarity and psi_m for m=2..10) and blind/background prevalence is tested.
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.preprocessing import normalize
from dscribe.descriptors import SOAP

HALOGENS = {"F", "Cl", "Br", "I"}
KNOWN_ID_BY_FORMULA = {
    "AlCl3":"mp-25470", "FeCl3":"mp-23204", "GaF3":"mp-588",
    "InBr3":"mp-570219", "TaCl5":"mp-29831", "ZrCl4":"mp-569175",
    "GaCl3":None,
}
BLIND_ID_BY_FORMULA = {"InI3":"mp-567789", "AlBr3":"mp-23288", "ZnCl2":"mp-22909", "SnCl2":"mp-29179"}


def is_binary_metal_halide(st):
    els=list(st.composition.elements)
    if len(els)!=2: return False
    syms={e.symbol for e in els}; hs=syms & HALOGENS
    if len(hs)!=1: return False
    other=next(s for s in syms if s not in HALOGENS)
    try: return bool(Element(other).is_metal)
    except Exception: return False


def d0_nn(st, r=10.0):
    out=[]
    for site in st:
        ds=[float(n.nn_distance) for n in st.get_neighbors(site,r) if float(n.nn_distance)>1e-7]
        if ds: out.append(min(ds))
    return float(np.median(out))


def anon_scaled(st):
    d0=d0_nn(st)
    sp=["He" if s.specie.symbol in HALOGENS else "H" for s in st]
    ss=Structure(Lattice(np.asarray(st.lattice.matrix)/d0),sp,st.frac_coords)
    return ss,d0


def select_idx(df, formula, mpid):
    sub=df[df.formula_norm==formula]
    if sub.empty: return None
    if mpid:
        h=sub[sub.material_id.astype(str)==mpid]
        if not h.empty: return int(h.index[0])
    def mpnum(x):
        try:return int(str(x).split('-')[-1])
        except:return 10**12
    return int(sorted(sub.index,key=lambda i:(mpnum(sub.loc[i,'material_id']),str(sub.loc[i,'cif_file'])))[0])


def build_soap():
    try:
        return SOAP(species=[1,2],periodic=True,r_cut=3.2,n_max=6,l_max=6,sigma=0.30,average="off",sparse=False)
    except TypeError:
        return SOAP(3.2,6,6,0.30,species=[1,2],periodic=True,average="off",sparse=False)


def local_soap(soap, st):
    at=AseAtomsAdaptor().get_atoms(st); at.set_pbc([True,True,True])
    X=np.asarray(soap.create(at),dtype=np.float32)
    return normalize(X,norm='l2',axis=1).astype(np.float32)


def psi_and_planarity(st, center_idx):
    """Post-hoc only: six nearest X around discovered center; no effect on discovery."""
    site=st[center_idx]
    neigh=[n for n in st.get_neighbors(site,4.0) if n.specie.symbol=="He"]
    # exclude zero-distance self images if any
    neigh=[n for n in neigh if n.nn_distance>1e-6]
    neigh=sorted(neigh,key=lambda n:n.nn_distance)[:6]
    if len(neigh)<6:
        return {"nX_available":len(neigh)}
    c=np.asarray(site.coords,float)
    V=np.asarray([np.asarray(n.coords,float)-c for n in neigh])
    d=np.linalg.norm(V,axis=1)
    C=V.T@V/max(len(V),1)
    evals,evecs=np.linalg.eigh(C); order=np.argsort(evals); evals=evals[order]; evecs=evecs[:,order]
    normal=evecs[:,0]; e1=evecs[:,2]; e2=evecs[:,1]
    x=V@e1; y=V@e2; ang=np.arctan2(y,x)
    psis={str(m):float(abs(np.mean(np.exp(1j*m*ang)))) for m in range(2,11)}
    best_m=max(psis,key=psis.get)
    return {
        "nX_available":6,
        "six_X_distances_scaled":[float(x) for x in d],
        "radial_cv":float(np.std(d)/max(np.mean(d),1e-12)),
        "planarity_ratio":float(evals[0]/max(np.sum(evals),1e-12)),
        "psi_m":psis,
        "best_m":int(best_m),
        "best_psi":float(psis[best_m]),
        "psi6":float(psis['6']),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cif-root',required=True); ap.add_argument('--metadata',required=True); ap.add_argument('--outdir',required=True)
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(a.metadata); meta['cif_file']=meta.cif_file.astype(str)
    m=meta.experimentally_observed.astype(str).str.lower().eq('yes') & meta.is_structure_representative.astype(str).str.lower().eq('true')
    meta=meta[m].copy(); paths={p.name:p for p in Path(a.cif_root).rglob('*.cif')}
    rows=[]; structs=[]; raw_structs=[]
    for _,r in meta.iterrows():
        p=paths.get(Path(str(r.cif_file)).name)
        if p is None: continue
        try:
            raw=Structure.from_file(str(p))
            if not is_binary_metal_halide(raw): continue
            st,d0=anon_scaled(raw)
        except Exception: continue
        rows.append({'material_id':str(r.material_id),'cif_file':p.name,'formula_norm':raw.composition.reduced_formula,'n_atoms':len(raw),'d0_A':d0})
        structs.append(st); raw_structs.append(raw)
    df=pd.DataFrame(rows).reset_index(drop=True)

    known={f:select_idx(df,f,mid) for f,mid in KNOWN_ID_BY_FORMULA.items()}; known={f:i for f,i in known.items() if i is not None}
    blind={f:select_idx(df,f,mid) for f,mid in BLIND_ID_BY_FORMULA.items()}; blind={f:i for f,i in blind.items() if i is not None}
    known_idx=list(known.values()); blind_idx=list(blind.values()); known_set=set(known_idx); blind_set=set(blind_idx)
    bg=[i for i in range(len(df)) if i not in known_set and i not in blind_set]
    print('DATASET',len(df),'structures',df.formula_norm.nunique(),'formulas')
    print('KNOWN',{f:str(df.loc[i,'material_id']) for f,i in known.items()})
    print('BLIND',{f:str(df.loc[i,'material_id']) for f,i in blind.items()})

    soap=build_soap(); local=[None]*len(df)
    # Compute positives/blinds first, then background for prevalence.
    order=known_idx+blind_idx+bg
    for k,i in enumerate(order):
        local[i]=local_soap(soap,structs[i])
        if (k+1)%100==0: print('SOAP local',k+1,'/',len(order))

    # Candidate environments are every atomic site in every known positive.
    candidates=[]
    for f,i in known.items():
        Xi=local[i]
        for si,v in enumerate(Xi):
            matches=[]
            per={}
            for f2,j in known.items():
                if j==i: continue
                s=float(np.max(local[j]@v))
                matches.append(s); per[f2]=s
            candidates.append({
                'origin_formula':f,'origin_material_id':str(df.loc[i,'material_id']),'origin_index':i,'site_index':si,
                'center_type':'X' if structs[i][si].specie.symbol=='He' else 'M',
                'commonality_min':float(min(matches)), 'commonality_mean':float(np.mean(matches)),
                'matches':per,
            })
    candidates.sort(key=lambda x:(x['commonality_min'],x['commonality_mean']),reverse=True)

    # Keep top diverse prototypes (SOAP cosine < 0.985 against already kept).
    selected=[]; selected_vec=[]
    for c in candidates:
        v=local[c['origin_index']][c['site_index']]
        if selected_vec and max(float(np.dot(v,u)) for u in selected_vec)>=0.985:
            continue
        cc=dict(c); cc['geometry_posthoc']=psi_and_planarity(structs[c['origin_index']],c['site_index'])
        selected.append(cc); selected_vec.append(v)
        if len(selected)>=12: break

    # For each selected motif, score every structure by its best local SOAP match.
    motif_results=[]
    for rank,(c,v) in enumerate(zip(selected,selected_vec),1):
        scores=np.asarray([float(np.max(local[i]@v)) for i in range(len(df))])
        bg_scores=scores[bg]
        blind_res={}
        for f,i in blind.items():
            blind_res[f]={
                'material_id':str(df.loc[i,'material_id']),
                'score':float(scores[i]),
                'percentile':float(100*np.mean(bg_scores<=scores[i])),
            }
        known_res={f:float(scores[i]) for f,i in known.items()}
        prevalence={str(t):float(np.mean(bg_scores>=t)) for t in (0.90,0.95,0.98)}
        motif_results.append({'rank':rank,**c,'known_best_match':known_res,'blind':blind_res,'background_prevalence':prevalence})

    payload={'n_structures':len(df),'n_formulas':int(df.formula_norm.nunique()),'motifs':motif_results}
    (out/'common_local_soap_motifs.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')

    lines=['# Common local SOAP motif discovery\n\n',f"Dataset: {len(df)} structures, {df.formula_norm.nunique()} formulas. Full anonymous M-X structures; no X-only filtering.\n\n"]
    lines.append('|rank|origin|center|min commonality|mean commonality|best m (post-hoc)|psi6|planarity|InI3 pct|AlBr3 pct|ZnCl2 pct|SnCl2 pct|\n')
    lines.append('|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in motif_results:
        g=r.get('geometry_posthoc',{}); b=r['blind']
        def pct(f): return b.get(f,{}).get('percentile',float('nan'))
        lines.append(f"|{r['rank']}|{r['origin_formula']} {r['origin_material_id']} site {r['site_index']}|{r['center_type']}|{r['commonality_min']:.4f}|{r['commonality_mean']:.4f}|{g.get('best_m','')}|{g.get('psi6',float('nan')):.3f}|{g.get('planarity_ratio',float('nan')):.4f}|{pct('InI3'):.1f}|{pct('AlBr3'):.1f}|{pct('ZnCl2'):.1f}|{pct('SnCl2'):.1f}|\n")
    lines.append('\n## Top motif details\n\n')
    for r in motif_results[:5]:
        lines.append(f"### Motif {r['rank']}: {r['origin_formula']} {r['origin_material_id']} site {r['site_index']} ({r['center_type']}-centered)\n\n")
        lines.append(f"- min/mean cross-positive SOAP similarity: {r['commonality_min']:.4f} / {r['commonality_mean']:.4f}\n")
        lines.append(f"- post-hoc geometry: `{json.dumps(r['geometry_posthoc'],ensure_ascii=False)}`\n")
        lines.append(f"- background prevalence >=0.95 SOAP: {100*r['background_prevalence']['0.95']:.1f}%\n")
        lines.append(f"- blind: `{json.dumps(r['blind'],ensure_ascii=False)}`\n\n")
    (out/'summary_local_motifs.md').write_text(''.join(lines),encoding='utf-8')
    df.to_csv(out/'dataset_used_local.csv',index=False)
    print(''.join(lines))

if __name__=='__main__': main()
