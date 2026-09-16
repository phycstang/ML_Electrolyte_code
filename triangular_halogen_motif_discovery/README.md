# Triangular Halogen Motif Discovery

This folder contains a reproducible, paper-oriented study of whether experimentally successful binary metal-halide precursors share a data-driven structural motif that can later be interpreted as a triangular/sixfold halogen arrangement.

## Scientific question

Can a common structural motif be discovered from complete binary metal-halide crystal structures **without** manually encoding a triangular lattice, sixfold order, `psi6`, exact-6 coordination, or an X-only sublattice?

## Dataset

Source repository assets:

- `MP_1702_cif_files.tar.gz`
- `mp_database/MP_1702_实验记录与结构去重表.csv`

The analysis keeps only experimentally observed, structure-representative, binary metal-halide structures.

Known positive formulas used for motif recurrence/enrichment:

- AlCl3 (`mp-25470`)
- FeCl3 (`mp-23204`)
- GaF3 (`mp-588`)
- InBr3 (`mp-570219`)
- TaCl5 (`mp-29831`)
- ZrCl4 (`mp-569175`)
- GaCl3 (`mp-30952`)

Blind formulas, excluded from all fitting and motif selection:

- InI3 (`mp-567789`)
- AlBr3 (`mp-23288`)
- ZnCl2 (`mp-22909`)
- SnCl2 (`mp-29179`)

## Representation

No atoms are removed. Every structure is converted to a complete anonymous periodic `M/X` framework:

- every metal -> generic `M`
- every F/Cl/Br/I -> generic `X`

The lattice is globally rescaled by the median nearest-neighbour distance calculated from **all atoms**, rather than an X-X-specific length, so the main representation does not privilege the halogen sublattice.

## Discovery routes

### A. Recurrent local SOAP motif

Each atomic site keeps its own local SOAP vector. No structure-level mean pooling is used.

For every local environment in the seven known positives, the code measures its best SOAP match in each of the other positive structures. Candidate motifs are ranked by recurrence across all positives and by rarity in the experimental background.

Only after a motif is selected is its geometry decoded using:

- center type (`M` or `X`)
- six nearest X distances
- radial coefficient of variation
- local planarity ratio
- `psi_m`, m=2...10

These quantities are **post-hoc interpretation only**.

### B. Fully unsupervised local-SOAP dictionary

All non-blind local environments are embedded by PCA and clustered with fixed MiniBatchKMeans dictionaries (`k=32,64,96`) without using labels.

After clustering, positive labels are overlaid and each cluster is tested for enrichment using a one-sided Fisher exact test with Benjamini-Hochberg FDR correction.

This gives a cleaner answer to: *does an independently discovered structural motif happen to be enriched among successful precursors?*

### C. Robustness

A pre-declared SOAP grid varies cutoff, radial basis size, angular basis size and Gaussian width. The hidden formulas remain excluded until each motif is frozen.

## Main script

```bash
python triangular_halogen_motif_discovery/deep_study.py \
  --cif-root data_cifs \
  --metadata 'mp_database/MP_1702_实验记录与结构去重表.csv' \
  --outdir triangular_halogen_motif_discovery/results
```

## Expected result files

- `results/RESULTS.md`
- `results/soap_robustness.csv`
- `results/matched_site_geometry.csv`
- `results/unsupervised_cluster_enrichment.csv`
- `results/unsupervised_hidden_validation.csv`
- `results/formula_motif_ranking_all.csv`
- `results/top50_unseen_candidates.csv`
- `results/deep_study.json`

## Interpretation rule

The study should not claim a *triangular halogen network* merely because a SOAP motif has high `psi6` after decoding. The strongest paper-level claim requires a separate topology check showing that the local sixfold environment extends through the halogen sublattice (for example, the existing exact-6/layer-connectivity analysis).

The intended evidence chain is:

`complete M-X CIF -> anonymous local representation -> motif discovery -> positive enrichment -> blind validation -> post-hoc sixfold decoding -> extended-network topology validation`.
