# Formula-clean final results

This is the paper-facing rerun. **All polymorphs of every known and blind formula are excluded from the background/fitting pool.** One pre-specified MP structure per formula is used only for the intended known/blind role.

## SOAP recurrent-motif robustness

| config       | origin_formula   | origin_center   |   commonality_min |   background_prev95 |   posthoc_best_m |   posthoc_psi6 |   InI3_pct |   AlBr3_pct |   ZnCl2_pct |   SnCl2_pct |
|:-------------|:-----------------|:----------------|------------------:|--------------------:|-----------------:|---------------:|-----------:|------------:|------------:|------------:|
| short        | GaCl3            | X               |             0.883 |               0.108 |                6 |          0.990 |     97.328 |     100.000 |      91.561 |      63.150 |
| baseline     | GaCl3            | X               |             0.914 |               0.121 |                8 |          0.400 |     96.484 |     100.000 |      94.515 |      72.293 |
| medium       | GaCl3            | X               |             0.944 |               0.264 |                7 |          0.301 |     98.172 |     100.000 |      86.779 |      53.305 |
| wide         | ZrCl4            | M               |             0.988 |               0.608 |                4 |          0.254 |     71.589 |      73.558 |      43.319 |      56.540 |
| radial_rich  | GaCl3            | X               |             0.936 |               0.183 |                7 |          0.301 |     98.453 |     100.000 |      90.436 |      54.571 |
| angular_rich | GaCl3            | X               |             0.910 |               0.104 |                8 |          0.400 |     96.203 |     100.000 |      94.233 |      72.152 |

## Species-channel ablation of the frozen short-range motif

| block   | group   |   mean_score |   mean_percentile |   min_percentile |
|:--------|:--------|-------------:|------------------:|-----------------:|
| MM      | blind   |        0.832 |            82.595 |           63.291 |
| MM      | known   |        0.862 |            85.835 |           58.228 |
| MX      | blind   |        0.894 |            79.255 |           62.447 |
| MX      | known   |        0.905 |            81.636 |           58.931 |
| XX      | blind   |        0.974 |            86.674 |           62.025 |
| XX      | known   |        0.946 |            68.997 |           20.816 |

## Independent X-layer exact-6 screen

| group   | target_formula   | material_id   |   frac_exact6 |   mean_psi6_exact6 |   mean_psi4_exact6 | strict_exact6_all   | fallback_sixfold   |   h |   k |   l |
|:--------|:-----------------|:--------------|--------------:|-------------------:|-------------------:|:--------------------|:-------------------|----:|----:|----:|
| known   | AlCl3            | mp-25470      |         1.000 |              1.000 |              0.000 | True                | True               |   0 |   0 |   1 |
| known   | FeCl3            | mp-23204      |         1.000 |              0.858 |              0.140 | True                | True               |   0 |   0 |   1 |
| known   | GaF3             | mp-588        |         0.667 |              0.800 |              0.090 | False               | True               |   2 |   1 |   1 |
| known   | InBr3            | mp-570219     |         1.000 |              0.984 |              0.027 | True                | True               |   1 |   1 |  -1 |
| known   | TaCl5            | mp-29831      |         1.000 |              0.947 |              0.074 | True                | True               |   1 |   0 |   0 |
| known   | ZrCl4            | mp-569175     |         1.000 |              0.966 |              0.042 | True                | True               |   1 |  -1 |  -2 |
| known   | GaCl3            | mp-30952      |         0.000 |              0.000 |              0.000 | False               | False              |   1 |   1 |   1 |
| blind   | InI3             | mp-567789     |         1.000 |              0.984 |              0.022 | True                | True               |   1 |   2 |  -2 |
| blind   | AlBr3            | mp-23288      |         1.000 |              0.964 |              0.065 | True                | True               |   0 |   1 |  -1 |
| blind   | ZnCl2            | mp-22909      |         1.000 |              0.976 |              0.077 | True                | True               |   1 |   1 |  -2 |
| blind   | SnCl2            | mp-29179      |         0.000 |              0.000 |              0.000 | False               | False              |   1 |   1 |   1 |

Formula-clean background topology prevalence: `{"strict_exact6_all": 0.5794655414908579, "frac_exact6_ge_0.8": 0.5879043600562588, "frac_exact6_ge_0.5": 0.6722925457102672, "fallback_sixfold": 0.6511954992967651}`.

## Fully unsupervised hard-cluster test

No hard local-SOAP cluster covered >=5/7 known positives. This argues for a continuous/fuzzy motif family rather than one discrete prototype cluster.

## Interpretation

A triangular-halogen network is **not** treated as a proven universal necessary condition. The recurrent SOAP family is robustly X-centered for most parameterizations, while the independent exact-6 screen tests extended triangular order separately. The final claim must reflect both results and the substantial background prevalence of exact-6 order.
