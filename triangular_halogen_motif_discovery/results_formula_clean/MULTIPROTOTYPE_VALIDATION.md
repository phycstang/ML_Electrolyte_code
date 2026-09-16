# Positive-only multi-prototype structural-family validation

Prototype selection uses only known-positive local SOAP environments and the facility-location representativeness objective. Background formulas are used only for percentile calibration. Blind structures are touched only after K is frozen by LOPO.

## 1. LOPO performance by K and SOAP setting

| config   |   k |   lopo_mean_pct |   lopo_median_pct |   lopo_min_pct |   lopo_top20_count |   lopo_top10_count |   train_min_coverage_mean |   train_mean_coverage_mean |
|:---------|----:|----------------:|------------------:|---------------:|-------------------:|-------------------:|--------------------------:|---------------------------:|
| baseline |   1 |          63.938 |            68.910 |         17.169 |                  1 |                  0 |                     0.969 |                      0.988 |
| baseline |   2 |          76.334 |            79.814 |         49.188 |                  3 |                  2 |                     0.987 |                      0.996 |
| baseline |   3 |          77.262 |            81.206 |         49.188 |                  4 |                  3 |                     0.994 |                      0.998 |
| baseline |   4 |          80.312 |            81.206 |         49.188 |                  5 |                  2 |                     0.997 |                      0.999 |
| short    |   1 |          69.473 |            74.710 |         16.705 |                  3 |                  1 |                     0.959 |                      0.985 |
| short    |   2 |          68.048 |            74.246 |         20.418 |                  2 |                  1 |                     0.977 |                      0.993 |
| short    |   3 |          69.175 |            75.174 |         20.186 |                  2 |                  0 |                     0.993 |                      0.998 |
| short    |   4 |          68.810 |            74.014 |         20.186 |                  1 |                  0 |                     0.995 |                      0.999 |

## 2. Pooled K selection across all 14 LOPO folds

|     k |   pooled_mean_pct |   pooled_median_pct |   pooled_min_pct |   pooled_top20_count |   pooled_top10_count |
|------:|------------------:|--------------------:|-----------------:|---------------------:|---------------------:|
| 4.000 |            74.561 |              79.582 |           20.186 |                6.000 |                2.000 |
| 3.000 |            73.218 |              77.610 |           20.186 |                6.000 |                3.000 |
| 2.000 |            72.191 |              74.942 |           20.418 |                5.000 |                3.000 |
| 1.000 |            66.705 |              73.086 |           16.705 |                4.000 |                1.000 |

Selected K = **4** by maximum pooled mean LOPO percentile (tie-break: smaller K).

## 3. Frozen prototypes trained on all seven known positives

| config   |   prototype_no | source_formula   | source_material_id   |   source_site | source_center_posthoc   |   posthoc_best_m |   posthoc_psi6 |   train_min_coverage_after_step |   train_mean_coverage_after_step |
|:---------|---------------:|:-----------------|:---------------------|--------------:|:------------------------|-----------------:|---------------:|--------------------------------:|---------------------------------:|
| short    |              1 | FeCl3            | mp-23204             |             4 | X                       |                5 |          0.609 |                           0.947 |                            0.985 |
| short    |              2 | GaCl3            | mp-30952             |             0 | M                       |                5 |          0.599 |                           0.981 |                            0.992 |
| short    |              3 | InBr3            | mp-570219            |             3 | X                       |                6 |          0.746 |                           0.991 |                            0.997 |
| short    |              4 | TaCl5            | mp-29831             |            24 | X                       |               10 |          0.212 |                           0.994 |                            0.999 |
| baseline |              1 | FeCl3            | mp-23204             |             3 | X                       |                5 |          0.609 |                           0.952 |                            0.987 |
| baseline |              2 | ZrCl4            | mp-569175            |             0 | M                       |                4 |          0.254 |                           0.985 |                            0.995 |
| baseline |              3 | GaCl3            | mp-30952             |             0 | M                       |                5 |          0.599 |                           0.992 |                            0.997 |
| baseline |              4 | InBr3            | mp-570219            |             3 | X                       |                6 |          0.746 |                           0.997 |                            0.999 |

## 4. Fixed-structure known and blind evaluation

### Known

| formula   | material_id   |   multiproto_median_pct |   multiproto_min_pct |   multiproto_max_pct |   configs_top20 |   configs_top10 |
|:----------|:--------------|------------------------:|---------------------:|---------------------:|----------------:|----------------:|
| AlCl3     | mp-25470      |                  80.394 |               75.638 |               85.151 |               1 |               0 |
| FeCl3     | mp-23204      |                 100.000 |              100.000 |              100.000 |               2 |               2 |
| GaCl3     | mp-30952      |                 100.000 |              100.000 |              100.000 |               2 |               2 |
| GaF3      | mp-588        |                  93.039 |               92.575 |               93.503 |               2 |               2 |
| InBr3     | mp-570219     |                 100.000 |              100.000 |              100.000 |               2 |               2 |
| TaCl5     | mp-29831      |                  92.807 |               85.615 |              100.000 |               2 |               1 |
| ZrCl4     | mp-569175     |                  95.012 |               90.023 |              100.000 |               2 |               2 |

### Blind

| formula   | material_id   |   multiproto_median_pct |   multiproto_min_pct |   multiproto_max_pct |   configs_top20 |   configs_top10 |
|:----------|:--------------|------------------------:|---------------------:|---------------------:|----------------:|----------------:|
| AlBr3     | mp-23288      |                  85.151 |               84.223 |               86.079 |               2 |               0 |
| InI3      | mp-567789     |                  40.835 |               33.179 |               48.492 |               0 |               0 |
| SnCl2     | mp-29179      |                  33.875 |               32.251 |               35.499 |               0 |               0 |
| ZnCl2     | mp-22909      |                  20.070 |               19.722 |               20.418 |               0 |               0 |

## 5. Top prospective formulas under the frozen multi-prototype family

| formula   |   multiproto_median_pct |   multiproto_min_pct |   configs_top20 |   configs_top10 |
|:----------|------------------------:|---------------------:|----------------:|----------------:|
| GdBr3     |                  99.884 |               99.768 |               2 |               2 |
| YCl3      |                  99.536 |               99.304 |               2 |               2 |
| RhBr3     |                  98.840 |               98.376 |               2 |               2 |
| IrBr3     |                  98.492 |               97.912 |               2 |               2 |
| IrCl3     |                  98.144 |               97.448 |               2 |               2 |
| YI3       |                  97.912 |               97.680 |               2 |               2 |
| FeBr3     |                  97.564 |               97.216 |               2 |               2 |
| RhCl3     |                  97.564 |               96.520 |               2 |               2 |
| ScCl3     |                  97.332 |               96.984 |               2 |               2 |
| CrBr3     |                  96.636 |               96.288 |               2 |               2 |
| VCl3      |                  96.636 |               96.056 |               2 |               2 |
| RuCl3     |                  95.940 |               95.592 |               2 |               2 |
| CrCl3     |                  95.940 |               95.128 |               2 |               2 |
| ZrCl3     |                  95.360 |               94.200 |               2 |               2 |
| TiCl3     |                  95.360 |               95.360 |               2 |               2 |
| ZrBr3     |                  94.896 |               93.735 |               2 |               2 |
| CrF3      |                  93.968 |               93.968 |               2 |               2 |
| CoF3      |                  93.968 |               93.039 |               2 |               2 |
| HfI3      |                  93.852 |               92.575 |               2 |               2 |
| TcCl3     |                  93.852 |               92.807 |               2 |               2 |

## Interpretation

If K>1 materially improves LOPO relative to K=1, the data support a **small family of local structural pathways** rather than one universal local motif. The selected prototypes should then be decoded physically (center type, coordination/connectivity, triangularity only post-hoc) and linked to reaction/reconstructability features in the next stage. If K=1 remains optimal or gains are weak, the evidence for a multi-pathway structural dictionary is also weak and should not be overstated.
