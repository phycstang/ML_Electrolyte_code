# Local motif validation and formula-balanced ranking

Candidate ranking is formula-best, but all known/blind validation below uses only the pre-specified MP structure. REMatch and topology are independent views and are not collapsed into a tuned score.

## LOPO local-motif recovery

| config   |   lopo_mean_pct |   lopo_median_pct |   lopo_min_pct |   lopo_top20_count |   lopo_top10_count |
|:---------|----------------:|------------------:|---------------:|-------------------:|-------------------:|
| baseline |          64.004 |            60.093 |         40.371 |                  1 |                  1 |
| short    |          65.694 |            55.684 |         44.548 |                  3 |                  0 |

| config   | heldout_formula   |   heldout_formula_background_percentile | motif_origin_formula   | motif_origin_center_posthoc   |   train_commonality_min |
|:---------|:------------------|----------------------------------------:|:-----------------------|:------------------------------|------------------------:|
| short    | AlCl3             |                                  44.548 | GaCl3                  | M                             |                   0.859 |
| short    | FeCl3             |                                  81.903 | GaCl3                  | X                             |                   0.883 |
| short    | GaF3              |                                  54.060 | GaCl3                  | X                             |                   0.883 |
| short    | InBr3             |                                  55.684 | GaCl3                  | X                             |                   0.883 |
| short    | TaCl5             |                                  87.007 | GaCl3                  | X                             |                   0.883 |
| short    | ZrCl4             |                                  89.327 | GaCl3                  | X                             |                   0.883 |
| short    | GaCl3             |                                  47.332 | FeCl3                  | M                             |                   0.916 |
| baseline | AlCl3             |                                  40.371 | GaCl3                  | X                             |                   0.925 |
| baseline | FeCl3             |                                  76.566 | GaCl3                  | X                             |                   0.914 |
| baseline | GaF3              |                                  56.613 | GaCl3                  | X                             |                   0.914 |
| baseline | InBr3             |                                  60.093 | GaCl3                  | X                             |                   0.914 |
| baseline | TaCl5             |                                  77.494 | GaCl3                  | X                             |                   0.914 |
| baseline | ZrCl4             |                                  90.951 | GaCl3                  | X                             |                   0.914 |
| baseline | GaCl3             |                                  45.940 | TaCl5                  | X                             |                   0.976 |

## Random-seven empirical null

```json
{
  "n_permutations": 100,
  "seed": 20260916,
  "observed_discovery_score": 0.7486407344689779,
  "observed_commonality_min": 0.9218975901603699,
  "null_discovery_score_mean": 0.5337971736279861,
  "null_discovery_score_95pct": 0.7730688733804338,
  "null_commonality_min_mean": 0.8508970740437508,
  "null_commonality_min_95pct": 0.9528835535049438,
  "empirical_p_discovery_score": 0.06930693069306931,
  "empirical_p_commonality_min": 0.2871287128712871
}
```

## Fixed-structure blind robustness

| formula   | material_id   |   fixed_median_pct |   fixed_min_pct |   fixed_max_pct |   fixed_top20_count |   fixed_top10_count |
|:----------|:--------------|-------------------:|----------------:|----------------:|--------------------:|--------------------:|
| AlBr3     | mp-23288      |            100.000 |          64.965 |         100.000 |                   5 |                   5 |
| InI3      | mp-567789     |             95.708 |          61.949 |          97.680 |                   5 |                   5 |
| SnCl2     | mp-29179      |             46.288 |          37.587 |          63.109 |                   0 |                   0 |
| ZnCl2     | mp-22909      |             86.891 |          28.074 |          92.343 |                   5 |                   2 |

## Recurrent baseline motif family

|   motif_rank | origin_formula   | origin_center_posthoc   |   commonality_min |   formula_background_prev95 |   posthoc_best_m |   posthoc_psi6 |
|-------------:|:-----------------|:------------------------|------------------:|----------------------------:|-----------------:|---------------:|
|            1 | GaCl3            | X                       |             0.914 |                       0.169 |                8 |          0.400 |
|            2 | GaCl3            | X                       |             0.920 |                       0.197 |                6 |          0.990 |
|            3 | GaCl3            | X                       |             0.922 |                       0.206 |                7 |          0.301 |
|            4 | TaCl5            | X                       |             0.951 |                       0.536 |                6 |          0.616 |
|            5 | ZrCl4            | M                       |             0.972 |                       0.580 |                4 |          0.254 |
|            6 | ZrCl4            | X                       |             0.966 |                       0.745 |                6 |          0.803 |
|            7 | InBr3            | M                       |             0.962 |                       0.782 |                6 |          0.954 |
|            8 | GaF3             | X                       |             0.930 |                       0.787 |                5 |          0.251 |

## Top prospective formulas — local evidence primary

| formula   |   local_median_pct |   local_min_pct |   local_top20_count |   local_top10_count |   median_mean_percentile |   median_nearest_percentile |   topology_max_frac_exact6 |
|:----------|-------------------:|----------------:|--------------------:|--------------------:|-------------------------:|----------------------------:|---------------------------:|
| TiF4      |             97.796 |          90.023 |                   6 |                   6 |                   87.412 |                      83.263 |                      0.500 |
| CrF4      |             96.172 |          96.056 |                   6 |                   6 |                   88.115 |                      89.311 |                      0.750 |
| HfI4      |             95.708 |          91.647 |                   6 |                   6 |                   97.187 |                      95.921 |                      1.000 |
| ZrI4      |             95.592 |          91.879 |                   6 |                   6 |                   97.398 |                      94.937 |                      1.000 |
| VF4       |             92.575 |          91.415 |                   6 |                   6 |                   91.983 |                      91.491 |                      1.000 |
| HfCl4     |             91.995 |          90.951 |                   6 |                   6 |                   92.264 |                      99.297 |                      1.000 |
| VCl4      |             99.072 |          81.206 |                   6 |                   5 |                   31.927 |                      64.205 |                      1.000 |
| TiCl4     |             98.840 |          82.367 |                   6 |                   5 |                   44.233 |                      63.502 |                      1.000 |
| TiBr4     |             98.376 |          83.527 |                   6 |                   5 |                   53.727 |                      61.744 |                      1.000 |
| AuCl3     |             96.520 |          81.671 |                   6 |                   5 |                   87.342 |                      63.361 |                      0.167 |
| SnCl4     |             96.404 |          86.079 |                   6 |                   5 |                   67.229 |                      61.322 |                      1.000 |
| CrCl3     |             96.288 |          82.831 |                   6 |                   5 |                   99.156 |                      78.129 |                      1.000 |
| SnBr4     |             95.360 |          84.687 |                   6 |                   5 |                   70.886 |                      62.658 |                      1.000 |
| ReCl4     |             94.200 |          80.510 |                   6 |                   5 |                   88.467 |                      80.169 |                      1.000 |
| SnI4      |             92.923 |          88.167 |                   6 |                   5 |                   73.699 |                      65.541 |                      1.000 |
| MnF4      |             92.111 |          88.863 |                   6 |                   5 |                   94.796 |                      92.968 |                      0.688 |
| TcCl4     |             90.719 |          88.167 |                   6 |                   5 |                   95.077 |                      96.554 |                      1.000 |
| VF5       |             91.299 |          87.935 |                   6 |                   3 |                   78.692 |                      90.506 |                      0.400 |
| TiI4      |             90.371 |          87.935 |                   6 |                   3 |                   94.866 |                      97.398 |                      1.000 |
| MoCl4     |             90.255 |          86.079 |                   6 |                   3 |                   93.179 |                      95.288 |                      1.000 |

## Interpretation

A claim of a common local structural family should be supported by LOPO and the empirical null. Exact-6 is retained only as post-hoc interpretation because its background prevalence is high and its Fisher tests are non-significant. Whole-crystal REMatch remains a complementary control: disagreement with local SOAP is evidence that the signal is localized rather than a reason to force a single global metric.
