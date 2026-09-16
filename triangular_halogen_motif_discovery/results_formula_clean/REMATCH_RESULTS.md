# Full-MX REMatch SOAP validation

REMatch compares the **complete sets of local M/X environments** between crystals. No X-only filtering or triangular descriptors are used. All known/blind formula polymorphs are removed from the background pool.

## LOPO summary

| soap_config   |   alpha |   lopo_mean_percentile_mean_agg |   lopo_min_percentile_mean_agg |   lopo_mean_percentile_nearest_agg |   lopo_min_percentile_nearest_agg |
|:--------------|--------:|--------------------------------:|-------------------------------:|-----------------------------------:|----------------------------------:|
| short         |   0.100 |                          75.929 |                         19.409 |                             72.031 |                            20.113 |
| short         |   1.000 |                          73.980 |                         18.706 |                             76.331 |                            19.691 |
| short         |  10.000 |                          73.980 |                         18.987 |                             76.492 |                            20.956 |
| baseline      |   0.100 |                          75.990 |                         20.253 |                             75.367 |                            21.800 |
| baseline      |   1.000 |                          76.492 |                         27.707 |                             75.708 |                            23.347 |
| baseline      |  10.000 |                          77.115 |                         34.880 |                             75.688 |                            24.473 |

## Blind consensus across all 6 REMatch settings

| formula   |   median_mean_percentile |   min_mean_percentile |   median_nearest_percentile |   min_nearest_percentile |   top20_mean_count |   top10_mean_count |
|:----------|-------------------------:|----------------------:|----------------------------:|-------------------------:|-------------------:|-------------------:|
| AlBr3     |                   26.160 |                20.956 |                      88.889 |                   84.529 |                  0 |                  0 |
| InI3      |                   74.754 |                69.198 |                      60.689 |                   30.098 |                  0 |                  0 |
| SnCl2     |                   34.037 |                33.193 |                      31.153 |                   27.567 |                  0 |                  0 |
| ZnCl2     |                   28.622 |                22.082 |                      19.761 |                   19.128 |                  0 |                  0 |

## Top 20 prospective formulas by REMatch mean-similarity consensus

| formula   |   median_mean_percentile |   min_mean_percentile |   median_nearest_percentile |   min_nearest_percentile |   settings |
|:----------|-------------------------:|----------------------:|----------------------------:|-------------------------:|-----------:|
| PtCl3     |                   99.930 |                99.859 |                      85.724 |                   84.248 |          6 |
| PtBr3     |                   99.789 |                99.578 |                      88.045 |                   86.217 |          6 |
| TiBr3     |                   99.508 |                98.875 |                      92.194 |                   90.999 |          6 |
| TiCl3     |                   99.367 |                97.328 |                      91.421 |                   90.155 |          6 |
| PtI3      |                   99.297 |                98.594 |                      84.599 |                   83.544 |          6 |
| CrCl3     |                   99.156 |                98.734 |                      78.129 |                   73.277 |          6 |
| OsBr4     |                   99.086 |                96.203 |                      89.662 |                   88.326 |          6 |
| OsCl4     |                   98.664 |                94.093 |                      88.889 |                   86.920 |          6 |
| MoCl3     |                   98.594 |                94.796 |                      91.772 |                   91.561 |          6 |
| TcBr4     |                   97.890 |                93.530 |                      94.023 |                   92.405 |          6 |
| TaCl4     |                   97.398 |                95.077 |                      87.412 |                   84.248 |          6 |
| ZrI4      |                   97.398 |                94.515 |                      94.937 |                   94.515 |          6 |
| PtI4      |                   97.257 |                97.046 |                      75.598 |                   70.886 |          6 |
| HfI4      |                   97.187 |                93.952 |                      95.921 |                   95.359 |          6 |
| TaI4      |                   97.187 |                94.374 |                      78.762 |                   74.824 |          6 |
| NbCl4     |                   97.117 |                94.515 |                      86.990 |                   84.388 |          6 |
| NbBr4     |                   96.906 |                94.233 |                      82.560 |                   78.200 |          6 |
| WCl4      |                   96.765 |                94.796 |                      80.942 |                   78.481 |          6 |
| TcCl3     |                   96.695 |                91.983 |                      94.023 |                   93.671 |          6 |
| TiI3      |                   96.484 |                93.812 |                      83.615 |                   81.575 |          6 |

Interpretation: mean aggregation tests similarity to the positive set as a whole; nearest aggregation allows multiple structural mechanisms. Neither is selected post-hoc as the sole metric.
