# Statistical validation

## Exact topology tests

| test                                                       |   positive_success |   positive_fail |   background_success |   background_fail |   odds_ratio |   p_one_sided |
|:-----------------------------------------------------------|-------------------:|----------------:|---------------------:|------------------:|-------------:|--------------:|
| known positives: strict exact-6 vs background              |                  5 |               2 |                  412 |               299 |        1.814 |        0.378  |
| known positives: fallback sixfold vs background            |                  6 |               1 |                  463 |               248 |        3.214 |        0.238  |
| core blind 3: strict exact-6 vs background (retrospective) |                  3 |               0 |                  412 |               299 |      inf     |        0.1958 |

With only seven known positives and a high background prevalence of exact-6/sixfold order, these tests are expected to have low power. A non-significant p-value is scientifically important: it prevents treating triangular order alone as a positive-specific descriptor.

## Blind SOAP robustness across the six pre-declared SOAP parameterizations

| formula   |   median_percentile_all6 |   min_percentile_all6 |   max_percentile_all6 |   n_configs_top20 |   n_configs_top10 |
|:----------|-------------------------:|----------------------:|----------------------:|------------------:|------------------:|
| InI3      |                   96.906 |                71.589 |                98.453 |                 5 |                 5 |
| AlBr3     |                  100.000 |                73.558 |               100.000 |                 5 |                 5 |
| ZnCl2     |                   90.999 |                43.319 |                94.515 |                 5 |                 4 |
| SnCl2     |                   59.845 |                53.305 |                72.293 |                 0 |                 0 |

## Source-motif stability

| origin_formula   | origin_center   |   n_of_6_configs |
|:-----------------|:----------------|-----------------:|
| GaCl3            | X               |                5 |
| ZrCl4            | M               |                1 |

The SOAP parameterizations are correlated representations of the same structures, so `n_of_6_configs` is a robustness count, not an independent-trial significance test.
