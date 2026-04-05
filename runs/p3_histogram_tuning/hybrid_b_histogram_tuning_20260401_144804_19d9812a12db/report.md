# P3 Histogram-Input Tuning Sweep

| Candidate | Phase | syndrome_limit | histogram_range_limit | sigma_measurement | LER | Overflow | Hist Sat | Corr Sat | Agg Param | Dominant Source | dLER vs short baseline | dHist vs short baseline | dLER vs confirm baseline | dHist vs confirm baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| baseline_current | short |  |  |  | 1.203069 | 0.022990 | 0.022990 | 0.000000 | 0.000000 | histogram_input | 0.000000 | 0.000000 |  |  |
| hist_range_165 | short |  | 2.067970 |  | 1.186035 | 0.007674 | 0.007674 | 0.000000 | 0.000000 | histogram_input | -0.017035 | -0.015316 |  |  |
| hist_range_180 | short |  | 2.255966 |  | 1.179376 | 0.002935 | 0.002935 | 0.000000 | 0.000000 | histogram_input | -0.023694 | -0.020054 |  |  |
| syndrome_120_hist_165 | short | 1.503977 | 2.067970 |  | 1.182569 | 0.007531 | 0.007531 | 0.000000 | 0.000000 | histogram_input | -0.020500 | -0.015458 |  |  |
| syndrome_120_hist_180 | short | 1.503977 | 2.255966 |  | 1.176324 | 0.003001 | 0.003001 | 0.000000 | 0.000000 | histogram_input | -0.026745 | -0.019989 |  |  |
| syndrome_125_hist_180 | short | 1.566643 | 2.255966 |  | 1.175063 | 0.002897 | 0.002897 | 0.000000 | 0.000000 | histogram_input | -0.028006 | -0.020093 |  |  |
| baseline_current | confirm |  |  |  | 1.388708 | 0.021543 | 0.021543 | 0.000000 | 0.000000 | histogram_input | 0.185639 | -0.001447 | 0.000000 | 0.000000 |
| syndrome_125_hist_180 | confirm | 1.566643 | 2.255966 |  | 1.375373 | 0.002756 | 0.002756 | 0.000000 | 0.000000 | histogram_input | 0.172304 | -0.020234 | -0.013335 | -0.018787 |
| hist_range_180 | confirm |  | 2.255966 |  | 1.378728 | 0.002785 | 0.002785 | 0.000000 | 0.000000 | histogram_input | 0.175659 | -0.020204 | -0.009980 | -0.018758 |
| syndrome_120_hist_180 | confirm | 1.503977 | 2.255966 |  | 1.377988 | 0.002802 | 0.002802 | 0.000000 | 0.000000 | histogram_input | 0.174919 | -0.020188 | -0.010720 | -0.018741 |

## Selection

- Shortlist: syndrome_125_hist_180, hist_range_180, syndrome_120_hist_180, syndrome_120_hist_165, hist_range_165, baseline_current
- Long confirm: baseline_current, syndrome_125_hist_180, hist_range_180, syndrome_120_hist_180
