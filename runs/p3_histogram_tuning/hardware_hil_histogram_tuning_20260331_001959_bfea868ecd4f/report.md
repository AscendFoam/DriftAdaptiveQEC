# P3 Histogram-Input Tuning Sweep

| Candidate | Phase | syndrome_limit | histogram_range_limit | sigma_measurement | LER | Overflow | Hist Sat | Corr Sat | Agg Param | Dominant Source | dLER vs short baseline | dHist vs short baseline | dLER vs confirm baseline | dHist vs confirm baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| baseline_current | short |  |  |  | 1.064131 | 0.370342 | 0.370342 | 0.000000 | 0.000000 | histogram_input | 0.000000 | 0.000000 |  |  |
| hist_range_125 | short |  | 1.566643 |  | 1.061625 | 0.116658 | 0.116658 | 0.000000 | 0.000000 | histogram_input | -0.002506 | -0.253683 |  |  |
| hist_range_150 | short |  | 1.879971 |  | 1.063546 | 0.022328 | 0.022328 | 0.000000 | 0.000000 | histogram_input | -0.000585 | -0.348014 |  |  |
| syndrome_110_hist_150 | short | 1.378646 | 1.879971 |  | 1.059311 | 0.021757 | 0.021757 | 0.000000 | 0.000000 | histogram_input | -0.004820 | -0.348584 |  |  |
| hist_150_sigma_meas_004 | short |  | 1.879971 | 0.040000 | 1.064018 | 0.021483 | 0.021483 | 0.000000 | 0.000000 | histogram_input | -0.000114 | -0.348858 |  |  |
| hist_150_sigma_meas_003 | short |  | 1.879971 | 0.030000 | 1.065414 | 0.021523 | 0.021523 | 0.000000 | 0.000000 | histogram_input | 0.001282 | -0.348819 |  |  |
| syndrome_110_hist_150_sigma_meas_004 | short | 1.378646 | 1.879971 | 0.040000 | 1.062906 | 0.021584 | 0.021584 | 0.000000 | 0.000000 | histogram_input | -0.001225 | -0.348757 |  |  |
| syndrome_115_hist_150_sigma_meas_003 | short | 1.441311 | 1.879971 | 0.030000 | 1.061947 | 0.020906 | 0.020906 | 0.000000 | 0.000000 | histogram_input | -0.002184 | -0.349435 |  |  |
| baseline_current | confirm |  |  |  | 1.069503 | 0.387092 | 0.387092 | 0.000000 | 0.000000 | histogram_input | 0.005372 | 0.016751 | 0.000000 | 0.000000 |
| syndrome_115_hist_150_sigma_meas_003 | confirm | 1.441311 | 1.879971 | 0.030000 | 1.046678 | 0.022181 | 0.022181 | 0.000000 | 0.000000 | histogram_input | -0.017453 | -0.348161 | -0.022825 | -0.364912 |
| hist_150_sigma_meas_004 | confirm |  | 1.879971 | 0.040000 | 1.060576 | 0.022864 | 0.022864 | 0.000000 | 0.000000 | histogram_input | -0.003555 | -0.347478 | -0.008927 | -0.364229 |
| hist_150_sigma_meas_003 | confirm |  | 1.879971 | 0.030000 | 1.062218 | 0.022755 | 0.022755 | 0.000000 | 0.000000 | histogram_input | -0.001913 | -0.347586 | -0.007285 | -0.364337 |

## Selection

- Shortlist: syndrome_115_hist_150_sigma_meas_003, hist_150_sigma_meas_004, hist_150_sigma_meas_003, syndrome_110_hist_150_sigma_meas_004, syndrome_110_hist_150, hist_range_150, hist_range_125, baseline_current
- Long confirm: baseline_current, syndrome_115_hist_150_sigma_meas_003, hist_150_sigma_meas_004, hist_150_sigma_meas_003
