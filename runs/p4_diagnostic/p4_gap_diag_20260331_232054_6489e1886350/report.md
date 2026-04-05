# P4 CNN Gap Diagnostic

- reference_mode: `static_linear`
- compare_modes: `window_variance, ekf, cnn_fpga`
- n_windows_per_scenario: `16`

## static_bias_theta

| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_variance | 0.630000 | 0.037370 | 0.025726 | 6.250000 | 0.296780 | 0.016402 | 0.892911 | 0.000918 |
| ekf | 0.553699 | 0.040634 | 0.024308 | 2.467790 | 0.273054 | 0.016786 | 0.869483 | 0.000999 |
| cnn_fpga | 0.477564 | 0.081173 | 0.037427 | 2.421373 | 0.257856 | 0.021694 | 0.854112 | 0.007764 |
