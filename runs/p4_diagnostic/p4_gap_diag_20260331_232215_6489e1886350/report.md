# P4 CNN Gap Diagnostic

- reference_mode: `static_linear`
- compare_modes: `window_variance, ekf, cnn_fpga`
- n_windows_per_scenario: `64`

## static_bias_theta

| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_variance | 0.630000 | 0.036615 | 0.025649 | 9.986030 | 0.367538 | 0.020262 | 0.862879 | 0.000879 |
| ekf | 0.609558 | 0.036112 | 0.026318 | 3.984121 | 0.360666 | 0.020493 | 0.862407 | 0.000828 |
| cnn_fpga | 0.471441 | 0.072658 | 0.035632 | 2.662283 | 0.316779 | 0.025466 | 0.816416 | 0.007641 |

## linear_ramp

| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_variance | 0.697298 | 0.015795 | 0.013823 | 7.821173 | 0.416183 | 0.007667 | 0.866012 | 0.001126 |
| ekf | 0.676856 | 0.013384 | 0.010526 | 3.796208 | 0.409788 | 0.007532 | 0.862345 | 0.000710 |
| cnn_fpga | 0.542777 | 0.052894 | 0.019575 | 1.688268 | 0.368897 | 0.012755 | 0.818158 | 0.007970 |

## step_sigma_theta

| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_variance | 0.690000 | 0.010384 | 0.007643 | 8.146933 | 0.410737 | 0.000612 | 0.864327 | 0.000866 |
| ekf | 0.669558 | 0.003173 | 0.002033 | 2.971095 | 0.404363 | 0.000396 | 0.862398 | 0.000559 |
| cnn_fpga | 0.535681 | 0.040555 | 0.011217 | 1.699611 | 0.363373 | 0.005567 | 0.818185 | 0.007873 |

## periodic_drift

| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_variance | 0.565108 | 0.016681 | 0.014285 | 7.818020 | 0.310029 | 0.008925 | 0.865763 | 0.001181 |
| ekf | 0.544665 | 0.014692 | 0.012516 | 4.767434 | 0.302964 | 0.008911 | 0.862411 | 0.000919 |
| cnn_fpga | 0.399444 | 0.053912 | 0.025431 | 3.666655 | 0.255632 | 0.014633 | 0.813366 | 0.007856 |
