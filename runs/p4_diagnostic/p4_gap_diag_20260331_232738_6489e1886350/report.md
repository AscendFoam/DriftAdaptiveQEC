# P4 CNN Gap Diagnostic

- reference_mode: `static_linear`
- compare_modes: `window_variance, ekf, cnn_fpga`
- n_windows_per_scenario: `64`

## static_bias_theta

| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_variance | 0.630000 | 0.036493 | 0.026677 | 7.397921 | 0.367523 | 0.020556 | 0.865351 | 0.001369 |
| ekf | 0.609558 | 0.037451 | 0.026253 | 3.077603 | 0.360625 | 0.020518 | 0.862383 | 0.000969 |
| cnn_fpga | 0.470396 | 0.075729 | 0.036755 | 2.618085 | 0.316349 | 0.025933 | 0.815983 | 0.008278 |

| Mode | mean pred mu_q | mean pred mu_p | mu_q sign mismatch | mu_p sign mismatch |
| --- | ---: | ---: | ---: | ---: |
| window_variance | -0.001493 | 0.001677 | 0.484375 | 0.609375 |
| ekf | -0.002451 | 0.001253 | 0.625000 | 0.656250 |
| cnn_fpga | -0.040729 | 0.011755 | 1.000000 | 0.953125 |

## linear_ramp

| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_variance | 0.697298 | 0.015421 | 0.011716 | 9.874839 | 0.415984 | 0.007414 | 0.863411 | 0.001056 |
| ekf | 0.676856 | 0.012117 | 0.009115 | 2.958445 | 0.409766 | 0.007333 | 0.862319 | 0.000679 |
| cnn_fpga | 0.542287 | 0.053240 | 0.019100 | 1.697519 | 0.368656 | 0.012856 | 0.817844 | 0.008091 |

| Mode | mean pred mu_q | mean pred mu_p | mu_q sign mismatch | mu_p sign mismatch |
| --- | ---: | ---: | ---: | ---: |
| window_variance | -0.001904 | -0.001010 | 0.593750 | 0.453125 |
| ekf | -0.002117 | -0.000885 | 0.703125 | 0.406250 |
| cnn_fpga | -0.043240 | 0.009076 | 1.000000 | 0.906250 |

## step_sigma_theta

| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_variance | 0.690000 | 0.009607 | 0.008675 | 6.983122 | 0.410847 | 0.000702 | 0.865893 | 0.000993 |
| ekf | 0.669558 | 0.002918 | 0.002586 | 3.081522 | 0.404350 | 0.000510 | 0.862271 | 0.000721 |
| cnn_fpga | 0.535004 | 0.038327 | 0.010662 | 1.915799 | 0.363094 | 0.005276 | 0.817897 | 0.007461 |

| Mode | mean pred mu_q | mean pred mu_p | mu_q sign mismatch | mu_p sign mismatch |
| --- | ---: | ---: | ---: | ---: |
| window_variance | 0.000521 | 0.000212 | 0.000000 | 0.000000 |
| ekf | 0.000074 | 0.000423 | 0.000000 | 0.000000 |
| cnn_fpga | -0.038327 | 0.009712 | 0.000000 | 0.000000 |

## periodic_drift

| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window_variance | 0.565108 | 0.017556 | 0.015625 | 9.923486 | 0.310231 | 0.009273 | 0.862768 | 0.000948 |
| ekf | 0.544665 | 0.014953 | 0.014770 | 5.593151 | 0.303030 | 0.009362 | 0.862334 | 0.000648 |
| cnn_fpga | 0.399230 | 0.055004 | 0.026728 | 4.036971 | 0.255658 | 0.014948 | 0.813281 | 0.008242 |

| Mode | mean pred mu_q | mean pred mu_p | mu_q sign mismatch | mu_p sign mismatch |
| --- | ---: | ---: | ---: | ---: |
| window_variance | -0.001329 | -0.000932 | 0.515625 | 0.453125 |
| ekf | 0.000120 | -0.000230 | 0.562500 | 0.328125 |
| cnn_fpga | -0.040004 | 0.011728 | 1.000000 | 0.937500 |
