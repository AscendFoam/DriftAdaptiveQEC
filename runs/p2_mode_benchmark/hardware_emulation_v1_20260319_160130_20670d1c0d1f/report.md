# P2 Runtime Mode Benchmark

## Mode Comparison

| Mode | Scenario | LER Mean | LER Std | Commit Mean | Overflow Mean | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fixed_baseline | linear_med | 0.816067 | 0.007063 | 7.00 | 0.254117 | mock |
| fixed_baseline | step_large | 0.937822 | 0.001278 | 7.00 | 0.306061 | mock |
| fixed_baseline | sinusoidal_mid | 1.019033 | 0.004828 | 7.00 | 0.342283 | mock |
| oracle_delayed | linear_med | 0.764239 | 0.007144 | 7.00 | 0.259200 | mock |
| oracle_delayed | step_large | 0.926650 | 0.004696 | 7.00 | 0.311261 | mock |
| oracle_delayed | sinusoidal_mid | 1.020556 | 0.005332 | 7.00 | 0.339767 | mock |
| model_artifact | linear_med | 0.696206 | 0.008786 | 7.00 | 0.257439 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| model_artifact | step_large | 0.731906 | 0.003280 | 7.00 | 0.307728 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| model_artifact | sinusoidal_mid | 0.759289 | 0.005996 | 7.00 | 0.339950 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| int8_artifact | linear_med | 0.701372 | 0.003477 | 7.00 | 0.255939 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz |
| int8_artifact | step_large | 0.732072 | 0.005045 | 7.00 | 0.309439 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz |
| int8_artifact | sinusoidal_mid | 0.761878 | 0.005690 | 7.00 | 0.342589 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz |

## Delta Summary

| Scenario | Mode | dLER vs Fixed | dLER vs Oracle | dLER vs Float | dCommit vs Fixed | dOverflow vs Fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| linear_med | fixed_baseline | 0.000000 | 0.051828 | 0.119861 | 0.000000 | 0.000000 |
| step_large | fixed_baseline | 0.000000 | 0.011172 | 0.205917 | 0.000000 | 0.000000 |
| sinusoidal_mid | fixed_baseline | 0.000000 | -0.001522 | 0.259744 | 0.000000 | 0.000000 |
| linear_med | oracle_delayed | -0.051828 | 0.000000 | 0.068033 | 0.000000 | 0.005083 |
| step_large | oracle_delayed | -0.011172 | 0.000000 | 0.194744 | 0.000000 | 0.005200 |
| sinusoidal_mid | oracle_delayed | 0.001522 | 0.000000 | 0.261267 | 0.000000 | -0.002517 |
| linear_med | model_artifact | -0.119861 | -0.068033 | 0.000000 | 0.000000 | 0.003322 |
| step_large | model_artifact | -0.205917 | -0.194744 | 0.000000 | 0.000000 | 0.001667 |
| sinusoidal_mid | model_artifact | -0.259744 | -0.261267 | 0.000000 | 0.000000 | -0.002333 |
| linear_med | int8_artifact | -0.114694 | -0.062867 | 0.005167 | 0.000000 | 0.001822 |
| step_large | int8_artifact | -0.205750 | -0.194578 | 0.000167 | 0.000000 | 0.003378 |
| sinusoidal_mid | int8_artifact | -0.257156 | -0.258678 | 0.002589 | 0.000000 | 0.000306 |
