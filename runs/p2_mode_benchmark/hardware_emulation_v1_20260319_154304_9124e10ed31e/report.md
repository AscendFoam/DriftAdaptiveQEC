# P2 Runtime Mode Benchmark

## Mode Comparison

| Mode | Scenario | LER Mean | LER Std | Commit Mean | Overflow Mean | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| mock | linear_med | 0.753950 | 0.005095 | 7.00 | 0.256167 | mock |
| mock | step_large | 0.768878 | 0.004248 | 7.00 | 0.314250 | mock |
| mock | sinusoidal_mid | 0.781933 | 0.006301 | 7.00 | 0.349761 | mock |
| model_artifact | linear_med | 0.771939 | 0.003687 | 7.00 | 0.255939 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| model_artifact | step_large | 0.881128 | 0.003424 | 7.00 | 0.315683 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| model_artifact | sinusoidal_mid | 0.952172 | 0.004388 | 7.00 | 0.350944 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |
| int8_artifact | linear_med | 0.774594 | 0.004891 | 7.00 | 0.255283 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz |
| int8_artifact | step_large | 0.880156 | 0.005790 | 7.00 | 0.317128 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz |
| int8_artifact | sinusoidal_mid | 0.951661 | 0.006878 | 7.00 | 0.348683 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz |

## Delta Summary

| Scenario | Mode | dLER vs Mock | dLER vs Float | dCommit vs Mock | dOverflow vs Mock |
| --- | --- | ---: | ---: | ---: | ---: |
| linear_med | mock | 0.000000 | -0.017989 | 0.000000 | 0.000000 |
| step_large | mock | 0.000000 | -0.112250 | 0.000000 | 0.000000 |
| sinusoidal_mid | mock | 0.000000 | -0.170239 | 0.000000 | 0.000000 |
| linear_med | model_artifact | 0.017989 | 0.000000 | 0.000000 | -0.000228 |
| step_large | model_artifact | 0.112250 | 0.000000 | 0.000000 | 0.001433 |
| sinusoidal_mid | model_artifact | 0.170239 | 0.000000 | 0.000000 | 0.001183 |
| linear_med | int8_artifact | 0.020644 | 0.002656 | 0.000000 | -0.000883 |
| step_large | int8_artifact | 0.111278 | -0.000972 | 0.000000 | 0.002878 |
| sinusoidal_mid | int8_artifact | 0.169728 | -0.000511 | 0.000000 | -0.001078 |
