# P3 HIL Mode Benchmark

| Mode | Backend | Status | LER | Overflow | Hist Sat | Corr Sat | Agg Param | Dominant Source | Commits | Slow Viol | Fast Viol | Artifact | Error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| fixed_baseline_mock | mock | ok | 1.199330 | 0.016157 | 0.016157 | 0.000000 | 0.000000 | histogram_input | 1500 | 0.000000 | 0.000017 |  |  |
| float_artifact_mock | mock | ok | 1.123811 | 0.022764 | 0.022764 | 0.000000 | 0.000000 | histogram_input | 1500 | 0.000000 | 0.000017 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |  |
| int8_artifact_mock | mock | ok | 1.124699 | 0.022638 | 0.022638 | 0.000000 | 0.000000 | histogram_input | 1500 | 0.000000 | 0.000017 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260328_012736.tflite |  |
| real_board | board | skipped_unavailable |  |  |  |  |  |  |  |  |  |  | board_device_missing:/dev/uio0,/dev/uio1 |
