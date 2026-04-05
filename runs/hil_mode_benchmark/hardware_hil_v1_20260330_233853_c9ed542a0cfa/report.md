# P3 HIL Mode Benchmark

| Mode | Backend | Status | LER | Overflow | Hist Sat | Corr Sat | Agg Param | Dominant Source | Commits | Slow Viol | Fast Viol | Artifact | Error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| fixed_baseline_mock | mock | ok | 1.198418 | 0.297219 | 0.297219 | 0.000000 | 0.000000 | histogram_input | 1500 | 0.000000 | 0.000017 |  |  |
| float_artifact_mock | mock | ok | 1.140519 | 0.394002 | 0.394002 | 0.000000 | 0.000000 | histogram_input | 1500 | 0.000000 | 0.000017 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |  |
| int8_artifact_mock | mock | ok | 1.140973 | 0.393696 | 0.393696 | 0.000000 | 0.000000 | histogram_input | 1500 | 0.000000 | 0.000017 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260328_012736.tflite |  |
| real_board | board | skipped_unavailable |  |  |  |  |  |  |  |  |  |  | board_device_missing:/dev/uio0,/dev/uio1 |
