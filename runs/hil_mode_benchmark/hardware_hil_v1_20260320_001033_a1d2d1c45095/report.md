# P3 HIL Mode Benchmark

| Mode | Backend | Status | LER | Overflow | Commits | Slow Viol | Fast Viol | Artifact | Error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| fixed_baseline_mock | mock | ok | 0.660778 | 0.300889 | 2 | 0.000000 | 0.000000 |  |  |
| float_artifact_mock | mock | ok | 0.554222 | 0.283556 | 2 | 0.000000 | 0.000000 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |  |
| int8_artifact_mock | mock | ok | 0.540444 | 0.274444 | 2 | 0.000000 | 0.000000 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz |  |
| real_board | board | skipped_unavailable |  |  |  |  |  |  | board_device_missing:/dev/uio0,/dev/uio1 |
