# P3 HIL Mode Benchmark

| Mode | Backend | Status | LER | Overflow | Commits | Slow Viol | Fast Viol | Artifact | Error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| fixed_baseline_mock | mock | ok | 1.198323 | 0.297045 | 1500 | 0.000000 | 0.000017 |  |  |
| float_artifact_mock | mock | ok | 1.140471 | 0.393327 | 1500 | 0.000000 | 0.000017 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |  |
| int8_artifact_mock | mock | ok | 1.071416 | 0.331180 | 1500 | 0.000000 | 0.000017 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260328_010548.tflite |  |
| real_board | board | skipped_unavailable |  |  |  |  |  |  | board_device_missing:/dev/uio0,/dev/uio1 |
