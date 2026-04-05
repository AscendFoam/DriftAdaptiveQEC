# P3 HIL Mode Benchmark

| Mode | Backend | Status | LER | Overflow | Commits | Slow Viol | Fast Viol | Artifact | Error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| fixed_baseline_mock | mock | ok | 0.677778 | 0.295889 | 2 | 0.000000 | 0.000000 |  |  |
| float_artifact_mock | mock | ok | 0.552778 | 0.284222 | 2 | 0.000000 | 0.000000 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz |  |
| int8_artifact_mock | mock | ok | 0.535556 | 0.277889 | 2 | 0.000000 | 0.000000 | /Users/qinchaoyang/Desktop/PC/codes/local/quantum/DriftAdaptiveQEC/artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260320_230659.tflite.json |  |
| real_board | board | skipped_unavailable |  |  |  |  |  |  | board_device_missing:/dev/uio0,/dev/uio1 |
