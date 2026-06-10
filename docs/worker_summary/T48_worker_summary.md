# T48 Worker Summary

## 改了什么

本轮只改了 `T48` 允许路径：
- 新增 helper：`cnn_fpga/model/build_t48_true_tflite_runtime_gate.py`
- 新增 focused tests：`tests/test_t48_true_tflite_runtime_gate.py`
- 新增 task-scoped config：`cnn_fpga/config/task_tmp/T48_static_theta_tflite_gate.yaml`
- 更新 isolated requirements：`requirements-tflite-win-py311.txt`
- 写入 isolated outputs：`artifacts/t48_true_tflite_runtime_gate/`
- 更新主报告：`docs/t48_true_tflite_runtime_gate.md`
- 更新 review 草案：`docs/review/T48_review.md`
- 更新人类说明：`docs/for_human/T48_explanation.md`
- 更新本文件：`docs/worker_summary/T48_worker_summary.md`

## 如何验证

实际执行并复核了：

1. `python -m py_compile cnn_fpga/model/build_t48_true_tflite_runtime_gate.py`
2. `python -m unittest tests.test_t48_true_tflite_runtime_gate`
3. isolated tf2.21 runtime env probe
   - `artifacts/t48_true_tflite_runtime_gate/runtime_env_probe_tf221.json`
4. isolated tf2.21 preserved true `.tflite` load probe
   - `artifacts/t48_true_tflite_runtime_gate/preserved_tflite_load_probe_tf221.json`
5. float `evaluate_tflite`
   - `eval_tflite_test_20260610_211759.json`
6. float `validate_export --max-samples 128`
   - `validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260610_211815.json`
7. int8 `evaluate_tflite`
   - `eval_tflite_test_20260610_211830.json`
8. int8 `validate_export --max-samples 128`
   - `validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260610_211845.json`
9. helper 真执行
   - `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json`
10. 边界检查
   - `git diff --name-only -- runs`
   - `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
   - `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

## 关键结果

- 最终有效解释器：`D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t48_tf221\Scripts\python.exe`
- Python：`3.11.15`
- runtime 包：`tensorflow==2.21.0`
- `tflite_runtime`：未安装
- `yaml`：未安装，config 读取走仓库 fallback parser

float 结果：
- eval 成功
- `mse = 0.2923593523678865`
- `mae = 0.22008557785115532`
- `r2_mean = 0.9943587472360158`
- `max_abs_diff = 0.11934027632297362`
- `mean_abs_diff = 0.006730998762145316`

int8 结果：
- eval 成功
- `mse = 0.29731706827032195`
- `mae = 0.22137534440302398`
- `r2_mean = 0.9941924482134047`
- `max_abs_diff = 0.2038934768686289`
- `mean_abs_diff = 0.008712156489137833`

最终 gate verdict：
- `GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`

## 剩余风险

1. 本轮成功依赖 isolated `tf2.21` 环境，不代表默认主环境已经恢复 `.tflite` runtime。
2. 本轮只覆盖 preserved `static_theta_v2` `.tflite` runtime truth，不支持扩写为 HIL / board / deployment closure。
3. `tf.lite.Interpreter` 在 `tensorflow 2.21.0` 下会给出 deprecation warning，后续若长期维护这条路径，可能要评估向 LiteRT 迁移。
