# T48 真 `.tflite` runtime gate

## 任务边界

`T48` 只验证一件事：当前这台 Windows 机器上，`static_theta_v2` 的 preserved true `.tflite` artifact 能否在真实 interpreter 环境里被真实加载、真实推理，并与 source artifact 做有界一致性校验。

本轮不做：
- 重新训练
- 重新导出 canonical `.tflite`
- benchmark / HIL / real-board / sidecar
- 改写 canonical historical artifact 目录

## 结论先行

最终 gate verdict：

`GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`

更准确地说：
- 默认可见的 `LPNEnv + tensorflow==2.13.0` 不能加载 preserved `.tflite`
- 但在本轮新建的隔离环境 `.venvs/t48_tf221` 中，`tensorflow==2.21.0` 可以成功加载并真实执行 float / int8 preserved `.tflite`
- 因此当前机器上已经存在一个真实可用的 `.tflite` 解释器环境，只是它不是默认环境，而是本轮建立的 task-scoped isolated environment

## 执行环境真值

- 最终验收解释器：`D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t48_tf221\Scripts\python.exe`
- Python 版本：`3.11.15`
- 真实可导入 runtime 包：`tensorflow==2.21.0`
- `tflite_runtime`：未安装
- `numpy==2.4.6`
- `yaml` / `PyYAML`：未安装；本轮 config 读取走仓库自带 fallback YAML parser

本轮环境真值记录：
- [runtime_env_probe_tf221.json](/D:/Codes/Quantum/DriftAdaptiveQEC/artifacts/t48_true_tflite_runtime_gate/runtime_env_probe_tf221.json)

补充调查结论：
- 中途排查还确认了 `LPNEnv + tensorflow==2.13.0` 会在 interpreter load 阶段报 `FULLY_CONNECTED` builtin op version `12` 不支持
- 这解释了为何历史 preserved `.tflite` 在默认环境里失败，而在 `tensorflow==2.21.0` 下恢复可用

## preserved artifact 选择

选择规则：
- source float / int8 artifact 只来自 `T50` pack 记录的 canonical `static_theta_v2`
- true `.tflite` 只从 `T50` pack 枚举的 preserved true `.tflite` 集合中选
- 默认选各自类别按文件名字典序最新的 preserved true `.tflite`
- 所有 `.tflite.json` stub manifest 明确拒绝，不计入“真实 runtime 成功”

本轮选中的 preserved pair：
- float source artifact：`artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- float true `.tflite`：`artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_tflite_20260328_012736.tflite`
- int8 source artifact：`artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756.npz`
- int8 true `.tflite`：`artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260328_012736.tflite`

明确拒绝的 stub：
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_tflite_20260320_230659.tflite.json`
- `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_tflite_20260320_230659.tflite.json`

## load probe 与真实执行

本轮做了两层证据：

1. isolated `tf2.21` load probe
   - [preserved_tflite_load_probe_tf221.json](/D:/Codes/Quantum/DriftAdaptiveQEC/artifacts/t48_true_tflite_runtime_gate/preserved_tflite_load_probe_tf221.json)
   - 结果：`6 / 6` preserved true `.tflite` 全部 `allocate_tensors()` 成功
2. 任务包要求的 4 条真实命令
   - float `evaluate_tflite`
   - float `validate_export --max-samples 128`
   - int8 `evaluate_tflite`
   - int8 `validate_export --max-samples 128`

四条真实命令全部成功，且都写到了本轮 isolated report 目录：
- `artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2/`

## 八个直接答案

1. 当前机器上是否真的有可用 `.tflite` 解释器环境？
   - 有。最终可用环境是 `.venvs/t48_tf221`。
2. 实际使用了哪个解释器和包？
   - `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t48_tf221\Scripts\python.exe`
   - `tensorflow==2.21.0`
3. 选中了哪些 preserved float / int8 artifact 对？
   - 见上文“preserved artifact 选择”。
4. float `.tflite` 是否真实执行成功？
   - 是。
5. int8 `.tflite` 是否真实执行成功？
   - 是。
6. artifact-vs-`.tflite` 的最大绝对误差 / 平均绝对误差是多少？
   - float：`max_abs_diff = 0.11934027632297362`，`mean_abs_diff = 0.006730998762145316`
   - int8：`max_abs_diff = 0.2038934768686289`，`mean_abs_diff = 0.008712156489137833`
7. 最终 gate verdict 是什么？
   - `GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`
8. 当前仍然不能支持哪些 `.tflite` / deployment claims？
   - 不能把本轮结论扩写成 real-board / HIL closure
   - 不能把本轮结论扩写成 cross-host / cross-OS / cross-environment portability
   - 不能把 isolated `tf2.21` 成功误写成“默认主环境已恢复”

## 关键结果

### float `.tflite`

- eval report：
  - [eval_tflite_test_20260610_211759.json](/D:/Codes/Quantum/DriftAdaptiveQEC/artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2/eval_tflite_test_20260610_211759.json)
- validate report：
  - [validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260610_211815.json](/D:/Codes/Quantum/DriftAdaptiveQEC/artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2/validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260610_211815.json)
- eval metrics：
  - `mse = 0.2923593523678865`
  - `mae = 0.22008557785115532`
  - `r2_mean = 0.9943587472360158`
- drift summary：
  - `max_abs_diff = 0.11934027632297362`
  - `mean_abs_diff = 0.006730998762145316`
  - `status = ok`

### int8 `.tflite`

- eval report：
  - [eval_tflite_test_20260610_211830.json](/D:/Codes/Quantum/DriftAdaptiveQEC/artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2/eval_tflite_test_20260610_211830.json)
- validate report：
  - [validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260610_211845.json](/D:/Codes/Quantum/DriftAdaptiveQEC/artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2/validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260610_211845.json)
- eval metrics：
  - `mse = 0.29731706827032195`
  - `mae = 0.22137534440302398`
  - `r2_mean = 0.9941924482134047`
- drift summary：
  - `max_abs_diff = 0.2038934768686289`
  - `mean_abs_diff = 0.008712156489137833`
  - `status = ok`

## 紧凑表格

| category | subject | result |
| --- | --- | --- |
| environment_truth | runtime_env | `.venvs/t48_tf221` / Python `3.11.15` / `tensorflow==2.21.0` 可导入；`tflite_runtime` 不存在 |
| artifact_selection | preserved pair | 选中 `20260328_012736` float / int8 true `.tflite`；2 个 `.tflite.json` stub 明确拒绝 |
| float_runtime_result | float preserved `.tflite` | 真实执行成功；`mse=0.2923593523678865`，`max_abs_diff=0.11934027632297362`，`status=ok` |
| int8_runtime_result | int8 preserved `.tflite` | 真实执行成功；`mse=0.29731706827032195`，`max_abs_diff=0.2038934768686289`，`status=ok` |
| supported_claims | T48 bounded support | 当前机器存在一个真实可用的 isolated `.tflite` interpreter 环境，并已完成 float / int8 preserved `.tflite` 的真实执行与 source-vs-`.tflite` 一致性校验 |
| unsupported_claims | T48 bounded exclusions | 不支持 real-board / HIL closure，不支持默认环境已恢复，不支持跨环境可移植性结论 |

## supported claims

当前可支持：
- 当前机器存在一个真实可用的 isolated `.tflite` interpreter 环境：`.venvs/t48_tf221 + tensorflow==2.21.0`
- selected preserved float `.tflite` 已在本机真实执行，并已与 source float artifact 做一致性校验
- selected preserved int8 `.tflite` 已在本机真实执行，并已与 source int8 artifact 做一致性校验
- `.tflite.json` stub 与 true `.tflite` 已被显式区分，stub 没有被冒充成真实 runtime 成功

## unsupported claims

当前仍不支持：
- real-board validation
- HIL closure
- deployment closure
- cross-host / cross-OS / cross-environment portability
- “默认主环境已经恢复 `.tflite` runtime” 这一表述

## 最终 gate JSON

- [t48_true_tflite_runtime_gate.json](/D:/Codes/Quantum/DriftAdaptiveQEC/artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json)

这个 JSON 已统一记录：
- 环境真值
- preserved artifact 选择
- float gate 结果
- int8 gate 结果
- drift summary
- 最终 gate verdict
