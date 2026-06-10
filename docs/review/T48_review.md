# T48 Review

## Verdict

`PASS`

本次 review 只审 `T48` 允许路径及其产物。当前工作区里还存在一批治理文档改动和 `T50` 相关文件，但它们不属于本次 `T48` worker 交付范围，我没有把那些改动计入 `T48` verdict。

`T48` 的核心任务目标已经完成，并且我做了不重跑长实验的轻量复核：

1. 现有报告与 gate JSON 自洽，最终 gate verdict 为 `GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`
2. `PYTHONDONTWRITEBYTECODE=1` 下运行 `python -m unittest tests.test_t48_true_tflite_runtime_gate`，结果 `Ran 5 tests`, `OK`
3. 用 `.venvs/t48_tf221` 解释器对 helper 做临时输出复核，成功生成与现有结论一致的 gate JSON
4. 用 `.venvs/t48_tf221` 解释器做只读 load 事实检查，选中的 float / int8 preserved `.tflite` 都能 `allocate_tensors()`
5. 三条边界 diff 检查均为空：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
   - `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

主报告也保持了边界诚实，没有把 isolated `.tflite` runtime 成功扩写成默认环境恢复、HIL closure、real-board 验证或 deployment closure：[docs/t48_true_tflite_runtime_gate.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/t48_true_tflite_runtime_gate.md#L13)

## Blocking issues

- 无。

## Non-blocking issues

- `cnn_fpga/model/build_t48_true_tflite_runtime_gate.py:14-16` 与 `:35-45` 的 helper 默认参数并不直接对应最终成功的 `*_tf221.json` probe 组合，最终 gate 复现依赖显式 CLI 参数而不是无参默认值。这不影响本次任务完成，但后续如果把它当“直接复跑工具”使用，必须保留准确调用方式。
- `tests/test_t48_true_tflite_runtime_gate.py:153-276` 已覆盖 stub 拒绝、runtime unavailable、float-only、兼容性失败等分支，但没有一个专门的“float + int8 全成功 -> GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8”单元测试。真实产物已经补足了这条证据，因此不是阻断项。

## Missing tests

- 增加一个显式的全成功单测：float / int8 报告都存在时，helper 应返回 `GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`。
- 增加一个 helper 默认参数回归测试，确认未来若调整 probe 文件命名或默认值，不会把最终成功结论重新退化成“只能靠手工传参复现”。

## Suspicious implementation details

- 未发现把 `.tflite.json` stub 冒充成真实 runtime 成功的情况。helper 明确拒绝 stub，主报告也明确区分 true `.tflite` 与 stub：[cnn_fpga/model/build_t48_true_tflite_runtime_gate.py](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/model/build_t48_true_tflite_runtime_gate.py:98)；[docs/t48_true_tflite_runtime_gate.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/t48_true_tflite_runtime_gate.md:42)
- `artifacts/t48_true_tflite_runtime_gate/` 下同时保留了旧的失败探针（`runtime_env_probe.json` / `preserved_tflite_load_probe.json`）和最终成功探针（`*_tf221.json`）。这不是伪实现，反而把“默认 `LPNEnv + tf2.13` 失败、isolated `tf2.21` 成功”的版本错配诊断保留下来了；只是后续转述时必须把两组 probe 区分清楚。
- `requirements-tflite-win-py311.txt` 只有 `tensorflow==2.21.0`，其余依赖依靠 transitive install 或仓库 fallback parser。对“最小 runtime gate 环境清单”这个任务目标来说这是可接受的，没有演变成过度工程。

## Recommended next action

- 接受 `T48` 在其有界范围内完成，并把 `docs/t48_true_tflite_runtime_gate.md` 与 `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json` 作为当前机器 `.tflite` runtime truth 的主引用。
- 保持边界表述：当前结论仅支持“isolated `tf2.21` 环境下，preserved `static_theta_v2` float/int8 `.tflite` 可真实执行并可做有界一致性校验”，不支持默认环境恢复、HIL closure、real-board 或 deployment closure。
- 如果后续还要长期复用这条 lane，建议单开一个极小后续任务，把 helper 默认输入对齐到最终成功 probe，并补上全成功单测；不要在 `T48` 上继续扩成 benchmark 或部署任务。
