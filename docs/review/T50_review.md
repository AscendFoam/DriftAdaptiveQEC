# T50 Review

## Verdict

`PASS`

任务包要求的核心交付都已落地并可被当前工作区复核：

- helper 已实现 canonical 材料链读取、主线 preserved reference 校验、bounded rerun 校验与 pack 汇总，代码边界与 `T50` 目标一致，未扩成 benchmark/HIL/runtime 主线改造：`cnn_fpga/model/build_training_reproducibility_pack.py`
- focused tests 已存在并通过：`tests/test_training_reproducibility_pack.py`
- bounded clean CPU-only rerun 与 eval rerun 的实际产物已存在于 `artifacts/t50_training_repro_pack/`
- 训练复现主报告明确列出 supported / unsupported claims，没有把 bounded rerun 写成 full reproducibility、`.tflite`、真板或 deployment 事实：`docs/training_reproducibility_and_material_regeneration_pack.md`

本次 reviewer 额外做了轻量复核：

1. `py_compile` 通过
2. `python -m unittest tests.test_training_reproducibility_pack` 通过
3. helper 重新执行成功，并重新生成 `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`
4. `git diff --name-only -- runs`
5. `git diff --name-only -- artifacts/models/static_theta_v2 artifacts/reports/static_theta_v2 artifacts/models/runtime_b_residual_v1 artifacts/reports/runtime_b_residual_v1`
6. `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt`

上述边界 diff 检查均为空。

## Blocking issues

- 无。

## Non-blocking issues

- `tests/test_training_reproducibility_pack.py` 目前只对 `p4_multiscenario_recovery_smoke.yaml` 做了 config-drift 负例回归；`experiment_runtime_b_residual.yaml`、`hardware_hil_recovery_smoke.yaml`、`p4_multiscenario_statcalib_extension_lane.yaml` 还没有对应的负例测试。helper 运行时确实会检查这些入口，所以这是覆盖面欠账，不是当前阻断项。
- `cnn_fpga/model/build_training_reproducibility_pack.py` 直接硬编码了 canonical `static_theta_v2` / `runtime_b_residual_v1` 锚点。这种写法对 `T50` 这个 task-scoped consolidator 是合理的，但它不是通用训练账本框架，后续不要把它无审查地复用成 repo-wide infrastructure。

## Missing tests

- 可补一个针对 `experiment_runtime_b_residual.yaml` 的漂移负例，确认其 dataset/model/report 三个 canonical anchor 任一漂移时 helper 会拒绝。
- 可补一个针对 `hardware_hil_recovery_smoke.yaml` 的负例，确认 `slow_loop.mode` 或 latest-float 选择语义漂移时 helper 会拒绝。
- 可补一个针对 `p4_multiscenario_statcalib_extension_lane.yaml` 的负例，确认 `hybrid_residual_b` 的 `model_dir` 或 `artifact_selector` 漂移时 helper 会拒绝。
- 可补一个 rerun 报告边界负例，确认 train/eval report 若落在 `artifacts/t50_training_repro_pack/` 之外、或 backend/device 不是 `numpy/cpu`，helper 会拒绝。

## Suspicious implementation details

- 未发现伪实现、mock、stub 冒充正式结果的问题。真实 rerun 模型与报告文件存在于 `artifacts/t50_training_repro_pack/`，pack JSON 也与这些产物一致。
- 单测中使用了临时 JSON 和占位 `.npz` 文件，但这是针对“路径/边界校验 helper”的正常单元测试手法；它没有替代任务包要求的真实 train rerun / eval rerun。
- canonical 历史 report 中 `training_backend` / `training_device` 为空是历史 artifact 事实，不是本轮实现偷偷省略。worker 在主报告中已明确把这一点写成剩余限制，而不是写成已解决。

## Recommended next action

- 接受 `T50` 在其有界范围内完成，后续把 `artifacts/t50_training_repro_pack/training_reproducibility_pack.json` 与 `docs/training_reproducibility_and_material_regeneration_pack.md` 作为训练材料/复现边界的主引用输入。
- 保持 `R11` 继续 open。`T50` 强化的是 clean CPU-only 训练证据和材料账本，不是 full training reproducibility closure。
- 如果要继续补强，请单开一个小任务补负例覆盖，不要把 `T50` 继续膨胀成通用训练基础设施或新的长实验任务。
