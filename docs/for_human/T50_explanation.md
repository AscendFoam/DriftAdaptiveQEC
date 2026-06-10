# T50 说明与 Review 解释

## 1. 通俗解释

`T50` 不是去证明“训练链已经完整复现”。

它做的是一件更务实的事：

1. 确认仓库里两条关键历史训练材料链还在
2. 确认当前 `P3/P4/HIL` 主线入口还在引用这些 preserved historical models
3. 在干净的 CPU-only 环境里，真实再跑一次有界的 train + eval
4. 把“历史材料”和“本轮新 rerun 证据”统一收成一个可审计的 pack

所以，`T50` 的价值不是把完成度说大，而是把“现在到底已经证到哪一步、还没证到哪一步”讲清楚。

## 2. 实现细解

### 2.1 任务目标

结合 `docs/04_task_board.md`、`docs/07_handoff.md` 和 `docs/08_risks_and_open_questions.md` 的上下文，`T50` 的位置很明确：

- 当前仓库仍处于 `Phase 2: Controlled Development`
- `T31 / T39 / T40` 已经把训练链推进到 clean CPU-only lane 可 dry-run、可 bootstrap、可做一次真实训练 smoke
- 但 `R11` 仍然 open，因为“完整训练可复现”“跨机器/跨系统可移植”“`.tflite`/真板正确性”都还没有被证明

因此 `T50` 的真正目标不是继续扩大实验面，而是补一个主线可以引用的、代码驱动的训练材料/复现证据包，把下列三层关系收清楚：

1. canonical 历史训练材料是否还完整存在
2. 当前主线配置是否还依赖这些 canonical 材料
3. clean CPU-only lane 现在是否能再次真实跑出一组隔离的 train/eval 证据

### 2.2 代码与配置改动

这次实现的核心是三个部分。

第一部分是 helper：

- `cnn_fpga/model/build_training_reproducibility_pack.py`

它做了四类工作：

1. 读取 canonical `static_theta_v2` 材料链
   - dataset manifest
   - float model
   - train report
   - int8 / export / eval / `.tflite` 派生物存在性摘要
2. 读取 canonical `runtime_b_residual_v1` 材料链
3. 校验四个主线入口的 preserved model references
   - `experiment_runtime_b_residual.yaml`
   - `hardware_hil_recovery_smoke.yaml`
   - `p4_multiscenario_recovery_smoke.yaml`
   - `p4_multiscenario_statcalib_extension_lane.yaml`
4. 读取本轮 bounded rerun 的 train/eval report，并生成统一 JSON pack

这个 helper 的设计重点不是“通用化”，而是“task-scoped 且边界清楚”。它直接锚定 canonical `static_theta_v2` 和 `runtime_b_residual_v1`，正好符合 `T50` 的任务包要求。

第二部分是 focused tests：

- `tests/test_training_reproducibility_pack.py`

当前测试覆盖了：

1. 正常情况下 pack 可以生成
2. canonical `static_theta_v2` float model 缺失时 helper 会拒绝
3. `p4_multiscenario_recovery_smoke.yaml` 漂移到缺失路径时 helper 会拒绝

第三部分是 bounded rerun 配置：

- `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`

它基于 `experiment_static_theta_v2.yaml`，只覆盖任务包允许覆盖的几项：

- 输出目录改到 `artifacts/t50_training_repro_pack/`
- `max_train_samples = 2048`
- `max_val_samples = 512`
- `epochs = 5`
- `patience = 3`

这意味着本轮 rerun 的目标不是追平 canonical 历史质量，而是拿到一组真实、隔离、可复核的 clean CPU-only 训练与评估证据。

### 2.3 任务流程

从执行顺序看，这个任务大致是这样完成的：

1. 先补 helper 和 focused tests
2. 用 clean CPU-only 解释器执行 bounded train rerun
3. 对新模型再执行一次真实的 test-split eval rerun
4. 用 helper 读取 canonical chain、主线 config、rerun report，生成统一 pack
5. 把 supported claims / unsupported claims 写进主报告

最终关键输出有四类：

1. 代码与测试
   - `cnn_fpga/model/build_training_reproducibility_pack.py`
   - `tests/test_training_reproducibility_pack.py`
2. 派生配置
   - `cnn_fpga/config/task_tmp/T50_static_theta_repro_pack.yaml`
3. 新证据产物
   - `artifacts/t50_training_repro_pack/models/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c.npz`
   - `artifacts/t50_training_repro_pack/reports/static_theta_v2/tiny_cnn_20260610_195014_7126933acb7c_train_report.json`
   - `artifacts/t50_training_repro_pack/reports/static_theta_v2/eval_test_20260610_195030.json`
   - `artifacts/t50_training_repro_pack/training_reproducibility_pack.json`
4. 主文档
   - `docs/training_reproducibility_and_material_regeneration_pack.md`

### 2.4 对后续开发的意义

`T50` 的意义，不是得到了“更强训练结论”，而是得到了“更可靠的证据边界”。

对后续开发，它至少有四个实际作用：

1. 给主线提供一份统一的训练材料台账，不用每次人工回忆 `static_theta_v2` / `runtime_b_residual_v1` 是否还在
2. 把 `P3/P4/HIL` 入口对 preserved models 的依赖关系落成代码可检查的事实
3. 把 clean CPU-only lane 从 `T40` 的 one-train smoke，推进到 one-train + one-eval bounded rerun
4. 明确告诉后续文档或论文写作：哪些可以说，哪些不能说，避免把 bounded rerun 写成 full reproducibility

这和 `docs/02_experiment_plan.md` 中对 canonical `static_theta_v2` 的历史定位并不冲突。`T50` 没有重写 canonical 历史结论，只是把这些历史材料与当前可复核的 clean-lane 证据接上了。

## 3. 为什么 review 结果是 PASS

我给 `PASS`，理由很直接：

1. 任务是否真的完成了
   - 完成了。helper、tests、派生 config、真实 train rerun、真实 eval rerun、pack JSON、主报告都存在。
2. 是否有伪实现、mock、stub、hardcode
   - 没发现把 mock/stub 冒充真实结果的问题。
   - 单测里的临时 JSON 与占位 `.npz` 只是单元测试手法，不替代真实 rerun。
   - helper 的 hardcode 主要是 canonical 锚点，这对 task-scoped `T50` 是合理设计，不是偷懒伪实现。
3. 是否缺测试或验证
   - 基础验证是够的：`py_compile`、`unittest`、helper 实跑、边界 diff 检查都能复核。
   - 真实 train/eval rerun 也已有落盘产物。
   - 但仍有非阻断测试欠账：还没分别给另外三个主线 config 补漂移负例。
4. 是否过度工程
   - 没有。实现保持在 `T50` 所需的 task-scoped consolidator，没有扩成新的训练基础设施。
5. 是否破坏已有功能
   - 没看到越界修改。`runs/`、canonical historical artifact 目录、`requirements`、`train.py`、`evaluate.py`、`export.py` 都没有被改写。
6. 文档是否把计划写成事实
   - 没有。主报告反复强调这不是 full reproducibility，也没有把 `.tflite`、真板、benchmark/HIL promotion 写成已完成事实。

所以这不是“完美无缺”的意思，而是“在任务包要求的有界范围内，已经达到了可接受完成态”。剩下的不足主要是覆盖面增强问题，不是当前 `BLOCK` 或 `PASS_WITH_WARNINGS` 级别的问题。

## 4. 对 worker 已写文档的检查与补充

worker 已经写了：

- `docs/training_reproducibility_and_material_regeneration_pack.md`
- `docs/review/T50_review.md`
- `docs/for_human/T50_explanation.md`

我的结论是：

1. `docs/training_reproducibility_and_material_regeneration_pack.md`
   - 基本准确。
   - 它把 supported / unsupported claims 区分得比较清楚，边界没有漂。
   - 我没有发现把计划写成事实的明显错误。
2. worker 原先写的 `docs/for_human/T50_explanation.md`
   - 内容方向基本对。
   - 但它更像“任务讲解”，还缺少“为什么 reviewer 给出这个 verdict”以及“对 worker 自检文档本身的补充说明”。
3. worker 原先写的 `docs/review/T50_review.md`
   - 那是一份 worker 自检草稿，不是正式外部 reviewer verdict。
   - 它本身没有明显事实错误，但角色定位不对，所以我把它改写成正式 review 结果。

这次补充的核心有两点：

1. 明确把 worker 自检稿和 reviewer 正式 verdict 区分开
2. 把当前仍然存在但不构成阻断的测试欠账写明，避免后续误以为 `T50` 已经把所有 config-drift 回归都补齐
