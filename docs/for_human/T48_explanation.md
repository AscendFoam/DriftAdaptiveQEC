# T48 说明与 Review 解释

## 1. 先用人话解释这个任务

`T48` 想回答的问题其实很简单：

仓库里早就有历史上导出的 `.tflite` 模型文件，但“文件存在”不等于“当前这台机器真的能跑它”。

所以 `T48` 做的不是新训练、也不是新 benchmark，而是一次很收敛的真值核验：

1. 当前机器上到底有没有真实可用的 `.tflite` 解释器环境
2. 历史保存下来的 float / int8 `.tflite` 文件到底能不能真加载、真推理
3. 它们和源 `.npz` 模型的输出差异大概在什么量级

一句话概括：

`T48` 不是在证明“部署链全通了”，而是在证明“这台机器上，历史 `.tflite` 文件到底能不能真跑”。

## 2. 这个任务具体做了什么

### 2.1 任务目标放在整个项目里是什么意思

从治理文档看，`T48` 是在 `T50` 之后接上的：

- `T50` 已经把 canonical 训练材料、主线 preserved model references、以及训练复现边界收清楚了
- 但那还没有回答 `.tflite` runtime 到底是不是真的可用
- `T49` 真板执行又还缺宿主机/bitstream/DMA 前提，所以不能直接跳过去

因此 `T48` 是一个很合理的“下一步”：

- 不碰 benchmark
- 不碰 HIL 主线
- 不碰真板
- 只把 `.tflite` runtime 这条边界查清楚

这也是为什么 `docs/04_task_board.md` 和 `docs/07_handoff.md` 都把它定义成一个 bounded gate，而不是新的性能任务或部署任务。

### 2.2 代码和配置改动

这次新增的核心实现有四块。

第一块是 helper：

- `cnn_fpga/model/build_t48_true_tflite_runtime_gate.py`

它负责：

1. 读取 `T50` 产出的训练材料 pack
2. 读取本轮 runtime probe / preserved `.tflite` load probe
3. 从 `static_theta_v2` 的 preserved materials 中挑选 float / int8 对应的 true `.tflite`
4. 明确拒绝 `.tflite.json` stub
5. 读取本轮 `evaluate_tflite` 和 `validate_export` 报告
6. 汇总成一个统一的 gate JSON，并给出最终 verdict

第二块是测试：

- `tests/test_t48_true_tflite_runtime_gate.py`

当前测试覆盖了：

1. stub 候选会被拒绝
2. runtime unavailable 时 helper 会给出 `NO_GO_TRUE_TFLITE_RUNTIME_UNAVAILABLE`
3. float 成功而 int8 不具备条件时，会给出 `GO_TRUE_TFLITE_RUNTIME_FLOAT_ONLY`
4. preserved `.tflite` 与可用 runtime 不兼容时，会给出 `NO_GO_PRESERVED_TFLITE_ARTIFACT_INVALID_OR_STUB_ONLY`
5. 缺关键报告文件时 helper 会拒绝

第三块是 task-scoped config：

- `cnn_fpga/config/task_tmp/T48_static_theta_tflite_gate.yaml`

这个配置没有改训练逻辑，只是把本轮 report 输出隔离到：

- `artifacts/t48_true_tflite_runtime_gate/reports/static_theta_v2`

并把 `evaluation.target_split` 固定在 `test`。

第四块是最小环境清单：

- `requirements-tflite-win-py311.txt`

这个文件没有试图做一个“大而全”的 runtime 环境，而是记录这次真正跑通所需的最小核心包：`tensorflow==2.21.0`。

### 2.3 任务流程

这次真正发生的流程可以理解成两段。

第一段是先查失败原因：

1. 先看默认可见环境 `LPNEnv + tensorflow==2.13.0`
2. 发现虽然能 import TensorFlow，但加载 preserved `.tflite` 会报 `FULLY_CONNECTED` builtin op version `12` 不支持
3. 这说明问题不在“历史 `.tflite` 文件坏了”，而在“当前 runtime 太老”

第二段是建立真实可用环境并做真实执行：

1. 新建 isolated 环境 `.venvs/t48_tf221`
2. 在其中用 `tensorflow==2.21.0` 重新做 runtime probe
3. 对 preserved true `.tflite` 做 load probe，确认能 `allocate_tensors()`
4. 对 float `.tflite` 真实执行 `evaluate_tflite`
5. 对 float `.npz` vs `.tflite` 真实执行 `validate_export --max-samples 128`
6. 对 int8 `.tflite` 和 int8 `.npz` 重复同样动作
7. 最后把这些结果收成 `t48_true_tflite_runtime_gate.json`

因此，这次不是靠“推测”说通了，而是靠真实执行证据说通了。

### 2.4 产出的关键证据是什么

这次最关键的证据有五类：

1. 失败环境探针
   - `artifacts/t48_true_tflite_runtime_gate/runtime_env_probe.json`
   - `artifacts/t48_true_tflite_runtime_gate/preserved_tflite_load_probe.json`
2. 成功环境探针
   - `artifacts/t48_true_tflite_runtime_gate/runtime_env_probe_tf221.json`
   - `artifacts/t48_true_tflite_runtime_gate/preserved_tflite_load_probe_tf221.json`
3. float 真执行报告
   - `eval_tflite_test_20260610_211759.json`
   - `validate_export_tiny_cnn_20260319_151717_b87c6c227b57_20260610_211815.json`
4. int8 真执行报告
   - `eval_tflite_test_20260610_211830.json`
   - `validate_export_tiny_cnn_20260319_151717_b87c6c227b57_int8_20260319_151756_20260610_211845.json`
5. 最终 gate 汇总
   - `artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json`

这些文件组合起来，足以支持一个有边界的结论：

- 当前机器上确实存在一个真实可用的 isolated `.tflite` runtime 环境
- 选中的 preserved float / int8 `.tflite` 都在这个环境里真实跑通了

### 2.5 对后续开发的意义

`T48` 的意义不是“性能更强了”，而是“部署边界更清楚了”。

它至少带来三件事：

1. 把 `.tflite` 路径从“文档里说可能能跑”推进到“当前机器上已有真实执行证据”
2. 把失败根因缩小为 runtime 版本错配，而不是 artifact 本体损坏
3. 给后续 `T49` 真板任务、以及任何部署相关表述，增加了一块真实但仍然有界的前置证据

但它的意义也必须被限制住：

- 这不是默认主环境恢复
- 这不是 HIL closure
- 这不是 real-board closure
- 这也不是跨机器可移植性证明

## 3. 为什么我的 review 结果是 PASS

我给 `PASS`，主要理由是这次任务的核心目标已经达成，而且没有越界或伪实现。

### 3.1 为什么说“真的完成了”

因为任务包要求的关键交付都存在，而且互相对得上：

1. helper 存在
2. focused tests 存在
3. runtime probe 和 load probe 存在
4. float / int8 的真实 eval 报告存在
5. float / int8 的真实 validate 报告存在
6. 最终 gate JSON 存在
7. 主报告明确给出最终 gate verdict：`GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`

我还做了轻量复核：

1. `PYTHONDONTWRITEBYTECODE=1` 下的 unit test 通过
2. helper 用现有 probe 和报告重新生成临时 gate JSON，结论一致
3. 在 `.venvs/t48_tf221` 里做了只读的 `.tflite` load 事实检查，float / int8 都能 `allocate_tensors()`

所以这不是“只写了文档没做事实执行”。

### 3.2 为什么不算伪实现

这次最关键的一点，是 worker 没有把 stub 当成功。

任务包最怕的就是这类伪完成：

- `.tflite.json` stub 还在，但文档写成“真实 `.tflite` runtime 已恢复”

而这次实现恰恰把 stub 拒绝做得很明确：

1. helper 代码里显式拒绝 `.json` / `.tflite.json`
2. 主报告也把 stub 跟 true `.tflite` 明确区分
3. 最终成功依据来自真实 `.tflite` 文件和真实执行报告，而不是 sidecar/stub

所以从 reviewer 角度，这一条是通过的。

### 3.3 为什么不是 PASS_WITH_WARNINGS

我确实发现了两个小问题，但我认为它们还不到 `PASS_WITH_WARNINGS` 的级别：

1. helper 默认参数没有直接指向最终成功的 `*_tf221.json` probe 组合
2. 单测缺少一个显式的“float + int8 全成功”路径测试

这两个问题都是真问题，但影响的是“后续复跑方便程度”和“测试覆盖完整度”，不是任务事实本身。

因为：

- 真实成功报告已经存在
- helper 也已经用现有报告生成了最终 gate JSON
- 我自己复核时也能用显式参数复现一致结论

所以更准确的判断是：完成态成立，但还有一点小的可维护性欠账。

## 4. 对 worker 已有 review / explanation 文档的补充

worker 已经写了：

- `docs/review/T48_review.md`
- `docs/for_human/T48_explanation.md`

它们的方向基本是对的，但我做了两类补充。

第一类补充是角色边界：

- worker 原来的 `docs/review/T48_review.md` 本质上是自检草稿
- 我把它改成了正式 reviewer verdict

第二类补充是 reviewer 视角下的欠账说明：

- 我补写了“helper 默认输入并不直接对齐最终成功 probe”这件事
- 我补写了“缺少 full-success 单测”这件事

这些都不是为了否定 worker 的完成度，而是为了让后续接手的人知道：

1. 这次成功是真的
2. 但如果后续想把它当长期复跑工具，还需要一个很小的整理任务

## 5. 一个容易被误解但必须记住的点

`T48` 成功，不等于“`.tflite` 路径彻底闭环”。

它只等于：

- 在当前这台机器上
- 用 isolated `tf2.21` 环境
- 对 `static_theta_v2` preserved float / int8 `.tflite`
- 已经有了真实 runtime 执行证据

这条边界一旦说清楚，`T48` 就是一个很扎实的 `PASS`。
