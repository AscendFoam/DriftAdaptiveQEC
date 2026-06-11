# T30 人话版说明（含 Reviewer 补充）

## 1. 这个 Task 在做什么（通俗版）

T26 说的是"statcalib 可以做，但只能单独做"。T30 做的是把这句话**落成一个真正能执行的接口合同**。

打个比方：T26 像是写了一份"可以盖新楼"的批文，但没画建筑图纸。T30 就是画出那张图纸——定义清楚这栋楼的入口在哪里、出口在哪里、每一层楼干什么用、什么情况下算"楼盖好了"、什么情况下算"还没盖"。

具体来说，这次加了：
- **输入合同**（`StatCalibInput`）：statcalib 模块需要接收什么数据——窗口编号、当前解码器参数、直方图统计摘要、校准特征等。
- **输出合同**（`StatCalibOutput`）：statcalib 模块吐出什么——更新后的解码器参数（K, b），以及一个明确的状态标签（`generated` / `not_generated` / `not_applicable` / `diagnostic_error`）。
- **转换规则**：只有状态为 `generated` 的输出才能转成现有快回路使用的 `DecoderRuntimeParams`。其他状态的输出会直接报错，防止下游把"没生成"的空值当成有效参数用。

这次**不是**加了新的 baseline 成绩，也不是跑了 benchmark。代码上只新增了一个独立的接口定义模块和一个测试文件，完全不碰现有的 `ParamMapper` 或 P4 冻结基准。

## 2. 任务实现详解

### 2.1 任务目标

T30 的目标是将 T26 gate 中的概念性 `StatCalibInput` / `StatCalibOutput` 收紧为**精确的、有类型的、带验证的接口合同**，并在接口层面完成最小验证。

任务包特别要求：
- 不修改现有 `ParamMapper.map_prediction()` 的语义
- 不运行 benchmark、不新增 run directory
- 不把 statcalib 加入冻结的 P4 排名表
- 验证只需做到接口级别

### 2.2 任务流程

Worker 执行了以下步骤：

1. **阅读前置文档**：读了 T26 gate、T26 review、ParamMapper 代码、SlowLoopRuntime 代码、P4 formal protocol 和风险清单。
2. **设计接口合同**：定义了 `StatCalibInput`（10 个字段）和 `StatCalibOutput`（8 个字段），包括状态值（4 种）、reason 值（5 种）和转换边界。
3. **实现独立模块**：新建 `cnn_fpga/decoder/statcalib.py`，包含 frozen dataclass、验证逻辑、工厂方法和序列化。
4. **写 focused test**：新建 `tests/test_statcalib_interface.py`，6 个测试用例覆盖关键路径。
5. **更新 gate 文档**：把 T26 gate 中的"Minimal Comparator Interface"从概念描述收紧为 exact field-level contract。
6. **补文档**：写了 review 文档、人话说明，在任务包中回填了 Worker Output 和 Verification Record。

### 2.3 文件变化

| 文件 | 变化类型 | 说明 |
|------|----------|------|
| `cnn_fpga/decoder/statcalib.py` | **新建** | 接口合同模块：`StatCalibInput`、`StatCalibOutput`、状态/原因常量、验证函数、工厂方法、转换方法 |
| `tests/test_statcalib_interface.py` | **新建** | 6 个 focused interface test |
| `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md` | **修改** | "Minimal Comparator Interface" 节从概念收紧为 exact field-level contract |
| `docs/review/T30_review.md` | **新建** | Worker 自审 + Reviewer 独立审查 |
| `docs/for_human/T30_explanation.md` | **新建** | 本文件 |
| `docs/tasks/Phase2/T30_statcalib_interface_contract.md` | **修改** | 追加 Worker Output 和 Verification Record |

**没有修改的文件**（关键边界确认）：
- `cnn_fpga/decoder/param_mapper.py` — 无改动
- `cnn_fpga/runtime/slow_loop_runtime.py` — 无改动
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` — 无改动
- 没有新 run directory

### 2.4 核心代码逻辑

**`StatCalibInput`** 是一个 frozen dataclass，包含：
- 窗口索引（`window_id`、`slow_update_index`）
- 前置解码器参数（`prior_decoder_params: DecoderRuntimeParams`）
- 统计输入（`histogram_summary`、`calibration_features`）
- 可选的教师信息（`teacher_prediction`、`teacher_decoder_params`）
- 追溯字段（`source`、`provenance`、`metadata`）

**`StatCalibOutput`** 是一个 frozen dataclass，包含：
- 状态和原因（`status`、`reason`）——只能从预定义的 4 种状态 × 5 种原因中选择
- 解码器参数（`K`、`b`、`delta_b`）——仅在 `generated` 状态时存在
- 追溯字段（`source`、`provenance`、`metadata`）

**关键设计决策**：
- 只有 `generated` 状态的输出才能调用 `to_runtime_params()` 转换成快回路参数。其他状态会抛 `ValueError`，防止空值被误用。
- `from_delta_b()` 工厂方法做最简单的 residual-b 逻辑：`K` 沿用 prior，`b` 在 prior 基础上加 delta。这是后续真实校准逻辑的最小基线。
- 模块完全不导入 `ParamMapper`，保证物理隔离。

### 2.5 对后续开发的意义

在项目整体路线图（[docs/04_task_board.md](../../04_task_board.md)）中，T30 属于 Milestone 2I: Mechanism Evidence Hardening 的延续。

**直接意义**：
1. 后续 statcalib 实现任务有了精确的 API 边界。不需要一边写代码一边临时发明输入输出语义。
2. 接口合同中的状态值（`generated` / `not_generated` / `not_applicable` / `diagnostic_error`）延续了 T28 的 missing-vs-zero 语义修复，避免了下游把"没生成参数"当成"生成了零值参数"的错误。
3. `to_runtime_params()` 的防护机制确保 statcalib 不会被静默地当成已有 baseline 的替代品。

**后续任务**：
- 如果继续推进 statcalib，下一步是把 `StatCalibInput` / `StatCalibOutput` 接入 `SlowLoopRuntime`，形成真正的 comparator lane。
- T36（seed 失败机理诊断）是独立优先级，与 statcalib 推进互不阻塞。
- R10（teacher diagnostics 可观察性）、R20（correction saturation structural zero）、R23（aggregation/report writer focused tests）仍然开放，需要在后续任务中持续追踪。

## 3. 为什么 Reviewer 给了 PASS

### 3.1 任务确实完成了

对照任务包的"Expected Output"逐项检查：

| 要求 | 结果 |
|------|------|
| `StatCalibInput` exact contract | 10 个字段，带类型和验证 |
| `StatCalibOutput` exact contract | 8 个字段，带状态/原因枚举和转换边界 |
| Status values 和 reason strings | 4 种状态 + 5 种原因 |
| Provenance / source 字段 | Input 和 Output 均包含 |
| 到 `DecoderRuntimeParams` 的转换边界 | `to_runtime_params()` 仅对 `generated` 状态开放 |
| 独立 `statcalib.py` 模块 | 新建，不导入 ParamMapper |
| 不改 ParamMapper 主线 | `git diff` 确认无改动 |
| Focused test | 6 个测试全部通过 |

### 3.2 没有伪实现、mock、stub 或 hardcode

- `from_delta_b()` 做的是真实的 residual-b 计算（`b = prior.b + delta_b`），不是 placeholder。
- 验证逻辑（shape check、finiteness check、status/reason enumeration）都是真实的，不是硬编码的 pass-through。
- 没有 mock backend 或 stub 返回值。

### 3.3 没有破坏已有功能

- `ParamMapper`、`SlowLoopRuntime`、benchmark runner 全部无改动。
- 没有新 run directory。
- 没有修改 config、scenario、baseline、seed/repeat policy。

### 3.4 没有过度工程

- frozen dataclass + 验证 + 工厂方法 = API 边界模块的标准做法。
- 没有多余的抽象层、继承体系或配置对象。
- 复杂度与"接口合同"的职责匹配。

### 3.5 文档没有把计划写成事实

- gate 文档的"Explicit Non-Claims"仍然说 statcalib 不存在正式 benchmark 验证。
- 接口合同的测试只证明"接口定义是对的"，不证明"statcalib 已经能产生有效的校准参数"。

### 3.6 非阻塞备注

Reviewer 标注了 4 个非阻塞项：
- **N1**：gate 文档中"no source code was changed"的声明是 T26 时期的，T30 已经添加了源码。下次 Captain closeout 时需要更新。
- **N2**：`tests/` 目录缺少 `__init__.py`，对 `python -m unittest` 无影响，但可能影响 IDE 或 pytest 发现。暂时不阻塞。
- **N3**：运行测试产生了 `__pycache__/.pyc` 副作用，按 T19/T28 的 tracked-cache 治理处理，不作为技术改动提交。
- **N4**：`from_delta_b()` 用的是最简单的 residual-b 逻辑，后续真实校准实现需要确认这是否是正确的基线行为。

这些都不影响 T30 的接口合同质量，只是后续任务需要注意的点。
