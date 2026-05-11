# T28 人话版说明

## 一、这个任务是做什么的？——通俗解释

这个项目是一个"量子纠错的自适应解码系统"。简单来说：量子计算容易出错，需要实时纠正；纠错的关键是持续估计物理噪声参数（比如噪声有多大、偏移了多少），然后调整解码策略。

系统里有多种"估计器"（UKF、EKF、CNN 等）。其中有一种叫 `hybrid_residual_b`，它的做法是：先用经典方法（叫"teacher"）给出一个估计，然后让一个小型 CNN 去学习对这个估计做残差修正。

**问题出在哪？**

系统在 benchmark 跑完后，会输出一份"成绩单"（CSV 和报告），里面有一栏叫"teacher diagnostics"——意思是"teacher 对 CNN 的修正贡献有多大"。之前的问题是：

1. `hybrid_residual_b` 这个主线路径用的是 **broadcast** 方式传入 teacher 信息（像广播一样把 teacher 参数展平拼到输入特征里），而不是 **scalar-branch** 方式（用专门的标量通道传入）。
2. 但 teacher diagnostics（诊断指标）只有在 scalar-branch 方式下才会被计算和输出。
3. 下游的汇总代码和 CSV 写出代码不知道"teacher diagnostics 没有生成"，就把缺失的值填成了 `0.0`。
4. 结果看起来像"teacher 贡献为零"，但实际上是"这个指标根本没算过"。

这就像体检报告里某个项目没做，但系统把空格填成了"正常"，让人误以为真的检查过了。

**T28 做了什么？**

T28 不是去补 teacher diagnostics 的计算能力，而是修"报告怎么写"——让报告能正确区分：

- **"没做过"**（`not_applicable`）：这个模式根本不需要 teacher，比如 UKF
- **"做了但没出结果"**（`not_generated`）：teacher 信息启用了，但当前是 broadcast 路径，不会产出 scalar-branch 诊断数据
- **"结果是零"**（`true zero`）：真的算出来了，值就是 0.0

同时确认了 `correction_saturation_rate = 0.0` 是真的零值，不是被伪造的。

## 二、实现的详细解释

### 任务目标

1. 修改三个代码文件，使 teacher diagnostics 的输出语义从"缺失 → 0.0"改为"缺失 → null + 显式状态标签"
2. 用最小 smoke 验证修改后的输出确实能区分三种情况
3. 不扩大 benchmark 口径，不改历史数据，不改场景/基线集合

### 任务流程

```
slow_loop_runtime.py（产生诊断数据）
       ↓
run_hil_suite.py（聚合多次慢回路更新的诊断数据）
       ↓
run_p4_multiscenario_benchmark.py（写出 comparison.csv 和报告）
```

### 代码变化

#### 1. `cnn_fpga/runtime/slow_loop_runtime.py` —— 诊断数据的源头

**修改位置：** `_teacher_branch_diagnostics()` 和 `_teacher_branch_input_summary()` 方法

**之前的行为：**
- 不管 teacher diagnostics 有没有生成，都会走完整个计算路径
- 最后无条件设置 `teacher_contribution_l2` 等字段，缺失时被隐式填为 0.0

**现在的行为：**
- 先检查 teacher features 是否启用 → 没启用 → 返回 `not_applicable`，直接结束
- 再检查 `scalar_feature_dim > 0` → 不满足（broadcast 路径下 dim=0）→ 返回 `not_generated`，数值字段设为 `None`
- 满足条件 → 走正常 explain 路径 → 返回 `generated`
- explain 出错 → 返回 `diagnostic_error`

新增字段：
- `teacher_diagnostics_status`：四种状态标签
- `teacher_diagnostics_status_reason`：更详细的原因
- `teacher_diagnostics_support_boundary`：固定为 `"scalar_branch_only"`，标明当前系统边界
- `teacher_feature_layout`：记录 teacher 特征的布局方式
- `teacher_features_enabled`：布尔值，标记是否启用了 teacher 特征

#### 2. `cnn_fpga/benchmark/run_hil_suite.py` —— 汇总层

**修改位置：** `_aggregate_teacher_branch_diagnostics()` 函数

**之前的行为：**
- 只聚合数值指标（contribution_l2、gate_mean 等）
- 缺失时通过 `or 0.0` 隐式填零

**现在的行为：**
- 新增 `status_counts`、`reason_counts` 等计数器，统计各状态出现次数
- 新增 `diagnostics_seen` 计数器，区分"完全没有诊断数据"和"有数据但状态不是 generated"
- 汇总时解析出 resolved_status（单一状态或 mixed）
- 数值指标缺失时保留为 `null`，不再填 `0.0`

#### 3. `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` —— CSV 和报告写出

**修改位置：** `_aggregate_metric()`、新增 `_optional_float()` 和 `_aggregate_status_field()`、`_write_report()`、per-repeat 数据组装、comparison 行聚合、CSV 列定义

**之前的行为：**
- `_aggregate_metric()` 用 `float(item[key])` 取值，key 不存在会崩溃
- per-repeat 行用 `or 0.0` 把 None 压成 0.0
- CSV 列不包含 teacher diagnostics 状态

**现在的行为：**
- `_aggregate_metric()` 改为 null-safe：跳过 None 值，缺失时返回 `None`
- 新增 `_optional_float()` 辅助函数，安全地将值转为 float 或保留 None
- 新增 `_aggregate_status_field()` 辅助函数，安全地聚合状态字符串
- per-repeat 行保留 `teacher_diagnostics_status`、`teacher_diagnostics_status_reason`、`teacher_diagnostics_support_boundary` 等字段
- comparison.csv 新增 5 列：status、reason、support_boundary、generated_repeats、scalar_feature_dim
- Markdown 报告新增 "Teacher Diag" 列

### 对后续开发的意义

1. **机制分析不再被误导：** 之前如果看到 "teacher contribution = 0.0"，可能会去分析"为什么 teacher 没用"；现在会直接看到 `not_generated`，知道是因为当前路径根本不产出这个指标。

2. **R10 进一步收窄：** teacher diagnostics 缺失问题现在被明确编码成状态标签，不再是隐式的数值混淆。但 R10 仍未关闭——broadcast 路径还是没有 scalar-branch explain 能力。

3. **R21 关闭：** downstream `0.0` 强制转换问题已在当前 writer 语义层面解决。

4. **为 T26（statcalib）铺路：** statcalib 设计需要准确的机制指标。如果指标本身是模糊的（不知道是真的零还是没算过），后续分析都会建立在错误基础上。

5. **为论文 claim ledger 铺路：** 论文需要写 teacher 对 CNN 修正的贡献有多大。之前的数据是"全零"（其实是缺失），现在能正确说"当前 broadcast 路径不支持此指标"。

### 配置文件变化

无。本次任务没有修改任何配置文件（yaml / txt 等）。

## 三、为什么 reviewer 给出了 PASS_WITH_WARNINGS

### 总体判断

任务**确实完成了**它的核心目标：teacher diagnostics 的 missing-vs-zero 语义已经被修复，smoke 验证也确认了修复后的输出是正确的。

### 为什么是 WARNING 而不是纯 PASS？

1. **代码有一个小 bug：** markdown 报告的表头写了两行——一行是旧的（11 列，没有 Teacher Diag），一行是新的（12 列，有 Teacher Diag）。旧的没删掉，会导致 markdown 表格渲染异常。这不影响 CSV（主要数据源），但影响人类可读的报告。

2. **`__pycache__` 文件跟着变了：** 因为代码被改过、smoke 也跑过，三个 `.pyc` 缓存文件跟着更新了。这些文件本来就不该被 git 追踪（T5 的治理结论），但当前还在历史里，所以会出现在 diff 里。

3. **没有单元测试：** 改动的三个文件目前没有对应的单元测试。这次靠 smoke 验证，对于 bounded repair 可以接受，但如果后续要继续改聚合逻辑，应该补测试。

### 审查中确认的积极点

- **没有伪实现：** 所有状态标签都是基于真实的代码路径条件，不是硬编码假值
- **没有越界：** 只改了 Allowed files 里的文件，没有碰 benchmark 口径、场景集、历史数据
- **文档诚实：** Worker 的 review 和 explanation 都明确说了"这不代表 teacher 机理证据已修复"
- **smoke 输出可验证：** CSV 和 JSON 里的状态值确实区分了 `not_applicable` 和 `not_generated`
- **`correction_saturation_rate = 0.0` 保留为真实零值，没有被误改成 null**

### 建议修复

修 markdown 表头的重复行（`run_p4_multiscenario_benchmark.py` 第 318 行删掉即可），然后就可以由 Captain 收口了。
