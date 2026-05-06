# CNN-FPGA-GKP Legacy Handoff

更新时间：2026-05-06  
用途：为后续新会话/新开发流程提供可直接接棒的上下文摘要，避免重复试错。

## 1. 项目当前所处阶段

项目已经明显不再处于“从零打通链路”的阶段，而是进入了：

1. `P4` 后期的机制验证阶段
2. `teacher representation` 设计收敛阶段
3. 从“微调超参”转向“结构性升级/机理诊断”的决策阶段

当前更准确的判断是：

- `Full / Hybrid Residual-B` 仍是正式主线基准
- `Gated v5` 是目前最强的 `teacher-representation` 候选
- `v8 / v9` 证明了“继续靠 gate/clip/scale 微调”边际收益很低
- 下一步更值得做的是失败机理诊断、paper-inspired 分支、或更高层级的噪声/闭环建模补强

## 2. 已形成的稳定结论

### 2.1 正式主线口径

- 正式主线仍应保留 `Full`
- 不应把 `v8 / v9` 升成主线
- `Gated v5` 可以作为当前最强 teacher-representation 候选保留

### 2.2 关于 teacher params 的核心认识

目前已经较明确地支持以下判断：

1. 问题不只是“teacher params 要不要删”
2. 更关键的是“teacher params 怎么编码、怎么注入”
3. 把全局 teacher 标量粗暴广播成整幅平面通道，不是好表示
4. 更合理的方向是低维 `scalar branch`、去冗余、gated 注入、以及更贴近闭环语义的训练目标

### 2.3 offline 改善不等于 formal HIL 改善

这是已经反复被强调的机制结论，必须保留：

- 离线训练指标变好，不代表 formal HIL 一定同步变好
- 某些 teacher 变体在离线或局部 benchmark 下有优势，但进到闭环语义后会翻转
- 因此不能只看 `MSE/MAE/R2` 或局部单 seed 表现决定主线

## 3. 近期 teacher-representation 关键实验结果

## 3.1 Gated v5

`Gated v5` 是目前最值得保留的分支。

历史关键结果可参考此前 paired benchmark 汇总：

- `seed=20260427`：`Full = 0.779861`，`Gated v5 = 0.547688`
- `seed=20260428`：`Full = 0.798706`，`Gated v5 = 0.710131`
- `seed=20260429`：`Full = 0.688990`，`Gated v5 = 0.674559`

三 seed / 四场景 paired 汇总里，`Gated v5` 整体优于 `Full`，但在 `20260429` 上接近持平甚至局部翻车。  
因此结论应写成：

- `Gated v5` 是当前最强 candidate
- 但还未强到可以完全替代 `Full`
- 它已经足够说明“少数关键 teacher 标量 + 低维 gated 注入”这条方向是对的

## 3.2 Gated v8

`v8` 的设计目标是：

- 保持 `aggressive_param_rate = 0`
- 比 `v7` 增强 `b_q / b_p` 主支路有效贡献

结果：

- 对 `v7` 有改进
- 但没有稳定超过 `Full`
- 在不同 seed 之间不够稳

相关汇总文件：

- [v8 paired summary.csv](/D:/Codes/Quantum/DriftAdaptiveQEC/runs/teachrepr_v8_bench_pair/paired_20260502_014626/summary.csv)

## 3.3 Gated v9

`v9` 的目标是进一步抑制 `v8` 的过冲风险，尤其是想缓解 `20260429`。

实际结果相反，`v9` 退化明显。

来自：

- [v9 paired summary.csv](/D:/Codes/Quantum/DriftAdaptiveQEC/runs/teachrepr_v9_pair/paired_20260502_115405/summary.csv)

三 seed 汇总：

- `20260427`: `Full = 0.836762`，`v8 = 0.737844`，`v9 = 0.806988`
- `20260428`: `Full = 0.828015`，`v8 = 0.738263`，`v9 = 0.819800`
- `20260429`: `Full = 0.532874`，`v8 = 0.766638`，`v9 = 0.833991`

整体平均：

- `Full = 0.732550`
- `v8 = 0.747582`
- `v9 = 0.820259`

这说明：

1. `v9` 不适合升主线
2. `v9` 的 teacher 分支被压得过弱
3. `v9` 不是“更稳的 v8”，而是“过度保守、收益流失”的版本

## 3.4 关于 v8 / v9 的共同教训

目前已经能明确得出一个重要设计结论：

- 单纯继续微调 `scalar_gate_init_bias / scalar_norm_clip / scalar_feature_weights / residual_clip_b / residual_scale_b`
- 很难再带来实质性提升

原因是：

1. 一旦 teacher 分支更激进，可能在部分 seed 翻车
2. 一旦 teacher 分支更保守，收益马上消失
3. 这说明当前瓶颈已不是“超参还没拧对”，而是“表征方式和闭环目标本身不够对”

## 4. 当前不建议继续投入的方向

以下方向不建议继续作为主要算力投入对象：

1. 继续做 `v8 -> v9 -> v10 -> v11` 这种小步 gate/clip/scale 微调
2. 继续做大规模“删/不删 teacher params”长跑
3. 继续扩 `PB Bound / PB ST` 作为论文主线
4. 继续把 `No TeacherParams` 当成主叙事

原因：

- 已经有足够结果说明这些方向很难再给主线带来决定性推进
- 进一步投入大概率是高成本、低信息增益

## 5. 当前更值得做的工作

这里是最关键的接棒建议。

### 5.1 第一优先级：做 `20260429` 的失败机理诊断

推荐优先做，而不是继续调新版本。

目标：

- 解释为什么 `Gated v5` 在多数 seed 上明显更好，但在 `20260429` 上不稳

建议产出：

1. `Full vs Gated v5` 的逐 window / 逐 commit 时间序列对照
2. 关键量可视化：
   - `teacher_b_q / teacher_b_p`
   - 预测 residual
   - commit 后实际 `b_q / b_p`
   - overflow / saturation
   - LER 恶化区间
3. 判断翻车机制更像哪一类：
   - 符号偏移
   - 幅度过冲
   - 响应滞后
   - teacher 本身在该 seed 下不稳

这一步的价值很高，因为它会直接决定后续该：

- 重写 loss
- 换 teacher 表征
- 做更强约束
- 还是补 teacher 估计质量

### 5.2 第二优先级：做 paper-inspired 分支

已有设计草案文件：

- [CNN_FPGA_GKP_paper_inspired分支实验设计草案.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md)

建议后续将其作为“结构升级”，而不是“再微调一个 gated vX”。

建议 framing：

1. 第一阶段：`effective noise state estimation`
2. 第二阶段：`residual correction / calibration`

建议要点：

- teacher 标量继续保留在低维 scalar branch
- 不再做 broadcast 平面注入
- loss 尽量更贴近闭环有用量，而不只是 `delta_b` 本身
- 可考虑把短 horizon surrogate 风险、映射后 `b` 误差、overflow 风险一起纳入

### 5.3 第三优先级：把机理文档化，而不是继续盲调

建议在新流程下，把以下内容系统沉淀：

1. 哪些结论已经稳定
2. 哪些结论只是在某个 seed / 场景下成立
3. 哪些改动是“高收益但有风险”
4. 哪些改动属于“已经基本证伪”

## 6. 近期做过的重要工程修复

### 6.1 benchmark 后台启动修复

已经对后台长任务启动做过一轮重要修复：

- 文件：[run_p4_teacher_representation_paired.py](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_p4_teacher_representation_paired.py)

做过的事情：

1. 增加了 `--detach` 和 `--detach-log-dir`
2. 增加 detached 自重启逻辑
3. 增加 stdout / stderr / meta 日志落盘
4. 修复 Windows 下 `DETACHED_PROCESS` 触发嵌套 `torch/python` 子进程异常退出的问题
5. 改为更稳的隐藏窗口后台方式

这个修复是有效的，后续如果继续跑长 benchmark，建议继续用这套 runner，而不是手工起后台进程。

## 7. 当前相关关键文件

建议后续新会话优先参考这些文件：

### 项目结论类

- [CNN_FPGA_GKP_阶段结论.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/CNN_FPGA_GKP_阶段结论.md)
- [CNN_FPGA_GKP_后续仿真与工程补强计划.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md)
- [CNN_FPGA_GKP_paper_inspired分支实验设计草案.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md)

### runner / benchmark 类

- [run_p4_teacher_representation_paired.py](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_p4_teacher_representation_paired.py)
- [run_p4_multiscenario_benchmark.py](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py)

### teacher-representation 配置类

- [experiment_runtime_b_residual.yaml](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/config/experiment_runtime_b_residual.yaml)
- [experiment_runtime_b_residual_norm_gated_teacher_v5.yaml](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml)
- [experiment_runtime_b_residual_norm_gated_teacher_v8.yaml](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v8.yaml)
- [experiment_runtime_b_residual_norm_gated_teacher_v9.yaml](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v9.yaml)

### 结果文件

- [v8 paired summary.csv](/D:/Codes/Quantum/DriftAdaptiveQEC/runs/teachrepr_v8_bench_pair/paired_20260502_014626/summary.csv)
- [v9 paired summary.csv](/D:/Codes/Quantum/DriftAdaptiveQEC/runs/teachrepr_v9_pair/paired_20260502_115405/summary.csv)

## 8. 建议的新会话启动方式

如果后续换新会话，建议不要从“继续调 v10”开始，而是从下面任一任务开始：

1. “请基于 `Gated v5` 与 `Full` 的已有结果，专门做 `seed=20260429` 的失败机理诊断工具与分析文档。”
2. “请基于 `CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`，实现第一版 paper-inspired 分支，不再继续 gated 微调。”
3. “请把 `P4` 当前 teacher-representation 相关结论系统回写到阶段结论文档，并形成新的实验决策树。”

## 9. 总结性判断

一句话总结当前项目状态：

- 现在继续做 `gated_v10 / v11` 这类小修小补，信息增益已经很低
- 更有意义的工作是：做失败机理诊断、做结构升级、做更贴近闭环语义的训练/评估设计

这份文档的核心目的，就是帮助后续会话直接从“更有意义的工作”开始，而不是再绕回高成本的微调循环。
