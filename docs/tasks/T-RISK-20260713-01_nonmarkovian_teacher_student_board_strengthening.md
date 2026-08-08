# T-RISK-20260713-01：Non-Markovian teacher-student 任务板补强

- Task ID：`T-RISK-20260713-01`
- 标题：根据 Non-Markovian feedback PRL 补强任务板
- 日期：2026-07-13
- 状态：`Done`

## 输入材料

- `docs/relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction.md`
- 该文 Supplemental Material 与 Markdown 中引用的 23 张图片
- `docs/task_board.md` v2
- `docs/experiment_plan.md` 第 14 节
- `docs/risks.md`
- `docs/tasks/T-RISK-20260712-01_experiment_aligned_task_board_restructure.md`

## 触发证据与判断

该 PRL 是 model-based numerical study，不是真实 cavity/transmon 实验。它与本项目最接近的部分不是“用 RNN 做 decoder”，而是：利用 ancilla measurement history 形成隐状态，逐 half-cycle 调整 sBs gate parameters，并以 Feedback-GRAPE 训练 non-Markovian controller。由此暴露出 v2 的五个证据缺口：

1. 原有 decoder、controller 和 oracle 术语不足以区分两类任务；
2. 缺少 latest-outcome Markovian、autonomous 和 finite-horizon control oracle 等 memory-specific 对照；
3. 未证明现有仿真能支撑可微 quantum trajectory 与 Feedback-GRAPE；
4. best-of-N、短训练 horizon 外推和 model mismatch 需要独立审计；
5. PRL-style 完整 GRU 未必适配约 300 元 FPGA，必须设置蒸馏 student 和硬件降级路径。

这些缺口会直接影响论文方法归属、因果解释和硬件 claim，因此需要插入治理任务；但不改变当前执行指针，也不预先承诺 NMF teacher 一定可行。

## 实际完成内容

### 1. 任务板 v2.1

- 在 M1.4 冻结 decoder/controller、decoder oracle/control oracle、teacher/student 术语。
- 在 M2.3 增加 differentiable sBs trajectory simulator、完整梯度校验、资源扫描和方向性 ranking reproduction gate。
- 在 M3.2 增加 memoryless FNN、autonomous sBs、finite-horizon control oracle、指数递推和 memory 消融。
- 在 Phase 4 增加 bounded residual RNN/GRU teacher、隐状态解释、teacher-to-student 蒸馏和证否分支。
- 在 Phase 5 增加物理时间公平性、实验可行性约束、all-agent selection audit、horizon extrapolation、model mismatch 和部署候选资源门。
- 在 Phase 6 将定点低维 student 设为默认板上主线；完整量化 GRU 仅在资源与 deadline 通过后作为可选对照。
- 在 Phase 7 重构主图和 reviewer challenge，使论文必须回答 memory 增益、PRL 近邻差异和低成本 FPGA 可部署性。
- 在跨阶段约束中写明可行性失败时回退 v2 drift/regime-aware MAP-LUT 主线。

### 2. 实验计划低频修订

在 `docs/experiment_plan.md` 新增第 15 节，登记术语边界、Feedback-GRAPE 可行性门、公平 baseline、teacher-to-student 路线以及强主张/证否主张。该修订由重要近邻文献核查触发，实质改变后续 Phase 2—7 的方法门和证据结构，满足低频更新条件。

### 3. 风险复核

在 `docs/risks.md` 新增并路由：

- `R-019`：controller/decoder 与两类 oracle 混称；
- `R-020`：可微 trajectory/Feedback-GRAPE 不可行；
- `R-021`：best-of-N 与短 horizon 外推偏差；
- `R-022`：完整 GRU 超出低成本 FPGA 资源或 deadline；
- `R-023`：策略在 `p(g)`、reset/leakage burden、slew 或 model mismatch 下不可实验。

所有风险均已由新增正常 task 覆盖，当前不再插入额外执行任务。

## 产物路径

- `docs/task_board.md`
- `docs/experiment_plan.md`
- `docs/risks.md`
- `docs/tasks/T-RISK-20260713-01_nonmarkovian_teacher_student_board_strengthening.md`

## 验证方式和结果

- 任务 ID 结构检查：通过；新增 task ID 唯一，正式任务表不存在重复 ID。
- 状态检查：通过；已完成主线 task 的 `Done` 状态未改变。
- 当前指针检查：通过；仍为 `T1.3.3`。
- 风险路由检查：通过；`R-019`—`R-023` 均关联到任务板中的明确任务。
- 冻结文件检查：通过；`docs/rough_plan.md` 未修改。
- 文档差异检查：通过；`git diff --check` 无空白错误。
- 本任务只修改治理和实验设计文档，未执行代码测试。

## 风险复核结论

本次补强没有把完整 RNN 强行变成项目主线。项目形成两条可证伪路径：

- 强路径：可微模型与 teacher 通过门控，student 保留足够 simulated gain，并在板上满足确定性 deadline；
- 降级路径：teacher、蒸馏或完整闭环任一关键门失败，删除 NMF 性能主张，保留 PRL-inspired recurrence 强 baseline，继续 v2 MAP-LUT + HIL 主线。

这种分支能避免单一高风险方法阻塞论文，同时保留对近邻 PRL 的正面、可审计回应。

## 是否需要插入新 task

否。`R-019`—`R-023` 已由 v2.1 新增正常 task 覆盖；只有未来执行中出现未被现有 gate 捕获且会使结论不可信的阻塞风险，才新增 `T-RISK-*`。

## 对任务板的同步说明

- `T-RISK-20260713-01` 已在“插入任务区”登记为 `Done`。
- 已追加 `Proposed -> In Progress -> Done` 进度日志。
- 当前推荐任务保持 `T1.3.3`，未提前启动任何新增实验 task。
