# Codex 工作规则

本文件是本仓库中 Codex/agent 的全局提示词和工作约束。除非用户明确覆盖，所有后续任务都应遵守本文件。

## 语言与沟通

- 文档优先使用中文；必要的公式、变量名、英文技术术语和论文标题可以保留英文。
- 最终回复要简洁说明做了什么、改了哪些文件、如何验证。
- 不要扩大任务范围。若发现需要新增工作，先记录风险，再判断是否插入新 task。

## 文档源规则

- `docs/new_task_board.md` 是当前任务状态源。任何 task 的状态变化、插入任务、阻塞、完成或废弃，都必须同步到该文件。
- `docs/new_tasks/` 是单 task 完成记录目录。每完成一个 task，都必须在该目录新增或更新对应记录。
- `docs/new_risks.md` 是风险登记与插入任务判断文档。每完成一个 task 后都要复核并更新风险。
- `README.md` 是项目入口和文档地图。新增重要目录、协议或治理文件时，要同步 README。
- `physics/` 是当前 GKP 物理仿真代码库，封装态制备、噪声、syndrome 测量、纠错和逻辑追踪等功能；后续可随 task 推进改进。

## 标准任务流程

1. 读取 `docs/new_task_board.md`，确认当前推荐任务和目标 task。
2. 开始执行前，将目标 task 状态改为 `In Progress`。
3. 执行 task 时保持证据优先：公式、代码、仿真、图、baseline、测试或文献必须能追溯。
4. 完成 task 后，在 `docs/new_tasks/` 下记录该 task 的输入、输出、验证、风险和后续建议。
5. 更新 `docs/new_risks.md`：新增风险、关闭风险、调整风险等级，并判断是否需要插入新 task。
6. 更新 `docs/new_task_board.md`：把状态改为 `Review` 或 `Done`，记录产物和验证摘要；若插入新 task，也要写入任务板。
7. 最后再给用户汇报改动和验证结果。
8. 如果现有代码/配置/数据与 task 要求有冲突，需要你自行选择更好的方案进行修改。然后在 `docs/new_tasks/<TaskID>_*.md` 中记录修改动机、行为变化、验证方式和对论文 claim 的影响。
9. 你可以随时阅读 @docs\任务版改进记录 中的记录，通过这些记录去根据任务查看对应的论文(放在 @relative_papers 中的，或者直接在网页搜索查阅)，通过论文来进一步强化实验。
10. 需经常参考的论文主要是 @relative_papers\Non-Markovian_feedback_for_optimized_quantum_error_correction 和 @relative_papers\Quantum_error_correction_of_a_qubit_encoded_in_grid_states_of_an_oscillator 和 @relative_papers\Real-time_quantum_error_correction_beyond_break-even 这3篇，对于综述 @relative_papers\Advances_in_Bosonic_Quantum_Error_Correction_with_Gottesman–Kitaev–Preskill_Codes_Theory_Engineering_and_Applications 主要看实验部分。注意你可以直接联网搜索并使用相关论文的源码(可以直接clone到本地)，比如Non-Markovian这篇文章就有github源码（仓库是 Matteo-Puviani/GQF ）
11. 每完成一个task，都要仔细检查是否真正完成了任务，是否有伪实现或者简化实现(偷懒简化代码导致形式上合理但是实际上不合理)，尤其是实验部分的代码是否真的能够给出理想的结果，如果不行那就要仔细分析并修改，直到能够给出理想的结果。
12. 每完成一个task，都允许你直接提交到 git 仓库，不需要等待用户确认。注意修改记录要用中文详细写。


## Task 完成记录要求

每个 task 完成记录建议命名为：

`docs/new_tasks/<TaskID>_<short_slug>.md`

示例：

`docs/new_tasks/T0.1.1_scope_freeze.md`

记录必须包含：

- task ID、标题、日期、状态；
- 输入材料；
- 实际完成内容；
- 产物路径；
- 验证方式和结果；
- 风险复核；
- 是否需要插入新 task；
- 对 `docs/new_task_board.md` 的同步说明。

## 风险与插入任务规则

- 风险等级使用 `Low`、`Medium`、`High`、`Critical`。
- 迫切程度使用 `Monitor`、`Soon`、`Immediate`。
- 若风险只是未来可能发生，登记在 `docs/new_risks.md`，不插入 task。
- 若风险会阻塞当前 task、使当前结论不可信，或可能导致错误论文 claim，应插入新 task。
- 插入任务 ID 使用 `T-RISK-YYYYMMDD-NN`，并写入 `docs/new_task_board.md` 的“插入任务区”。
- 插入任务必须说明来源风险、目标、产物、通过标准和建议执行顺序。

## 代码与实验实现规则

- 开始代码任务前，先确认对应 task 的输入、输出、通过标准和失败分支。
- 若 task 涉及 GKP 态制备、噪声通道、syndrome 测量、纠错或逻辑错误追踪，优先检查并复用 `physics/` 的既有接口。
- 修改 `physics/` 时，必须在对应 `docs/new_tasks/<TaskID>_*.md` 中说明修改动机、行为变化、验证方式和对论文 claim 的影响。
- 优先实现强 baseline 和可复现实验，再做 CNN claim。
- 不要只和 standard binning 比；涉及性能 claim 时至少考虑 static MAP、oracle MAP、Bayesian、EWMA/Kalman 或 sliding-window baseline 的适用性。
- 不要声称 CNN 超越 oracle MAP；贡献应表述为缩小 drift 下 static MAP 与 oracle MAP 的 gap。
- single-mode GKP 不使用 surface-code threshold 语言；使用 operational pseudo-threshold、break-even boundary 或 logical lifetime gain。
- FPGA 结果未实测前，只能写 hardware-aware simulation、fixed-point estimate 或 synthesis estimate，不能写成真实硬件结论。

## 文件修改纪律

- 保留用户已有改动，不回滚无关文件。
- 文档默认中文。
- 新增重要文档或目录时，同步 README 的文档地图。
