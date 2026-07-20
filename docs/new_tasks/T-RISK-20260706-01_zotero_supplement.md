# T-RISK-20260706-01 补充 High 优先级 Zotero 文献

- Task ID：`T-RISK-20260706-01`
- 来源风险：`R-008`
- 日期：2026-07-06
- 状态：Done

## 输入材料

- `docs/literature_matrix.md`：T0.2.1 形成的文献矩阵和优先补库清单。
- 本地 Zotero 选中目标：`我的文库 / 量子计算 / 量子纠错 / GKP码`。
- Zotero local API / Connector：`status --json` 显示 local API 与 connector 均可用。

## 实际完成内容

1. 对 High 优先级缺口做 Zotero 精确搜索，确认以下条目导入前未命中：
   - Brady et al., GKP/bosonic QEC 综述；
   - Fukui/Tomita/Okamoto 2017 analog-information GKP decoding；
   - Campagne-Ibarcq et al. 2020 grid-state oscillator QEC；
   - Fluehmann et al. 2019 trapped-ion oscillator GKP；
   - Bhardwaj/Takou/Lin/Brown 2025 adaptive drifting-noise estimation。
2. 额外补入一条与 R-008 直接相关的 noise-aware decoding 文献：
   - Hockings/Doherty/Harper 2025, Improving error suppression with noise-aware decoding。
3. 生成导入用 BibTeX 文件：
   - `docs/tasks/T-RISK-20260706-01_zotero_supplement.bib`
   - `docs/tasks/T-RISK-20260706-01_zotero_completion_round2.bib`
4. 通过 Zotero Connector 导入当前选中的 `GKP码` collection。
5. 将新增 Zotero item key 回填到 `docs/literature_matrix.md`。
6. 根据用户要求再次复查“是否齐全”，将矩阵中仍为 `缺` 或 `部分` 的实际论文/报告条目继续补齐；工具、索引和教程类条目保留为 `工具/泛称`。

## 导入结果

### 第一批：High 优先级缺口

| 文献 | Zotero item key | 来源 |
| --- | --- | --- |
| Brady et al., Advances in Bosonic Quantum Error Correction with Gottesman-Kitaev-Preskill Codes | `G8WF45GW` | `arXiv:2308.02913`, DOI `10.48550/arXiv.2308.02913` |
| Fukui/Tomita/Okamoto, Analog Quantum Error Correction with Encoding a Qubit into an Oscillator | `ELIT8KDY` | `arXiv:1706.03011`, DOI `10.1103/PhysRevLett.119.180507` |
| Campagne-Ibarcq et al., Quantum Error Correction of a Qubit Encoded in Grid States of an Oscillator | `SKTLAMK7` | `arXiv:1907.12487`, DOI `10.1038/s41586-020-2603-3` |
| Fluehmann et al., Encoding a Qubit in a Trapped-Ion Mechanical Oscillator | `BD9BUQQJ` | `arXiv:1807.01033`, DOI `10.1038/s41586-019-0960-6` |
| Bhardwaj et al., Adaptive Estimation of Drifting Noise in Quantum Error Correction | `LXHTGCCU` | `arXiv:2511.09491`, DOI `10.48550/arXiv.2511.09491` |
| Hockings/Doherty/Harper, Improving Error Suppression with Noise-Aware Decoding | `KIXGCRQB` | `arXiv:2502.21044`, DOI `10.48550/arXiv.2502.21044` |

### 第二批：计划池剩余实际论文/报告

第二批通过 `docs/tasks/T-RISK-20260706-01_zotero_completion_round2.bib` 补入 35 个条目，覆盖 Terhal review、GKP lattice perspective、finite-energy stabilization、loss logical channel、multimode GKP / QIFE、GKP 实验平台、NN/QEC decoder、自适应噪声估计、FPGA/real-time decoder、Toric-GKP 和 Color-GKP 背景。

代表性新增 Zotero key 包括：

- GKP 基础/有限能量：`QN2V2E9N`, `FCVY5YQF`, `Q8XYG2RV`
- GKP 本体解码：`VUWM6BWZ`, `7343N9VB`, `VSCNXUXR`, `ZN2J6VBA`, `6M7JUVYL`, `E65BLF44`
- GKP 实验与仿真：`6ZUJUWFJ`, `92EV5J6T`, `2BVT7J9L`, `QBSZYXFU`, `QH6FUBSF`, `26BDQDZ3`, `RGWXWMT3`, `MDT9NISD`, `KVR3T5GL`
- NN/QEC 与自适应估计：`ATZ593C4`, `RCEJFYWR`, `DM2HW3HW`, `WZ5MFK9G`, `AKA6SWVA`, `E4FLQ7NN`, `L8G5WJU7`, `G2J337CN`, `4GJNRHY5`, `RLV7AMCQ`, `QHVGC3Y5`
- FPGA/外层码背景：`WHLYSF34`, `NA9WGEM2`, `ESEZ6QQK`, `G2NMLD5W`, `TZXA9QEC`, `GJHZLV33`

## 验证方式和结果

- 导入前使用 Zotero `search --json` 对 5 个 High 优先级缺口逐项搜索，结果为空。
- `import-bibtex --file docs/tasks/T-RISK-20260706-01_zotero_supplement.bib --yes` 返回 HTTP `201`，并返回 6 个新增 item key。
- 导入后使用 Zotero `search --json` 逐项验证，6 个新增条目均可命中。
- 第二批 `import-bibtex --file docs/tasks/T-RISK-20260706-01_zotero_completion_round2.bib --yes` 在 CLI 侧超时，但 Zotero inventory 随后确认 35 个条目均已写入本地库。
- `docs/literature_matrix.md` 已同步回填新增 item key；计划池实际论文/报告条目的 Zotero 字段不再出现 `缺` 或 `部分`。
- `工具/泛称` 条目包括 Error Correction Zoo、The Walrus / Strawberry Fields、QuTiP / QuantumOptics.jl、Stim / PyMatching，不作为单篇 Zotero 论文强制补库。

## 风险复核

- `R-008` 的 Zotero 覆盖缺口已缓解：计划池中的实际论文/报告条目已经补齐。
- 后续风险不再是计划池缺文献，而是写作阶段可能需要额外的 forward/backward citation expansion。若 T0.2.2 超出当前参考池，应另行登记补库需求。

## 是否需要插入新 task

暂不需要插入新 task。若 T0.2.2 写 gap statement 时发现某个 Medium 缺口会直接支撑核心 claim，再按风险规则新增插入任务。

## 对 task_board 的同步说明

- `docs/task_board.md` 中 `T-RISK-20260706-01` 已同步为 `Done`。
- 当前推荐任务更新为 `T0.2.2`。
