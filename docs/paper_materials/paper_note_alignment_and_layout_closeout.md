# T78 Note Alignment and Layout Closeout

## 1. 作用与边界

本文件记录 `T78` 对 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的三类收口动作：

1. 非结果层 evidence-facing wording 校准；
2. `statcalib` 在 note 内的层级降权；
3. LaTeX warning / layout 收口。

它不新增实验，不重写 `T77` 的结果层事实，也不把 note 升级成新的证据来源。`T77` 负责结果层同步，`T78` 只负责 note 的非结果层与版面收口。

## 2. 本轮实际改动 section

| note section / area | 改动目标 | 直接 source task / review | `T78-SCOPE` 覆盖 |
| --- | --- | --- | --- |
| `Title` | 去掉把 `statcalib` 写成并列主线的标题暗示 | `T77` manifest, `T70`, `T77_review` | 是 |
| `Abstract` | 保留主线结果层不变，同时把 `statcalib` 明确压回 supplement-side lane | `T77` manifest, `T70`, `T77_review` | 是 |
| `Introduction` | 增加一句当前 evidence stack 的层级说明，阻断把 deployment / `statcalib` 读成并列主结果 | `T77` manifest, `T70`, `T72`, `T77_review` | 是 |
| `Summary of Contributions` | 把贡献 4 的 `statcalib` 口径压回 supplement-side extension lane | `T77` manifest, `T70`, `T77_review` | 是 |
| `Summary of Contributions / Metric-level advantages` 表 | 用 `raggedright` 列格式消掉表内可修的 `Underfull \hbox` | `T77_review` layout warning | 归属 `Summary of Contributions` |
| `Relationship to Existing Work` | 把 `Advantages...` 小节中的 deployment / comparator 叙事压回 architectural and evidence-bounded | `T70`, `T72`, `T77_review` | 是 |
| `Numerical Results and Benchmark Plan` 中 `statcalib` bridge + 三个小节 | 视觉层级降权：从与主结果同级的 `subsection` 降为 lower-level `subsubsection`，并加 bridge 句明确 supplement-side | `T70`, `T76`, `T77_review` | 否，单独在本文件记录 |
| `Discussion` | 把 real-board boundary 那一句拆开，保留 `NO_GO` 与 layered evidence 口径，同时顺手减少版式 warning | `T72`, `T76`, `T77_review` | 是 |
| `Conclusion` | 把 `statcalib` 改写成 supplement-side strong lane，而不是 promoted comparator | `T70`, `T72`, `T76`, `T77_review` | 是 |

## 3. `statcalib` 层级降权记录

### 3.1 调整前

- `Statistical calibration as a bounded extension lane`
- `Local sensitivity inside the statistical-calibration extension lane`
- `Per-scenario best variants and the no-promotion gate`

这三段都以 `\subsection` 出现在 `Numerical Results` 主体里，视觉上过于接近：

- `Four-scenario affine benchmark`
- `Feature and teacher ablations`
- `Mechanism probe for residual-b behavior`

容易让读者把 `statcalib` 误读成和主结果层、机制支撑层并列的 Results pillar。

### 3.2 调整后

- 先加一条 bridge 句，明确“后面三条只是 supplement-side extension lane record”；
- 再把三条 `statcalib` 标题全部降为 `\subsubsection`；
- 原有表格和数字不动，只改 note 内部阅读层级与 heading 强度。

### 3.3 不变边界

本轮仍然保留以下固定口径：

- `extension lane`
- `no promotion`
- `persistent tie`
- `no unique clean threshold`

本轮没有把 `statcalib` 改写成：

- promoted mature comparator
- `T24` 替代表
- `.tflite` lane
- real-board lane

## 4. 本轮明确未校准 section

以下 section 继续保持 `T78` 前状态，不应被误当成“已全稿重新校准”：

| 未校准 section / area | 原因 |
| --- | --- |
| `Brief Review of the GKP Code` | 任务包明确禁止扩大到理论章节重写 |
| `Noise and Drift Model` | 不属于当前 paper-facing 真实性冲突点 |
| `Model Architecture` 主体 | 本轮只允许处理 evidence-facing wording，不做方法章大改 |
| `Experimental Setup` | `T77` 的结果层同步已足够，本轮不需要再改 |
| `Numerical Results` 里除 `statcalib` hierarchy 之外的其他结果段 | `T77` 已完成结果层同步；`T78` 不重复改写主结果层 |

## 5. 编译与 warning before / after

### 5.1 工具链

- `latex_doctor.py --json`
  - 结果：`existing-usable`
  - 活跃工具链：`C:\texlive\2024\bin\windows`
- `compile_latex.py D:\Codes\Quantum\DriftAdaptiveQEC\docs\paper_notes\CNN_FPGA_GKP_theory_note_draft.tex --json`
  - 结果：编译成功
  - 编译器：`latexmk` / `pdfTeX` (`TeX Live 2024`)

### 5.2 warning 计数

| 项目 | `HEAD` 基线 | `T78` 编译后 |
| --- | --- | --- |
| `Underfull \hbox` | 32 | 0 |
| `Overfull \hbox` | 0 | 0 |
| `pdfTeX warning` | 不作为 `T78` 主问题 | 0 |

### 5.3 本轮消除 warning 的具体手段

1. 把 `Metric-level advantages` 表的四列改成 `raggedright` 列格式，避免窄列强制两端对齐带来的 `Underfull \hbox`；
2. 把 `Discussion` 里 `real-board gate/regeneration/provenance` 的长句拆得更易断行；
3. 在 `statcalib` heading 变更后重新完整编译，刷新 `.toc/.out/.aux` 等辅助文件。

### 5.4 剩余 warning 说明

当前 `log` 里没有残余 `Underfull \hbox`、`Overfull \hbox` 或 `pdfTeX warning`。`Select-String -Pattern 'Warning'` 还能匹配到 `infwarerr` 包名中的单词 `warning`，但这不是实际编译告警。

## 6. `T78-SCOPE` 覆盖情况

本轮 `T78-SCOPE` 注释覆盖的非结果层 section 如下：

- `Title`
- `Abstract`
- `Introduction`
- `Summary of Contributions`
- `Relationship to Existing Work`
- `Discussion`
- `Conclusion`

`Numerical Results` 中的 `statcalib` 层级降权是结果层内部的版面变化，因此没有用 `T78-SCOPE` 去把整个 Results 区重新标成“已校准”；它改由本文件和 `paper_results_section_assembly_pack.md` 的 `T78 note hierarchy supplement` 记录。
