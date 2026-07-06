# 投稿稿 FPGA datapath contract 补强记录

日期：2026-07-03

## 本次修改

本次只补强 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的方法叙事，不新增实验、不运行硬件、不升级证据等级。

新增内容：

1. 在 `Method` 的 `Affine fast path` 之后新增 `FPGA-facing datapath contract` 小节。
2. 新增 `tab:fpga-datapath-contract`，把 future FPGA fast path 拆成五个可审查对象：
   - wrapped syndrome input；
   - active/inactive `(K,b)` parameter bank；
   - fixed-point affine matrix-vector arithmetic；
   - corrected displacement / residual-boundary output；
   - timing、resource、power、source-vs-board 等 device evidence。
3. 在 `Analytical operation counts quantify the fast-path cost argument` 段落中回指该 contract，说明 operation-count、Q4.20 fixed-point emulation 和 runtime counters 共同构成当前软件/分析层支撑。
4. 将摘要中的 LER 收益句改为 descriptive mean 口径，明确 1.8--2.9% / 2.3% average 不是 inferential claim；同时把未完成硬件项改为 board timing、resource、power 和 source-vs-board measurements。
5. 清理 4 处 appendix / caption 中的内部指令式口吻，把 “manuscript should / paper may state / current figure package” 改为声明式的 source-data、claim-language 和 figure-scope 表述。

## 为什么需要这次补强

审稿人如果看到标题中含 `FPGA`，通常会追问三个问题：

1. fast path 到底是什么硬件对象，而不是泛泛的“可部署”说法；
2. 当前已有证据能支持到哪一层；
3. 什么证据仍然必须由真实板卡或综合报告提供。

此前稿件已经有 analytical cost、fixed-point parity 和 runtime counters，但这些材料分散在 Results。新增小节把它们前移为方法合同，使读者先理解“未来要验证的板级对象是什么”，再读后面的 cost / parity / counters。

## 证据边界

- 当前只支持 `software + analytical` 层的 datapath feasibility。
- `tab:fpga-datapath-contract` 中的 device evidence 明确标为未测。
- 本次没有补 FPGA synthesis、timing closure、resource、power、DMA/MMIO log、bitstream/RTL hash 或 source-vs-board vector。
- 不得把这次补强写成 real FPGA implementation success。
- 摘要中的 LER 数值仍来自 two-repeat software-HIL source data，只能写成 descriptive ranking 和 descriptive delta。

## 对投稿稿的作用

这次修改提高的是论文叙事完整性：把“低延迟、低成本、可验证”的优势从口号变成可审稿的接口定义。它不解决硬件实验缺口，但能让硬件实验作为 future validation placeholder 时更加可信和具体。
