# 投稿稿 future FPGA success 叙事补强记录

日期：2026-07-06

## 修改对象

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 修改内容

本次在 Discussion 的 supported-advantage table 后新增一个投稿式解释段落，回答审稿人和读者可能追问的核心问题：

- 如果未来真实 FPGA 实验成功，这项工作的物理优势到底会被证明成什么；
- 成功的 FPGA evidence 应当测量哪些具体对象；
- 该成功仍不能自动升级成哪些更强 claim。

新增段落把 future FPGA success 定义为一个 physical interface claim 的升级，而不是把当前 software result 直接贴上 hardware 标签。具体成功条件包括：

- 同一组 committed `(K,b)` banks 能够 bit-accurately 复现 source-level affine corrections；
- fast path 在 control-cycle latency budget 内具有 timing headroom；
- resource 和 power 成本相对 per-shot neural 或 posterior-branch decoder 保持小；
- adaptation 留在 slow calibration layer，latency-critical correction 仍是 deterministic physical displacement rule。

## 证据边界

本次没有新增硬件实验、没有新增 benchmark、没有改动表格数值，也没有把 future FPGA success 写成已完成事实。新增段落明确说明：即便未来 FPGA 路径成功，也不会自动证明 surface-code threshold、finite-energy logical-channel fidelity 或 broad holdout robustness。

## 预期验证

- LaTeX 编译 `CNN_FPGA_GKP_submission_draft.tex`；
- source-data audit；
- 内部项目语体扫描；
- overclaim 扫描；
- trailing whitespace 扫描。
