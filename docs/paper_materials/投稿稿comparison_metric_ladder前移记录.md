# 投稿稿 comparison metric ladder 前移记录

日期：2026-07-03

## 本次修改

本次只补强 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的 Introduction 与 Related Work 读者入口，不新增实验、不运行 benchmark、不改变任何结果数值。

修改内容：

1. 在 Introduction 中前移一段 comparison metric ladder，明确本文比较分三层：
   - 直接数值比较是受控漂移 benchmark 内的 protocol-defined `final_ler` ranking；
   - fidelity 相关文字只是 residual-boundary surrogate 与 channel language 的桥接，不是 finite-energy logical-channel reconstruction；
   - latency/resource 相关文字是 operation count、fixed-point software parity 和 future hardware measurement requirements 支撑的 datapath argument，不是 measured FPGA result。
2. 将贡献列表最后一项从泛泛的 validation roadmap 改成更明确的 metric and validation ladder。
3. 在 `Related Work` 的外部比较表前增加 comparison policy，明确该表是 metric-oriented positioning，不是 leaderboard-style normalized baseline。
4. 修改 `tab:external-comparison` caption，说明引用数字只是 adjacent work 的 representative reported metrics，不是本文归一化 baseline。
5. 将 narrow advantage paragraph 中的 cost claim 收紧为 controlled cost model 中的 analytical per-shot arithmetic count。
6. 在 `tab:metric-readiness` 后增加一句，说明该 ladder 是全文 cross-paper comparison 的阅读指南。

## 为什么需要这次补强

审稿人通常会把 LER、logical-channel fidelity、latency、resource 和 hardware result 放在同一个比较坐标中追问。若这些层级只在 Related Work 或 Discussion 后半段出现，读者可能在 Introduction 阶段误以为本文要同时竞争 surface-GKP overhead、logical-channel fidelity 和真实 FPGA latency。

这次补强把比较层级提前到问题陈述后，并把 Related Work 的外部指标表标为 metric-oriented positioning，使读者从一开始就知道：本文当前直接支持的是 controlled software benchmark 的 `final_ler` ranking 和 affine fast-path datapath rationale；fidelity 与 hardware 指标是外部标准和 future validation target。

## 证据边界

- 本次不新增 LER、fidelity、latency、resource 或 hardware measurement。
- 本次不把 `final_ler` proxy 升级为 logical-channel fidelity、hardware logical error rate 或 significance result。
- 本次不把 analytical operation count 或 Q4.20 software parity 写成 FPGA timing/resource/power result。
- 本次不改变 `docs/04_task_board.md`、`docs/07_handoff.md` 或任何 00-08 治理状态。

## 对投稿稿的作用

这次修改提高的是读者路径和审稿防御能力：Introduction 更早给出本文的比较层级，Related Work 的外部指标表明确不是 normalized leaderboard，Discussion 的 supported-advantage table 就不再显得像后置免责声明，而是从一开始服务同一条 argument chain。
