# 投稿稿 approximate-GKP affine theory 补强记录

日期：2026-07-03

## 本次修改

本次只补强 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的 Problem Setting 理论表述，不新增实验、不运行 benchmark、不升级硬件或 logical-channel 证据等级。

新增和调整内容：

1. 在 `Approximate GKP syndrome model` 中把已有 LMMSE 推导扩展为单分支局部条件均值表述。
2. 明确固定 lattice branch \(m_t\) 后的局部 residual \(r_t=e_t-\lambda m_t\)，并给出
   \[
   \hat r_t \approx \mu_r+\Sigma_{rs}\Sigma_{ss}^{-1}(s_t-\mu_s)=K(\theta_t)s_t+b(\theta_t).
   \]
3. 明确 affine fast path 不是 universal wrapped decoder，而是 slow loop 压缩近期 syndrome statistics 后提交给 fast path 的 single-branch local conditional-mean surface。
4. 把有效域失败模式整理为三类：branch ambiguity、finite-energy tails near half-lattice boundaries、drift faster than histogram window / commit cadence。
5. 将这些失败模式回连到当前投稿稿的 bounded evidence ladder：residual-MSE、\(\texttt{final\_ler}\) branch-cell crossing proxy、holdout drift stress 和 future hardware-facing datapath validation。

## 为什么需要这次补强

审稿人会要求 approximate-GKP 与 affine fast path 之间有一个正面的理论连接，而不只是“这不是完整物理模型”的防御性声明。此前稿件已经有 LMMSE 公式和有效域说明，但读者仍可能觉得 fast path 只是工程上方便的线性层。

这次补强把方法动机改写为可审查的局部命题：在单一 lattice branch、局部近似 Gaussian、慢漂移可由近期 histogram 跟踪的条件下，affine correction 是 local conditional mean 的低延迟实现。这样既能说明为什么 affine surface 合理，也能清楚说明它为什么会在 branch ambiguity、multimodal posterior 或 fast drift 下失效。

## 证据边界

- 本次不证明完整 finite-energy GKP logical channel。
- 本次不新增 logical-channel fidelity、process tomography、outer-code logical error rate 或硬件 logical-error measurement。
- 本次不把 `final_ler` 从 protocol-defined boundary-crossing proxy 升级为真实 logical error rate。
- 本次不新增 FPGA synthesis、timing、resource、power 或 source-vs-board evidence。
- 本次只改善投稿稿的理论可读性和 claim-evidence 对齐。

## 对投稿稿的作用

这次修改提高的是理论叙事的完整性：读者可以先看到 affine fast path 的正面近似命题，再看到它的失败模式和对应实验/分析 ladder。它仍然保留当前稿件的核心限制，即结果主要来自 controlled software / analytical evidence，而不是完整物理器件或真实板级验证。
