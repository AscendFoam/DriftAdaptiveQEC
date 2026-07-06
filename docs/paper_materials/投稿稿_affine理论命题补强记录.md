# 投稿稿 affine 理论命题补强记录

日期：2026-07-03

服务稿件：

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 本轮补强

本轮在 `Approximate GKP syndrome model` 附近加入 `Local affine fast path`
命题，将 fast-path affine rule 的理论依据写成可审查的局部最优线性估计结论。

命题内容：

- 固定 effective noise state 和 lattice branch 后，若局部 residual 与 wrapped
  syndrome 具有有限二阶矩且 \(\Sigma_{ss}\) 非奇异，则所有 affine correction
  \(a(s)=As+c\) 中的最小均方误差解为
  \(A^\star=\Sigma_{rs}\Sigma_{ss}^{-1}\)，
  \(c^\star=\mu_r-A^\star\mu_s\)。
- 若该局部分支内 \((r,s)\) 近似 joint Gaussian，则 affine rule 同时是
  conditional mean。

## 可写边界

- 可以写：affine fast path 是固定分支、固定 effective state 下的局部
  linear-MMSE / conditional-mean fast-path contract。
- 可以写：slow loop 的任务是更新或近似这些局部矩，从而把复杂估计移出
  latency-critical path。
- 可以写：这解释了为什么同一 committed \((K,b)\) interface 可以容纳
  teacher branch、statistical calibration 或未来 estimator。

## 不可外推边界

- 不能写成 global wrapped-posterior optimality。
- 不能写成 finite-energy logical-channel fidelity proof。
- 不能写成 real-board FPGA timing/resource/source-vs-board evidence。
- 不能写成 teacher-parameter necessity 或 universal SOTA decoder claim。

## 验证

本轮应以 LaTeX 编译、禁词扫描、source-data audit 和 symbol-boundary audit
作为收口验证。该记录本身不新增 benchmark，不新增实验数据，不改变硬件占位边界。
