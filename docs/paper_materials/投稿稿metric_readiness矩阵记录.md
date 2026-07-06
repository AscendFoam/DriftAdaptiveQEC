# 投稿稿 metric-readiness 矩阵记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。目标是把审稿人最关心的 LER、fidelity / infidelity、drift robustness、latency / resource 和 hardware validation 指标逐项拆开，说明当前稿件可以写什么、不能写什么、还缺什么材料。

## 新增文件

- `docs/paper_materials/submission_draft_metric_readiness_matrix.csv`
- `docs/paper_materials/submission_draft_metric_readiness_matrix.json`

## 写入稿件的核心口径

- `final_ler_mean`、paired descriptive deltas 和 non-inferential paired-bootstrap envelope 是当前主结果可写指标。
- logical-channel fidelity / process infidelity 是外部 GKP 文献中的 channel-level 标准；当前稿件没有估计，不能用 `1-final_ler` 替代。
- drift adaptation robustness 只在四个预声明 drift scenarios 和受控 local-Gaussian checks 内成立。
- fast-path latency / cost 当前只能写 analytical operation count、Q4.20 software fixed-point parity 和 low-complexity motivation。
- hardware-facing validation 仍是未来测量面，不能写成真实 FPGA timing/resource result。

## 不可外推边界

- 本记录不新增实验、不重跑 benchmark、不补 inferential CI / p-value / holdout。
- 本记录只新增 residual-boundary Pauli-event surrogate 的就绪度说明；它不估计 finite-energy logical-channel fidelity、process fidelity 或 channel tomography。
- 本记录不补硬件测量、不补 `.tflite` deployment closure、不改变 real-board 边界。
- fixed-point parity 只代表软件 emulation 数值一致性，不代表 source-vs-board agreement、timing closure、resource/power 或 board latency。
- 本记录只服务投稿稿 metric framing 和 source-data traceability。

## 2026-07-03 logical-channel surrogate 增量

- `Logical-channel fidelity or infidelity` 行已从 `not estimated` 改为 `residual-boundary Pauli-event surrogate only`。
- 对应 source data 为 `submission_draft_logical_channel_surrogate_analysis.csv/json`。
- 投稿稿主文只报告 `p_any`，即至少一个 q/p half-lattice residual crossing 的 union rate。
- `F_avg=(1+2p_I)/3` 只作为 source data 中的 identity-derived surrogate 字段，不作为主文性能指标，不与外部 finite-energy GKP channel-fidelity 或 infidelity 结果直接比较。

## 2026-07-03 holdout drift stress 增量

- `Drift adaptation robustness` 行已加入 `holdout drift stress tests`。
- 对应 source data 为 `submission_draft_holdout_drift_stress_analysis.csv/json`。
- 投稿稿主文报告 residual MSE，因为本组 controlled samples 中 affine 方法的 half-lattice crossing proxy 较稀疏，MSE 对漂移适应空间更敏感。
- `lagged_affine` 只代表 slow-commit known-state pressure reference，不代表 CNN residual branch。
- 该分析只能写成 controlled non-hardware stress diagnostic；不得写成正式 expanded benchmark、trained-branch holdout generalization、CI-backed robustness 或硬件结果。

## 2026-07-03 paired uncertainty 增量

- `Logical-error proxy` 行已加入 `non-inferential paired-bootstrap envelope`。
- 对应 source data 为 `submission_draft_paired_uncertainty_analysis.csv/json`。
- 投稿稿只把该 envelope 写成 `n=2` repeat-level descriptive resampling summary，用于透明展示 UKF-minus-Hybrid paired deltas 的方向和幅度。
- 该 envelope 不是 confidence interval、standard error、p-value、significance test 或 robustness proof；更强统计结论仍需要更多 repeats 和预声明 inferential protocol。
