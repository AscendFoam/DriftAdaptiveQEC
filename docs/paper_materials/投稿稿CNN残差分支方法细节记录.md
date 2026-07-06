# 投稿稿 CNN 残差分支方法细节记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。目标是把审稿人会追问的 CNN / teacher residual branch 细节从代码和 preserved artifacts 中抽取出来，补到方法章，而不是让该分支看起来像黑箱。

## 新增文件

- `docs/paper_materials/submission_draft_cnn_branch_method_details.csv`
- `docs/paper_materials/submission_draft_cnn_branch_method_details.json`

## 抽取事实

- 输入：5 个 `32 x 32` histogram 窗口、4 个 histogram delta planes、teacher prediction、teacher runtime bias 和 teacher temporal deltas，共 21 个空间通道。
- 标签：`b_q` / `b_p` 两个 residual-b 输出，语义是 target runtime bias 减去 teacher runtime bias。
- 数据：四个 drift scenarios，共 2048 个 runtime windows；train / val / test = 1638 / 204 / 206。
- 模型：Tiny-CNN，单层 `3 x 3` convolution，16 channels，ReLU，`2 x 2` average pooling，96 hidden units，2 outputs。
- 训练：weighted MSE，`b_q` / `b_p` label weight 均为 2；learning rate 0.002，weight decay 0.0001，batch size 64，最多 36 epochs，patience 8，按 validation loss 选模型。
- 泄漏边界：test split 没有进入拟合，但四个 scenario 在 split 间共享；这支持当前 split discipline，但不等于 unseen-drift generalization。

## 不可外推边界

- 本记录不重跑训练，不证明 full reproducibility。
- 本记录不补 holdout drift、不补 CI / p-value、不证明 causal mechanism closure。
- 本记录不把 CNN 分支写成 per-shot neural decoder；它仍是 slow-loop residual calibrator。
