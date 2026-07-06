# 投稿稿 formal interval 入口与正式语体补强记录

**日期**：2026-07-06

## 修改对象

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 修改内容

本次只做投稿稿正文层面的有界补强：

1. 在摘要中加入已经完成的 `static_bias_theta` 12 paired repeats formal interval 结果：
   - UKF-minus-Hybrid `final_ler_mean` paired-\(t\) 95% lower bound: `0.014778`
   - bootstrap 95% lower bound: `0.014934`
   - 明确 all-scenario interval evidence 仍未完成。
2. 在 Introduction 中把较强自查/材料包语气改成正式论文语气：
   - `This manuscript separates...` 改为 `We separate...`
   - `source-data analyses` 改为 `traceability analyses`
   - `This separation is central to the manuscript...` 改为 `Maintaining this separation...`
3. 在 contributions 中补充单场景 12 paired repeats interval check 的角色，但不外推到全场景。
4. 在 Discussion 与 Conclusion 中同步单场景 formal interval 的解释边界：
   - 它增强 `static_bias_theta` scenario-level 证据；
   - 它不支持 pooled / all-scenario significance wording。

## 边界

本次不新增实验，不运行 benchmark，不改变 `submission_draft_phase_a_paired_interval_analysis.csv/json`，不补 `linear_ramp`、`step_sigma_theta` 或 `periodic_drift` 的 formal interval。硬件、`.tflite`、real-board、statcalib 与 deployment 证据等级均不改变。

## 后续验证

- 重新编译 `CNN_FPGA_GKP_submission_draft.tex`。
- 扫描内部项目进度词和过强硬件/统计措辞。
- 确认新增 interval wording 只指向 `static_bias_theta` 单场景。
