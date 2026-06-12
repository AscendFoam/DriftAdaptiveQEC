# T75 Worker Summary

## 改了什么

本轮只改了 `T75` 任务包允许的路径，并完成了三组交付：

1. 新建 paper authoring 文档：
   - `docs/paper_materials/paper_maintext_results_authoring_pack.md`
   - `docs/paper_materials/paper_caption_lock_and_placement_notes.md`
   - `docs/paper_materials/paper_appendix_bridge_pack.md`
   - `docs/paper_materials/paper_authoring_do_not_write_list.md`
2. 新建 `docs/figure_assets/T75_maintext_results_authoring_pack/` 资产目录：
   - `README.md`
   - `authoring_manifest.json`
   - `asset_source_map.csv`
   - `t75_fig_m01_t24_frozen_summary.svg`
   - `t75_fig_m02_fr6_multi_seed_mechanism.svg`
   - `t75_fig_a01_boundary_schematic.svg`
3. 更新与补齐配套文档：
   - 更新 `docs/paper_materials/README.md`
   - 新建 `docs/review/T75_review.md`
   - 新建 `docs/for_human/T75_explanation.md`
   - 新建本文件 `docs/worker_summary/T75_worker_summary.md`

这轮没有运行任何 benchmark、training、`.tflite`、real-board 或其他长实验；全部工作都限定在 `T74` 已冻结结果之上的 paper-facing authoring 收口。

## 如何验证

我实际执行了以下验证：

1. `authoring_manifest.json` 解析与映射检查：
   - 解析成功；
   - manifest 资产 ID 为 `T75-FIG-M01`、`T75-FIG-M02`、`T75-FIG-A01`；
   - 每个资产都映射到了至少一个上游 `T74-*` stable ID。
2. `asset_source_map.csv` 一致性检查：
   - `t75_asset_id` 的唯一集合与 manifest 完全一致。
3. SVG 结构检查：
   - 三张 `.svg` 文件都真实存在；
   - 文件内容均包含 `<svg`；
   - 三张图都能被 XML 解析成功。
4. 主文 authoring 段落检查：
   - `paper_maintext_results_authoring_pack.md` 的段落 A/B/C 都包含至少一个上游 `T74-*` stable ID；
   - 三段都显式写出了 `可写表述` 和 `不可写表述`。
5. 禁写清单覆盖检查：
   - `paper_authoring_do_not_write_list.md` 已覆盖 `T48`、`T72`、`FR8`、`T74-FIG-04`。
6. 范围检查：
   - `git diff --name-only -- runs` 为空；
   - `git diff --name-only -- artifacts` 为空；
   - `git diff --name-only -- cnn_fpga physics benchmark tests` 为空；
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md` 非空，但这些治理文档改动在本轮开始前就已存在；本轮 `T75` 没有去修改它们。

## 剩余风险

1. `T75-FIG-M01` 虽然已经可用于主文，但它依然只是 `T74-TBL-01` 的可视化压缩。
   - 如果后续期刊样式或审稿偏好更适合主表，应该直接退回 `T74-TBL-01`，不要把 `T75-FIG-M01` 当成更强证据。
2. `T75-FIG-M02` 仍只支持 descriptive mechanism/intervention reading。
   - 不能把它写成 causal closure、intervention success proof 或 teacher necessity proof。
3. `T75-FIG-A01` 的价值在于解释边界，而不是关闭边界。
   - `T48` 仍只是 isolated current-host true runtime；
   - `T49/T71/T72` 仍只是 read-only real-board gate/provenance with current-host `NO_GO`；
   - `T74-FIG-04` 仍必须保持 `blocked`。
