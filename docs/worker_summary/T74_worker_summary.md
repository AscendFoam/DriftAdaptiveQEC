# T74 Worker Summary

## 改了什么

本轮只改了 `T74` 允许路径，并完成了三类产物：

1. 新建 paper-facing 主文档：
   - `docs/paper_materials/paper_simulation_result_table_pack.md`
   - `docs/paper_materials/paper_figure_caption_pack.md`
   - `docs/paper_materials/paper_maintext_insertion_map.md`
   - `docs/paper_materials/paper_submission_material_gap_checklist.md`
2. 新建 `docs/figure_assets/T74_paper_ready_simulation_result_pack/` traceability 目录：
   - `README.md`
   - `figure_manifest.json`
   - `result_source_map.csv`
   - `caption_source_map.csv`
   - `table_snapshot.csv`
   - `submission_bundle_manifest.json`
3. 更新与补齐配套文档：
   - 更新 `docs/paper_materials/README.md`
   - 新建 `docs/review/T74_review.md`
   - 新建 `docs/for_human/T74_explanation.md`
   - 新建本文件 `docs/worker_summary/T74_worker_summary.md`

这轮没有新跑 benchmark、training、`.tflite` 或 real-board，只把 `T24/T48/T50/T57/T58/T70/T72` 的已接受证据整理成 paper-ready simulation/material pack。

## 如何验证

我执行了以下验证：

1. machine-readable 清单解析：
   - `figure_manifest.json` 解析成功，共 `15` 个 item。
   - `submission_bundle_manifest.json` 解析成功，记录了 `7` 个 table、`4` 个 figure、`4` 个 supplement note。
   - `result_source_map.csv` 与 `caption_source_map.csv` 各 `15` 行。
   - `table_snapshot.csv` 共 `28` 行。
2. 表格证据回指检查：
   - `paper_simulation_result_table_pack.md` 共 `7` 个 `T74-TBL-*` section。
   - 共 `7` 个 `直接来源` block。
   - 共 `7` 个 `快照锚点`。
3. caption / note 回指检查：
   - `paper_figure_caption_pack.md` 共 `15` 个 `T74-(FIG|TBL|SUP)-*` section。
   - 共 `15` 个 `直接证据` block。
   - 共 `4` 个 `插入说明` block。
4. stable ID 一致性检查：
   - `T74-TBL-*` 在 table pack、caption pack、insertion map、result/caption source map、table snapshot、submission bundle 与 manifest 之间全部匹配。
   - `T74-FIG-*` 与 `T74-SUP-*` 在 caption pack、insertion map、result/caption source map、submission bundle 与 manifest 之间全部匹配。
5. task package 要求的范围检查：
   - `git diff --name-only -- runs` 为空。
   - `git diff --name-only -- artifacts` 为空。
   - `git diff --name-only -- cnn_fpga physics benchmark tests` 为空。
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md` 非空，但这些是本轮开始前就已存在的 Captain 侧治理文档改动；本轮 `T74` 没有去修改这些文件。

## 剩余风险

1. `T74-FIG-01` 仍是 `partial`。
   - `T24` 数据足够，但当前没有冻结到治理文档里的 paper-facing 绘图脚本。
   - 如果近期不想补图，直接使用 `T74-TBL-01` 作为 authoritative substitute 更稳。
2. `T74-TBL-07` 与 `T74-SUP-01` 仍必须把 `FR8` 保持为 supplement-only 的 extension lane。
   - 不能把它写成 promoted mature comparator。
   - 不能把 persistent clean tie set 写成唯一阈值。
3. `T48/T72` 的边界没有因为这次打包而被升级。
   - `.tflite` 仍只是 isolated current-host true runtime。
   - real-board 仍只是 read-only gate/provenance boundary。
   - 当前已经具备 simulation/material-first 的论文装配路线，但所有 hardware-dependent surface 仍然要等待真实 host/device 条件。
