# T73 Worker Summary

## 改了什么

1. 刷新了 `docs/paper_materials/paper_claim_evidence_ledger.md`，把主线 claim/evidence 口径更新到 post-`T72`：
   - 显式吸收 `T48/T50/T70/T72`
   - 把 isolated `.tflite` runtime、training/material pack、real-board gate/provenance pack、`statcalib` extension-lane closure 分别收入口径
2. 新增了 `docs/paper_materials/paper_result_figure_ledger.md`：
   - 给当前 paper-facing 图、表、result-pack 建立 `ready / partial / blocked` 台账
   - 每项都绑定到具体 run / review / asset / evidence pack
3. 新增了 `docs/paper_materials/paper_claim_risk_table.md`：
   - 将 paper-facing 风险绑定到 `R*` 或 review warning
   - 显式写入 `R31` closed by `T72` 与 `R32` open
4. 更新了 `docs/paper_materials/README.md` 和 `docs/paper_materials/paper_ablation_result_pack.md`：
   - README 改成 post-`T72` 的材料入口
   - `FR8` 从 pre-`T70` 的 `missing` 同步为 `partial` 的 extension-lane closure/no-promotion material

## 如何验证

已执行：

1. `git diff --name-only -- runs`
   - 空
2. `git diff --name-only -- artifacts`
   - 空
3. `git diff --name-only -- cnn_fpga physics benchmark tests`
   - 空
4. `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`
   - 空
5. `rg -n '^[|] `[^`]+` [|]' docs/paper_materials/paper_claim_evidence_ledger.md`
   - 自检 `C1`-`C12` 各行都带显式证据路径
6. `rg -n '^[|] `[^`]+` [|]' docs/paper_materials/paper_result_figure_ledger.md`
   - 自检 result/figure rows 都带显式来源列
7. `rg -n '^[|] `PR[0-9]+` [|]' docs/paper_materials/paper_claim_risk_table.md`
   - 自检 `PR1`-`PR10` 各行都带 `R*` 或 review warning

## 剩余风险

1. `R32` 仍开放：future-host 最小 config 场景下的 path provenance 标签还不够精确。
2. `R33` 仍开放：当前没有 `Linux + FPGA` 硬件宿主，real-board execution 仍不是近期主线。
3. `R24` 仍开放：`FR8` 只能作为 separately labeled extension-lane material，不能写成 promoted comparator。
4. `ready / partial` 是 paper-material readiness，不是 claim 自动升级。
