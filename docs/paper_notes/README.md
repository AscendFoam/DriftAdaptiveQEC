# Paper Notes

## 2026-07-03 submission-draft design-rationale update

- `CNN_FPGA_GKP_submission_draft.tex`: added a manuscript-facing
  `Design rationale and testable predictions` method subsection. The section
  explains four reviewer-facing predictions behind the affine runtime
  calibration design: drift adaptation through a committed `(K,b)` surface,
  low online cost from keeping estimation outside the per-shot path,
  source-vs-board checkability for a future FPGA implementation, and explicit
  failure modes when the single-branch local-affine assumption breaks.
- Boundary: this is a prose and argument-structure improvement only. It does
  not upgrade beyond the current `12/12` formal Phase A source-data status,
  does not add inferential confidence intervals or p-values, and does not
  supply hardware timing, resource, power, or source-vs-board evidence.
- 2026-07-06 later update: added one formal-length
  `static_bias_theta` paired repeat at
  `runs/paper_submission_phase_a/formal_probe_static_bias_theta_ukf_hybrid_r12_11_12_20260706`.
  The cumulative formal Phase A source-data status is now `12/12` for
  `static_bias_theta`, repeat indices `0,1,2,3,4,5,6,7,8,9,10,11`, with mean UKF-minus-Hybrid
  delta `0.015563263889` and `12/12` positive pairs. This completes one
  scenario-row source-data set, but remains non-upgraded until paired-interval
  analysis and the predeclared broader gate pass.
- 2026-07-06 later update: added `linear_ramp` formal-length paired repeats
  `9`、`10` 和 `11` at
  `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_09_10_20260706`,
  `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_10_11_20260706`
  and
  `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_11_12_20260706`.
  `linear_ramp` now has `12/12` formal scenario-row source data, mean
  UKF-minus-Hybrid delta `0.022416805556`, paired-\(t\) 95% interval
  `[0.021214967006, 0.023618644106]` and bootstrap 95% interval
  `[0.021404645833, 0.023439867477]`. This completes the second scenario-row
  interval check, but the all-scenario, pooled, holdout and hardware gates
  remain incomplete.
- 2026-07-06 note sync update: `CNN_FPGA_GKP_theory_note_draft.tex` has been
  compactly synchronized from `CNN_FPGA_GKP_submission_draft.tex` for selected
  theory and result material: the normalized \(\lambda=\sqrt{2\pi}\) residual
  coordinate caveat, branch-conditioned local-affine interpretation,
  software-only fast-path operation-count / Q4.20 parity diagnostics, the two
  completed Phase A scenario-level paired-interval rows, and controlled
  oracle/holdout diagnostic wording.  The synchronized note compiles to 22
  pages in a temporary verification build, below the 25-page hard limit.  This
  update does not add all-scenario interval evidence, holdout robustness,
  inferential \(p\)-values, FPGA synthesis/timing/resource measurements,
  source-vs-board agreement or real-board validation.
- Verification expected for this update: LaTeX compile of the submission
  draft, source-data audit, and project-progress wording scan.

本目录保存主线论文 `note`、LaTeX 草稿及其保留的编译产物。它是写作素材入口，不是项目完成态证据入口。

## 当前文件组

- `CNN_FPGA_GKP_theory_note_draft.tex`：当前主线 note 源文件。
- `CNN_FPGA_GKP_theory_note_draft.pdf`：对应编译产物。
- `CNN_FPGA_GKP_theory_note_draft.*`：保留的 LaTeX 辅助文件、日志与同步文件。
- 2026-07-06 update: `CNN_FPGA_GKP_submission_draft.tex` 已补充 \(\lambda=\sqrt{2\pi}\) 作为软件 residual coordinate 的解释，并在 FPGA-facing datapath contract 中新增 register-level fast-path 叙事；该修改不新增 finite-energy device calibration、synthesis、timing、resource、power 或板级证据。
- 2026-07-06 update: `CNN_FPGA_GKP_submission_draft.tex` 的 Discussion 已新增外部指标标尺解释，把 calibration-conditioned / learned decoder 的 LER 与 microsecond-scale latency、real-time hardware decoder 的 ns/sub-us timing、resource、area、power 和 cycle budget 统一读作 future validation standard；该修改不把外部数值写成本稿 baseline、结果或 normalized leaderboard。
- 2026-07-06 update: `CNN_FPGA_GKP_submission_draft.tex` 在 `\appendix` 后新增 Appendix overview，明确主文到 Outlook 结束，appendix 只承担 source-data routing、validation-scope tables、reproducibility limits 与 terminology controls；该修改不新增实验、不改数值、不升级统计、硬件、finite-energy fidelity 或 benchmark 证据。
- 2026-07-06 update: `CNN_FPGA_GKP_submission_draft.tex` 的 `tab:phase-a-upgrade-gate` 已同步 Phase A upgrade-gate source CSV，把 short-run repeat rehearsal 修正为当前 collector output 中 `0` 个 complete short-run scenario row；当前 formal interval 口径已随 `linear_ramp` completion 更新为 `static_bias_theta` 与 `linear_ramp` two completed scenario rows。该修改不新增 p-value、pooled/all-scenario gate、holdout 或 hardware 证据。
- 2026-07-06 update: Phase A 新增 `linear_ramp` formal-length repeat `0`--`11` paired runs；`linear_ramp` 当前进入 `submission_draft_phase_a_repeat_summary.csv/json` 和 `submission_draft_phase_a_paired_interval_analysis.csv/json`，作为第二个 `12/12` completed scenario-row positive interval check；该状态仍不构成 all-scenario gate、pooled analysis、holdout robustness、hardware 或 deployment 证据。
- 2026-07-06 update: `CNN_FPGA_GKP_submission_draft.tex` 的 bibliography 入口已改为根目录编译可解析的 `docs/paper_notes/CNN_FPGA_GKP_submission_refs`，以避免 LaTeX 从仓库根目录构建时依赖旧 `.bbl`；该修改只修复 BibTeX 编译契约，不新增引用、不改引用内容、不完成最终期刊参考文献格式。
- 2026-07-06 update: `CNN_FPGA_GKP_submission_draft.tex` 已清理正文中残留的 reviewer-facing 元语言，把“reviewer question / reviewer failure mode”改成 claim hierarchy、scientific questions 与 local technical question；该修改不新增实验、不改数值、不升级统计、硬件、finite-energy fidelity 或 benchmark 证据。
- 2026-07-06 update: `CNN_FPGA_GKP_submission_draft.tex` 的 Discussion 已补充 future FPGA success 叙事，把未来真板成功限定为 measured real-time calibration interface；该修改不新增硬件实验、不升级 surface-code threshold、finite-energy fidelity、holdout robustness 或 deployment 证据。
- 2026-07-06 update: `CNN_FPGA_GKP_submission_draft.tex` 的 Results 与 appendix 已将内部验证语体转为 validation requirements、short-run execution-coverage rehearsal 与 validation thresholds；该修改不新增实验、不改数值、不升级统计、硬件或 finite-energy fidelity 证据。
- `CNN_FPGA_GKP_submission_draft.tex`：2026-07-02 新建、2026-07-03 完成正式论文语体、比较指标、外部 metric 参照、metric comparability protocol、优势边界、literature metric crosswalk 登记、closest-work positioning 最近邻工作定位表、Discussion 比较优势补强、Discussion rival-explanations 约束段落、GKP boundary-sensitivity analytical bridge、controlled nearest-syndrome / oracle-affine / wrapped-Gaussian one-step sanity check、residual-boundary Pauli-event surrogate、surrogate average-fidelity readout、lattice logical-channel sanity summary、finite-squeezing toy-channel sanity check、nearest-syndrome / oracle / wrapped sequence-controlled baseline、controlled holdout drift stress diagnostic、affine local-validity 派生诊断、fast-path analytical cost model、Q4.20 fixed-point software parity、software runtime-discipline counters、validation-contract 图件汇总、non-inferential paired-bootstrap envelope、LER advantage-margin descriptive readout、all-scenario runner smoke matrix、内部审稿/项目 QA 语气清理、审稿意见驱动的 Introduction / metric ladder / approximate-GKP boundary / FPGA verification target / Outlook 强化、主文 Outlook 前移和 appendix 支撑材料降层、FPGA-facing datapath contract 方法叙事补强、Method algorithmic protocol 可复现叙述补强、approximate-GKP affine single-branch local-validity 理论表述补强、comparison metric ladder 前移、审稿残留元语言清理、项目状态式 closeout/closure 语体清理、fast-path / software-validation 过强语气降噪、source-data coverage 口径同步、实时/FPGA decoder 证据标准与当前 datapath contract 支撑层级补强、metric triangulation 证据链读法补强，以及 Results 开头 reviewer-facing evidence hierarchy 补强的投稿论文 TeX 初稿；基于 `审稿人视角_note成稿差距与材料清单.md` 与当前 note 重构，硬件结果仍为 future board-level measurements，不覆盖原 note。
- `CNN_FPGA_GKP_submission_refs.bib`：2026-07-02 为投稿论文 TeX 初稿生成的 active-key BibTeX 文件；来自 `docs/paper_materials/zotero_export_for_literature_review.bib` 的本地 Zotero 快照，只解决当前 27 个 active citation key 的覆盖与可编译引用，不等于最终期刊参考文献格式。
- `ppt_output/CNN_FPGA_GKP_submission_draft_two_slide_cn_v2_no_font_garble.pptx`：2026-07-03 基于 `CNN_FPGA_GKP_submission_draft.tex` 重写的两页中文 PPT 汇报素材；主体文字已栅格化为整页 PNG，以避免 PowerPoint 字体替换导致中文乱码或问号。第一页概括解码延迟/效率、LER、保真度和噪声适应优势，第二页说明 FPGA-facing 实时纠错闭环与工程实现边界。配套 `ppt_output/qa_report.md`、`ppt_output/asset_manifest.md`、`ppt_output/slide1_v2_no_font_garble.png` 与 `ppt_output/slide2_v2_no_font_garble.png` 记录回读检查和可视预览；该 PPT 只服务汇报表达，不升级 hardware、`.tflite`、benchmark、fidelity 或 deployment 证据等级。
- `../paper_materials/submission_draft_source_data_manifest.csv` 与 `../paper_materials/submission_draft_source_data_manifest.json`：2026-07-06 为投稿稿补充的 manuscript-facing source-data file-level hash manifest；当前行数以最新 `build_submission_draft_source_manifest.py` 输出为准，覆盖 source CSV/JSON、figure manifest/source-map、Phase A source-data 文件和生成/审计脚本，不等于 historical `runs/` 递归 hash closure、full reproducibility 或 hardware provenance。
- `../paper_materials/submission_draft_literature_metric_crosswalk.csv` 与 `../paper_materials/submission_draft_literature_metric_crosswalk.json`：2026-07-03 为投稿稿外部比较表补充的 literature metric anchor 交叉表，含 local literature-card source-anchor 与 final pinning follow-up；只服务 related-work positioning，不等于 normalized leaderboard、本稿实验源数据或硬件证据。
- `../paper_materials/submission_draft_benchmark_expansion_protocol.csv` 与 `../paper_materials/submission_draft_benchmark_expansion_protocol.json`：2026-07-03 为投稿稿统计/复现边界补充的 benchmark expansion protocol；只记录 repeat budget、holdout family 和 provenance-reporting plan，不等于已执行 expanded benchmark、CI、p-value、robustness proof 或硬件证据。
- `../paper_materials/submission_draft_runner_smoke_pair.csv` 与 `../paper_materials/submission_draft_runner_smoke_pair.json`：2026-07-03 为投稿稿 repeat-expanded benchmark 规划补充的 runner feasibility 材料；只记录单场景单重复 smoke pair，不等于主文性能证据、expanded benchmark、CI/p-value、holdout robustness 或硬件证据。
- `../paper_materials/submission_draft_runner_smoke_matrix.csv` 与 `../paper_materials/submission_draft_runner_smoke_matrix.json`：2026-07-03 为投稿稿 repeat-expanded benchmark 规划补充的全场景 runner feasibility 材料；只记录 short-run timing 下四个预声明场景、两个 paired repeats、UKF 与 Hybrid Residual-B 的执行覆盖，不等于主文性能证据、expanded benchmark、CI/p-value、holdout robustness 或硬件证据。
- `../paper_materials/submission_draft_phase_a_repeat_plan.csv` 与 `../paper_materials/submission_draft_phase_a_repeat_plan.json`：2026-07-03 为投稿稿补充的 Phase A repeat-expanded execution plan；只记录 formal-length 分块命令、smoke feasibility 命令、post-run artifact requirements 和不可外推边界，不等于已执行 expanded benchmark、CI/p-value、holdout robustness 或硬件证据。
- `../paper_materials/submission_draft_phase_a_repeat_summary.csv` 与 `../paper_materials/submission_draft_phase_a_repeat_summary.json`：2026-07-06 为投稿稿补充的 Phase A completed-run summary 汇总；当前 `static_bias_theta` 与 `linear_ramp` 已完成 formal cumulative `12/12` scenario-row source data；summary 仍不提供 all-scenario repeat-expanded evidence、CI/p-value、holdout robustness 或硬件证据。
- `../paper_materials/submission_draft_phase_a_paired_interval_analysis.csv` 与 `../paper_materials/submission_draft_phase_a_paired_interval_analysis.json`：2026-07-06 为投稿稿补充的 formal paired-interval source data；当前支持 `static_bias_theta` 与 `linear_ramp` 的 scenario-level positive interval checks，不等于 all-scenario repeat-expanded advantage、p-value、holdout robustness 或硬件证据。
- `../paper_materials/submission_draft_phase_a_upgrade_gate.csv` 与 `../paper_materials/submission_draft_phase_a_upgrade_gate.json`：2026-07-03 为投稿稿补充的 Phase A validation-threshold gate；只规定当前描述性结果、short-run rehearsal、formal repeat expansion、holdout drift 和 hardware measurements 的 permitted statement / validation requirement / unsupported inference，不运行 benchmark、不计算 CI/p-value、不补硬件证据。
- `../paper_materials/paper_note_results_sync_manifest.md`：`T77` 结果层同步 manifest。
- `../paper_materials/paper_note_alignment_and_layout_closeout.md`：`T78` 非结果层校准、`statcalib` 层级降权与排版收口记录。
- `../paper_materials/paper_bounded_prose_reopen_manifest.md`：`T80` 的 ready-section bounded prose reopen manifest。
- `../paper_materials/paper_methods_and_contribution_calibration_manifest.md`：`T81` 的 contribution/methods calibration manifest。
- `../paper_materials/paper_supporting_material_closeout_pack.md`：`T82` 的 supporting-material closeout pack。
- `../paper_materials/paper_manuscript_closeout_readiness_matrix.md`：`T82` 的 manuscript-facing readiness matrix。
- `../paper_materials/paper_fullnote_consistency_crosswalk.md`：`T83` 的全文 section-to-evidence consistency crosswalk。
- `../paper_materials/paper_closeout_gate_and_blocker_register.md`：`T83` 的 closeout gate 与 blocker register。
- `../paper_materials/paper_bounded_final_polish_change_map.md`：`T84` 的 reader-facing final polish 改动台账。
- `../paper_materials/paper_reader_facing_term_translation_table.md`：`T84` 的内部术语到读者化表述翻译表。
- `../paper_materials/paper_appendix_supplement_reader_assembly_map.md`：`T84` 的 main text / appendix / supplement / blocked 读者化装配表。
- `../paper_materials/paper_submission_readiness_preflight_gate.md`：`T85` 的 submission-readiness preflight gate。
- `../paper_materials/paper_submission_blocker_matrix.md`：`T85` 的 submission-facing blocker matrix。
- `../paper_materials/paper_residual_state_lag_sweep.md`：`T85` 的 residual wording/state-lag 清扫台账。
- `../paper_materials/paper_submission_pack_assembly_manifest.md`：`T86` 的 submission-facing assembly manifest。
- `../paper_materials/paper_submission_surface_route_map.md`：`T86` 的 main text / appendix / supplement / exclusion route map。
- `../paper_materials/paper_submission_exclusion_register.md`：`T86` 的显式 exclusion register。
- `../paper_materials/paper_submission_author_handoff.md`：`T86` 的作者 handoff 与禁写边界汇总。
- `../paper_materials/paper_author_final_qa_checklist.md`：`T87` 的作者终检 QA checklist。
- `../paper_materials/paper_presubmission_regression_gate.md`：`T87` 的 pre-submission regression gate。
- `../paper_materials/paper_submission_wording_redflag_register.md`：`T87` 的危险表述 red-flag register。
- `../paper_materials/paper_manual_finish_queue.md`：`T87` 的 bounded manual finish queue。
- `../paper_materials/paper_manual_finish_execution_log.md`：`T88` 的 manual-finish 执行日志。
- `../paper_materials/paper_mainline_surface_freeze_manifest.md`：`T88` 的主线 surface freeze manifest。
- `../paper_materials/paper_author_edit_decision_register.md`：`T88` 的作者编辑决策台账。
- `../paper_materials/paper_blocked_surface_disclaimer_table.md`：`T88` 的 blocked surface disclaimer table。
- `../paper_materials/paper_frozen_mainline_handoff_gate.md`：`T88` 的 frozen-mainline handoff gate。
- `../paper_materials/paper_frozen_mainline_handoff_packet.md`：`T89` 的 frozen-mainline handoff 单一入口包。
- `../paper_materials/paper_frozen_mainline_source_of_truth_map.md`：`T89` 的 frozen/mainline/blocked source-of-truth map。
- `../paper_materials/paper_postfreeze_change_control.md`：`T89` 的 post-freeze 变更分级与控制规则。
- `../paper_materials/paper_blocked_surface_reentry_conditions.md`：`T89` 的 blocked surface 重开条件表。

## 使用规则

1. 本目录中的 note 可以作为论文写作素材，不作为当前项目完成态证据。
1a. `CNN_FPGA_GKP_submission_draft.tex` 是独立投稿稿草案，不替代 frozen mainline note，不升级任何 benchmark、`.tflite`、real-board、training reproducibility 或 `statcalib` 证据等级；其中硬件相关表格和图位仍是 future board-level measurements，closest-work positioning 表格只是一项 literature-family 定位与边界对照、不是 normalized leaderboard，GKP boundary-sensitivity 表格只是一项 Gaussian residual analytical bridge，controlled nearest-syndrome / oracle-affine / wrapped-Gaussian 表格只是一项 one-step local-Gaussian sanity check，nearest-syndrome 只是 direct hard-correction sanity reference、不是 tuned finite-energy nearest-lattice decoder，residual-boundary Pauli-event surrogate、surrogate average-fidelity readout 与 lattice logical-channel sanity summary 只是一项 q/p half-lattice crossing event decomposition 及方法级聚合，finite-squeezing toy-channel sanity check 只是一项 simplified noisy wrapped-syndrome measurement-channel consistency check，不是 finite-energy logical-channel fidelity 或 process fidelity，sequence-controlled baseline 表格只是一项 controlled local-Gaussian short-sequence analysis，holdout drift stress 表格只是一项 controlled non-hardware stress diagnostic，affine local-validity 表格只是一项由既有 controlled CSV 派生的 oracle-MSE headroom / branch-risk / stale-commit-risk 读法，其中 surrogate average fidelity 仍然只是 residual-boundary Pauli-style surrogate，fast-path cost 表格只是一项 analytical operation-count model，fixed-point parity 表格只是一项 software-emulation numerical check，runtime-discipline counters 只是一项 software-in-the-loop protocol observability summary，runner smoke matrix 只是一项 all-scenario short-run runner feasibility check，paired-bootstrap envelope 只是一项 `n=2` descriptive resampling summary，LER advantage-margin / `delta/max SD` 只是一项 source-data descriptive scale readout，不是 finite-energy logical-channel fidelity、process tomography、inferential CI、standard error、p-value、significance test、expanded benchmark、hardware reliability、board latency 或 robustness proof。
1b. `CNN_FPGA_GKP_submission_refs.bib` 只服务当前投稿稿草案的引用控制；正式投稿前仍需按目标期刊模板、`.bbl` 要求和 preprint/venue 最终版本逐项核验。
1b-1. 仓库根目录编译 `CNN_FPGA_GKP_submission_draft.tex` 时，TeX 内部使用 `docs/paper_notes/CNN_FPGA_GKP_submission_refs` 作为 BibTeX database 路径；不要把这一点解读为引用内容或期刊格式已经终检。
1c. `submission_draft_source_data_manifest.csv/json` 只服务当前投稿稿 source-data / provenance 可审查性；它不补 row-level historical run hash closure、runner-version closure、training repeated-run closure、硬件 bitstream/DMA/MMIO/latency/resource 证据或统计推断。
1d. `submission_draft_benchmark_expansion_protocol.csv/json` 只服务当前投稿稿的下一步强统计规划；它不运行 benchmark，不补 inferential interval / p-value，不证明 holdout robustness，也不补 FPGA timing/resource/source-vs-board evidence。
1e. `submission_draft_runner_smoke_pair.csv/json` 只服务 runner feasibility 与后续 repeat-expanded benchmark 成本估算；它不进入主文性能 claim，不补强统计或硬件证据。
1f. `submission_draft_runner_smoke_matrix.csv/json` 只服务 all-scenario runner feasibility、缺失行检查和后续 repeat-expanded benchmark 成本估算；它使用 short-run timing，不进入主文性能 claim，不补 expanded benchmark、CI/p-value、holdout robustness 或硬件证据。
1g. `submission_draft_phase_a_repeat_plan.csv/json` 和 `submission_draft_phase_a_repeat_summary.csv/json` 只服务 Phase A repeat-expanded benchmark 的执行计划、run discovery 和 source-data 汇总；plan 不运行 benchmark，summary 即使包含 short-run rows 也不进入主文性能 claim，不补 inferential interval、p-value、holdout robustness、formal reproducibility 或硬件证据。
1g-1. `linear_ramp` 当前已有 formal-length repeat `0`--`11` 的 `12/12` completed scenario-row source data；它可以用于 `linear_ramp` scenario-level paired interval wording，但不能用于 all-scenario gate、pooled analysis、holdout robustness 或 hardware wording。
1h. `submission_draft_phase_a_paired_interval_analysis.csv/json` 只服务已经完成的 `static_bias_theta` 与 `linear_ramp` formal-length scenario-row paired interval checks；它不运行 benchmark、不计算 p-value、不证明 all-scenario repeat-expanded advantage、holdout robustness 或硬件证据。
1i. `submission_draft_phase_a_upgrade_gate.csv/json` 只服务投稿稿 claim wording 边界；它不运行 benchmark、不计算新的 inferential interval 或 p-value、不证明 holdout robustness，也不补 FPGA timing/resource/source-vs-board evidence。
2. 若 note 文本涉及 benchmark、`.tflite`、HIL、real-board、`statcalib` 或投稿完成态，必须先与 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 的当前边界对齐。
3. 重新编译后产生的 `.aux/.log/.fls/.fdb_latexmk/.synctex.gz/.toc/.out/.pdf` 若需保留，应继续放在本目录。
4. `T77` 之后，结果层同步优先查看 `paper_note_results_sync_manifest.md` 与源码中的 `% T77-SOURCE: ...` 注释；未被该链路覆盖的 section，不应默认视为“结果层已同步”。
5. `T78` 之后，标题、引言、`Relationship to Existing Work`、讨论、结论以及 note 内部 `statcalib` 层级的进一步校准，优先查看 `paper_note_alignment_and_layout_closeout.md` 与 `% T78-SCOPE: ...` 注释；这仍然只是 note 质量收口，不是证据升级。
6. `T80` 之后，若需判断当前 note 的 mainline prose 是否已经过 bounded reopen，应优先查看 `paper_bounded_prose_reopen_manifest.md` 与 `% T80-REOPEN: ...` 注释；该链路只覆盖 `Title`、`Abstract`、`Introduction`、`Relationship to Existing Work`、`Experimental Setup`、`Numerical Results`、`Discussion`、`Conclusion` 八个 ready sections。
7. `T81` 之后，若需判断 `Summary of Contributions` 与三章 methods 是否已经校准到当前 strongest supported truth，应优先查看 `paper_methods_and_contribution_calibration_manifest.md` 与 `% T81-CALIBRATION: ...` 注释；该链路只覆盖 `Summary of Contributions`、`Brief Review of the GKP Code`、`Noise and Drift Model`、`Model Architecture` 四个 target sections，不代表 full-manuscript reopen。
8. `T82` 之后，若需判断 supporting-boundary 段落是否已经按 `main text / appendix / supplement / blocked` 四层收口，应优先查看 `paper_supporting_material_closeout_pack.md`、`paper_manuscript_closeout_readiness_matrix.md` 与 `% T82-SUPPORT: ...` 注释；该链路只覆盖 `Runtime, quantization, and fixed-point degradation`、`Embedded runtime and board-level validation`、`Discussion` 中的 deployment/support boundary 段落、`Conclusion` 中的 remaining technical gap 段落，不代表 full-manuscript closeout。
9. `T83` 之后，如需判断当前 note 是否已经完成全文级 consistency sweep，以及后续是否只能进入 bounded final polish，应优先查看 `paper_fullnote_consistency_crosswalk.md`、`paper_closeout_gate_and_blocker_register.md` 与 `% T83-CLOSEOUT: ...` 注释；该链路只证明“当前主线 note 已形成可审计的一致性版本”，不等于 submission-ready pack、deployment closure 或 real-board success。
10. `T84` 之后，如需判断当前 note 哪些 section 已完成 reader-facing final polish、内部 task/provenance 术语该如何翻译、以及 appendix / supplement / blocked surface 应如何向读者装配，应优先查看 `paper_bounded_final_polish_change_map.md`、`paper_reader_facing_term_translation_table.md`、`paper_appendix_supplement_reader_assembly_map.md` 与 `% T84-POLISH: ...` 注释；该链路只做 translation / condensation / assembly，不等于 submission-ready pack。
11. `T85` 之后，如需判断当前 note 是否已经完成 residual wording-lag 清扫、是否允许进入下一张 bounded submission-pack assembly 任务、以及哪些 surface 仍必须保留为 blocker / exclusion，应优先查看 `paper_submission_readiness_preflight_gate.md`、`paper_submission_blocker_matrix.md`、`paper_residual_state_lag_sweep.md` 与 `% T85-PREFLIGHT: ...` 注释；该链路只做 preflight / blocker 明确化，不等于 submission-ready pack 已完成。
12. `T86` 之后，如需判断当前 mainline note/material 应如何组装成 submission-facing package、哪些 surface 进入 main text / appendix / supplement、哪些必须显式排除，应优先查看 `paper_submission_pack_assembly_manifest.md`、`paper_submission_surface_route_map.md`、`paper_submission_exclusion_register.md`、`paper_submission_author_handoff.md` 与 `% T86-ASSEMBLY: ...` 注释；该链路只做 assembly / exclusion 收口，不等于 submission-ready pack 已完成。
13. `T87` 之后，如需判断当前 note 是否已经完成作者终检级 QA、哪些表述仍属于 red-flag、以及后续只允许做哪些 bounded manual finish，应优先查看 `paper_author_final_qa_checklist.md`、`paper_presubmission_regression_gate.md`、`paper_submission_wording_redflag_register.md`、`paper_manual_finish_queue.md` 与 `% T87-QA: ...` 注释；该链路只服务 author-final QA / pre-submission regression gate，不等于 submission-ready completed。
14. `T88` 之后，如需判断哪些 bounded manual finish 已真实执行、当前 mainline surface 如何冻结、blocked disclaimer 该保留在哪些位置，以及后续是否只允许 frozen-mainline handoff，应优先查看 `paper_manual_finish_execution_log.md`、`paper_mainline_surface_freeze_manifest.md`、`paper_author_edit_decision_register.md`、`paper_blocked_surface_disclaimer_table.md`、`paper_frozen_mainline_handoff_gate.md` 与 `% T88-MANUAL: ...` 注释；该链路只服务 manual-finish execution / surface freeze / handoff 固化，不等于 submission-ready completed。
15. `T89` 之后，如需判断当前 frozen-mainline handoff 的唯一入口、哪些 surface 是当前允许引用的 authoritative source、post-freeze 手工改动是否需要 reopen 或 evidence task、以及 blocked surface 将来必须满足什么新证据才能重开，应优先查看 `paper_frozen_mainline_handoff_packet.md`、`paper_frozen_mainline_source_of_truth_map.md`、`paper_postfreeze_change_control.md`、`paper_blocked_surface_reentry_conditions.md`；该链路只服务 handoff 与 change-control，不升级证据等级，也不授权直接改写 note 或编译产物。
