# T88 Author Edit Decision Register

本表记录 `T88` 中真实发生的作者向编辑决策，而不是空泛原则。

| decision_id | decision_topic | options_considered | selected_option | reason | must_not_imply | evidence_anchor |
| --- | --- | --- | --- | --- | --- | --- |
| `ED01` | 主文 frozen result 的当前 note 主呈现 | `T75-FIG-M01`；`T74-TBL-01` | 在当前 note 中选择 `T74-TBL-01` / `Table~\\ref{tab:five-mode-benchmark}` | 当前 note 已内嵌 authoritative numeric table；直接冻结表格是最少歧义的 mainline handoff。 | 不得暗示 figure 比 table 更强，也不得暗示主结果层被重新定义。 | `paper_maintext_results_authoring_pack.md`; `paper_caption_lock_and_placement_notes.md`; `paper_mainline_surface_freeze_manifest.md` `FZ01` |
| `ED02` | 机制层的主文 vs 附录职责分配 | 主文保留机制 figure + 附录表；主文仅留文字、把全部表都下沉 | 保持“主文 descriptive reading + appendix authoritative numeric companion” | 这既保留可读性，也不把 `FR6/FR7` 压成主结果替代物。 | 不得暗示 causal closure、teacher necessity 或 intervention success。 | `paper_appendix_bridge_pack.md`; `paper_submission_surface_route_map.md`; `paper_mainline_surface_freeze_manifest.md` `FZ02` |
| `ED03` | appendix / supplement bridge 句如何冻结 | 继续写成泛化“supporting materials”；显式写出 appendix 与 supplement 各自承载什么 | 显式冻结 appendix 与 supplement 的分工 | `T88` 的目标是防漂移；因此必须把 route 从“允许继续整理”改成“当前写法已固定”。 | 不得暗示 appendix/support surface 可以再上移为主文结果。 | `paper_submission_pack_assembly_manifest.md`; `paper_submission_author_handoff.md`; `paper_mainline_surface_freeze_manifest.md` `FZ03/FZ05` |
| `ED04` | boundary schematic 是否在当前 note 中再造一套 caption | 在 note 内追加新 caption；沿用 `T75/T74` 已锁定的外部 caption 与 placement notes | 不在当前 note 内重复生成 caption | 当前 note 不内嵌该 schematic；重复造 caption 只会制造第二套 surface 描述。 | 不得暗示 boundary schematic 是主结果图，或允许它承载 deployment closure 叙事。 | `paper_caption_lock_and_placement_notes.md`; `paper_figure_caption_pack.md`; `paper_mainline_surface_freeze_manifest.md` `FZ04` |
| `ED05` | 结尾收束应该强调什么 | 继续写“还会继续 manual finish/assembly”；改写为 frozen-mainline handoff | 选择 frozen-mainline handoff | `T88` 之后主线需要的是稳定 handoff，而不是再为下一轮人工修改留下模糊入口。 | 不得暗示 submission-ready completed、hardware-ready 或 blocked surface 解锁。 | `paper_frozen_mainline_handoff_gate.md`; `paper_presubmission_regression_gate.md`; `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` |
