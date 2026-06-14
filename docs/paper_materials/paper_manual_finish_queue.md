# T87 Bounded Manual Finish Queue

本表只列作者在 `T87` 之后仍可继续做的人工终修动作。它不是 reopen 清单，更不是 claim promotion 清单。

| queue_id | allowed_manual_action | why_manual | depends_on | must_not_upgrade | owner |
| --- | --- | --- | --- | --- | --- |
| `MF01` | 对 `Numerical Results`、`Discussion`、`Conclusion` 做句法级流畅度润色 | 这些段落已经完成 route / boundary 固化，剩余工作主要是读者体验与句间衔接。 | `paper_author_final_qa_checklist.md` `QA01-QA03`; `% T87-QA` markers | 不得改动 evidence hierarchy，不得把 next step 写成 completed submission / deployment / hardware story。 | `author` |
| `MF02` | 在既有 manifest 允许范围内选择 `T75-FIG-M01` 或 `T74-TBL-01` 作为 frozen result 的主呈现 | 这是版面和读者负担的人工决策，不是证据决策。 | `paper_submission_pack_assembly_manifest.md` `PKG-MT-01`; `paper_submission_surface_route_map.md` | 不得借换图换表新增 benchmark、runtime 或 hardware 含义。 | `author` |
| `MF03` | 微调 appendix / supplement 的标题、顺序和 bridge 句 | 主文到 supporting surfaces 的过渡仍需要人工整理以适应期刊/会议结构。 | `paper_submission_surface_route_map.md`; `paper_submission_author_handoff.md` | 不得把 appendix/support surface 上移到 main text，不得取消 exclusion note。 | `author` |
| `MF04` | 对 `T75-FIG-A01` / `T74-FIG-03` 的 caption 做简化或读者化润色 | boundary schematic 的可读性受图注语言影响，适合人工细修。 | `paper_submission_pack_assembly_manifest.md` `PKG-APX-05`; `paper_appendix_bridge_pack.md` | 不得把 boundary schematic 改写成 result figure、deployment pipeline 或 portability closure 图。 | `author` |
| `MF05` | 依据目标 venue 的页数或模板做压缩、换行、段落断句与版式收束 | 这是投稿前常规人工动作，无法通过证据台账自动决定。 | `paper_presubmission_regression_gate.md`; current venue constraints | 不得因为篇幅压力删掉 blocked/exclusion 提示或把风险边界压成完成态。 | `author` |
