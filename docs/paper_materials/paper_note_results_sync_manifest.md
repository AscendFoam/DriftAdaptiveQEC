# T77 Note Results Sync Manifest

## 1. 作用与边界

本文件记录 `T77` 对 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的结果层同步范围、source chain、允许/禁止表述，以及本轮明确不同步的内容。

它只覆盖：

- `Abstract`
- `Summary of Contributions`
- `Experimental Setup`
- `Numerical Results and Benchmark Plan`
- `Discussion`
- `Conclusion`

它不把 note 升格为新的事实来源；真正的结果边界仍然回到 `T74/T75/T76` 与其上游 task/review/run/artifact。

`T78` 之后，本文件继续只负责结果层 section 的同步 trace。标题、引言、`Relationship to Existing Work`、`statcalib` 层级降权和 LaTeX warning 收口改由 `paper_note_alignment_and_layout_closeout.md` 记录。

## 2. 已同步 section 清单

| note section / subsection | 本轮修改重点 | 绑定 source IDs / task IDs | 允许表达 | 禁止表达 |
| --- | --- | --- | --- | --- |
| `Abstract` | 把主结果压回 locked four-scenario software-HIL；把 `statcalib` 改成 extension-lane / no-promotion | `T74-TBL-01`, `T74-TBL-02`, `T74-TBL-03`, `T74-TBL-07`; tasks `T24/T57/T58/T70`; reviews `T70/T76` | adaptive affine mainline result; bounded ablation support; descriptive multi-seed support; statcalib extension lane | paper-grade expanded benchmark; teacher necessity proved; promoted mature comparator |
| `Summary of Contributions` paragraph 4 | 把“benchmark suite”改成 mainline + bounded follow-up + extension-lane 分层 | `T74-TBL-01`, `T74-TBL-02`, `T74-TBL-03`, `T74-TBL-07`; tasks `T24/T57/T58/T70` | frozen mainline benchmark; bounded support layers; no-promotion lane | unified mature comparator story; statcalib replaces mainline |
| `Experimental Setup / Software-HIL protocol` | 明确 locked frozen protocol 是 mainline anchor | `T74-TBL-01`; tasks `T24/T25`; reviews `T24/T25` | authoritative frozen anchor; mock-backed software-HIL | board-level latency measured; hardware resource evidence complete |
| `Experimental Setup / Scenarios and modes` | 把 `statcalib` 从 mainline comparison 拆到 bounded extension lane | `T74-TBL-01`, `T74-TBL-02`, `T74-TBL-03`, `T74-TBL-07`; tasks `T24/T57/T58/T70` | mainline comparison vs support layers | one benchmark silently absorbing extension lane |
| `Numerical Results / Four-scenario affine benchmark` | 同步到 `T75-FIG-M01` / `T76-PREVIEW-M01` 的主图入口与 `T74-TBL-01` authoritative fallback | `T74-TBL-01`, `T75-FIG-M01`, `T76-PREVIEW-M01`; tasks `T24/T75/T76`; reviews `T25/T75/T76` | winner across four frozen scenarios; UKF runner-up; authoritative frozen result layer | expanded benchmark winner; `.tflite` ranking; real-board ranking |
| `Numerical Results / Feature and teacher ablations` | 定位为 appendix-side bounded support，并阻断 teacher-necessity 叙事 | `T74-TBL-02`, `T74-TBL-03`, `T75-FIG-M02`, `T76-PREVIEW-M02`; tasks `T57/T58/T75/T76`; reviews `T57/T58/T76` | bounded ablation support; descriptive mechanism reading | causal closure; teacher necessity proved |
| `Numerical Results / Statistical calibration ...` 三个 subsection | 改成 bounded extension lane / persistent tie / no-promotion gate | `T74-TBL-07`, `T74-SUP-01`; tasks `T64`-`T70`; review `T70` | separately labeled extension lane; no-promotion; no unique clean threshold | promoted mature comparator; unique threshold established |
| `Numerical Results / Mechanism probe for residual-b behavior` | 把主文图、附录查数表和保守解释绑在一起 | `T74-TBL-02`, `T74-TBL-03`, `T75-FIG-M02`, `T76-PREVIEW-M02`; tasks `T57/T58/T75/T76`; reviews `T58/T76` | descriptive multi-seed evidence; mixed / mostly harmful intervention | intervention fixes mechanism; teacher necessity proved |
| `Numerical Results / Runtime, quantization, and fixed-point degradation` | 把 runtime 从“已完成验证”压回 boundary-layer + future measurement | `T74-TBL-05`, `T74-SUP-03`; task `T48`; review `T48` | isolated current-host true runtime; future benchmark items | default-env recovery; HIL closure; deployment closure |
| `Numerical Results / Embedded runtime and board-level validation` | 明确 `T48` 与 `T49/T71/T72` 的分层边界 | `T75-FIG-A01`, `T76-PREVIEW-A01`, `T74-FIG-03`, `T74-TBL-05`, `T74-TBL-06`, `T74-SUP-03`, `T74-SUP-04`; tasks `T48/T49/T71/T72/T75/T76`; reviews `T72/T76` | isolated current-host true runtime; read-only real-board gate / provenance; current-host `NO_GO` | real-board execution success; hardware validated; deployment closure |
| `Discussion` | 把 strongest accepted result 固定回 `T24`，并把 `statcalib` 改成 supplement-side no-promotion | `T74-TBL-01`, `T74-TBL-07`, `T74-SUP-01`, `T75-FIG-M01`, `T75-FIG-M02`; tasks `T24/T70/T75/T76`; reviews `T70/T76` | fast/slow contract; mainline frozen result; replaceable slow-loop module | CNN universally superior; promoted comparator |
| `Conclusion` | 保留两层主结论，并显式保留 deployment / board layered boundary | `T74-TBL-01`, `T74-TBL-02`, `T74-TBL-07`, `T74-SUP-01`, `T75-FIG-A01`, `T76-PREVIEW-A01`, `T74-TBL-05`, `T74-TBL-06`; tasks `T24/T57/T70/T72/T75/T76`; reviews `T70/T72/T76` | bounded software-HIL evidence; no-promotion lane; layered deployment boundary | mature comparator closure; real-board completion; unified portability closure |

## 3. 与 `T76` 预览包的绑定

本轮 note-sync 直接复用了以下 paper-facing 预览链：

- `T75-FIG-M01` -> `T76-PREVIEW-M01` -> `T74-TBL-01`
- `T75-FIG-M02` -> `T76-PREVIEW-M02` -> `T74-TBL-02`, `T74-TBL-03`
- `T75-FIG-A01` -> `T76-PREVIEW-A01` -> `T74-FIG-03`, `T74-TBL-05`, `T74-TBL-06`, `T74-SUP-03`, `T74-SUP-04`, `T74-FIG-04`

聚合预览 `T76-PREVIEW-CS01` / `T76-PREVIEW-PDF01` 只用于 rendered-QA circulation，不构成单独的 note 结果结论入口。

## 4. 本轮明确未同步的内容

| 未同步内容 | 原因 | 当前处理 |
| --- | --- | --- |
| `Title` | 不在 `T77` 允许修改章节范围内 | 保持原样，等待后续专门 manuscript task 再判断是否需要降级命名 |
| `Introduction` / `Brief Review of the GKP Code` / `Noise and Drift Model` / `Model Architecture` / `Relationship to Existing Work` | `T77` 明确禁止扩大到非结果层章节 | 保持原样，不视为已按 `T74/T75/T76` 重新校准 |
| `T74-FIG-04` 统一 portability/deployment closure 图 | 仍是 blocked slot | 只允许在 note 与 callout 中保留 blocked reminder |
| `T49/T71/T72` real-board execution narrative | 证据仍停在 read-only gate/provenance with current-host `NO_GO` | 只在 boundary 段里保留 gate wording，不升级为结果 |
| `FR8` promoted comparator / unique threshold narrative | `T70` 已给出 no-promotion 与 future-selection gate | 只保留 extension-lane / persistent-tie / supplement-side 口径 |

## 5. 本地编译检查状态

- 状态：`COMPILED_WITH_TEXLIVE_2024`
- 工具链：
  - 诊断：`python scripts/latex_doctor.py --json`（需设置 `PYTHONUTF8=1` 以绕过 Windows 控制台解码冲突）
  - 编译：`python scripts/compile_latex.py D:\Codes\Quantum\DriftAdaptiveQEC\docs\paper_notes\CNN_FPGA_GKP_theory_note_draft.tex --json`
  - 实际编译器：`C:\texlive\2024\bin\windows\latexmk.EXE`
- 结果：
  - `CNN_FPGA_GKP_theory_note_draft.pdf` 已成功生成
  - `aux` / `fdb_latexmk` / `fls` / `log` / `out` / `synctex.gz` / `toc` 已刷新
- 编译备注：
  - 初次编译失败是因为 `NO_GO...` 字面量中的下划线未转义；修正为 `\texttt{NO\_GO...}` 后再次编译成功
  - 当前 `.log` 仍有 underfull hbox 与未解引用/引文 warning；这属于 note 草稿整体质量问题，不是本轮结果层 sync 的阻断项

## 6. 过程性残留清理状态

- `.tmp_t76_render_a01.png`：已删除
- `.tmp_t76_render_m02.png`：已删除
- `.tmp_t76_render_probe.png`：已删除
- `.tmp_t76_fontcache/`：已删除
