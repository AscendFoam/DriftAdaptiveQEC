# T77 任务解释与 Review 说明

## 1. 先用通俗的话说，`T77` 在做什么

`T74`、`T75`、`T76` 已经把论文结果层材料分三步收口出来了：

- `T74`：把表、图、caption、stable ID 和 traceability 冻住；
- `T75`：把这些稳定材料压成可以直接写 Results 的 prose 和最终成图；
- `T76`：把图真实渲染出来，检查可读性，并给出 Results section 的装配顺序。

但到 `T76` 结束时，还有两个“很像小问题，其实会影响后续写作”的缺口：

1. 预览包的 source-map 字段语义还不够干净；
2. note 草稿里的结果层还没有和最新审查过的材料完全同步。

所以 `T77` 做的事，不是新实验，也不是恢复全文扩写，而是：

- 把 `T76` 的 traceability 再打磨干净一点；
- 把已经审过的结果层材料同步到现有 note 草稿；
- 如果本机有 LaTeX 工具链，就做一次受控编译检查；
- 顺手清理 `T76` 留下的本地探针/缓存残留。

## 2. 这次实现到底改了什么

### 2.1 先修 `T76` 的 traceability

`T76` 的 review 里留下的主要 warning 是：contact sheet / PDF 这种“聚合预览”在 CSV 里把两种不同语义混在了一列里。

`T77` 的修法是：

- 在 [preview_source_map.csv](D:/Codes/Quantum/DriftAdaptiveQEC/docs/figure_assets/T76_rendered_figure_qa_pack/preview_source_map.csv:1) 里新增 `source_preview_ids`；
- 把 `upstream_t74_ids` 保留给真正的 `T74-*` stable IDs；
- 把直接拼接进去的 `T76-PREVIEW-*` 写到 `source_preview_ids`；
- 再让 [render_manifest.json](D:/Codes/Quantum/DriftAdaptiveQEC/docs/figure_assets/T76_rendered_figure_qa_pack/render_manifest.json)、[paper_rendered_figure_qa.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_rendered_figure_qa.md:75) 和 [README.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/figure_assets/T76_rendered_figure_qa_pack/README.md:31) 一起解释这条新 schema。

这一步的价值在于：以后不管是人读还是脚本读，都更容易分清“这张聚合图直接用了哪些 preview”和“这些 preview 再往上追到哪些 `T74-*`”。

### 2.2 再把逐图 QA 的回链写实

`T76` 原本已经能证明图可读，但 reviewer 当时指出一个问题：每张图的 QA 结论本身还不够“自带追溯信息”。

`T77` 把这件事补了：

- `T75-FIG-M01` / `M02` / `A01`
- 对应 `T76-PREVIEW-*`
- 对应全部上游 `T74-*`

现在都直接写进了 [paper_rendered_figure_qa.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_rendered_figure_qa.md:29)。

这意味着，后面就算不先打开 JSON/CSV，人也能从 QA 文档本身看出这张图到底在链路里的什么位置。

### 2.3 结果层 note-sync

这部分是 `T77` 的主体。

任务包只允许同步这些章节：

- `Abstract`
- `Summary of Contributions`
- `Experimental Setup`
- `Numerical Results and Benchmark Plan`
- `Discussion`
- `Conclusion`

Worker 也确实把同步重点放在这里，并在源码里加了 `% T77-SOURCE: ...` 注释，例如：

- [Abstract](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:18)
- [主结果段](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:664)
- [`statcalib` 结果层](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:726)
- [runtime / board boundary](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:853)
- [Discussion](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:981)
- [Conclusion](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:1028)

从内容上看，这次同步做了几件关键事：

1. 把 `T24` 重新压回“唯一主锚点”
   - 也就是 locked four-scenario、mock-backed software-HIL 主结果层。

2. 把 `FR6/FR7` 保持在 descriptive / bounded support
   - 不写成因果闭环，也不写成 teacher necessity 已证明。

3. 把 `FR8 statcalib` 压回 extension-lane / no-promotion
   - 甚至额外补了一条新的 note-facing callout 句式 [T77-CALLOUT-R7](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_results_callout_sheet.md:24)。

4. 把 `.tflite` 和 real-board 边界继续写成 layered boundary
   - `T48` 仍只是 isolated current-host true runtime；
   - `T49/T71/T72` 仍只是 gate/provenance with `NO_GO`；
   - 不把它们压成 deployment closure。

### 2.4 新增 note-sync manifest

这是这轮很有价值的新增文件：
[paper_note_results_sync_manifest.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_note_results_sync_manifest.md)。

它记录了：

- 哪些 note section 这次被同步了；
- 每个 section 绑定哪些 `T74/T75/T76` / task / review；
- 这些 section 允许写什么；
- 禁止写什么；
- 哪些章节这轮故意不同步，以及为什么不同步。

它本质上是在给“note 里的结果层说法”加一层 reviewer 可读的防错护栏。

### 2.5 编译检查和残留清理

Worker 还做了两件收尾动作：

1. 编译 note
   - 生成并刷新了 [PDF](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.pdf) 和辅助文件；
   - 当前日志显示编译完成，但仍有 `Underfull \hbox` 这类排版 warning。

2. 清理 `T76` 过程残留
   - `.tmp_t76_render_a01.png`
   - `.tmp_t76_render_m02.png`
   - `.tmp_t76_render_probe.png`
   - `.tmp_t76_fontcache/`

这四类路径当前都表现为受控删除 diff，说明这轮确实把之前 review 留下的临时文件噪声一并处理了。

## 3. 这件事对后续开发有什么意义

`T77` 的意义，不是“论文已经准备好扩写”，而是把“结果层材料”从纸面上再推进一小步：

- `T74` 解决的是结果包和 stable ID；
- `T75` 解决的是 bounded Results authoring；
- `T76` 解决的是真实渲染 QA 和段落装配；
- `T77` 解决的是 note 结果层同步和 traceability 精修。

这样下一步如果 Captain 要判断“是否可以开 paper draft reopen gate”，就不再只是看零散材料，而是能看一整条更完整的 paper-facing 链路。

同时，`T77` 也明确没有做这些事：

- 没有新 benchmark；
- 没有新 `.tflite` / real-board 事实；
- 没有治理文档更新；
- 没有源码或测试修改；
- 没有把全文都当成已同步。

## 4. 为什么我的 review 结果是 `PASS_WITH_WARNINGS`

### 4.1 为什么不是 `BLOCK`

因为任务包要求的核心交付基本都完成了，而且能被直接验证：

- `preview_source_map.csv` 的字段语义已经拆开；
- `paper_rendered_figure_qa.md` 逐图写清了 `T74/T75/T76` 回链；
- `paper_note_results_sync_manifest.md` 已建立；
- note 的结果层章节确实加入了 `T77-SOURCE` 注释；
- `runs/`、`artifacts/`、源码、测试、治理文档都没有被改动；
- note 的 PDF 和辅助文件已经刷新；
- `.tmp_t76_*` / `.tmp_t76_fontcache/` 已删除。

所以从任务完成度看，`T77` 是达标的。

### 4.2 为什么也不是完全无保留的 `PASS`

我保留 warning，主要是因为还有三件事情不该被说成“彻底收口”。

#### 第一件：整份 `.tex` 不能被当成“已全稿同步”

`T77` 同步的是结果层章节，不是整份 note。

这一点 Worker 自己其实也知道，并写进了 manifest：
[paper_note_results_sync_manifest.md:49-50](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_note_results_sync_manifest.md:49)。

问题在于，当前 `git diff` 里的 `.tex` 仍然混有非结果层 hunks。最明显的是 `Relationship to Existing Work` 里这一段：
[CNN_FPGA_GKP_theory_note_draft.tex:575-605](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:575)

它不在 `T77` 允许同步的章节范围内，也没有 `T77-SOURCE` 标记。所以：

- 我不把它算成 `T77` 失败；
- 但我也不会把整份 `.tex` 当成“已经全部按 T77 校准完毕”。

这就是第一条 warning。

#### 第二件：`statcalib` 虽然被压回 no-promotion，但视觉层级仍偏高

在正文里，`statcalib` 的语言已经非常克制：

- [726-735](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:726)
- [759-766](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:759)
- [791-799](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:791)

这些地方都在说：

- 这是 extension lane；
- 不能 promoted；
- 没有 unique clean threshold。

但问题是，这些内容仍然以多个 subsection 和表格的形式留在 `Numerical Results` 主体里。对内部 note 这还可以接受；但如果以后向作者或外部读者继续传播，单靠文字解释还不够，版面层级最好也一起降下来。

这不是任务失败，但它解释了为什么我保留第二条 warning。

#### 第三件：编译成功不等于排版完全收口

PDF 已经生成，辅助文件也刷新了，这说明“有本地可用工具链并且能编过”这一点是成立的。

但日志里仍有 `Underfull \hbox` 一类 warning，例如：
[CNN_FPGA_GKP_theory_note_draft.log:405](D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.log:405)

这不会否定 `T77` 的任务结论，但它说明：

- 当前 note 仍是工作稿；
- 不是已经进入版面最终收口状态。

## 5. 如果 Worker 已经写了 review / explanation，我怎么处理

这轮比较特别，因为 Worker 已经在 [docs/review/T77_review.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/review/T77_review.md) 里写了一个“自检包”，而不是正式 adversarial review。

我对它的判断是：

- 它作为 self-check 是有价值的；
- 但它不能替代真正的 reviewer verdict；
- 尤其不能替代对 coexisting `.tex` diff 和 whole-note 边界的判断。

所以我保留了 Worker 文档里正确的部分，例如：

- 任务目的的理解；
- traceability/schema 的修法；
- note-sync manifest 的作用；
- LaTeX 编译与 `.tmp_t76_*` 清理。

然后补上了它没真正对抗性检查的部分：

1. 当前整份 note 仍不能被视为“全稿同步”；
2. `statcalib` 的视觉层级问题仍未彻底解决；
3. 编译成功只说明可编译，不说明排版最终收口。

## 6. 我建议的下一步

如果 Captain 接受这轮工作，下一步最合理的不是继续扩大 `T77`，而是二选一：

1. 如果要推进 paper reopen gate
   - 新开一个更窄的 gate/review 任务；
   - 明确判断当前材料是否足以支持 prose reopen。

2. 如果要继续清理 note/manuscript
   - 新开一个只处理非结果层章节校准与版面层级的任务；
   - 包括 `Relationship to Existing Work`、`statcalib` 视觉降权、排版 warning 收口。

不建议把这些事情继续叠加到 `T77` 上，因为 `T77` 的任务边界本来就不是 full-manuscript reopen。
