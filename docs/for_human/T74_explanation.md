# T74 任务解释与本次 Review 说明

## 1. 先用通俗的话解释：T74 到底在做什么

`T74` 不是去跑新实验，也不是去补真板执行。

它做的是一件更“写论文前的整理工作”：
把仓库里已经被前面任务验证过的 simulation/material 证据，整理成一套可以直接服务论文写作的材料包。这个材料包要回答几个很实际的问题：

- 现在最稳、最适合放进论文的主结果表是哪几张？
- 哪些图已经 ready，哪些还只是 partial？
- 哪些内容只能放附录，哪些只能放补充材料？
- 哪些东西绝对不能写成“已经完成”？
- 如果后面的人继续写论文，他应该去哪里找每张表、每张图、每条 caption 的证据来源？

所以，`T74` 的本质不是“新增结果”，而是“把已有结果整理成可投稿材料入口”。

## 2. 这次实现具体做了什么

### 2.1 任务目标

结合 `docs/04_task_board.md` 和 `docs/07_handoff.md`，`T74` 是 `T73` 之后的主线收口动作。`T73` 解决的是 claim / result / risk 三本账刷新，`T74` 解决的是“这些账怎么变成论文作者可直接复用的表、图、caption、插入位置和缺口清单”。

当前版本的 `T74` 任务包还比最初口头理解更强，额外要求了：

- 全部 paper-facing 项目统一采用 stable ID；
- 给出 `main text / appendix / supplement only` 的插入映射；
- 生成任务级 `submission_bundle_manifest.json`；
- 在任务级目录下补齐 traceability 资产。

这说明 `T74` 不是一般的“写几份说明文档”，而是一个有明确可审计结构的材料打包任务。

### 2.2 实际交付物

本次交付主要分成三组。

第一组是四份 paper-facing 主文档：

- `docs/paper_materials/paper_simulation_result_table_pack.md`
- `docs/paper_materials/paper_figure_caption_pack.md`
- `docs/paper_materials/paper_maintext_insertion_map.md`
- `docs/paper_materials/paper_submission_material_gap_checklist.md`

它们分别负责：

- 汇总哪些表可以进入论文；
- 汇总每张图、每张表和每条 supplement note 的 caption / safe wording；
- 明确主文、附录、补充材料的放置层级；
- 说明当前还缺什么，但这些缺口是不是已经阻断了 simulation/material-first 的投稿路径。

第二组是任务级 traceability 资产目录：

- `docs/figure_assets/T74_paper_ready_simulation_result_pack/README.md`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/figure_manifest.json`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/result_source_map.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/caption_source_map.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/table_snapshot.csv`
- `docs/figure_assets/T74_paper_ready_simulation_result_pack/submission_bundle_manifest.json`

这一组不是新实验结果目录，而是“论文材料索引目录”。它的作用是把每个 stable ID 和它对应的证据路径、边界口径、状态、表格快照、提交包位置串起来。

第三组是配套更新：

- 更新 `docs/paper_materials/README.md`
- 更新 reviewer 文档
- 更新面向人的解释文档
- 更新 worker summary

### 2.3 这套实现怎么组织材料

本次 `T74` 用三类 stable ID 管理全部材料：

- `T74-TBL-*`：表
- `T74-FIG-*`：图
- `T74-SUP-*`：补充说明

我复核时确认，当前实现共整理出：

- `7` 个表项
- `4` 个图项
- `4` 个补充说明项

对应状态统计为：

- `ready = 11`
- `partial = 3`
- `blocked = 1`

这里最关键的不是数量，而是边界处理方式：

- `T74-TBL-01` 把 `T24` 冻结四场景结果收成主文主表；
- `T74-FIG-02` 复用 `FR6` 六 seed 图包作为主文可用图；
- `T74-TBL-02` 到 `T74-TBL-05` 把 `FR7`、`FR6`、`T50`、`T48` 收成附录候选；
- `T74-TBL-06` 把 `T49/T71/T72` 收成 real-board gate/provenance 边界表，但只允许作为 supplement；
- `T74-TBL-07` 把 `FR8 statcalib` 收成 extension-lane closure 表，但仍是 `partial + supplement only`；
- `T74-FIG-04` 则被明确标成 `blocked`，因为当前证据不支持诚实画出统一 portability/deployment closure 图。

这意味着作者没有把“还不能说的话”偷偷写成“差一点就能说的话”，而是显式保留下来，供后续任务继续处理。

### 2.4 对后续开发和论文推进的意义

`T74` 的意义不在于新增实验，而在于把主线证据第一次整理成了“能直接进入写作流程”的形式。

它带来的实际收益有三点：

- 以后写论文主文、附录、补充材料时，不需要重新发明命名，也不需要重新梳理每张图表的证据来源。
- 边界口径被提前写进材料包本身，后续作者更难把 `T48/T72/FR8` 这类边界证据误写成完成态。
- 即便真板条件暂时不具备，主线也已经拥有一条诚实的 `simulation/material-first` 路径，不必把“没有真板”误等价成“论文材料完全不能推进”。

## 3. 为什么这次 Review 给出 `PASS`

我给 `PASS`，不是因为“文档写得多”，而是因为它满足了这类任务真正该满足的三件事。

### 3.1 任务确实完成了

任务包要求的核心交付都已经落地：

- 四份主文档齐全；
- traceability 目录齐全；
- `README` 已更新；
- `submission_bundle_manifest.json` 已补齐；
- stable ID 体系已形成。

这不是“只起了几个文件名”的假完成，而是一套完整的 paper-facing 材料组织结构。

### 3.2 我做了独立的一致性核查

按用户要求，我没有重跑任何长实验，但做了足够强的只读结构验证。

我实际核查了这些点：

- `figure_manifest.json`、`submission_bundle_manifest.json`、`result_source_map.csv`、`caption_source_map.csv`、`table_snapshot.csv` 之间的 stable ID 是否能对上；
- 所有 direct source / supporting source 路径是否真实存在；
- `status_counts` 是否一致；
- `paper_simulation_result_table_pack.md`、`paper_figure_caption_pack.md`、`paper_maintext_insertion_map.md`、`paper_submission_material_gap_checklist.md` 是否真的使用了同一套 stable ID；
- `runs/`、`artifacts/`、源码目录、测试目录是否被这次任务改动。

核查结果是：

- manifest 共 `15` 个 stable ID；
- result/caption source map 各 `15` 行；
- snapshot 共 `28` 行；
- 状态统计一致：`ready=11`、`partial=3`、`blocked=1`；
- 直接证据与支撑证据路径未发现缺失；
- 没有发现对 `runs/`、`artifacts/`、`cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 的越权修改。

### 3.3 没有把计划写成事实

这是这次 review 最看重的一点。

我重点确认了以下边界没有被升级：

- `T24` 没有被写成 expanded benchmark、`.tflite` 主结果或 real-board 主结果；
- `T48` 仍只是 isolated current-host true `.tflite` runtime，不是 default-env/HIL/deployment closure；
- `T72` 仍只是 read-only gate / regeneration / provenance 边界，不是 real-board execution success；
- `FR8` 仍只是 extension lane，不是 mature comparator；
- `T74-FIG-04` 被明确保留为 `blocked`，没有被包装成“先留个坑，后面补一下就算完成”。

这说明作者是在诚实整理证据，而不是借文档打包顺手升级结论。

## 4. Worker 已写文档的情况：有没有问题，补充了什么

Worker 原先写的 `docs/review/T74_review.md` 和 `docs/for_human/T74_explanation.md` 大方向基本是对的，没有发现明显事实性错误。

但它们有两个不足：

- review 还是更像 self-check，不够强调“独立 reviewer 已重新核对过 manifest/source map/snapshot/path existence”；
- for-human 说明虽然能讲清任务用途，但对“为什么这次可以判 PASS”说得还不够具体。

所以我这次做的不是推翻，而是补强：

- 把 review 文档改成正式 reviewer 口径；
- 把验证依据写得更清楚；
- 把 `PASS` 的理由落在“任务完成、结构一致、边界诚实、无越权修改”四个点上；
- 保留一个非阻塞提醒：当前工作区还有 `T74` 之外的既有 diff，后续提交时需要精确暂存。

## 5. 一句话总结

`T74` 已经把当前仓库中分散的 simulation/material 主线证据，整理成了一套真实、可追溯、边界受控、可直接服务论文写作的材料包；我给 `PASS`，是因为它完成的是“证据组织和写作入口收口”，而不是伪装成新增实验或硬件闭环。
