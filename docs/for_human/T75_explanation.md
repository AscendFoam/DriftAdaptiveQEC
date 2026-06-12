# T75 任务解释与本次 Review 说明

## 1. 先用通俗的话解释：T75 到底在做什么

`T75` 不是去跑新实验，也不是去补 `.tflite`、真板、训练复现。

它做的是更靠近论文作者的一层“写作收口”工作：
把 `T74` 已经整理好的 stable-ID 材料包，进一步压缩成可以直接放进主文 Results 的文字和图。

如果把 `T74` 理解成“论文作者可查账的材料目录”，那么 `T75` 就是在回答更实际的问题：

- 主文 Results 现在到底应该怎么写，才既清楚又不越界？
- 哪三张图可以直接作为主文/附录最终图？
- 每张图的标题、caption 核心句、放置位置、替代表述应该怎么锁定？
- 哪些话现在绝对不能写？

所以，`T75` 的本质不是“新增事实”，而是“把已有事实变成可直接落笔的 authoring pack”。

## 2. 这次实现具体做了什么

### 2.1 任务目标

从 `docs/04_task_board.md` 和 `docs/07_handoff.md` 的当前状态看，`T75` 是 `T74` 之后的下一层主线动作。

`T74` 已经完成了：

- stable-ID 的表、图、补充说明体系；
- source map / caption map / submission-bundle manifest；
- main text / appendix / supplement 的分层路线；
- gap checklist。

但 `T74` 还停留在“材料可追溯”的层面，不等于“作者已经可以直接拿去写主文 Results 并定稿成图”。

`T75` 的任务，就是把这层再往前推一步，形成：

- bounded 的主文 Results prose；
- 最终三张可直接引用的 SVG 成图；
- caption / placement lock；
- appendix bridge；
- do-not-write guardrail。

### 2.2 实际交付物

这次交付主要有三组。

第一组是四份 authoring 文档：

- `docs/paper_materials/paper_maintext_results_authoring_pack.md`
- `docs/paper_materials/paper_caption_lock_and_placement_notes.md`
- `docs/paper_materials/paper_appendix_bridge_pack.md`
- `docs/paper_materials/paper_authoring_do_not_write_list.md`

它们分别负责：

- 把主文 Results 的推荐写法固定下来；
- 锁定三张图的标题、caption 核心句、放置层级和替代方案；
- 说明主文如何自然过渡到 appendix / supplement；
- 冻结当前绝对不能写的 overclaim 句式。

第二组是任务级 figure assets：

- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m01_t24_frozen_summary.svg`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_m02_fr6_multi_seed_mechanism.svg`
- `docs/figure_assets/T75_maintext_results_authoring_pack/t75_fig_a01_boundary_schematic.svg`

这三张图分别服务于：

- 主文冻结主结果；
- 主文机制/解释层；
- 附录边界说明层。

第三组是追溯与索引：

- `docs/figure_assets/T75_maintext_results_authoring_pack/authoring_manifest.json`
- `docs/figure_assets/T75_maintext_results_authoring_pack/asset_source_map.csv`
- `docs/figure_assets/T75_maintext_results_authoring_pack/README.md`
- `docs/paper_materials/README.md`

这一组确保每个 `T75-FIG-*` 都能明确映射回 `T74-*` stable IDs 和更上游的证据路径。

### 2.3 这次实现如何组织主文与附录

这次实现实际上锁定了一条相当清晰的写作路线。

主文层：

- `T75-FIG-M01` 对应 `T24` 冻结四场景结果，是主结果的图形压缩版本；
- `T74-TBL-01` 保留为数值上的 authoritative substitute；
- `T75-FIG-M02` 对应 `FR6` 六 seed 的 descriptive mechanism/intervention 读法，负责解释层，不替代主结果。

附录层：

- `T74-TBL-02` 到 `T74-TBL-05` 继续作为 `FR7`、`FR6` 数值快照、`T50`、`T48` 的 supporting tables；
- `T75-FIG-A01` 作为边界图，解释 frozen benchmark、isolated true runtime、real-board gate 之间的分层关系。

补充材料层：

- `T74-TBL-06`
- `T74-TBL-07`
- `T74-SUP-01` 到 `T74-SUP-04`

仍被明确保留在 supplement，不进入主结果层。

这说明作者没有试图把 deployment-facing 或 `FR8` extension-lane 材料强行挤进主文，而是继续保持层次分明。

### 2.4 这对后续开发和论文推进意味着什么

`T75` 的价值，在于它把“可追溯材料包”推进成了“可直接写作的 authoring 包”。

这会带来三方面收益：

- 后续作者写 Results 时，不需要重新组织语言和图表层次，可以直接从 authoring pack 落笔。
- review / rebuttal 时，更容易说明“哪句话对应哪份证据、哪句话绝对不能写”。
- 即便当前硬件 host 条件仍不具备，simulation/material-first 的论文路线也已经有了更成熟的落地入口，不必把“真板尚不可用”误等价成“主文 Results 无法推进”。

## 3. 为什么这次 Review 给出 `PASS`

我给 `PASS`，原因不是“文件又多了几份”，而是这次任务真正该完成的东西已经完成了，而且没有借 authoring 工作去偷偷升级证据。

### 3.1 任务要求的交付物确实都落地了

`T75` 要求的核心交付包括：

- 四份 authoring 文档；
- 三张最终 SVG；
- manifest/source map；
- README 更新；
- review / explanation / worker summary。

这些内容现在都已经在任务允许路径下落地，没有发现“只有标题没有内容”或“只写了意图没真正做成”的情况。

### 3.2 我做了独立的一致性核查

按你的要求，我没有重跑任何长实验，但做了足够强的只读验证。

我独立核查了这些点：

- `authoring_manifest.json` 是否能解析；
- `T75-FIG-M01`、`T75-FIG-M02`、`T75-FIG-A01` 是否都映射回了上游 `T74-*` stable IDs；
- `asset_source_map.csv` 的资产 ID 集合是否与 manifest 一致；
- source map 里的证据路径是否存在；
- 三张 SVG 是否真实存在、含有 `<svg`、并能被 XML 正常解析；
- `paper_maintext_results_authoring_pack.md` 是否真的按任务要求给出了上游 ID、safe wording、forbidden wording；
- `paper_authoring_do_not_write_list.md` 是否覆盖了 `T48`、`T49/T71/T72`、`FR8`、`T74-FIG-04` 等关键 overclaim 风险；
- `runs/`、`artifacts/`、`cnn_fpga/`、`physics/`、`benchmark/`、`tests/` 是否被改动。

核查结果是：

- 三个 `T75-FIG-*` 资产都能稳定回链；
- manifest 和 source map 一致；
- 三张 SVG 结构上成立；
- 文档内容和 stable-ID 路线一致；
- 没有发现越权改代码、改测试、改历史 run 或 artifact 的情况。

### 3.3 没有把 authoring 工作写成证据升级

这次 review 最重要的一点，是确认 `T75` 只是“压缩和锁定表述”，不是“借写作顺手升级事实”。

我重点确认了以下边界仍然被保留：

- `T48` 仍只是 isolated current-host true `.tflite` runtime；
- `T49/T71/T72` 仍只是 read-only real-board gate / regeneration / provenance，current-host 仍是 `NO_GO`；
- `FR8` 仍只是 extension lane，不能写成 promoted mature comparator；
- `T74-FIG-04` 仍是 blocked slot，不能被补画成统一 portability/deployment closure 图；
- `FR6/FR7` 仍只是 descriptive / bounded explanation support，不能写成 causal closure 或 teacher necessity proof；
- `T24` 仍只是 locked frozen-set software-HIL 排名，不是 expanded benchmark。

这说明 `T75` 做到的是“把话写准”，而不是“把证据写大”。

## 4. 为什么我没有把“未做真实渲染预览”当成阻塞项

这次我确实额外检查了一点：三张 SVG 有没有经过真正的渲染预览。

结论是：

- 结构级验证是成立的；
- 但我没有拿到一份明确的“已渲染并人工查看”的证据；
- 我也尝试用本机无头浏览器做临时 PNG 渲染抽查，但当前环境的 GPU 进程在 Edge/Chrome 下都失败了，所以没法独立补上这个视觉验证。

我没有因此给 `BLOCK`，原因有两个：

- 任务包本身要求的验证，重点是 manifest/source-map/SVG 结构和边界表述，不是投稿级的最终视觉 QA；
- 这不影响判断 `T75` 是否完成了“bounded authoring pack”这件事。

但我把它写成了 non-blocking issue，因为一旦后续真的进入期刊排版或投稿包整合，就仍应补一次真实渲染预览，确认：

- 字体和符号是否正常；
- 文本是否重叠；
- 图例与颜色是否可读；
- 版式是否适合目标模板。

## 5. Worker 已写文档的情况：有没有问题，补充了什么

Worker 原先写的 `docs/review/T75_review.md` 和 `docs/for_human/T75_explanation.md` 总体方向是对的，没有明显事实性错误。

但我认为它们还缺两点 reviewer 视角上的强化：

- 对“真实渲染预览未验证”的说明不够正式；
- 对“为什么这次可以判 PASS”的论证还不够明确地区分“完成 authoring pack”和“完成投稿排版”。

所以我做的不是推翻，而是补强：

- 把 review 文档改成更正式的 reviewer 口径；
- 明确写出 non-blocking issue；
- 把结论聚焦到“交付完成、回链完整、边界诚实、无越权 diff”；
- 同时把“未做真实渲染预览”保留成一个诚实的后续验证建议。

## 6. 一句话总结

`T75` 已经把 `T74` 的 stable-ID 材料包推进成一套真正可用于主文 Results 写作的 authoring pack：有 prose、有 final figure、有 caption lock、有 appendix bridge，也有明确的 do-not-write guardrail；我给 `PASS`，是因为它完成的是“写作收口”，并且没有借写作去偷升证据等级。
