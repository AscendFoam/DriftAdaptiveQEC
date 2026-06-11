# T72 任务解释与本次 Review 说明

## 1. 先用通俗的话解释这个 task

`T49` 回答了一个很朴素的问题：这台当前宿主机能不能直接进入真板 smoke？答案是不能，结论是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。

`T71` 又往前走了一步：它把这套结论整理成一套 checked-in、只读、role-aware、可 replay / regeneration 的 gate 包，方便 future-host 复核。

`T72` 再做的事情，不是“把真板跑起来”，而是“把这套 gate 包上的说明写得更诚实、更可迁移”。可以把它理解成：

- `T49` 像是第一次体检，确认“现在还不能上场”
- `T71` 像是把体检单收成一个标准档案包
- `T72` 像是把体检单上的注释、来源、默认值说明都写严谨，避免以后别人误把旧注释当成这次真实检查结果

所以，`T72` 的价值是 provenance hardening，不是 execution success。

## 2. 这个实现到底做了什么

从 `docs/04_task_board.md`、`docs/07_handoff.md` 和 `docs/02_experiment_plan.md` 看，`T72` 的定位非常明确：它是 `T71` 留下的 `R31` 收口任务，只处理 read-only real-board gate pack 的 provenance 问题，不解锁 `T37`，也不允许扩写成真板执行成功。

本轮实现主要做了三件事。

第一，collector 不再把 probe 限制写成固定文案，而是把每条 probe 的实际状态结构化记录下来。现在 `host_fact_manifest.json` 里有：

- `probe_execution_records`
- `probe_limitations`

并且每条 probe 会被明确标记为：

- `ok`
- `command_failed`
- `not_applicable`

这解决了 `T71` 的一个核心问题：以前 reviewer 看到 `access denied` 一类文案，无法立刻判断这到底是本次真的探测出来的，还是沿用了旧观察；现在这个歧义被明显缩小了。

第二，config/path provenance 变成了动态导出，而不是默认话术。`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py` 现在会把下面这些信息一起写进 artifact：

- 当前实际使用的 `config_path`
- 这是默认 config 还是 `--config` override
- `candidate_mmio_path` / `candidate_dma_path` 的 effective value
- 对应路径来自 config 还是来自 CLI override
- `bitstream_evidence.source_records` 的真实 config 路径和值

这意味着 future-host 如果改了 `--config`、`--mmio-path` 或 `--dma-path`，artifact 里不仅“值会变”，连“来源说明”也会跟着变。

第三，`expected_byte_count_basis` 不再写死成默认配置文案，而是根据当前 config 里的 `histogram_shape`、`dtype`、`buffer_bytes` 动态推导。这样 future-host 如果换了 buffer 规模或 dtype，不会再出现“数字是新的，但解释还是旧的”这种 provenance 漂移。

## 3. 它对后续开发有什么意义

这个 task 的意义，不是让仓库突然具备了真板执行能力，而是让真板 gate 包更适合被 future-host 复核。

它的直接价值是：

- future-host 更容易看清哪些事实真的是这次执行探测到的
- future-host 更容易看清哪些值来自默认 config，哪些来自 override
- current-host regeneration 与 `T49` checked-in replay 仍保持同一个 honest `NO_GO`

这和计划文档里的定位完全一致：`docs/02_experiment_plan.md` 已把 `T49/T71/T72` 统一归到 real-board gate / provenance 边界，而不是 real-board execution success；`docs/04_task_board.md` 和 `docs/07_handoff.md` 也都明确写了 `T37` 仍然 blocked。

换句话说，`T72` 让“不能跑真板”这件事的证据变得更干净，但没有把“不能跑”变成“能跑”。

## 4. 为什么我的 review 结果是 `PASS_WITH_WARNINGS`

我没有给 `BLOCK`，因为 T72 任务包要求解决的主问题，确实基本都解决了：

- `probe_limitations` 已经从固定字符串变成执行派生记录
- `source_records` / `repo_board_defaults` / `expected_byte_count_basis` 不再写死默认口径
- `--config` / `--mmio-path` / `--dma-path` 的 focused regression 已补
- 我复跑的编译、单测、collector 再生成、gate helper 对比都表明 verdict 没漂移，仍然是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- 文档整体也没有把本轮结果写成 real-board ready 或 execution success

我也没有给纯 `PASS`，因为还留着一个小但真实的 provenance 边角：

- 当前路径 provenance 还不能区分“YAML 里明确写了这个路径”和“配置文件没写，代码默认补成了 `/dev/uio0` / `/dev/uio1`”

我专门用一个临时 config 去掉 `hil.board_io.axi_uio_path` / `dma_buffer_path` 做了轻量复核，artifact 仍会把这两条记录写成 `source_kind=config_field`。这不会推翻 T72 已完成的主体目标，但说明它对 future-host 的“最小 config 回退场景”还不算完全无歧义，所以我给的是 `PASS_WITH_WARNINGS`。

## 5. Worker 已有 review / explanation 文档有没有问题

总体方向是对的，而且最关键的一点没有写错：Worker 现有文档一直把 `T72` 描述成 provenance hardening，而不是 real-board success。这一点很重要，也是我没有 blocking 的主要原因之一。

我这里主要补充两点：

- 补出一个更细的残余风险：代码默认值回退场景下，path provenance 仍可能比真实来源更乐观
- 补出一个更具体的后续建议：如果未来真的要把这套 transfer-pack 交给更多宿主机迁移使用，最好再补一个极小 follow-up，把 `config_field_present` / `code_default` 这类来源状态补齐

另外，Worker 原始提交里主报告最开始落在 `docs/t72_real_board_transfer_pack_provenance_hardening.md`；当前 `HEAD` 已经通过后续文档整理把它归到 `docs/evidence_packs/deployment_boundary/`。所以从现在的仓库状态看，文档落点已经正常，但从“原始 worker diff”角度看，这一点仍值得保留为 warning，而不必升级成 block。

## 6. 一句话总结

`T72` 不是把真板门打开了，而是把“这扇门为什么还打不开”的证据包写得更诚实、更适合 future-host 复核了。
