# T65 解释文档

## 1. 这个任务在做什么

`T65` 不是新实验任务，而是一个“结果包收口任务”。

前一轮 `T64` 已经跑出了一个 bounded 的 FR8 extension-lane 结果：`statcalib` 作为第六条 lane，在锁定四场景里赢了冻结五模式历史冠军。但 `T64` review 也指出了两个很具体的问题：

1. 报告里对执行形态的措辞比 artifact 真正能证明的范围更大。
2. 报告把一个来自文件系统时间戳的信息，误写成了 `summary.json` 自带字段。

此外还缺第三样东西：一个自动化小检查器，去防止以后继续把 `T64` 报告和真实 artifact 写偏。

所以 `T65` 的任务不是“再证明一次 `statcalib` 很强”，而是：

- 把 `T64` 报告改得更严谨
- 加一个自动审计 helper
- 加 focused tests
- 出一份一致性 audit 文档

## 2. 这个任务为什么重要

这个项目当前处在 `Research Reality Recovery Mode`。在这个模式下，最重要的不是把结果说得更大，而是把边界说得更准。

`T64` 已经把实验结果跑出来了，但如果报告写法本身不够严谨，那么以后别人引用 `T64` 时，还是可能会出现两种问题：

1. 把 artifact 证明不了的事情写成 artifact 已经证明了。
2. 把 extension lane 误读成重写了 `T24` 冻结主表，或者误读成更高等级的 comparator 结论。

`T65` 的意义就在这里：它不增加新 benchmark evidence，但它提高了“现有 benchmark evidence 能否被安全复用”的可信度。

## 3. Worker 实际做了什么

如果只看 T65 任务本地允许范围内的内容，Worker 这次主要做了四件事。

第一件，是修 `docs/fr8_statcalib_extension_lane_benchmark.md`。

这次修正把 `T64` 报告里的 provenance 和 execution-shape 表述重新写严谨了：

- 不再把执行形态说成 artifact 无法证明的 `detached one-shot`
- 不再写“finish timestamp from `summary.json`”
- 明确区分：
  - artifact-recorded fields
  - observed outside preserved artifacts
  - auxiliary filesystem metadata

第二件，是新增审计 helper：

- `cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`

这个 helper 会直接读取：

- `T64` 的 `summary.json`
- `launch_plan.json`
- `progress.jsonl`
- `comparison.csv`
- `T24` 的 `comparison.csv`
- `T64` task package
- `T64` report

它检查的重点包括：

- 报告措辞是否和真实 artifact 对齐
- 四场景、五个 frozen mode 加 `statcalib` 的边界是否漂移
- `paired_seeds=true`、`repeats=2` 是否保持
- `progress.jsonl` 是否有重复 `running`
- `T64` 的冻结子表是否仍和 `T24` 完全一致
- 报告是否继续保留 mock-backed、separate extension lane、not `.tflite`、not real-board 等边界句子

第三件，是新增 focused tests：

- `tests/test_fr8_extension_lane_consistency.py`

这些测试不是大而全，而是专门盯这次任务要防的回归：

- 错误 execution wording 会不会被抓住
- 错误 provenance wording 会不会被抓住
- duplicate `running` 会不会被抓住
- 当前保留的真实 `T64/T24` artifact 集是否能通过 full audit

第四件，是新增一份显式审计文档：

- `docs/fr8_statcalib_extension_lane_consistency_audit.md`

它把“审了什么、怎么审的、结果是什么、以后哪些边界必须继续保留”写清楚了。

## 4. 我为什么一开始给了 `BLOCK`，后来又改了

我一开始给 `BLOCK`，不是因为 T65 本地实现坏了，而是因为我按“当前整份 diff”审时，看到了很多超出 T65 Allowed Files 的额外改动，比如：

- `docs/follow-up_plan/**`
- `docs/汇报用/**`

而 `docs/follow-up_plan/**` 在 T65 任务包里还是显式 Forbidden Scope。只按 git 面看，这会让我把本次提交理解成“混入了越界文件的 T65 diff”，所以只能先 block。

但你后来补充说明：这些额外 diff 你已经单独审核过了，不应该再算成这次 T65 的阻塞项。

这个澄清很关键。它改变的不是 T65 技术内容，而是 review 的边界定义：

- 原先我是按“当前 mixed diff 整体”判
- 现在是按“只看 T65 task-local 内容”判

在这个新前提下，T65 的本地内容是可以接受的。

## 5. 为什么最终结果是 `PASS_WITH_WARNINGS`

我最后没有给纯 `PASS`，而是给了 `PASS_WITH_WARNINGS`，原因很简单：

T65 的技术内容本身基本过关，但这个结论依赖了你的补充说明，而不是单靠当前 git diff 的纯净性就能独立得出。

所以 warning 的核心不是“代码还有明显问题”，而是：

- 当前接受结论带有一层外部边界澄清
- review 不是仅凭提交面本身就能完全自证

这和以前某些任务里出现的“混入了已知但非本任务的改动，需要用户或 Captain 明确说明归属”是同一类情况。

## 6. 有没有伪实现、mock、stub、hardcode

这次没有发现伪实现。

新增的 helper 不是把 `PASS` 写死，也不是只输出漂亮文本。它真的去读取 artifact 和报告，再根据检查结果返回成功或失败。这个层面上，它是一个真实的 guard。

但它确实是有意做窄的：

- 预期 scenarios 是写死的
- frozen modes 顺序是写死的
- 某些报告 required phrases 也是写死的

这不算坏事，因为 T65 本来就不是做通用框架，而是做 `T64 closeout guard`。只是要明白：

- 它不是 fake
- 但它也不是“以后所有 FR8 lane 通吃的通用系统”

## 7. 还缺什么

对 T65 本身来说，没有阻塞级别的缺测。我实跑了三项轻量验证：

- `python -m unittest tests.test_fr8_extension_lane_consistency`
- `python -m py_compile cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`
- `python -m cnn_fpga.benchmark.audit_fr8_extension_lane_consistency ...`

都通过了。

但仍有两个可以继续补的非阻塞点：

1. 增加 synthetic failure test，专门构造 `frozen_subset_matches_t24` 失败路径。
2. 增加 synthetic failure test，专门构造 boundary phrase 缺失路径。

这两项不是当前必须补的 blocker，但如果以后继续扩这个 helper，值得加上。

## 8. 这个任务对后续开发的意义

T65 的真正价值不是“让 T64 结果变得更强”，而是“让 T64 结果变得更难被误用”。

它对后续开发的意义主要有三点：

1. 以后再引用 `T64` 时，报告和 artifact 不一致的风险更低了。
2. `T24` 冻结主表和 `T64` extension lane 的边界，被代码级 guard 又钉了一次。
3. 项目继续坚持了当前最重要的治理原则：结果可以强，但表述必须诚实。

所以最准确的说法是：

`T65` 没有新增实验新证据，但它新增了结果包的一致性护栏，使 `T64` 更适合被当作一个 self-audited、bounded、mock-backed software-HIL extension-lane artifact 来复用。
