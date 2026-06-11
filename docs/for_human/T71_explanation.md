# T71：真板 gate 再生成包到底补了什么，以及为什么 review 是 `PASS_WITH_WARNINGS`

## 1. 先用大白话解释这个任务

`T49` 已经把一个很关键的问题说清楚了：

“当前这台 Windows 机器，不能诚实地说已经具备真板 smoke 前提。”

它给出的结论不是“未知”，而是明确的：

`NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

但 `T49` 还有一个现实缺口：

- 它更像是“这一次任务，把当前宿主查明白了”
- 还不够像“以后换一台候选真板宿主，也能用同一套 repo 内入口重新检查”

`T71` 做的就是这件事：

把 `T49` 的一次性 current-host gate 结果，往前推进成一个 checked-in、可 replay、可 regeneration 的只读 gate 包。

换句话说，`T71` 不是去跑真板，而是把“检查真板前提”的方法本身做得更稳定、更可迁移。

## 2. 这个任务为什么会排在现在

结合 [04_task_board.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/04_task_board.md) 和 [07_handoff.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/07_handoff.md) 的当前状态，主线已经把 `T49` 定义为：

- 当前宿主真板 gate 的第一次诚实收口
- 结论是 current-host `NO_GO`
- 不允许据此直接打开 `T37`

所以 `T71` 成为下一步是合理的，因为当前最重要的问题已经不再是：

- “这台机器到底有没有查过？”

而是：

- “以后换宿主时，有没有一套 repo 内可复用的、不会过度乐观的 read-only gate 入口？”

这和 [02_experiment_plan.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/02_experiment_plan.md) 的长期边界也是一致的。主线一直在强调：

- `P3 软件 HIL` 已完成
- `P3 真板 HIL` 未完成
- `board_backend.py` 仍然是 placeholder

所以 `T71` 的正确方向，必须是“加固 gate 再生成能力”，而不是“假装更接近真板已打通”。

## 3. 这次具体改了什么

### 3.1 第一件事：把 device 判定从“数数量”改成“看角色”

`T49` review 里已经指出一个 warning：

- 如果只用“可打开路径数量 >= 2”来判定 device 层 ready
- 那未来可能出现“两条路径都能打开，但其实都属于同一类角色”的误判

这次 `T71` 的核心代码改动就在 [build_t49_real_board_smoke_gate.py](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/hwio/build_t49_real_board_smoke_gate.py:129)：

- 现在会分别统计：
  - `openable_mmio_paths`
  - `openable_dma_paths`
  - `openable_unknown_role_paths`
- ready 条件变成：
  - 至少 1 条 `mmio`
  - 且至少 1 条 `dma`

这意味着：

- “两条都是 `dma`” 不再能误判 ready
- “一条 `mmio` + 一条 `dma`” 才算真正满足设备层前提

这一步很重要，因为真板前提不是“有两条路径”，而是“必须同时有寄存器侧入口和 DMA 侧入口”。

### 3.2 第二件事：新增 checked-in 的只读 collector

这次新增了 [collect_t71_real_board_gate_artifacts.py](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py:1)。

它的作用是统一生成三份 gate 输入：

1. `host_fact_manifest.json`
2. `device_path_probe.json`
3. `code_side_audit.json`

也就是把原先更像“任务期间人工收集”的事实包，推进成 repo 内一个可直接运行的只读入口。

这一步对后续开发的意义很直接：

- 以后如果换到 future-host，不需要重新发明一套 probe 方式
- 可以直接复用主线已有 collector + gate helper
- 这样“真板前提检查”就更像一个工程入口，而不是一次性 task 结果

### 3.3 第三件事：补 focused tests

测试这次补得也比较对路。

[test_t49_real_board_smoke_execution_gate.py](/D:/Codes/Quantum/DriftAdaptiveQEC/tests/test_t49_real_board_smoke_execution_gate.py:159) 新增了同角色误判保护测试：

- 两条都是 `dma`，不能判 ready
- 一条 `mmio` + 一条 `dma`，才允许 ready

[test_t71_real_board_gate_regeneration_pack.py](/D:/Codes/Quantum/DriftAdaptiveQEC/tests/test_t71_real_board_gate_regeneration_pack.py:15) 则验证了：

- 回放 `T49` checked-in artifacts 时，verdict 仍然不漂移
- 当前宿主重新生成 artifacts 后，再跑 helper，结论仍然和 `T49` 一致

这说明 `T71` 没有把事情做成“又一套平行逻辑”，而是明确围绕 `T49` 的历史结论做加固。

## 4. 这次产出的 artifacts 和文档在回答什么

`artifacts/t71_real_board_gate_regeneration_pack/` 里这次最重要的几份文件分别回答了不同问题：

- `host_fact_manifest.json`
  - 当前宿主是谁、解释器是什么、repo 默认 board/bitstream 记录是什么
- `device_path_probe.json`
  - 当前宿主上到底有没有 `mmio + dma` 双角色路径
- `code_side_audit.json`
  - AXI、DMA、placeholder 语义现在在代码里到底长什么样
- `current_host_regenerated_gate.json`
  - 当前宿主重新生成后的最终 gate verdict
- `t49_checked_in_replay_gate.json`
  - 用 `T49` 历史 artifacts 回放后的 gate verdict
- `replay_vs_regeneration_comparison.json`
  - 回放和再生成是否一致

而主报告 [t71_real_board_gate_regeneration_pack.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/t71_real_board_gate_regeneration_pack.md) 则把这些结果收成一句主结论：

- `T71` 没有改变 `T49` 的 current-host `NO_GO`
- 它只是把这个 `NO_GO` 变得更可再生成、可迁移、可复核

## 5. 为什么我的 review 不是 `PASS`，而是 `PASS_WITH_WARNINGS`

### 5.1 为什么不是 `BLOCK`

先说为什么我没有 block：

- 任务要求的主要事情都做了
- role-aware 逻辑确实落到了代码里
- 同角色误判问题确实被测试防住了
- 当前宿主 regeneration 和 `T49` replay 结论一致
- unit tests 真实通过
- collector 真实可运行
- 没有触碰 forbidden files
- 没有 write-side MMIO / DMA / register 行为
- 文档没有把任务写成真板执行成功、真板 ready 或真板 HIL 完成

所以从任务完成度来说，`T71` 是成立的。

### 5.2 为什么有 warning

warning 的核心不在主 verdict，而在 provenance 的严谨度。

这次最值得记账的问题出现在 [collect_t71_real_board_gate_artifacts.py](/D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py) 里：

1. `probe_limitations` 是固定写进去的
   - `278-283` 直接写了几条 “access denied” 文案
   - 但 collector 本身并没有实际去执行这些探针
   - 这意味着它把“旧任务里观察到的限制”写成了“当前 / future-host 重新收集出来的事实”
2. `source_records` 是固定写当前默认配置的话术
   - `367-370` 直接写死了 `hardware_hil.yaml: hil.board=ZCU111`
   - 也写死了 `hil.bitstream_version=fpga_linear_v1`
   - 但脚本自己又支持 `--config`
3. `expected_byte_count_basis` 也是固定文案
   - `473` 直接写了 `32 x 32 float32 histogram -> 4096 bytes`
   - 如果 future-host 用的是另一份 config，这段说明可能就不再完全准确

这几个问题都不会改变当前最重要的事：

- 当前 verdict 仍然是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

但它们会削弱一个更细的承诺：

- “这是不是一个已经足够严谨、足够成熟的 future-host host-transfer pack”

所以我给的是 `PASS_WITH_WARNINGS`，不是 `PASS`。

## 6. 这算不算伪实现、mock、stub 或 hardcode

不算伪实现，也不算 mock/stub。

原因很简单：

- collector 是真的会去收集 host 信息、跑 `pnputil`、检查候选路径、生成 JSON
- helper 也是真的会去读取这些 JSON 并生成 gate verdict
- tests 也是真的会跑

所以它不是“看起来像做了，但实际上没有做”。

但它确实存在 hardcode，而且是“说明文字层面的 hardcode”：

- 某些 provenance 描述写死了当前默认配置
- 某些 probe limitations 不是实时探测，而是固定文案

这类问题更适合记成 warning，而不是把整个任务打成伪实现。

## 7. Worker 已写的说明哪里是对的，哪里需要补充

worker 原先写的：

- [T71_worker_summary.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/worker_summary/T71_worker_summary.md)
- 原 `docs/review/T71_review.md`
- 原 `docs/for_human/T71_explanation.md`

大方向是对的，尤其是这几点：

- `T71` 没有改变 `T49` 当前宿主 verdict
- replay 与 regeneration 一致
- future-host 现在至少有两步命令入口
- `T37` 仍然应该 blocked

但原文档里少记了一件重要的 reviewer 视角问题：

- collector 中有几处 provenance 字段是写死的，不是动态探测出来的

这不推翻 worker 的主结论，但如果不把它写出来，后续接手的人很容易误以为：

- “这已经是一个完全成熟的 future-host 迁移包了”

而我认为更准确的说法应该是：

- “它已经是一个可用的再生成入口，但 provenance 严谨度还没完全收口”

## 8. 对后续开发意味着什么

`T71` 的价值不是让仓库离真板执行“更近一大步”，而是让真板前提检查更工程化了一步。

它对后续开发的实际意义主要有三点：

1. `T49` 的 current-host honest `NO_GO` 不再只是历史任务结果，而变成了可回放、可再生成的 gate 包。
2. future-host 如果真的出现了候选板卡宿主，现在至少有一套 repo 内入口，可以先做只读前提核查。
3. `T37` 为什么还不能开工，边界更清楚了。
   - 不是因为没人查过
   - 而是因为：
     - `device_path_truth` 还没成立
     - `bitstream_and_contract_truth` 还没成立
     - `repo_execution_path_truth` 还没成立

## 9. 接下来最合理的动作是什么

如果还要继续推进这一条 lane，最合理的下一步不是直接开 `T37`，而是单开一个很小的后续任务，只收两件事：

1. 去掉 collector 里的写死 provenance 文案
   - 让 `probe_limitations`、`source_records`、`expected_byte_count_basis` 都从当前执行上下文真实生成
2. 补 future-host 相关回归
   - 特别是 `--config`、`--mmio-path`、`--dma-path` 的非默认路径测试

在这之前，最强可支持的结论仍然应该是：

- 仓库已经有一个 checked-in、只读、role-aware、可 replay / regeneration 的 real-board gate 包
- 当前宿主重放后仍然是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- 这仍不等于真板执行成功、真板 ready、`P3 real-board HIL complete` 或 deployment closure
