# T49：这次任务在做什么，以及为什么 review 是 `PASS_WITH_WARNINGS`

## 1. 先用通俗的话解释这个任务

`T49` 不是去“跑真板实验”，而是先回答一个更基础的问题：

“当前这台机器，究竟有没有资格进入真板 smoke？”

这个问题之所以要单独做，是因为前一项 `T48` 只证明了软件侧 `.tflite` runtime 在一个隔离环境里可以真实加载和执行，但这离“真板能跑”还差很远。根据 `docs/02_experiment_plan.md` 的主口径，当前仓库的状态一直是：

- `P3 软件 HIL` 已完成
- `P3 真板 HIL` 未完成
- `board_backend.py` 仍然是 placeholder

所以 `T49` 的作用，不是补一个“看起来像真板”的故事，而是把“为什么现在还不能诚实地说真板可跑”做成一份有证据的 gate 包。

## 2. 这次实现具体做了什么

### 2.1 任务目标

结合 `docs/04_task_board.md` 和 `docs/07_handoff.md`，`T49` 是 `T48` 之后的下一步 deployment-boundary 收口任务。它要把当前宿主上的真板前提拆成四层：

1. `host_environment`
2. `device_path_truth`
3. `bitstream_and_contract_truth`
4. `repo_execution_path_truth`

然后明确给出一个 gate verdict，而不是模糊说“未来也许可以跑真板”。

### 2.2 本次交付的文件

worker 主要交付了 3 类东西：

1. 证据产物
   - `artifacts/t49_real_board_smoke_execution_gate/host_fact_manifest.json`
   - `artifacts/t49_real_board_smoke_execution_gate/device_path_probe.json`
   - `artifacts/t49_real_board_smoke_execution_gate/code_side_audit.json`
   - `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`
2. 逻辑与测试
   - `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
   - `tests/test_t49_real_board_smoke_execution_gate.py`
3. 解释文档
   - `docs/t49_real_board_smoke_execution_gate.md`
   - `docs/worker_summary/T49_worker_summary.md`
   - `docs/review/T49_review.md`
   - `docs/for_human/T49_explanation.md`

### 2.3 实现流程

这次实现的核心流程其实很清楚：

1. 先记录当前宿主事实
   - Python 路径与版本
   - Windows 版本
   - 主板/BIOS 标识
   - repo 里现存的 board / bitstream 默认配置
2. 再检查当前机器上是否真的有候选设备路径
   - `/dev/uio0`
   - `/dev/uio1`
   - `\\\\.\\XilinxDMA`
   - `\\\\.\\XDMA`
   - `\\\\.\\uio0`
   - `\\\\.\\uio1`
3. 再审代码侧已知事实
   - `axi_map.py` 里的寄存器地址和 mask
   - `dma_client.py` 里的 buffer / byte-count / histogram 约定
   - `board_backend.py` 是否仍是 placeholder
4. 最后由 helper 聚合出 gate verdict

也就是说，这次不是在“跑硬件”，而是在“核对进入硬件前必须先成立的事实”。

### 2.4 当前收口出的事实

当前最关键的事实是：

- 宿主能识别出来，解释器和 OS 信息都能记录
- 仓库里确实有 `board/real` 入口、AXI 地址表和 DMA 结构约定
- 但当前这台 Windows 主机上，没有发现任何可读打开的真板候选设备路径
- 另外，bitstream / RTL / DMA contract 也还没有和当前宿主绑定确认
- `board_backend.py` 仍然是 placeholder-only execution path

所以最终 verdict 是：

`NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

这个结论非常重要，因为它是“诚实的 NO_GO”，而不是“任务失败”。它把当前最先卡住的层清楚指出来了：不是文档问题，不是口头计划问题，而是 host/device 这一层根本没有成立。

## 3. 代码和配置变化的意义

这次任务的意义，不是让仓库多了一段新硬件功能代码，而是让“真板仍未具备前提”这件事第一次有了结构化证据。

结合 `docs/02_experiment_plan.md`、`docs/real_board_smoke_execution_plan.md`、`docs/review/T20_review.md`、`docs/review/T22_review.md` 来看，前面的工作更多是在做：

- readiness checklist
- execution plan
- placeholder 边界说明

而 `T49` 往前迈了一步：它不再停留在“应该检查什么”，而是对“当前这台机器查出来的事实是什么”给了一个任务包级别的落点。

对后续开发的意义主要有三点：

1. 它把“真板未完成”的原因从泛泛而谈，收敛成了可引用的四层 gate。
2. 它给未来真板任务设了一个更严格的起点。
   - 后续任务不能再跳过 host/device 事实，直接讨论 benchmark 或 HIL promotion。
3. 它保护了 Phase 2 的边界。
   - 不会因为已经有 `.tflite` 真执行证据，就顺势把真板语义也写成既成事实。

## 4. 为什么这次 review 不是 `PASS`，而是 `PASS_WITH_WARNINGS`

我给 `PASS_WITH_WARNINGS`，不是因为 worker 把事情做错了，而是因为它虽然完成了任务，但还有两处值得明确记账的工程问题。

### 4.1 任务完成度本身是够的

先说为什么不是 `BLOCK`：

- helper、测试、artifact、主报告都交付了
- `python -m py_compile` 和 `python -m unittest` 能通过
- 当前宿主的 Python / Windows / device path 事实，我做了轻量复核，和 artifact 一致
- `board_backend.py`、`axi_map.py`、`dma_client.py` 没有被改
- 没有 `runs/` 修改
- 没有治理文档越界修改
- 没有证据表明执行了 MMIO 写、DMA 写、寄存器写或真板 smoke
- 文档没有把结果写成 `real-board validated`、`P3 真板 HIL complete` 或 `deployment closure`

所以它满足了“有界完成 + 边界诚实”的要求。

### 4.2 为什么有 warning

第一处 warning 在 helper 逻辑本身：

- `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py` 现在把 device 层 ready 条件写成“可只读打开的路径数量 >= 2”
- 但 probe 里其实已经有 `role` 字段，区分了 `mmio` 和 `dma`
- 真正更稳妥的判定应该是：至少 1 条 `mmio` 路径和 1 条 `dma` 路径都成立

当前真实 artifact 的 `openable_count = 0`，所以这不会改变本次 `NO_GO`。但如果以后继续复用这段 helper，这个判定会偏松。

第二处 warning 在再生性：

- 本次最关键的三个 probe JSON 已经交付
- 但 diff 里没有把它们的生成入口也一并 check in
- 也就是说，仓库里现在更像是“交付了事实包结果”，而不是“交付了完整的 probe 生成链”

这不等于伪实现，因为我已经对关键事实做了独立交叉验证，确认它们不是编造出来的；但它确实意味着，后续若要重复生成同类事实包，仍会依赖人工命令或临时脚本，而不是 repo 内统一入口。

## 5. Worker 已写文档有哪些是对的，哪些需要补充

### 5.1 已有文档基本是对的

worker 原本写的：

- `docs/t49_real_board_smoke_execution_gate.md`
- `docs/worker_summary/T49_worker_summary.md`
- 原 `docs/review/T49_review.md` 预审草稿
- 原 `docs/for_human/T49_explanation.md`

整体上都保持了一个关键优点：边界很稳，没有把“只读事实探测”写成“真板已打通”。

尤其是这几点写得是对的：

- 当前最先触发的是 host/device 层 `NO_GO`
- `fpga_linear_v1` 只是 config-level record，不是当前宿主可执行 bitstream 证据
- `board_backend.py` 仍然是 placeholder-only
- service name clues 不能当成板卡存在证据

### 5.2 这次补充的地方

我这次主要补了两点：

1. 把 `T49` 放回到更大的主线语境里解释清楚
   - 它为什么跟在 `T48` 后面
   - 它和 `T20`/`T22` 的关系是什么
   - 它对 Phase 2 受控开发有什么意义
2. 把 warning 记清楚
   - helper 的 device-role 判定偏松
   - probe artifacts 的再生链还不完整

这两点不是为了否定 worker 的工作，而是为了让后续接手的人更清楚：

- 现在已经知道什么
- 还不能声称什么
- 下一步应该补哪里，而不是误以为只差“再跑一次真板命令”

## 6. 建议如何继续

如果后续还要推进真板方向，我建议把下一步拆得很小，只做下面两类事情之一：

1. 小型工程补强任务
   - 修正 helper 的 device-role 判定
   - 补针对 checked-in artifacts 的回归测试
   - 补一个只读 probe 生成入口，提升再生性
2. 新的 bounded real-board execution task
   - 前提是目标宿主上真实出现设备节点
   - 并且 bitstream / RTL / DMA contract 能与当前宿主绑定

在这两个前提没成立前，不应该把 `T49` 往“真板 smoke 已执行”方向继续扩写。
