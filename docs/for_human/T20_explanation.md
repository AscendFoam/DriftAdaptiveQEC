# T20: Real-Board HIL Readiness Checklist — 通俗解释与 Review 说明

## 一、这个 Task 在做什么？（通俗版）

### 背景

这个项目的终极目标之一是：把训练好的 AI 模型部署到真实的 FPGA 板卡上，让它在硬件上实时运行量子纠错解码。

为了这个目标，项目里已经写好了一些"真板接口"代码（`board_backend.py`、`fpga_driver.py`），但这些代码目前只是**空壳**——它们定义了接口形状，但内部实现要么返回占位数据，要么抛出"设备缺失"错误。

这就像你给房子画好了门的设计图纸，门框装上了，但门本身还没装——推一下就看到墙后面什么都没有。

### 问题

现在有三件事绝对不能搞混：

1. **Mock FPGA**（`mock_fpga.py`）：纯软件模拟，不需要任何硬件。项目当前的所有测试都用的这个。
2. **真板 Backend**（`board_backend.py`）：设计给真实 FPGA 用的接口，但目前是空壳。
3. **`fpga_driver.py`**：统一入口，能根据配置选择走 mock 还是真板。但真板路径一走就报错（因为设备不存在）。

如果在文档或论文里把 mock 的测试结果写成"真板已通过"，那就是严重的学术不诚实。

### T20 做了什么

T20 的任务就是：**不碰代码，只写一份"真板准备清单"**。具体来说：

- 清点当前真板相关代码的占位证据（证明它确实是空壳，不是已完成）
- 列出如果要推进真板，需要补齐哪些前置条件（设备、地址、权限、日志等）
- 制定最小 smoke 测试的验收标准（分四层，逐步递进）
- 列出绝对不能写的禁止表述

用一句话总结：**T20 是一份"真板路线路勘报告"，搞清楚从空壳到真实验收之间还差什么。**

## 二、任务实现详细解释

### 2.1 任务目标

为真板 HIL 路径补 readiness checklist 与缺口清单，明确当前 placeholder 证据、真板前置条件、最小 smoke 验收标准与禁止表述。不实现真板 backend，不改写 placeholder 语义。

### 2.2 任务流程

1. **只读代码审计**：Worker 读取了 `board_backend.py`、`fpga_driver.py`、`run_hil_suite.py` 和 `docs/03_hil_p4_boundary_audit.md`，定位了所有 placeholder 证据点。

2. **撰写 readiness checklist**：产出 `docs/real_board_hil_readiness.md`，包含 7 个章节：
   - Purpose（目的与边界）
   - Current Placeholder Evidence（3 个文件的占位证据点）
   - Required Preconditions（4 类前置条件：设备拓扑、权限、驱动/寄存器、运行时日志）
   - Minimum Smoke Acceptance Standard（4 层递进验收：A 设备存在 → B 寄存器活性 → C DMA 读出 → D commit/ack）
   - Required Evidence Pack（未来真板任务应产出的 9 类证据）
   - Prohibited Wording（5 条禁止表述 + 3 条允许表述）
   - Recommended Next-Step Boundary（后续任务应保持的 4 条限制）

3. **更新治理文档**：同步更新了 `03_hil_p4_boundary_audit.md`、`04_task_board.md`、`07_handoff.md`、`08_risks_and_open_questions.md`。

4. **刻意不做的事**：
   - 没有修改 `board_backend.py` 或 `fpga_driver.py`
   - 没有调用任何硬件命令
   - 没有写成真板已完成
   - T20 在 task board 上仍为 `[ ]`（等审核后才勾）

### 2.3 代码/配置文件的变化

**没有代码变更。** 没有配置文件变更。所有改动都在 `docs/` 目录内。

具体文件变化：

| 文件 | 变化类型 | 内容 |
|------|----------|------|
| `docs/real_board_hil_readiness.md` | 新增 | 真板 readiness checklist 全文（7 节，164 行） |
| `docs/tasks/Phase2/T20_real_board_readiness_checklist.md` | 追加 | Worker Output Summary 段落 |
| `docs/03_hil_p4_boundary_audit.md` | 修改 | 新增 T20 产出引用和推荐表述 |
| `docs/04_task_board.md` | 修改 | Current Unique Task 区域追加 Worker 只读产出 |
| `docs/07_handoff.md` | 修改 | 追加 T20 Worker 只读结果，更新状态 |
| `docs/08_risks_and_open_questions.md` | 修改 | 新增 R13（真板缺口高风险），新增 Q16 |

### 2.4 对后续开发的意义

1. **真板路径的"施工许可前置清单"**：现在有了一份精确的四层验收标准（A→B→C→D），后续任何真板执行任务都必须逐层补证，不能跳过。

2. **与 T3 边界审计形成互补**：T3 在 Phase 0 建立了 mock/real-board 的标签体系（`placeholder_real_board_backend`），T20 把这个标签的具体证据点和缺口清单全部固定下来。

3. **禁止表述清单防止误写**：第 6 节列出了 5 条明确禁止写的表述（如 `real-board HIL complete`），以及 3 条允许写的表述。这对后续论文写作和文档维护有直接约束力。

4. **Phase 2 最后一个有界任务**：T20 是 Phase 2 任务队列中的最后一个任务（Milestone 2D）。完成后，Phase 2 的四个 Milestone（证据增强、环境清单、仓库清理、硬件准备）将全部收口。

5. **风险文档同步**：R13（真板缺口）被标记为"高"风险，与 R3（HIL 边界易被误写，"高"）和 R12（TFLite 运行时不可用，"高"）形成三条高风险并列的格局，提示项目当前最关键的三个缺口。

## 三、为什么给出 PASS 的 Review 结果？

### 审查逻辑（Adversarial Mode）

本次是 **adversarial review**（对抗性审查），审查标准比普通 review 更严格。我从 6 个维度进行了检查，同时额外做了 5 项攻击性检查：

1. **任务是否真的完成** — 是。T20 目标是补 readiness checklist，不是实现真板。checklist 已产出，且覆盖了任务包要求的全部 5 项内容（placeholder 证据、前置条件、设备/地址/权限/日志、最小 smoke 验收标准、禁止表述）。

2. **是否有伪实现 / mock / stub / hardcode** — 无。本任务是只读文档任务。所有 placeholder 证据经我用 `grep` 独立验证，7 项占位符证据全部与源码逐字吻合：
   - `"Placeholder real-board backend..."` — `board_backend.py:1`
   - `"board_device_missing:..."` — `board_backend.py:168`
   - `"target_bank": None, "version": None, "ack_delay_us": None` — `board_backend.py:257`
   - `return []` — `board_backend.py:285`
   - `allow_missing_device: bool = True` — `board_backend.py:58`
   - `"reserved for future real-board integration"` — `fpga_driver.py:77`
   - `"mock and future real HIL backends"` — `fpga_driver.py:1`

3. **是否缺测试或验证** — 不缺。任务包限定的验证是"只读审计"，Worker 已执行。我做了独立交叉验证，全部吻合。

4. **是否过度工程** — 否。7 节内容全部服务于单一目标（真板缺口清单），没有多余抽象。四层验收标准是合理的递进结构，不是过度拆分。

5. **是否破坏已有功能** — 否。无代码变更。

6. **文档是否把计划写成事实** — 否。多处显式区分"checklist 存在"与"真板已验证"。

### 攻击性检查结果

- **伪装成真板证据的隐蔽表述？** — 未发现。所有表述都保持了"placeholder"语义。
- **配置项等同于硬件能力的暗示？** — 未发现。第 3 节全部用探询语气。
- **过度承诺的四层标准？** — 未发现。四层是合理的最小必要条件。
- **未覆盖的占位代码路径？** — `axi_map.py` 和 `dma_client.py` 未在 Inputs to read 中列出，但寄存器名与 `board_backend.py` 引用一致，不构成 blocking issue。
- **计划写成事实的隐蔽风险？** — 未发现。治理文件更新全部诚实。

### 三个非阻塞提醒

- **N1**：第 3.3 节的寄存器名称来源不透明——可能是从 `board_backend.py` 推断而来，而非直接审计 `axi_map.py`。后续真板任务应直接审计寄存器定义文件。
- **N2**：四层验收标准缺乏量化阈值（如"epoch_id 多快变化"、"超时多少"）。这不是当前 blocking issue，因为量化需要真板环境，强行指定反而是伪精确。
- **N3**：权限描述偏 Linux（`/dev/uio*`、udev），而项目运行在 Windows 上。后续执行任务应确认目标平台。

### 结论

即使在 adversarial 模式下，也未发现任何 blocking issue。Worker 完全按任务包执行，placeholder 证据全部可交叉验证，边界表述严格诚实。**PASS**。
