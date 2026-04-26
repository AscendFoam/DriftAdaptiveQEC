# CNN-FPGA-GKP 后续仿真与工程补强计划

## 1. 文档目的

本文档用于系统整理当前 `CNN-FPGA-GKP` 主线在“物理真实性、运行时真实性、板级真实性、误差判定真实性”几个维度上仍可继续补强的方向，并给出一套可执行的分阶段计划。

这份文档不否定当前已经得到的阶段结论。相反，它的目标是把两件事分开说明：

1. 当前主线已经足够回答什么问题。
2. 若后续要进一步提高论文说服力、工程可信度或真板落地性，还应补哪些仿真与实现细节。

---

## 2. 当前主线已经做到什么

结合现有 `P0 -> P4` 阶段结果，当前主线已经具备以下能力：

1. 能在统一有效噪声模型下，比较不同解码/自适应方案的相对优劣。
2. 能用双回路软件 HIL 口径验证 `teacher + residual` 方案在部署约束下的闭环表现。
3. 能把快回路固定点、参数 bank、commit、DMA 读窗、慢回路推理延迟这些关键工程环节纳入同一条实验链路。
4. 能对 `float / int8 / 真实部署产物` 的一致性做软件侧验收。

因此，当前主线足以支撑这样的表述：

- 当前项目已经有“有效模型 + 软件 HIL + 部署产物一致性”的正式实验闭环。
- 当前 `P4` 的排序结论，是在这套口径下成立的工程结论。

但当前主线还不能直接等价为以下更强说法：

- “这已经是真正完整的板级仿真。”
- “这已经覆盖了更完整的物理噪声通道与模拟读出链。”
- “这已经是逐级位宽完全精确的 RTL 级定点等价。”
- “这已经是更完整电路级容错口径下的最终逻辑错误率。”

也就是说，当前主线已经足够支撑“阶段性工程结论”，但还没有走到“全链路高保真数字孪生”。

---

## 3. 当前仍未覆盖的关键细节总览

| 方向 | 当前状态 | 为什么重要 | 主要代码入口 | 优先级 |
| --- | --- | --- | --- | --- |
| 多类物理噪声接入 | `physics/noise_channels.py` 已实现，但未接入 P2/P3/P4 主线 | 提升物理建模说服力 | `physics/noise_channels.py`、`cnn_fpga/benchmark/run_hil_suite.py` | 高 |
| 真板 I/O 语义补全 | `board_backend.py` 仍是 placeholder 级 | 提升板级可信度、方便后续真板联调 | `cnn_fpga/hwio/board_backend.py`、`cnn_fpga/hwio/fpga_driver.py` | 中 |
| 延迟模型负载耦合 | 目前是独立高斯/常数抽样 | 提升运行时口径真实性 | `cnn_fpga/runtime/latency_injector.py`、`cnn_fpga/runtime/scheduler.py` | 高 |
| 慢回路故障模型细化 | 目前主要是独立伯努利注错 | 提升闭环鲁棒性分析质量 | `cnn_fpga/benchmark/run_hil_suite.py`、`cnn_fpga/runtime/slow_loop_runtime.py` | 高 |
| 固定点逐级位宽精确 | 当前是“接近硬件”的固定点与饱和统计 | 提升数值一致性可信度 | `cnn_fpga/runtime/fast_loop_emulator.py`、`cnn_fpga/model/quantize.py` | 高 |
| syndrome 读出链细化 | 当前是统计化测量模型 | 提升物理测量口径真实性 | `physics/syndrome_measurement.py` | 中高 |
| 逻辑错误定义扩展 | 当前是有效模型，不是更完整电路级模型 | 提升论文外推边界的清晰度 | `physics/logical_tracking.py` | 中 |

这里的“优先级”是相对当前项目主线而言：

- 高：值得在近期逐步补入，能直接增强当前 `P4` 结论的可信度。
- 中高：重要，但可以在主线稳定后补。
- 中：更偏向增强项或为后续真板/论文扩展服务，不应阻塞当前 teacher 编码主线。

---

## 4. 方向一：把 `noise_channels.py` 的多类噪声真正接入主线

### 4.1 当前状态

`physics/noise_channels.py` 已经实现了较完整的玻色量子噪声通道，包括：

1. 光子损失 `photon loss`
2. 热噪声 `thermal noise`
3. 位移噪声 `displacement noise`
4. 相位噪声 `phase noise / dephasing`

这些通道是基于 Wigner 函数演化或其近似来写的，属于“更物理”的噪声层。

但当前 `P2/P3/P4` 主线里，真正驱动快回路闭环的噪声并不是这套通道，而是 `run_hil_suite.py` 中 `_build_mock_noise_provider()` 生成的有效参数序列：

- `sigma`
- `mu_q`
- `mu_p`
- `theta_deg`

再叠加场景化漂移形式：

- `static / constant`
- `linear`
- `step`
- `periodic / sin`
- `random_walk`

所以当前主线的噪声建模，本质上是“有效噪声参数 + 场景化漂移”的工程模型，而不是“底层量子噪声通道直接驱动闭环”。

### 4.2 为什么重要

如果后续要更强地回答“系统到底在跟踪什么”，就需要把两层概念区分清楚：

1. 底层真实噪声源是什么。
2. 这些底层噪声在解码层面等效成什么样的 `(sigma, mu_q, mu_p, theta)` 漂移。

`noise_channels.py` 的意义不在于简单替换掉当前场景模型，而在于把“物理噪声来源”与“控制层可见参数”串起来。

这能带来三类收益：

1. 论文层面可以更自然地解释场景设置与物理来源的对应关系。
2. 数据集可以更丰富，不再只依赖人工设定的 `linear / periodic / step` 漂移。
3. 可以更清楚地区分“真实物理噪声变化”和“人为施加的场景 envelope”。

### 4.3 不建议的直接接法

不建议把 `noise_channels.py` 直接塞进当前每个 fast cycle 的主循环里，原因有三点：

1. 当前代码主线是围绕有效参数 `(sigma, mu_q, mu_p, theta)` 建立的，直接替换会牵动数据、训练、benchmark 全链路。
2. `noise_channels.py` 更偏 Wigner/状态级演化，计算代价较大，不适合原样放进高频循环。
3. 当前 teacher / CNN / 参数映射都默认“输入是一组有效噪声统计量”，如果底层噪声变量直接换口径，会破坏已有结果的可比性。

### 4.4 最合理的接法

最合理的方案是采用“两层噪声模型”：

第一层：物理通道层  
- 用 `noise_channels.py` 生成底层噪声状态，例如 `gamma, n_bar, sigma_disp, sigma_phase`。

第二层：有效参数层  
- 把底层物理噪声通过标定、采样或近似映射，折算成当前主线仍可消费的 `(sigma, mu_q, mu_p, theta)`。

第三层：场景 envelope 层  
- 再让这些底层参数随时间以 `linear / periodic / step / random_walk` 等方式变化。

这样可以保持当前主线接口不变，同时把噪声来源做得更物理。

### 4.5 建议改造步骤

#### 步骤 A：离线标定层

新增一个“物理噪声到有效参数”的离线标定脚本，核心目标是回答：

- 在不同 `gamma / n_bar / sigma_phase / sigma_disp` 下，最终会得到怎样的有效 `(sigma, mu_q, mu_p, theta)` 分布。

建议新增文件：

- `physics/noise_effective_mapper.py`
- `cnn_fpga/data/build_physical_augmented_dataset.py`

产出物建议包括：

- `physical_noise_lookup.json`
- `physical_noise_samples.npz`
- “物理通道参数 -> 有效统计量”散点图和拟合报告

#### 步骤 B：dataset 侧接入

在现有 dataset builder 基础上新增一条配置分支：

- 保持现有 `anisotropic_gaussian` 路线不删
- 新增 `distribution: physical_mixed_channels`

对应入口建议在：

- `cnn_fpga/data/dataset_builder.py`

做法：

1. 先从底层物理通道采样。
2. 折算成有效标签。
3. 仍输出与当前模型兼容的 histogram + label 格式。

这样做的好处是：

- 模型结构、训练脚本、benchmark 入口都不用立刻重写。
- 可以先比较“现有有效噪声数据集”和“物理增强数据集”训练出来的模型差异。

#### 步骤 C：runtime 侧接入

在 `run_hil_suite.py` 的 `_build_mock_noise_provider()` 外再包一层更物理的 provider，例如：

- `build_physical_noise_provider()`

它不直接返回底层 Wigner，而是返回：

- 当前时刻的物理通道参数
- 折算后的 `sigma / mu_q / mu_p / theta_deg`
- 元数据中记录真实物理来源

这样 `FastLoopEmulator` 仍可以按旧接口运行，但 benchmark 输出会额外保留物理来源信息。

### 4.6 风险与代价

1. 参数维度会显著上升，容易导致数据覆盖不足。
2. 物理通道与有效参数之间未必是一一对应，可能出现不可辨识性。
3. 若映射标定做得太粗，最后只是“多加了一层名字”，并未真正提升建模质量。

### 4.7 验收标准

这一方向的第一阶段，不要求马上改写全部主线，只要求达到以下口径：

1. 能从底层物理通道生成与当前主线兼容的数据样本。
2. benchmark 输出中能明确区分“物理噪声来源”和“场景 envelope”。
3. 至少完成一组“统计噪声数据集 vs 物理增强数据集”的对照训练与 benchmark。

---

## 5. 方向二：把真板 I/O 语义从 placeholder 提升到板级语义

### 5.1 当前状态

`cnn_fpga/hwio/board_backend.py` 目前已经提供了一个统一接口骨架：

- AXI-Lite 寄存器映射
- DMA buffer 的 mmap 访问
- active/staged params 影子状态
- commit 控制位与状态读取

但它仍然是明显的 placeholder 级实现，典型特征包括：

1. 读写路径主要还是 shadow state 驱动，不是真正的板级状态机。
2. `schedule_commit()` 里 active params 会立即跟 shadow staged 同步，语义仍偏“软件近似”。
3. 没有真实的 DMA descriptor ring、ownership、interrupt 驱动、backpressure、reset 语义。
4. `allow_missing_device` 与 `/dev/uio*` 检查也表明它目前主要是为未来扩展预留接口，而非完整真板仿真。

### 5.2 为什么重要

当前软件 HIL 已经足够支撑“运行时逻辑是通的”，但还不能支撑“板级 I/O 语义已经充分验证”。

后续一旦要做真板联调或更像板卡的仿真，最容易出问题的通常不是 CNN 本身，而是：

1. DMA 缓冲的所有权切换
2. 中断/轮询的时序差异
3. reset 与异常恢复
4. stale buffer、重复 buffer、丢 buffer
5. commit 与读取边界上的竞争条件

### 5.3 最合理的补法

这里不建议直接跳到“真板一次性打通”，而是建议先补“板级语义仿真层”。

也就是说，先在软件里把下面这些语义补齐：

1. DMA ring buffer 状态机
2. buffer ownership
3. commit ack 的板端行为
4. reset / reinit / timeout / stale data 语义
5. interrupt 与 polling 两种驱动模式

等这些在软件板级语义里已经稳定，再迁移到真正板卡。

### 5.4 建议改造步骤

#### 步骤 A：把 board backend 的影子状态机补全

重点文件：

- `cnn_fpga/hwio/board_backend.py`
- `cnn_fpga/hwio/fpga_driver.py`
- `cnn_fpga/hwio/dma_client.py`

建议新增或补齐：

1. DMA descriptor/slot 状态枚举
2. buffer 的 producer-consumer ownership
3. commit 触发后到 ack 返回之间的真实 pending 态
4. reset 后寄存器和 buffer 状态的统一回收

#### 步骤 B：增加板级异常事件

建议支持以下事件类型：

1. `dma_stale_buffer`
2. `dma_partial_write`
3. `commit_ack_lost`
4. `irq_missed`
5. `board_reset_mid_run`

这些事件先不要求真板真实出现，而是要让 HIL 主线能承受这些情况并给出日志。

#### 步骤 C：增加板级一致性测试

建议新增一组最小板级语义测试：

- 连续多次 commit 不越界
- reset 后状态归零
- buffer sequence 单调
- stale buffer 能被识别
- ack timeout 不会 silently pass

### 5.5 风险与代价

1. 这类工作很容易工程量膨胀。
2. 如果当前真板还没到位，过早追求“像真板的一切细节”可能性价比不高。
3. 若没有明确板卡接口文档，可能会做出一套和未来 RTL/驱动不一致的中间语义。

### 5.6 当前建议优先级

这项工作重要，但不应阻塞当前 `teacher params` 编码主线。  
它更适合按“可渐进增强”的方式做，而不是现在就变成唯一主线。

---

## 6. 方向三：把延迟模型从独立高斯抽样升级为负载耦合模型

### 6.1 当前状态

`cnn_fpga/runtime/latency_injector.py` 当前每个阶段的延迟是独立采样的：

- `dma`
- `preprocess`
- `inference`
- `writeback`
- `commit_ack`
- `fast_cycle`

支持的分布也比较简单：

- `constant / fixed`
- `normal / gaussian`

这意味着当前延迟模型默认隐含了一个假设：

- 每次慢回路更新的 DMA、推理和写回延迟彼此独立；
- 它们也不依赖系统当前排队长度、CPU 负载、待处理窗口数量或总线竞争情况。

### 6.2 为什么重要

这种独立抽样模型适合做“第一阶段工程口径”，但它会低估一些真实运行时问题：

1. 慢回路一旦积压，后续延迟往往会继续变大，而不是独立重置。
2. DMA 延迟与推理延迟在真实系统中经常共振，不完全独立。
3. 若 host 端偶尔抖动，会带来一段持续的 backlog，而不是单次随机尖峰。

这些问题对双回路系统尤其关键，因为双回路最怕的不是单次小波动，而是持续积压导致的参数过时。

### 6.3 最合理的补法

建议把延迟建模改成“条件分布 + 运行时状态耦合”：

延迟不再只由静态均值和方差决定，还要依赖当前状态，例如：

- `pending_windows`
- `slow_job_inflight`
- `recent_commit_density`
- `host_busy_level`
- `dma_queue_depth`

这样可以把“系统负载”纳入延迟模型。

### 6.4 建议改造步骤

#### 步骤 A：先把 runtime 状态记录下来

在现有 benchmark 输出中补更多状态字段，例如：

- 每次 slow job 启动时的 pending window 数
- 当前是否有 in-flight job
- 最近 N 个窗口的平均处理时间
- 当前 commit 是否积压

主要入口：

- `cnn_fpga/runtime/scheduler.py`
- `cnn_fpga/benchmark/run_hil_suite.py`

#### 步骤 B：实现 load-aware latency injector v1

建议新增：

- `LoadAwareLatencyInjector`

规则不用一开始就很复杂，可以先做简单条件偏置：

1. 若 `pending_windows > 0`，则 `dma / inference / writeback` 均值抬高
2. 若连续多窗积压，则抖动方差也抬高
3. 若出现 timeout/重试，则下一次写回和 commit 延迟增加

#### 步骤 C：加入 burst / correlated latency

进一步支持“持续一段时间系统都很忙”的相关延迟，而不是每次独立采样。

可以用一个隐藏状态机建模：

- `idle`
- `busy`
- `congested`

各状态对应不同延迟分布，并允许状态转移。

### 6.5 风险与代价

1. 模型更真实后，benchmark 方差会变大，结果更难看但更可信。
2. 若缺少真实主机测量数据，负载模型容易变成“看上去更复杂”的主观设定。
3. 如果一开始做得太细，会拖慢 teacher 编码主线。

### 6.6 验收标准

第一阶段验收建议是：

1. 延迟不再完全独立于 backlog。
2. benchmark 报告里能看到“轻载/重载/拥塞”三种状态统计。
3. 至少证明一种方案在相关延迟下仍保持排序稳定，或明确指出排序何时会变化。

---

## 7. 方向四：把慢回路故障模型从独立伯努利扩展为状态化故障模型

### 7.1 当前状态

当前 HIL 主线对慢回路故障的注入，主要是独立事件级故障，例如：

- `dma_timeout_prob`
- `axi_write_fail_prob`
- `cnn_infer_fail_prob`

这类故障注入的优点是简单、直观、便于扫参。  
但它的问题也很明显：

1. 默认每次故障相互独立。
2. 不区分“短暂抖动”与“持续性退化”。
3. 不建模重试、退避、降级模式、恢复窗口等系统行为。

### 7.2 为什么重要

在真实双回路系统里，最危险的往往不是“一次失败”，而是：

1. 连续几窗都没更新，参数逐步陈旧
2. commit 写失败后反复重试
3. 推理服务卡住，但快回路还在继续跑
4. 主机侧进入 fallback 模式后性能长期退化

如果只用独立伯努利注错，系统鲁棒性会被看得过于乐观。

### 7.3 建议补法

建议把故障模型升级成“状态机 + 恢复策略”形式。

建议最少包含以下状态：

- `normal`
- `retrying`
- `degraded_hold_last`
- `dropped_update`
- `service_stalled`

并规定状态转移逻辑，例如：

1. 一次 DMA timeout 后进入 `retrying`
2. 多次失败后进入 `degraded_hold_last`
3. 若推理恢复成功，再回到 `normal`

### 7.4 建议改造步骤

#### 步骤 A：故障事件对象化

建议把当前分散在 benchmark 中的故障概率配置，收敛成统一故障注入器，例如：

- `cnn_fpga/runtime/fault_injector.py`

定义故障类型、持续时长、恢复策略和影响范围。

#### 步骤 B：把故障和 scheduler 事件联动

在 `scheduler.py` 里记录：

- 本次窗口是否因为故障被跳过
- 当前 active params 持续陈旧了多久
- fallback 持续了多少窗

#### 步骤 C：扩展 benchmark 输出

建议新增指标：

- `stale_param_window_count`
- `retry_count`
- `consecutive_fail_max`
- `fallback_ratio`
- `recovery_time_windows`

### 7.5 验收标准

第一阶段不要求穷尽所有异常，只要求：

1. 故障不再是完全独立、无记忆的。
2. 能区分“偶发失败”和“持续服务退化”。
3. benchmark 报告能体现故障对参数陈旧度和最终 LER 的影响。

---

## 8. 方向五：把固定点仿真从“接近硬件”推进到逐级位宽精确

### 8.1 当前状态

当前固定点相关工作已经做了不少：

1. `fast_loop_emulator.py` 中对 syndrome、correction、histogram 输入做了饱和/溢出统计。
2. `AxiRegisterMap` 和运行时参数里已经使用固定点语义。
3. `quantize.py` 已支持部署模型的对称 int8 量化。

这些已经足够支持：

- “当前数值范围是否会炸”
- “float 与 int8 是否明显偏离”

但它仍不是更严格意义上的逐级位宽精确模型。  
当前缺口主要包括：

1. 未严格区分每一级乘加的中间位宽。
2. rounding mode、truncate、saturate 的具体规则还不够细。
3. 部分路径是“近似硬件”而不是“逐级硬件等价”。
4. CNN 量化目前偏部署近似，不是严格对齐具体 RTL 数据通路。

### 8.2 为什么重要

当模型和控制参数越来越接近硬件极限时，小的位宽差异就可能造成：

1. 参数映射边界处的不连续
2. correction 饱和与 clip 行为不同
3. long-run benchmark 中小偏差逐渐累积

这会直接影响“为什么 offline 好、HIL 却翻转”这类机制判断。

### 8.3 最合理的补法

建议把固定点建模分成两层：

第一层：控制路径位宽精确  
- 先把 `(K, b)` 到 correction 的主控制链路做到逐级位宽精确。

第二层：模型推理位宽精确  
- 再考虑 CNN 权重、激活、缩放、重排等更严格的硬件对齐。

原因是当前主线最敏感的是控制路径，不一定要先把全部 CNN RTL 等价化。

### 8.4 建议改造步骤

#### 步骤 A：整理位宽规范

建议先形成一份明确的位宽表，至少包含：

- syndrome 输入位宽
- histogram 累积位宽
- `K` 和 `b` 的寄存器位宽
- 乘法器输出位宽
- 加法树中间位宽
- correction 输出位宽
- saturation / rounding 策略

建议新增文档：

- `docs/CNN_FPGA_GKP_固定点与位宽规范.md`

#### 步骤 B：实现 bit-accurate shadow pipeline

建议新增：

- `cnn_fpga/runtime/fixed_point_pipeline.py`

把快回路关键路径拆成显式 stage：

1. 输入量化
2. 乘法
3. 累加
4. bias 注入
5. rounding / truncate
6. saturation

并输出每一级中间值和溢出标记。

#### 步骤 C：用小样本做逐条对比

不要一开始就拿大 benchmark 做。  
先准备一批固定输入，逐条比较：

- float reference
- 当前 approximate fixed-point
- bit-accurate fixed-point

先看差异来自哪里，再决定是否把新链路接入正式 benchmark。

### 8.5 验收标准

1. 主控制路径各 stage 的位宽和 rounding 规则都被显式实现。
2. 小样本逐条比对结果可复现。
3. 能解释当前 approximate fixed-point 与 bit-accurate 版本之间的主要差异来源。

---

## 9. 方向六：把 syndrome 读出链路从统计模型推进到“测量链模型”

### 9.1 当前状态

`physics/syndrome_measurement.py` 当前已经包含：

1. 有限压缩噪声
2. 测量效率损失
3. ancilla error
4. shot noise

这已经是一个比纯理想测量更真实的统计模型。  
但它仍然是“从真实位移到 noisy syndrome”的有效近似，不是完整的模拟/读出链路模型。

目前没有细到：

1. 模拟前端增益/失调
2. 滤波与积分窗口
3. ADC 量化位数与满量程
4. threshold / unwrap / calibration drift
5. I/Q 不平衡或通道串扰

### 9.2 为什么重要

当前主线里，syndrome 是整个快回路的入口。  
如果以后要更认真地讨论：

- overflow 为什么出现
- 输入范围为什么该这样调
- 测量链误差如何影响最终解码

那就需要把“读出链”显式分层。

### 9.3 建议补法

建议新增一个比当前 `RealisticSyndromeMeasurement` 更细的“测量链后端”，但不直接替换当前实现，而是并行存在。

建议分成三层：

第一层：理想综合征  
- 仍然由当前 modulo 测量得到。

第二层：模拟前端层  
- 增益、偏置、滤波、额外模拟噪声。

第三层：数字读出层  
- ADC 量化、裁剪、校准、阈值处理。

### 9.4 建议改造步骤

#### 步骤 A：轻量版 ADC/AFE 模型

建议新增：

- `physics/readout_chain.py`

先实现最小版本：

1. 模拟增益 `gain`
2. 固定偏置 `offset`
3. 模拟高斯噪声
4. ADC 位宽 `n_bits`
5. 满量程 `full_scale`
6. 量化与饱和

#### 步骤 B：与 fast loop 的输入范围联动

把 `readout_chain.py` 的输出直接接到 fast loop 输入侧，比较：

- 当前统计测量模型
- 增加 ADC/AFE 后的模型

看 overflow、saturation、LER 是否显著变化。

#### 步骤 C：再决定是否继续加复杂前端

若轻量版 ADC/AFE 已经能解释大部分现象，就不必立刻做更重的模拟模型。  
只有当它解释不了关键差异时，再考虑加入更细的积分窗口或校准漂移。

### 9.5 验收标准

1. 至少有一套可开关的 ADC/AFE 读出链模型。
2. 能量化读出链误差对 overflow 与 LER 的影响。
3. 能明确回答“当前输入范围默认值是否依赖理想化读出假设”。

---

## 10. 方向七：把逻辑错误定义从有效模型扩展到更完整口径

### 10.1 当前状态

`physics/logical_tracking.py` 当前的逻辑错误定义是：

- 根据每轮真实误差与校正后的残差累积
- 一旦累计位移越过 GKP 决策边界 `±sqrt(2pi)/2`
- 就记为逻辑错误

这是一种非常合理、也很常见的有效模型。  
它的优点是：

1. 与当前连续变量/GKP 主线高度一致
2. 易于长时间统计
3. 易于和解码控制量建立直接联系

但它仍不是更完整的电路级或容错调度级口径。  
它没有显式纳入：

1. 更复杂的多轮门序列
2. 更丰富的 ancilla/circuit fault propagation
3. 更完整的 fault-tolerant schedule 结构

### 10.2 为什么重要

这一点主要关系到“结论的外推边界”。

当前主线可以说：

- 在当前有效模型下，某方案的闭环表现更好。

但不能直接说：

- 在更完整电路级容错系统里，它一定也以同样幅度更好。

如果论文里把这条边界写清楚，反而会更严谨。

### 10.3 最合理的补法

这一方向不建议现在就重做整个主线，而是建议先补“多口径并行评估”。

也就是说，先保留当前 `LogicalErrorTracker` 不动，再新增一个更完整但更慢的评估分支，用于抽样对照。

### 10.4 建议改造步骤

#### 步骤 A：先把当前口径写清楚

在文档和 benchmark 输出中明确标注：

- 当前 `LER` 是有效模型口径，不是完整电路级 `LER`

这是最低成本、但收益很高的一步。

#### 步骤 B：实现扩展型 tracker

建议新增：

- `physics/logical_tracking_extended.py`

它可以先不追求完全电路级，只需要比当前更细一些，例如：

1. 区分不同类型的逻辑翻转来源
2. 记录跨窗积累与恢复
3. 记录错误触发前的残差轨迹

#### 步骤 C：抽样对照而不是全量替换

先挑少量 scenario 与 seed，对比：

- 当前有效模型 LER
- 扩展型 tracker 指标

如果两者排序高度一致，则说明当前口径足以做主实验；若差异明显，再考虑是否继续升级。

### 10.5 验收标准

1. benchmark 报告明确写清 LER 口径。
2. 至少有一个扩展型 tracker 分支。
3. 至少完成一组抽样 scenario 的双口径对照。

---

## 11. 当前最建议优先做的不是全部一起补，而是分层推进

为了避免工程量失控，建议后续按“对当前主线结论的影响程度”来分层推进。

### 11.1 第一优先层：直接增强当前 P4 可信度

这一层最值得先做：

1. `noise_channels.py` 的两层接入
2. 负载耦合延迟模型
3. 状态化慢回路故障模型
4. 快回路控制路径 bit-accurate 固定点链

原因：

- 它们最直接影响当前 benchmark 的解释力；
- 不需要等待真板也能推进；
- 对“为什么 offline 与 formal HIL 不一致”这类机制问题也更有帮助。

### 11.2 第二优先层：增强物理入口真实性

这一层建议随后推进：

1. syndrome 读出链的 ADC/AFE 轻量模型
2. 更清晰的逻辑错误口径扩展

原因：

- 它们更偏向物理真实性和论文边界清晰化；
- 重要，但不一定立刻改变当前 teacher 编码路线的工程判断。

### 11.3 第三优先层：为真板联调铺路

这一层建议作为条件性扩展：

1. 板级 I/O 语义补全
2. 更接近真实中断/DMA/descriptor 的 backend

原因：

- 这部分主要服务于未来真板落地；
- 在当前论文主线仍可用软件 HIL 完成时，不应反过来阻塞主实验。

---

## 12. 建议里程碑

## 12.1 里程碑 M1：两周内可完成的增强项

目标：在不打断当前 teacher 编码主线的前提下，先把最关键的仿真增强做出第一版。

建议交付：

1. `noise_channels` 到有效参数的离线映射脚本
2. `physical_mixed_channels` dataset 配置雏形
3. load-aware latency injector v1
4. stateful fault injector v1
5. runtime 报告新增 backlog / stale / retry 统计

验收重点：

- 不追求全替换，只追求能做小规模对照实验。

## 12.2 里程碑 M2：三到四周内完成的数值与测量增强

目标：提高“数值与输入侧真实性”。

建议交付：

1. 位宽规范文档
2. bit-accurate control pipeline v1
3. `readout_chain.py` 轻量 ADC/AFE 模型
4. 一组小规模双口径对照实验

验收重点：

- 能解释当前 overflow、输入范围和定点差异的主要来源。

## 12.3 里程碑 M3：中期增强项

目标：提高板级语义和评估口径完整性。

建议交付：

1. board backend 的 ring/state 语义补齐
2. reset / stale / ack timeout 测试集
3. 扩展型 logical tracker

验收重点：

- 真板语义与更完整错误评估成为“可选增强层”，而不是口头描述。

---

## 13. 与当前主线实验推进的关系

这些补强方向不应该替代当前主线。  
当前主线仍然应该保持为：

1. 继续收敛 `teacher params` 的编码方式
2. 用受控 benchmark 验证 `Full / Selective / Gated` 等模式
3. 优先回答“怎样把离线优势转成 formal HIL 收益”

更准确的推进方式应是：

- 主线实验继续跑 teacher 表征问题；
- 补强计划并行推进，但先做不会重写主线接口的版本；
- 每补一项，都优先做“小样本对照”，而不是一上来替换整套 benchmark。

这样才能避免项目陷入“为了更真实而把当前已验证链路全部推倒重来”。

---

## 14. 推荐的具体执行顺序

如果按当前性价比排序，我建议后续执行顺序如下：

1. 先做 `noise_channels -> effective parameters` 的离线桥接。
2. 再做 `load-aware latency + stateful fault` 的 runtime 增强。
3. 然后补 `bit-accurate control pipeline`。
4. 再补 `ADC/AFE` 轻量读出链。
5. 最后视真板条件决定是否重点投入 `board_backend` 补全。

原因如下：

1. 前三项最能增强当前 `P4` 结果的可解释性。
2. 它们都能在现有软件 HIL 主线内完成，不依赖额外硬件条件。
3. `board_backend` 的收益更偏中长期，不适合现在压过 teacher 编码主线。

---

## 15. 最终判断

当前项目并不是“仿真太粗所以结果都不可信”，而是处在一个很典型的中间阶段：

- 已经有足够强的有效模型和软件 HIL 主线，能够产出有价值的工程结论；
- 但离“更完整物理噪声 + 更完整板级语义 + 更完整误差口径”的高保真数字孪生，还有一段明确且可规划的路要走。

后续最合理的做法不是推翻当前主线，而是围绕以下目标逐步补强：

1. 让噪声来源更物理
2. 让运行时语义更接近真实系统负载
3. 让固定点和读出链更可解释
4. 让板级与逻辑错误口径的边界更清楚

如果这些增强按本文档的顺序逐步落地，那么后续论文和工程报告中的表述就可以从：

- “我们在有效模型和软件 HIL 下验证了该方法”

逐步升级为：

- “我们在物理增强噪声、负载耦合运行时、可解释固定点链和更完整板级语义下，验证了该方法的稳定性与边界。”

