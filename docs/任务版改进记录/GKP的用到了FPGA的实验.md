## 总结判断

`new_task_board.md` 应当大改。

当前任务板的主问题不是仿真不足，而是：

- 把 FPGA 放在论文初稿之后，像“补充实验”；
- 默认 FPGA 可接收较理想的连续 syndrome；
- 硬件任务部分隐含了高性能 ADC/DAC、AWG、微波控制链；
- 真实量子硬件仍占据 Phase 7 的完成路径；
- 论文证据结构按模块展开，尚未围绕最终 claim 闭环。

在“唯一可用硬件是约 300 元 FPGA”的条件下，论文应重定位为：

> **Physics-aligned drift-adaptive GKP decoding on a resource-constrained FPGA**

对应的一句话论文论点可以冻结为：

> We demonstrate a physics-aligned dual-timescale decoder for approximate single-mode GKP quantum memory, in which a host-side estimator tracks non-stationary noise while a resource-constrained FPGA executes deterministic fixed-point MAP-LUT decisions, supported by multi-fidelity simulation, strong adaptive baselines, bit-accurate hardware validation and cycle-accurate hardware-in-the-loop experiments, without claiming direct quantum-hardware control.

中文含义是：

> 本文证明，在资源极受限 FPGA 上，仍可通过“主机慢速噪声估计 + FPGA 确定性 MAP-LUT 快回路”缩小漂移条件下 static MAP 与 oracle MAP 的差距；证据来自物理对齐仿真、强 baseline、真实板级测量和闭环 HIL，而非真实 cavity–transmon 实验。

这是用 claim–evidence–boundary 反推任务板后的主轴。

## 术语需要先冻结

| 规范术语 | 本项目定义 |
| --- | --- |
| low-cost FPGA prototype | 约 20K LUT、无高速 RF ADC/DAC 的入门 FPGA |
| measurement record | 数字化的等效读出记录；不是实际 transmon IQ 数据 |
| derived syndrome | 从 measurement record 和协议状态推导出的 syndrome |
| cycle-accurate HIL | PC 物理仿真器与真实 FPGA 按 round/cycle 交换数据 |
| measured FPGA result | 真实板上的数值、资源和时延结果 |
| physics-aligned simulation | 根据 GKP 实验协议与噪声预算构建的仿真 |
| real quantum experiment | 只有接入真实 bosonic mode/ancilla 才能使用；本项目不声明 |

约 300 元级 Tang Nano 20K 一类板已经具有约 20K LUT、828 Kbit Block SRAM、64 Mbit SDRAM 和48个乘法器，足够实现本项目的 MAP-LUT、frame、fallback 和日志路径。[官方资源规格](https://en.wiki.sipeed.com/hardware/en/tang/tang-nano-20k/nano-20k.html)

## 两篇论文对任务板提出的硬约束

### 1. 不能只模拟抽象 decoder

2020 实验的实际链路是：

```text
conditional displacement
→ transmon readout
→ FPGA数字判决
→ round-dependent feedback displacement
→ next round
```

square code 不是每轮直接获得完整连续 `(s_q,s_p)`，而是以 sharpen/trim 为单位形成四轮周期。FPGA 还维护协议状态和反馈方向。[实验协议](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Quantum_error_correction_of_a_qubit_encoded_in_grid_states_of_an_oscillator.md:595>)

因此仿真器必须输出：

```text
r_t = measurement record
m_t = round/cycle state
s_hat_t = derived syndrome
a_t = controller action
```

而不应只输出理想 `s_t`。

### 2. 噪声必须具有实验因果路径

综述第 III 章和实验误差预算要求至少覆盖：

- photon loss；
- additive Gaussian displacement；
- phase diffusion；
- transmon bit flip；
- transmon phase flip/readout error；
- `|f⟩` leakage/reset failure；
- Kerr/spurious rotation；
- pulse-amplitude calibration drift；
- latency/stale feedback。

其中 transmon bit flip 与 photon loss 是2020实验的主要限制，不能被一个总的 `sigma_eff` 完全替代。[Noise Models](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Advances_in_Bosonic_Quantum_Error_Correction_with_Gottesman–Kitaev–Preskill_Codes_Theory_Engineering_and_Applications.md:773>)、[实验误差预算](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Quantum_error_correction_of_a_qubit_encoded_in_grid_states_of_an_oscillator.md:427>)

### 3. 论文指标不能只有 LER

需要形成与物理实验相呼应的指标阶梯：

- stabilizer convergence；
- peak/envelope proxy；
- `X/Y/Z` logical decay；
- `F_avg/F_e/PTM`；
- logical lifetime；
- QEC ON/OFF gain；
- oracle-gap closure；
- tail-window LER；
- FPGA latency/resource/precision；
- HIL 闭环一致性。

这些指标中，前半部分来自仿真，后半部分来自真实 FPGA；必须分开标注。

# 建议的 task board v2

## 保留既有成果

以下部分不要推翻：

- Phase 0 已完成的范围、平台和双回路接口；
- T1.1 理想 GKP MAP；
- T1.2 finite-energy 状态、逻辑通道和趋势；
- T1.3 漂移生成器与 oracle MAP；
- 当前推荐的 T1.3.3、T1.3.4。

这些是后续重构的理论资产。

---

## 新增 Milestone 1.4：论文与低成本硬件契约

建议放在 `T1.3.4` 后，并作为进入 Phase 2 的强制 gate。

| ID | 任务 | 通过标准 |
| --- | --- | --- |
| T1.4.1 | 冻结低成本 FPGA 资源上限 | 给出 LUT/FF/BRAM/DSP、时钟、片上存储、UART/SPI 和无 RF ADC/DAC 边界 |
| T1.4.2 | 冻结论文主 claim 与非 claim | 明确只声明 FPGA-HIL，不声明真实 GKP 实验或真实微波控制 |
| T1.4.3 | 冻结 measurement-record 接口 | 定义 `r_t/m_t/s_hat_t/a_t`、时间戳、valid、round type、fallback |
| T1.4.4 | 预注册证据门 | 冻结 baseline、场景、seed、CI、消融和失败标准 |
| T1.4.5 | 冻结硬件选择门 | 先综合目标设计，确认能放入约20K LUT级器件后再购买开发板 |

这一步应新增单独任务记录，而不是只在任务板备注里写“用廉价 FPGA”。

---

## 新增 Milestone 2.0：实验协议数字孪生

把它放在现有 syndrome-level simulator 之前。

| ID | 任务 | 产物/通过标准 |
| --- | --- | --- |
| T2.0.1 | 实现 square sharpen–trim 四轮状态机 | 支持 q sharpen、q trim、p sharpen、p trim 和 cycle index |
| T2.0.2 | 实现 measurement backaction 与条件反馈 | outcome 改变后续状态；不是纯标签生成器 |
| T2.0.3 | 建立实验参数表 | 对应 conditional displacement、readout、feedback、loss、ancilla error |
| T2.0.4 | 实现 phase-of-cycle fault injection | bit flip 发生时间能改变错误后果 |
| T2.0.5 | 复现实验定性趋势 | 约20 rounds 收敛、QEC ON/OFF 衰减趋势、主要误差排序一致 |
| T2.0.6 | 建立 sBs 可选参考 | 只作为增强协议/讨论基准，不扩成新的主线 |

该 milestone 是论文物理可信度的核心。综述明确显示，2020 sharpen–trim 与2023 sBs 代表不同工程阶段，[实验实现](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/relative_papers/Advances_in_Bosonic_Quantum_Error_Correction_with_Gottesman–Kitaev–Preskill_Codes_Theory_Engineering_and_Applications.md:1647>)；本项目宜以前者作为数字孪生主协议、后者作为 Discussion 和敏感性参照。

---

## 重写 Phase 2：多保真、控制器耦合仿真

### Milestone 2.1：高速闭环仿真器

修改现有 T2.1：

- 生成 `measurement record`，不只生成连续 syndrome；
- 输出 round/cycle metadata；
- 支持 FPGA action 回灌；
- 跟踪 residual、Pauli frame、confidence 和 stale update；
- 支持 on-chip trace 导出格式。

### Milestone 2.2：finite-energy effective model

保留现有任务，但增加：

- round-dependent ancilla/readout noise；
- sharpen 与 trim 的不同 backaction；
- active displacement calibration error；
- FPGA quantization 与物理控制误差分开建模。

### Milestone 2.3：Fock/master-equation 验证

用途从“完整主仿真器”改为“独立高保真审计器”：

- 只验证少量关键参数点；
- 验证 fast simulator 没有制造虚假 CNN 优势；
- 验证 finite-energy envelope、loss、Kerr 和 ancilla error 下排序不翻转。

### Milestone 2.4：时序数字孪生

应拆成两个延迟：

\[
L_{\mathrm{system-assumed}}
=
L_{\mathrm{meas}}+L_{\mathrm{ADC}}+L_{\mathrm{FPGA}}+L_{\mathrm{command}}+L_{\mathrm{pulse}},
\]

\[
L_{\mathrm{FPGA-measured}}
=
L_{\mathrm{quantize}}+L_{\mathrm{lookup}}+L_{\mathrm{decision}}+L_{\mathrm{state}}.
\]

第一项是基于文献的系统假设；第二项才是本项目可真实测量的结果。不能混为一个“measured closed-loop latency”。

---

## 强化 Phase 3：协议原生强 baseline

除现有 baseline 外，增加：

| Baseline | 作用 |
| --- | --- |
| fixed-step sharpen–trim controller | 对齐2020实验的基础反馈 |
| static round-aware MAP | 固定实验参数的可部署基线 |
| calibration-aware static MAP | 排除收益只是简单偏置校准 |
| Bayesian memory | 强多轮基线 |
| EWMA/Kalman adaptive MAP | 最重要传统自适应基线 |
| sliding-window estimator | 与 CNN 窗口估计公平比较 |
| oracle protocol-aware MAP | 不可部署上界 |
| no-adaptation FPGA LUT | 隔离慢回路贡献 |

所有 baseline 必须使用相同 measurement record、相同动作空间、相同 latency budget 和相同随机种子。

---

## 重写 Phase 4：资源受限软硬件协同设计

### Milestone 4.1：主机侧慢回路

CNN 保留在 PC/CPU，不部署到300元 FPGA：

- causal 1D-CNN；
- 输出低维 `mu/sigma/rho/readout bias/p_outlier`；
- 低频更新；
- uncertainty/OOD 输出；
- 与 Kalman/EWMA 使用相同历史窗口。

### Milestone 4.2：FPGA 快回路

建议改为：

| ID | 任务 |
| --- | --- |
| T4.2.1 | round-aware quantizer/wrapper |
| T4.2.2 | parametric MAP-LUT 或 PWL likelihood |
| T4.2.3 | LLR/confidence/fallback |
| T4.2.4 | Pauli-frame 与 controller state |
| T4.2.5 | 双参数 bank、版本校验和原子切换 |
| T4.2.6 | trace replay、cycle counter 和自检 |
| T4.2.7 | 4/6/8/10/12/16-bit design-space sweep |
| T4.2.8 | 目标 FPGA 预综合与资源 gate |

资源 gate 建议要求：

- timing closure；
- 至少保留约15%资源余量；
- 完整参数 bank 能放入片上存储；
- 不依赖开发板没有的高速接口；
- 不因 UART 吞吐限制 decoder core。

### Milestone 4.3：双回路安全更新

除 hysteresis 外增加：

- stale parameter detection；
- CRC/version check；
- update reject；
- bank rollback；
- history starvation；
- UART/SPI transmission fault；
- fallback telemetry。

---

## 重写 Phase 5：按论文 claim 组织证据

当前 Phase 5 按指标分类，可以改为按审稿问题分类。

### Evidence Gate E1：物理可信度

- reproduce stabilizer convergence；
- reproduce square-code `X/Y/Z` 不对称趋势；
- reproduce photon-loss/bit-flip 主导关系；
- Fock/master-equation spot checks；
- 所有物理参数可追溯。

### Evidence Gate E2：算法有效性

- average LER；
- worst-window/tail LER；
- oracle-gap closure；
- logical lifetime proxy；
- static noise 下不退化；
- paired confidence interval。

### Evidence Gate E3：贡献因果性

必须消融：

- 去掉 CNN；
- CNN 换 Kalman/EWMA；
- 去掉 history；
- 去掉 confidence；
- 去掉 fallback；
- 单 bank vs 双 bank；
- float vs fixed-point；
- round-aware vs round-agnostic；
- ideal syndrome vs measurement record。

### Evidence Gate E4：失效边界

覆盖：

- unseen drift family；
- burst/outlier；
- incorrect noise model；
- readout contrast drift；
- bit-flip rate升高；
- update lag；
- parameter corruption；
- overflow/saturation；
- history loss；
- LUT 超界。

### Evidence Gate E5：资源—性能 Pareto

报告：

- bit width–LER；
- LUT size–LER；
- DSP/BRAM–latency；
- update interval–adaptation lag；
- board frequency–throughput；
- accuracy–resource Pareto front。

---

## 用新的 Phase 6 替换“论文初稿先行”

现有任务板在 FPGA 之前先写论文，[当前 Phase 6–7](</D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md:213>)。对于本项目这是不合适的，因为真实板级证据是主贡献之一。

建议新的 Phase 6 改为：

## Phase 6：低成本 FPGA 实测

### Milestone 6.1：板卡 bring-up

- 固定板卡、芯片和工具链版本；
- clock/PLL 自检；
- UART/SPI 通信；
- BRAM/SDRAM trace replay；
- GPIO `start/done`；
- bitstream hash 和构建记录。

### Milestone 6.2：bit-accurate 验证

逐 shot 对齐：

```text
Python float
→ Python fixed-point
→ RTL simulation
→ post-route simulation
→ real FPGA
```

通过标准：

- action/fallback/frame 100% 一致；
- LLR 误差满足预注册界限；
- 无未解释 overflow；
- 所有 corner vectors 通过。

### Milestone 6.3：真实板级测量

- post-route Fmax；
- LUT/FF/BRAM/DSP；
- core cycles；
- initiation interval；
- worst-case latency；
- update latency；
- GPIO/logic-analyzer 交叉测量；
- 功耗只作为 board-level 辅助指标。

必须分别报告：

- decoder-core latency；
- trace replay latency；
- UART transfer latency；
- emulated system latency。

### Milestone 6.4：cycle-accurate HIL

```text
PC physics simulator
    ↓ measurement record
real FPGA decoder
    ↓ action/confidence/fallback
PC state update
    ↓ next round
```

通过标准：

- FPGA-HIL 与 Python fixed-point 闭环结果落在同一统计区间；
- 板上结果不是离线预存 action；
- 参数更新与 fallback 真正改变后续 trace；
- 可完整重放失败样本。

---

## 新 Phase 7：论文、主图和开放复现

推荐主图压缩为六张：

1. GKP物理实验—数字孪生—廉价 FPGA 的边界图；
2. sharpen–trim 数字孪生及实验趋势复现；
3. drift 下 strong-baseline、dual-loop 与 oracle 主结果；
4. OOD/tail/failure-mode 和消融；
5. precision–resource–LER Pareto；
6. 真实 FPGA latency/resource 与 cycle-accurate HIL。

任务包括：

- claim–evidence matrix；
- figure-first 结果冻结；
- Methods、Results、Discussion；
- limitation 和非 claim 扫描；
- 代码、RTL、测试向量、配置、seed、bitstream hash；
- one-command reproduction；
- adversarial reviewer audit。

只有 E1–E5 和 Phase 6 通过后，才写完整论文。可以提前写 Methods 草稿和图模板，但不能提前冻结 Results 叙事。

---

## 新 Phase 8：可选外部验证

把原 `T7.4` 的真实 GKP 数据/量子硬件接入从主线移出，改为 optional：

- 公开或合作方 GKP measurement record replay；
- 第三方在另一块廉价 FPGA 上复现；
- 第二种 FPGA family 的综合可移植性；
- 未来 cavity–transmon 合作实验。

这些任务不再阻塞论文完成，也不得被写成当前已完成工作。

## 建议删除或降级的任务

- 删除“量化 CNN 部署到廉价 FPGA”作为主线；
- 删除真实 ADC/AWG/microwave latency 的实测承诺；
- 将真实 cavity–transmon 接入降为 optional；
- 不以 surface-code FPGA decoder 作为直接性能公平对手，只作为工程背景；
- 不把 board-level USB 功耗写成 decoder energy；
- 不把 UART round-trip 写成 decoder latency；
- 不把 synthetic measurement record 称为 experimental data；
- 将 operational pseudo-threshold 降为次指标，主指标转向 logical lifetime gain、LER、oracle gap 和 tail risk。

## 建议新增风险

| Risk ID | 等级/迫切度 | 风险 |
| --- | --- | --- |
| R-014 | High/Soon | 理想 syndrome 与真实协议记录不一致 |
| R-015 | High/Soon | 把 simulated/HIL 写成真实量子实验 |
| R-016 | High/Soon | UART/I/O 掩盖或夸大 decoder latency |
| R-017 | Medium/Soon | 训练、验证使用同一 simulator，产生循环论证 |
| R-018 | Medium/Monitor | 廉价 FPGA 资源不足导致模型选择偏差 |
| R-019 | Medium/Monitor | Gowin/板卡工具链降低复现性 |
| R-020 | Medium/Soon | 没有强 adaptive baseline，CNN claim 不可信 |
| R-021 | Medium/Monitor | 板级功耗测量精度不足 |
| R-022 | High/Soon | 只有“烧板成功”，缺少算法和物理新意 |

其中 R-014、R-015、R-017 会直接影响论文结论可信度，应由新增的 T1.4/T2.0 解决，不应只登记后监控。

## 推荐执行顺序

当前不必回滚已有工作：

1. 完成 `T1.3.3` regret/oracle-gap；
2. 完成 `T1.3.4` adaptive baseline 对齐；
3. 执行新增 `T1.4.1–T1.4.5`；
4. 完成 `T2.0` 实验协议数字孪生；
5. 重构 Phase 2–5 仿真与证据门；
6. 预综合确认约20K LUT板可容纳；
7. 仿真实验冻结后购买约300元开发板；
8. 完成 Phase 6 真实板级验证和 HIL；
9. 通过证据门后完成论文和开放复现。
