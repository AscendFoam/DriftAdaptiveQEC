# 第一篇论文 Claim Ladder 与证据升级契约

**Task：** T1.4.1  
**冻结日期：** 2026-07-14  
**机器可读源：** `docs/claim_ladder.json`  
**作用域：** 第一篇论文的 simulation、synthesis estimate、board measurement、HIL/replay、quantum experiment 主张

## 1. 一句话论证

第一篇论文只把每项结论提升到当前可直接复核的最高证据层；架构存在、配置字段、host
runtime、read-only gate、外部论文或未来计划都不能代替缺失的综合、真板、HIL 或量子实验
证据。

## 2. 术语账本

| Canonical term | 本文定义 | 不得混用为 |
| --- | --- | --- |
| `simulation` | Python/NumPy/runtime emulator 上的数学、Monte Carlo、software protocol 或 mock-backed software-HIL | synthesis、board measurement、quantum experiment |
| `mock-backed software-HIL` | host 内用 mock backend 执行 HIL-shaped 调度和事件链；**software-HIL 属于 CL1** | real-board HIL |
| `synthesis estimate` | 有 target device、tool/version、constraints 和 report 的 post-synthesis/post-implementation 估计 | analytical operation count、software fixed-point parity、board measurement |
| `board measurement` | 在明确板卡、bitstream、clock 和 host/device path 上取得的数字控制平面测量 | 真实微波控制、量子读出、量子 QEC |
| `board HIL/replay` | 真板消费可追溯 trace 并产生可与 reference 对齐的输出、事件和 latency/fallback 记录 | mock-backed software-HIL、量子闭环实验 |
| `quantum experiment` | 在明确量子平台上按可追溯 protocol 得到 logical lifetime/error/fidelity 等物理测量 | simulation LER、surrogate fidelity、外部论文结果 |
| `isolated host true .tflite runtime` | 选定 preserved artifact 在记录 interpreter 环境中的真实加载/执行与一致性校验 | synthesis、HIL、deployment closure |
| `model oracle` | 知道 assumed noise state 的不可部署 decoder reference | optimal recovery、control oracle、真实 decoder |

所有稿件、图注、补充材料和回复信都使用上述 canonical terms；禁止只写未限定的
“HIL”“hardware validated”“FPGA result”或“oracle”。

## 3. 五层 Claim Ladder

五层给出部署/实验成熟度顺序，但不是严格可互换的单标量：某一层的通过只升级与该层
直接对应的 claim。尤其是 host `.tflite` runtime、training reproducibility 和 read-only
real-board gate 属于正交 supporting lanes，不会自动把结论提升到综合或真板层。

| ID | 证据层 | 当前状态 | 当前最高允许表述 | 禁止表述 | 升级到本层的最低门槛 |
| --- | --- | --- | --- | --- | --- |
| `CL1` | 软件仿真与 mock-backed software-HIL | `supported_bounded` | `simulation-derived`、`hardware-aware simulation`、`mock-backed software-HIL`、`pre-registered causal synthetic benchmark` | `FPGA measured`、`hardware validated`、real-board HIL、quantum experiment | 冻结模型/场景/baseline/seed/metric；同 trace 与因果无泄漏；paired uncertainty、失败分支、模型 fidelity 边界 |
| `CL2` | 综合/实现估计 | `blocked_missing_synthesis_report` | 当前只能写“尚需 target-specific synthesis/implementation” | measured board latency/power、timing closed on hardware、FPGA validated | RTL/HLS source hash、tool/device/clock/constraints；LUT/FF/BRAM/DSP、WNS/TNS/clock report；power 仍标 estimate |
| `CL3` | 真实板卡数字控制平面测量 | `blocked_missing_board_host_execution` | 当前只能写 planned measurement / blocked evidence | quantum control validated、microwave/readout integrated、real-time QEC demonstrated | 板卡/器件/bitstream/tool/clock/I/O provenance；core/transport/end-to-end 分测；重复、分位数、失败计数和原始报告 |
| `CL4` | 真实板卡 HIL / trace replay | `blocked_no_board_hil` | 当前只能写 planned board replay，不能写已完成 | closed-loop quantum experiment、logical lifetime gain、beyond break-even、deployment closure | CL3 contract；非 placeholder MMIO/DMA/commit/fallback；输入 trace、board output、event log、deadline/mismatch 可追溯 |
| `CL5` | 真实量子实验 | `blocked_no_quantum_experiment` | 当前只能把外部实验当文献事实或未来接入条件 | 从 software LER 推断 beyond break-even、从 residual variance 推断 squeezing、quantum advantage | 明确量子平台与校准；readout/reset/leakage/protocol；预注册 metric/baseline/uncertainty；授权原始数据与分析 provenance |

### 3.1 不允许跨层继承

1. `CL1 -> CL2`：operation count、software latency 或 Q4.20 parity 不等于 synthesis report。
2. `CL2 -> CL3`：综合资源/时序估计不等于板卡测量；estimated power 不等于 measured power。
3. `CL3 -> CL4`：单个 kernel latency 不等于 HIL；还需要 transport、state、commit、fallback 和 trace equivalence。
4. `CL4 -> CL5`：数字 replay 不等于量子装置闭环；没有 cavity/transmon/readout/reset 数据就不能写量子性能。
5. 高层证据也不能反向替代公平 simulation baseline、statistical uncertainty 或 model-mismatch 审计。

## 4. 当前证据快照

### 4.1 已支持但有边界

| 证据 | 当前事实 | 归属 | 可写边界 |
| --- | --- | --- | --- |
| T24 | 四场景、五模式、`repeats=2`、paired seeds 的 frozen-set formal software revalidation；run artifact 当前存在 | `CL1` | 仅限 mock-backed software-HIL frozen set；不外推 expanded benchmark、`.tflite` ranking 或 real board |
| T1.3.4 | 72k 同 trace samples；static/Window/EKF/oracle 为 `0.06139/0.02264/0.02532/0.01139`；EKF gap closure CI `[0.7044,0.7378]` | `CL1` task evidence | 仅限预注册 Gaussian step、one-window delay、model oracle；尚未自动晋升 frozen paper mainline |
| Q4.20 parity | 四个受控软件场景的 fixed-point emulation 与 float affine path 数值差异已记录 | `CL1` supporting | software numerical parity；不是 RTL、综合、时序或板测 |
| T48 | `final_gate_verdict=GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8`；记录的 interpreter 和两份 true `.tflite` 当前仍存在 | `OL1` orthogonal | selected preserved artifacts 的 isolated current-host true runtime；不升级 `CL2`/`CL3`/`CL4` |
| T49/T71/T72 | checked-in read-only gate/provenance 可 replay/regenerate | `OL2` orthogonal | 当前 verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`；不能写真板执行成功 |

### 4.2 当前明确缺失

- 仓库扫描未发现 `.rpt/.bit/.xclbin/.dcp` synthesis/implementation/bitstream 证据文件；
- `submission_draft_fast_path_cost_model.json` 自身声明只含 analytical operation counts；
- `submission_draft_fixed_point_parity_analysis.json` 自身声明只含 software fixed-point emulation；
- T72 当前 device paths 不可用，bitstream/RTL address/DMA/fixed-point contract 未确认；
- `board_backend.py` 的执行路径仍被 gate 判定为 `placeholder_only`；
- 没有本项目 cavity/transmon、真实 syndrome history、readout/reset/leakage 或 logical lifetime
  原始数据。

因此当前项目最高 evidence rung 为 `CL1`。这个结论不否认未来 `CL2`--`CL5` 的计划，只是
禁止把计划写成结果。

## 5. 第一篇论文逐项 Claim Registry

| ID | Claim | 当前状态 | Allowed wording | Forbidden wording |
| --- | --- | --- | --- | --- |
| `PC01` | T24 frozen-set 排名 | `CL1 / frozen_supported` | “在 T24 四场景、五模式、`repeats=2` 的 mock-backed software-HIL frozen set 内，`hybrid_residual_b` 为四场景 winner。” | “CNN 已在真实 FPGA、expanded benchmark 或量子实验中全面优于传统 baseline。” |
| `PC02` | T1.3.4 static-to-model-oracle gap | `CL1 / task_evidence_not_yet_promoted` | “在预注册、同 trace、一窗因果延迟的 Gaussian synthetic step drift 中，existing EKF 显著缩小 static-to-model-oracle logical-error gap。” | “已覆盖一般 loss/outlier/leakage drift”或“超过 oracle”。 |
| `PC03` | Q4.20 software parity | `CL1 / supporting_only` | “Q4.20 software emulation 在受控样本上与 float affine fast path 保持所报告的数值一致性。” | “Q4.20 RTL 已完成综合、时序收敛或板级验证。” |
| `PC04` | 真实综合资源/时序/功耗 | `blocked` | `[Evidence needed: tool/device/constraint-specific synthesis and implementation reports]` | operation count 或 software parity 等价于 synthesis/resource/timing result |
| `PC05` | 约 300 元真实板 core/transport/end-to-end 测量 | `blocked` | `[Evidence needed: named-board measured rows with bitstream and host provenance]` | ZCU111 config、placeholder backend 或 read-only gate 等价于真实板测 |
| `PC06` | 真实板 HIL/replay、deadline/fallback | `blocked` | `[Evidence needed: board replay outputs, event logs, source-vs-board comparison and latency distributions]` | software-HIL 或 isolated host `.tflite` runtime 等价于 real-board HIL |
| `PC07` | 真实 GKP 量子 logical lifetime/error/fidelity gain | `blocked` | `[Evidence needed: identified quantum platform, authorized raw data, protocol and uncertainty]` | software LER、surrogate fidelity 或外部论文数据等价于本项目量子实验 |
| `PC08` | selected artifact 的 isolated host true `.tflite` runtime | `OL1 / supported_bounded` | “选定 preserved float/int8 artifact 已在记录的 isolated current-host interpreter 环境真实执行并做一致性校验。” | 默认环境、跨主机 portability、software-HIL integration 或 deployment closure 已完成 |

## 6. 稿件写作与升级规则

1. 每个结果句必须带 `PCxx` claim ID 或能回指到 claim-evidence ledger；没有 ID 的新强 claim
   默认不进入主文。
2. 每次提升 level 必须新增证据 artifact、任务记录、验证和风险复核；只改 prose 不构成升级。
3. `current_status=blocked` 的 claim 只允许出现在 Limitations、Outlook、planned measurement
   或 `[Evidence needed: ...]` 占位中。
4. figure caption 不能比正文 claim 高一层；摘要、标题和结论不能比 Results 高一层。
5. external literature 只证明他人装置/方法的事实，不能把对应 rung 赋给本项目。
6. `Go`、`PASS`、`gate passed` 等治理词不直接进入论文；必须翻译成相应 evidence statement。
7. software-HIL 必须带 `mock-backed`；HIL 若不带限定词，默认视为违规措辞。
8. oracle 必须写成 `model oracle`、`control oracle` 或 `channel-recovery bound` 之一；三者不互换。
9. 任何“measured”必须同时给出测量对象、设备、时间/资源口径和 provenance；否则改为
   `estimated`、`simulated` 或删除。
10. 只有新的 review 明确把 claim promotion 标为通过，才更新 JSON 的 `current_level`；历史
    evidence pack、sidecar 或本地临时 run 不能自动改值。

## 7. 失败与降级策略

| 失败情况 | 必须降级为 |
| --- | --- |
| synthesis 无 timing closure 或 report 不全 | 保留 CL1 analytical/software evidence，不报告 CL2 |
| 真板只有单次 kernel timing，没有 transport/trace provenance | CL3 的 preliminary measurement，不升级 CL4 |
| board output 与 source/reference 不一致 | 报告 mismatch 和 failure diagnosis，不写 HIL validation |
| HIL deadline/fallback 未闭环 | 只报告 replay/partial-path，不写 end-to-end closure |
| 量子 trace 无可靠 truth/tomography | 只报告 calibration/diagnostic metric，不报告 logical gain |
| future CNN 不优于强 adaptive baseline | 删除 CNN 性能主张，保留 adaptive MAP/dual-loop co-design 叙事 |

## 8. 机器可读 contract 与复核

`docs/claim_ladder.json` 是本文件的结构化同源 contract。测试
`tests/test_claim_ladder_contract.py` 验证：

- 五层 ID、顺序和 canonical name 唯一；
- 每层都有 allowed/forbidden wording、至少三条 promotion gate 和存在的证据锚点；
- 当前只允许 `CL1=supported_bounded`，`CL2`--`CL5` 全部 fail closed；
- T48/real-board gate 两条正交 lane 不会静默提升 ladder；
- blocked claim 没有伪造 `current_level`，并保留 `[Evidence needed: ...]`；
- Markdown 覆盖全部 `CLxx/PCxx` ID 和 current-host `NO_GO` 边界。

任何后续 task 若修改 claim status，必须同时修改 Markdown、JSON、测试期望、任务板和风险表，
并重新运行该 contract test。
