# Route-A claim contract、canonical roles 与不可混排比较 lane

## 一句话论证

在同一可部署执行合同下，MAP 负责平均逻辑错误性能，regime/event/fallback 逻辑负责 tail
安全，RTL 负责确定性执行；CNN、teacher 和 student 是可替换扩展，只有通过各自独立证据门
才允许升级对应主张。

推荐英文主语固定为 **regime-aware safe adaptive dual-loop**，不再把 `CNN + FPGA` 当作论文
主角。当前已经支持的是架构/证据合同和 board-independent RTL qualification；Route-A 的 LER、
tail、GQF lifetime 和真板速度优势仍是条件主张或当前禁止主张。

## canonical terminology ledger

| canonical term | 可接受简称/历史变体 | 冻结决定 |
| --- | --- | --- |
| regime-aware safe adaptive dual-loop | Route A, safe adaptive dual-loop | 唯一 paper-level 主角 |
| static/adaptive joint MAP decoder | MAP, adaptive MAP | 唯一拥有 decoder-lane LER/logical decision 的模块 |
| causal regime posterior estimator | HMM, regime detector | 输出 posterior/update permission；不直接输出 logical action |
| event and leakage/reset FSM | event FSM | 输出 mode/reset/hysteresis；不拥有 average LER 优势 |
| conservative fallback and health monitor | fallback, health monitor | 输出 fail-closed reason/fallback/rollback/health |
| versioned trusted A/B MAP bank | A/B bank, parameter bank | 输出 bank/version/commit ack；不解释 regime |
| FPGA fast-path executor | fast path, RTL core | 执行已锁存 fixed-point action；不运行 host estimator |
| legacy CNN residual module | CNN residual | 可替换 slow-loop proposal/消融；不是系统主角 |
| Feedback-GRAPE/NMF teacher | teacher, NMF teacher | controller training target；不是 decoder 或 oracle |
| distilled low-dimensional student | student | 可选 controller recurrence；不等于 FPGA system |
| hidden-state decoder/model oracle | oracle MAP, model oracle | 明示 hidden-truth、不可部署、只作上界 |

禁用含混术语：`the oracle`、`RNN decoder`、`deployed teacher`、`student equals the FPGA
system`、`FPGA runs the host estimator in the critical path`。

## 角色与责任边界

| role | 类型 | 唯一责任/输出权 | 不允许冒领 |
| --- | --- | --- | --- |
| safe adaptive dual-loop | primary system | 组合 performance/safety/atomic update/deterministic execution contracts | 不生成新的 decoder 分数或全局总分 |
| static/adaptive MAP | required | logical-coset LLR/decision、candidate MAP image | HMM posterior、硬件 latency |
| regime posterior | required | normal/smooth/shift/burst/leakage posterior、update permission | logical action、LER 最优性 |
| event FSM | required | hold/recovery/reset mode 与 hysteresis | posterior 校准、MAP likelihood |
| fallback/health | required | fallback/rollback/reason/health | “更优 decoder”或 oracle action |
| trusted A/B bank | required | stage/CRC/version/CAS/commit/readback/LKG | regime 分类 |
| FPGA fast path | required | registered action/state 与 core-cycle contract | host estimation、真实 transport 未测字段 |
| CNN residual | replaceable | matched-budget bounded proposal | primary contribution、普遍 LER 优势 |
| NMF teacher | replaceable/offline | controller training target | decoder oracle、部署主路径 |
| student | replaceable/candidate | bounded controller extension | 整个 FPGA system、official GQF 结果 |
| hidden-state oracle | privileged reference | decoder/model upper reference | deployable aggregate、硬件表 |

这保证旧叙事可以保留为模块史和消融：CNN/teacher/student 的已有证据不会被删除，但论文即使
移除这些模块，仍可退化为 `adaptive/static MAP + event/fallback + trusted bank + deterministic
fast path`，主系统定义不坍塌。

## 三条不可混排 evaluation lane

| lane | 数据域与 privilege | 允许指标 | 绝对禁止 |
| --- | --- | --- | --- |
| same-trace GKP decoder | protocol-aligned syndrome；deployable observed-only，oracle 物理分栏 | `p_L,p_X,p_Y,p_Z`、average/p95/worst LER、static-to-oracle gap、lag、false update/fallback、avoided/induced error、decoder cost | 把 GQF lifetime 减 LER；跨 squeezing/noise/round 拼 raw LER；oracle 进入 deployable aggregate |
| official GQF controller | 固定 commit 的官方 GQF；同 simulator/training/selection/seeds | `T_X,T_Y,T_Z,T_ch,F_avg`、gain retention、controller params/MAC/memory、unsafe/fallback | 用 T2.3.7 冒充 official reproduction；exact reproduction 前写 surpass NMF |
| task-normalized FPGA hardware | 同 code/problem/precision/latency boundary，逐项标 evidence level | core/source-to-action/closed-loop latency、II、deadline、Fmax、LUT/FF/BRAM/DSP/power | surface-code raw latency直接排 single-mode GKP；P&R 冒充板测；core-only 对 source-to-action |

不存在第四条“综合排行榜”。三条 lane 的 metric namespace 在机器合同中互斥；跨 lane 只能做
qualitative innovation/evidence map，不能求和、相减、排序或生成 global score。

## 逐 claim 冻结矩阵

状态定义：

- `SUPPORTED_CONTRACT`：当前只支持设计/证据合同；
- `SUPPORTED_BOUNDED`：当前在明确证据层和数据域内可写；
- `SUPPORTED_EXTENSION_ONLY`：可写成模块扩展证据，不进入主比较；
- `CONDITIONAL_FUTURE`：对应 task/gate 通过后才能激活；
- `ABLATION_ONLY`：只允许消融/负结果；
- `PROHIBITED_NOW`：当前正文/摘要/结论禁止作为事实。

| claim ID | 状态 | 最强可支持主张 | metric/domain/evidence | 激活或撤销条件 |
| --- | --- | --- | --- | --- |
| CLAIM-ARCH-01 | SUPPORTED_CONTRACT | 已定义职责不重叠、可降级的 safe adaptive dual-loop | cross-lane governance | 任一模块冒领其他 lane 指标或移除 nonmixing 即撤销 |
| CLAIM-RTL-QUAL-01 | SUPPORTED_BOUNDED | production core 与独立整数 golden 在 1e6 board-independent CXXRTL cycles 全字段一致 | core cycle/II；RTL/CXXRTL | source/trace 变化未重跑或出现 mismatch/undefined 即撤销 |
| CLAIM-FPGA-EST-01 | SUPPORTED_BOUNDED | 当前 fast path 有目标器件三 seed P&R timing/resource estimate | Fmax/resources/core cycles；P&R estimate | hash/constraint 变化或不写 target/clock qualifier 即撤销 |
| CLAIM-STATIC-01 | CONDITIONAL_FUTURE | Route-A 相对最强 deployable static GKP decoder 改善 aggregate/tail LER | decoder same-trace；T6.7.1/2、T6.8.1 | paired 95% LCB 不正或任一 tail gate 失败即撤销 |
| CLAIM-DRIFT-01 | CONDITIONAL_FUTURE | matched history/cadence/budget 下取得一般漂移优势且不违反 OOD 安全门 | decoder held-out；T6.7/T6.8.2 | 无外部复现、catastrophic 或 nominal margin 失败则降为“相对内部强基线”或撤销 |
| CLAIM-TAIL-01 | CONDITIONAL_FUTURE | freeze/trusted-bank/fallback 降低有害 tail 且 nominal 代价可接受 | p95/worst/avoided/induced；T6.7.2/4 | `55/512 > 37/512` 类反例保留或 nominal margin 失败即撤销 |
| CLAIM-CNN-01 | ABLATION_ONLY | legacy CNN residual 是可替换 matched-budget ablation | decoder cost/LER；T6.6/7 | 没有真实 checkpoint、成本不匹配或 comparison 失败时不得升级 |
| CLAIM-STUDENT-01 | SUPPORTED_EXTENSION_ONLY | 现有受限项目 controller simulator 中 student retention/cost 证据可写，但不属于 official GQF/decoder lane | T4.4.5 component evidence | 任何 official GQF、长时/OOD 或 decoder 暗示都会撤销 |
| CLAIM-GQF-01 | PROHIBITED_NOW | 当前不得声称超过 Puviani NMF lifetime | official GQF；T6.8.3/4/5 | exact reproduction + paired lifetime LCB>0 才可激活 |
| CLAIM-HW-SPEED-01 | PROHIBITED_NOW | 当前不得声称比已有 FPGA QEC decoder 更快 | matched real-board；T6.8.6/T6.9.2 | 只有同任务可比子集和同 bitstream 板测通过才可激活 |
| CLAIM-BREAK-EVEN-01 | PROHIBITED_NOW | 当前不得声称真实 physical break-even/寿命纪录 | physical oscillator measurement | simulation/CXXRTL/P&R/纯数字板测均不能激活 |

## 摘要、讨论与结论的用语门

### 当前可用于方法/讨论的英文

> We define a contract-centric, regime-aware safe adaptive dual-loop in which MAP decoding owns logical-error performance, regime/event/fallback logic owns tail safety, and the FPGA fast path owns deterministic execution. The production core is cycle-exact against an independent integer reference over one million board-independent CXXRTL qualification cycles; this evidence does not include physical transport or board measurement.

这段不能提前加入 “improves LER”, “surpasses NMF”, “faster than existing FPGA decoders” 或
“break-even”。在 T6.7.4 前，摘要若必须写结果，应只写 qualification 数字并明确
board-independent；更稳妥的做法是等 Route-A promotion 后再冻结摘要。

### 未来条件式结果模板

只有对应 gate 通过后才可把方括号替换成真实数据：

> Under a preregistered shared execution contract, Route-A [reduced aggregate LER with a paired 95% lower confidence bound above zero] while [meeting the abrupt/OOD tail and nominal non-inferiority gates].

> In the fixed-commit official GQF environment, the Route-A extension [improved paired logical-channel lifetime over reproduced NMF], with [confidence interval and compute burden].

> On the matched real-board task, the integrated fast path achieved [source-to-action p50/p95/p99/worst latency] with [zero deadline misses and its confidence upper bound].

任何一个方括号对应的 task 失败，整句删除或按任务板分支降级，不能用另一条 lane 的成功补门。

## 立即禁用的表述

- `CNN is generally/universally optimal`、`CNN drives the primary contribution`；
- `adaptive decoding is universally superior`、`state-of-the-art GKP decoder`；
- single-mode GKP 的 `surface-code threshold`；
- `currently surpasses Puviani NMF` 或把 T2.3.7 写成 official reproduction；
- `fastest FPGA decoder`、`measured FPGA latency/power`、`transport qualified`；
- `exceeds Sivak 2023`、`real break-even`、`physical lifetime record`；
- static GKP、drift decoder、GQF controller 与 FPGA hardware 的 global leaderboard/score。

## fail-closed 降级链

1. CNN matched gate 失败：CNN 只留消融，Route-A 其余模块不受影响；
2. tail gate 失败：降为 smooth-only adaptive MAP，不写安全自适应；
3. aggregate LER gate 失败：回退 static MAP-LUT + deterministic FPGA；
4. GQF exact reproduction 失败：只写 partial/directional reproduction，不做 surpass；
5. GQF lifetime LCB 不正：只写 retention/compression/safety extension 或负结果；
6. 真板缺失/失败：只写 CXXRTL/synthesis/P&R estimate，不做 measured/faster；
7. 物理装置实验缺失：break-even 与 physical lifetime 始终关闭。

## 机器产物与复现

- `docs/t6_5_1_route_a_claim_contract.json`
- `docs/t6_5_1_route_a_claim_contract_source_data.csv`
- `cnn_fpga/benchmark/route_a_claim_contract.py`
- `tests/test_route_a_claim_contract.py`

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.benchmark.route_a_claim_contract
python -m pytest -q tests\test_route_a_claim_contract.py
```

机器合同绑定 claim ladder、术语 registry、T5.1.4 证否分支、T4.4.5 bounded student、T5.5.2
P&R estimate、T6.2.2 long RTL qualification、实验计划与任务板；20 个 contract gates 与 10 个
越界语义 mutations 必须全部通过。

