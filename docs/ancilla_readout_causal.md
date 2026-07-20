# T5.2.2 ancilla bit/phase flip 与 readout error 独立因果注入

**日期：** 2026-07-16  
**状态：** PASS（protocol-native effective simulation）  
**实现：** `cnn_fpga/benchmark/ancilla_readout_causal.py`  
**机器证据：** `docs/t5_2_2_ancilla_readout_causal.json`  
**Source Data：** `docs/t5_2_2_ancilla_readout_source_data.csv`

## 1. 结论与证据边界

T2.2.2 已实现 sBs stage-resolved ancilla/readout overlay，T5.1.2 也有 readout/ancilla drift lane；但二者的
正式产物把 bit、phase 与 readout 同时改变，并主要依赖单 seed/聚合 endpoint，不能据此声称某一个通道的
因果敏感度。本任务没有改名复用旧 PASS，而是执行 3 个互斥实验族：

1. `ancilla_bit_flip`：只改变 X constituent 的 big-CD bit-flip probability；
2. `ancilla_phase_flip`：只改变同一位置的 phase-flip probability；
3. `readout_error`：只改变 4×3 hidden-to-observed classifier 的 symmetric g/e confusion。

每族使用 6 个注入率、8 个全新独立 seed clusters、每格 4,096 cycles；同一 family/seed 内跨 rate 使用
common random numbers，CI 用 20,000-repeat whole-seed bootstrap。结果只能说明当前 effective model 的
协议方向与通道隔离成立，不是 cavity--transmon master equation、装置标定、实验 65× 比例复现或 repeated
physical-memory LER。

## 2. 一手机制锚

机器 JSON 将本地 Sivak 2023 补充材料逐行 hash 绑定：

- 第 1121 行：sBs 对 ancilla phase flip 具有设计上的 fault tolerance；big-CD 中的 phase flip 等价于小参数误差；
- 第 1157 行：big-CD 中部 bit flip 可产生 logical error；readout misclassification 会驱动错误 virtual rotation；
- 第 1159 行：实验采用分别提高 bit-flip rate 或 phase-flip rate 的 selective noise injection；
- 第 1165 行：实验 65× 差异只作定性方向锚，不作为本项目数值拟合目标。

## 3. 冻结实验设计

注入率为 `0, 0.005, 0.01, 0.02, 0.04, 0.08`。每个 cycle 的 ideal label 按
`K_gg -> K_ge -> K_eg -> K_ee` 平衡循环，使 g/e 和 X/Z constituent 都被覆盖。固定项包括：

- 无 leakage injection；复用已登记 conditional reset kernel；
- bit event 后 hidden logical backaction conditional probability `0.5`；
- phase continuous-backaction scale `0.01`；
- virtual-rotation range `[-0.6, 0.6] rad`；
- truth 只在 simulator evaluator lane，deployable record 仍只有 observed syndrome/reset/counters。

readout-only 中，误判可能选择错误的既有 reset action；这是 readout intervention 的下游因果结果，不是同时
注入 reset-failure channel。T5.2.3 才单独改变 leakage/reset probabilities。

## 4. 分离 estimand 与结果

三族不生成 global sensitivity score：

| family，rate=0.08 | 主事件率，95% cluster CI | 协议效应，95% cluster CI | 必须为 0 的交叉效应 |
| --- | ---: | ---: | --- |
| bit-only | bit `0.080078 [0.077545,0.082306]` | logical backaction `0.039093 [0.037720,0.040436]` | phase/readout/rotation `0` |
| phase-only | phase `0.079468 [0.077698,0.081268]` | mean absolute backaction `0.0007947 [0.0007770,0.0008127]` | Z-toggle/logical/label-change `0` |
| readout-only | mismatch `0.080750 [0.079300,0.082199]` | mean absolute rotation `0.024537 [0.023977,0.025092] rad` | bit/phase/logical/label-change `0` |

六点曲线中，bit event/toggle/logical、phase event/backaction、readout mismatch/rotation 均严格递增。解析检查分别为：

- bit event/toggle/label-change `p`，logical backaction `0.5p`；
- phase event/nonzero-backaction `p`，mean absolute backaction `0.01p`；
- readout mismatch/nonzero-rotation `p`，均匀旋转的 mean absolute value `0.3p`。

22/22 gates 全通过；三族 rate=0 negative control 的 11 个指标全部精确为 0，所有不属于本族的交叉指标也
精确为 0。

## 5. 反简化与失败分支审计

`tests/test_ancilla_readout_causal.py` 的 23 项 direct/mutation tests 覆盖：

- 144 个 family×seed×rate cell 和 18 个 cluster summary 的完整成员关系；
- 拒绝删 family/rate/seed、降到 1,024 cycles 或少于 20,000 cluster bootstraps；
- exact one-channel config、paired stream identity、trace hash 和 balanced label schedule；
- zero controls、全部 cross-channel zeros、解析 rate 与方向性；
- truth/deployable schema、parent/code/source/CSV hash；
- 同步改写 phase logical path、同时改变 readout+bit、添加 global score、删 cell 和 stale parent 的 fail-close。

Source Data 共 1,960 行，逐 seed 保存 11 个主/交叉 estimand、intervention hash、trace hash、cluster CI、
parent/implementation/source bindings 与 22 个 gates。相邻 T2.2.2/T5.1.2/T5.1.6/T5.2.1 回归共
`139 passed`。

## 6. 允许与禁止 claim

允许：

- 当前 sBs effective overlay 下，三种 intervention 已做到一次只改变一个注册通道；
- bit、phase、readout 的协议原生敏感度方向、解析 rate 和交叉负控通过；
- phase flip 不切换 Z-basis outcome，readout error 通过 observed feedback/virtual-rotation 路径传播。

禁止：

- 把本项目概率、0.5 logical conditional、0.01 backaction 或 0.6 rad 上限写成目标装置标定；
- 把 phase 的零 logical truth 写成一般性的实验 logical-rate 定理；
- 声称复现 Sivak 实验的 65× bit/phase sensitivity；
- 把 evaluator logical backaction 写成 physical-memory LER、break-even、QPU 或 FPGA 实测；
- 把 readout-only 的错误 reset action 与 T5.2.3 的独立 reset-failure injection 混为一谈。
