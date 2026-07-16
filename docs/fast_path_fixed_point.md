# T4.2.4 Fast path 端到端定点化

## 1. 与 T2.4.3 的差异

T2.4.3 是旧单轴、one-window adaptive LUT stress model；它不消费 T4.2.1 的双轴 version-bound image，
也不经过 T4.2.2 event/frame 和 T4.2.3 health/fallback。T4.2.4 新增 `BitAccurateFastPath`，组合链为：

`integer ADC code -> version-bound X/Z MAP ROM/interpolation -> health/integrity -> event FSM -> Pauli/phase-frame action`

float syndrome/OOD 只允许经离线 replay adapter 转成 code；在线 `step_codes` 只接收整数、布尔、固定 digest
和 observed class，AST 不含 float division、`exp/log/sqrt`。

## 2. Selected word contract

| 字段 | 位宽/格式 | 规则 |
| --- | --- | --- |
| syndrome ADC | unsigned 10 bit | half-open cell，floor 到 bin；越界/非有限只为 trace 生成 clipped code，同时 observation invalid |
| LUT address/fraction | 8/2 bit | 257 guard-node/phase；half-bin numerator |
| LLR | signed Q9.12，22 bit | compile 与 interpolation 均 nearest-even，signed saturation；`code<0` 才 flip |
| event mode/counters | 3 bit / 6×3 bit | 六态；unsigned saturation，不回绕 |
| Pauli/phase frame | 2×1 bit / 2×8 bit | GF(2) XOR；modulo-256 half-turn add |
| OOD/parameter age | unsigned 8/16 bit | OOD nearest-even unit code；age unsigned saturation |
| version/fault mask | unsigned 16/14 bit | exact/monotonic comparison |
| health counters | 18×8 bit | fault/good、累计 fault/leakage、14 个 per-flag，全部 saturation |
| integrity | CRC32 + SHA256 | exact equality，不作数值量化 |

selected representation proxy：每 bank 双 phase ROM `11,308` bits，double-bank ROM `22,616` bits，8-bank
artifact table `90,464` bits；event live state `55` bits，health state/input `182` bits，image integrity metadata
`288` bits。它们是精确表示账本，不含 BRAM packing/routing/control overhead，不是综合资源。

## 3. 四档精度与 exhaustive code audit

| profile | ADC/address/LLR | exhaustive rows | mean/max LLR value error | hard mismatch |
| --- | --- | ---: | ---: | ---: |
| low | 6/4/Q5.6 | 1,024 | `1.78986e-2 / 0.53125` | 0 |
| medium | 8/6/Q7.10 | 4,096 | `1.11079e-3 / 0.0732422` | 0 |
| selected | 10/8/Q9.12 | 16,384 | `9.46671e-5 / 0.00488281` | 0 |
| dense | 12/10/Q10.14 | 65,536 | `1.34995e-5 / 0.000305176` | 0 |

总计 87,040 code rows 覆盖 4 profiles×8 banks×X/Z×完整 ADC domain。所有 profile 在 ADC centre 上 hard
action mismatch 为零；这不等于低精度无影响，因为 float syndrome 到粗 ADC code 仍会在 decision boundary
附近改变 action。

## 4. Paired LER impact

每 profile 使用相同 8 banks×4 seeds×2,048 samples 的 model-matched raw displacement，共 128 cluster rows。
exact-float 和量化链共享 raw trace/truth；truth 只存在于离线 evaluator，不进入 `FastPathCodeInput`。

| profile | quantized LER | float LER | action disagreement | paired ΔLER [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| low | `0.0397644` | `0.0395966` | `1.11389e-3` | `+1.67847e-4 [-7.629e-5, 4.272e-4]` |
| medium | `0.0395660` | `0.0395966` | `2.13623e-4` | `-3.05176e-5 [-1.373e-4, 6.104e-5]` |
| selected | `0.0396271` | `0.0395966` | `9.15527e-5` | `+3.05176e-5 [-4.578e-5, 1.221e-4]` |
| dense | `0.0395966` | `0.0395966` | `0` | `0 [0,0]` |

selected CI 跨零，不能声称量化提高或降低 LER；只可称当前 model-matched finite trace 上影响受限。low 的
CI 也跨零，尽管 disagreement 更大。健康 replay 中所有 MAP 均被接受、fallback 为零、双轴 frame 均更新，
source-to-action 固定 6 cycles、II=1。

## 5. 验证与边界

21/21 gates；production 过程独立重编译并重跑两次，再比较 87,040-code 与 128-LER rows 的 canonical
hash。NaN/out-of-cell、OOD 192/193 code、age overflow 均有显式 saturation/fallback provenance。

当前是 model-matched axis-marginal software reference；没有 correlated 2D optimum、OOD/fault 条件 LER、
device ADC/calibration、RTL、synthesis/post-route、FPGA 或 board evidence。LUT/FF/BRAM/DSP/Fmax 保持
`null`，T5/T6 之前不得把 representation bits 写成硬件利用率。

