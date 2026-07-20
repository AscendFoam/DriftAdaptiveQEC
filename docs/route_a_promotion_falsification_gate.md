# T6.7.4 Route-A promotion / falsification gate

## 结论

机器判定为：

`GO_ROUTE_A_CONTRACT_SYSTEM_RESTRICTED_SIMULATOR_AND_PREBOARD_CLAIMS`

这不是“Route-A 是最优解码器”的 GO，而是一个范围受限的系统晋级：在冻结的 simulator、协议、EWMA primary baseline 和 V4 阈值下，smooth aggregate 主门、abrupt/OOD/nominal safety 门以及百万周期 integrated correctness 门同时通过，因此可以把 **contract-centric、regime-aware safe adaptive dual-loop** 作为论文系统主语。global deployable LER、static GKP superiority、tail improvement、HMM-on-FPGA 和 measured FPGA speed 均没有被晋级。

## 独立性与证据绑定

- 未重跑、删除或重加权 formal scenario，也未重选 baseline/阈值；
- 从 T6.7.1 的 576 条和 T6.7.2 的 888 条逐 trajectory 原始 Pauli/window/paired counts 重新执行完整分析与 20,000 次 formal-seed cluster bootstrap；
- 逐字节复核 498,240-row smooth CSV、686,104-row tail CSV 与 131,000,000-byte million-cycle binary trace；
- 重新计算 T6.7.1、T6.7.2、T6.7.3 各自 evidence gates 和当前 source hashes；
- 输出报告在读取时再次核对父 artifact、Source Data、trace、实现源码和本任务派生 CSV，文件漂移会 fail closed；
- 8/8 semantic mutations 覆盖父哈希、Source Data、primary LCB、tail gate、RTL mismatch、baseline reselection、claim deletion 与错误 global promotion。

## 预注册主门

| 门 | 结果 | 解释 |
| --- | --- | --- |
| locked-EWMA smooth aggregate | PASS | `EWMA LER - Route-A LER = 2.1687e-5`，95% CI `[1.9003e-5, 2.4548e-5]` |
| abrupt/OOD catastrophic | PASS | 六个 family 全部在预注册 margin 内；不能解释为全部获得改善 |
| calibration strict counterexample | PASS | Route-A / EWMA global worst 均为 `181/512`，不是原先的 `55/512 > 37/512` |
| nominal non-inferiority | PASS | average 与 induced-minus-avoided 不劣；fallback `0.119%` |
| integrated long RTL | PASS | 10×100,000 cycles、0 mismatch、0 undefined、0 CRC、0 silent overflow |
| evidence/integrity/claim registry | PASS | 总计 10/10 promotion gates |

## 同时成立的证伪结果

### 1. 不是最强 deployable smooth decoder

equal-family/equal-seed formal average LER 为：

| 方法 | average LER |
| --- | ---: |
| Window MAP | `8.96419e-4` |
| static joint MAP | `9.68191e-4` |
| standard binning | `9.78611e-4` |
| proposed Route-A | `9.92740e-4` |
| locked EWMA | `1.01443e-3` |
| Kalman | `1.78574e-3` |

所以 Route-A 只相对锁定 EWMA aggregate 改善；它没有超过 Window 或 static joint MAP，static-to-oracle gap closure 仍为负。四个 smooth family 中只有 `periodic_drift` 通过 Holm family-wise discovery。

### 2. tail 门是 safety / non-inferiority，不是 tail improvement

六个 abrupt/OOD family 中没有任何一个取得“proposed-minus-EWMA average 95% UCB < 0”的确认性改善。step、telegraph、readout/reset、leakage、compound 的 average 差恰为零；burst 只出现极小的 avoided/induced 差异。高 fallback、unnecessary fallback 与 truth-side false-update 仍须在正文呈现。

### 3. CNN、HMM 与 FPGA 结论

- legacy CNN checkpoint 确实执行，但输入 schema 和 `3,489,984 MAC / 1,586,368 B` matched budget 失败，只保留消融；
- HMM/posterior 是软件 slow loop，RTL 只执行量化后的 action/state/version 与 fast path；
- CXXRTL host runtime 不是板上 latency，Yosys structural check 不是 P&R、资源、功耗或板测；
- “比已有 FPGA decoder 更快”仍禁止，直到 T6.8.6 与 T6.9.2 的 same-task real-board 门通过。

## 分支判定

- primary smooth average 门通过，因此不触发 `static MAP-LUT + deterministic FPGA` 的全退化分支；
- tail safety 门通过，因此不触发 smooth-only 分支；
- CNN matched gate 失败，因此 CNN 自动降为 ablation-only；
- 由于 Window/static 反例成立，Route-A 的晋级范围固定为合同系统与 locked-EWMA contrast，不形成 global rank。

## 产物与复现

- runner：`cnn_fpga/benchmark/route_a_promotion_gate.py`
- machine report：`docs/t6_7_4_route_a_promotion_gate.json`
- Source Data：`docs/t6_7_4_route_a_promotion_gate_source_data.csv`
- tests：`tests/test_route_a_promotion_gate.py`

运行：

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.benchmark.route_a_promotion_gate
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests/test_route_a_promotion_gate.py
```

当前验证：`10/10 gates`、`8/8 semantic mutations`、`7 passed`。

