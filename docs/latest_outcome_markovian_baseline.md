# T3.2.7 精确预算 latest-outcome Markovian feedback baseline

## 1. 结论

在与冻结 history GRU 完全相同的 `72,853` 参数、`72,266` dense MAC、15 维 sBs
residual action、5-agent training seeds、300 epochs、validation/test/confirmation split 和
held-out trace 下，新的 latest-outcome FNN 推翻了旧弱 MF 所暗示的稳定 memory gain：

- cutoff 12：exact MF logical-Z lifetime 为 `6.888249` cycles，高于 history NMF 的
  `6.740785`；`NMF - exact MF = -0.147464 [-0.386866, 0.147532]`，CI 跨零；
- cutoff 16：history NMF 为 `7.708351`，exact MF 为 `7.168269`；从 artifact 中五个
  配对 agent 值按同一 20,000 次 bootstrap 重算，`NMF - exact MF =
  +0.540082 [0.231972, 0.785521]`；
- cutoff 方向反转，因此当前证据不支持“memory 在该模型中稳健提高 lifetime”；它只证明
  memory effect 对 finite cutoff/model lane 敏感；
- exact MF 相对旧 70,159 参数 MF 在 cutoff 12 提高 `+0.353578
  [0.084842,0.571642]` cycles，说明旧 `NMF > MF` 排名部分来自 comparator 容量/结构不足。

这是一项保留的负结果，不通过删除 simple baseline 或要求正向 gate 来维持 NMF claim。

## 2. 信息集与结构

canonical learned API 为 `forward_latest(latest_outcomes)`，输入仅是当前 observed token：

- `g = 0`；
- `e = 1`；
- `leakage = 2`；
- 第一 half-cycle 使用全零 start token。

simulator adapter 虽接收 history tensor，但源码只读取 `history[:, -1]`。测试对相同最新
outcome、不同任意前缀做独立等形状调用，输出 bit-exact；模型无 hidden/rollout state。

| 策略 | 输入 | front-end | 总参数 | dense MAC / half-cycle | 输出 |
| --- | --- | --- | ---: | ---: | --- |
| 旧 MF | latest binary g/e scalar | 1→256 dense | 70,159 | 较低但未匹配 | 15 residuals |
| exact MF | latest g/e/leakage token | static 390-param / 330-MAC front | **72,853** | **72,266** | 15 residuals |
| history NMF | full causal g/e history | GRU(1,10), 390 params / 330 MAC | **72,853** | **72,266** | 15 residuals |

exact MF 的 390 参数 front 不是 dummy padding：

- `Linear(3,10)`：40；
- `Linear(10,15)` + `Linear(15,10)`：325；
- `LayerNorm(10)`：20；
- 五个 feature-pair scales：5；
- 合计 390，所有 tensor 都进入 forward。

三种 token 的直接梯度测试覆盖全部 72,853 个参数元素。正式二能级 simulator 不产生
leakage，所以每个 agent 恰有 `outcome_encoder.weight[:, leakage]` 的 10 个元素在实际训练
中保持未观测；artifact 明确记录 `72,843/72,853` 覆盖和这一原因。

## 3. 冻结训练与评测合同

从 T2.3.7 PASS artifact/checkpoint 加载 frozen history NMF 与旧 MF，并逐源文件、checkpoint、
单模型 state hash 复核。exact MF 使用相同：

- cutoff 12 primary，cutoff 16 confirmation；
- 10 full cycles，float64/complex128 physics；
- 5 个 agent seeds：`101/211/307/401/503`；
- 300 epochs，batch 6，Adam `1e-4`；
- 2 个 validation seeds，只用于 checkpoint selection；
- 8 个 test seeds × 64 trajectories；
- 4 个 confirmation seeds × 32 trajectories；
- 同一 Feedback-GRAPE reward path + score path、EMA train-only baseline、residual/slew penalty；
- 同一随机 seed trace、同一 100 us physical horizon、同一 15 维 nominal-residual action。

5 个 exact MF best epochs 为 `300/300/100/300/250`；3/5 位于训练上限，仍不能声称 optimizer
convergence。所有 agents 均保留，没有按 test lifetime 选择最好一套。

## 4. 主结果

### 4.1 Cutoff 12 primary

| 策略 | logical-Z lifetime | logical-Z AUC | fidelity lifetime | p(g) | residual RMS | slew RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 2.747662 | 0.267471 | 3.553781 | 0.665625 | 0 | 0 |
| 旧 MF | 6.534671 | 0.511419 | 8.314757 | 0.877070 | 0.159357 | 0.062788 |
| exact MF | **6.888249** | **0.526542** | **8.719894** | 0.829512 | 0.167119 | 0.047274 |
| history NMF | 6.740785 | 0.520568 | 8.528953 | 0.874531 | 0.164489 | **0.020140** |

exact MF 的 lifetime/AUC 更高，但 p(g) 低于 NMF，NMF 的 parameter slew 明显更平滑。单一
lifetime 指标不能抹去 experimental burden trade-off。

### 4.2 Cutoff 16 confirmation

| 策略 | logical-Z lifetime | logical-Z AUC | fidelity lifetime | p(g) |
| --- | ---: | ---: | ---: | ---: |
| standard | 5.144172 | 0.440632 | 6.013342 | 0.775781 |
| 旧 MF | 7.245903 | 0.541062 | 9.057210 | 0.882188 |
| exact MF | 7.168269 | 0.538059 | 8.942740 | 0.828906 |
| history NMF | **7.708351** | **0.558686** | **9.460991** | 0.884063 |

该 lane 的 NMF 优势显著，但它与 cutoff 12 的排序相反。T3.2.11 必须在多个 cutoff 上继续
shuffle/truncation/reset，不能只选择支持 memory 的一侧。

## 5. 反简化审计

完成的门包括：live parent source/checkpoint、严格 seed split、精确参数/MAC、15 维动作、
latest-only 行为、三 token、实际梯度覆盖、五模型 checkpoint hash、held-out 冻结评测、density
数值诊断、signed comparison 与 leakage evidence boundary。正式 artifact 为 13/13 PASS，Source
Data 共 18,023 行，保留 1,500 个 training epochs、全部 validation points、agent/seed/cycle raw
curves 和数值健康指标。

## 6. 复现

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.benchmark.latest_outcome_markovian_baseline `
  --artifact docs\t3_2_7_latest_outcome_markovian_validation.json `
  --checkpoint docs\t3_2_7_latest_outcome_markovian_checkpoints.pt `
  --source-data docs\t3_2_7_latest_outcome_markovian_source_data.csv
```

已有合同一致 checkpoint 时会逐模型 hash 后恢复；配置、parent evidence、simulator 或训练/报告
源码变化均 fail closed。

## 7. Claim boundary

允许：finite-cutoff two-level simulator 中 exact-budget latest-outcome MF 与 history NMF 的 signed
comparison，以及 cutoff-sensitive memory effect。

禁止：稳定 memory mechanism、论文 1000-cycle 六态 lifetime 精确复现、multilevel leakage
robustness、optimizer optimality、真实装置、RTL、FPGA latency/resource/performance。

