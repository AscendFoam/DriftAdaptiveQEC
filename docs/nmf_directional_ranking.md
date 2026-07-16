# T2.3.7 PRL-inspired MF/NMF 方向性排名

**任务：** T2.3.7  
**状态：** PASS（有限截断、二能级 ancilla、10-cycle 方向性门）  
**正式合同：** `T237-V3-STRICT-SPLIT-AREA-EQUIVALENT-10-CYCLE-LIFETIME`  
**训练合同：** `T237-FEEDBACK-GRAPE-TRAIN-ONLY-EMA-BASELINE-V1`

## 1. 结论与适用范围

在同一个 finite-cutoff joint cavity--two-level-ancilla simulator、同一个 Puviani Table S1 `10 us` full-cycle timing、同一组 high-noise lifetime 和相同 10-cycle 物理时间下，5 个独立训练 seed 的 NMF agent 在 8 个独立 test seeds 上全部高于配对 MF agent，也高于 standard sBs：

- primary cutoff 12 的 projected-logical-Z area-equivalent lifetime：standard `2.7477`、MF `6.5347`、NMF `6.7408` cycles；
- NMF−MF 的配对 agent 均值差为 `+0.2061` cycles，20,000 次 bootstrap 95% CI `[0.0842, 0.3281]`；
- 5/5 个 NMF agents 高于其同 seed MF comparator；
- cutoff 16 独立 confirmation 仍保持 `NMF 7.7084 > MF 7.2459 > standard 5.1442`；
- physical target-state fidelity 的方向一致：cutoff 12 为 `NMF 8.5290 > MF 8.3148 > standard 3.5538`。

因此，本任务通过的是 **同模型内 10-cycle directional ranking**。它不是原论文 1000-cycle、六 Pauli eigenstates、完整 logical-channel exponential lifetime 的精确复现，也不证明 optimizer optimality、multilevel leakage/SPAM robustness、device calibration、实验 lifetime 或 FPGA latency。

## 2. 一手来源与实现对齐

正式文献记录为：

- Puviani *et al.*, *Phys. Rev. Lett.* **134**, 020601 (2025)；
- DOI `10.1103/PhysRevLett.134.020601`；arXiv `2312.07391`；
- 官方代码 `Matteo-Puviani/GQF`；
- Zotero item `IJ5EDGZF`，BibTeX key `puviani_non-markovian_2025`。

本实现复用论文 Supplement 的主要结构约束：

| 项目 | MF | NMF |
| --- | ---: | ---: |
| observation | 最新一次 `g/e` | 完整既往 `g/e` history |
| architecture | Dense 256--256--15 | GRU 10--Dense 256--256--15 |
| 参数量 | 70,159 | 72,853 |
| 输出 | 每 half-cycle 15 个 bounded residual controls | 同左 |
| 初始化 | nominal sBs residual，固定可复现 initializer | 同左，GRU recurrent matrix orthogonal |

15 个输出不直接替换 nominal controls，而是经 simulator 的硬边界形成 residual action：前 14 个 control residual 落在 `[-2,2]`，VR residual 落在 `[-1,1]`。这避免无界网络输出通过数值爆炸伪造 reward。

## 3. 仿真器与公平比较合同

三种主策略共享：

1. `physics/differentiable_sbs_trajectory.py` 的 joint `2N` density trajectory；
2. 四个 ancilla rotations、三个 complex ECD、固定 layer-4 displacement 和 VR；
3. Puviani Table S1 七段 idle，总计 `5 us`/half-cycle、`10 us`/full cycle；
4. cavity pure loss、two-level ancilla amplitude damping/dephasing；
5. 相同 cutoff、物理时间、噪声、初态和 held-out evaluation seeds；
6. physical target-state fidelity、code survival、projected logical-Z 和完整 density diagnostics。

`standard` 使用 nominal sBs controls；MF 与 NMF 使用相同 optimizer、learning rate、epoch、batch、validation/test budget。`nmf_latest_only` 不重新训练，而是在冻结 NMF 权重上每一步清零 hidden state，只保留最新 outcome，用作 history-reset 因果消融。

## 4. 严格数据拆分与 Feedback-GRAPE

正式配置：

| split | seeds | 用途 |
| --- | --- | --- |
| agent/train | `101,211,307,401,503` | model initialization；每 epoch 的 trajectory seed 由确定性 train-only 规则派生 |
| validation | `10007,10009` | 只选每个 agent 的 checkpoint |
| test | `20011,20021,20023,20029,20047,20051,20063,20071` | cutoff 12 最终结果，64 trajectories/seed，512/agent |
| confirmation | `30011,30013,30029,30047` | cutoff 16 独立确认，32 trajectories/seed，128/agent |

每个 MF/NMF agent 训练 300 epochs、batch 6，即每个 family 共 9,000 个 gradient-update trajectories；另有 train-only no-gradient warm-up batch 初始化 EMA baseline。5 个 MF 的 validation best epoch 为 `200--250`；5 个 NMF 为 `250--300`。模型只按 validation score 选择，test 和 confirmation 在全部 checkpoint 冻结后才运行。

训练目标保留 Feedback-GRAPE 两条路径：

\[
J(\theta)=\mathbb{E}[R]+\mathbb{E}[(R_{\mathrm{stop}}-b)\log P_\theta(\mathbf m)]
-\lambda_r\|\delta u\|_2^2-\lambda_s\|\Delta u\|_2^2.
\]

这里第一项对 selected conditional state 保留 reward path，第二项是 measurement likelihood score path；`b` 是 previous train-only EMA baseline。Adam learning rate 为 `1e-4`，gradient norm clip 为 `10`，residual/slew penalty 各 `1e-5`。

## 5. V2 反简化审计与 V3 修复

首轮 V2 正式运行虽然通过方向门，但审计发现 training baseline 初值来自 initial validation fidelity。常数 baseline 在无限样本期望下不改变梯度，但有限 batch 下会改变实际更新，因此 validation 不能称为纯 checkpoint-selection split。该 V2 JSON/checkpoint 已移到 `runs/t237_v2_validation_baseline_audit/` 并标记 invalid，未进入正式证据。

V3 做了以下修复：

1. baseline 只由独立 train-only no-gradient warm-up trajectory 初始化；
2. 记录 warm-up seed、每个 epoch 的真实 trajectory seed 和 validation-only 声明；
3. checkpoint gate 由 validation history 的 argmax 重新计算，不再硬编码 `True`；
4. non-exponential lifetime gate 从每条 raw curve 的 method flag 重新计算；
5. primary 与 confirmation 的 trace/Hermiticity/eigenvalue 全部进入 density gate；
6. schema-v3 checkpoint 同时绑定 config hash、两份 executable source byte hash 和 training protocol ID；
7. 逐模型 state-dict SHA-256、seed completeness 和最终 checkpoint SHA-256 进入必过门。

Windows 在第 8 个 agent 的原子 `os.replace` 上发生一次瞬时 `WinError 5`。临时文件已经完整写出；恢复前逐模型重算 hash，确认包含 5 MF + 3 NMF 且全部匹配，再从 schema-v3 checkpoint 恢复剩余 2 个 agents。最终 artifact 记录 `resumed_agents=8`、`newly_trained_agents=2`，没有丢弃或重选任何 agent。

## 6. 指标定义

10-cycle curve 有 correction transient，直接单指数 log-linear fit 的 `R^2` 可能很差甚至为负。它不适合作为短时域主 gate。主指标采用 finite-horizon area-equivalent lifetime：先计算 normalized AUC `A`，再求唯一的 `T` 使

\[
A=\frac{T}{H}\left(1-e^{-H/T}\right),\qquad H=10\ \text{cycles}.
\]

直接 log-linear lifetime 与 `R^2` 仍保留在 JSON 中作为“不应外推”的诊断，但不参与通过判据。主 gate 同时要求 projected logical-Z lifetime、logical-Z AUC 和 physical fidelity lifetime 方向一致。

## 7. 正式结果

### 7.1 Primary cutoff 12

| strategy | logical-Z lifetime | logical-Z AUC | fidelity lifetime | fidelity AUC | mean `p(g)` |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 2.7477 | 0.2675 | 3.5538 | 0.3340 | 0.6656 |
| MF | 6.5347 | 0.5114 | 8.3148 | 0.5814 | 0.8771 |
| NMF | **6.7408** | **0.5206** | **8.5290** | **0.5885** | 0.8745 |
| NMF, history reset | 6.0317 | 0.4876 | 7.5726 | 0.5547 | 0.8473 |

配对 agent logical-Z lifetime：

| train seed | MF | NMF | NMF history reset | NMF−MF |
| ---: | ---: | ---: | ---: | ---: |
| 101 | 6.5512 | 6.6105 | 5.8998 | +0.0593 |
| 211 | 6.4562 | 6.7576 | 6.1028 | +0.3013 |
| 307 | 6.4784 | 6.7356 | 6.1190 | +0.2572 |
| 401 | 6.6413 | 6.6638 | 6.1667 | +0.0225 |
| 503 | 6.5462 | 6.9364 | 5.8701 | +0.3902 |

NMF−history-reset 的配对均值差为 `+0.7091` cycles，95% CI `[0.5638, 0.9017]`。

### 7.2 Confirmation cutoff 16

| strategy | logical-Z lifetime | logical-Z AUC | fidelity lifetime | fidelity AUC | mean `p(g)` |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 5.1442 | 0.4406 | 6.0133 | 0.4872 | 0.7758 |
| MF | 7.2459 | 0.5411 | 9.0572 | 0.6048 | 0.8822 |
| NMF | **7.7084** | **0.5587** | **9.4610** | **0.6166** | 0.8841 |
| NMF, history reset | 8.2720 | 0.5781 | 9.9319 | 0.6294 | 0.8802 |

cutoff 16 确认了主任务要求的 `NMF > MF > standard`，但 history reset 在该 lane 反而高于 full-history NMF。它说明“完整历史相对最新 outcome 的优势”尚未跨 cutoff 稳健，不能用 primary PASS 掩盖；该反证交给 T3.2.11/T5.4.5 的 history truncation、shuffle、hidden reset 和长时域外推继续检验。

### 7.3 与论文机制的差异

原论文报告 MF 与 standard 大致相当、NMF 显著领先；本有限截断 10-cycle 模型中 MF 已远高于 standard，而 NMF 只在 MF 之上增加一个较小但配对为正的边际。因此允许声称“方向顺序被复现”，不允许声称“复现了论文增益幅度或同一物理机制”。

## 8. 数值、控制与非 demo 检查

- 10 个 agent 的 recorded training wall time 合计 `3208.03 s`，单 agent `317.38--330.04 s`；
- cutoff 12 每个 feedback agent 评测 `8 x 64 = 512` trajectories；
- cutoff 16 每个 feedback agent 评测 `4 x 32 = 128` trajectories；
- NMF primary/confirmation 的最小 agent-mean `p(g)` 为 `0.8526/0.8594`；
- NMF 最大 residual/slew RMS 为 `0.1774/0.0264`，远低于注册边界；
- 所有 primary/confirmation raw records 的最大 trace error `5.55e-16`、最大 Hermiticity error `0`、最小 final eigenvalue `0`；
- 14/14 required gates、31 个 focused implementation/artifact/figure tests 通过；
- JSON 约 `2.039 MiB`，schema-v3 checkpoint 约 `5.872 MiB`，不是只保存一个 best scalar 的 demo artifact。

## 9. Figure contract 与 Source Data

`docs/figures/t2_3_7_nmf_directional_ranking.*` 为 183 mm × 120 mm Python-only figure：

- a：held-out projected logical-Z curves；阴影为 standard 的 8 个 test seeds 或 feedback 的 5 个 independently trained agents 上 95% bootstrap CI；
- b：5 个配对 agent 的 MF/NMF/history-reset lifetime，黑横线为 median，灰线保留配对关系；
- c：cutoff 12 与独立 cutoff 16 confirmation；点为 median，error bar 为 agent IQR；standard 只有一个 seed-averaged estimate。

`docs/t2_3_7_nmf_directional_ranking.csv` 共 8,450 行，其中 6,336 行是 lane/strategy/training-seed/evaluation-seed/cycle 级 raw curves。SVG 保留 editable text，PDF 嵌入 TrueType 字体，TIFF 为 600 dpi。Source data are provided as a Source Data file.

## 10. 复现命令

```powershell
$env:PYTHONUTF8='1'
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m physics.nmf_directional_ranking `
  --artifact docs\t2_3_7_nmf_directional_ranking.json `
  --checkpoint docs\t2_3_7_nmf_directional_ranking_checkpoints.pt `
  --device cuda --cutoff 12 --confirmation-cutoff 16 --cycles 10 `
  --epochs 300 --train-batch 6 --validation-batch 24 `
  --test-batch 64 --confirmation-batch 32 --validation-interval 50 `
  --training-seeds 101,211,307,401,503 `
  --validation-seeds 10007,10009 `
  --test-seeds 20011,20021,20023,20029,20047,20051,20063,20071 `
  --confirmation-seeds 30011,30013,30029,30047
```

已有且合同一致的 schema-v3 checkpoint 会按 agent hash 恢复；config、source 或 training protocol 任一变化均 fail closed。

## 11. Claim boundary

允许：

> In a finite-cutoff, two-level-ancilla, high-noise literature-scenario simulator, five independently trained non-Markovian feedback agents directionally exceeded paired memoryless agents and standard sBs over a held-out 10-cycle evaluation, with the NMF-over-MF direction preserved at cutoff 16.

禁止：

- 论文 1000-cycle 六态 logical-channel lifetime 已精确复现；
- NMF 已达到最优或完整 history gain 已跨 cutoff/长时域稳健；
- MF/NMF 增益幅度或机制与原论文一致；
- 已覆盖 multilevel leakage、SPAM、pulse Hamiltonian 或 device calibration；
- 得到真实 quantum hardware、FPGA 或实时闭环 lifetime/latency 结论。
