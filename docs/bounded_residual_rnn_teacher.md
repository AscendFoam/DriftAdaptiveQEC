# T4.4.1：fresh bounded-residual GRU teacher

## 结论

T4.4.1 已训练并冻结一个**全新**的 72,853 参数
`GRU10-Dense256-Dense256-Out15` offline teacher。正式运行使用三个新 restart seed，未加载、复制或重命名
T2.3.7 的任何 state dict；checkpoint、训练曲线、validation-only 选择、held-out evaluation 和 cutoff-16
confirmation 均有 hash-bound 机器证据。21/21 production gates 通过。

该结果只支持 finite-cutoff、two-level、10-cycle differentiable sBs 模型内的 bounded-residual teacher；不支持
optimizer 全局收敛、论文 1000-cycle 六态 channel lifetime、multilevel leakage/SPAM/pulse、在线 FPGA 或真实装置结论。

## 动作与初始化合同

teacher 每个 half-cycle 输出 15 个 raw residual coordinates，物理动作统一为

\[
\boldsymbol u_t=\boldsymbol u_{\mathrm{nominal}}
+\boldsymbol b\odot\tanh(\boldsymbol r_t),
\]

其中前 14 个 residual bound 为 `2.0`，virtual rotation bound 为 `1.0`。因此：

- raw residual 为零时严格得到 nominal sBs，不是全零物理门参数；
- 所有有限 raw output 都被解析地限制在 hard box 内；
- 全零物理门向量并未被偷换成 initializer，且 `layer2_beta_real=2.5066` 的 nominal 幅度已经超过其
  `2.0` residual bound，所以全零物理向量不在完整安全 box 内；
- 三个 fresh initializer 的最大归一化 residual 都小于 `0.0011`，确实从 nominal 邻域开始。

## 严格 split 与训练来源

| 用途 | seeds / 规模 | 是否参与选模 |
| --- | --- | --- |
| fresh restart | `601, 709, 811`；各 320 epochs×6 trajectories | 产生梯度 |
| validation | `41011, 41017`；每 seed 24 trajectories | 只选 checkpoint |
| primary evaluation | 8 个独立 seeds；每 seed 64 trajectories | 否 |
| cutoff-16 confirmation | 4 个独立 seeds；每 seed 32 trajectories | 否 |

四组 seeds 两两不交，restart seeds 也与 T2.3.7 的 `101/211/307/401/503` 不交。三个 initial/final
state SHA 均互异并与五个旧 NMF state SHA 不交；`resumed_restarts_this_invocation=0`。

## 训练结果与未隐藏失败

| restart | initial validation score | best score | best epoch | 结论 |
| ---: | ---: | ---: | ---: | --- |
| 601 | 0.303499 | 0.587632 | 320 | success；触及 budget cap |
| 709 | 0.303363 | 0.585264 | 280 | success；未触及 cap |
| 811 | 0.302319 | 0.586702 | 320 | success；触及 budget cap |

validation-only 选择 restart 601。三个 restart 都超过 `+0.10` validation gain gate，没有失败 restart；但
601/811 的 best epoch 位于 320 上限，artifact 明确记录 `training_cap_hit_indices=[0,2]`，并将所有 restart 的
`optimizer_global_convergence_claimed` 固定为 `false`。通过 task gate 不等于证明全局或渐近收敛。

## Held-out 数值

| lane | selection score | fidelity lifetime | logical-Z lifetime | `p(g)` |
| --- | ---: | ---: | ---: | ---: |
| cutoff 12 nominal | 0.309941 | 3.650224 | 2.876278 | 0.671484 |
| cutoff 12 selected teacher | 0.563544 | 8.743995 | 7.001796 | 0.881641 |
| cutoff 16 nominal | 0.455477 | 5.859962 | 5.006519 | 0.765234 |
| cutoff 16 frozen teacher | 0.597035 | 9.637760 | 8.042886 | 0.886719 |

selected teacher 相对 nominal 的 primary/confirmation score gain 为 `0.253603/0.141557`。primary 八个
seed 的 logical-Z lifetime 配对 bootstrap 差为 `+4.125518` cycles，95% CI
`[3.899115, 4.339331]`。这只是当前 assumed-model finite-horizon 证据，T4.4.4 才负责 teacher、student、
MF、standard 与 control oracle 的统一同轨 gain-retention gate。

## 非 demo 审计

- 三个 72,853 参数模型都执行 320 个真实 Feedback-GRAPE epoch；不是用旧 checkpoint 生成标签。
- 每个 epoch 的 reward、score path、gradient norm、control residual/slew 与 train-only EMA baseline 进入
  1,074-row Source Data；validation、24 组 held-out seed-metric 和 15 个 action bounds 也全部保留。
- selected teacher 在全部 256 个 8-bit histories 上接受梯度审计；每个参数张量的梯度均 finite/nonzero，最小
  nonzero fraction 为 `1.0`。
- full-history replay、cached GRU step、不同 future suffix 的共同 prefix 和重复 replay 最大误差均为 `0`。
- checkpoint reload 的 state SHA 与 selected state 完全一致，probe output 另有独立 SHA。
- demo-scale pilot 会返回 `FAIL`，不能取得 production teacher 身份。

## 产物与复现

- 训练器：`cnn_fpga/benchmark/bounded_residual_rnn_teacher.py`
- 测试：`tests/test_bounded_residual_rnn_teacher.py`
- 机器报告：`docs/t4_4_1_bounded_residual_rnn_teacher_validation.json`
- checkpoint：`docs/t4_4_1_bounded_residual_rnn_teacher_checkpoints.pt`
- Source Data：`docs/t4_4_1_bounded_residual_rnn_teacher_source_data.csv`

正式复现需要 DLEnv/CUDA：

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.benchmark.bounded_residual_rnn_teacher
```

若 gate 失败，teacher hidden-state/student gain 主张停止，主线回退到已经成立的 drift/regime-aware MAP-LUT；
不得用文字解释覆盖失败结果。
