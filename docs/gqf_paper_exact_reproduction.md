# T6.8.4：Puviani GQF paper-exact reproduction 审计与执行结果

## 结论

T6.8.4 已完成，但结论为：

`COMPLETE_GQF_PAPER_EXACT_ATTEMPT_NO_GO_SOURCE_INCOMPLETE`

这不是“复现失败后继续称方向一致”，而是 exact 前置证据不足，按预注册停止规则不启动不可审计的
20-agent full training。当前只建立了官方 simulator 的 reduced standard-path 诊断；
paper-exact、MF/NMF ordering 和“超过 Puviani NMF”均禁止。

一手来源为 [PRL 论文](https://doi.org/10.1103/PhysRevLett.134.020601)、
[arXiv 2312.07391](https://arxiv.org/abs/2312.07391) 和
[官方 GQF 仓库](https://github.com/Matteo-Puviani/GQF)。本地论文/补充材料与固定 commit 均有
SHA-256 绑定。

## 论文预注册规格

从正文和补充材料冻结的核心规格为：Fock cutoff 100；训练态 `Delta=0.2`；Fig. S2/Sivak2023
时序；10 full cycles/20 measurements；1000 epochs；20 agents；batch 6 或 8；学习率 `1e-4`；
15 个 controls；GRU 架构 `[10,256,256,15]`；六逻辑态、1000-cycle evaluation；NMF 510、
standard 512 条轨迹；单指数 `T_X/T_Y/T_Z` 拟合与调和 `T_ch`。

论文中可直接冻结的标量锚点包括低噪声 `T_Z(std)=700 cycles`、`T_Z(NMF)=1500 cycles`、
`T_-Y(NMF)=770 cycles`，以及 entanglement infidelity `1.5e-3/4.3e-4`。完整 strategy×noise 的
`T_X,T_Y,T_Z,T_ch` 数值表没有公开，不能从图像读取后冒充精确 ground truth。

## 18 个 exact 阻断项

最关键的阻断项包括：

- 官方仓库没有论文所述 20 个 agent 的 checkpoint、seed、selection ledger 或 raw trajectories；
- 论文 RNN `[10,256,256,15]`，官方实现却是 `[30,30,30,15]`；MF 也从论文
  `[256,256,15]` 变成 `[30,30,15]`；
- 论文训练 1000 epochs，官方 runner 为 101，class default 为 500；
- 论文主要 agent 训练 `Delta=0.2`，runner 为 `0.34`，论文不同段落对 evaluation Delta 也不唯一；
- 论文 bias 描述本身存在 `0.01`、`[-0.1,0.1]` 两种口径，官方仅 output layer 使用
  `[-0.01,0.01]`；
- runner 传 `RNN`，class 只接受 `rNN`；`previous_reward` 从不更新，不能实现论文所述 best-agent
  选择；
- 官方没有 20-agent orchestration、六态 1000-cycle evaluator、lifetime fitter、fit window 或
  weighting；
- `max_steps=21` 在官方环境实际执行 21 个 `env.step`，与论文 20 half-cycles/10 cycles 的描述不一致；
- 官方 HEAD 有语法缺陷且依赖未固定；当前主机的 full GQF GPU state path 发生 cuSolver fatal。

完整 18 项差异保存在 `docs/t6_8_4_gqf_paper_exact_reproduction.json`。

## 降级的真实执行证据

为了避免只做静态代码审计，在 T6.8.3 的 patched official worktree 上执行了：

- strategy：standard sBs，`TEST=True`；
- cutoff 8、`Delta=0.34`、low noise、Sivak2023；
- 六逻辑态、3 seeds、batch 2；
- official `max_steps=21` 的完整行为；
- 36 条批轨迹、378 次环境 step、756 行 Source Data；
- wall-clock `218.58 s`。

第一次运行按论文 20 half-steps 预期锁定 720 行，官方实际返回 756 行，因此主动失败。第二版不删除
额外数据，而是同时保留第 20 步 paper-prefix 和第 21 步 official terminal。所有概率范围和 density
trace 检查通过。

cutoff 8、36 条轨迹、无训练的结果不用于 lifetime 或 MF/NMF ordering；其作用仅是证明 official
多态长路径不是 import-only smoke，并提供当前 CPU 成本下界。

## 反简化检查

- 13/13 integrity gates 通过；
- 15/15 exact gates 失败，不能由总 PASS 字符串掩盖；
- 13/13 target-specific mutations 被拒绝；
- 20 个缺失 agents 均逐行记录，seed/checkpoint/所有寿命指标为 `null`；
- 三个 exact strategy 的 `T_X,T_Y,T_Z,T_ch,F_avg` 均为 `null`；
- 相关 intake/reproduction/governance tests 合计 `21 passed`；
- live paper、official tree、patch/runner manifest、probe code、probe artifact、prereg 和 CSV 均 hash-bound。

## 后续约束

T6.8.5 的“同 GQF 环境 matched comparison”前置条件未满足，因此不得执行或宣称 Route-A 对 NMF
的 lifetime superiority。T6.8.5 只能完成其预注册的 negative branch：记录 ineligible、禁止比较，
并把可恢复条件写入 claim matrix。

