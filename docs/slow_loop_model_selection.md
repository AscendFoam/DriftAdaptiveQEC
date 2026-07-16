# T4.1.1 匹配预算慢回路模型选型

## 结论

在冻结的 synthetic four-regime pilot 上，validation-only 规则选择
`gaussian_hmm`，不是 causal TCN。HMM 的 validation/evaluation NLL 分别为
`0.454975/0.455711`；validation runner-up causal TCN 为
`0.476180/0.511936`。按 8 个 evaluation seed 聚类，runner-up-minus-HMM NLL
为 `+0.056225 [0.046709,0.065742]`。该结果只冻结 T4.1.1 pilot backbone，
不证明 HMM 对 T4.1.2 的 richer history、OOD regime 或真实装置普遍最优。

## 公平比较合同

- 共同任务：normal/burst/leakage/calibration-shift 四类 posterior；
- 共同输入：T3.2.6 每 32 cycle 形成的 14 项 observed summary；
- 共同 history：严格最近 8 个窗口，禁止 hidden regime、future、evaluation label
  与 logical truth；
- 共同 split：3 training、3 validation、8 evaluation seeds，互不重叠；
- 共同上限：每次更新不超过 4096 MAC、4096 B 常驻 float32 模型/状态、4096 B
  transient workspace；host latency 只是同机 software diagnostic；
- 选择规则：先排除越预算模型，再按 validation NLL、validation Brier、MAC、
  参数量、family name 词典序决胜；evaluation 不参与选择。

T4.1.2 尚未完成的 analog/action/LLR/deadline 等 richer history 不在本任务内；
这里复用已注册的 T3.2.6 共同任务，避免拼接不同 T3 metric 伪造 winner。

## 候选与结果

| family | validation NLL | evaluation NLL | evaluation accuracy | evaluation Brier | MAC/update | 常驻 B | scratch B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian HMM | 0.454975 | 0.455711 | 0.837376 | 0.239679 | 926 | 3728 | 104 |
| causal TCN | 0.476180 | 0.511936 | 0.829208 | 0.256349 | 3556 | 2508 | 464 |
| small GRU | 0.503134 | 0.509619 | 0.829455 | 0.258132 | 2300 | 1916 | 156 |
| exponential recurrence | 0.539824 | 0.595195 | 0.813119 | 0.286238 | 504 | 1084 | 72 |
| diagonal Kalman | 0.545693 | 0.560321 | 0.813366 | 0.277373 | 1064 | 1248 | 184 |
| run-length FSM | 0.785817 | 0.824837 | 0.714356 | 0.445747 | 1408 | 1044 | 72 |

validation 排序为 HMM、TCN、GRU、指数递推、Kalman、FSM；evaluation 中 GRU/TCN
和 Kalman/指数递推发生局部换位，但 HMM 仍第一。HMM 的 transition detection delay
为 `1.8992` windows，明显慢于 TCN 的 `1.5505` 和 Kalman 的 `1.3269`，所以本任务
只按预注册 primary NLL 选择，不能删除 latency/delay trade-off。

## 实现与反简化审计

- causal TCN 固定 14×7 两层 causal convolution；small GRU 固定 hidden size 5；
  两者各训练 5 个独立 restart、最多 240 epochs、validation early stopping 和
  validation-only temperature calibration，保存 selected state dict 与全部 restart 记录；
- HMM、Kalman、指数递推和 FSM 都有完整 training/validation grid，而不是手填参数；
- 14 条 simulator trajectory 共 229,376 cycles；evaluation 形成 4,040 个独立
  bounded-history prediction，六模型长表共 24,240 rows；
- 初版 HMM 资源画像在深审计中会因朴素重放 8 个 emission 而低估 MAC；已实现
  `RollingGaussianHMMAdapter`，只计算新窗口 emission，缓存 8×4 emission vector 后
  重放四态 forward recursion。多个截断点与朴素 last-8 replay 在 `1e-13` 内一致；
- FSM 极端 posterior 曾因 `1-p_active` 消去误差失败，已改为直接累加非 active mass
  并归一化；没有放宽 posterior gate；
- JSON、24,240-row CSV 和 checkpoint 均保存 SHA-256/implementation binding；13/13
  生产 gates 与 33 focused tests 通过。

## 证据边界

允许表述：在注册的 synthetic four-regime、8-window、匹配预算 pilot 中，validation
选择 HMM，evaluation 保持第一。禁止表述：HMM 普遍优于神经网络、T4.1.2 rich input
已完成、获得 logical-error/control gain、设备鲁棒性、bit-accurate、综合、FPGA latency
或量子实验结果。

