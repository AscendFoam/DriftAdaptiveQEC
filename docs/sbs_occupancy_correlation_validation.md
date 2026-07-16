# T2.0.6 sBs occupancy 与 leakage-correlation 交叉验证

**日期：** 2026-07-14  
**实现：** `physics/sbs_occupancy_correlation.py`  
**机器结果：** `docs/t2_0_6_occupancy_correlation.json` / `docs/t2_0_6_correlation_tail.csv`  
**证据口径：** `protocol_aligned_occupancy_correlation_effective_model_not_device_calibrated`

## 1. 两条独立证据链

Sivak S4E 对 observed syndrome string 使用

```text
P([gg]^n) = a lambda^n
p_err ≈ 1-lambda
<Pi0> ≈ a lambda
```

并把一阶近似的系统误差估为 `p_err^2`。本实现的 estimator 只接受 `observed_all_gg_boolean_matrix`；hidden code state 不在函数参数中，只用于独立 truth comparison。

S4F 的 correlation 验证则是另一条证据链：对 observed full-cycle non-gg activity 计算 lag correlation，再丢弃任何出现连续 `>=2` leakage cycles 的整条 trajectory。该操作是离线 post-selection，不是在线 leakage controller。

## 2. Shared trajectory 与 source-inspired 参数

- production scale：`600 shots × 1200 cycles`，另有 `200` burn-in cycles；
- simulation seed `2026071407`，bootstrap seed `2026071408`，400 次 95% shot-cluster bootstrap；
- physical error `0.13/cycle`，no-leakage target occupancy `0.82`，由二态平衡关系得到 recovery probability `0.592222`；
- total leakage start rate `6.76e-4/cycle`，其中 single-cycle `f` 为 `5.48e-4`、length-`>=2` higher 为 `1.28e-4`；
- higher event source-inspired mean duration `17.2 cycles`；
- readout diagonal 使用 primary `F_g=0.9997`、`F_e=0.9914`；f/higher 精确归类 leakage 是显式 assumption；
- higher leakage 时 ancilla inactive，每周期增加一个 coarse recovery level；这是 effective error-depth law，不是 Fock-space 动力学。

truth lane 保存 start-of-cycle depth、code occupancy 和 f/higher kind；observed lane只保存 X/Z classified syndrome、all-gg、non-g activity 和 observed leakage。测试锁定 `code=(depth==0 and no leakage)`，避免更新顺序导致 truth 自相矛盾。

## 3. Occupancy 结果

| 方法 | point estimate | 95% statistical CI | first-order / combined |
| --- | ---: | ---: | ---: |
| hidden truth | 0.813565 | [0.811663, 0.815456] | — |
| syndrome-only `a lambda` | 0.813524 | [0.811288, 0.815992] | `p_err^2=0.017427`; combined [0.793861, 0.833419] |

拟合得到 `lambda=0.867989`、`p_err=0.132011`、log-probability `R^2=0.99999987`。hidden 与 syndrome point difference 为 `4.10e-5`；hidden truth 同时位于 syndrome statistical CI 和包含一阶模型误差的 combined CI 内。

该一致性来自同一批 trajectory 的独立视图，不等于真实实验 Wigner tomography。文献的 `0.825±0.003` 只作 trend/reference anchor，本任务没有重建实验 density matrix。

## 4. Leakage correlation 结果

注册 long lags 为 `40,60,...,200 cycles`。observed leakage-run post-selection 保留 `507/600=0.845` trajectories。

| 指标 | point | 95% CI |
| --- | ---: | ---: |
| mean tail before removal | 0.002976 | [0.001069, 0.004924] |
| mean tail after removal | -0.000192 | [-0.001166, 0.000715] |
| paired before-after | 0.003168 | [0.001684, 0.005058] |

按 absolute post-removal noise floor 计算 point shrink ratio 为 `15.52`。higher-leakage-rate 置零的因果消融得到 retained fraction `1.0`、before=after、paired difference `0`，相应 shrink gates 明确失败；因此 PASS 不是 post-selection 代码自动生成的。

## 5. 预注册 gates

11 个 gate 全部 PASS：

- hidden occupancy 距 `0.82` 不超过 `0.02`；
- syndrome/hidden point difference 不超过 `0.02`，且 combined CI 包含 truth；
- estimated/configured `p_err` 差不超过 `0.02`；
- all-gg fit `R^2>=0.999` 且 probabilities 严格下降；
- retained fraction 在 `[0.75,0.95]`；
- removal 前 mean tail `>=0.001`；
- paired shrink CI-low `>=3e-4`；
- shrink ratio `>=2`；
- removal 后 absolute mean tail `<=0.0015`。

JSON 保存每条 check 的 observed/criterion/limit/detail、逐 lag correlation、all-gg probabilities、两个 seed 和所有 CI。

## 6. 证据边界

本结果允许写“observed-only estimator recovers hidden occupancy within the preregistered interval, and leakage-run post-selection removes the simulated long-lag tail”。禁止写实验 raw syndrome/Wigner 数据复现、真实 transmon higher-state model、在线 leakage-removal controller、真实 nonstationarity 消除或 device-calibrated correlation length。
