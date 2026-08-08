# T-RISK-20260728-02：标量 UQ 三分裂校准

- **日期**：2026-07-28
- **状态**：Done
- **来源风险**：R-N188、R-N189
- **父任务**：T-RISK-20260728-01

## 输入材料

- `docs/t_risk_20260728_01_density_uq_preflight.json` 与 Source Data：density UQ 已用 72 cells、18,432 records、384 clusters/state 独立 PASS。
- `docs/t_risk_20260728_01_scalar_uq_preflight.json` 与 Source Data：factor=1.0、288 trials/cell 的 scalar UQ 完成 192 cells、55,296 trials，但 coverage 与 boundary-power 门 fail-closed。
- `cnn_fpga/benchmark/phase9_paired_cluster_uq.py`：冻结的 paired vector-norm multiplier UCB 实现。
- `configs/phase9/t_risk_20260728_02_scalar_uq_calibration.json`：三分裂 V2 预注册合同。
- V1 基础设施失败证据：`docs/t_risk_20260728_02_scalar_uq_failed_seed_collision_v1.json`、393,216-row selection A、resource report 与 run identity。

## 执行方案

1. 原 55,296-row scalar NO-GO 永久降格为 diagnostic，不得用同批样本改阈值、改 factor 或充当 confirmation。
2. 在 fresh outcome 前冻结：
   - 三个 family、8 个 margin、`n={12,384}`、4 个 effect ratio，共 192 cells；
   - factor grid `{1.0,1.1,1.2,1.3,1.4,1.5}`；
   - 两份独立 selection fold 与一份 untouched confirmation；
   - 每 cell 2,048 trials、B=199、factor selection 取“双 fold 全门通过的最小 factor”；
   - 每 split 256-way Bonferroni-adjusted Wilson coverage/power 门；
   - 4-worker、每 worker 单 BLAS thread、resource preflight、owner lock 与原子发布。
3. selection receipt 原子封印之前不得生成 confirmation；confirmation 只评估已选 factor，不得重选。
4. 用不导入 production runner 的独立 verifier 逐行重算全部 seed、cell、UCB、Wilson、factor selection 与 confirmation。

## 实际完成内容

### 1. 原 scalar NO-GO 的根因分离

- 原运行完整产生 55,296 trials。
- coverage 总门失败 74/192 cells，其中 64 个为 `n=12`、10 个为 `n=384`。
- effect-ratio=1.0 的 16 个 primary power strata 中有 3 个 simultaneous UCB 超过 0.10。
- 288 trials 在 256-way Bonferroni 下，即使 empirical coverage 约 0.95，其 Wilson LCB 通常也低于 0.90；因此原门的样本量不足，不能据此宣布 factor=1.0 不可用。

### 2. V1 seed-collision fail-closed 与 V2 修复

- V1 release commit：`36f82a29fca12ab88676d5c4b721e821bca9cae6`。
- V1 resource preflight PASS，selection A 完整原子发布 393,216 rows。
- selection B 在 raw seed/index 唯一性门被拒绝；未发布 selection receipt、factor 决策、confirmation 或 final report。
- 精确碰撞：
  - split：`selection_b`
  - cell：`rare_heavy_tail__m0p005000__n384__r0p500`
  - trial：358 与 908
  - 两者均映射到 seed `70,942,503`
- 根因是 `base + SHA256(address) mod 1e9` 非单射。
- V2 release commit：`4e9336dc525d0d27549443cdeec4395129e59ef5`。
- V2 不改变 factor grid、统计门、样本量、family、margin 或 split role，只改为可证明单射的
  `base + cell_index × 2048 + trial_index`，并使用六段间隔 1,000,000、最大 offset 393,215 的全新不重叠 scientific seed ranges。

### 3. Fresh2 三分裂结果

- run id：`250d6089-9dc0-4f7d-9257-3ba4338303a5`
- resource preflight：PASS，估计 wall time 332.76 s。
- 每 split：
  - 192 cells；
  - 2,048 trials/cell；
  - 393,216 raw rows；
  - 无缺行、重复 trial、重复 seed、非有限值或 BLAS 线程漂移。
- 双 selection fold 选择的最小合格 factor：`1.0`。
- untouched confirmation 只评估 factor=1.0，并终态 PASS。

| Split | 最差 coverage rate | 最差 simultaneous Wilson LCB | Coverage | 4 组 power IUT |
| --- | ---: | ---: | --- | --- |
| selection A | 0.9443359375 | 0.9223013532 | PASS | 全 PASS |
| selection B | 0.9448242188 | 0.9228626940 | PASS | 全 PASS |
| confirmation | 0.9409179688 | 0.9183825856 | PASS | 全 PASS |

四组 power 规则分别为：

- null equivalence LCB `>=0.80`；
- half-margin equivalence LCB `>=0.65`；
- boundary equivalence UCB `<=0.10`；
- outside-margin equivalence UCB `<=0.05`。

三个 split 的 16-stratum primary IUT 在四组规则下均无失败项。

## 产物路径

- Runner：`cnn_fpga/benchmark/phase9_scalar_uq_three_split_calibration.py`
- 独立 verifier：`cnn_fpga/benchmark/phase9_scalar_uq_three_split_verifier.py`
- 配置：`configs/phase9/t_risk_20260728_02_scalar_uq_calibration.json`
- 主报告：`docs/t_risk_20260728_02_scalar_uq_calibration.json`
  - report analysis：`de82db4d12459e3b54174d6cf738953ed242b153d32de329c1094cae9ef80db4`
- 独立验证报告：`docs/t_risk_20260728_02_scalar_uq_calibration_verification.json`
  - verification analysis：`9dfe4506e8dffd7bd5a8d4e9e19e5bf02e0abab27fba521c3b3c449237878dd9`
- Selection receipt：`docs/t_risk_20260728_02_scalar_uq_selection_receipt.json`
- Run identity：`docs/t_risk_20260728_02_scalar_uq_run_identity.json`
- Resource report：`docs/t_risk_20260728_02_scalar_uq_resource_preflight.json`
- Lossless Source Data：
  - `docs/t_risk_20260728_02_scalar_uq_selection_a.csv`
  - `docs/t_risk_20260728_02_scalar_uq_selection_b.csv`
  - `docs/t_risk_20260728_02_scalar_uq_confirmation.csv`
- V1 失败证据：
  - `docs/t_risk_20260728_02_scalar_uq_failed_seed_collision_v1.json`
  - `docs/t_risk_20260728_02_scalar_uq_selection_a_failed_seed_collision_v1.csv`
  - `docs/t_risk_20260728_02_scalar_uq_resource_preflight_failed_seed_collision_v1.json`
  - `docs/t_risk_20260728_02_scalar_uq_run_identity_failed_seed_collision_v1.json`

## 验证方式和结果

1. Production runner 自验证：
   - final report self-hash；
   - config/implementation/paired-UQ/run identity/selection receipt/resource/三份 CSV/两类 parent 的 expected-name + live binding；
   - 192×2,048 行数、claim boundary、selection/confirmation 时序与单因子门。
2. 独立 verifier：
   - 不导入 production runner；
   - 逐行重算 1,179,648 rows；
   - 精确重算 cell-major seed、UCB、Wilson、64 primary power strata/split、双 fold 最小 factor 与 confirmation；
   - verdict=`VERIFIED_PASS_SCALAR_UQ_THREE_SPLIT_CALIBRATION`。
3. 测试：
   - `63 passed in 16.65s`；
   - 覆盖 V1 精确碰撞 regression、V2 全域 393,216-seed 单射与六 ranges disjoint、真实 Windows ProcessPool smoke、单 BLAS thread 等价、config mutation、denominator/seed/raw schema、factor reselection、claim firewall、owner lock、CLI 零写与原 density/scalar UQ 回归。

## 反简化审计

- 不是把 288-trial 门改松：coverage/power 数值门完全保持不变，只把每 cell trial 数提升到预注册的 2,048。
- 不是在同一数据上选 factor 并确认：两份 selection 与 confirmation 使用完全不重叠的 fresh seed ranges。
- 不是挑一个有利 factor：只允许按冻结 grid 取双 fold 同时全门通过的最小值，最终仍为 1.0。
- 不是 mock 并行：测试包含真实 Windows ProcessPool/BLAS worker smoke；production 完成 1,179,648 raw rows。
- 不是 runner 自证：独立 verifier 不导入 production runner并逐行重算。
- V1 基础设施失败没有被删除、覆盖或改写为 PASS。
- 本 task 只资格化 scalar UQ 统计合同，不证明 twin qualification、LER、lifetime、physical break-even、hardware measured、official/Puviani 或 external SOTA。

## 风险复核

- R-N188：由 Open 降为 Mitigated。factor=1.0 已在三分裂合同下通过，但只有 T-RISK-20260728-01 后续 design/formal 原样消费该 analysis 与 factor 时才能关闭。
- R-N189：Closed。V1 碰撞已有精确复现，V2 单射映射、range firewall、逐行 verifier 和真实 ProcessPool 均通过。
- 不插入新 task；恢复 T-RISK-20260728-01。

## 对任务板的同步

- `T-RISK-20260728-02`：`In Progress -> Done`。
- `T-RISK-20260728-01`：`Blocked (T-RISK-20260728-02) -> In Progress`。
- 当前推荐任务恢复为 `T-RISK-20260728-01` 的 cutoff32/36 design extension。
- 六个 twin/performance downstream 仍保持 Blocked；本 task PASS 不释放它们。
