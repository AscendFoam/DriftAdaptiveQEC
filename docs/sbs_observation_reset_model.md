# T2.0.3 sBs g/e/leakage observation 与 reset model

**日期：** 2026-07-14  
**实现：** `physics/sbs_observation_reset.py`  
**协议 ID：** `PROTO-SBS-MAIN`  
**证据范围：** protocol-aligned hidden/observed/reset effective model；参数默认不代表 Sivak 装置

## 1. 四层状态分离

一次 full cycle 的信息流被拆成四层：

1. **ideal Kraus branch**：T2.0.2 的 `K_gg/K_ge/K_eg/K_ee`；
2. **hidden ancilla state**：`g/e/f/higher`，只供 simulator truth lane；
3. **observed class**：`g/e/leakage`，是可部署算法允许读取的输入；
4. **conditional reset**：只按 observed class 选择动作，输出 reset 后 hidden carry state。

模型不允许 controller 直接读取 hidden state。`SBSObservedCycle.as_deployable_dict()` 不含 ideal label、hidden pre/post state、truth flag 或 carry state；这些只在独立 `SBSTruthCycle` 中出现。

## 2. Pair order：Kraus `(Z,X)` 与执行 `(X,Z)`

论文给出

\[
K_{ge}=K_g^Z K_e^X.
\]

因此 label 的字符顺序是 `(Z,X)`，算符执行与本项目接口的时间顺序是 `(X,Z)`。例如 `K_ge` 解码为 `PairedSyndrome(x="e", z="g")`。代码和 `docs/protocol_hierarchy.json` 同时冻结：

- `kraus_label_character_order = ["Z","X"]`；
- `chronological_constituent_order = ["X","Z"]`。

这使 X/Z 的 recovery e-run 不会因字符串直读而互换。

## 3. 三个显式 stochastic kernels

### 3.1 Preparation / carry kernel

`P(hidden_pre | hidden_carry, ideal_g/e)` 的 shape 是 `[4,2,4]`。常用 builder 的规则为：

- carry 为 g：以显式 injection probability 把 ideal g/e 变成 g/e、f 或 higher；
- carry 为 e/f/higher：先保持未清除 state，使 reset failure 能跨 constituent/cycle 持续。

### 3.2 Readout confusion matrix

`P(observed | hidden_pre)` 的 shape 固定为完整 `[4,3]`，四行对应 g/e/f/higher，三列对应 g/e/leakage。每行必须非负且和为 1。

Sivak 文献局部给出 `F_g=0.9997`、`F_e=0.9914`，但本地文本没有可安全读出的 `F_f` 和完整 off-diagonal matrix。因此代码不提供文献默认 confusion matrix，也不从两个对角元补全 4×3 矩阵；调用者必须显式给出 full matrix 与 provenance。

### 3.3 Observation-conditioned reset kernel

`P(hidden_post | observed, hidden_pre)` 的 shape 是 `[3,4,4]`。常用 builder 冻结以下作用域：

- observed g：不发 reset pulse；
- observed e：尝试 e→g；
- observed leakage：尝试 f/higher→g；
- action 与 hidden state 不匹配时，hidden state 保持，不允许 simulator 偷看 truth 自动选择正确 pulse。

`f_reset_success` 与 `higher_reset_success` 分开。前者表达 `|f>` measurement-feedback reset；后者可设为 0 表达论文所述“高于 f 的 states 未被 reset 覆盖”，也可作为显式 assumption 表达其他 decay/active recovery。两者均无装置式默认值。

## 4. e-run、leakage streak 与 truth counters

- `x_e_run/z_e_run`：分别按 full cycles 跟踪同一 quadrature 的 observed e，不能把 X/Z 交错串直接累计；
- `leakage_constituent_run`：按实际 X→Z constituent 顺序累计连续 observed leakage；
- `leakage_cycle_run`：一个 full cycle 任一 constituent 出现 observed leakage 就累计；
- `hidden_higher_run/hidden_f_run`：只在 truth lane 基于 hidden pre-readout state 累计；
- 所有 counter 使用显式 `counter_max` 饱和，不发生整数无限增长或静默 overflow。

hidden higher 可以被错分为 g，此时 hidden streak 会增加，而 observed leakage counter 保持 0；测试专门验证了这种差异，防止把 truth counter 当可部署 feature。

## 5. 文献数值与模型参数边界

文献报告：`|f>` 通常一周期内被 reset；长事件在其后呈 17.2 cycles 衰减；任意 leakage 与持续至少两周期 leakage 的装置率分别约 `6.76e-4` 和 `1.28e-4` per cycle。它们只作为原装置事实保存在 paper-parameter registry，不是本 model builder 的默认值。

`make_persistent_leakage_model` 强制显式输入：完整 confusion matrix、g/e 条件下的 f/higher injection、e/f/higher reset success、counter max 和 provenance。正式实验前仍需 calibration gate。

## 6. API

- `ideal_syndrome_from_kraus`：明确 `(Z,X)` label → `(X,Z)` execution mapping；
- `HiddenAncillaMemory` / `ObservedSyndromeMemory`：truth 与 deployable memory 分离；
- `SBSObservationResetModel.step`：单 full-cycle 复现；
- `simulate`：同一 seeded RNG 下运行理想 branch sequence；
- `SBSObservationResetTrajectory.deployable_records`：只导出 observed lane；
- `make_persistent_leakage_model`：参数全显式的 assumption-driven builder。

## 7. 验证与反 demo 审计

`tests/test_sbs_observation_reset.py` 覆盖：

1. 四个 Kraus label 的 Z/X 字符顺序和 X/Z 执行顺序；
2. ideal paired syndrome、e reset 与 hidden post state；
3. X/Z 分离 e-run 及 saturation；
4. `|f>`→leakage→g reset；
5. higher state 多 constituent/多 cycle streak；
6. higher→g misclassification 时 hidden/observed counter 分歧；
7. action 只按 observation 选择，不能 truth-assisted reset；
8. readout confusion Monte Carlo 与解析行一致；
9. f reset success Monte Carlo 与输入概率一致；
10. higher leakage streak survival 与 `(1-p_reset)^k` 一致；
11. T2.0.2 ideal trajectory 到 T2.0.3 observed trajectory 的集成与 seeded reproducibility；
12. deployable records 无 hidden/ideal/truth 字段；
13. paper registry 的 `F_f/off-diagonal=null` 保持 fail closed；
14. 非归一化、负概率、缺 provenance、非法 counter、memory index 不一致和坏 label 全部拒绝。

本层已实现 observation/reset semantics，但仍没有 Table S3 timing、模拟 ADC/IQ threshold、装置校准矩阵或 Fock-space ancilla evolution；这些不得从当前结果外推。
