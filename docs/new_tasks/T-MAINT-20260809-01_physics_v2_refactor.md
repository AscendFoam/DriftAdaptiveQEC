# T-MAINT-20260809-01：physics v2 可维护性重构

- 日期：2026-08-09
- 状态：Done
- 类型：用户授权的破坏旧哈希兼容维护；不改变当前科学任务执行指针

## 输入与目标

- 输入：`physics/` 当前源码、测试、包根导出和仓内调用方。
- 目标：缩短超长文件，复用语义一致的代码，移除没有当前调用价值的包装与防御分支，同时保持物理模型、数值流程和公开行为尽量不变。
- 证据边界：旧 path/bytes/SHA-256 绑定全部降级为 v1 历史快照；不得只替换 manifest 哈希并声称旧资格仍验证当前实现。

## 通过标准

1. 普通源码文件不超过 1000 行，不再保留基于旧源码字节的超长例外。
2. 现有公开导出和仓内模块入口保持可用，除非删除的是确认无调用方的包装。
3. 关键模型固定 seed/config 的结构化结果与重构前一致，或差异有明确理论理由和记录。
4. Phase 9 backend A/B 的科学传播、likelihood 和 RNG 继续独立。
5. 定向测试、模块导入、CLI 和规模检查通过。

## 失败分支

- 若某次抽取改变数值结果且无法由理论或已知 bug 解释，则回退该抽取并保留原科学实现。
- 若旧证据 verifier 因源码字节变化失败，只记录为 v1 historical mismatch，不通过修改旧证据来伪造当前资格。

## 实际完成内容

1. 将 `phase9_twin_contract.py` 缩为 129 行公开入口，合同实现按 schema、recurrence、enumeration、qualification 放入 `_phase9_contract/`；完整固定输入 characterization 的 38,639-byte canonical JSON 与重构前逐字节一致。
2. 将 backend A/B 分别迁入 `_phase9_backend_a/`、`_phase9_backend_b/`。A 为 17 行入口 + 946/861/730 行实现，总行数仍为 2,554；B 为 63 行入口 + 908/896/581 行实现，总行数仍为 2,448。两套传播、RNG、likelihood、reset、logical projector 和资格门继续物理隔离。
3. 将 `protocol_ancilla_errors.py` 缩为 101 行入口，sBs fault overlay、sharpen--trim 和 validation 分入 `_protocol_ancilla/`；固定 seed characterization 的 44,352-byte canonical JSON 逐字节一致。
4. 拆分其余超长边界：
   - `differentiable_sbs_trajectory.py` 1,000 行，validation 237 行；
   - `differentiable_sbs_feasibility.py` 447 行，worker 732 行；
   - `nmf_directional_ranking.py` 892 行，execution 587 行；
   - `control_imperfections.py` 896 行，validation 272 行；
   - `cross_fidelity_validation.py` 970 行，reporting 160 行。
5. 在 `_shared/` 只保留四个按语义命名的小核：validation、categorical sampling、Hermite numerics 和 Torch checkpoint serialization；删除各模型中的同义重复实现，没有抽取 backend A/B 的科学 kernel。
6. 将包根惰性导出从 325 个缩到 10 个基础别名；仓内生产调用改为具体模块导入，保留标准 `physics.<module>` 路径和冷导入零重依赖行为。
7. 删除超长源码 SHA allowlist；`scripts/check_physics_module_size.py` 现在递归要求所有 Python 文件不超过 1,000 行。

## 产物路径

- 公共入口：`physics/phase9_twin_contract.py`、`physics/phase9_backend_a.py`、`physics/phase9_backend_b.py`、`physics/protocol_ancilla_errors.py`。
- 私有实现：`physics/_phase9_contract/`、`physics/_phase9_backend_a/`、`physics/_phase9_backend_b/`、`physics/_protocol_ancilla/`、`physics/_control_imperfections/`、`physics/_cross_fidelity/`、`physics/_differentiable_sbs/`、`physics/_nmf_ranking/`。
- 公共复用核：`physics/_shared/`。
- 契约与守卫：`tests/test_physics_api_contract.py`、`tests/test_physics_module_size.py`、`scripts/check_physics_module_size.py`。
- 阅读入口：`physics/README.md`。

## 规模结果

- 顶层 `physics/*.py`：56 个，与整理前相同；实现文件按领域进入私有目录，不再平铺在顶层。
- 全部 Python 文件：83 个、38,883 行；整理前为 38,807 行，净增 76 行（约 0.2%），主要是稳定入口和显式 import，不再以复制科学逻辑换取少文件。
- 最大单文件：由 2,835 行降为 1,000 行；83/83 全部满足上限。
- 包根 API：325 -> 10 个惰性别名。

## 验证方式和结果

- 统一定向回归：742 passed、2 skipped、21 deselected；其中 Phase 9 核心 288 passed，拆分模块 198 passed/1 skipped，共享核相关 199 passed/1 skipped，其余控制/边界 52 passed，API/规模合同 5 passed。
- Backend A 固定 seed 的 config hash、最终 density、IQ、posterior/reset、drift、logical 数值一致；Backend B 的 canonical/qualification/trajectory/config 四项 hash 与重构前一致。
- API 合同在 Python 3.9/3.11 均验证；冷 `import physics` 不加载 NumPy/SciPy/Torch/Matplotlib。
- 6 个保留 CLI 的 `--help` smoke 全过；83 个文件全部 compile；`git diff --check` 无错误。
- 旧 backend A/B qualification 测试为 26 passed、9 failed。9 项均是预期的 v1 parent/source bytes、analysis/toolchain 或 live hash 绑定失配；数值 gates 未回归。缺失旅行快照中的历史 JSON/CSV/PDF/论文 Markdown 的测试仍不能运行，本任务未伪造或补写这些证据。
- NumPy 2 环境仍报告既有 `np.trapz` deprecation warning；一个 Petz fidelity 单测存在约 `2.57e-9` 的环境数值差、略超原 `2e-9` 容差，均与本次校验复用无执行路径关系，未放宽测试。

## 风险复核与论文 claim 影响

- `R-N199` 调整为 Open / High / Immediate：旧 path/bytes/SHA 证据只允许解释为 v1 历史快照。
- 当前源码用于任何新论文 claim 前必须 fresh 重跑对应 qualification，并建立多文件 manifest；禁止回写旧 pin 伪造连续资格。
- 本任务未运行新科学 outcome，不改变模型范围、物理结论、性能结论或当前推荐科学任务 `T-RISK-20260728-04`。
- 不另插入科学 task；fresh qualification 是恢复后续 claim 的前置硬门。

## 任务板同步

- `docs/new_task_board.md` 已将 `T-MAINT-20260809-01` 从 `In Progress` 更新为 `Done`；当前科学执行指针保持不变。
