# T-MAINT-20260808-01：physics 模块边界整理

- 日期：2026-08-08
- 状态：Done
- 类型：用户请求的仓库维护；不改变当前科学任务执行指针

## 输入材料

- `physics/` 的 56 个 Python 源文件、公开惰性导出表和仓内导入关系。
- `configs/phase9/`、机器 JSON/CSV、checkpoint/source manifest 中的 path、bytes 和 SHA-256 绑定。
- `tests/test_control_imperfections.py`、`tests/test_cross_fidelity_validation.py`、`tests/test_quadrature_conventions.py` 及相关兼容测试。

## 实际完成内容

1. 将 `control_imperfections.py` 的验证 runner、writer 和 CLI 实现迁入 `control_imperfections_validation.py`；旧模块保留原类型、公开函数和 `python -m` 入口。
2. 将 `cross_fidelity_validation.py` 的检查组装、失效归因、JSON writer 实现和 CLI 迁入 `cross_fidelity_reporting.py`；旧模块保留公开 writer wrapper，并把主 runner 拆为有名私有步骤。
3. 将 `error_correction.py`、`noise_channels.py` 对 `LATTICE_CONST` 的偶然转导依赖改为直接依赖 `constants.py`。
4. 新增模块规模守卫：普通 Python 文件不得超过 1000 行；7 个已有 source-hash/release/checkpoint 绑定的超长 v1 文件按当前字节冻结，修改时必须新建并资格化 v2。
5. 新增公开 API、冷导入和 55 个旧模块路径的 characterization tests；更新 `physics/README.md` 与根 README 阅读地图。

## 产物路径

- `physics/control_imperfections.py`
- `physics/control_imperfections_validation.py`
- `physics/cross_fidelity_validation.py`
- `physics/cross_fidelity_reporting.py`
- `physics/error_correction.py`
- `physics/noise_channels.py`
- `physics/README.md`
- `scripts/check_physics_module_size.py`
- `tests/test_physics_api_contract.py`
- `tests/test_physics_module_size.py`

## 验证方式和结果

- 行数：`control_imperfections.py` 904 行、companion 272 行；`cross_fidelity_validation.py` 988 行、companion 160 行。
- `control_imperfections` 新旧固定输入的 `as_dict()` 完全相等。
- `cross_fidelity` 新旧固定配置的 canonical JSON SHA-256 同为 `d4718e33e32018cc325a5d48aef2a1d4844ab38a2fb9fd1c2e197073bfc332cd`。
- Python 3.11 定向回归：`99 passed, 32 subtests passed`。
- NumPy 2.0.2 兼容环境的 cross-fidelity + quadrature 回归：`75 passed`，仅 3 个既有 `np.trapz` 弃用警告。
- CLI smoke：cross-fidelity 退出码 0，`15/15` checks 为真；control-imperfections 原 CLI `--help` 可用。
- 规模守卫、源码编译、README 导入 smoke 与 `git diff --check` 通过。

## 风险复核

- 新增 `R-N199`：超长 v1 科学源被证据链按字节绑定，不能用普通 facade 重构。当前以精确 SHA-256 守卫缓解；未来只能走 v2 + 重资格化。
- Phase 9 backend A/B 的科学传播、likelihood 和 RNG 继续物理隔离；本次没有为消除表面重复而建立共同科学内核。
- 新增 companion 不改变 325 个包根导出或 55 个旧模块路径；旧结果与 CLI 兼容性已有 characterization。

## 是否需要插入新 task

不插入。剩余超长文件不是普通代码债，而是 frozen evidence migration；只有实际启动 v2/requalification 时才应建立独立科学 task。当前推荐任务仍为 `T-RISK-20260728-04`。

## 对论文 claim 的影响

无。没有生成新物理结果、性能结果或硬件证据，也没有修改现有 release-pinned 科学源；只改善代码职责边界、阅读性和回归守卫。

## 任务板同步

已在 `docs/new_task_board.md` 进度日志登记为一次性维护任务；不改变任何科学 task 状态或当前推荐任务。
