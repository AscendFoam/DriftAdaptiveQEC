# T17 Review: Training Manifest Bootstrap

**Reviewer**: Claude Code (normal review)
**Date**: 2026-05-10
**Task package**: `docs/tasks/Phase2/T17_training_manifest_bootstrap.md`

---

## Verdict: **PASS**

---

## 1. Blocking Issues

None.

---

## 2. Non-blocking Issues

### N1: `torch` 版本为 dev build，文档未显式标注风险

bootstrap 文档记录了 `torch = 2.8.0.dev20250405+cu128`，这是一个 nightly/dev build，而非标准 release。dev build 意味着：
- API 可能在小版本内发生变化
- 同版本号在不同日期拉取到的实际包可能不同
- 其他机器上 `pip install torch` 不会得到同一构建

文档已正确声明"不写成跨机器保证"，但 dev build 这一点本身可以更显式地标注，让后续 Worker 知道这不仅是版本号差异，而是构建渠道差异。

**影响**：低。当前 bootstrap 的作用域是本机入口说明，不是跨机器依赖锁定。

### N2: 未产出 `requirements-train.txt`

任务允许选择 `requirements-train.txt` 或 `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md`。Worker 选择了文档路径。

这在当前阶段是合理的：dev torch 版本不适合写入正式 requirements 文件；文档方式更诚实地反映了"当前只有本机环境探测结果"的事实。但如果后续需要训练链的可移植性，仍需补一份格式化的依赖声明。

---

## 3. Scope Compliance Verification

### 3.1 Allowed files — 全部合规

| 文件 | Allowed? | 实际变更 |
|------|----------|----------|
| `docs/tasks/Phase2/T17_training_manifest_bootstrap.md` | Yes | Worker Output Summary 追加 |
| `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md` | Yes | 新增 147 行 |
| `docs/04_task_board.md` | Yes | T17 标记完成 + Current Task 更新 |
| `docs/07_handoff.md` | Yes | T17 完成记录 + 当前状态更新 |
| `docs/08_risks_and_open_questions.md` | Yes | R2/Q6/Q7/Q9 更新 |

### 3.2 Forbidden scope — 全部合规

| 禁止项 | 是否违反 |
|--------|----------|
| 不改训练代码 | 未违反（`cnn_fpga/`、`physics/`、`benchmark/` 无 diff） |
| 不启动训练长跑 | 未违反（无 run 输出、无训练 artifact 变更） |
| 不改模型主线 | 未违反 |
| 不把 DLEnv 探测结果写成跨机器保证 | 未违反（多处显式标注"只确认了这台机器上可用"） |

### 3.3 Worker Output Summary 一致性

Worker 声称更新了 4 个文件 + 新增 1 个文件，实际 diff 覆盖范围与声明一致。

---

## 4. Code Accuracy Cross-Check

Bootstrap 文档 Section 4 声明 `train.py` 支持两类路径。交叉验证：

| 声明 | 代码验证 | 结果 |
|------|----------|------|
| `linear_regression_baseline` 不依赖 torch | `train.py:36-50` 用纯 numpy 实现，无 torch import | 通过 |
| `tiny_cnn` 支持 numpy/torch 双后端 | `train.py:131-142` 调用 `fit_tiny_cnn`，`tiny_cnn.py` 支持 `backend` 参数 | 通过 |
| CLI 参数只有 `--config / --train-split / --val-split` | `train.py:65-70` `_arg_parser()` 定义 | 通过 |
| 典型配置文件存在 | glob 确认 `experiment_static_theta_v2.yaml`、`experiment_runtime_b_residual.yaml`、`..._v5.yaml` 均存在 | 通过 |

---

## 5. Document Consistency Cross-Check

| 检查项 | 结果 |
|--------|------|
| `04_task_board.md` T17 标记为 `[x]` | 通过 |
| `04_task_board.md` Current Task 包含"已完成" | 通过 |
| `07_handoff.md` Section 1 当前任务更新为 T17 已完成 | 通过 |
| `07_handoff.md` Section 2 追加 T17 条目（21/22） | 通过 |
| `07_handoff.md` Section 4 补充 T17 判断 | 通过 |
| `07_handoff.md` Section 5 关键产出追加 `training_chain_bootstrap.md` | 通过 |
| `07_handoff.md` Section 6 任务摘要更新为 T17 | 通过 |
| `07_handoff.md` Section 7 下一步建议已更新 | 通过 |
| `08_risks_and_open_questions.md` R2 Evidence/Mitigation 更新 | 通过 |
| `08_risks_and_open_questions.md` Q6 更新 | 通过 |
| `08_risks_and_open_questions.md` Q7 更新 | 通过 |
| `08_risks_and_open_questions.md` Q9 更新 | 通过 |
| `docs/05_decision_log.md` 未修改（正确：无状态切换） | 通过 |
| 三份文档关于训练链的表述口径一致 | 通过 |

---

## 6. Missing Tests

本任务只要求 import 级和 `--help` 级检查，不要求完整训练。Worker 已执行：
- DLEnv 下 `numpy / yaml / torch` 导入检查
- `python -m cnn_fpga.model.train --help` 检查

符合任务包的 Verification 要求，无额外缺失。

---

## 7. Suspicious Implementation Details

无。bootstrap 文档没有引入 hardcode、假结果或跳过验证。边界表述与 `docs/03_hil_p4_boundary_audit.md` 的风格一致。

---

## 8. Recommended Next Action

1. **Captain 整合**：接受 T17 完成，指定下一任务（建议 `T18` TFLite manifest）。
2. **T18 边界**：T18 应继续按同样的拆分模式处理 `.tflite` export/runtime 路径。
3. **可选后续**：若训练链后续需要可移植性，可在更晚的任务中补 `requirements-train.txt`（使用 `pip freeze` 并标注 dev build 渠道限制）。
