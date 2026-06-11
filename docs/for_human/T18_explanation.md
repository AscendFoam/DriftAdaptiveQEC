# T18: TFLite Runtime Bootstrap — 通俗解释与 Review 说明

## 一、这个 Task 在做什么？（通俗版）

### 背景

这个项目在做一个"量子纠错解码器"，其中有一个关键环节：训练好的 AI 模型需要被部署到实际硬件（FPGA 板）上运行。为了把模型从电脑上的训练格式变成硬件能跑的格式，需要经过一个"导出"步骤——把模型导出成 `.tflite` 文件（TensorFlow Lite 格式），这样嵌入式设备才能加载和执行。

### 问题

现在有两件事容易搞混：

1. **真实的 `.tflite` 导出和运行**：需要安装 TensorFlow 或 TFLite 运行库，真正把模型转换成 `.tflite` 格式，然后在 TFLite 解释器上跑推理。这是"真刀真枪"的部署路径。

2. **`tflite_stub_v1` 回退路径**：如果机器上没有 TensorFlow，代码会退而求其次，生成一个 `.tflite.json` 文件作为"替身清单"。这个替身能让后续流程继续跑下去（接口通了），但它**不是真正的 TFLite 部署**，只是让管线不中断。

### T18 做了什么

T18 的任务就是：把上面两条路径的边界搞清楚，写成一份独立的文档。具体来说：

- 列出真实 `.tflite` 路径需要什么依赖（TensorFlow / tflite_runtime 等）
- 列出 `tflite_stub_v1` 是什么、能干什么、不能干什么
- 在当前机器上探测一下环境，看看能不能跑真实 `.tflite`
- 如果不能跑（事实确实如此），就把阻塞原因写清楚，**不伪造成能跑**

用一句话总结：**T18 是一份"环境清查报告"，搞清楚 .tflite 部署路径目前走到了哪一步，还差什么。**

## 二、任务实现详细解释

### 2.1 任务目标

为 `.tflite` export/runtime 路径补一份独立的 bootstrap 文档（`docs/evidence_packs/deployment_boundary/TFLite_runtime_bootstrap.md`），使得后续任何开发者或 AI 会话拿到这份文档就能知道：

- 当前仓库里有哪些 `.tflite` 相关的代码入口
- 真实 runtime 和 stub 回退各自是什么
- 当前环境能不能跑真实的 `.tflite`
- 如果不能，阻塞在哪里

### 2.2 任务流程

1. **只读代码审计**：Worker 阅读了 `cnn_fpga/model/export.py`、`cnn_fpga/runtime/inference_service.py`、`cnn_fpga/model/evaluate_tflite.py`、`cnn_fpga/model/validate_export.py`，理解两条路径的实现。

2. **环境探测**：在本机运行了最小验证：
   - `export.py --help`：确认导出入口存在
   - `evaluate_tflite.py --help`：确认评估入口存在
   - `validate_export.py --help`：确认一致性验证入口存在
   - `import tensorflow` / `import tflite_runtime`：确认两者都不可用

3. **撰写 bootstrap 文档**：产出 `docs/evidence_packs/deployment_boundary/TFLite_runtime_bootstrap.md`，包含 10 个章节，覆盖目的、边界判断、环境事实、入口命令、依赖边界、stub 边界、最小 smoke、未覆盖项、与其他 bootstrap 的关系、推荐表述。

4. **更新治理文档**：同步更新了 task board、handoff 和 risks 文档。

### 2.3 代码/配置文件的变化

**没有代码变更。** 本次任务完全在 `docs/` 目录内完成。

具体文件变化：

| 文件 | 变化类型 | 内容 |
|------|----------|------|
| `docs/evidence_packs/deployment_boundary/TFLite_runtime_bootstrap.md` | 新增 | `.tflite` 路径的独立 bootstrap 文档 |
| `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md` | 追加 | Worker Output Summary 段落 |
| `docs/04_task_board.md` | 修改 | T18 标记为 `[x]`，更新 Current Unique Task 状态 |
| `docs/07_handoff.md` | 修改 | 追加 T18 完成记录，更新判断和建议 |
| `docs/08_risks_and_open_questions.md` | 修改 | 新增 R12 风险项，更新开放问题 Q6/Q7/Q9/Q13 |

### 2.4 对后续开发的意义

1. **环境基线固定**：后续任何 AI 会话如果需要判断 `.tflite` 路径是否可用，只需读 `docs/evidence_packs/deployment_boundary/TFLite_runtime_bootstrap.md`，不需要重新审计代码。

2. **与 T17 形成互补**：T17 固定了训练链环境，T18 固定了 `.tflite` 部署链环境。两条路径现在都有独立的 bootstrap 文档。

3. **R12 风险显式化**：真实 TFLite 运行时不可用被正式记录为高风险项。这防止了后续文档或论文误写"TFLite 已部署"。

4. **为 T19/T20 铺路**：明确了 `.tflite` 路径的边界后，项目可以安心进入 T19（仓库清理）和 T20（真板准备清单），而不会在缺乏环境的情况下冒然尝试 `.tflite` 实际操作。

## 三、为什么给出 PASS 的 Review 结果？

### 审查逻辑

我作为 Reviewer，从以下 6 个维度进行了检查：

1. **任务是否真的完成** — 是。T18 目标是补 bootstrap 文档和边界说明，文档已产出且覆盖完整。环境探测结论（tensorflow 不可用）与代码审计结论一致。

2. **是否有伪实现 / mock / stub / hardcode** — 无。整个任务是文档任务，没有代码变更。文档中明确把"真实 runtime 不可用"写成了阻塞事实，没有伪造成已通过。

3. **是否缺测试或验证** — 不缺。任务包限定的验证方式是"只读代码审计 + 环境探测"，Worker 已按此执行。由于没有 tensorflow，更深层的 runtime 验证不可行，且已明确记录为阻塞。

4. **是否过度工程** — 否。文档 10 节内容全部服务于任务目标，没有多余抽象。

5. **是否破坏已有功能** — 否。无代码变更。

6. **文档是否把计划写成事实** — 否。文档在多处显式区分了"代码路径存在"和"runtime 已恢复"，且列出了 4 项"故意不承诺"。

### 唯一的非阻塞问题（N1）

第 10 节"推荐表述"的 Markdown 排版稍有不规范（中文反引号包裹长段落），但不影响语义准确性。这只是一个格式建议，不阻塞任务通过。

### 结论

Worker 完全按任务包执行，没有越界，没有伪实现，边界表述诚实，文档质量合格。**PASS**。
