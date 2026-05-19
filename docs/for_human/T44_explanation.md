# T44：论文目标计划书（工作稿）

## 1. 这份文档是什么

这不是最终论文，也不是结果扩写稿。  
它的目标是把当前项目整理成一份**可执行、可审稿、可继续补证据**的论文目标计划书。

写作原则只有三条：

1. 只写当前证据真的能支撑的部分。
2. 不能支撑的地方直接写 `xxx`，或明确标成 `blocked / partial`。
3. 论文叙事要像 NVIDIA 类系统论文一样，先讲问题和系统，再讲方法和实验，不把空白写成结论。

---

## 2. 先给结论

以当前仓库证据看，这篇论文**可以开始规划**，但**还不能直接进入完整成稿**。

更严谨地说：

- 可以写成一篇**方法 + 系统 + 受控复验**的论文目标计划；
- 还不能写成一篇**真实部署完成**或**全面 benchmark 已闭环**的论文；
- 当前最稳妥的主线是：`teacher-guided residual-b` + 双回路运行时 + frozen-set formal software revalidation；
- 当前最不该写强的部分是：真板验证、真实 `.tflite` runtime、跨平台训练复现、integrated statcalib comparator、expanded benchmark。
- 更关键的一点是：**`T44` 本身只能冻结问题，不能补出问题。**
- 而且只看当前 task board 里已经显式可见的后续 pending 项（`T32`, `T37`），它们也**不足以单独把论文推到高质量投稿标准**。

---

## 3. 推荐论文定位

### 3.1 推荐主标题

**A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction**

### 3.2 保守备选标题

**Evidence-Bounded Recovery of a CNN-Assisted QEC Decoding Pipeline Under Mock-Backed Software HIL**

### 3.3 选择理由

- 主标题更像方法论文，适合较好的期刊/会议；
- 但标题里的"framework"必须被正文真正支撑，不能只是口号；
- 如果后续证据没有补齐，保守标题更安全；
- 目前不建议写成"deployment complete""hardware validated""state-of-the-art broad benchmark"之类标题。

### 3.4 关于 `延伸改进思路.md` 的定位

这份文档值得读，但当前更合适的定位是：

1. 它是**后续延伸研究参考**；
2. 它不是当前主线 truth baseline；
3. 它不能自动改写当前论文主线；
4. 其中任何一个想法如果要进入主线，都必须单开 bounded task。

---

## 4. 论文的核心目标

这篇论文最值得成立的目标，不是"CNN 全面替代经典解码器"，而是：

> 在实时硬件约束下，把学习模块限制为对快回路真正有用的残差修正，让稳定的 classical teacher 负责锚定，让轻量 CNN 负责补偿漂移。

换句话说，论文要回答的是：

1. 为什么 GKP 漂移自适应需要双回路？
2. 为什么直接绝对回归不如 teacher-guided residual？
3. 为什么快回路必须保持线性、确定、可硬件化？
4. 这种设计在当前冻结协议下到底验证到了什么、没验证到什么？

---

## 5. 章节结构计划

下面按更接近 NVIDIA 风格的系统论文方式排版。  
每一节都写清楚"该写什么""当前能写到哪里""还缺什么"。

### 5.1 Abstract

**建议结构**

1. 问题：GKP 漂移自适应需要实时解码。
2. 方法：双回路 + teacher-guided residual-b。
3. 结果：frozen-set formal software revalidation 中表现最好。
4. 边界：当前仍是 mock-backed software HIL，不是真板，不是真实 `.tflite` runtime。

**当前可直接写**

- `C1`：bounded P3 software HIL 路径已恢复并复验；
- `C2/C3`：frozen-set formal software revalidation 已完成，`hybrid_residual_b` 在四个场景中都赢；
- `C5`：有一次 clean-environment CPU-only training smoke；
- `C9`：statcalib 只有接口契约和测试。

**必须补充**

- `xxx`：真实 `.tflite` runtime；
- `xxx`：real-board HIL；
- `xxx`：更广泛 benchmark；
- `xxx`：多 seed 机制结论；
- `xxx`：features 组正式结果。

---

### 5.2 Introduction

**建议写法**

1. 先讲 GKP 解码为什么需要漂移自适应。
2. 再讲为什么在线控制语义不能简单等同于离线回归。
3. 再讲为什么本项目选择双回路，而不是端到端替换。
4. 最后给出贡献列表。

**建议贡献点**

- 一个可复验的双回路 GKP 解码系统；
- 一个 teacher-guided residual-b 学习方案；
- 一组 frozen-set formal software revalidation 结果；
- 一个 clean CPU-only training smoke；
- 一个独立的 statcalib interface contract。

**不能写强的地方**

- 不能写"全流程部署完成"；
- 不能写"广泛优于所有 baseline"；
- 不能写"真板已验证"；
- 不能写"训练完全可复现"。

---

### 5.3 Background / Related Work

**建议排版**

1. GKP 与 syndrome / drift 的物理背景
2. 快回路 / 慢回路的时间尺度分离
3. ML-assisted QEC 与 classical adaptive estimators
4. residual / teacher-guided 这种方法为什么不是普通回归
5. quantum systems 论文里的证据边界

**当前可直接写**

- 物理背景和系统背景已经足够；
- classical baselines 也已经有明确位置；
- "teacher + residual" 与绝对回归的区别已经能讲清。

**必须补充**

- `xxx`：与更强 GKP / soft-information 工作的定量对照表；
- `xxx`：文献引用清单的最终整理；
- `xxx`：如果要写"novelty"句子，必须先确认不会超出证据边界。

---

### 5.4 Method / System

**建议小节**

1. 双回路架构
2. Fast loop: linear decode + param bank
3. Slow loop: histogram window + teacher + residual CNN
4. Runtime-consistent feature construction
5. ParamMapper 与 `(K, b)` 提交语义

**当前可直接写**

- `physics/README.md` 支持 GKP 物理链路；
- `cnn_fpga/runtime/README.md` 支持双回路与 param bank；
- `cnn_fpga/hwio/README.md` 支持 mock/board 后端统一接口；
- `cnn_fpga/decoder/README.md` 支持 teacher、UKF、RLS、statcalib 等基线。

**必须补充**

- `xxx`：真实 board backend 的可执行日志；
- `xxx`：true `.tflite` 推理链路；
- `xxx`：`board_backend.py` 不再只是 placeholder 的证据；
- `xxx`：参数提交延迟和回滚失败率的正式统计。

---

### 5.5 Experimental Protocol

**建议小节**

1. frozen-set formal protocol
2. 场景、seed、repeats
3. baseline 列表
4. ablation 计划
5. 订练 smoke / runtime boundary / deployment boundary

**当前可直接写**

- `C2/C3` 已经有 formal software revalidation；
- `C5` 已经有一次 CPU-only smoke；
- `C9` 已经有 statcalib interface contract；
- `C4` 只能作为单 seed trace-supported diagnosis。

**必须补充**

- `xxx`：features 组正式 ablation；
- `xxx`：更广场景的 benchmark 扩展；
- `xxx`：至少多 seed 的机制验证；
- `xxx`：真实 runtime / board 执行计划对应的实测结果。

---

### 5.6 Results

**当前最稳的结果顺序**

1. frozen-set ranking
2. scenario-wise 对比
3. single-seed mechanism diagnosis
4. training smoke
5. statcalib status

**当前可直接写**

- `Hybrid Residual-B` 在 frozen set 内赢四个场景；
- 最强经典基线是 `UKF`；
- `correction_saturation_rate = 0` 和 `aggressive_param_rate = 0` 不能被写成"系统更强"的证据，只能说没有更激进控制；
- `seed=20260429` 的机制诊断只能写成"trace-supported hypothesis"。

**必须补充**

- `xxx`：features 组的正式对比表；
- `xxx`：多 seed 机制结论；
- `xxx`：如果要写更强的结果段，就必须先补可复验的 runtime 或 board 证据。

---

### 5.7 Discussion / Limitations

**这一节必须写得最诚实**

要直接承认：

- `C6` 训练完整复现性仍未证实；
- `C7` true `.tflite` runtime 仍未恢复；
- `C8` real-board HIL 仍未验证；
- `C10` statcalib 还不是 integrated comparator；
- `C11` benchmark 还不是 broad / paper-grade expanded benchmark；
- `C4` 仍只是单 seed 机制诊断。

**不能写的句式**

- "almost complete"
- "nearly deployment-ready"
- "effectively reproducible"
- "hardware validated"
- "comprehensive benchmark"

**建议写法**

- 直接写"blocked / partial / supported"；
- 直接写"what is verified / unverified / missing"；
- 直接写"future work needs xxx"；
- 不要写成"只是差一点就完成了"。

---

### 5.8 Conclusion

**应该总结什么**

1. 这是一条可复验的双回路 teacher-guided residual 路线；
2. 当前证据支持 frozen-set formal software revalidation；
3. 当前证据还不足以支撑完整部署完成叙事；
4. 下一步不是继续扩写结论，而是补证据。

---

## 6. 图表计划

### 6.1 推荐图

1. **系统架构图**
   - fast loop / slow loop / param bank / HIL
2. **证据边界图**
   - mock-backed software HIL / true `.tflite` / real-board / training smoke
3. **benchmark 结构图**
   - scenarios / baselines / repeats / seeds
4. **机制诊断图**
   - `seed=20260429` 的 trace-supported path

### 6.2 推荐表

1. baseline 对比表
2. claim/evidence 对照表
3. evidence boundary 表
4. ablation 表

### 6.3 当前缺口

- `xxx`：features 组正式表格；
- `xxx`：true `.tflite` 表格；
- `xxx`：real-board smoke 表格；
- `xxx`：multi-seed 机制表格。

这部分也直接说明了一件事：

- 当前 T44 与当前已知 pending 项，**还不能自动补齐完整 paper-grade 图表包**。
- 最缺的不是"排版"，而是：
  - 缺正式 result pack
  - 缺多 seed 机制图
  - 缺 comparator extension 表
  - 缺稳定 regeneration ledger

---

## 7. 证据边界清单

### 已支持

- `C1` mock-backed software HIL bounded revalidation
- `C2` frozen-set formal software revalidation
- `C3` frozen set 内四场景胜出
- `C5` clean CPU-only training smoke
- `C9` statcalib interface contract

### 部分支持

- `C4` `seed=20260429` 的 trace-supported diagnosis

### 仍然阻塞

- `C6` full training reproducibility
- `C7` true `.tflite` runtime
- `C8` real-board HIL
- `C10` integrated statcalib comparator
- `C11` broad benchmark / paper-grade expanded benchmark

还应额外补一句当前 recovery 视角下的判断：

- `T44` 不是这些 blocked claim 的解决任务；
- `T32`、`T37` 即使后续完成，也只能分别缩窄 `C7/C8`；
- 论文要达到"较好投稿标准"，还需要额外的 benchmark / mechanism / ablation / reproducibility 类 bounded tasks。

---

## 8. 写作风格建议

要尽量像 NVIDIA 一类系统论文，但要保持科研诚实：

1. **先结论，后展开**
   - 每一节开头先告诉读者这节回答什么问题。
2. **方法和系统优先**
   - 不要把论文写成"实验日志合集"。
3. **边界显式**
   - 每个结果都要写清楚属于哪一层证据。
4. **少形容词，多条件**
   - 用"supported / partial / blocked"，少用"很强 / 很稳 / 几乎完成"。
5. **图表先于长论述**
   - 系统论文要让图和表先说话。
6. **把空白写成空白**
   - 该写 `xxx` 的地方就写 `xxx`，不要用模糊语句伪装完成。

---

## 9. 现在最适合的投稿定位

以当前证据，最稳妥的定位是：

- **方法/系统论文**
- **受控软件 HIL + 冻结协议复验**
- **teacher-guided residual adaptive decoding**

还不适合直接定位成：

- 真实硬件验证论文
- 完整部署论文
- 广义 state-of-the-art 论文
- 完整可复现训练链论文

如果只问"能不能投"，答案是：

- **可以朝 evidence-bounded 方法/系统论文去准备**

如果问"现在够不够好期刊/会议的强版本"，答案是：

- **还不够**
- 当前最主要不是 prose 不够，而是证据包不够完整。

---

## 10. 下一步执行顺序

1. 先把这份计划作为论文蓝图固定下来。
2. 再把 `Abstract / Introduction / Method` 逐节补正文。
3. 每写一节都回头检查 claim/evidence 边界。
4. 任何缺证据的位置都保留 `xxx`。
5. 等 `C6 / C7 / C8 / C10 / C11` 里真正有新证据后，再考虑把计划升级成完整论文稿。

更严格地说，后续真正需要的不是"继续写"，而是新增几类有界任务：

1. **paper-grade benchmark 扩展任务**
   - 用于回答 frozen-set 之外的说服力问题。
2. **multi-seed mechanism / intervention 任务**
   - 用于把 `seed=20260429` 从单 seed 诊断推进到更可信的机制证据。
3. **formal ablation result-pack 任务**
   - 用于收齐 features / teacher / comparator 的正式结果。
4. **reproducibility and figure-material pack 任务**
   - 用于把训练、图表、表格的 regeneration path 固定下来。
5. **runtime / board evidence 任务**
   - 若投稿定位仍保留部署或系统约束亮点，则还要补 true `.tflite` 与 bounded board smoke。

这些类别里，前四类更接近**主线补证据**；  
第五类更像**高价值质量增强项**。  
而 `docs/reference/延伸改进思路.md` 中的更激进方法，则应继续留在**后续延伸研究**而不是当前主线。
