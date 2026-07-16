# 翻译与来源说明

## 当前状态

这是服务于 T3.2.1 的 **task-scoped draft**，不是 nature-reader 所定义的完整全文双语 reader。它保留稳定 source ID、页码索引、全部已提取图卡和核心机制公式，但没有逐段翻译以下内容：

- p.1 引言中的完整相关工作与有限能 GKP 定义推导；
- p.2–3 q/p-SE 波函数变换的全部正文；
- p.5–10 Appendix A–F 的 theorem/proof 全文；
- p.11 tracking/truncation probability integral 的逐式推导；
- bibliography、acknowledgements 与 data availability statement。

因此，不得把 `paper.md` 描述成“全文翻译”“完整复现”或“逐页无遗漏 reader”。若后续 task 要依赖附录中的具体闭式公式，必须先回到 arXiv v3 TeX/PDF 对应 equation 和 proof block 复核，再扩展本包。

## 来源与版本

- arXiv 标识：1912.00829v3
- DOI：10.1103/PhysRevResearch.2.043280
- 使用文件：公开 arXiv v3 TeX source 与 PDF
- PDF 视觉检查：p.1–4 和 p.11 可读，双栏顺序与图号一致；红色方框是本地 PDF 字体/内部链接渲染标记，不是原文内容。
- 图像：直接从 TeX source 中的原始 figure assets 提取；所有 assets 均在 `paper.md` 有对应图卡。

## 术语决策

- `memory-assisted` 保留英文并译作“记忆辅助”，避免误解为通用外部存储硬件。
- `syndrome extraction` 保留 syndrome，强调它是物理测量过程，不等同于仓库的观测数据结构。
- `correction` 在原论文语境译作“纠正位移”；T3.2.1 只输出逻辑 coset 决策，不能写成物理 displacement gate 已实现。
- `posterior mean` 只用于描述原论文的 Gaussian 近似 estimator；T3.2.1 用 posterior logical-class mass 做分类。

## T3.2.1 映射审计

| 论文机制 | T3.2.1 对应 | 一致性 | 禁止外推 |
|---|---|---|---|
| 多轮观测联合 posterior | 20-cycle recursive periodic Bayesian filter | 机制级一致 | 不是论文公式等价实现 |
| 末端一次纠正 | episode 末端一次逻辑 coset 决策 | 时序级一致 | 未实现物理 corrective displacement |
| 已知 Gaussian error width | 冻结 process/measurement covariance | 假设级一致 | 未做在线参数辨识 |
| q/p 可分近似 | 保留相关二维 covariance | 本任务主动强化 | 不能称为论文原算法 |
| 有限能态 fidelity | logical error/NLL/Brier | 指标不同 | 不可比较绝对数值 |
| 电路重编译/offline squeezing | 无 | 未迁移 | 不可形成硬件 claim |

## 不确定性

- 原论文 main text 把多轮 posterior 近似为 multivariate Gaussian；其公式的高阶截断条件只在原模型下成立。
- 当前 task 的 periodic Gaussian observation 是独立工程模型，不能以原论文 tracking/truncation 界替代网格收敛检查。
- reader 中 `Original` 为短摘述/短摘录，旨在控制来源范围；不是逐字全文。

