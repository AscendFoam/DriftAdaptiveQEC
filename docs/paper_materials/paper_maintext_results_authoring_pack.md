# T75 Main-Text Results Authoring Pack

## 1. 任务边界

`T75` 不是新实验，也不是 full-manuscript reopen。它只把 `T74` 已经冻结的 stable-ID 材料包压缩成一套可以直接用于主文 Results authoring 的句胚、图表入口和边界说明。

本文件所有主段落都必须：

- 显式回指上游 `T74-*` stable IDs；
- 写清 `可写表述` 与 `不可写表述`；
- 保持 `T24` / `T48` / `T72` / `FR8` 的既有 evidence boundary 不升级。

## 2. 主文 authoring 路线

| authoring slot | 推荐资产 | 上游 `T74` IDs | 当前状态 | 说明 |
| --- | --- | --- | --- | --- |
| 主结果可视化入口 | `T75-FIG-M01` | `T74-TBL-01`, `T74-FIG-01` | `ready` | `T75-FIG-M01` 是对 `T24` 冻结主表的 task-local 最终成图；若版面不适合，`T74-TBL-01` 仍是 authoritative substitute |
| 机制/解释入口 | `T75-FIG-M02` | `T74-FIG-02`, `T74-TBL-03` | `ready` | 主文只保留 descriptive mechanism/intervention reading；更细的 `FR7` 数值放附录 |
| 边界说明入口 | `T75-FIG-A01` | `T74-FIG-03`, `T74-TBL-05`, `T74-TBL-06`, `T74-SUP-03`, `T74-SUP-04` | `ready` | 这是 appendix boundary schematic，不把 deployment-adjacent 材料挤进主结果段 |

## 3. 主文段落包

### 段落 A：冻结主结果段

- 上游 `T74` IDs：
  - `T74-TBL-01`
  - `T74-FIG-01`
- 推荐主文入口：
  - 首选 `T75-FIG-M01`
  - 若版面或审稿风格更偏向表格，可直接退回 `T74-TBL-01`
- 中文句胚：
  - `在锁定的四场景、五模式、paired-seeds+repeats=2 的 formal software revalidation 中，hybrid_residual_b 在四个冻结场景中均取得最低 final LER，ukf 在四个场景中均为 runner-up。我们因此将 T24 的 frozen-set 排名作为当前最稳的主结果入口，并把它限定在 mock-backed software-HIL 的证据层内，而不把它外推成 expanded benchmark、.tflite 或 real-board 结论。`
- 极短英文句胚：
  - `Under the locked T24 protocol, hybrid_residual_b is the winner in all four frozen scenarios, with ukf as the runner-up throughout.`
- 可写表述：
  - `frozen four-scenario benchmark result`
  - `authoritative frozen ranked table`
  - `mock-backed software-HIL ranking`
  - `bounded formal software revalidation`
- 不可写表述：
  - `paper-grade expanded benchmark`
  - `deployment-facing result`
  - `real-board validated ranking`
  - `runtime-agnostic superiority`

### 段落 B：机制与解释段

- 上游 `T74` IDs：
  - `T74-FIG-02`
  - `T74-TBL-03`
  - `T74-TBL-02`
- 推荐主文入口：
  - 主文图用 `T75-FIG-M02`
  - `FR7` 作为附录 bridge 支持解释层，而不是主结果替代表
- 中文句胚：
  - `作为对主结果的解释层补充，FR6 的六 seed 图包显示，Gated v5 相对 Full 的 instability pattern 在多个 seed 上广泛出现，但幅度与方向存在差异；同一批证据还表明，本次 lower-clip I1 intervention 的效果是 mixed 且多数更差。与此一致，FR7 的 bounded frozen-set ablation 说明 feature/teacher 配置会显著影响结果，但它并不支持把 teacher 相关输入写成简单的必要性或因果闭环结论。`
- 极短英文句胚：
  - `The multi-seed evidence is descriptive rather than causal: the instability pattern repeats broadly, while the tested lower-clip intervention is mixed and mostly harmful.`
- 可写表述：
  - `descriptive multi-seed mechanism evidence`
  - `bounded ablation support`
  - `interpretation layer for the frozen-set result`
  - `feature/teacher configuration materially changes the frozen-set outcome`
- 不可写表述：
  - `causal closure`
  - `teacher design is proved necessary`
  - `the intervention fixes the mechanism`
  - `general mechanism proof beyond the bounded six-seed pack`

### 段落 C：边界与部署限制段

- 上游 `T74` IDs：
  - `T74-FIG-03`
  - `T74-TBL-05`
  - `T74-TBL-06`
  - `T74-SUP-03`
  - `T74-SUP-04`
- 推荐放置：
  - 主文末尾用一句或两句收口
  - 详细图解与表格放 appendix / supplement
- 中文句胚：
  - `在 simulation/material 主结果之外，当前仓库还具备两层部署相邻证据：其一是 current-host isolated true .tflite runtime 已对选定 preserved float/int8 artifact 完成真实执行与一致性校验；其二是 read-only real-board gate / regeneration / provenance 包已经形成，但 current-host verdict 仍然是 NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE。我们因此把 deployment-facing 材料明确保留为分层边界说明，而不把它们压缩成统一 portability 或 deployment closure 叙事。`
- 极短英文句胚：
  - `The deployment-facing evidence remains layered: isolated true runtime is verified on the current host, whereas the real-board lane is still gate-only and NO_GO.`
- 可写表述：
  - `layered boundary evidence`
  - `isolated current-host true runtime`
  - `read-only real-board gate/provenance`
  - `no unified portability/deployment closure claim`
- 不可写表述：
  - `default environment recovered`
  - `HIL closure`
  - `real-board execution success`
  - `deployment closure`

### 段落 D：主文收尾到附录的过渡句

- 上游 `T74` IDs：
  - `T74-TBL-02`
  - `T74-TBL-03`
  - `T74-TBL-04`
  - `T74-TBL-05`
  - `T74-FIG-03`
- 中文句胚：
  - `附录随后补充三类 supporting material：一是 FR7/FR6 的 ablation 与 descriptive cross-seed 数值表，二是 training/material 与 isolated true .tflite runtime 的边界表，三是分层 evidence-boundary schematic，用以说明哪些材料属于 supporting evidence，哪些仍然不能升级为主结果。`
- 可写表述：
  - `appendix support tables`
  - `boundary tables`
  - `evidence-boundary schematic`
- 不可写表述：
  - `appendix contains stronger deployment validation`
  - `appendix upgrades the main result`

## 4. `T75-FIG-M01` 与 `T74-TBL-01` 的关系

- `T75-FIG-M01` 是对 `T74-TBL-01` 的 task-local publication-facing visual compression。
- 数值 authoritative source 仍然是 `T74-TBL-01` 与其上游 `comparison.csv`。
- 如果后续稿件版面、期刊风格或审稿意见更偏向表格，而不是结果图，允许直接用 `T74-TBL-01` 替代 `T75-FIG-M01`，且不改变当前主结果语义。
- 不允许反过来把 `T75-FIG-M01` 写成比 `T74-TBL-01` 更强的证据。

## 5. 本文件的最安全用法

1. 先用段落 A 锁定主文 frozen result wording。
2. 再用段落 B 给出最保守的 mechanism/explanation layer。
3. 最后用段落 C 显式阻断 deployment overclaim，并用段落 D 过渡到 appendix。

### T76 rendered QA 备注

- `T76` 只对 `T75-FIG-M01`、`T75-FIG-M02` 和 `T75-FIG-A01` 做了 rendered-preview 可读性修正：缩短或换行图中说明文字，避免真实预览时出现裁切。
- 这些修正不改变本文件的 Results authoring 语义，也不改变 `T75-FIG-*` 到 `T74-*` 的回链关系。

这样写的核心目标不是“把 paper 写满”，而是先把当前最稳、最不容易越界的 Results authoring 路线固定下来。
