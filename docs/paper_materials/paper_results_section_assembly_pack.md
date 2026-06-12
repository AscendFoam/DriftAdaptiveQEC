# T76 Results Section Assembly Pack

## 1. 目的

`T75` 已经给出了 bounded prose、caption lock 和 appendix bridge；`T76` 在此基础上再固定一层更接近落稿的装配顺序：

- 主文段落先后顺序；
- 图表插入点；
- 若版面受限时的 fallback 路线；
- 每一段必须保留的边界句。

本文件仍然不等于 full-manuscript reopen，也不把 supporting material 升格成主结果。

## 2. 推荐主文顺序

### 段落 A：冻结主结果段

- 主锚点：`T75-FIG-M01`
- authoritative numeric fallback：`T74-TBL-01`
- 推荐位置：Results 第一段或第一小节
- 建议写法：
  - 先给出 locked `T24` protocol 下四场景 winner / runner-up 关系；
  - 再补一句 “the figure is a publication-facing compression of `T74-TBL-01` only”。
- 必留边界：
  - frozen-set
  - mock-backed software-HIL only
  - not expanded benchmark / `.tflite` ranking / real-board ranking

### 段落 B：机制/解释层段

- 主锚点：`T75-FIG-M02`
- appendix fallback：`T74-TBL-03`
- 推荐位置：紧跟段落 A
- 建议写法：
  - 先说明 instability pattern 在多 seed 上广泛存在；
  - 再说明 tested lower-clip intervention 是 mixed 且多数 harmful；
  - 最后补一句 “descriptive only, not causal closure”。
- 必留边界：
  - descriptive multi-seed evidence
  - no intervention-success proof
  - no teacher-necessity proof

### 段落 C：边界/限制收口段

- 主文只保留 1 到 2 句
- appendix 主锚点：`T75-FIG-A01`
- appendix / supplement 关联项：
  - `T74-TBL-05`
  - `T74-TBL-06`
  - `T74-SUP-03`
  - `T74-SUP-04`
- 建议写法：
  - 先说明 deployment-facing evidence 仍然是 layered；
  - 再明确 isolated true runtime 与 read-only real-board gate 的不同边界；
  - 最后阻断 unified portability / deployment closure 叙事。
- 必留边界：
  - `T48` = isolated current-host true runtime only
  - `T49/T71/T72` = read-only real-board gate/provenance with current-host `NO_GO`
  - `T74-FIG-04` remains blocked

## 3. 图表插入建议

| 段落 | 首选图表 | 次选 / fallback | 推荐落点 | 备注 |
| --- | --- | --- | --- | --- |
| A. 冻结主结果 | `T75-FIG-M01` | `T74-TBL-01` | 第一组 Results 主图 | 如果版面更偏表格，直接退回 `T74-TBL-01` |
| B. 机制/解释层 | `T75-FIG-M02` | 文中简述 + `T74-TBL-03` 放 appendix | 主图后的解释图 | 不要把它改写成“为什么一定成功”的证明图 |
| C. 边界/限制收口 | 主文只写文字；appendix 用 `T75-FIG-A01` | `T74-TBL-05` + `T74-TBL-06` + `T74-SUP-03/04` | Results 末尾 bridge 到 appendix | 保持 appendix-only boundary schematic 定位 |

## 4. 最小安全装配包

如果当前目标只是形成 paper-ready 的主文 Results + appendix 路线，最小安全组合是：

1. 主文：
   - `T75-FIG-M01`
   - `T75-FIG-M02`
   - 必要时正文内提及 `T74-TBL-01`
2. Appendix：
   - `T74-TBL-02`
   - `T74-TBL-03`
   - `T74-TBL-04`
   - `T74-TBL-05`
   - `T75-FIG-A01`
3. Supplement：
   - `T74-TBL-06`
   - `T74-TBL-07`
   - `T74-SUP-01` 到 `T74-SUP-04`

## 5. 不能改动的 fallback 原则

### `T75-FIG-M01`

- 允许 fallback 到 `T74-TBL-01`
- 不允许把 `T75-FIG-M01` 写成比 `T74-TBL-01` 更强的证据

### `T75-FIG-M02`

- 允许把查数职责退回 `T74-TBL-03`
- 不允许把 `T75-FIG-M02` 改写成 causal explanation

### `T75-FIG-A01`

- 允许退回 `T74-TBL-05` / `T74-TBL-06` / `T74-SUP-03/04` 的分表表达
- 不允许把 `T75-FIG-A01` 扩写成 deployment success schematic

## 6. 收口句模板

- `The frozen benchmark result is anchored by the locked T24 protocol and remains the paper's main result layer.`
- `The multi-seed mechanism figure is descriptive support rather than causal closure.`
- `Deployment-facing evidence remains layered: isolated true runtime is verified on the current host, whereas the real-board lane remains gate-only and NO_GO.`

## 7. 本文件的边界

本文件只锁定 Results 段落装配顺序，不处理：

- 全文引言/方法/讨论重写；
- 新图位发明；
- `.tflite`、real-board、`FR8` 的证据升级；
- `T74-FIG-04` 的补画。
