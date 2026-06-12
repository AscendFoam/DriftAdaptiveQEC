# T75 Caption Lock And Placement Notes

## 1. 锁定原则

- `T75-FIG-*` 只服务于当前主线 paper-facing authoring，不新增实验含义。
- 每个 `T75` 资产都必须能回链到上游 `T74-*` stable IDs。
- 如果某个图形位不适合最终使用，必须明确表格替代方案，而不是临时改写证据口径。

## 2. 最终成图锁定表

| `T75` 资产 | 上游 `T74` IDs | 推荐标题 | caption 核心句 | 建议位置 | 版式/尺寸建议 | 替代方案 |
| --- | --- | --- | --- | --- | --- | --- |
| `T75-FIG-M01` | `T74-TBL-01`, `T74-FIG-01` | `Frozen Four-Scenario Benchmark Summary Under the Locked T24 Protocol` | `hybrid_residual_b` 在四个冻结场景均为 winner，`ukf` 均为 runner-up；该图只表示 mock-backed software-HIL 内的 frozen-set ranking | `main text` | 横版；双栏优先；宽 `170-180 mm`；图例置右上，差值注释放右侧 | `T74-TBL-01` |
| `T75-FIG-M02` | `T74-FIG-02`, `T74-TBL-03` | `Six-Seed Mechanism and Intervention Summary` | instability pattern 在多个 seed 上广泛存在，而 lower-clip intervention 的结果是 mixed 且多数 harmful；该图是 descriptive only | `main text` | 横版双 panel；双栏优先；宽 `170-180 mm`；Panel A/B 共用 seed 顺序 | `T74-TBL-03` |
| `T75-FIG-A01` | `T74-FIG-03`, `T74-TBL-05`, `T74-TBL-06`, `T74-SUP-03`, `T74-SUP-04` | `Layered Evidence Boundary From Frozen Benchmark To Deployment-Facing Gates` | 当前 deployment-facing 证据仍是分层边界：isolated true runtime 已验证，而 real-board lane 仍是 gate-only 且 current-host `NO_GO` | `appendix` | 横版 schematic；可单独整页或半页宽；宽 `160-180 mm` | `T74-TBL-05` + `T74-TBL-06` + `T74-SUP-03/04` |

## 3. 锁定后的 caption 草案

### `T75-FIG-M01`

- 推荐标题：
  - `Frozen Four-Scenario Benchmark Summary Under the Locked T24 Protocol`
- 主 caption 核心句：
  - `在锁定的四场景、五模式、paired-seeds+repeats=2 的 formal software revalidation 中，hybrid_residual_b 在四个冻结场景均优于 ukf；该图只表示 mock-backed software-HIL 内的 frozen-set ranking。`
- 推荐补充句：
  - `若最终稿件不保留该图，T74-TBL-01 仍可作为完全等价的 authoritative numeric substitute。`
- 图例与标注锁定：
  - `Hybrid Residual-B` 用冷色主色。
  - `UKF` 用暖色对照色。
  - `Lower is better` 必须显式出现。
  - 每个场景与 `ukf` 的差值可以作为右侧小注释保留。
  - `T76` rendered QA 已把 authoritative-source 说明缩短，并把底部 boundary note 改为双行；语义不变。

### `T75-FIG-M02`

- 推荐标题：
  - `Six-Seed Mechanism and Intervention Summary`
- 主 caption 核心句：
  - `Panel A 汇总各 seed 上的 baseline gap，Panel B 汇总同一批 seed 上的 I1 intervention delta；该图说明 instability pattern 广泛存在，但 tested lower-clip intervention 的效果是 mixed 且多数更差。`
- 推荐补充句：
  - `This figure is descriptive only and must not be read as causal proof, mechanism closure, expanded benchmark evidence, .tflite validation, or real-board validation.`
- 图例与标注锁定：
  - Panel A 用 seed category 配色。
  - Panel B 用 intervention verdict 配色。
  - 两个 panel 的零线都必须明显可见。
  - seed 顺序固定为 `20260425, 20260427, 20260428, 20260429, 20260430, 20260510`。
  - `T76` rendered QA 已把底部 boundary note 改为双行；语义不变。

### `T75-FIG-A01`

- 推荐标题：
  - `Layered Evidence Boundary From Frozen Benchmark To Deployment-Facing Gates`
- 主 caption 核心句：
  - `该图把当前主线 paper-facing 证据划分为 frozen benchmark、descriptive mechanism/ablation、isolated current-host true runtime，以及 read-only real-board gate/provenance 四层，并显式阻断统一 portability/deployment closure 的过度叙事。`
- 推荐补充句：
  - `The blocked portability/deployment closure slot remains intentionally unfilled.`
- 图例与标注锁定：
  - 主结果层用蓝/青色系。
  - `.tflite` boundary 用黄/琥珀色系。
  - real-board gate 与 blocked slot 用红/砖色系。
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` 必须以精简形式出现。
  - `T76` rendered QA 已把三层说明中的长句改为双行；语义不变。

## 4. 表格替代锁定

| 若以下图最终不使用 | 保留的替代表达 |
| --- | --- |
| `T75-FIG-M01` | 直接使用 `T74-TBL-01`，并保留“authoritative frozen ranked table”口径 |
| `T75-FIG-M02` | 主文只留机制/解释段，附录改查 `T74-TBL-03` |
| `T75-FIG-A01` | 直接在 appendix 组合 `T74-TBL-05`、`T74-TBL-06`、`T74-SUP-03`、`T74-SUP-04` |

## 5. 非图资产 placement lock

虽然本文件主要锁定 `T75-FIG-*`，但以下非图资产位置也一并固定，避免后续在 authoring 时漂移：

- `T74-TBL-02`：`appendix`
- `T74-TBL-03`：`appendix`
- `T74-TBL-04`：`appendix`
- `T74-TBL-05`：`appendix`
- `T74-TBL-06`：`supplement only`
- `T74-TBL-07`：`supplement only`
- `T74-SUP-01` 到 `T74-SUP-04`：`supplement only`

## 6. 最重要的锁定点

1. `T75-FIG-M01` 是主结果图，但不是比 `T74-TBL-01` 更强的证据。
2. `T75-FIG-M02` 只能强化 descriptive mechanism layer，不能推动 causal closure。
3. `T75-FIG-A01` 是 boundary schematic，不是 deployment success schematic。
