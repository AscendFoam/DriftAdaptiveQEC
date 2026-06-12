# T75 Authoring Do-Not-Write List

## 1. 使用方式

本文件不是一般性的写作建议，而是当前主线 authoring 的禁写清单。任何主文、caption、appendix bridge 或 rebuttal 草稿，只要踩到这里列出的表述，就应回退到更保守的替代口径。

## 2. 核心禁写项

| 主题 | 不得写成什么 | 为什么不能这么写 | 当前安全替代表述 |
| --- | --- | --- | --- |
| `T48` | `default-env recovered` / `HIL closure` / `deployment closure` | `T48` 只验证了 isolated current-host true `.tflite` runtime | `isolated current-host true runtime verified for selected preserved float/int8 artifacts` |
| `T49/T71/T72` | `real-board execution success` / `hardware validated` / `P3 real-board HIL complete` | 当前 replay 与 regeneration 都仍是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` | `read-only real-board gate / regeneration / provenance boundary with current-host NO_GO` |
| `FR8` | `promoted mature comparator` / `final calibrated mainline` / `unique threshold found` | `T70` 明确仍是 extension lane + no-promotion + persistent tie set | `separately labeled extension-lane closure with no-promotion and no unique clean threshold` |
| `T74-FIG-04` | `可以补一张统一 portability/deployment 图` | 当前证据不支持诚实画出统一闭环图 | `blocked figure slot; no unified portability/deployment closure claim` |
| `FR6/FR7` | `causal closure` / `teacher necessity proved` / `mechanism solved` | 现有证据只支持 descriptive mechanism + bounded ablation | `descriptive multi-seed mechanism evidence` / `bounded frozen-set ablation support` |
| `T24` | `paper-grade expanded benchmark` / `runtime-agnostic superiority` / `board-level result` | `T24` 仍只是 frozen-set mock-backed software-HIL revalidation | `authoritative frozen ranked table inside the locked software-HIL benchmark set` |

## 3. 典型危险句式

以下句式当前都不应出现在主文、caption、附录桥接或图注中：

1. `We validate deployment on preserved TFLite and board-facing paths.`
2. `The real-board execution lane confirms the same ranking.`
3. `Statcalib emerges as the promoted calibration comparator.`
4. `The six-seed mechanism figure establishes the causal reason for the gain.`
5. `Teacher-side inputs are shown to be necessary for the improvement.`
6. `A unified portability pipeline is now in place from training to deployment.`

## 4. 对应替代表述

| 如果你想写的是 | 请改成 |
| --- | --- |
| `.tflite` 路径已经打通 | 当前机器在 isolated environment 下已验证 selected preserved true `.tflite` artifacts 的真实执行 |
| 真板方向也有准备 | 仓库已有 read-only real-board gate / regeneration / provenance 包，但 current-host 仍是 `NO_GO` |
| `statcalib` 很有前景 | `statcalib` 在 bounded extension lane 中展示了有价值的 closure signal，但当前 gate 仍为 no-promotion |
| 机制已经解释清楚 | 现有 multi-seed 证据支持 descriptive interpretation，而不是 causal closure |

## 5. 图表 authoring 专项禁写

### `T75-FIG-M01`

- 不得暗示：
  - expanded benchmark
  - `.tflite` ranking
  - real-board ranking
- 只能表示：
  - `T24` 锁定协议下的 frozen-set ranking

### `T75-FIG-M02`

- 不得暗示：
  - causal mechanism closure
  - intervention success proof
  - teacher necessity proof
- 只能表示：
  - descriptive multi-seed instability/intervention reading

### `T75-FIG-A01`

- 不得暗示：
  - deployment success schematic
  - unified portability pipeline
  - board-ready execution path
- 只能表示：
  - layered evidence boundary and blocked closure slot

## 6. 最高优先级提醒

当前最需要避免的不是“写少了”，而是把分层证据压扁成统一闭环叙事。只要出现下面任一倾向，就应立刻回退：

1. 把 `isolated current-host` 写没了。
2. 把 `NO_GO` 写没了。
3. 把 `extension lane` 写没了。
4. 把 `blocked` 写成“只差一张图”。

### T76 rendered QA 备注

- `T76` 即使对 `T75` 图资产做了版式可读性修正，也不改变本清单中的任何禁写边界。
