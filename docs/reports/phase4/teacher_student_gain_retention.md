# T4.4.4：teacher-student physical gain-retention gate

## 结论

18/18 gates 通过。4-state distilled student 在全新、未参与 teacher 训练或 student 拟合的 paired seeds 上，
保留了 fresh teacher 相对 standard 的绝大部分 finite-model physical gain。cutoff12 三个预注册指标的 point
retention 为 `99.85%/99.74%/99.66%`，paired-bootstrap 95% CI 下界最低 `98.24%`；cutoff16 为
`98.78%/98.98%/98.15%`，最低下界 `94.45%`。全部高于运行前冻结的 `90%` point 与 CI-lower gate。

该结果支持进入 T4.4.5 的**有界 strong branch**：student 高保真保留当前 teacher-vs-standard gain。它不支持
“NMF 普遍优于 MF”：cutoff12 的五-agent MF 平均略高于 teacher，cutoff16 才反转为 teacher 更高。

## 双 horizon 公平协议

- 10-cycle 主 lane：standard、5 个 exact-budget MF agents、selected teacher、冻结 T3.2.10 handcrafted
  recurrence 的显式 horizon extrapolation、distilled student；
- 新 primary seeds 8×64 trajectories，cutoff12；新 confirmation seeds 4×32，cutoff16；共 5,760 条
  policy trajectories，所有模型均冻结，evaluation 不再选 agent、模型或阈值；
- 2-cycle exact lane：对 16 条全部 `g/e` branches 精确枚举，并加入 T3.2.9 finite-horizon control oracle；
- control oracle 从未进入 10-cycle lane，也没有 block-reset/receding-horizon 偷换；
- retention 定义为 `(student-standard)/(teacher-standard)`，对 selection score、fidelity lifetime 和
  logical-Z lifetime 分别计算；20,000 次 paired-seed bootstrap 的每次 teacher gain 分母都为正。

## 10-cycle physical 结果

### Cutoff 12 primary

| strategy | selection score | fidelity lifetime | logical-Z lifetime | observed `p(g)` | residual RMS | slew RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | `0.302451` | `3.570623` | `2.772217` | `0.667578` | `0` | `0` |
| 5-agent MF mean | `0.557115` | `8.622787` | `6.795550` | `0.822793` | `0.167239` | `0.047723` |
| teacher | `0.552952` | `8.439925` | `6.756956` | `0.863965` | `0.166539` | `0.023351` |
| handcrafted recurrence | `0.512103` | `6.642441` | `6.432883` | `0.979102` | `0.675113` | `0.033249` |
| distilled student | `0.552572` | `8.427107` | `6.743237` | `0.864258` | `0.166448` | `0.024415` |

### Cutoff 16 confirmation

| strategy | selection score | fidelity lifetime | logical-Z lifetime | observed `p(g)` | residual RMS | slew RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | `0.462725` | `5.993625` | `5.135107` | `0.774219` | `0` | `0` |
| 5-agent MF mean | `0.579684` | `9.155712` | `7.507638` | `0.825234` | `0.167205` | `0.047751` |
| teacher | `0.593930` | `9.525396` | `7.960814` | `0.875781` | `0.168122` | `0.022955` |
| handcrafted recurrence | `0.502312` | `6.380656` | `6.268072` | `0.978906` | `0.675158` | `0.033267` |
| distilled student | `0.592331` | `9.489395` | `7.908418` | `0.875781` | `0.168083` | `0.024142` |

所有 5 个 MF agents 都逐 seed 保留，以上 MF 是 agent mean，不是按 test lifetime 选择的 best agent。cutoff12
的 MF/teacher 次序与 cutoff16 相反，构成必须保留的 model/cutoff counterevidence。

## Retention 与 burden

| lane | metric | point retention | paired 95% CI |
| --- | --- | ---: | ---: |
| cutoff12 | selection score | `0.998481` | `[0.990787,1.004547]` |
| cutoff12 | fidelity lifetime | `0.997368` | `[0.987511,1.004886]` |
| cutoff12 | logical-Z lifetime | `0.996557` | `[0.982442,1.008014]` |
| cutoff16 | selection score | `0.987812` | `[0.967051,0.999901]` |
| cutoff16 | fidelity lifetime | `0.989806` | `[0.973811,0.999138]` |
| cutoff16 | logical-Z lifetime | `0.981457` | `[0.944501,1.000600]` |

student/teacher 的 stochastic `p(g)` 差为 `0.000293/0`，exact 2-cycle 为 `0.002126/0.002328`，均小于
冻结容差 `0.02`。10-cycle cutoff12 的 observed e fraction 为 teacher `0.136035`、student `0.135742`；
cutoff16 两者均 `0.124219`。handcrafted recurrence 的 e fraction 仅约 `0.021`，但 residual RMS 约为
teacher/student 的 4 倍，且 physical score 更低，因此不能只用 `p(g)` 选择策略。

当前 simulator 为 two-level ancilla，原生 multilevel leakage event 不存在；artifact 对每个 burden row 把 leakage
写为 `null`，没有把 e 偷标为 leakage。student 的外部 leakage token 仍只触发 reset+exact zero residual，
不构成 trained physical leakage response。

## 2-cycle exact control-oracle lane

16 条 branches 的概率和误差小于 `1e-12`。cutoff12 control oracle terminal fidelity 为 `0.815799`，高于
teacher/student 的 `0.596000/0.595391`，但其 area selection score `0.591201` 低于 teacher/student
`0.668528/0.667209`，因为它只优化 terminal fidelity。cutoff16 是 frozen transfer，oracle terminal fidelity
`0.638688` 也不构成全局上界。其非凸 multi-start、短 horizon 和 objective-specific 性质均保留，不能写成
10-cycle oracle lifetime。

exact student gain retention 在 cutoff12 四指标为 `98.49%--99.69%`，cutoff16 为 `97.61%--98.14%`。

## 成本与非 demo 审计

| strategy | stored scalars | state | analytic MAC/half-cycle | 边界 |
| --- | ---: | ---: | ---: | --- |
| exact-budget MF | 72,853 | 0 | 72,266 | float model |
| fresh teacher | 72,853 | 10 | 72,266 | offline float teacher |
| handcrafted recurrence | 105 | 15 | 45 | 2-cycle trained；10-cycle frozen extrapolation |
| distilled student | 95 | 4 | 87 | pure-NumPy float artifact |
| 2-cycle control oracle | 225 | 4-bit history | table read | exponential table growth；nondeployable |

student 相对 teacher 的 stored scalars/MAC 降低 `99.8696%/99.8796%`。这些是解析 float counts，未包含
fixed-point、address/control、BRAM/DSP、Fmax、deadline、RTL、FPGA 或 board measurement。

448-row Source Data 保留 108 个 stochastic policy-agent-seed summaries、320 个 exact branches、14 个
retention gates 和 6 个 cost rows。direct test 还覆盖 threshold 防篡改、undefined bootstrap fail-closed、
student raw-head simulator mapping、全部 MF agents、oracle horizon、burden null 与 artifact/source hashes。

产物：

- `cnn_fpga/benchmark/teacher_student_gain_retention.py`
- `tests/test_teacher_student_gain_retention.py`
- `docs/t4_4_4_teacher_student_gain_retention.json`
- `docs/t4_4_4_teacher_student_gain_retention_source_data.csv`
