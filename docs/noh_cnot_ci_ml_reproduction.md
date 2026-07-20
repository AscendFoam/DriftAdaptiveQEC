# T6.17.2 Noh 2022 双 GKP CNOT：CI / analog ML 独立复现

- verdict：`PASS_PROJECT_NATIVE_MATCHED_NOH_CNOT_CI_ML_REPRODUCTION`
- source-sufficiency：`PASS_SOURCE_SUFFICIENCY`
- gates / mutations：15/15 / 15/15
- correctness boundary：0/100,000 mismatch
- runtime：3.063 s（Python correctness workload，非 decoder latency）

## Table I 锚点复现

| dB | method | failures / trials | estimate | literature | abs diff | rel diff | Wilson 95% |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 9 | CI | 6,563 / 65,536 | 0.10014343 | 0.101 | 0.000857 | 0.85% | [0.097868521, 0.10246522] |
| 9 | ML | 4,518 / 65,536 | 0.068939209 | 0.0689 | 3.92e-05 | 0.06% | [0.067024686, 0.070904263] |
| 12 | CI | 5,146 / 589,824 | 0.0087246365 | 0.00869 | 3.46e-05 | 0.40% | [0.0084904825, 0.0089651897] |
| 12 | ML | 2,158 / 589,824 | 0.0036587185 | 0.00361 | 4.87e-05 | 1.35% | [0.0035078345, 0.0038160678] |
| 13 | CI | 6,362 / 2,424,832 | 0.0026236869 | 0.0026 | 2.37e-05 | 0.91% | [0.0025600839, 0.0026888658] |
| 13 | ML | 2,037 / 2,424,832 | 0.0008400582 | 0.000853 | 1.29e-05 | 1.52% | [0.00080437516, 0.00087732279] |

## 证据边界

本 task 只复现 Table I 的 gate-level 对象：两个 square-lattice GKP qubit、`lambda=1`、有限 squeezing 高斯位移是唯一噪声。8 个独立位移和四个净位移按 Appendix C Eq. (C19) 直接采样；CI 按 Eq. (27)，ML 按 Eqs. (30)/(33) 与 Algorithms 1/2；任一整数为奇数即是非平凡 Pauli failure。

100,000 个样本由 50,000 对实际跨越 q/p ML Voronoi facet 的单侧点组成；paper algorithm 与独立 25-candidate likelihood oracle 为零 mismatch。exact tie 的 argmin 非唯一，因此不拿 tie convention 制造假 mismatch。

`9.9 dB` 仍只是文献中 full surface–GKP finite-size threshold。本 task 没有 outer-code lattice、matching graph 或 code-distance crossing，故该值保持 `LITERATURE_ONLY_NULL`；CI `<50 ns`、ML `>1 ms` 和硬件资源也没有被本仿真补写。

## 产物

- report：`docs/t6_17_2_noh_cnot_ci_ml_reproduction.json`
- Source Data：`docs/t6_17_2_noh_cnot_ci_ml_reproduction_source_data.csv`（119 rows）
- implementation：`cnn_fpga/benchmark/surface_gkp_cnot_reproduction.py`
