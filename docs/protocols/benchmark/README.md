# Benchmark Protocols

本目录保存 P4 与论文级 benchmark 扩展相关协议。

## 文件清单

| 文件 | 用途 |
| --- | --- |
| `P4_benchmark_development_protocol.md` | development-layer P4 benchmark protocol |
| `P4_benchmark_formal_protocol.md` | frozen formal software-HIL benchmark protocol and evidence-level rules |
| `paper_benchmark_expansion_protocol.md` | future paper-grade benchmark expansion protocol |

## 边界

`P4_benchmark_formal_protocol.md` 继续保护 `T24` frozen-set anchor；未来 expanded benchmark 不得静默改写历史 `T24` 表格或把 `statcalib` extension lane 并入 frozen table。
