# 03 Sidecar Artifact Schema

## 1. Manifest

每个 sidecar run root 必须包含：

```text
sidecar_manifest.json
```

最小 schema：

```json
{
  "schema_version": "sidecar_manifest_v2",
  "lane_id": "sidecar_slowloop_temporal_tcn",
  "run_id": "20260612_120000_abcdef0",
  "level": "S1_toy_or_replay",
  "created_from_main_commit": "",
  "workspace_status": "clean_or_dirty_recorded",
  "code_policy": "additive_default_off",
  "historical_run_roots_read": [],
  "new_run_roots": [],
  "source_files_added": [],
  "source_files_modified": [],
  "docs_modified": [],
  "summary_files": [],
  "metrics_files": [],
  "forbidden_claims_acknowledged": {
    "does_not_rewrite_t24": true,
    "does_not_promote_statcalib": true,
    "does_not_claim_tflite_deployment": true,
    "does_not_claim_real_board_validation": true,
    "does_not_claim_paper_grade_expanded_benchmark": true
  },
  "promotion_status": "not_requested"
}
```

## 2. Summary

推荐：

- `sidecar_summary.json`
- `sidecar_summary.csv`

summary 必须包含：

1. lane id。
2. run id。
3. evidence level。
4. 输入来源。
5. 输出指标。
6. 与 frozen anchor 的关系。
7. 明确 negative / inconclusive 结果。

## 3. Metrics

除非任务包另有说明，sidecar metrics 不应覆盖主线 LER 表。可以使用：

- `sidecar_metrics.csv`
- `sidecar_safety_checks.csv`
- `sidecar_replay_table.csv`
- `sidecar_contract_test_results.json`

## 4. 禁止字段语义

manifest 或 summary 不得出现以下完成态 claim，除非处于明确否定语境：

- `real-board validated`
- `tflite deployed`
- `mature calibration comparator`
- `paper-grade benchmark completed`
- `T24 replaced`
- `mainline promoted`

## 5. 保留历史输入

如果读取历史 run root，必须写成精确路径，例如：

```text
runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743
```

不得写成“读取 `runs/` 中所有历史结果”。

