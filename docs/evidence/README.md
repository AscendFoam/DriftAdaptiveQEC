# 新机器证据落盘规则

本目录接收后续任务新生成的 JSON/CSV 机器证据。既有 `docs/` 顶层证据已被代码路径、manifest、release pin 或自哈希绑定，暂不搬入这里；它们通过 [`../evidence_catalog/README.md`](../evidence_catalog/README.md) 阅读。

## 目录与命名

新文件按阶段和 milestone 放置：

```text
docs/evidence/
└── phase9/
    └── milestone_9_3/
        ├── t9_3_1_<slug>.json
        └── t9_3_1_<slug>_source_data.csv
```

- JSON 保存合同、摘要、receipt、verification 或 verdict。
- CSV 保存可复核的 Source Data；字段名、单位和分母必须明确。
- 人类完成记录仍写入 `docs/new_tasks/`，并链接对应机器证据。
- smoke、probe 和未封存中间结果写入系统临时目录，不进入 `docs/`。

## 既有证据何时迁移

只有在对应 task 被重新生成、所有引用同步更新、哈希/manifest 重新封存且相关测试通过时，才迁移一个既有证据组。禁止只为目录美观批量移动。

证据新增或迁移后，运行：

```bash
python scripts/build_evidence_catalog.py
```

