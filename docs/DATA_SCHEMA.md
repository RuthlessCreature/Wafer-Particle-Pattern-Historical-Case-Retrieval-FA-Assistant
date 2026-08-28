# 数据模型

SQLite 表：`cases`

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 主键 |
| image_path | TEXT | 入库后的原图副本 |
| normalized_path | TEXT | wafer 归一化预览 |
| comment | TEXT | 历史工程师 comment / FA 结论 |
| metadata_json | TEXT | 可扩展 JSON |
| feature_blob | BLOB | numpy compressed feature bundle |
| feature_version | TEXT | 特征算法版本 |
| created_at | TEXT | 入库时间 |

## Metadata 建议

```json
{
  "tool": "OX-01",
  "chamber": "C3",
  "recipe": "RCP-A",
  "process_step": "OXIDE-1",
  "lot": "LOT123",
  "wafer_id": "W07",
  "fa_root_cause": "downstream line contamination",
  "action": "wet clean",
  "result": "recovered"
}
```

v0.1.0 不强制 schema，是为了先让现场低成本收集数据；进入正式生产阶段后应把关键字段升级成受控枚举/结构化字段。
