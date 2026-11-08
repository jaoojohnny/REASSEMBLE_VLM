# 切分摘要：bnc_alldev

insert BNC｜group key：`recording_id`｜分層：`has_failure`｜seed 0｜比例 dev 100%

| 集合 | group 數 | 段數 | 失敗 | 成功 | 失敗率 | 含失敗的 group |
|---|---|---|---|---|---|---|
| dev | 62 | 86 | 30 | 56 | 35% | 22 |
| all | 62 | 86 | 30 | 56 | 35% | 22 |

## 依日期（段數／失敗）

| 日期 | dev |
|---|---|
| 2025-01-09 | 23／11 |
| 2025-01-10 | 12／2 |
| 2025-01-11 | 11／3 |
| 2025-01-13 | 34／14 |
| 2025-01-14 | 6／0 |

## 失敗原因分布（gold 標註）

| 原因 | dev |
|---|---|
| board_lifted | 2 |
| incomplete_insert | 13 |
| misalignment | 6 |
| other | 1 |
| slip_or_drop | 3 |
| unknown | 2 |
| wrong_contact | 3 |
