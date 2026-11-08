# 切分摘要：bnc_dev25_test75_s0

insert BNC｜group key：`recording_id`｜分層：`has_failure`｜seed 0｜比例 dev 25%, test 75%

| 集合 | group 數 | 段數 | 失敗 | 成功 | 失敗率 | 含失敗的 group |
|---|---|---|---|---|---|---|
| dev | 14 | 20 | 7 | 13 | 35% | 4 |
| test | 48 | 66 | 23 | 43 | 35% | 18 |
| all | 62 | 86 | 30 | 56 | 35% | 22 |

## 依日期（段數／失敗）

| 日期 | dev | test |
|---|---|---|
| 2025-01-09 | 1／0 | 22／11 |
| 2025-01-10 | 4／1 | 8／1 |
| 2025-01-11 | 3／0 | 8／3 |
| 2025-01-13 | 12／6 | 22／8 |
| 2025-01-14 | 0／0 | 6／0 |

## 失敗原因分布（gold 標註）

| 原因 | dev | test |
|---|---|---|
| board_lifted | 0 | 2 |
| incomplete_insert | 4 | 9 |
| misalignment | 1 | 5 |
| other | 0 | 1 |
| slip_or_drop | 2 | 1 |
| unknown | 0 | 2 |
| wrong_contact | 0 | 3 |
