# 第 1 包人工核對結果（2026-10-04）

工具：`check_segments.bat`（`data/derived/manual_check/`，4 視角依時間戳對齊，綠框＝標註區間，前後各 1.5 秒緩衝）
核對者：使用者

| # | 錄製檔 | segment | 動作描述 | 標註 success | 長度 (s) | 起訖 | success |
|---|---|---|---|---|---|---|---|
| 1 | 2025-01-09-13-57-17 | /segments_info/2 | Insert Ethernet. | False | 24.1 | 正確 | 正確 |
| 2 | 2025-01-11-14-35-33 | /segments_info/8 | Insert USB. | False | 7.7 | **結束早 1–2 秒** | 正確 |
| 3 | 2025-01-09-13-57-17 | /segments_info/3 | Insert Ethernet. | False | 18.3 | 正確 | 正確 |
| 4 | 2025-01-11-14-35-33 | /segments_info/4 | Insert waterproof. | True | 29.9 | 正確 | 正確 |
| 5 | 2025-01-14-12-05-01 | /segments_info/4 | Insert square peg 1. | True | 14.4 | 正確 | 正確 |

## 觀察

- 相鄰的高層 segment 幾乎無縫相接：3 個樣本共 59 個間隔，最小 0.011 s、中位 0.017 s、最大 0.020 s。
  第 2 段沒包進去的 1–2 秒落在下一段 "Place USB." 的開頭。
- 第 2 段的低階技能只有一個 "Approach"（success=False），涵蓋整段。

## 決定

- `configs/field_mapping.yaml` 的 `extraction.end_buffer_sec = 2.0`：取「最後狀態」影格時看到標註結束後 2 秒，允許跨進下一段。
- 第 2 包索引記錄每段與下一段的間隔（`next_gap_s`），方便之後檢查緩衝是否碰到下一個動作的實質內容。
- 其他決定：best_camera 候選 hama1 / hama2 / hand；排除 "No action."；第一個字＝動作、其餘去句號＝零件；全量 149 檔過夜索引。
