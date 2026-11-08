# 第 2 包人工核對結果（2026-10-05）

工具：`check_part.bat`（從 `data/index/segments.csv` 抽 Insert BNC 5 段，seed 0，只取驗證 ok 的段，讀 `data/slim/` 精簡版）
輸出：`data/derived/manual_check_part/`（4 視角依時間戳對齊，綠框＝標註區間，前後各 1.5 秒緩衝）
核對者：使用者　結論：**5 段起訖與 success 全部正確**

| # | 錄製檔 | segment | 標註 success | 長度 (s) | 起訖 | success |
|---|---|---|---|---|---|---|
| 1 | 2025-01-09-17-23-07 | /segments_info/14 | False | 36.8 | 正確 | 正確 |
| 2 | 2025-01-10-17-25-29 | /segments_info/30 | False | 23.6 | 正確 | 正確 |
| 3 | 2025-01-13-11-23-17 | /segments_info/26 | False | 9.5 | 正確 | 正確 |
| 4 | 2025-01-09-14-28-47 | /segments_info/11 | True | 45.7 | 正確 | 正確 |
| 5 | 2025-01-09-17-03-05 | /segments_info/6 | True | 35.2 | 正確 | 正確 |

第 2 包停止條件「選出一種操作，索引 5 段人工核對」已達成：研究零件 Insert BNC（86 段、失敗 30、62 個錄製檔）。
