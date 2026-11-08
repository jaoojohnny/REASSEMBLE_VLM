# 待確認問題：synthetic_normal.h5

由 inspect-h5 依實際檔案自動產生（2026-10-04T17:55:07+08:00）。所有欄位意義都只是推測，請逐項確認或更正；確認後再整理成 `configs/field_mapping.yaml`（第 2 包 indexer 會讀取）。

## 1. 問題清單

| # | 主題 | 問題 |
|---|---|---|
| Q1 | segment 欄位 | `/segments_info`（第 0 層，共 4 段）：start=`start`、end=`end`、success=`success`、text=`text`，其他欄位 無。這些是否就是 segment 起訖、成功標記與動作描述？ |
| Q2 | segment 欄位 | `/segments_info/*/Low_level`（第 1 層，共 12 段，分散在 4 個容器）：start=`start`、end=`end`、success=`success`、text=`text`，其他欄位 無。這些是否就是 segment 起訖、成功標記與動作描述？ |
| Q3 | 時間單位與基準 | segment 起訖範圍 1736420000.1–1736420002.9 落在 `/timestamps/cam_left` 的範圍內（100%），推測兩者同一時間基準、單位 s。請確認單位與是否為 Unix 時間。 |
| Q4 | action / part 解析 | 高層 text 第一個詞的分佈：{'pick': 2, 'insert': 2}。是否以第一個詞當 action、其餘字串當零件（part）？ |
| Q5 | Insert 零件候選 | 這些樣本中的 Insert 零件：{'round peg 1': '1 段（成功 1 / 失敗 0）', 'square peg 2': '1 段（成功 0 / 失敗 1）'}。研究要選哪一種？（失敗數需等第 2 包全量索引才準） |
| Q6 | 失敗原因與證據時間 | 資料中沒有失敗原因、證據時間欄位（符合規格預期）。這些將在第 3 包走人工標註流程，請確認。 |
| Q7 | 低階 segment | 第 1 層（如 Low_level）動作分佈：{'approach': 4, 'align': 4, 'push': 4}。要不要拿來當證據時間的候選區間？ |
| Q8 | 相機清單 | 偵測到的內嵌影片：{'cam_left': '32x24, 30 fps, 90 幀', 'cam_right': '32x24, 30 fps, 90 幀', 'cam_wrist': '32x24, 30 fps, 90 幀'}。哪些是外部視角、哪個是腕部相機？best_camera 的候選集合為何？ |
| Q9 | 影格與時間戳 | cam_left：影片 90 幀、標稱 30 fps；`/timestamps/cam_left` 有 90 筆，實測約 30.0447 Hz。第 i 幀是否對應第 i 個時間戳？ |
| Q10 | 影格與時間戳 | cam_right：影片 90 幀、標稱 30 fps；`/timestamps/cam_right` 有 90 筆，實測約 29.9591 Hz。第 i 幀是否對應第 i 個時間戳？ |
| Q11 | 影格與時間戳 | cam_wrist：影片 90 幀、標稱 30 fps；`/timestamps/cam_wrist` 有 90 筆，實測約 29.8491 Hz。第 i 幀是否對應第 i 個時間戳？ |

## 2. 人工核對 5 段 segment

規格要求在真實樣本中人工核對至少 5 段的起訖與 success。請對照影片（第 3 包會產生預覽圖）。

| # | segment | text | start | end | 長度 (s) | success | 核對 | 備註 |
|---|---|---|---|---|---|---|---|---|
| 1 | `/segments_info/0` | Pick round peg 1 | 1736420000.1 | 1736420000.7 | 0.600 | True | ☐ 正確 ☐ 有誤 | |
| 2 | `/segments_info/1` | Insert round peg 1 | 1736420000.7 | 1736420001.4 | 0.700 | True | ☐ 正確 ☐ 有誤 | |
| 3 | `/segments_info/2` | Pick square peg 2 | 1736420001.4 | 1736420002.1 | 0.700 | True | ☐ 正確 ☐ 有誤 | |
| 4 | `/segments_info/3` | Insert square peg 2 | 1736420002.1 | 1736420002.9 | 0.800 | False | ☐ 正確 ☐ 有誤 | |

## 3. 異常（error / warn）

- 無

資訊類（info）：TS_GAPS

## 4. 偵測摘要

- segment 容器 `/segments_info`：第 0 層，4 段（1 個容器），start=`start` end=`end` success=`success` text=`text`
- segment 容器 `/segments_info/*/Low_level`：第 1 層，12 段（4 個容器），start=`start` end=`end` success=`success` text=`text`
- 影片 `/cam_left`：mpeg4 32x24，標稱 30 fps，90 幀；時間戳 `/timestamps/cam_left`（90 筆，約 30.0447 Hz）
- 影片 `/cam_right`：mpeg4 32x24，標稱 30 fps，90 幀；時間戳 `/timestamps/cam_right`（90 筆，約 29.9591 Hz）
- 影片 `/cam_wrist`：mpeg4 32x24，標稱 30 fps，90 幀；時間戳 `/timestamps/cam_wrist`（90 筆，約 29.8491 Hz）
- 音訊 `/mic_wrist`：? Hz；時間戳 `/timestamps/mic_wrist`
- 時間陣列 7 個；未配對：無
- 時間基準推測：same_as（參考 `/timestamps/cam_left`，單位 s）
