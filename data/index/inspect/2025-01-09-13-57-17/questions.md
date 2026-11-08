# 待確認問題：2025-01-09-13-57-17.h5

由 inspect-h5 依實際檔案自動產生（2026-10-04T18:01:05+08:00）。所有欄位意義都只是推測，請逐項確認或更正；確認後再整理成 `configs/field_mapping.yaml`（第 2 包 indexer 會讀取）。

## 1. 問題清單

| # | 主題 | 問題 |
|---|---|---|
| Q1 | segment 欄位 | `/segments_info`（第 0 層，共 5 段）：start=`start`、end=`end`、success=`success`、text=`text`，其他欄位 ['index']。這些是否就是 segment 起訖、成功標記與動作描述？ |
| Q2 | segment 欄位 | `/segments_info/*/low_level`（第 1 層，共 8 段，分散在 3 個容器）：start=`start`、end=`end`、success=`success`、text=`text`，其他欄位 無。這些是否就是 segment 起訖、成功標記與動作描述？ |
| Q3 | 時間單位與基準 | segment 起訖範圍 1736427437.795092–1736427539.861794 落在 `/timestamps/hand` 的範圍內（100%），推測兩者同一時間基準、單位 s。請確認單位與是否為 Unix 時間。 |
| Q4 | action / part 解析 | 高層 text 第一個詞的分佈：{'no': 2, 'insert': 2, 'pick': 1}。是否以第一個詞當 action、其餘字串當零件（part）？ |
| Q5 | Insert 零件候選 | 這些樣本中的 Insert 零件：{'Ethernet.': '2 段（成功 0 / 失敗 2）'}。研究要選哪一種？（失敗數需等第 2 包全量索引才準） |
| Q6 | 失敗原因與證據時間 | 資料中沒有失敗原因、證據時間欄位（符合規格預期）。這些將在第 3 包走人工標註流程，請確認。 |
| Q7 | 低階 segment | 第 1 層（如 Low_level）動作分佈：{'grasp': 1, 'lift': 3, 'approach': 2, 'align': 2}。要不要拿來當證據時間的候選區間？ |
| Q8 | 相機清單 | 偵測到的內嵌影片：{'capture_node-camera-image': '346x260, 5.72 fps, 611 幀', 'hama1': '640x480, 30 fps, 3062 幀', 'hama2': '640x480, 30 fps, 3062 幀', 'hand': '640x480, 30 fps, 2934 幀'}。哪些是外部視角、哪個是腕部相機？best_camera 的候選集合為何？ |
| Q9 | 影格與時間戳 | capture_node-camera-image：影片 611 幀、標稱 5.72 fps；`/timestamps/capture_node-camera-image` 有 611 筆，實測約 5.71513 Hz。第 i 幀是否對應第 i 個時間戳？ |
| Q10 | 影格與時間戳 | hama1：影片 3062 幀、標稱 30 fps；`/timestamps/hama1` 有 3062 筆，實測約 31.1173 Hz。第 i 幀是否對應第 i 個時間戳？ |
| Q11 | 影格與時間戳 | hama2：影片 3062 幀、標稱 30 fps；`/timestamps/hama2` 有 3062 筆，實測約 31.0643 Hz。第 i 幀是否對應第 i 個時間戳？ |
| Q12 | 影格與時間戳 | hand：影片 2934 幀、標稱 30 fps；`/timestamps/hand` 有 2934 筆，實測約 29.976 Hz。第 i 幀是否對應第 i 個時間戳？ |

## 2. 人工核對 5 段 segment

規格要求在真實樣本中人工核對至少 5 段的起訖與 success。請對照影片（第 3 包會產生預覽圖）。

| # | segment | text | start | end | 長度 (s) | success | 核對 | 備註 |
|---|---|---|---|---|---|---|---|---|
| 1 | `/segments_info/0` | No action. | 1736427437.795092 | 1736427449.495045 | 11.700 | True | ☐ 正確 ☐ 有誤 | |
| 2 | `/segments_info/1` | Pick Ethernet. | 1736427449.511793 | 1736427461.895114 | 12.383 | True | ☐ 正確 ☐ 有誤 | |
| 3 | `/segments_info/2` | Insert Ethernet. | 1736427461.911807 | 1736427485.967381 | 24.056 | False | ☐ 正確 ☐ 有誤 | |
| 4 | `/segments_info/3` | Insert Ethernet. | 1736427485.978471 | 1736427504.278364 | 18.300 | False | ☐ 正確 ☐ 有誤 | |
| 5 | `/segments_info/4` | No action. | 1736427504.297776 | 1736427539.861794 | 35.564 | True | ☐ 正確 ☐ 有誤 | |

## 3. 異常（error / warn）

- 無

資訊類（info）：TS_GAPS

## 4. 偵測摘要

- segment 容器 `/segments_info`：第 0 層，5 段（1 個容器），start=`start` end=`end` success=`success` text=`text`
- segment 容器 `/segments_info/*/low_level`：第 1 層，8 段（3 個容器），start=`start` end=`end` success=`success` text=`text`
- 影片 `/capture_node-camera-image`：h264 346x260，標稱 5.72 fps，611 幀；時間戳 `/timestamps/capture_node-camera-image`（611 筆，約 5.71513 Hz）
- 影片 `/hama1`：h264 640x480，標稱 30 fps，3062 幀；時間戳 `/timestamps/hama1`（3062 筆，約 31.1173 Hz）
- 影片 `/hama2`：h264 640x480，標稱 30 fps，3062 幀；時間戳 `/timestamps/hama2`（3062 筆，約 31.0643 Hz）
- 影片 `/hand`：h264 640x480，標稱 30 fps，2934 幀；時間戳 `/timestamps/hand`（2934 筆，約 29.976 Hz）
- 時間陣列 17 個；未配對：無
- 時間基準推測：same_as（參考 `/timestamps/hand`，單位 s）
