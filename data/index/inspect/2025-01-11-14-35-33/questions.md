# 待確認問題：2025-01-11-14-35-33.h5

由 inspect-h5 依實際檔案自動產生（2026-10-04T18:01:06+08:00）。所有欄位意義都只是推測，請逐項確認或更正；確認後再整理成 `configs/field_mapping.yaml`（第 2 包 indexer 會讀取）。

## 1. 問題清單

| # | 主題 | 問題 |
|---|---|---|
| Q1 | segment 欄位 | `/segments_info`（第 0 層，共 19 段）：start=`start`、end=`end`、success=`success`、text=`text`，其他欄位 ['index']。這些是否就是 segment 起訖、成功標記與動作描述？ |
| Q2 | segment 欄位 | `/segments_info/*/low_level`（第 1 層，共 40 段，分散在 17 個容器）：start=`start`、end=`end`、success=`success`、text=`text`，其他欄位 無。這些是否就是 segment 起訖、成功標記與動作描述？ |
| Q3 | 時間單位與基準 | segment 起訖範圍 1736602533.597588–1736602747.397726 落在 `/timestamps/hand` 的範圍內（100%），推測兩者同一時間基準、單位 s。請確認單位與是否為 Unix 時間。 |
| Q4 | action / part 解析 | 高層 text 第一個詞的分佈：{'pick': 8, 'insert': 7, 'no': 2, 'place': 2}。是否以第一個詞當 action、其餘字串當零件（part）？ |
| Q5 | Insert 零件候選 | 這些樣本中的 Insert 零件：{'USB.': '2 段（成功 1 / 失敗 1）', 'waterproof.': '1 段（成功 1 / 失敗 0）', 'Ethernet.': '1 段（成功 1 / 失敗 0）', 'bolt 4.': '1 段（成功 1 / 失敗 0）', 'large gear.': '1 段（成功 1 / 失敗 0）', 'BNC.': '1 段（成功 0 / 失敗 1）'}。研究要選哪一種？（失敗數需等第 2 包全量索引才準） |
| Q6 | 失敗原因與證據時間 | 資料中沒有失敗原因、證據時間欄位（符合規格預期）。這些將在第 3 包走人工標註流程，請確認。 |
| Q7 | 低階 segment | 第 1 層（如 Low_level）動作分佈：{'grasp': 8, 'lift': 7, 'approach': 9, 'release': 6, 'align': 6, 'push': 2, 'twist': 2}。要不要拿來當證據時間的候選區間？ |
| Q8 | 相機清單 | 偵測到的內嵌影片：{'capture_node-camera-image': '346x260, 5.72 fps, 1231 幀', 'hama1': '640x480, 30 fps, 6415 幀', 'hama2': '640x480, 30 fps, 6414 幀', 'hand': '640x480, 30 fps, 5941 幀'}。哪些是外部視角、哪個是腕部相機？best_camera 的候選集合為何？ |
| Q9 | 影格與時間戳 | capture_node-camera-image：影片 1231 幀、標稱 5.72 fps；`/timestamps/capture_node-camera-image` 有 1231 筆，實測約 5.71941 Hz。第 i 幀是否對應第 i 個時間戳？ |
| Q10 | 影格與時間戳 | hama1：影片 6415 幀、標稱 30 fps；`/timestamps/hama1` 有 6415 筆，實測約 31.0432 Hz。第 i 幀是否對應第 i 個時間戳？ |
| Q11 | 影格與時間戳 | hama2：影片 6414 幀、標稱 30 fps；`/timestamps/hama2` 有 6414 筆，實測約 31.0181 Hz。第 i 幀是否對應第 i 個時間戳？ |
| Q12 | 影格與時間戳 | hand：影片 5941 幀、標稱 30 fps；`/timestamps/hand` 有 5941 筆，實測約 29.9658 Hz。第 i 幀是否對應第 i 個時間戳？ |

## 2. 人工核對 5 段 segment

規格要求在真實樣本中人工核對至少 5 段的起訖與 success。請對照影片（第 3 包會產生預覽圖）。

| # | segment | text | start | end | 長度 (s) | success | 核對 | 備註 |
|---|---|---|---|---|---|---|---|---|
| 1 | `/segments_info/4` | Insert waterproof. | 1736602565.830866 | 1736602595.764194 | 29.933 | True | ☐ 正確 ☐ 有誤 | |
| 2 | `/segments_info/6` | Insert Ethernet. | 1736602603.014243 | 1736602613.330857 | 10.317 | True | ☐ 正確 ☐ 有誤 | |
| 3 | `/segments_info/8` | Insert USB. | 1736602621.214198 | 1736602628.914117 | 7.700 | False | ☐ 正確 ☐ 有誤 | |
| 4 | `/segments_info/11` | Insert USB. | 1736602644.914262 | 1736602661.730831 | 16.817 | True | ☐ 正確 ☐ 有誤 | |
| 5 | `/segments_info/13` | Insert bolt 4. | 1736602668.83398 | 1736602684.180838 | 15.347 | True | ☐ 正確 ☐ 有誤 | |

## 3. 異常（error / warn）

- 無

資訊類（info）：TS_GAPS

## 4. 偵測摘要

- segment 容器 `/segments_info`：第 0 層，19 段（1 個容器），start=`start` end=`end` success=`success` text=`text`
- segment 容器 `/segments_info/*/low_level`：第 1 層，40 段（17 個容器），start=`start` end=`end` success=`success` text=`text`
- 影片 `/capture_node-camera-image`：h264 346x260，標稱 5.72 fps，1231 幀；時間戳 `/timestamps/capture_node-camera-image`（1231 筆，約 5.71941 Hz）
- 影片 `/hama1`：h264 640x480，標稱 30 fps，6415 幀；時間戳 `/timestamps/hama1`（6415 筆，約 31.0432 Hz）
- 影片 `/hama2`：h264 640x480，標稱 30 fps，6414 幀；時間戳 `/timestamps/hama2`（6414 筆，約 31.0181 Hz）
- 影片 `/hand`：h264 640x480，標稱 30 fps，5941 幀；時間戳 `/timestamps/hand`（5941 筆，約 29.9658 Hz）
- 時間陣列 17 個；未配對：無
- 時間基準推測：same_as（參考 `/timestamps/hand`，單位 s）
