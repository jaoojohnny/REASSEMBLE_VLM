# 第 1 包交付報告：檢查現有檔案、建立環境、inspect HDF5

日期：2026-10-04 ｜ 電腦：desktop-or33krg（Windows 10）｜ 專案：`C:\Users\user\Desktop\REASSEMBLE_VLM`

## 1. 本包結果

| 規格項目 | 狀態 |
|---|---|
| 檢查使用者現有檔案與程式 | 使用者表示沒有既有探索程式；從零建立 |
| 建立環境 | conda 環境 `reassemble-vlm`：Python 3.11.16、h5py 3.16.0 / HDF5 2.0.0、numpy 2.4.6、PyAV 18.1.0、OpenCV 5.0.0、pytest 9.1.1 |
| 硬體 | NVIDIA GeForce RTX 3060 12 GB（driver 560.94）、24 核心、31.7 GB RAM、C 槽剩 314 GB |
| 取得 1–3 個真實樣本 | 3 個 .h5（下載 720.8 MB、解壓 3.2 GB），CRC-32 全部通過，SHA-256 記在 `data/raw/manifest.json` |
| inspect_h5（P1-1、P1-2） | 3 檔都產生 tree.txt / candidates.json / questions.md / 映射提案；0 error、0 warn |
| 合成資料測試 | Windows 上 pytest 36/36 通過（之後新增 4 個音訊相關測試，雲端 40/40；下次執行 `setup_windows.bat` 會一起跑） |
| **停止點** | **等使用者核對 tree.txt、欄位候選與 5 段 segment** |

## 2. 執行過的命令

```bat
setup_windows.bat        :: 建環境 → 測試 → 下載樣本（使用者輸入 y）→ inspect
:: 等同於：
conda create -n reassemble-vlm -c conda-forge --override-channels python=3.11 pip
pip install -r requirements.txt && pip install -e .
python -m pytest -q
python -m scripts.fetch_samples            :: spread 策略：依錄製時間取頭/中/尾
python -m scripts.inspect_h5 data\raw
```

## 3. 樣本與輸出

| 樣本 | 解壓大小 | 長度 | 高層段數 | 低層段數 | Insert（成/敗） |
|---|---|---|---|---|---|
| 2025-01-09-13-57-17.h5 | 364 MB | 約 102 s | 5 | 8 | 2（0/2） |
| 2025-01-11-14-35-33.h5 | 1.2 GB | 約 214 s | 19 | 40 | 7（5/2） |
| 2025-01-14-12-05-01.h5 | 1.7 GB | 約 273 s | 38 | 75 | 1（1/0） |

輸出：`data/index/inspect/<檔名>/`（tree.txt、nodes.json、candidates.json、questions.md、field_mapping.proposed.yaml）與總表 `data/index/inspect/_summary.md`；zip 全部 149 個檔的清單在 `data/index/zip_listing.csv`。

## 4. 實際觀察到的結構（三檔一致）

```text
/                                   11 項
├── hama1, hama2, hand              內嵌 mp4（h264 640x480，容器標稱 30 fps）
├── capture_node-camera-image       內嵌 mp4（h264 346x260，標稱 5.7–7.6 fps）
├── hama1_audio, hama2_audio, hand_audio   int64 一維陣列，每個元素是一個位元組（開頭 FF F3 = MP3）
├── events                          (N, 3) int64，每秒約 9–18 萬筆
├── robot_state/                    12 個陣列（力/力矩、夾爪、關節、pose、velocity）
├── timestamps/                     17 個 float64 陣列，Unix 秒；相機、events、每個 robot_state 各一個；沒有音訊
└── segments_info/<i>/              start, end（Unix 秒）, success（bool）, text（"Insert Ethernet."）, index（int64）
    └── low_level/<j>/              start, end, success, text（定長 bytes："Approach"、"Align"…）
```

- **時間**：segment 起訖與所有時間戳都是 Unix 秒；segment 100% 落在 `/timestamps/hand` 範圍內。
- **影格對時間**：4 台相機 × 3 檔，影片幀數都等於對應時間戳筆數，推測第 i 幀對應第 i 個時間戳。
- **不能用 frame_idx/30**：mp4 標稱 30 fps，但 hama1/hama2 時間戳中位頻率約 31.0 Hz、hand 約 30.0 Hz 且有缺口。
- **高層動作**：pick / insert / remove / place，每檔頭尾各一段 "No action."。
- **低層技能**：approach、align、grasp、lift、push、pull、twist、release。
- **失敗原因、證據時間、best_camera**：資料中沒有（符合規格預期）→ 第 4 包人工標註（原寫「第 3 包」為筆誤，2026-10-05 更正）。

### 與官方 README 不一致的地方（所以不能照文件寫死 key）

| 項目 | 官方 README | 實際檔案 |
|---|---|---|
| 事件相機影片 key | `capture_node` | `capture_node-camera-image` |
| 低階分段群組 | `Low_level`（樹狀圖） | `low_level` |
| segment 欄位 | start / end / success / text | 另有 `index` |
| robot_state | 10 個 | 12 個（多 gripper_efforts、gripper_velocities） |
| measured_torque / velocity 維度 | ×7 / ×7 | ×3 / ×6 |
| 音訊 | mp3 位元組字串 | int64 陣列逐位元組存放、沒有時間戳 |
| text 格式 | — | 高層有句號（"Insert USB."）；高層 vlen 字串、低層定長 bytes |

## 5. 需要你核對的項目

1. **5 段 segment 人工核對**：每個檔的 `questions.md` 第 2 節，對照影片確認起訖與 success（規格 P1 必要測試）。
2. `success=False` 是否就代表該動作失敗？有沒有「無法判定」的標記？
3. text 解析規則：第一個詞＝action，其餘去掉句號＝零件（例："Insert Ethernet." → insert / Ethernet）。
4. "No action." 段是否排除。
5. 相機角色：hama1、hama2＝外部相機；hand＝腕部；capture_node-camera-image＝事件相機影像（DAVIS346，解析度 346x260 吻合）。
6. best_camera 候選要不要包含事件相機。
7. low_level 區間要不要當證據時間候選。

## 6. 第 2 包 indexer 欄位映射提案

見 `configs/field_mapping.proposed.yaml`。核對後另存為 `configs/field_mapping.yaml` 並把 `status` 改成 `confirmed`。

## 7. 尚未完成與下一包依賴

- 尚未實作：indexer / validation（P1-3、P1-4）、Insert 數量報表、影格擷取與預覽、標註、切分、VLM、評估。
- **需要你決定**：選定 Insert 零件需要全部 149 檔的 segment 統計。zip 是 deflate 壓縮，讀任何一檔的 segments_info 都得把該檔整個下載解壓。全量約 54.8 GB；以這次實測約 2.2 MB/s 估計要 7 小時左右。可以邊下載邊建索引、不保留原檔（只留索引），之後只重新下載選中零件的檔。也可以先用部分檔案，或改用官方 README 提到的 Google Drive 來源。
- 下一包依賴：核對後的 `configs/field_mapping.yaml`、上述下載策略的決定。

## 8. 授權與引用

REASSEMBLE：Sliwowski et al., 2025, arXiv:2502.05086；TU Wien Research Data，CC BY 4.0。原始檔放在 `data/raw/`，已列入 `.gitignore`。
