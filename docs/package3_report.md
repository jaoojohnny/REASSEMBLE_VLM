# 第 3 包交付報告：影格擷取、時間同步、40 段 pilot 預覽

日期：2026-10-05 ｜ 專案：`C:\Users\user\Desktop\REASSEMBLE_VLM` ｜ 研究零件：Insert BNC

## 1. 執行結果

| 項目 | 結果 |
|---|---|
| 執行 | `extract_pilot.bat`：2026-10-05 14:45:05 → 14:45:31（26 秒，4 個 worker），rc=0 |
| 測試 | Windows：pytest 55 passed（14.6 秒）；雲端：55 passed |
| pilot 段落 | 40 段＝失敗 20 + 成功 20，來自 32 個錄製檔（其中 8 個檔各 2 段，都是同一檔裡先失敗、後成功） |
| 取樣設定 | 3 組：均勻取 1 / 4 / 8 幀；視窗 `[標註起點 − 0.5 s, 標註終點 + 2.0 s]`；JPEG 品質 95；相機 hama1、hama2、hand |
| 輸出量 | 影格 1,560 張（120 + 480 + 960，原解析度 640×480）、contact sheet 120 張、meta.json 120 份，共 111 MB |
| 缺幀（MISSING） | 0 |
| 時間差過大（FAR，> 0.25 s） | 6 張＝2 段 × 3 台相機，都在最後一欄（原因見 §4） |
| 預覽頁 | `data/derived/frames/pilot_index.html`（自動轉到 n=8 那一組；#1–#20 失敗、#21–#40 成功） |

輸出位置：`data/derived/frames/insert_BNC/<設定>/`，設定資料夾為 `uniform_n1_sb0.5_eb2_jpg`、`uniform_n4_…`、`uniform_n8_…`；
抽樣清單 `data/derived/frames/insert_BNC/selection_pilot.csv`。範例輸出（文字檔）在 `docs/examples/package3/`。

## 2. 規格驗收對照（P2-1～P2-5）

| ID | 需求 | 驗收條件 | 做法與證據 |
|---|---|---|---|
| P2-1 | 由 segment ID 定位原始檔，依真實時間戳擷取 start 至 end，可設前後緩衝 | 影格時間落在所標示範圍 | `segments.csv` → `data/slim/<錄製檔>.h5`；視窗 `[start − 0.5, end + 2.0]`（`--start-buffer`／`--end-buffer`，預設讀 `field_mapping.yaml` 的 `extraction`）。真實資料 1,560 張影格的實際時間全部落在視窗內；測試 `test_extract_end_to_end`、`test_pick_frames_window_and_missing` |
| P2-2 | 1、4、8 幀均勻取樣；保留首尾與每張原始秒數 | 同 seed 多次執行得到同樣 frame ID | 目標點含視窗首尾（n=1 取視窗最後＝終點 + 2 秒的最後狀態），每個目標取時間戳最接近的實際幀；`samples.csv`、`meta.json` 記目標秒數、實際秒數、Unix 時間與 frame_idx。測試：強制重做後 frame ID 相同；pilot 抽樣同 seed 結果相同 |
| P2-3 | 每個視角獨立輸出；可產生同時間多視角預覽 | 檔名與圖上標記 camera、timestamp | 檔名如 `hama1__03__t+016.331s__f006266.jpg`（相機、序號、相對秒數、幀號）；`sheet.jpg` 每列一台相機、每欄同一目標時間，格內左上標相機與幀號、左下標實際秒數 |
| P2-4 | 輸出 PNG/JPEG 影格、contact sheet、preview metadata JSON；缺幀報錯或標記 | 至少 20 成功 + 20 失敗可人工檢視 | `--format jpg` 或 `png`；`meta.json` 每台相機含規格範例的 `segment_id`、`camera`、`source_times_sec`、`frame_paths`、`sampling`、`n_frames`；缺幀時該格畫成 MISSING 並記原因（`no_frame_in_window`／`camera_missing`／`decode_failed`）；pilot 40 段＝20 + 20 |
| P2-5 | 以來源版本和取樣設定作快取鍵；不覆蓋原檔 | 第二次同設定不重做，換設定可重建 | 快取鍵＝精簡版檔案指紋（大小 + 修改時間）+ 取樣設定 + 相機；換設定會另開資料夾；只讀 `data/slim/`，不寫回。測試：第二次全部「沿用」、換設定另建 |

## 3. 時間同步精度

每張影格的秒數都來自該相機自己的時間戳（hama 約 31 Hz，不是 `frame_idx / 30`）。

| 設定 | 影格數 | \|實際 − 目標\| 中位 | p95 | 最大（不含 FAR） |
|---|---|---|---|---|
| n=1 | 120 | 0.016 s | 0.036 s | 0.087 s |
| n=4 | 480 | 0.012 s | 0.031 s | 0.092 s |
| n=8 | 960 | 0.010 s | 0.030 s | 0.092 s |

- 最大值 0.09 s 來自腕部相機偶爾較長的幀間隔，遠低於 FAR 門檻 0.25 s。
- 同一欄三台相機之間的實際時間差：中位 0.019 s、p95 0.044 s、最大 0.077 s。所以同一欄可以視為同一時刻的多視角畫面。

## 4. FAR 的 2 段：錄影在標註結束後不到 2 秒就停了

| # | 錄製檔 | segment | 標註 | 標註結束後還有幾秒影像 | 最後一欄實際取到 |
|---|---|---|---|---|---|
| 9 | 2025-01-11-14-35-33 | 17 | 失敗 | 0.97 s | 終點 + 0.97 s（目標是 + 2 s） |
| 20 | 2025-01-13-19-04-46 | 5 | 失敗 | 0.58 s | 終點 + 0.58 s |

兩段都是該錄製檔最後一次 Insert，後面只接一小段 "No action."。影格本來就不存在，不是程式錯誤。
預覽圖上這一格左下的實際秒數會以黃字加 `FAR` 標示，`samples.csv`／`meta.json` 也有 `far: true`。
目前做法：最後一欄用錄影的最後一幀。第 5 包把影格交給 VLM 時，會一併附上這個標記。

## 5. 初步觀察（只看了 #3、#20、#21 三段的縮圖與 2 段原尺寸影格，僅供參考，以使用者核對為準）

- **腕部相機（hand，第三列）最能看到接觸區。** 夾爪上的 BNC 接頭全程清楚。最後一欄（終點 + 2 秒）：
  成功段 #21 是夾爪張開、接頭已插在插座上（俯視清楚）；失敗段 #3 則是接頭仍夾在夾爪裡。
- **外部相機（hama1、hama2，第一、二列）主要看得到手臂位置、是否在板子上方。** 在 640×480 原圖中插入點很小，常被夾爪擋住。
- 對第 5 包的意義：腕部相機在最後狀態的畫面可能是最強的證據，外部相機提供脈絡。這要等 40 段都看過才能確定。

## 6. 停止點：使用者核對 40 段 pilot 預覽（**已完成：40 段全部可以**，見 docs/manual_check_package3.md）

規格品質檢查：「讓 Claude 製作 40 段 pilot 預覽後先停下，使用者需確認接觸區是否可見、時間是否對齊、影片是否完整。」

每一段檢查了三件事（2026-10-05 使用者回覆「全都可以」）：

1. **接觸區看得到**：至少有一個視角（通常是第三列腕部相機）看得到 BNC 接頭和插座，最後一欄看得出插上了沒有。
2. **時間對齊**：第一欄（起點前 0.5 秒）手臂還沒開始插；綠色欄位是靠近、對準、插入的過程；最後一欄看得到結果。
   畫面要和標註一致：標「成功」的看起來插好了，標「失敗」的看起來沒插好。
3. **畫面完整**：沒有暗紅色 MISSING 格。黃字 FAR 只會出現在 #9、#20 的最後一欄（原因見 §4）。

## 7. 本包改動與命令

- 新增：`src/assembly_monitor/extract.py`、`scripts/extract_frames.py`、`tests/test_extract.py`、`extract_pilot.bat`、
  `docs/package3_report.md`、`docs/manual_check_package3.md`、`docs/examples/package3/`
- 修改：`src/assembly_monitor/cli.py`（`extract-frames` 入口）、`pyproject.toml`（console script）、
  `src/assembly_monitor/check_clips.py`（影片上的字改成黑底白字，比較清楚）、`README.md`（第 3 包說明、預覽怎麼看、各包狀態表）
- 更正：第 1 包寫的「第 3 包人工標註」應為**第 4 包**（規格的人工標註在第 4 包）。已改 `configs/field_mapping*.yaml` 註解、
  `report.py`／`candidates.py` 產生的問題文字、`README.md`、`docs/package1_report.md`
- 命令：

```bat
extract_pilot.bat                                         :: 雙擊：自我測試 → pilot 擷取 → 打開預覽頁
python -m scripts.extract_frames                          :: pilot（失敗 20 + 成功 20，1/4/8 幀）
python -m scripts.extract_frames --set all                :: BNC 全部有效段
python -m scripts.extract_frames --n-frames 8 --end-buffer 3   :: 換設定（另開資料夾，不覆蓋）
python -m scripts.extract_frames --format png --cameras hama1,hand --force
```

- 測試：55 項（第 3 包新增 `tests/test_extract.py` 5 項：目標時間含首尾、視窗內取幀與缺幀標記、pilot 抽樣同 seed 同結果且分散、
  合成檔端到端（時間落在視窗、檔名含相機與時間、缺視角標 `camera_missing`、快取不重做、強制重做 frame ID 相同、換設定另建資料夾）、pilot 模式與預覽頁）

## 8. 需要使用者核對或決定

| 項目 | 預設 | 需要 |
|---|---|---|
| 40 段 pilot 預覽（停止點） | — | **已完成**：40 段全部可以 |
| #9、#20 最後一欄只到終點後 0.97 / 0.58 秒 | 用錄影最後一幀，標 FAR | **已接受**（維持預設） |
| 取樣設定是否沿用到第 5 包 | 1 / 4 / 8 幀、緩衝 0.5 / 2.0 秒 | 可等第 5 包小樣本測試後再定 |

## 9. 尚未完成與下一包依賴

- **第 4 包**：標註 schema（result＝success / failure / uncertain；失敗原因類別依 pilot 修訂，至少允許 unknown）、CSV／JSONL 匯入匯出
  （保留標註者、版本、審核狀態）、HTML 預覽加標註表（多視角與秒數）、雙人標註一致率與爭議清單（30 段 pilot）、
  以錄製檔為單位的 split／leakage audit。停止點是 pilot 標註與無洩漏報告。
- **依賴**：本包核對已通過（2026-10-05）；切分比例（建議開發集 20–30%／測試集 70–80%，以錄製檔分組，§1 那 8 個檔的兩段必須在同一集合）；
  雙人標註需要第二位標註者（或決定替代做法）。
