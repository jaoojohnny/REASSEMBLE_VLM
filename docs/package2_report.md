# 第 2 包交付報告：indexer、validation、候選 Insert 數量報表

日期：2026-10-05 ｜ 專案：`C:\Users\user\Desktop\REASSEMBLE_VLM`

## 1. 全量索引執行結果

| 項目 | 結果 |
|---|---|
| 執行 | `index_all.bat`：2026-10-05 02:10 → 05:47（約 3 小時 37 分），同時 2 個下載 |
| 錄製檔 | **149 / 149 完成，0 失敗**（data/index/recordings/*.json） |
| 下載量 | 54.8 GiB（平均約 4.5 MB/s）；完整原檔處理完即刪除，暫存資料夾已清空 |
| 精簡版 | `data/slim/` 149 檔、共 7.9 GB（每檔 3.7–111 MB，中位 52 MB；只含 4 台相機 mp4、相機時間戳、segments_info） |
| 測試 | 開跑前 Windows 上 pytest 全過；data/raw 快速檢查通過 |

## 2. 與官方統計比對（indexer 正確性）

| | 官方（論文 / README） | 本索引 |
|---|---|---|
| 示範（高層動作段，不含 "No action."） | 4,551 | **4,551**（pick 1,195 + insert 1,165 + remove 1,100 + place 1,091） |
| 成功 | 4,035 | **4,035** |
| 失敗 | 516 | **516** |
| 物件 | 17 | **17**（Insert 的零件種類） |

數字完全吻合，代表欄位映射、文字解析與排除規則都正確。

## 3. 驗證（validation.json）

- segment：ok 4,547、warn 4、error 0、excluded 312（"No action."）
- 錯誤碼：`W_NO_FRAMES:hand` 6（腕部相機在該段沒有畫面）、`W_SHORT` 3（長度 < 0.3 秒）
- 錄製檔層級：`W_KNOWN_ISSUE` 10（官方 README 列出的問題檔）、`W_CAMERA_MISSING:hand` 1（2025-01-10-16-17-40，官方註記 missing hand cam）
- 所有相機幀數 = 時間戳筆數（沒有 `W_FRAMES_TS_MISMATCH`）

## 4. 動作 × 成敗

| 動作 | 段數 | 成功 | 失敗 | 失敗率 |
|---|---|---|---|---|
| pick | 1,195 | 1,095 | 100 | 8% |
| insert | 1,165 | 918 | 247 | 21% |
| remove | 1,100 | 953 | 147 | 13% |
| place | 1,091 | 1,069 | 22 | 2% |

Insert 各零件完整表見 `data/index/action_summary.md` 與 `insert_candidates.csv`。失敗數前五：BNC 30、bolt 4 30、USB 21、square peg 1 20、Ethernet 20。

## 5. 選定零件：Insert BNC（使用者 2026-10-05 決定）

| 項目 | 數值 |
|---|---|
| 段數 / 成功 / 失敗 | 86 / 56 / 30（失敗率 35%） |
| 錄製檔 | 62 個；19 個檔有多次嘗試 |
| 每檔嘗試型態 | 只有成功 40、先敗後成 14、只有失敗 3、敗敗 2、敗敗成 1、敗敗敗成 1、連敗 4 次 1 |
| 依日期（失敗 / 成功） | 01-09：11 / 12；01-10：2 / 10；01-11：3 / 8；01-13：14 / 20；01-14：0 / 6 |
| 長度中位數 | 成功 24.6 s（10.2–59.6）；失敗 19.7 s（3.3–42.4） |
| 失敗後的下一段 | 再次 Insert BNC 12、No action 6、Pick BNC 6、Place BNC 6 |
| 驗證 | ok 85、warn 1（`W_NO_FRAMES:hand`，該段腕部相機無畫面） |
| 與下一段間隔 | 中位 0.017 s（仍是無縫相接 → 取最後狀態沿用 end_buffer 2 秒） |

對後續各包的意義：

- **第 3 包 pilot 預覽**：規格要求至少 20 成功 + 20 失敗可人工檢視，BNC 有 30 失敗、56 成功，足夠。
- **第 4 包切分**：必須以錄製檔為 group（19 個檔有多次嘗試，同檔的段不能跨集合）。這個研究是現成 VLM 零樣本推論、不訓練，70/15/15 會讓 test 只剩約 4–5 個失敗；建議改成「開發集（調 prompt）約 20–30% / 測試集 70–80%」，第 4 包時再請使用者決定。
- `configs/field_mapping.yaml` 的 `target_part` 已設為 BNC。

## 6. 停止點：BNC 索引 5 段人工核對（**已完成：5 段全部正確**，見 docs/manual_check_package2.md）

`check_part.bat` 會從 `segments.csv` 抽下列 5 段（seed 0；失敗 3、成功 2，各來自不同錄製檔），讀精簡版產生短片到 `data/derived/manual_check_part/`：

| # | 錄製檔 | segment | 標註 | 長度 (s) |
|---|---|---|---|---|
| 1 | 2025-01-09-17-23-07 | /segments_info/14 | 失敗 | 36.8 |
| 2 | 2025-01-10-17-25-29 | /segments_info/30 | 失敗 | 23.6 |
| 3 | 2025-01-13-11-23-17 | /segments_info/26 | 失敗 | 9.5 |
| 4 | 2025-01-09-14-28-47 | /segments_info/11 | 成功 | 45.7 |
| 5 | 2025-01-09-17-03-05 | /segments_info/6 | 成功 | 35.2 |

## 7. 本包改動與命令

- 新增：`mapping.py`、`index.py`、`stream.py`、`slim.py`、`configs/field_mapping.yaml`（confirmed）、`index_all.bat`、`check_part.bat`、`tests/test_index.py`、`tests/test_stream.py`
- 修改：`check_clips.py`（從索引抽段）、`fetch.py`（多工下載不顯示進度條）、`cli.py`、`synthetic.py`（合成用映射）
- 命令：`python -m scripts.index_segments --source remote`、`python -m scripts.index_segments --report-only`、`python -m scripts.export_check_clips --from-index --part BNC`
- 測試：雲端 50/50（第 2 包新增 index / stream / 從索引抽段）

## 8. 尚未完成 / 下一包

- 第 3 包：依時間戳擷取 BNC 段落影格（1、4、8 幀均勻取樣，保留首尾與原始秒數）、多視角預覽、40 段 pilot 預覽，讓使用者確認接觸區是否可見。
- 依賴：第 4 包的切分比例決定（5 段人工核對已通過）。
