# 資料字典

所有路徑相對專案資料夾。秒數若標「相對」，都是相對資料集標註的起點（`start_unix`）。

## 1. 索引（第 2 包，`data/index/`）

| 檔案 | 主要欄位 |
|---|---|
| `segments.csv` | `recording_id`、`segment_id`（錄製檔內的編號）、`text`、`action`、`part`、`start_unix`／`end_unix`（Unix 秒）、`start_sec`／`end_sec`（相對錄製檔第一個相機時間戳）、`duration_s`、`success`、`prev_gap_s`／`next_gap_s`、`frames_<相機>`、`n_skills`、`validation_status`（ok／warn／error／excluded）、`error_code` |
| `skills.csv` | 低階技能：`recording_id`、`segment_id`、`skill_id`、`text`（Approach／Align／Twist／Release…）、`start_unix`、`end_unix`、`success` |
| `recordings/<錄製檔>.json` | 單一錄製檔完整索引：各相機時間範圍（`t_first`／`t_last`）、來源檔 CRC-32／SHA-256、各段與技能 |

段的全域代號（之後各包共用）：`segment_id`＝`<錄製檔>__seg<編號>`，例如 `2025-01-09-14-28-47__seg11`。

## 2. 影格（第 3 包，`data/derived/frames/insert_BNC/`）

| 檔案 | 內容 |
|---|---|
| `selection_pilot.csv` | 第 3 包已核對的 40 段 |
| `<設定>/segments/<段>/meta.json` | 每台相機的 `segment_id`、`camera`、`source_times_sec`（相對秒）、`frame_paths`、`sampling`、`n_frames`、每張影格的 `frame_idx`、`delta_s`、`far`、`missing` |
| `<設定>/samples.csv` | 每張影格一列 |

## 3. 切分（第 4 包，`data/splits/<切分名稱>/`，預設 `bnc_dev25_test75_s0`）

| 檔案 | 內容 |
|---|---|
| `split.csv` | `segment_id`、`recording_id`、`segment_index`、`group`（分組依據的值，預設＝錄製檔）、`split`（dev／test）、`dataset_result`、`duration_s`、`date`、`start_unix`、`end_unix` |
| `split_summary.md`／`.json` | 各集合的 group 數、段數、失敗／成功、依日期分布；有 gold 時加失敗原因分布 |
| `leakage_report.md`／`.json` | 洩漏檢查：每段只出現一次、group 不跨集合、同錄製檔不同集合的段視窗不重疊、不同集合的錄製檔時間不重疊、來源檔雜湊不重複、衍生檔歸屬、影格位元組完全相同（任何一張就算未通過）、近似影格（dHash 漢明距離 ≤ 4，以「段配對 × 相機」彙總：同一對段每台相機都有 ≥ 50% 影格近似才算疑似重複序列＝警告；只有背景固定的外部相機相似不算。合成圖 sheet.jpg、review/ 不比近似）。報告最後列出近似最多的 5 組段配對與各相機覆蓋率，供人工確認 |
| `derived_manifest.csv` | 每個衍生檔（各相機、1/4/8 幀、標註影格、檢視圖）屬於哪一段、哪個集合 |
| `manifest.json` | 設定（比例、seed、group key、分層）、各集合數量、`lock.segments_sha256`、`lock.per_split.<集合>.sha256`、`split_csv_sha256`、`locked: true` |

開發階段只用 `dev`。程式裡用 `split.load_split(dir)` 只會拿到 dev；要拿 test 必須 `allow_test=True`，而且會先驗證雜湊。

## 4. 標註（第 4 包，`data/annotations/<任務>/`，預設 `pilot30`）

| 檔案 | 內容 |
|---|---|
| `human__<標註者>.jsonl` | 使用者的標註（標註頁匯出，`compare_pilot.bat` 收進來並合併） |
| `ai__claude.jsonl` | Claude 的獨立標註 |
| `reference__dataset_lowlevel.jsonl` | 由資料集低階技能推得的參考標籤 |
| `adjudication__<標註者>.jsonl` | 裁決頁匯出的最後答案 |
| `gold.jsonl`／`gold.csv` | 合併後的標準答案：`review_status`＝approved（無爭議）／adjudicated（已裁決）／disputed（尚未裁決，不算正式答案）。正式答案一律標成目前版本（v2），原版本在 `annotated_label_version`／`annotated_reason_set`；無爭議段的畫面難度預設 clear，並記 `visual_difficulty_source: default_for_approved`；裁決頁匯出的無爭議段沒填難度時也一樣（有主標註者的難度就沿用，記 `primary_annotation`）。爭議段的難度一定由裁決者選 |
| `inbox/` | 收進來的原始匯出檔（保留歷史） |
| `reports/agreement.md`／`.html`／`.json` | 各來源兩兩比較：結果、失敗原因、最佳視角的一致率與 κ（含 95% bootstrap 信賴區間）、證據區間 IoU 與命中率、可見度一致率 |
| `reports/disputes.csv` | 需要裁決的段：`disputes`＝需要裁決的欄位（result、failure_reason、best_camera，或 evidence_length＝證據區間短於 0.2 秒）；`differences`＝所有不同的欄位（含只列入統計的 evidence）；另附參考標籤的原因與區間 |
| `reports/gold_summary.json`、`needs_adjudication.flag` | 合併結果（同意／裁決／未裁決段數、版本、結果與失敗原因分布、畫面難度分布、預設難度的段）；還有未裁決的段時會有 flag 檔 |

### 標註紀錄欄位（JSONL 每行一筆；CSV 把 `camera_visibility` 展開成 `vis_<相機>` 欄）

| 欄位 | 型別 | 允許值 | 何時必填 |
|---|---|---|---|
| `segment_id` | 字串 | `<錄製檔>__seg<編號>` | 一律 |
| `recording_id`、`segment_index`、`part` | 字串／整數／字串 | — | 標註頁自動帶入 |
| `dataset_result` | 字串 | success／failure | 自動帶入；**不可修改**（匯入時與索引不符會報錯） |
| `result` | 字串 | success／failure／uncertain | 一律 |
| `failure_reason` | 字串或 null | `configs/annotation.yaml` 的類別：misalignment、incomplete_insert、slip_or_drop、board_lifted（v2）、wrong_contact、other、unknown | result＝failure（草稿除外）；其他結果必須是 null |
| `evidence_start_sec`、`evidence_end_sec` | 數字或 null | 相對秒，起 ≤ 迄，落在 [−0.5, 段長 + 2]；v2 長度至少 0.2 秒 | result＝success／failure 且由人或 AI 標註（草稿除外） |
| `best_camera` | 字串或 null | hama1／hama2／hand | 同上 |
| `visual_difficulty` | 字串或 null | clear／subtle／not_visible（結果在畫面上有多容易看出來） | v2：目前版本的人工／AI 正式紀錄必填 |
| `camera_visibility` | 物件 | 每台相機：clear／partial／occluded／null | 選填 |
| `visible_evidence` | 字串 | — | 同證據區間 |
| `notes` | 字串 | — | 選 other 時必填 |
| `annotator` | 字串 | — | 一律 |
| `annotator_type` | 字串 | human／ai／reference | 預設 human |
| `label_version`、`reason_set` | 字串 | 目前 v2、bnc-v2（pilot 原始標註為 v1、bnc-pilot-v1） | 一律／選填 |
| `review_status` | 字串 | draft／submitted／approved／disputed／adjudicated／rejected | 一律 |
| `created_at`、`updated_at`、`source` | 字串 | ISO 時間；來源（annotate-page、adjudication-page、rules:lowlevel-v1…） | 選填 |

其他自訂欄位會原樣保留（例如 gold 的 `annotated_label_version`、`visual_difficulty_source`）。JSON Schema：`configs/annotation.schema.json`（`python -m scripts.annotate schema` 產生）；
跨欄位規則（起 ≤ 迄、落在該段視窗內）由 `assembly_monitor.schema.validate_record` 檢查。

## 5. 標註頁（`data/derived/annotation/<任務>/`）

| 檔案 | 內容 |
|---|---|
| `index.html`、`adjudicate.html` | 標註頁、裁決模式（雙擊開啟，不需要網路） |
| `data.js` | 段落清單、各相機影格的相對秒數、低階技能、設定 |
| `compare_data.js` | 比較後才有：各來源答案與爭議，供裁決模式使用 |
| `frames/<段>/<相機>__0000.jpg`、`frames.json` | 每秒 5 張的密集影格（依各相機時間戳挑最近幀）與其秒數、幀號 |
| `review/<段>__overview.jpg`、`<段>__hand.jpg` | 總覽（8 個時間點 × 3 視角）與腕部相機 12 格圖（標相對秒數與當下的低階技能），供 Claude 獨立標註 |
| `selection.csv`、`task.json` | 任務的段落清單（含低階技能序列）與設定 |

## 6. VLM 試跑（第 5 包，`experiments/<實驗名稱>/`，預設 `dev_smoke`）

| 檔案 | 內容 |
|---|---|
| `<模型>/f<幀數>/run_config.json` | 模型設定與套件版本、GPU、`prompt_version`／`prompt_sha256`、`sampling_key`、切分、`config_sha256` |
| `<模型>/f<幀數>/raw.jsonl` | `segment_id`、`model`、`n_frames`（0＝不看影格）、`images`（`i`、`camera`、`time_sec` 相對秒、`source`）、`raw_text`、`repair_text`、`latency_s`、`input_tokens`、`output_tokens`、`peak_vram_mb`、`stopped_by_time`、`error`／`error_detail` |
| `<模型>/f<幀數>/predictions.jsonl` | 規格固定欄位 `result`、`failure_reason`、`evidence_start_sec`、`evidence_end_sec`、`best_camera`、`visible_evidence`、`confidence`；`status`（ok／invalid／error）、`errors`、`warnings`、`attempt`（first／repair）、`first_errors`、`dataset_result`、`prompt_version`、`sampling_key` |
| `<模型>/f<幀數>/errors.csv` | `segment_id`、`model`、`n_frames`、`kind`（invalid／error）、`detail`、`created_at` |
| `summary.csv`／`summary.md` | 每個組合：段數、格式正確、無效、錯誤、無法判斷、格式修正成功、有警告、快速準確率、快速失敗召回、平均／最長秒數、峰值顯存 |

`warnings` 可能的值：`missing_failure_reason`、`bad_failure_reason`、`reason_given_for_non_failure`、`evidence_not_number`、
`evidence_start_after_end`、`evidence_out_of_range`、`evidence_half_missing`、`bad_best_camera`、`bad_confidence`、`stopped_by_time`。
有警告的欄位一律是 null（原值留在 `raw_text`）。
