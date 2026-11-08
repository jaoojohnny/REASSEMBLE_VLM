# REASSEMBLE × VLM 組裝視覺監督

依《REASSEMBLE × VLM 組裝視覺監督：給 Claude 的程式開發規格與驗收清單》（v1.0，2026-10-04）逐包開發。
研究目標：從 REASSEMBLE 選一種 Insert 操作，用現成 VLM 離線判斷 success / failure / uncertain，並輸出失敗原因、證據時間與畫面依據。

**目前進度：第 1～7 包都已完成（2026-10-16）**——第 1～3 包已驗收（選定 **Insert BNC**，86 段、失敗 30）；第 4 包 86 段的人工標準答案（gold）全部完成；
第 5 包開發集實驗（Qwen3-VL、Qwen3.5、Gemma 4、提示詞 p1／p2、推理 R1～R3、最後一張 +2／+1／+0.5 秒、做法 A）已完成；
第 6 包：選定的方法凍結後在測試集（66 段）跑過一次（2026-10-14，`experiments/final_test/report.md`）。
第 7 包：兩種訓練都只用其他零件訓練，測試集已跑（2026-10-16，`experiments/final_test_train/report.md`）：LoRA 微調後一律答成功（65%、兩類平均 50%）；同提示詞原始模型 73%（67%）；改良小分類器 68%（66%）；整體最好仍是第 6 包的小分類器 82%（75%）。結論見 `docs/package7_training.md`。再訓練（VLM 特徵＋小分類器，只看開發集）也沒有比不訓練好（開發集 16/20 vs 18/20）。
**第 8 包（2026-10-25～30）**：新的測試資料 **USB**（70 段、失敗 21）、商業模型 Claude Sonnet 5.5 與 GPT-6.1 Sol（API）、各方法的執行時間、給 Qwen 的 Skill（第一版放大鏡、第二版手腕固定裁切、第三版從商業模型推理整理的檢查清單；都只在 BNC 開發集）。說明與結果見 `docs/package8_usb.md`。

**第 9 包（2026-10-30）**：新的測試資料 **bolt 4**（83 段、失敗 30）：Skill 第三版 62/83，明顯比沒有 Skill 的 Qwen（46/83）好；GPT-6.1 Sol 78/83。說明見 `docs/package9_bolt4.md`。

**第 10 包（2026-10-30）**：兩階段（本地 Qwen＋Skill 第三版先判，沒答成功才用 GPT-6.1 Sol 的答案），在 **Ethernet**（72 段）與 **square peg 1**（73 段）兩個新測試集驗證：兩階段都明顯比只用第三版好；Ethernet 70/72（和只用 GPT 相同，API 費用約 60%），square peg 1 只有 48/73（GPT 47/73，所有方法都沒超過一律答成功 53/73）。說明與結果見 `docs/package10_hybrid.md`；一鍵檔 `run_final_test_p10.bat`。

**第 11 包（2026-10-30 開發階段）**：讓本機 Qwen 判斷更準的四種推理方式（機率門檻、前後對照、答失敗再檢查一次、看範例），在 4 個新零件的開發集（square peg 2、round peg 3、D-SUB、medium gear）挑選；說明見 `docs/package11_reason.md`；一鍵檔 `run_reason_dev.bat`；第二輪（結合 Codex 的 evidence_review 修改 Skill 與提示詞）`run_reason_dev2.bat`；測試集（4 個新零件，2026-10-31 凍結）`run_final_test_p11.bat`：**機率門檻 191/265（72%）vs 第三版 126/265（48%）**，事先登記的主要問題成立；兩類平均 63% vs 57%（差的區間包含 0）；方塊插銷進步大、waterproof 變差，仍低於一律答成功（82%）。之後使用者指定「以資料集標記為主、判斷和標記一樣的段才用」：Claude 盲評 241 段開發段，195 段和標記一致＝乾淨開發集（失敗 25 段；`python -m scripts.clean_dev` → `experiments/clean_dev_p11/report.md`），測試集照原本標記算。

**第 12 包（2026-10-31 開發階段）**：兩階段——本機 Qwen 機率門檻先判，判失敗才問 GPT-6.1 Sol；GPT 先在 4 個開發零件的 241 段 dev 段各跑一次，用乾淨開發集照事先寫好的規則挑方法與門檻；測試零件 round peg 1、large gear（round peg 4、small gear 繼續保留）。說明見 `docs/package12_cascade.md`；一鍵檔 `run_cascade_dev.bat`。開發集：照規則選出**只用 GPT**（乾淨開發集兩類平均 97%、全部 241 段 82%；只用 Qwen 機率門檻 76%／66%；兩階段 Qwen 機率門檻→GPT 90%／75%，只問 39% 的段）。測試集（2026-11-02 凍結）`run_final_test_p12.bat`：round peg 1＋large gear，只用 GPT vs 只用 Qwen 機率門檻：**GPT 109/128（兩類平均 70%）vs Qwen 89/128（53%）**，主要問題成立（27 : 7，p＜0.001）；但 GPT 只抓到 8/16 段失敗、沒有超過一律答成功（112/128）；兩階段 第三版→GPT 68%（問 56% 的段）、門檻→GPT 57%（失敗段只剩 3/16）。

## 快速開始（Windows）

1. 雙擊 **`setup_windows.bat`**。它會依序：
   1. 找到 Anaconda / Miniconda，建立獨立環境 `reassemble-vlm`（Python 3.11，只用 conda-forge）
   2. `pip install` 套件，並把本專案以可編輯模式安裝
   3. 記錄 Python、套件版本與 GPU（`logs/envinfo.txt`）
   4. 跑單元測試，並用真 h5py 對合成檔做一次煙霧測試（結果在 `docs/examples/`）
   5. 若 `data/raw/` 沒有 `.h5`：先列出要下載的 1–3 個 REASSEMBLE 檔與大小，**輸入 `y` 才下載**（只抓這幾個檔，不是整包 54.8 GB）
   6. 對 `data/raw/*.h5` 做結構檢查，結果在 `data/index/inspect/`
2. 執行紀錄都在 `logs/`：`setup_status.txt`（每步結果）、`setup_windows.log`、`pytest.txt`、`fetch_samples.log`、`inspect.log`。
3. 之後只想重跑檢查：雙擊 **`run_inspect.bat`**（可加參數，例如 `run_inspect.bat --expect-cameras hama1,hama2,hand`）。
4. 人工核對 5 段 segment：雙擊 **`check_segments.bat`**，會在 `data/derived/manual_check/` 產生多視角對齊短片（綠框＝標註區間）、五格縮圖與 `checklist.md`。

`setup_windows.bat` 可以重複執行：已存在的環境與已下載的檔案會沿用。

### 第 2 包：全量索引（過夜跑）

雙擊 **`index_all.bat`**：先跑自我測試與 `data/raw` 的快速檢查，確認後才開始。它會逐一下載 149 個錄製檔（共約 54.8 GB），
依 `configs/field_mapping.yaml` 建索引，並把「影片 + 相機時間戳 + 標註」存成精簡版到 `data/slim/`（約 8 GB）。
完整原檔處理完就刪除。執行期間電腦不會睡眠；中途關掉視窗也沒關係，再雙擊一次會從沒做完的檔繼續，失敗的檔會自動重試。

輸出在 `data/index/`：

| 檔案 | 內容 |
|---|---|
| `segments.csv` | 每個高層 segment 一列：recording_id、segment_id、action、part、起訖（Unix 秒與相對秒）、success、各相機幀數、前後間隔、validation_status、error_code |
| `skills.csv` | 低階技能（low_level）每個一列 |
| `recordings.csv` | 每個錄製檔的相機幀數、時間戳筆數、頻率與問題 |
| `validation.json` | 錯誤碼統計與每個有問題的 segment（含來源檔） |
| `insert_candidates.csv` / `action_summary.md` | 各動作成敗統計；Insert 各零件的段數、失敗數、錄製檔數、test 預估失敗數 |
| `recordings/<id>.json` | 單一錄製檔的完整索引（續跑依據） |
| `stream_progress.json` | 全量索引進度與預估剩餘時間 |

選定零件後，雙擊 **`check_part.bat`**：從 `segments.csv` 抽 5 段（約一半失敗、各來自不同錄製檔、只取驗證 ok 的段），
讀 `data/slim/` 精簡版產生多視角短片到 `data/derived/manual_check_part/`。

### 第 3 包：影格擷取與 pilot 預覽

雙擊 **`extract_pilot.bat`**：先跑自我測試，再對 BNC 取 40 段（失敗 20、成功 20，各段儘量來自不同錄製檔），
依各相機時間戳在 `[標註起點 − 0.5 秒, 標註終點 + 2 秒]` 內均勻取 1 / 4 / 8 幀（含首尾；1 幀＝視窗最後一幀，也就是最後狀態），
輸出到 `data/derived/frames/insert_BNC/<取樣設定>/`，並在瀏覽器打開預覽頁 `data/derived/frames/pilot_index.html`。

| 輸出 | 內容 |
|---|---|
| `segments/<錄製檔>__seg<id>/` | 各相機影格（檔名含相機、序號、相對秒數、幀號）、`sheet.jpg`（多視角 contact sheet）、`meta.json` |
| `samples.csv` | 每張影格一列：目標秒數、實際秒數、Unix 時間、幀號、時間差、缺幀原因 |
| `manifest.json` / `index.html` | 設定與統計 / 預覽頁（失敗在前，每段附檢查勾選欄） |

快取鍵＝精簡版檔案指紋 + 取樣設定 + 相機：同設定再跑會直接沿用；換設定（例如 `--end-buffer 3`、`--n-frames 16`）會另開資料夾。
全部 BNC 段：`python -m scripts.extract_frames --set all`。

**預覽怎麼看**：每段一張 `sheet.jpg`，三列依序是 hama1、hama2（外部）與 hand（腕部），每欄是同一個目標時間。
標頭綠色＝標註區間內，灰色＝前後緩衝（第一欄是起點前 0.5 秒，最後一欄是終點後 2 秒，用來看最後狀態）；
每格左上是相機與幀號，左下是該幀的實際秒數。暗紅色格子寫 `MISSING` 表示視窗內沒有影格；左下黃字 `FAR` 表示最近的影格和目標差超過 0.25 秒
（pilot 中只有 2 段：錄影在標註結束後不到 2 秒就停了）。要看原尺寸 640×480 的單張影格，打開該段的 `segments\<錄製檔>__seg<id>\` 資料夾。

錯誤碼：`E_START_GE_END`、`E_MISSING_TIME`、`E_SUCCESS_NOT_BOOL`、`E_DUP_ID`、`E_OUT_OF_RANGE`、`E_NO_FRAMES` 會讓該段成為 error；
`W_NO_FRAMES:<相機>`、`W_SHORT`、`W_TEXT_UNPARSED`、`W_SKILL_OUTSIDE`，以及錄製檔層級的 `W_CAMERA_MISSING:<相機>`、
`W_FRAMES_TS_MISMATCH:<相機>`、`W_TS_NON_MONOTONIC:<相機>`、`W_KNOWN_ISSUE` 為 warn。被排除的段（"No action."）標 excluded。

### 第 4 包：標註、切分與洩漏檢查

1. 雙擊 **`package4_prepare.bat`**：安裝新套件（jsonschema）→ 自我測試 → 產生 pilot30 標註頁（失敗 20 + 成功 10，
   三視角每秒 5 張）→ 資料集參考標籤 → 切分（以錄製檔分組，開發 25%／測試 75%，seed 0）與洩漏檢查 → 打開標註頁。
2. 依 **`docs/annotation_manual.md`** 標完 30 段，按「匯出 JSONL」（存到下載資料夾）。
3. 雙擊 **`compare_pilot.bat`**：收檔 → 依 `configs/annotation.yaml` 更新標註頁 → 和 Claude（AI 第二位標註者）、資料集參考標籤比較 →
   打開比較報告；有爭議（結果、失敗原因、最佳視角不同，或證據區間短於 0.2 秒）時打開裁決頁。
   裁決時每段都要選畫面難度；裁決完再匯出、再雙擊一次，就會產生 `data/annotations/pilot30/gold.jsonl`，並更新切分摘要（含失敗原因分布）。

**其餘 56 段**（不在 pilot30 的段：測試 41、開發 15；同樣的 v2 規則和標註頁，同樣由你和 Claude 各自標一份再比較）：

1. 雙擊 **`annotate_rest.bat`**：自我測試 → 產生 `rest56` 標註頁（密集影格、Claude 用的檢視圖）→ 資料集參考標籤 → 打開標註頁。
   可以分幾次標，草稿留在瀏覽器裡；標測試段時看到的東西不要拿來改 VLM 的提示詞。
2. 標完按「匯出 JSONL」，雙擊 **`compare_rest.bat`**：收檔 → 和 Claude、資料集參考標籤比較 → 有爭議時打開裁決頁 →
   裁決完匯出、再雙擊一次，產生 `data/annotations/rest56/gold.jsonl`。

| 輸出 | 內容 |
|---|---|
| `data/splits/bnc_dev25_test75_s0/` | `split.csv`、`split_summary.md`、`leakage_report.md`、`derived_manifest.csv`、`manifest.json`（test 清單雜湊鎖定）。近似影格以段配對彙總，只有「每台相機都相似」才警告 |
| `data/derived/annotation/pilot30/` | 標註頁 `index.html`、裁決頁 `adjudicate.html`、密集影格、Claude 用的檢視圖 |
| `data/annotations/pilot30/` | 各來源標註、`gold.jsonl`、`reports/agreement.html`、`reports/disputes.csv` |

欄位定義見 `docs/data_dictionary.md`。標註格式在 `configs/annotation.yaml`（失敗原因類別、可見度、比較門檻），JSON Schema 在 `configs/annotation.schema.json`。

### 第 5 包：VLM 試跑（開發集）

1. 雙擊 **`setup_vlm.bat`**：顯示 GPU 驅動 → 安裝 PyTorch（CUDA）與 transformers、bitsandbytes → 檢查 GPU →
   列出模型的大小、授權、是否已下載，**按 Y 才下載**（存到 `models\hf`；已下載的不會重抓，全部約 63 GB）。
2. 雙擊 **`run_vlm_dev.bat`**：自我測試 → 擷取開發集 20 段的 1/4/8 幀 → 每個模型先跑 1 段（8 幀）快速檢查 →
   多數類、SmolVLM2、Qwen3-VL-4B／8B、Qwen3.5-4B／9B、Gemma 4 E4B／12B × 1/4/8 幀跑完 20 段（可中斷續跑；已跑完的模型不會重新載入）→
   提示詞第 2 版以後的設定檔（`configs\experiment_p*.yaml`）→
   做法 A（有訓練的對照組）：用其他零件的插入片段訓練小分類器（DINOv2 特徵＋邏輯迴歸），也在開發集評分 →
   打開 `experiments\prompt_versions.md`（各提示詞版本並排比較）。已跑完的部分都會略過。

**下週實驗（事先登記）**：雙擊 **`run_vlm_reasoning.bat`**：最後一張改成段結束後 1／0.5 秒、分步驟推理、模型內建思考模式，
以及做法 A 的 1／0.5 秒版本，都只在開發集；結果併進 `experiments\prompt_versions.md`。

測試集保持鎖定（要 `--unlock-test` 才讀）。說明見 `docs/package5_vlm.md`。

### 第 6 包：評分與測試集最終評估

1. 兩個 run_vlm 檔最後會自動跑 `python -m scripts.evaluate` → `experiments\dev_evaluation\report.md`：開發集所有組合的任務判斷成功率、抓到失敗、成功被誤判、兩類平均、95% 信賴區間（依錄影重抽樣）。
2. 選定方法後寫進 `configs\final_test.yaml` 並設 `frozen: true`，雙擊 **`run_final_test.bat`**：測試集只跑一次 → `experiments\final_test\report.md`（含 McNemar 比較、依畫面難度分組、失敗原因與證據命中）。凍結後設定一改就拒絕執行。說明見 `docs/package6_evaluation.md`。

### 第 7 包：訓練（只用其他零件，BNC 開發集只看、不挑）

1. 雙擊 **`run_probe_v2.bat`**：先列出兩個新影像編碼器的大小、問要不要下載（約 2.7 GB），再跑所有候選（編碼器 × 最後一張 × 幀數 × 時間彙整），
   依訓練池的交叉驗證挑一個 → `experiments\probe_selection\selection.md`、`configs\probe_v2_selected.yaml`。
2. 雙擊 **`run_finetune.bat`**：安裝 peft → 試跑 6 段（顯存、速度、估計時間）→ 問要不要開始 → 正式訓練約 1–2 小時 →
   開發集上比較「微調後」和「同提示詞的原始模型」→ `experiments\dev_evaluation\report.md`。說明與事先登記的設定見 `docs/package7_training.md`。
3. Claude 看完訓練結果、把 `configs\final_test_train.yaml` 設成 `frozen: true` 之後，雙擊 **`run_final_test_train.bat`**：測試集只跑這一次
   （LoRA、同提示詞原始模型、第 6 包主方法與做法 A、改良做法 A、一律答成功）→ `experiments\final_test_train\report.md`。第 6 包的結果不會被覆蓋。

### 再訓練：VLM 特徵＋小分類器（只看開發集）

雙擊 **`run_vlm_probe.bat`**：Qwen3-VL-8B 不改權重，讀它看完畫面後的內部特徵，用其他零件訓練邏輯迴歸，依交叉驗證挑特徵 →
`experiments\vlm_probe_selection\selection.md`。不讀測試集。說明見 `docs/package7_training.md` 最後一節。

### 第 8 包：USB 新測試集、商業模型、執行時間、Skill

1. 雙擊 **`setup_api.bat`**：安裝 anthropic、openai SDK；建立 `.env` 並用記事本打開，貼上 API key（ANTHROPIC_API_KEY、OPENAI_API_KEY）；每個模型送一句測試。
   key 只放在這台電腦的 `.env`（不進 Git、程式不印出）。ChatGPT Plus／Claude Pro 訂閱不含 API。
2. 雙擊 **`run_p8_dev.bat`**：BNC 86 段（開發集）跑 Qwen3-VL-8B、Qwen＋放大鏡 Skill、Claude、GPT → `experiments\p8_dev_evaluation\report.md`；
   第 6、7 包各方法的執行時間 → `experiments\timing\timing.md`。
3. 雙擊 **`run_final_test_usb.bat`**：USB 測試集（凍結，只跑一次）→ `experiments\final_test_usb\report.md`（成功率＋執行時間＋API 費用）。

## 手動指令（Anaconda Prompt，先 `cd` 到專案資料夾）

```bat
conda activate reassemble-vlm

python -m scripts.fetch_samples --list                 :: 只列 zip 內容，存 data\index\zip_listing.csv
python -m scripts.fetch_samples --n 2                  :: 取 2 個（預設 spread：依錄製時間取頭/中/尾）
python -m scripts.fetch_samples --members 2025-01-09-13-59-54.h5
python -m scripts.fetch_samples --zip-path D:\data.zip  :: 已用瀏覽器下載整包時，只解出選中的檔

python -m scripts.inspect_h5 data\raw                  :: 檢查資料夾內所有 .h5
python -m scripts.inspect_h5 data\raw\xxx.h5 --expect-cameras hama1,hama2,hand

python -m scripts.export_check_clips --n 5 --pad 1.5  :: 人工核對短片（data\derived\manual_check）
python -m scripts.index_segments                       :: 第 2 包：索引 data\raw
python -m scripts.index_segments --source remote       :: 第 2 包：全部 149 檔邊下載邊索引（可續跑）
python -m scripts.index_segments --report-only         :: 只重建 CSV 與報表
python -m scripts.export_check_clips --from-index --part BNC --n 5   :: 第 2 包人工核對短片
python -m scripts.extract_frames                       :: 第 3 包 pilot（失敗 20 + 成功 20，1/4/8 幀）
python -m scripts.extract_frames --set all             :: 第 3 包：BNC 全部有效段
python -m scripts.annotate build                       :: 第 4 包：pilot30 標註頁（密集影格）
python -m scripts.annotate reference                   :: 第 4 包：資料集低階技能參考標籤
python -m scripts.annotate collect                     :: 第 4 包：從下載資料夾收標註頁匯出的檔
python -m scripts.annotate compare                     :: 第 4 包：一致率、爭議清單、裁決頁資料
python -m scripts.annotate merge                       :: 第 4 包：依裁決合併 gold
python -m scripts.annotate refresh                     :: 改了 configs/annotation.yaml 後只更新標註頁（不重抽影格）
python -m scripts.annotate validate data\annotations\pilot30\human__A.jsonl
python -m scripts.make_split                           :: 第 4 包：開發 25%／測試 75%＋洩漏檢查
python -m scripts.make_split --verify bnc_dev25_test75_s0   :: 檢查切分是否被改動
python -m scripts.make_split --seq-coverage 0.3          :: 疑似重複序列的門檻（每台相機近似影格比例，預設 0.5）
python -m scripts.extract_frames --set dev             :: 第 5 包：開發集 20 段的 1/4/8 幀
python -m scripts.run_vlm --check                      :: 第 5 包：檢查 GPU 與套件
python -m scripts.run_vlm                              :: 第 5 包：開發集 × 模型 × 1/4/8 幀（可續跑）
python -m scripts.run_vlm --use qwen3vl_8b --frames 8 --limit 3   :: 小樣本
python -m scripts.run_vlm --use mock --limit 2 --name deploy_check  :: 不載入模型，只檢查流程
python -m scripts.download_models --plan               :: 第 5 包：列出模型大小與授權（不下載）
python -m scripts.probe pool                           :: 做法 A：列出訓練池（其他零件、排除 BNC 測試錄影）
python -m scripts.probe all                            :: 做法 A：擷取影格 → 特徵 → 交叉驗證 → 預測 BNC 開發集
python -m scripts.envinfo                              :: 環境與 GPU
python -m scripts.make_synthetic --out data\derived\synthetic   :: 產生合成示範檔
python -m pytest -q
```

安裝後也可以直接用 `inspect-h5`、`fetch-samples`、`am-envinfo`、`export-check-clips`、`index-segments`、`extract-frames` 指令。
所有路徑都由命令列或 `configs/*.yaml` 指定，沒有寫死任何電腦路徑。

## `inspect-h5` 的輸出

每個 HDF5 一個資料夾 `data/index/inspect/<檔名>/`：

| 檔案 | 內容 |
|---|---|
| `tree.txt` | 人可讀的結構樹：group / dataset、shape、dtype、壓縮、儲存大小、少量樣本值、內嵌影片資訊、時間軸統計 |
| `nodes.json` | 每個節點的完整描述（機器可讀，可跨檔比對） |
| `candidates.json` | 推測的 segment 容器與欄位、相機與時間戳配對、時間基準、異常清單、人工核對樣本 |
| `questions.md` | **待確認問題** + 5 段 segment 人工核對表 + 異常摘要 |
| `field_mapping.proposed.yaml` | 第 2 包 indexer 的欄位映射提案（`status: proposed`，核對前不可使用） |

多個檔案時另有 `_summary.md` / `_summary.json`：各檔結構差異（可抓出缺少的視角）、相機出現次數、高層動作分佈、樣本內 Insert 零件的成功 / 失敗數、異常統計。

## 設計原則（對應規格「Claude 必須遵守」）

- **不預設任何 HDF5 key。** segment 容器靠「數字命名子群組 + 子群組裡有起訖純量」的結構找；相機靠內嵌位元組的 mp4 檔頭找；時間軸靠名稱（time / stamp / clock…）或數值範圍（像 Unix 時間）找。全部標為推測，並列進 `questions.md` 讓使用者核對。測試裡有一個把所有名稱都換掉的合成檔，用來確認沒有寫死欄位。
- **大型陣列不整段載入。** 只讀頭尾或左上角；內嵌影片用檔案位移開一個「視窗」交給 PyAV（失敗再用 OpenCV）探測幀數、fps、解析度；時間陣列超過 `max_read_mb` 時抽樣統計。
- **秒數只來自時間戳。** 影片標稱 fps 與時間戳實測頻率不一致、或幀數與時間戳筆數不同時都會列出（不能用 `frame_idx / 30` 推秒數）。
- **不假設失敗原因、證據時間存在。** 找不到時明確列為「第 4 包人工標註」。
- **單檔壞掉不中止整批**，錯誤寫進總表。
- **不先下載整包資料。** `fetch-samples` 用 HTTP Range 只讀 zip 檔尾目錄與選中的檔，逐檔驗證 CRC-32、記錄 SHA-256（`data/raw/manifest.json`）；伺服器不支援 Range 時立即停止並提示改用瀏覽器下載。預設排除官方 README 註記有問題的 10 個錄製檔。
- **原始資料不進 Git**（見 `.gitignore`）。

## 測試

| 檔案 | 涵蓋 |
|---|---|
| `tests/test_inspect_h5.py` | 規格要求的四種合成情境：正常、多視角缺失、非法時間、重複 ID；另加改名情境、大陣列只預覽、輸出檔內容 |
| `tests/test_candidates.py` | 欄位推測規則（不需 h5py） |
| `tests/test_timeaxis.py` | 時間單位推測、單調性、取樣率、缺口 |
| `tests/test_media.py` | 檔頭辨識、檔案視窗、真 mp4 探測、假 mp4 失敗時不中斷 |
| `tests/test_fetch.py` | 本機 HTTP 伺服器：Range 分段下載、zip64、略過已下載、不支援 Range 時拒絕、取消與總量上限 |
| `tests/test_cli.py` | 批次檢查、壞檔不中止、總表；本機 zip 取樣 |
| `tests/test_check_clips.py` | 核對段落挑選（先失敗後成功、分散各檔）、多視角對齊短片與清單 |
| `tests/test_index.py` | 映射必須 confirmed、文字解析規則、四種情境的錯誤碼、segments.csv / validation.json / Insert 報表 |
| `tests/test_stream.py` | 本地 Range 伺服器上的 zip：多工下載、精簡版、刪暫存、沿用本機檔、壞檔隔離、續跑 |
| `tests/test_extract.py` | 影格時間落在視窗、含首尾、同設定同幀、快取不重做、換設定另開資料夾、缺幀標記、pilot 抽樣 |
| `tests/test_schema.py` | 非法類別與時間被拒（Python 驗證與 JSON Schema）、草稿可不完整、CSV／JSONL 往返不遺失備註與原始標籤、缺席欄位保留、Big5 CSV |
| `tests/test_split.py` | 同錄製檔兩段必在同一集合、三相機與 1/4/8 幀衍生檔不跨集合、同 seed 同結果／不同 seed 不同、人造重複影格被抓出、以段分組時視窗重疊被抓出、鎖定與防看 test |
| `tests/test_agreement.py` | κ 與 IoU、30 段合成 pilot 的一致率與爭議清單、資料集參考標籤、裁決合併 gold |
| `tests/test_annotate.py` | 選段、密集影格覆蓋視窗、匯出檔名、合成檔端到端產生標註頁與快取、收檔→比較→裁決→gold 全流程 |

## `tree.txt` 範例（合成資料，節錄）

完整範例在 `setup_windows.bat` 跑完後的 `docs/examples/synthetic_normal/`。

```text
/ [G] 8 項  attrs {"note": "synthetic test file (not REASSEMBLE data)", "schema": "synthetic-v1"}
├── cam_left [D] shape=() opaque[6.8 KB]
│     ↳ 偵測為 mp4（6.8 KB，讀取方式 file-offset）：FMP4 32x24，標稱 30 fps，90 幀（header），[opencv 5.0.0]
├── events [D] shape=(4000, 3) int16  layout=chunked  chunks=(1024, 3)  gzip(4)  儲存 11.7 KB
│     ↳ 角落 [0:5, 0:3] [[255, 191, 153], [80, 92, 12], [22, 4, 52], [243, 194, 273], [151, 181, 291]]
├── segments_info [G] 4 項
│   ├── 0 [G] 5 項
│   │   ├── end [D] shape=() float64  = 1736420000.7
│   │   ├── Low_level [G] 3 項
│   │   ├── start [D] shape=() float64  = 1736420000.1
│   │   ├── success [D] shape=() bool  = true
│   │   └── text [D] shape=() str(vlen,utf-8)  = "Pick round peg 1"
└── timestamps [G] 7 項
    ├── cam_left [D] shape=(90,) float64
    │     ↳ 時間軸：n=90，1736420000.001871 → 1736420002.96681（推測單位 s、Unix 時間，2025-01-09T10:53:20.001+00:00 起），中位間隔 33.284 ms ≈ 30.045 Hz，單調不減
```

## 專案結構

```text
REASSEMBLE_VLM/
  setup_windows.bat / run_inspect.bat / check_segments.bat / index_all.bat / check_part.bat / extract_pilot.bat
                                        一鍵安裝 / 重跑檢查 / 第 1 包核對短片 / 全量索引 / 第 2 包核對短片 / 第 3 包 pilot
  package4_prepare.bat / compare_pilot.bat   第 4 包：標註頁＋切分 / 收檔、比較、裁決、gold
  annotate_rest.bat / compare_rest.bat       第 4 包：其餘 56 段的標註頁 / 收檔、比較、裁決、gold
  run_vlm_reasoning.bat / run_final_test.bat 推理與最後一張時間的實驗（開發集）/ 第 6 包測試集最終評估
  run_probe_v2.bat / run_finetune.bat / run_final_test_train.bat   第 7 包：改良小分類器 / LoRA 微調（開發集）/ 測試集
  run_vlm_probe.bat                     再訓練：VLM 特徵＋小分類器（開發集）
  setup_api.bat                         第 8 包：API SDK、.env（API key）、連線測試
  run_p8_dev.bat / run_final_test_usb.bat  第 8 包：BNC 開發集（Qwen、Skill、Claude、GPT、執行時間）／USB 新測試集
  run_skill_v2.bat / run_reasoning_bnc.bat / run_skill_v3.bat  第 8 包：Skill 第二版／蒐集商業模型推理／Skill 第三版（BNC 開發集）
  run_final_test_bolt4.bat              第 9 包：bolt 4 新測試集（驗證 Skill 第三版）
  run_final_test_p10.bat                第 10 包：Ethernet、square peg 1 新測試集（兩階段：Qwen 先判、沒答成功才用 GPT）
  run_reason_dev.bat                    第 11 包：推理方式（4 個新零件的開發集，不開測試集）
  run_reason_dev2.bat                   第 11 包第二輪：結合 Codex 的規則、最後兩張手腕裁切、手腕第二次觀察（開發集）
  run_final_test_p11.bat                第 11 包：4 個新測試集（square peg 4／3、round peg 2、waterproof），機率門檻 vs 第三版（跑一次）
  run_cascade_dev.bat                   第 12 包：GPT 在 4 個開發零件的 dev 段各跑一次 → 兩階段（Qwen 機率門檻→GPT）的開發集報告
  run_final_test_p12.bat                第 12 包：2 個新測試集（round peg 1、large gear），只用 GPT vs Qwen 機率門檻，另列兩階段（跑一次）
  setup_vlm.bat / run_vlm_dev.bat       第 5 包：GPU 套件與模型下載 / 開發集試跑
  configs/model.yaml / experiment.yaml / prompts/ / probe.yaml   第 5 包：模型、試跑設定、prompt 範本、做法 A
  configs/annotation.yaml               標註格式、失敗原因類別、比較門檻（annotation.schema.json 由它產生）
  configs/field_mapping.yaml            已確認的欄位映射（indexer 只讀這份）
  pyproject.toml / requirements.txt / environment.yml
  configs/dataset.yaml                  資料集來源、路徑、取樣與檢查預設值（不含 HDF5 key）
  src/assembly_monitor/
    inspect_h5.py   HDF5 走訪與資料集描述（不叫 inspect.py，避免和標準庫撞名）
    candidates.py   欄位角色推測、相機 / 時間軸配對、異常與待確認問題
    report.py       tree.txt / questions.md / 映射提案 / 總表
    media.py        內嵌 mp4/mp3 辨識與探測
    timeaxis.py     時間軸統計
    fetch.py        HTTP Range 取樣、CRC 與 manifest
    envinfo.py      環境與 GPU 紀錄
    check_clips.py  人工核對用的多視角對齊短片
    mapping.py      讀取並檢查 configs/field_mapping.yaml
    index.py        第 2 包 indexer / validation / Insert 候選報表
    stream.py       全量串流索引（下載 → 索引 → 精簡版 → 刪原檔，可續跑、防睡眠）
    slim.py         精簡版 HDF5（影片 + 相機時間戳 + 標註）
    extract.py      第 3 包影格擷取、contact sheet、metadata、預覽頁
    schema.py       第 4 包標註格式、驗證、JSON Schema、CSV／JSONL 匯入匯出
    split.py        第 4 包切分、洩漏檢查、test 清單鎖定
    agreement.py    第 4 包一致率、爭議清單、資料集參考標籤、gold 合併
    annotate.py     第 4 包標註任務（密集影格、標註頁、收檔、比較）
    vlm.py          第 5 包 ModelAdapter、PromptBuilder、Parser、Runner（續跑、錯誤隔離）
    probe.py        做法 A：其他零件訓練的小分類器（凍結 DINOv2＋邏輯迴歸；第 7 包加上編碼器與彙整方式的掃描）
    evaluate.py     第 6 包評分（任務判斷成功率、信賴區間、McNemar、依難度分組）
    final.py        第 6 包測試集凍結與執行（凍結鎖）
    finetune.py     第 7 包 Qwen3-VL-8B 的 QLoRA 微調
    vlm_probe.py    再訓練：Qwen3-VL-8B 內部特徵＋邏輯迴歸
    api_models.py   第 8 包：Claude／GPT API（同提示詞、同影格；延遲、token、估計費用）
    skill.py        第 8 包：Qwen＋放大鏡工具（框出零件 → 從原圖裁切放大 → 再判斷）
    timing.py       第 8 包：各方法的執行時間（每段平均、中位數、合計、模型載入、API 費用）
    web/annotate.html  離線標註頁（三視角同步、時間軸、快捷鍵、裁決模式）
    synthetic.py    合成測試檔
    cli.py          命令列入口
  scripts/          python -m scripts.xxx 入口
  tests/            pytest
  data/raw/         HDF5 原檔（不進 Git）
  data/index/       索引、zip 清單與結構報告
  data/slim/        全量索引時留下的精簡版（影片 + 時間戳 + 標註，不進 Git）
  data/derived/     取樣影格、預覽、標註頁與密集影格（不進 Git）
  data/splits/      切分與洩漏檢查報告（小檔，可進 Git）
  data/annotations/ 標註檔、gold 與比較報告（小檔，可進 Git）
  experiments/      VLM 原始回答、解析結果、錯誤與摘要（不進 Git）
  models/           模型權重快取（HF_HOME，不進 Git）
  docs/             說明與範例輸出
  logs/             執行紀錄
```

## 資料與授權

REASSEMBLE：Sliwowski et al., 2025, *A Multimodal Dataset for Contact-rich Robotic Assembly and Disassembly*, arXiv:2502.05086。
資料託管於 TU Wien Research Data（<https://researchdata.tuwien.ac.at/records/0ewrv-8cb44>），授權 CC BY 4.0，使用時請引用原論文。

## 各包狀態

| 包 | 內容 | 停止點與通過條件 | 狀態 | 報告 |
|---|---|---|---|---|
| 1 | 檢查檔案、建立環境、inspect HDF5 | 拿到 tree.txt 與欄位候選；使用者核對 | 已驗收 | `docs/package1_report.md` |
| 2 | indexer、validation、候選 Insert 數量報表 | 選出一種操作，索引 5 段人工核對 | 已驗收（Insert BNC） | `docs/package2_report.md` |
| 3 | 影格擷取、時間同步、40 段預覽 | 接觸畫面清楚，時間正確 | 已驗收（40 段全部可以） | `docs/package3_report.md` |
| 4 | 標註 schema、匯出、split/leakage audit | pilot 標註與無洩漏報告 | 完成：86 段 gold（pilot 30＋其餘 56） | `docs/package4_report.md` |
| 5 | VLM adapter、prompt、parser、續跑 | 小樣本 raw 與 parsed 輸出可核對 | 完成：開發集實驗 | `docs/package5_vlm.md` |
| 6 | metrics、圖表、完整 reproduce 命令 | 從固定輸入一鍵重算主表 | 完成：測試集跑過一次（2026-10-14） | `docs/package6_evaluation.md` |
| 7 | 訓練：LoRA 微調、改良小分類器 | 只用其他零件訓練；開發集看結果後凍結，測試集再跑一次 | 完成：測試集跑過一次（2026-10-16）；訓練沒有贏過不訓練 | `docs/package7_training.md` |

## Docker 與 GitHub（2026-11-09）

- 私人 GitHub repo `REASSEMBLE_VLM`：雙擊 **`push_github.bat`** 上傳／更新（用這台電腦的 Git 和你自己的 GitHub 登入）。
  不上傳 `.env`（API key）、原始資料、影格、模型權重、實驗結果、紀錄、進度報告簡報與 PDF；上傳前 `scripts/repo_check.py` 會再檢查一次。
- Docker：`Dockerfile`、`docker-compose.yml`（Python 3.11＋CUDA 12.6 版 torch 2.14.1，和本機環境相同）；
  `docker compose build` → `docker compose run --rm vlm`（單元測試）。Windows 可雙擊 `docker_build_test.bat`。說明見 `docs/docker.md`。
