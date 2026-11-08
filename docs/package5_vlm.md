# 第 5 包：VLM 推論器與結構化輸出

停止點：**小樣本 raw 與 parsed 輸出可核對**。先在開發集 20 段試跑三個候選模型，決定主力模型；測試集保持鎖定。

## 怎麼跑

1. `setup_vlm.bat`：安裝 PyTorch（CUDA，預設 cu126）、transformers、accelerate、bitsandbytes → `run_vlm --check` 檢查 GPU →
   列出模型大小與授權 → 按 Y 才從 Hugging Face 下載（`models\hf`）。
2. `run_vlm_dev.bat`：自我測試 → `extract_frames --set dev`（開發集 20 段的 1/4/8 幀）→
   每個模型 1 段快速檢查（`experiments\dev_quick`）→ 全部組合（`experiments\dev_smoke`）。執行時設 `HF_HUB_OFFLINE=1`，不會臨時下載。

中斷後再執行會接著做：已有答案（ok 或 invalid）的段略過，error（OOM、例外、缺影格）的段重試。

## 候選模型（`configs/model.yaml`）

| key | 模型 | 角色 | 設定 | 授權 |
|---|---|---|---|---|
| `qwen3vl_8b` | Qwen/Qwen3-VL-8B-Instruct | 主力 | 4-bit NF4（約 6 GB）、長邊 512 px | Apache-2.0 |
| `qwen3vl_4b` | Qwen/Qwen3-VL-4B-Instruct | 對照（只差大小） | 4-bit NF4（約 3.3 GB）、長邊 512 px | Apache-2.0 |
| `qwen35_9b` | Qwen/Qwen3.5-9B | 新一代（2026-02 發布），和 Qwen3-VL 比較 | 4-bit NF4、長邊 512 px、**關閉思考模式** | Apache-2.0 |
| `qwen35_4b` | Qwen/Qwen3.5-4B | 新一代的 4B | 同上 | Apache-2.0 |
| `gemma4_12b` | google/gemma-4-12B-it | 另一家的同級模型（2026-10-07 加入） | 4-bit NF4、長邊 512 px、思考模式關；沒有獨立視覺編碼器 | Apache-2.0（不需申請） |
| `gemma4_e4b` | google/gemma-4-E4B-it | Gemma 4 的小模型，對照 Qwen3.5-4B | 同上 | Apache-2.0 |
| `smolvlm2` | HuggingFaceTB/SmolVLM2-2.2B-Instruct | 極小基準 | bf16、多張圖不切塊 | Apache-2.0 |
| `majority` | — | 多數類基準（一律 success） | 不看影格 | — |
| `mock` | — | 部署檢查 | 不載入模型 | — |

顯存是 2026-10 查到的網路估計；實際值看 `raw.jsonl` 的 `peak_vram_mb` 與摘要。Qwen3-VL 需要 transformers 4.57 以上；Qwen3.5 需要 transformers 5（用 `model_class: AutoModelForMultimodalLM` 載入）。

**Qwen3.5（2026-10-06 加入）**：預設會先輸出一段 `<think>…</think>` 思考再回答，很慢也容易被 `max_new_tokens` 截斷，所以用 `chat_template_kwargs: {enable_thinking: false}` 關掉，和其他模型一樣直接回答 JSON。Parser 仍會先去掉 `<think>` 區塊；思考沒寫完就被截斷時記 invalid（`unfinished_thinking`）。Qwen3.5 的語言部分是混合架構（線性注意力＋一般注意力），Windows 上沒有 flash-linear-attention 時 transformers 會改用較慢的 PyTorch 版本，速度可能比 Qwen3-VL 慢。

新增模型只要在 `model.yaml` 加新的 key，**不要改既有 key 的設定**（會讓已跑完的結果被判定「設定已改」）。`run_vlm_dev.bat` 重跑時，所有組合都已跑完的模型會直接略過、不載入。

## 元件（`src/assembly_monitor/vlm.py`）

| 元件 | 行為 |
|---|---|
| ModelAdapter | `load()`、`predict(prompt)`、`repair(raw_text)`、`metadata()`；`TransformersAdapter`（Hugging Face）、`MajorityAdapter`、`MockAdapter` 可互換 |
| PromptBuilder | 範本 `configs/prompts/bnc_p1.txt`＋`experiment.yaml` 的零件、完成條件、相機、失敗原因（key 必須和 `annotation.yaml` 一致）；每張圖前標「Image i: camera X, t = ±s.ss s」；明示看不清就答 uncertain。版本 `bnc-p1`，內容雜湊記在每筆紀錄 |
| Parser | 保留原文；取出 JSON；`result` 不合法就是 invalid；其他欄位不合法時設為 null 並記在 `warnings`（例如 `bad_failure_reason`、`evidence_out_of_range`、`bad_confidence`）。**不替模型補答案** |
| 格式修正 | JSON 格式錯（找不到、解析失敗、不是物件）時，請同一模型把原文改寫成 JSON 一次（不給新資訊）；仍失敗記 invalid |
| Runner | 依切分讀 dev；每個（模型 × 幀數）一個資料夾；每段各自捕捉 OOM、例外、缺影格；生成超過 `timeout_sec` 會停止並記 `stopped_by_time`；改了 prompt／模型／取樣而沒換名稱會拒絕執行（避免新舊混在一起） |

影格依（時間、相機）排列：1 幀＝三台相機各一張最後狀態（標註結束後 2 秒）；4 幀＝12 張；8 幀＝24 張。送進模型前縮到長邊 512 px。

## 輸出（`experiments/<name>/<模型>/f<幀數>/`）

| 檔案 | 內容 |
|---|---|
| `run_config.json` | 模型設定、套件版本、GPU、prompt 版本與雜湊、取樣、切分；`config_sha256` 用來判斷能不能續跑 |
| `raw.jsonl` | 每段一行（重跑會追加）：送進模型的影格（相機、秒數、原始檔）、原始回答 `raw_text`、格式修正 `repair_text`、延遲、token 數、`peak_vram_mb`、錯誤 |
| `predictions.jsonl` | 解析結果：規格的固定欄位（result、failure_reason、evidence_start_sec／end_sec、best_camera、visible_evidence、confidence）＋ `status`（ok／invalid／error）、`errors`、`warnings`、`attempt`（first／repair）、`dataset_result` |
| `errors.csv` | invalid 與 error 的段 |
| `../summary.csv`、`../summary.md` | 每個組合的格式正確、無效、錯誤、無法判斷段數，快速準確率與失敗召回（用資料集成敗標記，僅供試跑參考），平均秒數、峰值顯存 |

正式評估（Macro-F1、失敗召回、拒答率、原因、證據命中，依畫面難度分組）在第 6 包，用人工標準答案，並可從 `raw.jsonl` 重算。

## 做法 A：有訓練的對照組（`configs/probe.yaml`、`src/assembly_monitor/probe.py`）

使用者 2026-10-06 決定加入。主實驗仍是「現成模型不訓練」；做法 A 回答「簡單訓練能多多少、能不能套用到沒看過的零件」。

- **訓練池**：插入動作、零件不是 BNC、而且錄影不在 BNC 測試集的 48 支錄影裡（只讀錄影 ID 來排除）。
  目前 368 段（成功 293、失敗 75），來自 16 種零件、35 支錄影。標記只用資料集的成功／失敗。
- **特徵**：凍結的 DINOv2-base（Apache-2.0，約 350 MB）對每張影格取 CLS＋patch 平均；每段依（時間、相機）串接，1/4/8 幀各做一次。
- **分類器**：標準化 → PCA（32／64 維）→ 邏輯迴歸（失敗、成功等權）；用「依錄影分組」的 5 折交叉驗證挑參數，再用整個訓練池重訓。
- **評分**：BNC 開發集 20 段（摘要表的 `probe_dinov2` 列）；最終才用 `python -m scripts.probe run --unlock-test` 評 BNC 測試集。
- **注意**：訓練池含 BNC 開發集那 14 支錄影裡的其他零件，開發集分數可能略為樂觀；測試集不受影響。分類器只輸出成功／失敗與機率，不給原因、證據時間。

## 提示詞版本（只用開發集調整）

每一版是一份獨立的實驗設定檔，跑過就不再改；要再調整就另存新檔（新名稱、新版本號），舊結果都保留。
`run_vlm_dev.bat` 會依序跑 `configs/experiment.yaml`（第 1 版）和所有 `configs/experiment_p*.yaml`，已跑完的組合略過，
最後寫 `experiments/prompt_versions.md`：同一模型、同一幀數的各版本放在一起，附多數類與做法 A 當參考線。

| 版本 | 設定檔 | 實驗名稱 | 模型 | 改了什麼 |
|---|---|---|---|---|
| bnc-p1 | `experiment.yaml` | dev_smoke | 全部候選 | 第一版 |
| bnc-p2 | `experiment_p2.yaml` | dev_p2 | Qwen3.5-9B、Qwen3-VL-8B、Gemma 4 12B／E4B | 見下 |

**第 2 版（2026-10-06）為什麼改**：第 1 版所有 VLM 都太常判失敗。開發集有 5 段成功段，每個 VLM 都判成 `incomplete_insert`，
理由幾乎都是「對準了但沒轉、沒鎖上」，是在換句話說提示詞裡的定義；旋轉從幾張靜止畫面看不到。信心分數也幾乎都是 0.95。
第 2 版：明講旋轉通常看不到、不能因此判沒鎖上；只有畫面中指得出具體證據時才判失敗，並寫出第幾張圖；看不出來答 uncertain；
提醒同時看外部相機；失敗原因改寫成看得到的線索；confidence 改成「答對的機率」。切分、取樣、相機、幀數都和第 1 版相同。

預期取捨：畫面看不出來的失敗（標準答案 `visual_difficulty: not_visible`）可能改判成功或 uncertain，這是照實回報，
第 6 包會依畫面難度分開評分。開發集只有 20 段，提示詞最多改一兩版，避免變成專門配合這 20 段。

摘要另外列「成功被誤判」（真的成功卻答失敗的段數）與「兩類平均」（失敗召回與成功答對率的平均；一律答成功＝50%），
因為開發集成功段比較多，只看準確率會被誤導。

## 下週實驗：只用提示詞引導推理（事先登記，2026-10-07）

使用者排程：本週評估 Qwen3.5、Gemma 4 等；下週不訓練、只用提示詞引導模型推理，算任務判斷成功率；下下週再嘗試訓練。
下面的設定在 Claude 看任何測試段之前寫好並部署（Claude 接著要獨立標註其餘 56 段，其中 41 段是測試段），
避免看了測試段再回頭設計提示詞。下週只在這些變體裡、用開發集挑；之後若要新增變體，只能根據開發集的觀察。
執行：雙擊 `run_vlm_reasoning.bat`（`run_vlm_dev.bat` 不會跑這些）。

| 代號 | 設定檔 | 和第 1 版的差別 | 模型 | 幀數 |
|---|---|---|---|---|
| E1 | `experiment_e1.yaml`（dev_eb1） | 最後一張改成段結束後 1 秒 | Qwen3.5-9B、Qwen3-VL-8B、Gemma 4 12B | 1/4/8 |
| E0.5 | `experiment_e05.yaml`（dev_eb05） | 最後一張改成段結束後 0.5 秒 | 同上 | 1/4/8 |
| R1 | `experiment_r1.yaml`（dev_r1） | 先分步驟寫推理（最後狀態 → 過程 → 逐條檢查 → 結論）再給 JSON；明講要看放開當下、不要看下一個動作 | 同上 | 4/8 |
| R3 | `experiment_r3.yaml`（dev_r3，2026-10-14 加入） | 以 E0.5 為基礎（第 1 版提示詞、+0.5 秒），只多一段「對準時接頭下墜」線索：腕部相機中對準成功時接頭會一瞬間明顯下墜、套進插座（插座方形法蘭從接頭上方跳到夾爪指尖附近）；整段沒有下墜＝對不準。下墜只代表套上，還要旋轉鎖上、放開後留在插座上才算成功。R3 對 E0.5 的差＝線索的效果 | Qwen3.5-9B、Qwen3-VL-8B、Gemma 4 12B | 4/8 |
| R2 | `experiment_r2.yaml`（dev_r2） | 提示詞同第 1 版，只打開模型內建的思考模式 | Qwen3.5-9B、Gemma 4 12B | 8 |
| A-E1／A-E0.5 | `probe_eb1.yaml`、`probe_eb05.yaml` | 做法 A 改用 +1／+0.5 秒的影格 | DINOv2＋邏輯迴歸 | 1/4/8 |

**R3 的線索從哪裡來**：使用者（標註者）2026-10-14 說明 BNC 接頭的結構——對準時若有對準，腕部相機會看到接頭一瞬間明顯下墜。這是接頭的機械特性，不是從測試段的結果歸納的；Claude 在開發集的腕部影格上確認過例子（成功段 `2025-01-10-18-31-44__seg2` 約 8.5→9.1 秒法蘭往下跳，對不準的 `2025-01-13-11-07-43__seg6` 整段法蘭都停在接頭上方）。用腕部畫面背景的縮放自動量測則分不出成功與失敗（下墜幅度小於量測誤差），所以交給模型看接頭與法蘭。R3 在跑測試集之前加入，和其他變體一樣只用開發集比較；以 E0.5 為基礎，是因為 2026-10-13 的開發集結果中「第 1 版提示詞＋最後一張 +0.5 秒」最好，R1（分步驟）與 R2（思考模式）都沒有幫助。

**為什麼測最後一張的時間**：第 1 版的最後一張在段結束後 2 秒，常已拍到下一個動作。開發集 13 段成功裡 12 段的下一個動作是「拿別的零件」
（手臂移走，腕部相機看到別的零件，模型說「接頭掉在板子上」，例如 `2025-01-13-11-07-43__seg7`）；失敗段的下一個動作是重插。
下一個動作和答案相關（操作者看到失敗才重插），所以 2 秒的畫面可能讓模型或分類器「看下一個動作」猜答案。
2 秒當初是刻意選的（插入段的標註常早 1–2 秒結束，見 `field_mapping.yaml`），所以要選一個看得到放開、又還沒進入下一個動作的長度。

**解析**：推理輸出取最後一個有 `result` 的 JSON 物件；Qwen3.5 的 `<think>…</think>` 與 Gemma 4 的 `<|channel>thought…<channel|>` 會先去掉，
思考沒寫完就被截斷記 invalid（`unfinished_thinking`）。實驗可用 `model_overrides` 覆寫模型設定（例如打開思考、放寬輸出長度），
不改 `model.yaml` 的既有 key。

**成功率怎麼報**：答對率（任務判斷成功率）之外，一定一起報抓到失敗、成功被誤判與兩類平均——一律答成功就有 65% 答對率。
測試集的成敗只需要資料集標記，不必等標註；但設定必須先凍結，測試集只跑一次。

## 怎麼選主力模型

在開發集看三件事：跑不跑得動（`peak_vram_mb`、平均秒數、OOM）、會不會照格式回答（invalid 與 warnings 的比例）、成敗判斷大概準不準。
prompt 與取樣設定的調整只用開發集；改了 prompt 就換 `experiment.name`（或版本號），保留每一版的結果。
