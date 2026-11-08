# 第 6 包：評分與測試集最終評估

兩個工具：

| 工具 | 用途 | 會不會讀測試集 |
|---|---|---|
| `python -m scripts.evaluate`（`run_vlm_dev.bat`、`run_vlm_reasoning.bat` 最後會自動跑） | 開發集所有已跑過的組合一起評分 → `experiments/dev_evaluation/report.md`，用來挑方法 | 不會 |
| `run_final_test.bat`（`python -m scripts.final_test`） | 用凍結的設定在測試集跑一次並評分 → `experiments/final_test/report.md` | 會，只在凍結後 |

## 指標（`src/assembly_monitor/evaluate.py`）

- **任務判斷成功率**＝答對段數 ÷ 全部段數。無法判斷、無效、錯誤、沒有預測都算答錯。
- 一律答成功的基準就有「成功段比例」那麼高（開發集 65%、測試集 65%），所以一定一起報：抓到失敗（失敗召回）、
  成功被誤判、兩類平均（平衡準確率）、Macro-F1、無法判斷段數、無效＋錯誤段數。
- **95% 信賴區間**：以錄影為單位重抽樣（同一支錄影的段一起抽）2000 次，固定 seed。
- **和主方法比較**：McNemar 精確檢定（只看一個方法對、另一個錯的段）。
- **真實答案**：有正式人工標準答案（gold，approved／adjudicated）的段用 gold 的結果，其餘用資料集的成功／失敗標記。
- **依畫面難度分組**（clear／subtle／not_visible／未標）、**失敗原因正確率**、**證據區間命中**（區間有重疊）與平均 IoU：只在有 gold 的段上算。

## 測試集只跑一次：凍結流程

1. 開發集實驗跑完（`run_vlm_reasoning.bat`），看 `experiments/dev_evaluation/report.md` 和 `experiments/prompt_versions.md`。
2. 和 Claude 一起選方法，寫進 `configs/final_test.yaml` 的 `methods`（kind：`majority`、`vlm`、`probe`），填 `primary`（主方法）、
   `decided`（一句話理由）、`frozen_at`，再把 `frozen` 改成 `true`。
3. 雙擊 `run_final_test.bat`：自我測試 → 列出凍結的方法（沒凍結就停下，不會打開測試集）→ 擷取測試段影格 → 每個方法跑一次 → 評分。
4. 第一次執行會寫 `experiments/final_test/lock.json`：`final_test.yaml`、`model.yaml`、每個方法的實驗設定與提示詞範本、做法 A 設定的 SHA-256。
   之後任何一個檔案內容改變就拒絕執行。中斷後再雙擊會接著跑（同一份設定）。
5. 真的必須重跑（例如發現程式錯誤）時，把整個 `experiments/final_test` 改名保留，並在報告中說明原因。

測試集的成敗只需要資料集標記；原因、證據與難度分組需要 gold，56 段裁決完成後重新執行 `python -m scripts.final_test evaluate`
即可用同一批預測重算（不必重跑模型）。
