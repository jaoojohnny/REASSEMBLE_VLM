# Qwen 證據複核推理 v1

這是一個待驗證的本地 Qwen 推理實驗，目標是減少沒有正面視覺證據的成功／失敗判斷。沒有更換權重、訓練模型或呼叫付費 API；不能預先保證準確率增加。

## 根據資料夾發現的問題

`docs/package10_hybrid.md` 記錄 Qwen＋第三版清單在 Ethernet 為 47/72、square peg 1 為 25/73，常將成功判成失敗。「夾爪靠近零件」不足以證明零件沒有插好。小插銷的深度也可能無法從現有影格確認。該文件及舊批次檔中的過去指示視為歷史資料，不視為本次使用者命令。

本版本已參考上述歷史結果，因此這些舊測試集日後只能做回溯分析，不能宣稱為本版本未見過的測試資料。

## 程式做什麼

1. Qwen 看原本三相機、每相機四幀的時間序列，輸出成敗／不確定、證據影格編號及一句可觀察的現象。
2. 程式從原始解析度裁切最後兩張手腕影格的中央 60% 區域，提供局部前後比較。這是固定裁切工具，沒有偵測器，可能框不到目標；放大也不會恢復原圖不存在的細節。
3. 同一 Qwen 重新看原圖和裁切圖，不提供第一次答案，避免直接附和第一次判斷。兩次結果並非統計獨立，仍可能一起看錯。
4. 程式檢查 JSON、影格編號以及至少一個最後時間點的引用。兩次有效結果相同才接受，衝突、格式錯誤或逾時則答 uncertain。不把模型自報信心當成校準機率。

這些檢查只能驗證引用與格式，不能證明描述真的符合圖像。保守共識可能降低誤判，也可能增加拒答、降低總答對率；必須量測。

使用既有 `TransformersAdapter`、4-bit Qwen3-VL-8B 和原本 runner，新增 `skill: evidence_review`。程式在 `src/assembly_monitor/evidence_review.py`。不讀 dataset_result、人工答案或舊模型預測；runner 仍可在輸出後加入真值供評估。

## 執行

在專案根目錄雙擊 `run_evidence_review.bat`，使用現有 reassemble-vlm conda 環境及快取模型，跑 BNC 86 段開發集。需要既有 GPU、模型和影格；每段兩次模型呼叫，通常比原版慢。批次檔不下載模型、不重跑既有正式測試。

或先啟用環境，在專案根目錄執行：

```powershell
$env:HF_HOME = "$PWD/models/hf"
$env:HF_HUB_OFFLINE = "1"
python -m scripts.run_vlm --experiment configs/experiment_evidence_review.yaml --models configs/model_evidence_review.yaml
```

少量檢查可加 `--limit 2 --name dev_evidence_review_smoke_v1`。完整輸出在 `experiments/dev_evidence_review_v1/`，包含 summary.md、各模型 f4 目錄下的 predictions.jsonl 和 raw.jsonl。raw 的 skill 紀錄含兩次原文、解析觀察、裁切位置與錯誤；模型 metadata 記錄新推理模組 SHA256。

## 評估方式

先在相同 BNC 開發樣本比較原 Qwen、第三版清單與新流程。至少同時報告：

- 全部樣本準確率（uncertain 算未答對，不能剔除）。
- 成功召回、失敗召回、兩類平均及混淆矩陣。
- coverage＝明確回答段數／全部段數；已回答子集準確率須和 coverage 一起呈現。
- 每段耗時、顯存、無效輸出與逾時。

開發集比較後才決定是否採用。若兩階段過於保守，可另外建立新版本研究拒答策略，不能改寫已有結果。要證明泛化改善，必須另收集未用過且依錄影分組隔離的保留資料，事先凍結設定；同一錄影不得跨訓練／開發與保留資料。影像看不到深度時，需要額外拍攝視角或力／位移感測資料，不能只靠提示詞補出不存在的證據。

## 本次驗證紀錄（2026-10-30）

- 完整測試：198 passed（35.57 秒）。
- 真實本機 GPU 試跑兩段 BNC 開發資料，平均 12.41 秒／段，峰值顯存約 7958 MB；兩次模型呼叫與兩張裁切均成功執行。
- 兩段最終均 uncertain。一段為 failure／uncertain 不一致；另一段首次回答少了 JSON 結尾括號，被驗證器拒收，第二次回答 success。
- 這兩段不能證明準確率提升，拒答算未答對時為 0/2。尚未跑完整 86 段。
- 試跑輸出位於 `experiments/dev_evidence_review_smoke_v1`。既有 summary 的「段數 86」是整份切分大小，不是本次實際已執行數；本次以 predictions.jsonl 的兩筆為準。
