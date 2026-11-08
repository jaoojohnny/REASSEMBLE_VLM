# 標註比較報告：pilot30

產生時間：2026-10-06T18:05:05+08:00

## 來源

| 代號 | 標註者 | 類型 | 已完成段數 |
|---|---|---|---|
| A | 張立醒 | 人 | 30 |
| B | claude | AI（Claude） | 30 |
| C | dataset_lowlevel | 資料集低階技能（規則） | 30 |

說明：κ＝Cohen's kappa（扣除碰巧一致後的一致程度，1＝完全一致、0＝與碰巧相同）；失敗原因只比兩邊都判「失敗」的段（分母 n）；證據區間「命中」＝IoU ≥ 門檻，或起訖都在容許誤差內。
需要裁決的欄位依 configs/annotation.yaml 的 compare.blocking_fields；其他欄位的差異只列入統計。證據區間短於 evidence.min_length_sec（例如起點＝終點）的段一律列入裁決（evidence_length）。

## A vs B

共同段數 30（只有 A：0、只有 B：0）；需要裁決的爭議 14 段 {'result': 5, 'best_camera': 7, 'failure_reason': 5, 'evidence_length': 1}；有任何差異 29 段 {'result': 5, 'evidence': 27, 'best_camera': 7, 'failure_reason': 5, 'evidence_length': 1}

| 項目 | n | 一致 | 一致率 | κ |
|---|---|---|---|---|
| 結果 | 30 | 25 | 83%（95% CI 70%–97%） | 0.70（95% CI 0.49–0.93） |
| 失敗原因 | 15 | 10 | 67%（95% CI 40%–87%） | 0.47（95% CI 0.06–0.80） |
| 最佳視角 | 30 | 23 | 77%（95% CI 60%–90%） | 0.00（95% CI 0.00–0.00） |

證據區間（n=30）：平均 IoU 0.18（95% CI 0.12–0.24）、中位 0.11、命中 3/30（10%）、起點平均差 6.37 秒、終點平均差 1.43 秒

失敗原因對照（列＝A、欄＝B）

| | incomplete_insert | misalignment | other |
|---|---|---|---|
| incomplete_insert | 4 | 1 | 0 |
| misalignment | 2 | 5 | 0 |
| other | 0 | 0 | 1 |
| slip_or_drop | 2 | 0 | 0 |

各視角可見度一致率：hama1 40%（n=30）、hama2 60%（n=30）、hand 83%（n=30）

## A vs C

共同段數 30（只有 A：0、只有 C：0）；與參考標籤有差異 29 段 {'evidence': 26, 'failure_reason': 9, 'evidence_length': 1}

| 項目 | n | 一致 | 一致率 | κ |
|---|---|---|---|---|
| 結果 | 30 | 30 | 100%（95% CI 100%–100%） | 1.00（95% CI 1.00–1.00） |
| 失敗原因 | 20 | 11 | 55%（95% CI 35%–75%） | 0.34（95% CI 0.08–0.60） |
| 最佳視角 | 0 | 0 | — | — |

證據區間（n=27）：平均 IoU 0.21（95% CI 0.13–0.28）、中位 0.22、命中 1/27（4%）、起點平均差 9.28 秒、終點平均差 0.76 秒

失敗原因對照（列＝A、欄＝C）

| | incomplete_insert | misalignment | unknown |
|---|---|---|---|
| incomplete_insert | 6 | 1 | 1 |
| misalignment | 2 | 5 | 0 |
| other | 0 | 0 | 1 |
| slip_or_drop | 1 | 1 | 0 |
| wrong_contact | 1 | 0 | 1 |

## B vs C

共同段數 30（只有 B：0、只有 C：0）；與參考標籤有差異 17 段 {'result': 5, 'evidence': 13, 'failure_reason': 4}

| 項目 | n | 一致 | 一致率 | κ |
|---|---|---|---|---|
| 結果 | 30 | 25 | 83%（95% CI 70%–97%） | 0.70（95% CI 0.49–0.93） |
| 失敗原因 | 15 | 11 | 73%（95% CI 47%–93%） | 0.53（95% CI 0.12–0.86） |
| 最佳視角 | 0 | 0 | — | — |

證據區間（n=27）：平均 IoU 0.38（95% CI 0.26–0.49）、中位 0.38、命中 14/27（52%）、起點平均差 5.49 秒、終點平均差 0.81 秒

失敗原因對照（列＝B、欄＝C）

| | incomplete_insert | misalignment | unknown |
|---|---|---|---|
| incomplete_insert | 6 | 2 | 0 |
| misalignment | 1 | 5 | 0 |
| other | 0 | 0 | 1 |

## 爭議清單（A vs B，共 14 段，需要裁決）

| 段 | 資料集 | 爭議 | 結果 | 原因 | 視角 | 證據區間 / IoU |
|---|---|---|---|---|---|---|
| 2025-01-09-15-37-27__seg25 | failure | result | failure / uncertain | — / — | hand / hand | [29.56, 31.34] / [31.0, 34.5] IoU 0.0688 |
| 2025-01-09-17-03-05__seg4 | failure | best_camera | failure / failure | other（其他） / other（其他） | hama1 / hand | [1.97, 3.92] / [0.0, 4.8] IoU 0.4062 |
| 2025-01-09-17-03-05__seg6 | success | best_camera | success / success | — / — | hama1 / hand | [21.62, 34.04] / [33.1, 35.9] IoU 0.0658 |
| 2025-01-09-17-23-07__seg14 | failure | failure_reason, best_camera | failure / failure | incomplete_insert（沒插到底／沒鎖上） / misalignment（對不準） | hama1 / hand | [33.91, 36.11] / [9.0, 36.1] IoU 0.0808 |
| 2025-01-09-18-08-36__seg16 | failure | best_camera | failure / failure | incomplete_insert（沒插到底／沒鎖上） / incomplete_insert（沒插到底／沒鎖上） | hama1 / hand | [14.62, 17.32] / [17.7, 19.3] IoU 0.0 |
| 2025-01-10-17-25-29__seg30 | failure | evidence_length | failure / failure | misalignment（對不準） / misalignment（對不準） | hand / hand | [23.59, 23.59] / [7.1, 24.1] IoU 0.0 |
| 2025-01-10-18-12-06__seg31 | failure | failure_reason, best_camera | failure / failure | misalignment（對不準） / incomplete_insert（沒插到底／沒鎖上） | hama2 / hand | [33.99, 42.5] / [42.3, 44.3] IoU 0.0194 |
| 2025-01-11-14-35-33__seg17 | failure | result, best_camera | failure / uncertain | — / — | hama2 / hand | [21.04, 24.36] / [24.4, 27.1] IoU 0.0 |
| 2025-01-11-15-18-22__seg36 | failure | failure_reason | failure / failure | misalignment（對不準） / incomplete_insert（沒插到底／沒鎖上） | hand / hand | [24.96, 29.41] / [28.7, 30.5] IoU 0.1282 |
| 2025-01-13-11-23-17__seg26 | failure | failure_reason | failure / failure | slip_or_drop（滑脫或掉落） / incomplete_insert（沒插到底／沒鎖上） | hand / hand | [7.65, 9.0] / [8.1, 10.5] IoU 0.3158 |
| 2025-01-13-15-50-28__seg11 | failure | failure_reason | failure / failure | slip_or_drop（滑脫或掉落） / incomplete_insert（沒插到底／沒鎖上） | hand / hand | [13.19, 16.25] / [15.1, 17.5] IoU 0.2668 |
| 2025-01-13-16-19-55__seg4 | failure | result | failure / uncertain | — / — | hand / hand | [22.93, 28.52] / [25.1, 29.5] IoU 0.5205 |
| 2025-01-13-17-40-45__seg16 | failure | result | failure / uncertain | — / — | hand / hand | [12.04, 14.69] / [12.0, 16.7] IoU 0.5638 |
| 2025-01-13-19-04-46__seg5 | failure | result, best_camera | failure / uncertain | — / — | hama1 / hand | [15.33, 17.99] / [17.0, 21.9] IoU 0.1507 |
