# 標註比較報告：rest56

產生時間：2026-10-14T13:02:32+08:00

## 來源

| 代號 | 標註者 | 類型 | 已完成段數 |
|---|---|---|---|
| A | 張立醒 | 人 | 56 |
| B | claude | AI（Claude） | 56 |
| C | dataset_lowlevel | 資料集低階技能（規則） | 56 |

說明：κ＝Cohen's kappa（扣除碰巧一致後的一致程度，1＝完全一致、0＝與碰巧相同）；失敗原因只比兩邊都判「失敗」的段（分母 n）；證據區間「命中」＝IoU ≥ 門檻，或起訖都在容許誤差內。
需要裁決的欄位依 configs/annotation.yaml 的 compare.blocking_fields；其他欄位的差異只列入統計。證據區間短於 evidence.min_length_sec（例如起點＝終點）的段一律列入裁決（evidence_length）。

## A vs B

共同段數 56（只有 A：0、只有 B：0）；需要裁決的爭議 7 段 {'failure_reason': 7, 'best_camera': 1}；有任何差異 54 段 {'evidence': 49, 'failure_reason': 7, 'best_camera': 1}

| 項目 | n | 一致 | 一致率 | κ |
|---|---|---|---|---|
| 結果 | 56 | 56 | 100%（95% CI 100%–100%） | 1.00（95% CI 1.00–1.00） |
| 失敗原因 | 10 | 3 | 30%（95% CI 0%–60%） | 0.14（95% CI -0.21–0.44） |
| 最佳視角 | 56 | 55 | 98%（95% CI 95%–100%） | 0.79（95% CI 0.00–1.00） |

證據區間（n=56）：平均 IoU 0.23（95% CI 0.17–0.30）、中位 0.13、命中 7/56（12%）、起點平均差 7.40 秒、終點平均差 1.91 秒

失敗原因對照（列＝A、欄＝B）

| | incomplete_insert | misalignment | slip_or_drop | unknown | wrong_contact |
|---|---|---|---|---|---|
| incomplete_insert | 1 | 1 | 0 | 0 | 0 |
| misalignment | 2 | 1 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 1 | 1 |
| slip_or_drop | 2 | 0 | 1 | 0 | 0 |

各視角可見度一致率：hama1 93%（n=56）、hama2 95%（n=56）、hand 95%（n=56）

## A vs C

共同段數 56（只有 A：0、只有 C：0）；與參考標籤有差異 54 段 {'evidence': 45, 'failure_reason': 9}

| 項目 | n | 一致 | 一致率 | κ |
|---|---|---|---|---|
| 結果 | 56 | 56 | 100%（95% CI 100%–100%） | 1.00（95% CI 1.00–1.00） |
| 失敗原因 | 10 | 1 | 10%（95% CI 0%–30%） | 0.00（95% CI -0.14–0.17） |
| 最佳視角 | 0 | 0 | — | — |

證據區間（n=53）：平均 IoU 0.13（95% CI 0.05–0.21）、中位 0.00、命中 8/53（15%）、起點平均差 8.94 秒、終點平均差 1.23 秒

失敗原因對照（列＝A、欄＝C）

| | incomplete_insert | misalignment | unknown | wrong_contact |
|---|---|---|---|---|
| incomplete_insert | 0 | 0 | 2 | 0 |
| misalignment | 1 | 1 | 0 | 1 |
| other | 0 | 0 | 0 | 2 |
| slip_or_drop | 1 | 1 | 1 | 0 |

## B vs C

共同段數 56（只有 B：0、只有 C：0）；與參考標籤有差異 48 段 {'evidence': 44, 'failure_reason': 6}

| 項目 | n | 一致 | 一致率 | κ |
|---|---|---|---|---|
| 結果 | 56 | 56 | 100%（95% CI 100%–100%） | 1.00（95% CI 1.00–1.00） |
| 失敗原因 | 10 | 4 | 40%（95% CI 10%–70%） | 0.25（95% CI -0.04–0.58） |
| 最佳視角 | 0 | 0 | — | — |

證據區間（n=53）：平均 IoU 0.38（95% CI 0.33–0.43）、中位 0.31、命中 9/53（17%）、起點平均差 1.97 秒、終點平均差 0.74 秒

失敗原因對照（列＝B、欄＝C）

| | incomplete_insert | misalignment | unknown | wrong_contact |
|---|---|---|---|---|
| incomplete_insert | 2 | 0 | 2 | 1 |
| misalignment | 0 | 1 | 1 | 0 |
| slip_or_drop | 0 | 1 | 0 | 0 |
| unknown | 0 | 0 | 0 | 1 |
| wrong_contact | 0 | 0 | 0 | 1 |

## 爭議清單（A vs B，共 7 段，需要裁決）

| 段 | 資料集 | 爭議 | 結果 | 原因 | 視角 | 證據區間 / IoU |
|---|---|---|---|---|---|---|
| 2025-01-09-14-47-32__seg32 | failure | failure_reason | failure / failure | other（其他） / unknown（看不出原因） | hama1 / hama1 | [0.01, 12.59] / [0.0, 12.6] IoU 0.9984 |
| 2025-01-09-17-23-07__seg12 | failure | failure_reason | failure / failure | misalignment（對不準） / incomplete_insert（沒插到底／沒鎖上） | hand / hand | [16.36, 32.98] / [16.4, 34.4] IoU 0.9191 |
| 2025-01-09-17-23-07__seg17 | failure | failure_reason | failure / failure | slip_or_drop（滑脫或掉落） / incomplete_insert（沒插到底／沒鎖上） | hand / hand | [7.49, 10.68] / [7.5, 11.5] IoU 0.793 |
| 2025-01-09-17-40-57__seg14 | failure | failure_reason | failure / failure | incomplete_insert（沒插到底／沒鎖上） / misalignment（對不準） | hand / hand | [20.26, 33.08] / [6.5, 33.3] IoU 0.4784 |
| 2025-01-09-18-08-36__seg17 | failure | failure_reason, best_camera | failure / failure | other（其他） / wrong_contact（接觸錯誤） | hama1 / hand | [-0.01, 3.26] / [0.0, 3.3] IoU 0.9849 |
| 2025-01-13-11-23-17__seg27 | failure | failure_reason | failure / failure | slip_or_drop（滑脫或掉落） / incomplete_insert（沒插到底／沒鎖上） | hand / hand | [4.87, 7.96] / [5.4, 8.4] IoU 0.7252 |
| 2025-01-13-19-04-46__seg2 | failure | failure_reason | failure / failure | misalignment（對不準） / incomplete_insert（沒插到底／沒鎖上） | hand / hand | [12.39, 17.95] / [17.5, 20.3] IoU 0.0569 |
