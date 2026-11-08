# 第 13 包：外部插頭資料 REBOOT（16 mm 圓柱當開發集）

更新：2026-11-06｜狀態：**流程建好、圓柱開發集跑完；下一階段待定**。報告：`experiments/p13_reboot_cyl/report.md`。
（下面先是整理好的流程總覽；最後面是依時間的工作紀錄。）

## 1. 目的與定位

- 到第 12 包為止所有結果都只來自 REASSEMBLE。第 13 包找外部資料，檢驗方法換機器人、換實驗室、換視角後還準不準。使用者 11/4 指定先以插頭類為主。
- ViFailback 的插頭題太少（插頭插入只有 16 段、3 段成功）→ 改用 **REBOOT**（2026-09；NIST 組裝任務板：USB-A、USB-C、RJ45、NEMA 1-15P 電源插頭、RCA、HAN 10E，另有 M12、16 mm 方柱／圓柱；雙臂 WidowX；4 台相機；每任務 60 段專家示範＋60 段救援段；Apache-2.0）。
- 插頭零件沒有公開失敗時間點；只有官方範例（16 mm 圓柱救援段）附 `meta/phase.json` → 使用者選「先做圓柱」：
  - **圓柱＝開發集**：用來設計流程（怎麼切段、用哪台相機）。
  - **插頭零件＝留作測試**：影格都沒看過，等流程定案後只測一次。

## 2. 流程（4 個 .bat，依序）

| 步驟 | .bat | 做什麼 | 下載／讀取 | 主要輸出 |
|---|---|---|---|---|
| 1 說明檔檢查 | `run_reboot_meta.bat` | 4 種插頭（專家＋救援）＋官方範例共 10 個資料集：只抓 README、meta；查失敗時間點、AV1 能不能解、下載量 | 約 5 MB；環境加裝 pyarrow | `experiments/p13_reboot_meta/` |
| 2 圓柱下載＋檢查圖 | `run_reboot_cyl.bat` | 救援段 60＋專家示範 62；俯視＋左右手腕彩色影片；每段一張檢查圖 | 4.64 GB | `data/raw/reboot/`、`experiments/p13_cyl_check/` |
| 3 找放開時刻、切段 | `run_reboot_seg.bat` | 只讀 data/ 的關節欄位（深度不讀）→ 右手夾爪曲線 → 放開時刻 → 成功段；失敗段用官方「救援開始」 | 6.5 MB | `experiments/p13_cyl_segments/` |
| 4 凍結方法 × 圓柱 | `run_reboot_cyl_models.bat` | 加抓低角度前視 cam_low；做成 REASSEMBLE 格式的影格與切分；GPT、Qwen 跑；計分 | 0.93 GB；GPT 147 次 US$1.01 | `data/derived/frames/reboot_cyl/`、`data/splits/reboot_cyl_dev/`、`experiments/p13_reboot_cyl/` |

每一步都有自我測試（使用者電腦最後一次 35 項通過）；中途關掉可以接著跑。

## 3. 段的定義（定案）

- **成功段**（專家示範）：右手夾爪曲線每段都是「閉合（原位）→ 張開 → 半開（夾著圓柱）→ 張開」，取最後一次放開；62/62 段都找到，放開在 13.8～23.6 秒。
- **失敗段**（救援段）：失敗發生在放置階段（Align／Engage place）的段；結尾＝官方 `recovery_started_at_frame`。
  - 官方標註有雜訊：只用「失敗開始／救援開始」，不用階段邊界（常錯）；去掉 bad_order、lucky success、階段編號錯 → 23 段。
  - retention_failure（夾爪不放開，圓柱可能已插好）3 段照跑，主要成績排除 → **20 段**。
- **長度與取樣**：每段 3 秒；凍結取樣 uniform_n4_sb0.5_eb0.5（4 格），段結尾放在最後一格前 0.5 秒，所以最後一格＝判斷時刻：
  - 成功 A（放開前）：最後一格＝夾爪開始放開
  - 成功 B（放開後）：最後一格＝放開後 1 秒
  - 失敗：最後一格＝救援開始（A、B 共用）
- **相機**：hama1＝俯視 cam_high、hama2＝低角度前視 cam_low、hand＝右手腕；640×480。
- **方法**：第 12 包凍結的實驗設定原封不動（只換名稱、切分、影格資料夾；測試會檢查其他欄位完全相同）；零件名稱仍是 round peg；手腕固定裁切照舊。

## 4. 結果（圓柱開發集）

- **成功段 A（放開前）**：所有方法把 62 段成功全判成失敗（成功段答對 0/62）→ 凍結提示詞要看到「夾爪放開」才判成功，A 不能用。
- **成功段 B（放開後）**（成功 62、失敗 20）：

| 方法 | 失敗段答對 | 成功段答對 | 兩類平均 | 問 GPT |
|---|---|---|---|---|
| 只用 GPT ★ | 19/20 | 62/62 | 98% | 82 |
| 兩階段 第三版→GPT | 19/20 | 62/62 | 98% | 67 |
| 兩階段 門檻→GPT | 18/20 | 62/62 | 95% | 38 |
| Qwen 機率門檻 | 19/20 | 43/62 | 82% | 0 |
| Qwen 第三版 | 20/20 | 15/62 | 62% | 0 |
| 一律答成功 | 0/20 | 62/62 | 50% | 0 |

- **解讀**：98% 主要是在分「夾爪放開了沒」，不是看圓柱有沒有插到底。
  1. A 版本夾爪還夾著的成功段全部被判失敗；
  2. REBOOT 的失敗都是操作員在放開前介入救援，失敗段結尾夾爪幾乎都還夾著；
  3. 唯一一段結尾已經放開的失敗段 rec59（sub_mm_misalignment，圓柱看起來立在孔裡）GPT 判成功——就是 REASSEMBLE 的瓶頸題。

## 5. 限制

- 失敗是人故意做的，而且在放開前就介入 → 測不到「看起來插好、其實沒插到底、已經放開」的難題。
- 成功段全來自專家示範、失敗段全來自救援段（不同錄製時段，背景擺設略有不同）。
- 段長 3 秒，比 REASSEMBLE 插入段（約 15 秒）短很多。
- 只有一個零件（圓柱），失敗段只有 20 段；官方標註本身有雜訊。

## 6. 要套用到插頭零件時

- **可以直接沿用**：只下載需要的相機（每種零件專家＋救援約 5.8–6.4 GB）、讀夾爪自動切成功段、做 REASSEMBLE 格式的影格與切分、跑凍結方法、計分。
- **缺**：失敗段的時間點（插頭沒有 phase.json）→ 向作者要，或人工標。
- **程式要改**：目前寫死圓柱（資料集名稱、右手、段名 reboot-cyl-），要改成依零件設定、自動判斷哪隻手。
- **同樣的限制可能會出現**：插頭的失敗如果也是放開前就救援，結果同樣只是在分「放開了沒」。

## 7. 下一階段選項（待決定）

1. **收尾**：把 REBOOT 結果寫進報告（流程可用、GPT 能認出夾爪放開與否、這批資料測不到難題、rec59 例子）。
2. **新方向**：用 A（放開前）版本，設計「只看插入深度」的新提示詞——兩類都還夾著，差別只在有沒有插到底，直接針對瓶頸。圓柱上設計 → 插頭零件測一次（要先拿到失敗時間點）。

## 8. 檔案

- 程式：`src/assembly_monitor/reboot_meta.py`、`reboot_data.py`、`reboot_seg.py`、`reboot_run.py`；`scripts/reboot_meta_check.py`、`reboot_cyl.py`、`reboot_seg.py`、`reboot_run.py`
- 設定：`configs/experiment_reboot_cyl_gpt.yaml`、`configs/experiment_reboot_cyl_v3.yaml`（從第 12 包複製，新檔）
- 測試：`tests/test_p13_meta.py`、`test_p13_cyl.py`、`test_p13_seg.py`、`test_p13_run.py`
- .bat：`run_reboot_meta.bat`、`run_reboot_cyl.bat`、`run_reboot_seg.bat`、`run_reboot_cyl_models.bat`
- 資料：`data/raw/reboot_meta/`（說明檔）、`data/raw/reboot/`（圓柱影片＋gripper.json，約 5.6 GB）、`data/derived/frames/reboot_cyl/`、`data/splits/reboot_cyl_dev/`
- 結果：`experiments/p13_reboot_meta/`、`p13_cyl_check/`、`p13_cyl_segments/`、`p13_reboot_cyl/`（report.md、per_segment.csv、inputs/）
- `requirements.txt` 加了 `pyarrow>=15`。

## 9. 沒有動到的東西

- 第 12 包以前的凍結設定、提示詞、lock.json 都沒改（REBOOT 的實驗設定是新檔）。
- 插頭零件的影格都沒看過；REASSEMBLE 保留的 round peg 4、small gear 也沒動。

---

## 工作紀錄（依時間）

建立：2026-11-04

## 為什麼

到第 12 包為止，所有結果都只來自 REASSEMBLE。REBOOT（2026-09，NIST 組裝任務板，雙臂 WidowX，
4 台相機：俯視、低角度前視、左右手腕）有插頭類零件（USB-A、USB-C、RJ45、NEMA 1-15P 電源插頭、RCA、HAN 10E），
可以檢驗方法換機器人、換實驗室、換視角後還準不準。它的失敗模式含「卡住」（插進去但沒到底），正是第 12 包的瓶頸。

- 每個任務：60 段專家示範（成功）＋60 段救援段（故意製造失敗再救回，最後都成功）
- 論文說救援段標有 t_f（失敗開始）、t_r（開始救援）、t_s（救回），但 Hugging Face 資料卡沒有寫
- 授權：資料卡 Apache-2.0（論文附錄寫 CC-BY-4.0）
- 影片 AV1 編碼；每任務約 37 GB，其中約 34 GB 是深度資料（不抓），彩色影片約 2.8 GB

## 第一步（run_reboot_meta.bat）

只下載 4 種插頭零件（USB-A、RJ45、USB-C、NEMA）專家＋救援共 9 個資料集，和官方範例 1 個的說明檔（README、meta/，約 5 MB），
檢查有沒有失敗時間點標註、這台電腦能不能解 AV1，並估算只抓彩色影片的下載量。
輸出：`data/raw/reboot_meta/`、`experiments/p13_reboot_meta/report.md`。

## 之後的判斷

- 有時間點 → 切出「失敗那一段」當失敗段、專家示範的插入結尾當成功段，原封不動套用已凍結的方法（不在 REBOOT 上調任何參數）
- 沒有時間點 → 寫信向作者要標註；同時用 USB-A 試標 10 段，太花時間或難判就放棄
- 注意：失敗是人故意做出來的，可能比自然失敗明顯，分數可能偏樂觀

## 第一步結果（2026-11-04）

- 4 種插頭零件 9 個資料集：**沒有失敗時間點**。檔案和欄位都是標準 LeRobot v3；每段固定約 30 秒（897 影格）。
  段數：USB-A 60＋60、RJ45 63＋60、USB-C 60＋63（救援段分兩個資料集）、電源插頭 61＋60。
- 只有官方範例 `sample_recovery-demonstration`（16 mm 圓柱插入，不是插頭）有 `meta/phase.json`，格式為：
  五個階段邊界 tau_0～tau_5、`failure.originating_phase`、`failure_mode`、`induced_at_frame`、`recovery_started_at_frame`、文字描述。
  - 60 段都是救援段；失敗發生在放置階段：Align(place) 14 段（都是 misalignment）、Engage(place) 13 段
    （excessive_force 5、retention_failure 3、sub_mm_misalignment 3、freeze 1、premature_release 1）
  - 標註有雜訊：58 段中 26 段的失敗影格不在它標的階段內；3 段救援開始 ≤ 失敗開始（含 360→51 的筆誤）；
    階段編號出現 0 和 21；有幾段寫「lucky success」（其實沒失敗）
- AV1 解碼：可以（PyAV 18.1.0，libdav1d）。
- 彩色影片下載量（專家＋救援）：每種插頭零件約 5.8–6.4 GB（4 台相機），只抓俯視＋兩個手腕 4 種共約 20 GB。
- `usbc_recovery_install` 的 episodes 表下載不完整（196,608 位元組，讀不開），之後重抓。

## 第二步：16 mm 圓柱當 REBOOT 的開發集（run_reboot_cyl.bat）

決定（2026-11-04）：先只做圓柱，暫不寫信給作者。圓柱有官方失敗時間點（meta/phase.json），
只拿來設計流程（怎麼切段、用哪台相機、取哪幾格）；4 種插頭零件不看影格，留到流程凍結後只測一次。

- 下載：救援段 `REBOOT26/sample_recovery-demonstration`（60 段）＋專家示範 `REBOOT26/16mm-cylinder_install`（約 62 段），
  只抓 README、meta/、俯視＋左右手腕 3 台相機的彩色影片，共約 4.6 GB（上限 6 GB，超過就停）；不抓 data/ 和低角度前視
- 檢查：每段一張拼圖（3 台相機 × 最多 8 格）。救援段取：放置階段開始、失敗開始、中間、救援前一格、救援後 1.5 秒、之後、最後一格；
  專家示範平均取 8 格。標註旗標：bad_order、lucky、bad_phase、phase_out、bad_bounds
- 輸出：`data/raw/reboot/`、`experiments/p13_cyl_check/report.md`、`experiments/p13_cyl_check/sheets/`

### 第二步結果（2026-11-05）

- 下載 4.64 GB（47 檔）全部成功；拼圖 122 張，2,898 格全部解碼，時間誤差 0 毫秒。
- 放置階段失敗 27 段，去掉 bad_order／lucky／bad_phase 後 23 段。
- 看拼圖（救援 36、39、47、49、56、57，專家 0、10）：
  - 「失敗開始／救援開始」大致合理；五個階段邊界常常錯（例：47 號 Engage(place) 標在 22 秒，那時手臂已回原位）→ 只用失敗開始、救援開始，不用階段邊界
  - 救援前一格：右手夾爪幾乎都還夾著圓柱，停在孔上方或旁邊；sub_mm_misalignment 的圓柱底端壓在孔口，最接近 REASSEMBLE 的瓶頸題
  - 圓柱都是右手：右手腕相機最清楚；俯視相機圓柱很小；左手腕只拍到右手臂側面，沒用
  - 結尾不能直接比：專家示範插好後手臂停在板子上方直到 30 秒；救援段救回後手臂馬上回原位 → 用各自最後一格會洩漏答案
  - 專家示範沒有標註，插入時間每段不同（約 13～18 秒）
- 下一步提案：讀專家示範的夾爪開合資料（只讀 data/ 裡的關節欄位，不讀深度）找「放開圓柱」那一刻；
  成功段結尾＝放開那一刻，失敗段結尾＝救援開始，兩邊都停在「插入動作剛結束、夾爪還在孔旁」。

## 第三步：找放開時刻、切段（run_reboot_seg.bat）

- 讀專家示範 data/*.parquet 的關節欄位（observation.state、action、episode_index、frame_index、timestamp），
  用部分讀取：先讀檔尾目錄、只抓這幾欄的位元組，深度不讀；單檔預估超過 20 MB 或累計超過 300 MB 就停。結果存 `gripper.json`。
- 放開時刻：選夾爪動得較大的手臂；夾住＝和結尾狀態相反、持續 ≥1 秒的最後一段；onset＝夾爪走完 10% 的那一格。
  排除：multi_hold（夾了不只一次）、late（結尾前 0.5 秒內才放開）、hold_at_start（一開始就是「夾住」，多半是誤判）。
- 段：成功段＝[onset−3 秒, onset]（專家示範）；失敗段＝[救援開始−3 秒, 救援開始]（23 段乾淨的放置階段失敗）。
- 輸出：`experiments/p13_cyl_segments/`（report.md、gripper_plots.jpg、sheets/、segments.json、releases.json）。

### 第三步第一次結果與修正（2026-11-06）

- 夾爪資料只讀了 6.5 MB（62 檔，每檔約 0.1 MB），深度沒讀。
- 第一次跑：62 段專家示範全部被標 multi_hold → 成功段 0。看 gripper_plots.jpg：右手夾爪曲線每段都一樣是
  閉合（原位）→ 張開 → 半開（夾著 16 mm 圓柱）→ 張開，程式把「原位時的閉合」也算成一次夾住。綠線（onset）位置都對。
- 修正：從第 0 格開始的那段不算夾住。用已存的 gripper.json 重算：62/62 找到放開時刻、沒有旗標；
  夾住從約 6 秒開始，放開 onset 13.8～23.6 秒（中位 18.0）。
- 失敗段檢查圖（rec56、rec40）：結尾時右手夾爪都還夾著圓柱、停在孔上方或附近，沒插進去。

### 第三步完成（2026-11-06）

- 成功段 62、失敗段 23（排除 rec35、37、43、48：筆誤或 lucky success）；檢查圖 1,020 格全部解碼。
- 看檢查圖（成功 exp00、exp20；失敗 rec36、rec47、rec53、rec56）：
  - 成功段結尾（夾爪開始放開）：圓柱被夾著、已經往下進到孔裡；+0.5 秒夾爪張開、+1.5 秒夾爪抬起，圓柱留在孔裡
  - 失敗段結尾（救援開始）：圓柱被夾著停在孔口或上方（rec36、rec56），或歪斜壓在孔旁（rec53 excessive_force）
  - 兩邊結尾同樣是「夾爪還夾著、在孔旁」，差別只在圓柱有沒有進到孔裡 → 很難，接近 REASSEMBLE 的瓶頸題
  - retention_failure（rec47、51、55）＝夾爪不放開：圓柱可能已經插好，只是沒放手 → 對「插入成不成功」來說答案不明確，主要成績排除、另外列
- 待決定：(1) 凍結的成功定義寫「夾爪張開移開後零件還在」，REASSEMBLE 的段也包含放開；
  停在放開前（同一種時刻，較難）或放開後（符合 REASSEMBLE 慣例，但失敗段夾爪還夾著，可能變成看夾爪就能分）
  (2) 要不要加抓低角度前視相機 cam_low（兩個資料集共約 0.93 GB），側面看得到插入深度；
  相機對應 hama1＝俯視、hama2＝低角度前視、hand＝動作那隻手的手腕

## 第四步：圓柱開發集 × 第 12 包凍結的方法（run_reboot_cyl_models.bat）

使用者 2026-11-06 選：成功段結尾兩種都跑（A＝放開前、B＝放開後）；多抓低角度前視 cam_low（約 0.93 GB）。

- 段：成功 A（最後一格＝夾爪開始放開）、成功 B（最後一格＝放開後 1 秒）、失敗（最後一格＝救援開始，A、B 共用）。
  每段 3 秒，段結尾＝最後一格前 0.5 秒，所以凍結取樣（uniform_n4_sb0.5_eb0.5）的窗＝[最後一格−4 秒, 最後一格]。
- 段名：reboot-cyl-expNN__seg1（A）、__seg2（B）、reboot-cyl-recNN__seg0（失敗）；切分 data/splits/reboot_cyl_dev（全部 dev）。
- 影格：data/derived/frames/reboot_cyl/uniform_n4_sb0.5_eb0.5_jpg/（和 REASSEMBLE 相同格式）；
  hama1＝cam_high、hama2＝cam_low、hand＝右手腕；640×480 原解析度。
- 實驗設定 configs/experiment_reboot_cyl_{gpt,v3}.yaml：從第 12 包 experiment_test_rpeg1_*.yaml 複製，只改名稱、切分、集合（dev）、
  影格資料夾（測試會檢查其他欄位完全相同）。零件名稱仍是 round peg。手腕固定裁切 [0.2–0.8] 照舊。
- 方法：只用 GPT、只用 Qwen 機率門檻、兩階段（門檻→GPT、第三版→GPT）、Qwen 第三版、一律答成功。
- retention_failure 3 段照跑，主要成績排除、另外列。
- 輸出：experiments/p13_reboot_cyl/（report.md、per_segment.csv、results.json、inputs/ 模型輸入拼圖）。

### 第四步結果（2026-11-06）

- 147 段全部跑完（GPT 147 次，US$1.01；Qwen 兩種各 147 段）。
- 成功段結尾＝放開前：所有方法把 62 段成功全判成失敗（成功段答對 0/62）→ 凍結的提示詞要看到「夾爪放開」才判成功，這個版本不能用。
- 成功段結尾＝放開後：GPT 失敗段答對 19/20、成功段答對 62/62（兩類平均 98%）；Qwen 機率門檻 19/20、43/62（82%）；
  第三版→GPT 98%（問 GPT 67 次）；門檻→GPT 95%（38 次）；Qwen 第三版 20/20、15/62（62%）。
- 解讀：98% 主要是在分「夾爪放開了沒」，不是在看圓柱有沒有插到底。證據：
  (1) 放開前版本，夾爪還夾著的成功段全部被判失敗；
  (2) REBOOT 的失敗都是操作員在放開前就介入救援，所以失敗段結尾夾爪幾乎都還夾著；
  (3) 唯一一段結尾已經放開的失敗段 rec59（sub_mm_misalignment，圓柱看起來立在孔裡）GPT 判成功——正是 REASSEMBLE 的瓶頸題。
- 結論：REBOOT 的格式與流程可以用（放開後版本），但它的失敗類型測不到「看起來插好但沒插到底、已經放開」這種難題；
  插頭零件如果照同樣方式測，也會遇到同樣問題（另外還缺失敗時間點）。
