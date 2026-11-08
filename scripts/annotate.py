"""第 4 包標註工具。用法：
  python -m scripts.annotate build                 # pilot30：失敗 20 + 成功 10，密集影格與標註頁
  python -m scripts.annotate reference             # 由資料集低階技能產生參考標籤
  python -m scripts.annotate collect               # 從下載資料夾收標註頁匯出的檔
  python -m scripts.annotate compare               # 一致率、爭議清單、裁決頁資料
  python -m scripts.annotate merge                 # 依裁決結果合併 gold
  python -m scripts.annotate validate FILE...      # 驗證標註檔
  python -m scripts.annotate schema                # 輸出 configs/annotation.schema.json
"""

from assembly_monitor.cli import annotate_main

if __name__ == "__main__":
    raise SystemExit(annotate_main())
