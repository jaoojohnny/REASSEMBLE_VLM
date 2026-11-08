"""再訓練：VLM 特徵＋小分類器（只看 BNC 開發集）。用法：
  python -m scripts.vlm_probe plan      # 列出資料與已抽的特徵（不用 GPU）
  python -m scripts.vlm_probe extract   # 抽 Qwen3-VL-8B 的內部特徵（GPU，約 25 分鐘，可續跑）
  python -m scripts.vlm_probe fit       # 依錄影分組交叉驗證挑特徵 → 訓練 → 預測開發集
"""

from assembly_monitor.cli import vlm_probe_main

if __name__ == "__main__":
    raise SystemExit(vlm_probe_main())
