"""做法 A：有訓練的對照組（凍結影像編碼器＋邏輯迴歸）。用法：
  python -m scripts.probe pool          # 列出訓練池（其他零件、排除 BNC 測試錄影）
  python -m scripts.probe extract       # 擷取訓練池與 BNC 開發集的 1/4/8 幀
  python -m scripts.probe run           # 特徵 → 依錄影分組交叉驗證 → 預測 BNC 開發集
  python -m scripts.probe all           # extract + run
"""

from assembly_monitor.cli import probe_main

if __name__ == "__main__":
    raise SystemExit(probe_main())
