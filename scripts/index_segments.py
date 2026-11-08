"""segment 索引與驗證。用法：
  python -m scripts.index_segments                    # 本機 data/raw
  python -m scripts.index_segments --source remote    # 全部 149 檔邊下載邊索引（可續跑）
  python -m scripts.index_segments --report-only      # 只重建 CSV 與報表
"""

from assembly_monitor.cli import index_main

if __name__ == "__main__":
    raise SystemExit(index_main())
