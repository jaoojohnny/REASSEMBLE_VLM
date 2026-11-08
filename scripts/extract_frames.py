"""第 3 包影格擷取。用法：
  python -m scripts.extract_frames                     # pilot：失敗 20 + 成功 20，取樣 1/4/8 幀
  python -m scripts.extract_frames --set all           # 選定零件全部有效段
  python -m scripts.extract_frames --n-frames 8 --end-buffer 3
"""

from assembly_monitor.cli import extract_main

if __name__ == "__main__":
    raise SystemExit(extract_main())
