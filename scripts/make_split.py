"""第 4 包切分與洩漏檢查。用法：
  python -m scripts.make_split                                   # 開發 25%／測試 75%，以錄製檔分組，seed 0
  python -m scripts.make_split --ratios train=0.7,val=0.15,test=0.15 --name bnc_701515_s0
  python -m scripts.make_split --verify bnc_dev25_test75_s0      # 檢查已鎖定的切分是否被改動
"""

from assembly_monitor.cli import split_main

if __name__ == "__main__":
    raise SystemExit(split_main())
