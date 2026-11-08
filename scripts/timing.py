"""第 8 包：已跑過的凍結測試各方法的執行時間（只讀輸出）。用法：
  python -m scripts.timing --final configs/final_test.yaml configs/final_test_train.yaml --out experiments/timing
"""

from assembly_monitor.cli import timing_main

if __name__ == "__main__":
    raise SystemExit(timing_main())
