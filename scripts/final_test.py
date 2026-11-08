"""第 6 包：測試集最終評估（只跑一次）。用法：
  python -m scripts.final_test plan       # 列出 configs/final_test.yaml 的方法與狀態（不解鎖測試集）
  python -m scripts.final_test all        # 凍結後：擷取測試段影格 → 跑各方法 → 評分
"""

from assembly_monitor.cli import final_test_main

if __name__ == "__main__":
    raise SystemExit(final_test_main())
