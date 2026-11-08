"""第 6 包：開發集評分。用法：
  python -m scripts.evaluate            # 所有已跑過的開發集組合 → experiments/dev_evaluation/report.md
"""

from assembly_monitor.cli import evaluate_main

if __name__ == "__main__":
    raise SystemExit(evaluate_main())
