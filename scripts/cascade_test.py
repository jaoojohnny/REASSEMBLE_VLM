"""第 12 包測試集：2 個零件（round peg 1、large gear）的凍結測試合併報告（只讀輸出，不跑模型）。用法：
  python -m scripts.cascade_test                       # → experiments/p12_final_pooled/report.md
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor.cascade_dev import TEST_CONFIGS, pool_test, write_test_report
from assembly_monitor.util import setup_logging

log = logging.getLogger("cascade_test")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cascade_test", description="第 12 包測試集：2 個零件合併")
    ap.add_argument("--configs", nargs="+", default=TEST_CONFIGS)
    ap.add_argument("--out", default="experiments/p12_final_pooled")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    res = pool_test(args.configs)
    md = write_test_report(res, args.out)
    log.info("合併報告：%s（主要問題%s）", md, "成立" if res["holds"] else "不成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
