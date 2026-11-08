"""第 11 包測試集：4 個零件的凍結測試合併報告（只讀輸出，不跑模型、不解鎖其他東西）。用法：
  python -m scripts.reason_test                         # → experiments/p11_final_pooled/report.md
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor.reason_eval import pool_final, write_pooled_report
from assembly_monitor.util import setup_logging

log = logging.getLogger("reason_test")
CONFIGS = ["configs/final_test_sqpeg4.yaml", "configs/final_test_sqpeg3.yaml", "configs/final_test_rpeg2.yaml",
           "configs/final_test_wproof.yaml"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reason_test", description="第 11 包測試集：4 個零件合併")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--out", default="experiments/p11_final_pooled")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    res = pool_final(args.configs)
    md = write_pooled_report(res, args.out)
    log.info("合併報告：%s（主要問題%s）", md, "成立" if res["holds"] else "不成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
