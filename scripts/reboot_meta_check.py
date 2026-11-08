"""第 13 包第一步：REBOOT（插頭類 4 種零件）只下載說明檔，檢查有沒有失敗時間點標註。用法：
  python -m scripts.reboot_meta_check            # → data/raw/reboot_meta/、experiments/p13_reboot_meta/report.md
  python -m scripts.reboot_meta_check --parts usb_a rj45
不下載影片、不下載深度資料（data/ 底下的 parquet）。
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import reboot_meta as rm
from assembly_monitor.util import setup_logging

log = logging.getLogger("reboot_meta_check")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reboot_meta_check", description="REBOOT 說明檔檢查")
    ap.add_argument("--parts", nargs="+", default=list(rm.PARTS), choices=list(rm.PARTS))
    ap.add_argument("--no-sample", action="store_true", help="不抓官方範例資料集")
    ap.add_argument("--dest", default="data/raw/reboot_meta")
    ap.add_argument("--out", default="experiments/p13_reboot_meta")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    repos = rm.repo_list(args.parts, sample=not args.no_sample)
    log.info("列出 %d 個資料集的檔案，只下載小檔（README、meta/）……", len(repos))
    results = [rm.analyze(r) for r in rm.fetch(repos, args.dest)]
    av1 = rm.av1_check()
    md = rm.write_report(results, av1, args.out)
    v = rm.verdict(results)
    log.info("失敗時間點標註：%s", v["zh"])
    log.info("AV1 影片解碼：%s", av1["zh"])
    log.info("報告：%s", md)
    return 2 if v["state"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
