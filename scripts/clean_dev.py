"""第 11 包：乾淨開發集（Claude 盲評判斷和資料集標記一致的開發段）與各方法在上面的成績。只讀已存在的輸出，不跑模型。用法：
  python -m scripts.clean_dev        # → experiments/clean_dev_p11/{usable_dev_segments.json, report.md, results.json}
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor.clean_dev import build, write_outputs
from assembly_monitor.util import setup_logging

log = logging.getLogger("clean_dev")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clean_dev", description="第 11 包：乾淨開發集")
    ap.add_argument("--config", default="configs/reason_dev.yaml")
    ap.add_argument("--clean-dir", default="experiments/clean_dev_p11")
    ap.add_argument("--blind-dir", default="experiments/blind_review_dev30_v1")
    ap.add_argument("--out", default=None, help="輸出資料夾（預設 --clean-dir）")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    res = build(args.config, args.clean_dir, args.blind_dir)
    md = write_outputs(res, args.out or args.clean_dir)
    log.info("報告：%s（乾淨段 %d／%d）", md, res["agree_all"]["usable"], res["agree_all"]["n"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
