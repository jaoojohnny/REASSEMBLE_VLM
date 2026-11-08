"""第 12 包：兩階段（Qwen 機率門檻先判，判失敗才問 GPT）在開發集的成績與挑選。只讀已存在的輸出，不跑模型。用法：
  python -m scripts.cascade_dev                      # 讀 configs/cascade_dev.yaml → experiments/p12_cascade_dev/report.md
  python -m scripts.cascade_dev --check-run experiments/cdev_sqpeg2_gpt/gpt61_sol/f4   # 小樣本檢查：有錯就回傳 2
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor.cascade_dev import LABELS, check_runs, evaluate, load_dev_cfg, write_report
from assembly_monitor.util import setup_logging

log = logging.getLogger("cascade_dev")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cascade_dev", description="第 12 包：兩階段的開發集成績與挑選")
    ap.add_argument("--config", default="configs/cascade_dev.yaml")
    ap.add_argument("--out", default=None, help="輸出資料夾（預設設定檔的 out_dir）")
    ap.add_argument("--log", default=None)
    ap.add_argument("--check-run", nargs="+", default=None, help="只檢查這些 GPT 結果資料夾：每段都 ok 才回傳 0")
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    if args.check_run:
        bad, msgs = check_runs(args.check_run)
        for m in msgs:
            (log.error if bad else log.info)(m)
        return 2 if bad else 0
    cfg = load_dev_cfg(args.config)
    res = evaluate(cfg)
    md = write_report(res, args.out or cfg.get("out_dir", "experiments/p12_cascade_dev"))
    s = res["selection"]
    log.info("報告：%s（選出：%s；%s）", md, LABELS[s["pick"]], "通過" if s["passes"] else "沒有通過")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
