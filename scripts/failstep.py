"""第 18 包：失敗在哪一步（bolt 4 主要、BNC 對照）。用法：
  python -m scripts.failstep            # 打亂答案 100 次
  python -m scripts.failstep --perm 0   # 不跑打亂答案檢定（快，但不下判定）
說明：docs/package18_failstep.md；輸出：experiments/p18_failstep/
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import failstep as fs
from assembly_monitor.util import KeepAwake, setup_logging

log = logging.getLogger("failstep")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="failstep", description="第 18 包：失敗在哪一步（步驟標記當答案，F2S2 只換答案）")
    ap.add_argument("--perm", type=int, default=fs.N_PERM, help="打亂答案的次數（預設 100）")
    ap.add_argument("--jobs", type=int, default=-2, help="打亂答案時同時跑幾個（-2＝全部核心留 1 個；1＝不平行）")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    with KeepAwake():
        try:
            res = fs.run(n_perm=args.perm, n_jobs=args.jobs)
        except fs.Stop as e:
            log.error("停止：%s", e)
            return 3
    pr = res["parts"].get(fs.PRIMARY)
    if pr:
        log.info("%s：%s", fs.PRIMARY, fs.verdict(pr))
    log.info("報告：experiments/p18_failstep/report.md（%.0f 秒）", res["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
