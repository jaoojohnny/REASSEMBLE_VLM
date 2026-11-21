"""第 14 包：每種零件各自一個小分類器（依錄影分組的巢狀交叉驗證，特徵 10 個以內）。用法：
  python -m scripts.perpart prepare   # 補擷取 15 個零件「段結束後 1 秒」的影格；算 DINOv2 特徵（全畫面；開發零件另算手腕裁切）
  python -m scripts.perpart dev       # 5 個開發零件 × 6 種做法 → 依規則選做法並凍結 → experiments/p14_perpart_dev/report.md
  python -m scripts.perpart test      # 凍結的做法套到其他 10 個零件＋依日期分組的檢查 → experiments/p14_perpart_test/report.md
保留零件 round peg 4、small gear 不碰。說明：docs/package14_perpart.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from assembly_monitor import perpart as pp
from assembly_monitor.util import KeepAwake, setup_logging

log = logging.getLogger("perpart")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perpart", description="第 14 包：每種零件各自一個小分類器")
    ap.add_argument("cmd", choices=["prepare", "dev", "test"])
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    with KeepAwake():
        if args.cmd == "prepare":
            log.info("補擷取影格（15 個零件，段結束後 1 秒，3 台相機）……")
            st = pp.extract_frames(pp.DEV_PARTS + pp.TEST_PARTS)
            log.info("影格：新擷取 %d 段、已有 %d 段、缺格 %d；錯誤 %d", st["n_done"], st["n_skipped"],
                     st["n_missing_frames"], len(st["errors"]))
            for e in st["errors"][:10]:
                log.error("  %s", e)
            encs = pp.make_encoders()
            missing_total = 0
            try:
                for part in pp.DEV_PARTS + pp.TEST_PARTS:
                    need = ("F1", "F2", "F3") if part in pp.DEV_PARTS else ("F1",)
                    segs = pp.part_segments("data/index", part)
                    _, keep, missing = pp.segment_vectors(segs, encs, Path("experiments/_features"), need=need)
                    missing_total += len(missing)
                    log.info("  特徵 %-13s %3d/%3d 段（%s）%s", part, len(keep), len(segs), "、".join(need),
                             f"；缺影格 {len(missing)} 段" if missing else "")
            finally:
                for k in ("full", "crop"):
                    encs[k].close()
            return 2 if st["errors"] else 0
        if args.cmd == "test":
            try:
                res = pp.run_test()
            except ValueError as e:
                log.error("停止：%s", e)
                return 3
            s = res["summary"]
            log.info("做法 %s：測試零件平均 依錄影 %.1f%%、依日期 %.1f%%、混合分類器 %.1f%%；比混合分類器好 %d/10 個零件",
                     res["recipe"], s["rec_mean"] * 100, s["date_mean"] * 100, s["pooled_mean"] * 100,
                     s["n_better_than_pooled"])
            log.info("開發零件：依錄影 %.1f%%、依日期 %.1f%%；報告：experiments/p14_perpart_test/report.md",
                     s["dev_rec_mean"] * 100, s["dev_date_mean"] * 100)
            return 0
        res = pp.run_dev()
        sel = res["selection"]
        log.info("選出的做法：%s（5 個開發零件平均兩類平均 %.1f%%；最高 %s %.1f%%）", sel["recipe"],
                 sel["avg"][sel["recipe"]] * 100, sel["best_recipe"], sel["best_avg"] * 100)
        log.info("凍結設定：%s；報告：experiments/p14_perpart_dev/report.md", res["selected_config"])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
