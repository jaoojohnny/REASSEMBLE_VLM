"""第 19 包：每個步驟自己的小分類器（這一步成功了嗎）。用法：
  python -m scripts.stepcls dev    # 5 個開發零件的題目：擷取步驟結束後 0／0.5／1 秒的手腕影格 → DINOv2 → F2S2 → 選時間點
  python -m scripts.stepcls test   # 開發過關後：凍結的時間點套到 bolt 4 的旋轉、對準，各一次；另外算「失敗在哪一步」
說明：docs/package19_stepcls.md；輸出：experiments/p19_stepcls_dev/、experiments/p19_stepcls_test/
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import stepcls as sc
from assembly_monitor.util import KeepAwake, setup_logging

log = logging.getLogger("stepcls")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stepcls", description="第 19 包：每個步驟自己的小分類器")
    ap.add_argument("cmd", choices=["dev", "test"])
    ap.add_argument("--perm", type=int, default=sc.N_PERM, help="test：打亂答案的次數（預設 100）")
    ap.add_argument("--jobs", type=int, default=-2, help="test：打亂答案時同時跑幾個（-2＝全部核心留 1 個；1＝不平行）")
    ap.add_argument("--force", action="store_true", help="test：已經有結果也重跑（一般不用）")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    if args.cmd == "test":
        with KeepAwake():
            try:
                res = sc.run_test(n_perm=args.perm, n_jobs=args.jobs, force=args.force)
            except sc.Stop as e:
                log.error("停止：%s", e)
                return 3
        ws = res["which_step"]
        log.info("失敗在哪一步：答對 %d/%d 段（一律答最多的那一步 %d/%d）；成功段每一步都沒被判失敗 %d/%d", ws["n_right"],
                 ws["n_failed"], ws["always_most"], ws["n_failed"], ws["ok_clean"], ws["n_ok"])
        log.info("報告：experiments/p19_stepcls_test/report.md（%.0f 秒）", res["seconds"])
        return 2 if res.get("extract_errors") else 0
    with KeepAwake():
        try:
            res = sc.run_dev()
        except sc.Stop as e:
            log.error("停止：%s", e)
            return 3
    s = res["summary"]
    log.info("開發 %d 題平均：%s", s["n_tasks"], "、".join(f"結束後 {k} 秒 {v * 100:.0f}%" for k, v in s["avg"].items()))
    log.info("選結束後 %g 秒（%.0f%%）；門檻 %.0f%% → %s", s["offset"], s["selected_avg"] * 100, s["gate"] * 100,
             "過關，已凍結" if s["gate_passed"] else "沒過，不碰 bolt 4")
    log.info("報告：experiments/p19_stepcls_dev/report.md（%.0f 秒）", res["seconds"])
    return 2 if res.get("extract_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
