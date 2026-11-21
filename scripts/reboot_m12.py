"""第 16 包：bolt 4 的最終模型直接套到 REBOOT M12 螺絲件（不重新訓練）。用法：
  python -m scripts.reboot_m12 prepare   # 下載 meta＋俯視＋手腕影片（上限 5 GB）、讀夾爪、做標註拼圖
  python -m scripts.reboot_m12 apply     # Claude 標完失敗時刻之後：取影格 → DINOv2 → bolt 4 模型 → 計分
說明：docs/package16_m12.md；輸出：experiments/p16_m12/
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import reboot_m12 as rm
from assembly_monitor.util import KeepAwake, setup_logging

log = logging.getLogger("reboot_m12")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reboot_m12", description="第 16 包：bolt 4 模型 → REBOOT M12")
    ap.add_argument("cmd", choices=["prepare", "apply"])
    ap.add_argument("--max-gb", type=float, default=rm.MAX_GB)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    with KeepAwake():
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    try:
        if args.cmd == "prepare":
            res = rm.prepare(max_gb=args.max_gb)
            ex, rc = res["kinds"]["expert"], res["kinds"]["recovery"]
            ok = sum(d["model_frame"] is not None for d in ex["episodes"].values())
            log.info("成功 %d 段（找到放開 %d）、失敗 %d 段；相機 %s；影片 %.2f GB",
                     ex["n"], ok, rc["n"], "、".join(res["cams"]), res["download"]["total_gb"])
            log.info("報告：experiments/p16_m12/prepare_report.md；拼圖：experiments/p16_m12/sheets/")
            return 0
        res = rm.apply()
        s = res["score"]
        if s:
            log.info("失敗段答對 %d/%d、成功段答對 %d/%d、兩類平均 %.1f%%、AUC %s", s["ok_failure"], s["n_failure"],
                     s["ok_success"], s["n_success"], s["bal"] * 100,
                     "–" if s["auc"] is None else f"{s['auc']:.2f}")
        log.info("報告：experiments/p16_m12/report.md（%.0f 秒）", res["seconds"])
        return 0
    except rm.Stop as e:
        log.error("停止：%s", e)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
