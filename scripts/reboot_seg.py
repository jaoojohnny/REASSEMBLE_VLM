"""第 13 包第三步：REBOOT 16 mm 圓柱——讀夾爪開合（只讀關節欄位）、找放開時刻、切段、做檢查圖。用法：
  python -m scripts.reboot_seg        # → experiments/p13_cyl_segments/report.md
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import reboot_seg as rs
from assembly_monitor.util import setup_logging

log = logging.getLogger("reboot_seg")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reboot_seg", description="REBOOT 圓柱：找放開時刻、切段")
    ap.add_argument("--dest", default="data/raw/reboot")
    ap.add_argument("--out", default="experiments/p13_cyl_segments")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    log.info("讀專家示範的夾爪資料（只讀關節欄位，不讀深度）……")
    try:
        res = rs.run(args.dest, args.out)
    except rs.TooBig as e:
        log.error("停止：%s（沒有繼續讀）", e)
        return 3
    st = res["gripper_stats"]
    ok = sum(1 for r in res["releases"].values() if r.get("ok"))
    log.info("夾爪資料 %.1f MB；找到放開時刻 %d/%d 段", st["bytes_read"] / 2**20, ok, len(res["releases"]))
    n_ok = sum(1 for s in res["segments"] if s["label"] == "success")
    log.info("成功段 %d、失敗段 %d；報告：%s/report.md", n_ok, len(res["segments"]) - n_ok, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
