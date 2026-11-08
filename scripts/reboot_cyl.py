"""第 13 包：REBOOT 16 mm 圓柱（開發集）。用法：
  python -m scripts.reboot_cyl plan        # 只列出要下載的檔案和大小
  python -m scripts.reboot_cyl download    # 下載 README、meta/、3 台相機的彩色影片（約 4.6 GB）
  python -m scripts.reboot_cyl check       # 建索引＋每段一張拼圖 → experiments/p13_cyl_check/
  python -m scripts.reboot_cyl all         # 以上全部
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import reboot_data as rd
from assembly_monitor.util import setup_logging

log = logging.getLogger("reboot_cyl")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reboot_cyl", description="REBOOT 16 mm 圓柱（開發集）")
    ap.add_argument("cmd", choices=["plan", "download", "check", "all"])
    ap.add_argument("--dest", default="data/raw/reboot")
    ap.add_argument("--out", default="experiments/p13_cyl_check")
    ap.add_argument("--max-gb", type=float, default=rd.MAX_GB)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    if args.cmd in ("plan", "download", "all"):
        plan = rd.plan_all()
        total = sum(s for files in plan.values() for _, s in files)
        for kind, files in plan.items():
            log.info("  %s（%s）：%d 個檔案，%.2f GB", kind, rd.CYL[kind], len(files), sum(s for _, s in files) / 1e9)
        log.info("合計 %.2f GB", total / 1e9)
        if total > args.max_gb * 1e9:
            log.error("超過上限 %.1f GB，停止（沒有下載）。", args.max_gb)
            return 3
        if args.cmd == "plan":
            return 0
        ok, free = rd.disk_ok(args.dest, total)
        if not ok:
            log.error("磁碟空間不足：剩 %.1f GB，需要 %.1f GB＋2 GB。", free / 1e9, total / 1e9)
            return 3
        st = rd.download(plan, args.dest)
        log.info("下載 %d 個檔案（%.2f GB），已存在略過 %d 個，失敗 %d 個",
                 st["files"], st["bytes"] / 1e9, st["skipped"], len(st["failed"]))
        if st["failed"]:
            for f in st["failed"]:
                log.error("  %s", f)
            return 2
    if args.cmd in ("check", "all"):
        res = rd.check(args.dest, args.out)
        d = res["decode"]
        log.info("解碼 %d/%d 格；報告：%s/report.md", d["ok"], d["asked"], args.out)
        if d["ok"] < d["asked"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
