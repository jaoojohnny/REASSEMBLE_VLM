"""第 13 包第四步：REBOOT 16 mm 圓柱（開發集）× 第 12 包凍結的方法。用法：
  python -m scripts.reboot_run cam-low     # 下載低角度前視相機 cam_low（約 0.93 GB）
  python -m scripts.reboot_run prepare     # 影格（REASSEMBLE 格式）＋切分 data/splits/reboot_cyl_dev＋輸入拼圖
  （模型用 scripts.run_vlm 跑：configs/experiment_reboot_cyl_gpt.yaml、configs/experiment_reboot_cyl_v3.yaml）
  python -m scripts.reboot_run evaluate    # 計分 → experiments/p13_reboot_cyl/report.md
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import reboot_run as rr
from assembly_monitor.util import setup_logging

log = logging.getLogger("reboot_run")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reboot_run", description="REBOOT 圓柱 × 凍結的方法")
    ap.add_argument("cmd", choices=["cam-low", "prepare", "evaluate"])
    ap.add_argument("--dest", default="data/raw/reboot")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    if args.cmd == "cam-low":
        try:
            st = rr.download_cam_low(args.dest)
        except RuntimeError as e:
            log.error("停止：%s", e)
            return 3
        log.info("cam_low：下載 %d 個檔案（%.2f GB），已存在略過 %d 個，失敗 %d 個",
                 st["files"], st["bytes"] / 1e9, st["skipped"], len(st["failed"]))
        return 2 if st["failed"] else 0
    if args.cmd == "prepare":
        res = rr.prepare(args.dest)
        segs = res["segments"]
        log.info("段：%d（放開前成功 %d、放開後成功 %d、失敗 %d）；影格 新做 %d、沿用 %d、缺 %d 格；拼圖 %d 張",
                 len(segs), sum(s["variant"] == "A" for s in segs), sum(s["variant"] == "B" for s in segs),
                 sum(s["variant"] == "F" for s in segs), res["extract"]["done"], res["extract"]["skipped"],
                 res["extract"]["missing"], res["sheets"])
        return 2 if res["extract"]["missing"] else 0
    res = rr.evaluate()
    for v in ("A", "B"):
        g = res["variants"][v]["gpt"]
        log.info("成功段結尾＝%s：GPT 失敗段答對 %d/%d、成功段答對 %d/%d", rr.VARIANT_ZH[v],
                 g["ok_failure"], g["n_failure"], g["ok_success"], g["n_success"])
    log.info("報告：%s/report.md", rr.OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
