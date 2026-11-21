"""第 17 包：M12 影格先用 OpenCV 調光線（L0 不調、L1 整批、L2 每張），再套 bolt 4 模型。用法：
  python -m scripts.m12_light
說明：docs/package17_m12_light.md；輸出：experiments/p17_m12_light/
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import m12_light as ml
from assembly_monitor.util import KeepAwake, setup_logging

log = logging.getLogger("m12_light")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="m12_light", description="第 17 包：OpenCV 調光線 → bolt 4 模型 → REBOOT M12")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    with KeepAwake():
        try:
            res = ml.run()
        except ml.Stop as e:
            log.error("停止：%s", e)
            return 3
    if not res["l0_same_as_p16"]:
        log.warning("L0 和第 16 包不一樣（最大差 %.6f），結果先不要用，告訴 Claude。", res["l0_max_diff_vs_p16"])
    log.info("報告：experiments/p17_m12_light/report.md；圖：experiments/p17_m12_light/before_after.jpg（%.0f 秒）",
             res["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
