"""第 20 包：補強步驟分類器（只在開發零件比三種改法）。用法：
  python -m scripts.stepcls2 dev   # 擷取外部相機影格、算 3×3 區塊與外部相機的 DINOv2 數字 → 四種改法 × 5 題 → 依規則選
說明：docs/package20_stepboost.md；輸出：experiments/p20_stepboost_dev/
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import stepcls2 as s2
from assembly_monitor.util import KeepAwake, setup_logging

log = logging.getLogger("stepcls2")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stepcls2", description="第 20 包：補強步驟分類器（開發零件）")
    ap.add_argument("cmd", choices=["dev"])
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    with KeepAwake():
        try:
            res = s2.run_dev()
        except s2.Stop as e:
            log.error("停止：%s", e)
            return 3
    s = res["summary"]
    log.info("平均：%s", "、".join(f"{c} {s['avg'][c] * 100:.0f}%（難題 {s['hard_hit'][c]}/{s['n_hard']}）" for c in s2.CANDS))
    log.info("選擇：%s", f"{s['pick']} {s2.CAND_NAME[s['pick']]}，已凍結" if s["improved"] else "保留 C0（沒有補強成功），bolt 4 不重跑")
    log.info("報告：experiments/p20_stepboost_dev/report.md（%.0f 秒）", res["seconds"])
    return 2 if res.get("extract_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
