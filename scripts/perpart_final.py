"""每零件分類器的最終模型：用凍結的做法 F2S2 在一個零件全部的段訓練一次並存檔。用法：
  python -m scripts.perpart_final --part "bolt 4"            # 訓練、存檔、從影格重算 3 段檢查
  python -m scripts.perpart_final --part "bolt 4" --check 0  # 不做影格檢查（不用 GPU）
輸出：models/perpart_final/<零件>.npz／.json、experiments/perpart_final/<零件>.md。保留零件不碰。
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import perpart_final as pf
from assembly_monitor.util import setup_logging

log = logging.getLogger("perpart_final")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perpart_final", description="每零件分類器的最終模型（F2S2）")
    ap.add_argument("--part", default="bolt 4")
    ap.add_argument("--check", type=int, default=3, help="從影格重新算特徵檢查幾段（0＝不檢查）")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    log.info("零件 %s：用全部的段訓練最終模型（凍結做法 F2S2）……", args.part)
    try:
        res = pf.train_part(args.part, check_encode=args.check)
    except ValueError as e:
        log.error("停止：%s", e)
        return 3
    m, tc = res["model"], res["train_counts"]
    log.info("訓練資料 %d 段（失敗 %d）；挑到 k=%d（留下 %d 個特徵）、C1=%s、C2=%s；4 折兩類平均 %.1f%%",
             res["n"], res["n_failure"], m["k"], len(m["features"]), m["c1"], m["c2"], m["inner_bal"] * 100)
    log.info("訓練段上（會偏高）：失敗段 %d/%d、成功段 %d/%d、兩類平均 %.1f%%", tc["ok_failure"], tc["n_failure"],
             tc["ok_success"], tc["n_success"], res["train_bal"] * 100)
    if res["cv"]:
        log.info("預期成績（第 14 包交叉驗證）：%.1f%%", res["cv"]["bal_mean"] * 100)
    log.info("存檔：%s；重新讀入預測相同：%s", "、".join(res["files"]), res["reload_same"])
    ch = res.get("check")
    if ch:
        log.info("影格重算檢查：%d 段，答案全部相同：%s，機率最大差 %s", len(ch["rows"]), ch["all_same"],
                 None if ch["max_p_diff"] is None else f"{ch['max_p_diff']:.4f}")
    log.info("報告：experiments/perpart_final/%s.md（%.0f 秒）", pf.slug(args.part), res["seconds"])
    if not res["reload_same"] or (ch and not ch["all_same"]):
        log.error("檢查沒有通過，請告訴 Claude")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
