"""第 15 包：每零件分類器（第 14 包 F2S2）＋本機 Qwen 機率門檻（τ 0.999089）。用法：
  python -m scripts.combo qwen    # 補跑機率門檻：之前沒跑過的段（BNC、USB、bolt 4、Ethernet、square peg 1、4 個開發零件的 ref 段）
  python -m scripts.combo dev     # 5 個開發零件 × 6 種方法 → 依規則選一種並凍結（configs/combo_selected.yaml）
  python -m scripts.combo test    # 凍結的方法套到 10 個測試零件＋依日期分組 → experiments/p15_combo_test/report.md
說明：docs/package15_combo.md
"""

from __future__ import annotations

import argparse
import logging

from assembly_monitor import combo as cb
from assembly_monitor.util import KeepAwake, setup_logging

log = logging.getLogger("combo")
MODELS = ["configs/model.yaml", "configs/model_p8.yaml", "configs/model_skill2.yaml", "configs/model_reason.yaml"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="combo", description="第 15 包：每零件分類器＋Qwen 機率門檻")
    ap.add_argument("cmd", choices=["qwen", "dev", "test"])
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    with KeepAwake():
        if args.cmd == "qwen":
            from assembly_monitor.mapping import load_mapping
            from assembly_monitor.schema import load_annotation_config

            st = cb.extract_for_runs()
            log.info("影格：新擷取 %d 段、已有 %d 段、錯誤 %d", st["n_done"], st["n_skipped"], len(st["errors"]))
            reasons = list(load_annotation_config("configs/annotation.yaml", load_mapping("configs/field_mapping.yaml")).reasons)
            out = cb.run_qwen(MODELS, reasons)
            log.info("Qwen 機率門檻補跑完成：" + "、".join(f"{r['key']} {r['n_segments']} 段" for r in out))
            return 0
        try:
            if args.cmd == "dev":
                res = cb.run_dev()
                sel = res["selection"]
                log.info("選出：%s %s（開發零件平均 %.1f%%；最高 %s %.1f%%）", sel["method"], cb.NAMES[sel["method"]],
                         sel["avg"][sel["method"]] * 100, sel["best_method"], sel["best_avg"] * 100)
                return 0
            res = cb.run_test()
        except ValueError as e:
            log.error("停止：%s", e)
            return 3
        s = res["summary"]
        log.info("測試零件平均：%s %.1f%% vs 每零件分類器 %.1f%%（Qwen 門檻 %.1f%%）；比分類器好 %d/10；依日期 %.1f%%",
                 res["method"], s["sel_mean"] * 100, s["m0_mean"] * 100, s["m1_mean"] * 100, s["n_better_than_m0"],
                 s["date_sel_mean"] * 100)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
