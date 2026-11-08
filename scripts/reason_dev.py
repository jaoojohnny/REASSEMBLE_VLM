"""第 11 包：推理方式在開發集（4 個新零件）的成績與挑選。只讀已存在的輸出，不跑模型。用法：
  python -m scripts.reason_dev                          # 讀 configs/reason_dev.yaml → experiments/p11_reason_dev/report.md
  python -m scripts.reason_dev --config configs/reason_dev.yaml --log logs/reason_dev.log
  python -m scripts.reason_dev --check-run experiments/rdev_sqpeg2_v3/qwen3vl_8b_rverify/f4   # 小樣本檢查：有錯就回傳 2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from assembly_monitor.reason_eval import evaluate, load_cfg, write_report
from assembly_monitor.util import setup_logging

log = logging.getLogger("reason_dev")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reason_dev", description="第 11 包：推理方式的開發集成績與挑選")
    ap.add_argument("--config", default="configs/reason_dev.yaml")
    ap.add_argument("--out", default=None, help="輸出資料夾（預設設定檔的 out_dir）")
    ap.add_argument("--log", default=None)
    ap.add_argument("--check-run", nargs="+", default=None, help="只檢查這些結果資料夾：每段都 ok 才回傳 0（正式跑之前的小樣本檢查）")
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    if args.check_run:
        return check_runs(args.check_run)
    cfg = load_cfg(args.config)
    res = evaluate(cfg)
    md = write_report(res, args.out or cfg.get("out_dir", "experiments/p11_reason_dev"))
    log.info("報告：%s（候選最好：%s；%s）", md, res["best_label"], "通過" if res["passes"] else "沒有通過")
    return 0


def check_runs(dirs: list[str]) -> int:
    bad = 0
    for d in dirs:
        rd = Path(d)
        preds = [json.loads(x) for x in (rd / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] \
            if (rd / "predictions.jsonl").exists() else []
        raws = [json.loads(x) for x in (rd / "raw.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] \
            if (rd / "raw.jsonl").exists() else []
        preds = list({p["segment_id"]: p for p in preds}.values())  # 續跑時同一段可能有多筆：取最後一筆
        raws = list({r["segment_id"]: r for r in raws}.values())
        n_ok = sum(1 for p in preds if p.get("status") == "ok")
        errs = [r.get("error_detail") for r in raws if r.get("error_detail")]
        ok = bool(preds) and n_ok == len(preds) and all("reason" in r for r in raws)
        log.info("%s：%d 段，ok %d%s", d, len(preds), n_ok, "" if ok else f"；錯誤：{errs[:2]}")
        if ok:
            r = raws[-1]["reason"]
            log.info("  例：P(yes)=%s，P(result=failure)=%s，機率質量 %s，%.1f 秒", r.get("p_yes"), r.get("p_result_failure"),
                     r.get("mass"), raws[-1].get("latency_s") or 0)
            if r.get("second"):
                log.info("  第二次觀察：P(yes)=%s，機率質量 %s", r["second"].get("p_yes"), r["second"].get("mass"))
        bad += 0 if ok else 1
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
