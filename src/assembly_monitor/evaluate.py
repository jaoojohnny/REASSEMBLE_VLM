"""第 6 包：評分。

任務判斷成功率＝答對的段 ÷ 全部段（無法判斷、無效、錯誤都算答錯）。一律一起報：抓到失敗（失敗召回）、成功被誤判、
兩類平均（平衡準確率）、Macro-F1、拒答率，以及依錄影重抽樣（cluster bootstrap）的 95% 信賴區間。
有人工標準答案（gold）的段另外報：依畫面難度分組、失敗原因正確率、證據區間命中。

真實答案：有正式 gold（approved／adjudicated）的段用 gold 的 result，其餘用資料集的成功／失敗標記。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from .util import now_iso

log = logging.getLogger(__name__)
CLASSES = ("success", "failure")
OFFICIAL = ("approved", "adjudicated")
DIFFICULTY_ORDER = ("clear", "subtle", "not_visible", "（未標）")


# ---------------------------------------------------------------- 讀檔


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def latest_predictions(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """predictions.jsonl 可能有續跑時追加的多筆，取每段最後一筆。"""
    out: dict[str, dict[str, Any]] = {}
    for r in _read_jsonl(Path(run_dir) / "predictions.jsonl"):
        out[str(r["segment_id"])] = r
    return out


def load_gold(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """只收正式答案（approved／adjudicated）。"""
    gold: dict[str, dict[str, Any]] = {}
    for p in paths:
        for r in _read_jsonl(Path(p)):
            if r.get("review_status") in OFFICIAL:
                gold[str(r["segment_id"])] = r
    return gold


def default_gold_paths(ann_root: str | Path = "data/annotations") -> list[str]:
    return sorted(str(p) for p in Path(ann_root).glob("*/gold.jsonl"))


# ---------------------------------------------------------------- 逐段結果


def segment_rows(preds: dict[str, dict[str, Any]], segments: list[dict[str, Any]],
                 gold: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    gold = gold or {}
    rows = []
    for s in segments:
        sid = str(s["segment_id"])
        g = gold.get(sid)
        truth = g["result"] if g and g.get("result") in CLASSES else s.get("dataset_result")
        p = preds.get(sid)
        status = (p or {}).get("status") or "missing"
        pred = p.get("result") if p and status == "ok" else None
        rows.append({"segment_id": sid, "recording_id": s.get("recording_id") or sid.split("__")[0], "truth": truth,
                     "pred": pred, "status": status, "correct": pred == truth,
                     "difficulty": (g or {}).get("visual_difficulty") or "（未標）", "has_gold": g is not None})
    return rows


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def result_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    fails = [r for r in rows if r["truth"] == "failure"]
    succs = [r for r in rows if r["truth"] == "success"]
    caught = sum(1 for r in fails if r["pred"] == "failure")
    succ_ok = sum(1 for r in succs if r["pred"] == "success")
    rec_f, rec_s = _div(caught, len(fails)), _div(succ_ok, len(succs))
    f1s = []
    for c in CLASSES:
        tp = sum(1 for r in rows if r["truth"] == c and r["pred"] == c)
        npred, ntrue = sum(1 for r in rows if r["pred"] == c), sum(1 for r in rows if r["truth"] == c)
        prec, rec = _div(tp, npred), _div(tp, ntrue)
        f1s.append(0.0 if not prec or not rec else 2 * prec * rec / (prec + rec))
    answered = [r for r in rows if r["pred"] in CLASSES]
    return {
        "n": n, "n_correct": sum(1 for r in rows if r["correct"]),
        "accuracy": _div(sum(1 for r in rows if r["correct"]), n),
        "n_failures": len(fails), "n_caught": caught, "failure_recall": rec_f,
        "n_successes": len(succs), "n_success_correct": succ_ok, "success_recall": rec_s,
        "n_false_alarms": sum(1 for r in succs if r["pred"] == "failure"),
        "balanced_accuracy": (rec_f + rec_s) / 2 if rec_f is not None and rec_s is not None else None,
        "macro_f1": sum(f1s) / len(f1s) if n else None,
        "n_uncertain": sum(1 for r in rows if r["pred"] == "uncertain"),
        "n_invalid_or_error": sum(1 for r in rows if r["status"] in ("invalid", "error", "missing")),
        "abstain_rate": _div(sum(1 for r in rows if r["pred"] == "uncertain"), n),
        "answered_accuracy": _div(sum(1 for r in answered if r["correct"]), len(answered)),
    }


CI_KEYS = ("accuracy", "failure_recall", "balanced_accuracy", "macro_f1")


def bootstrap_ci(rows: list[dict[str, Any]], keys: Iterable[str] = CI_KEYS, n_boot: int = 2000, seed: int = 0,
                 fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = result_metrics) -> dict[str, tuple[float, float] | None]:
    """依錄影重抽樣（同一支錄影的段一起抽），取 2.5／97.5 百分位。"""
    by_rec: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_rec.setdefault(r["recording_id"], []).append(r)
    recs = sorted(by_rec)
    keys = list(keys)
    if not recs:
        return {k: None for k in keys}
    rng = random.Random(seed)
    vals: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = [r for _ in recs for r in by_rec[recs[rng.randrange(len(recs))]]]
        m = fn(sample)
        for k in keys:
            if m.get(k) is not None:
                vals[k].append(float(m[k]))
    out: dict[str, tuple[float, float] | None] = {}
    for k in keys:
        v = sorted(vals[k])
        out[k] = (v[int(0.025 * (len(v) - 1))], v[int(math.ceil(0.975 * (len(v) - 1)))]) if v else None
    return out


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict[str, Any]:
    """同一批段上兩個方法的答對／答錯：只看「一個對一個錯」的段，雙尾精確二項檢定。"""
    if len(a_correct) != len(b_correct):
        raise ValueError("兩個方法的段數不同")
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p_value": p}


def by_difficulty(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for d in DIFFICULTY_ORDER:
        rs = [r for r in rows if r["difficulty"] == d]
        if rs:
            out[d] = result_metrics(rs)
    return out


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else (1.0 if a0 == b0 else 0.0)


def gold_metrics(preds: dict[str, dict[str, Any]], segment_ids: list[str], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """失敗原因與證據區間（只在有正式 gold 的段上算）。"""
    fails = [sid for sid in segment_ids if sid in gold and gold[sid].get("result") == "failure"]
    pred_fail = [sid for sid in fails if (preds.get(sid) or {}).get("status") == "ok" and preds[sid].get("result") == "failure"]
    reason_ok = sum(1 for sid in pred_fail if preds[sid].get("failure_reason") == gold[sid].get("failure_reason"))
    hits, ious, n_iv = 0, [], 0
    for sid in segment_ids:
        g, p = gold.get(sid), preds.get(sid) or {}
        if not g or p.get("status") != "ok" or p.get("result") != g.get("result"):
            continue
        vals = [g.get("evidence_start_sec"), g.get("evidence_end_sec"), p.get("evidence_start_sec"), p.get("evidence_end_sec")]
        if any(v is None for v in vals):
            continue
        n_iv += 1
        iou = _iou(float(vals[2]), float(vals[3]), float(vals[0]), float(vals[1]))
        ious.append(iou)
        hits += iou > 0
    return {"n_gold": sum(1 for sid in segment_ids if sid in gold), "n_gold_failures": len(fails),
            "n_pred_failure_on_gold_failures": len(pred_fail), "reason_correct": reason_ok,
            "reason_accuracy": _div(reason_ok, len(pred_fail)),
            "n_evidence_pairs": n_iv, "evidence_hit_rate": _div(hits, n_iv),
            "evidence_mean_iou": sum(ious) / len(ious) if ious else None}


# ---------------------------------------------------------------- 一個方法、一份報告


def evaluate_run(run_dir: str | Path, segments: list[dict[str, Any]], gold: dict[str, dict[str, Any]] | None = None,
                 n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    from .timing import run_timing

    preds = latest_predictions(run_dir)
    rows = segment_rows(preds, segments, gold)
    return {"run_dir": str(run_dir), "metrics": result_metrics(rows), "ci": bootstrap_ci(rows, n_boot=n_boot, seed=seed),
            "by_difficulty": by_difficulty(rows), "gold": gold_metrics(preds, [r["segment_id"] for r in rows], gold or {}),
            "timing": run_timing(run_dir, [r["segment_id"] for r in rows]), "rows": rows}


def describe_run(run_dir: str | Path) -> dict[str, Any]:
    """從 run_config.json 取出方法的描述（模型、幀數、提示詞、最後一張的時間）。"""
    p = Path(run_dir) / "run_config.json"
    cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    m = re.search(r"_eb([0-9.]+)_", str(cfg.get("sampling_key") or ""))
    return {"model": cfg.get("model_key") or Path(run_dir).parent.name, "n_frames": cfg.get("n_frames"),
            "prompt": cfg.get("prompt_version") or "—", "end_buffer": f"+{m.group(1)} s" if m else "–"}


def _pct(v: Any) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def _ci(t: Any) -> str:
    return "–" if not t else f"{t[0] * 100:.0f}–{t[1] * 100:.0f}%"


def _frac(a: Any, b: Any) -> str:
    return "–" if a is None or not b else f"{a}/{b}"


def write_report(results: list[dict[str, Any]], out_dir: str | Path, *, title: str, split_name: str,
                 primary: str | None = None, notes: list[str] | None = None, n_boot: int = 2000) -> dict[str, str]:
    """results：每個方法一筆 {"id", "desc", **evaluate_run(...)}。寫 report.md、metrics.csv、metrics.json、per_segment.csv。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = results[0]["metrics"]["n"] if results else 0
    n_fail = results[0]["metrics"]["n_failures"] if results else 0
    lines = [f"# {title}", "", f"產生時間：{now_iso()}　集合：{split_name}　段數：{n}（失敗 {n_fail}）", "",
             "**任務判斷成功率**＝答對段數 ÷ 全部段數（無法判斷、無效、錯誤都算答錯）。"
             "一律答成功的基準就有「成功段比例」那麼高，所以一定要一起看抓到失敗、成功被誤判與兩類平均。",
             f"括號是依錄影重抽樣 {n_boot} 次的 95% 信賴區間（失敗段集中在少數錄影時，失敗召回的區間會很寬）。", ""]
    lines += ["| 方法 | 模型 | 幀數 | 提示詞 | 最後一張 | 任務判斷成功率 | 抓到失敗 | 成功被誤判 | 兩類平均 | Macro-F1 | 無法判斷 | 無效＋錯誤 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m, ci, d = r["metrics"], r["ci"], r["desc"]
        lines.append(f"| {r['id']}{' ★' if r['id'] == primary else ''} | {d['model']} | {d['n_frames'] or '不看影格'} | {d['prompt']} | "
                     f"{d['end_buffer']} | {_frac(m['n_correct'], m['n'])}＝{_pct(m['accuracy'])}（{_ci(ci.get('accuracy'))}） | "
                     f"{_frac(m['n_caught'], m['n_failures'])}（{_ci(ci.get('failure_recall'))}） | "
                     f"{_frac(m['n_false_alarms'], m['n_successes'])} | {_pct(m['balanced_accuracy'])}（{_ci(ci.get('balanced_accuracy'))}） | "
                     f"{_pct(m['macro_f1'])} | {m['n_uncertain']} | {m['n_invalid_or_error']} |")
    if primary and any(r["id"] == primary for r in results) and len(results) > 1:
        base = next(r for r in results if r["id"] == primary)
        bc = [x["correct"] for x in base["rows"]]
        lines += ["", f"## 和主方法（{primary}）比較：McNemar 精確檢定", "",
                  "只看「一個方法對、另一個錯」的段；p < 0.05 才算明顯不同。", "",
                  "| 方法 | 只有主方法對 | 只有這個方法對 | p 值 |", "|---|---|---|---|"]
        for r in results:
            if r["id"] == primary:
                continue
            t = mcnemar_exact(bc, [x["correct"] for x in r["rows"]])
            lines.append(f"| {r['id']} | {t['a_only']} | {t['b_only']} | {t['p_value']:.3f} |")
    if any(r.get("timing") for r in results):
        from .timing import timing_lines

        lines += ["", "## 執行時間", "",
                  "每段時間＝模型從收到影格到給出答案的時間（VLM 含格式修正；API 含網路來回）；影格擷取與縮圖不算。", ""]
        lines += timing_lines([{"id": r["id"], "timing": r["timing"]} for r in results if r.get("timing")])
    n_gold = results[0]["gold"]["n_gold"] if results else 0
    lines += ["", "## 依畫面難度（有人工標準答案的段）", "", f"有正式 gold 的段：{n_gold}/{n}。", ""]
    if n_gold:
        lines += ["| 方法 | " + " | ".join(DIFFICULTY_ORDER) + " |", "|---|" + "---|" * len(DIFFICULTY_ORDER)]
        for r in results:
            cells = []
            for dd in DIFFICULTY_ORDER:
                m = r["by_difficulty"].get(dd)
                cells.append("–" if not m else f"{m['n_correct']}/{m['n']}（失敗 {_frac(m['n_caught'], m['n_failures'])}）")
            lines.append(f"| {r['id']} | " + " | ".join(cells) + " |")
        lines += ["", "## 失敗原因與證據區間（有 gold 的段）", "",
                  "原因正確率＝gold 是失敗、模型也答失敗的段中，原因相同的比例；證據命中＝結果相同、雙方都有區間時，區間有重疊的比例。", "",
                  "| 方法 | 原因正確 | 證據命中 | 平均 IoU |", "|---|---|---|---|"]
        for r in results:
            g = r["gold"]
            iou = "–" if g["evidence_mean_iou"] is None else f"{g['evidence_mean_iou']:.2f}"
            lines.append(f"| {r['id']} | {_frac(g['reason_correct'], g['n_pred_failure_on_gold_failures'])} | "
                         f"{_pct(g['evidence_hit_rate'])}（{g['n_evidence_pairs']} 段） | {iou} |")
    if notes:
        lines += ["", "## 備註", ""] + [f"- {x}" for x in notes]
    lines.append("")
    md = out / "report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    flat_cols = ["id", "model", "n_frames", "prompt", "end_buffer"] + list(results[0]["metrics"]) if results else []
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=flat_cols + [f"ci_{k}" for k in CI_KEYS])
        w.writeheader()
        for r in results:
            w.writerow({"id": r["id"], **r["desc"], **r["metrics"],
                        **{f"ci_{k}": "" if not r["ci"].get(k) else f"{r['ci'][k][0]:.3f}-{r['ci'][k][1]:.3f}" for k in CI_KEYS}})
    (out / "metrics.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in results],
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "per_segment.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "recording_id", "truth", "difficulty"] + [r["id"] for r in results])
        if results:
            for i, row in enumerate(results[0]["rows"]):
                w.writerow([row["segment_id"], row["recording_id"], row["truth"], row["difficulty"]]
                           + [r["rows"][i]["pred"] or r["rows"][i]["status"] for r in results])
    return {"md": str(md), "csv": str(out / "metrics.csv")}


# ---------------------------------------------------------------- 開發集：所有已跑過的組合


def evaluate_dev(exp_paths: list[str | Path], out_dir: str | Path = "experiments/dev_evaluation",
                 gold_paths: list[str] | None = None, n_boot: int = 2000) -> dict[str, Any]:
    from .split import load_split
    from .vlm import load_experiment

    gold = load_gold(gold_paths if gold_paths is not None else default_gold_paths())
    results, segments, seen = [], None, set()
    for p in exp_paths:
        exp = load_experiment(p)
        if "test" in exp["include"]:
            continue
        root = Path(exp["out_dir"]) / exp["name"]
        if not root.exists():
            continue
        segs = load_split(exp["split_dir"], include=exp["include"])
        segments = segments or segs
        for rc in sorted(root.glob("*/f*/run_config.json")):
            rd = rc.parent
            desc = describe_run(rd)
            key = (desc["model"], desc["n_frames"], desc["prompt"], desc["end_buffer"])
            if key in seen:
                continue
            seen.add(key)
            rid = f"{desc['model']}·f{desc['n_frames'] or 0}·{desc['prompt']}·{desc['end_buffer']}"
            results.append({"id": rid, "desc": desc, **evaluate_run(rd, segs, gold, n_boot=n_boot)})
    results.sort(key=lambda r: -(r["metrics"]["balanced_accuracy"] or 0))
    if not results:
        raise FileNotFoundError("沒有找到任何開發集結果（先跑 run_vlm_dev.bat）")
    paths = write_report(results, out_dir, title="開發集評分（所有已跑過的組合，依兩類平均排序）", split_name="dev",
                         notes=[f"開發集 {len(segments)} 段（{len({s.get('recording_id') for s in segments})} 支錄影）"
                                "：用來挑設定與檢查新做法，不當最後結果；段數少時信賴區間很寬。",
                                "挑設定時優先看兩類平均，再看成功被誤判；最後成功率以測試集為準。"], n_boot=n_boot)
    return {"results": results, **paths}
