"""第 4 包：標註比較、一致率與爭議清單、資料集參考標籤、裁決後合併（規格 P3-4）。

來源（每個來源是一份標註檔）：
  human      使用者（主標註者，最後的 gold 以他為準）；也可以有第二位人類標註者
  ai         Claude 依同一份手冊、同一批畫面獨立標註
  reference  由資料集低階技能推得：第一個失敗的技能 → 原因（Align→misalignment、Twist→incomplete_insert…），
             失敗技能的時間 → 證據區間；成功段取最後一個技能的時間。只是啟發式參考，不是標準答案。

一致率：result、failure_reason（兩邊都判失敗的段）、best_camera 用一致比例與 Cohen's κ；證據區間用 IoU，
IoU ≥ iou_threshold 或起訖都在 endpoint_tolerance_sec 內算「命中」。主標註者與第二位標註者有任一項不一致就列為爭議，
由使用者裁決；裁決結果另存一份（review_status=adjudicated），merge_gold 依此產生 gold。
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .schema import AnnotationConfig, SegmentInfo, normalize_record, segment_key
from .util import now_iso

FINAL_STATUSES = ("submitted", "approved", "adjudicated")


# ---------------------------------------------------------------- 統計


def cohen_kappa(a: list[Any], b: list[Any]) -> float | None:
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0 - 1e-12:
        return 1.0 if po >= 1.0 - 1e-12 else 0.0
    return (po - pe) / (1 - pe)


def interval_iou(a0: float, a1: float, b0: float, b1: float, point_tol: float = 0.2) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 1e-9:
        return 1.0 if abs(a0 - b0) <= point_tol else 0.0
    return inter / union


def bootstrap_ci(items: list[Any], stat: Callable[[list[Any]], float | None], n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> list[float] | None:
    if len(items) < 2 or n_boot <= 0:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [items[rng.randrange(len(items))] for _ in items]
        v = stat(s)
        if v is not None and not math.isnan(v):
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    lo = vals[int(alpha / 2 * (len(vals) - 1))]
    hi = vals[int((1 - alpha / 2) * (len(vals) - 1))]
    return [round(lo, 4), round(hi, 4)]


def _r(v: float | None, k: int = 4) -> float | None:
    return None if v is None else round(v, k)


# ---------------------------------------------------------------- 兩兩比較


def by_segment(recs: Iterable[dict[str, Any]], statuses: tuple[str, ...] | None = FINAL_STATUSES) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in recs:
        if statuses is not None and r.get("review_status") not in statuses:
            continue
        out[str(r["segment_id"])] = r
    return out


def _has_interval(r: dict) -> bool:
    a, b = r.get("evidence_start_sec"), r.get("evidence_end_sec")
    return isinstance(a, (int, float)) and isinstance(b, (int, float))


def _too_short(r: dict[str, Any], cfg: AnnotationConfig) -> bool:
    if not cfg.evidence_min_len or r.get("annotator_type") == "reference" or r.get("result") not in cfg.evidence_required_for:
        return False
    if not _has_interval(r):
        return False
    return float(r["evidence_end_sec"]) - float(r["evidence_start_sec"]) < cfg.evidence_min_len


def compare_pair(A: dict[str, dict], B: dict[str, dict], cfg: AnnotationConfig,
                 names: tuple[str, str] = ("A", "B")) -> dict[str, Any]:
    thr = float(cfg.compare.get("iou_threshold", 0.5))
    tol = float(cfg.compare.get("endpoint_tolerance_sec", 1.0))
    n_boot = int(cfg.compare.get("bootstrap", 1000))
    seed = int(cfg.compare.get("seed", 0))
    common = sorted(set(A) & set(B))
    rows: list[dict[str, Any]] = []
    for sid in common:
        a, b = A[sid], B[sid]
        row: dict[str, Any] = {"segment_id": sid, "dataset_result": a.get("dataset_result") or b.get("dataset_result"),
                               "result_a": a.get("result"), "result_b": b.get("result")}
        row["result_agree"] = a.get("result") == b.get("result")
        disputes = [] if row["result_agree"] else ["result"]
        if a.get("result") == "failure" and b.get("result") == "failure":
            row["reason_a"], row["reason_b"] = a.get("failure_reason"), b.get("failure_reason")
            row["reason_agree"] = row["reason_a"] == row["reason_b"]
            if not row["reason_agree"]:
                disputes.append("failure_reason")
        if a.get("best_camera") and b.get("best_camera"):
            row["camera_a"], row["camera_b"] = a["best_camera"], b["best_camera"]
            row["camera_agree"] = a["best_camera"] == b["best_camera"]
            if not row["camera_agree"]:
                disputes.append("best_camera")
        if _has_interval(a) and _has_interval(b):
            a0, a1, b0, b1 = (float(a["evidence_start_sec"]), float(a["evidence_end_sec"]),
                              float(b["evidence_start_sec"]), float(b["evidence_end_sec"]))
            iou = interval_iou(a0, a1, b0, b1)
            row.update({"evidence_a": [a0, a1], "evidence_b": [b0, b1], "iou": round(iou, 4),
                        "d_start": round(b0 - a0, 3), "d_end": round(b1 - a1, 3)})
            row["evidence_hit"] = iou >= thr or (abs(b0 - a0) <= tol and abs(b1 - a1) <= tol)
            if not row["evidence_hit"]:
                disputes.append("evidence")
        elif _has_interval(a) != _has_interval(b) and a.get("result") in cfg.evidence_required_for \
                and b.get("result") in cfg.evidence_required_for and "reference" not in (a.get("annotator_type"), b.get("annotator_type")):
            disputes.append("evidence")
        vis_diff = {}
        for c in cfg.cameras:
            va = (a.get("camera_visibility") or {}).get(c)
            vb = (b.get("camera_visibility") or {}).get(c)
            if va and vb and va != vb:
                vis_diff[c] = [va, vb]
        row["visibility_diff"] = vis_diff
        row["differences"] = list(disputes)                      # 所有不同的欄位（報告用）
        row["disputes"] = [f for f in disputes if f in cfg.blocking_fields]   # 需要裁決的欄位
        short = [n for n, x in zip(names, (a, b)) if _too_short(x, cfg)]
        if short:                                                 # 證據區間沒設好（例如起點＝終點）：一定要裁決修正
            row["disputes"].append("evidence_length")
            row["differences"].append("evidence_length")
            row["evidence_too_short"] = short
        rows.append(row)

    def agree_stats(key_a: str, key_b: str, flag: str) -> dict[str, Any]:
        rs = [r for r in rows if flag in r]
        xa, xb = [r[key_a] for r in rs], [r[key_b] for r in rs]
        n = len(rs)
        ag = sum(1 for r in rs if r[flag])
        pairs = list(zip(xa, xb))
        return {"n": n, "agree": ag, "rate": _r(ag / n) if n else None, "kappa": _r(cohen_kappa(xa, xb)),
                "kappa_ci95": bootstrap_ci(pairs, lambda s: cohen_kappa([p[0] for p in s], [p[1] for p in s]), n_boot, seed),
                "rate_ci95": bootstrap_ci(pairs, lambda s: sum(1 for p in s if p[0] == p[1]) / len(s), n_boot, seed)}

    metrics: dict[str, Any] = {
        "names": list(names), "n_a": len(A), "n_b": len(B), "n_common": len(common),
        "only_a": sorted(set(A) - set(B)), "only_b": sorted(set(B) - set(A)),
        "result": agree_stats("result_a", "result_b", "result_agree"),
        "failure_reason": agree_stats("reason_a", "reason_b", "reason_agree"),
        "best_camera": agree_stats("camera_a", "camera_b", "camera_agree"),
    }
    conf: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if "reason_agree" in r:
            conf[str(r["reason_a"])][str(r["reason_b"])] += 1
    metrics["failure_reason"]["confusion"] = {k: dict(v) for k, v in sorted(conf.items())}
    ev = [r for r in rows if "iou" in r]
    ious = [r["iou"] for r in ev]
    metrics["evidence"] = {
        "n": len(ev),
        "mean_iou": _r(statistics.fmean(ious)) if ious else None,
        "median_iou": _r(statistics.median(ious)) if ious else None,
        "mean_iou_ci95": bootstrap_ci(ious, statistics.fmean, n_boot, seed),
        "hit": sum(1 for r in ev if r["evidence_hit"]),
        "hit_rate": _r(sum(1 for r in ev if r["evidence_hit"]) / len(ev)) if ev else None,
        "start_mae_s": _r(statistics.fmean(abs(r["d_start"]) for r in ev), 3) if ev else None,
        "end_mae_s": _r(statistics.fmean(abs(r["d_end"]) for r in ev), 3) if ev else None,
        "iou_threshold": thr, "endpoint_tolerance_sec": tol,
    }
    vis: dict[str, Any] = {}
    for c in cfg.cameras:
        pairs = [((A[s].get("camera_visibility") or {}).get(c), (B[s].get("camera_visibility") or {}).get(c)) for s in common]
        pairs = [p for p in pairs if p[0] and p[1]]
        vis[c] = {"n": len(pairs), "agree": sum(1 for p in pairs if p[0] == p[1]),
                  "rate": _r(sum(1 for p in pairs if p[0] == p[1]) / len(pairs)) if pairs else None}
    metrics["visibility"] = vis
    disputes = [r for r in rows if r["disputes"]]
    metrics["n_disputes"] = len(disputes)
    metrics["dispute_fields"] = dict(Counter(f for r in disputes for f in r["disputes"]))
    metrics["blocking_fields"] = list(cfg.blocking_fields)
    metrics["n_with_differences"] = sum(1 for r in rows if r["differences"])
    metrics["difference_fields"] = dict(Counter(f for r in rows for f in r["differences"]))
    return {"metrics": metrics, "rows": rows, "disputes": disputes}


# ---------------------------------------------------------------- 資料集參考標籤


def load_skills(index_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    p = Path(index_dir) / "skills.csv"
    out: dict[str, list[dict]] = defaultdict(list)
    if not p.exists():
        return out
    with p.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                out[segment_key(r["recording_id"], r["segment_id"])].append({
                    "skill_id": int(r["skill_id"]) if str(r.get("skill_id", "")).isdigit() else 0,
                    "text": (r.get("text") or "").strip().rstrip("."),
                    "start_unix": float(r["start_unix"]), "end_unix": float(r["end_unix"]),
                    "success": {"True": True, "False": False}.get(r.get("success")),
                })
            except (KeyError, ValueError):
                continue
    for v in out.values():
        v.sort(key=lambda s: (s["start_unix"], s["skill_id"]))
    return out


def skill_sequence(skills: list[dict[str, Any]]) -> str:
    return " > ".join(f"{s['text']}{'✗' if s['success'] is False else '✓'}" for s in skills)


def reference_from_lowlevel(segs: dict[str, SegmentInfo], skills: dict[str, list[dict]], cfg: AnnotationConfig) -> list[dict[str, Any]]:
    ref = cfg.reference or {}
    mapping = {str(k).lower(): v for k, v in (ref.get("skill_to_reason") or {}).items()}
    default = ref.get("no_failed_skill", "unknown")
    annotator = ref.get("annotator", "dataset_lowlevel")
    out = []
    for sid, seg in segs.items():
        sk = skills.get(sid, [])
        w0, w1 = seg.window(cfg.start_buffer, cfg.end_buffer)

        def rel(t: float) -> float:
            return round(min(max(t - seg.start_unix, w0), w1), 3)

        failed = [s for s in sk if s["success"] is False]
        reason = interval = None
        if seg.dataset_result == "failure":
            if failed:
                reason = mapping.get(failed[0]["text"].lower(), default)
                interval = (rel(failed[0]["start_unix"]), rel(failed[-1]["end_unix"]))
            else:
                reason = default
        elif seg.dataset_result == "success" and sk:
            interval = (rel(sk[-1]["start_unix"]), rel(sk[-1]["end_unix"]))
        rec = {
            "segment_id": sid, "recording_id": seg.recording_id, "segment_index": seg.segment_index, "part": seg.part,
            "dataset_result": seg.dataset_result, "result": seg.dataset_result or "uncertain",
            "failure_reason": reason,
            "evidence_start_sec": interval[0] if interval else None, "evidence_end_sec": interval[1] if interval else None,
            "annotator": annotator, "annotator_type": "reference", "label_version": cfg.label_version,
            "reason_set": cfg.reason_set, "review_status": "submitted", "created_at": now_iso(), "updated_at": now_iso(),
            "source": "rules:lowlevel-v1", "notes": f"低階技能：{skill_sequence(sk) or '（無）'}",
        }
        out.append(normalize_record(rec, cfg))
    return out


# ---------------------------------------------------------------- 報告


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.0%}"


def _num(v: float | None, k: int = 2) -> str:
    return "—" if v is None else f"{v:.{k}f}"


def _ci(ci: list[float] | None, pct: bool = False) -> str:
    if not ci:
        return ""
    return f"（95% CI {_pct(ci[0])}–{_pct(ci[1])}）" if pct else f"（95% CI {ci[0]:.2f}–{ci[1]:.2f}）"


def render_agreement_md(task: str, sources: dict[str, dict], pairs: dict[str, dict], primary: str,
                        second: str | None, reason_zh: dict[str, str]) -> str:
    lines = [f"# 標註比較報告：{task}", "", f"產生時間：{now_iso()}", "", "## 來源", "", "| 代號 | 標註者 | 類型 | 已完成段數 |", "|---|---|---|---|"]
    tname = {"human": "人", "ai": "AI（Claude）", "reference": "資料集低階技能（規則）"}
    for k, s in sources.items():
        lines.append(f"| {k} | {s['annotator']} | {tname.get(s['annotator_type'], s['annotator_type'])} | {s['n']} |")
    lines += ["", "說明：κ＝Cohen's kappa（扣除碰巧一致後的一致程度，1＝完全一致、0＝與碰巧相同）；"
              "失敗原因只比兩邊都判「失敗」的段（分母 n）；證據區間「命中」＝IoU ≥ 門檻，或起訖都在容許誤差內。",
              "需要裁決的欄位依 configs/annotation.yaml 的 compare.blocking_fields；其他欄位的差異只列入統計。"
              "證據區間短於 evidence.min_length_sec（例如起點＝終點）的段一律列入裁決（evidence_length）。", ""]
    for key, p in pairs.items():
        m = p["metrics"]
        a, b = m["names"]
        is_ref = "reference" in (sources.get(a, {}).get("annotator_type"), sources.get(b, {}).get("annotator_type"))
        if is_ref:
            head = f"與參考標籤有差異 {m.get('n_with_differences', m['n_disputes'])} 段 {m.get('difference_fields') or ''}"
        else:
            head = (f"需要裁決的爭議 {m['n_disputes']} 段 {m['dispute_fields'] or ''}；"
                    f"有任何差異 {m.get('n_with_differences', m['n_disputes'])} 段 {m.get('difference_fields') or ''}")
        lines += [f"## {a} vs {b}", "", f"共同段數 {m['n_common']}（只有 {a}：{len(m['only_a'])}、只有 {b}：{len(m['only_b'])}）；{head}", "",
                  "| 項目 | n | 一致 | 一致率 | κ |", "|---|---|---|---|---|"]
        for f, label in (("result", "結果"), ("failure_reason", "失敗原因"), ("best_camera", "最佳視角")):
            s = m[f]
            lines.append(f"| {label} | {s['n']} | {s['agree']} | {_pct(s['rate'])}{_ci(s['rate_ci95'], True)} | {_num(s['kappa'])}{_ci(s['kappa_ci95'])} |")
        e = m["evidence"]
        lines += ["", f"證據區間（n={e['n']}）：平均 IoU {_num(e['mean_iou'])}{_ci(e['mean_iou_ci95'])}、中位 {_num(e['median_iou'])}、"
                  f"命中 {e['hit']}/{e['n']}（{_pct(e['hit_rate'])}）、起點平均差 {_num(e['start_mae_s'])} 秒、終點平均差 {_num(e['end_mae_s'])} 秒", ""]
        conf = m["failure_reason"]["confusion"]
        if conf:
            cols = sorted({c for v in conf.values() for c in v})
            lines += [f"失敗原因對照（列＝{a}、欄＝{b}）", "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for r_, v in conf.items():
                lines.append(f"| {r_} | " + " | ".join(str(v.get(c, 0)) for c in cols) + " |")
            lines.append("")
        vis = m["visibility"]
        if any(v["n"] for v in vis.values()):
            lines.append("各視角可見度一致率：" + "、".join(f"{c} {_pct(v['rate'])}（n={v['n']}）" for c, v in vis.items()))
            lines.append("")
    if second:
        key = f"{primary}|{second}"
        if key in pairs:
            ds = pairs[key]["disputes"]
            lines += [f"## 爭議清單（{primary} vs {second}，共 {len(ds)} 段，需要裁決）", "",
                      "| 段 | 資料集 | 爭議 | 結果 | 原因 | 視角 | 證據區間 / IoU |", "|---|---|---|---|---|---|---|"]
            for r in ds:
                ev = f"{r.get('evidence_a')} / {r.get('evidence_b')} IoU {r.get('iou', '—')}" if "iou" in r else "—"
                lines.append(f"| {r['segment_id']} | {r['dataset_result']} | {', '.join(r['disputes'])} | {r['result_a']} / {r['result_b']} | "
                             f"{reason_zh.get(str(r.get('reason_a')), r.get('reason_a', '—'))} / {reason_zh.get(str(r.get('reason_b')), r.get('reason_b', '—'))} | "
                             f"{r.get('camera_a', '—')} / {r.get('camera_b', '—')} | {ev} |")
    return "\n".join(lines) + "\n"


def compare_sources(sources: dict[str, list[dict[str, Any]]], cfg: AnnotationConfig, out_dir: str | Path, task: str,
                    include_drafts: bool = False) -> dict[str, Any]:
    """sources＝{代號: 紀錄}，代號依序如 A（主標註者）、B（第二位）、C（參考）。輸出 agreement.json/.md、disputes.csv、compare_data.js。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    statuses = None if include_drafts else FINAL_STATUSES
    segd = {k: by_segment(v, statuses) for k, v in sources.items()}
    info = {}
    for k, recs in sources.items():
        first = next(iter(recs), {}) if recs else {}
        info[k] = {"annotator": first.get("annotator", "?"), "annotator_type": first.get("annotator_type", "human"), "n": len(segd[k])}
    keys = list(sources)
    pairs: dict[str, dict] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs[f"{keys[i]}|{keys[j]}"] = compare_pair(segd[keys[i]], segd[keys[j]], cfg, (keys[i], keys[j]))
    primary = keys[0]
    second = next((k for k in keys[1:] if info[k]["annotator_type"] != "reference"), None)
    reason_zh = {k: f"{k}（{v.get('zh', '')}）" for k, v in cfg.reasons.items()}
    md = render_agreement_md(task, info, pairs, primary, second, reason_zh)
    (out / "agreement.md").write_text(md, encoding="utf-8")
    (out / "agreement.json").write_text(json.dumps({"task": task, "generated_at": now_iso(), "sources": info,
                                                    "pairs": {k: v["metrics"] for k, v in pairs.items()},
                                                    "primary": primary, "second": second}, ensure_ascii=False, indent=1), encoding="utf-8")
    disputes = pairs[f"{primary}|{second}"]["disputes"] if second else []
    ref = next((k for k in keys if info[k]["annotator_type"] == "reference"), None)
    with (out / "disputes.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        cols = ["segment_id", "dataset_result", "disputes", "differences", "result_a", "result_b", "reason_a", "reason_b",
                "camera_a", "camera_b", "evidence_a", "evidence_b", "iou", "reference_reason", "reference_evidence"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in disputes:
            rr = segd[ref].get(r["segment_id"]) if ref else None
            w.writerow({**{c: r.get(c, "") for c in cols}, "disputes": ";".join(r["disputes"]),
                        "differences": ";".join(r.get("differences", r["disputes"])),
                        "evidence_a": r.get("evidence_a", ""), "evidence_b": r.get("evidence_b", ""),
                        "reference_reason": (rr or {}).get("failure_reason") or "",
                        "reference_evidence": [rr.get("evidence_start_sec"), rr.get("evidence_end_sec")] if rr else ""})
    data = {"task": task, "primary": primary, "second": second, "reference": ref,
            "sources": {k: {**info[k], "records": segd[k]} for k in keys},
            "disputes": {r["segment_id"]: r["disputes"] for r in disputes}}
    (out / "compare_data.js").write_text("window.ANNOT_COMPARE = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    return {"pairs": pairs, "primary": primary, "second": second, "reference": ref, "disputes": disputes,
            "agreement_md": str(out / "agreement.md"), "agreement_json": str(out / "agreement.json"),
            "disputes_csv": str(out / "disputes.csv"), "compare_js": str(out / "compare_data.js")}


# ---------------------------------------------------------------- 合併 gold


def merge_gold(primary: list[dict[str, Any]], adjudicated: list[dict[str, Any]], disputes: Iterable[str],
               cfg: AnnotationConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """gold＝主標註者的紀錄；有爭議的段以裁決紀錄取代；有爭議但還沒裁決的段標 disputed（不算正式答案）。

    正式答案（approved／adjudicated）一律蓋上目前的 label_version／reason_set（原本的版本留在
    annotated_label_version／annotated_reason_set）；沒有畫面難度的無爭議段補 visual_difficulty.default_for_approved。
    裁決紀錄沒填難度時：沿用主標註者的難度；沒有、且該段不在爭議清單，也補預設值（爭議段一定要裁決者選）。
    """
    prim = by_segment(primary)
    adj = by_segment(adjudicated, ("adjudicated", "approved", "submitted"))
    dis = set(disputes)
    gold, report = [], {"approved": 0, "adjudicated": 0, "disputed_unresolved": [], "n": 0,
                        "label_version": cfg.label_version, "reason_set": cfg.reason_set, "difficulty_defaulted": []}
    for sid in sorted(set(prim) | set(adj)):
        if sid in adj:
            r = dict(adj[sid])
            r["review_status"] = "adjudicated"
            report["adjudicated"] += 1
            if cfg.difficulty and r.get("visual_difficulty") in (None, ""):
                # 裁決頁也會匯出沒有爭議、只是看過的段；它們沒填難度時，先沿用主標註者的，
                # 再不行就和無爭議段一樣補預設值（2026-10-06 修：pilot30 有 15 段因此變成空白）
                pv = (prim.get(sid) or {}).get("visual_difficulty")
                if pv not in (None, ""):
                    r["visual_difficulty"], r["visual_difficulty_source"] = pv, "primary_annotation"
                elif sid not in dis and cfg.difficulty_default:
                    r["visual_difficulty"], r["visual_difficulty_source"] = cfg.difficulty_default, "default_for_approved"
                    report["difficulty_defaulted"].append(sid)
        elif sid in dis:
            r = dict(prim[sid])
            r["review_status"] = "disputed"
            report["disputed_unresolved"].append(sid)
        else:
            r = dict(prim[sid])
            r["review_status"] = "approved"
            report["approved"] += 1
        if r["review_status"] in ("approved", "adjudicated"):
            if r.get("label_version") != cfg.label_version or r.get("reason_set") != cfg.reason_set:
                r.setdefault("annotated_label_version", r.get("label_version"))
                r.setdefault("annotated_reason_set", r.get("reason_set"))
                r["label_version"], r["reason_set"] = cfg.label_version, cfg.reason_set
            if cfg.difficulty and r.get("visual_difficulty") in (None, "") and cfg.difficulty_default:
                if r["review_status"] == "approved":
                    r["visual_difficulty"] = cfg.difficulty_default
                    r["visual_difficulty_source"] = "default_for_approved"
                    report["difficulty_defaulted"].append(sid)
        r["updated_at"] = now_iso()
        gold.append(normalize_record(r, cfg))
    report["n"] = len(gold)
    official = [g for g in gold if g["review_status"] in ("approved", "adjudicated")]
    report["results"] = dict(Counter(g["result"] for g in official))
    report["failure_reasons"] = dict(Counter(g["failure_reason"] for g in official if g["result"] == "failure"))
    if cfg.difficulty:
        report["visual_difficulty"] = dict(Counter(str(g.get("visual_difficulty")) for g in official))
    return gold, report
