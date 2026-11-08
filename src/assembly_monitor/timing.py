"""第 8 包：各方法的執行時間（學長要求：報告成功率之外，也報告執行時間）。

來源都是已經存在的輸出，不必重跑模型：
- VLM／API：raw.jsonl 每段的 latency_s（＋格式修正的 repair_latency_s）；模型載入時間在 run_config.json 的 metadata.load_s。
- 小分類器（做法 A）：測試段的特徵是當場算的，run_probe 會把「影像編碼＋分類器」的時間寫進 raw.jsonl；
  特徵已在快取裡的舊結果沒有時間（另外用 python -m scripts.probe time 量）。
- API：另外記 token 與估計費用（設定檔單價 × token）。
時間是「這台電腦、這次執行」的量測值；API 含網路來回，會隨時間與流量變動。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from .util import now_iso

log = logging.getLogger(__name__)


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _latest(recs: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    d: dict[str, dict[str, Any]] = {}
    for r in recs:
        if r.get("segment_id"):
            d[r["segment_id"]] = r
    return d


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def run_timing(run_dir: str | Path, segment_ids: Iterable[str] | None = None) -> dict[str, Any]:
    rd = Path(run_dir)
    raws = _latest(_read_jsonl(rd / "raw.jsonl"))
    if segment_ids is not None:
        keep = set(segment_ids)
        raws = {k: v for k, v in raws.items() if k in keep}
    rc = {}
    if (rd / "run_config.json").exists():
        try:
            rc = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
        except ValueError:
            rc = {}
    md = rc.get("metadata") or {}
    lat = [r["latency_s"] + (r.get("repair_latency_s") or 0.0) for r in raws.values() if r.get("latency_s") is not None]
    costs = [(r.get("cost_usd") or 0.0) + (r.get("repair_cost_usd") or 0.0) for r in raws.values() if r.get("cost_usd") is not None]
    loc = [r["skill"]["locate_latency_s"] for r in raws.values() if isinstance(r.get("skill"), dict) and r["skill"].get("locate_latency_s") is not None]
    mtype = (rc.get("model") or {}).get("type") or ("probe" if rc.get("encoder") else None)
    if mtype in ("anthropic", "openai"):
        where = f"{mtype} API（含網路）"
    elif mtype == "cascade":
        where = "兩階段：本機＋部分段用 API"
    elif mtype == "majority":
        where = "不需計算"
    else:
        where = md.get("gpu") or rc.get("gpu") or "本機"
    n_in = [r["input_tokens"] for r in raws.values() if r.get("input_tokens") is not None]
    esc = [bool(r["escalated"]) for r in raws.values() if r.get("escalated") is not None]
    return {
        "n_segments": len(raws), "n_timed": len(lat),
        "mean_s": round(sum(lat) / len(lat), 3) if lat else None,
        "median_s": round(_q(lat, 0.5), 3) if lat else None,
        "p90_s": round(_q(lat, 0.9), 3) if lat else None,
        "total_s": round(sum(lat), 1) if lat else None,
        "load_s": md.get("load_s") if md.get("load_s") is not None else rc.get("encoder_load_s"),
        "cost_usd": round(sum(costs), 4) if costs else None,
        "cost_per_segment_usd": round(sum(costs) / len(costs), 5) if costs else None,
        "mean_input_tokens": round(sum(n_in) / len(n_in)) if n_in else None,
        "locate_mean_s": round(sum(loc) / len(loc), 3) if loc else None,
        "n_escalated": sum(esc) if esc else None,
        "model_type": mtype, "where": where,
    }


def fmt_s(v: Any) -> str:
    if v is None:
        return "–"
    v = float(v)
    if v < 1:
        return f"{v * 1000:.0f} ms"
    if v < 120:
        return f"{v:.1f} 秒"
    return f"{v / 60:.1f} 分"


def timing_lines(rows: list[dict[str, Any]]) -> list[str]:
    """rows：{"id", "timing"}。回傳 markdown 表格的各行。"""
    lines = ["| 方法 | 在哪裡跑 | 每段平均 | 中位數 | 90% 的段在 | 全部段合計 | 模型載入（一次） | API 費用 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["timing"]
        if t.get("model_type") == "majority":
            lines.append(f"| {r['id']} | 不需計算 | – | – | – | – | – | – |")
            continue
        if not t.get("n_timed"):
            if t.get("model_type") == "probe":
                why = "沒有記錄（特徵在快取裡）"
            elif t.get("model_type") in ("anthropic", "openai"):
                why = "沒有結果（API 呼叫失敗）"
            else:
                why = "沒有時間記錄"
            lines.append(f"| {r['id']} | {t['where']} | {why} | – | – | – | {fmt_s(t.get('load_s'))} | – |")
            continue
        cost = "–" if t.get("cost_usd") is None else f"US${t['cost_usd']:.3f}（每段 US${t['cost_per_segment_usd']:.4f}）"
        extra = f"（找位置 {fmt_s(t['locate_mean_s'])}）" if t.get("locate_mean_s") is not None else ""
        if t.get("n_escalated") is not None:
            extra += f"（第二階段 {t['n_escalated']}/{t['n_segments']} 段）"
        lines.append(f"| {r['id']} | {t['where']} | {fmt_s(t['mean_s'])}{extra} | {fmt_s(t['median_s'])} | "
                     f"{fmt_s(t['p90_s'])} 以內 | {fmt_s(t['total_s'])} | {fmt_s(t['load_s'])} | {cost} |")
    return lines


def timing_report(final_cfg_paths: list[str | Path], out_dir: str | Path = "experiments/timing") -> dict[str, Any]:
    """已跑過的凍結測試（例如第 6、7 包）各方法的執行時間；只讀輸出，不解鎖、不重跑。"""
    from . import final

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# 各方法的執行時間", "", f"產生時間：{now_iso()}", "",
             "每段時間＝模型從收到影格到給出答案的時間（VLM 含格式修正；API 含網路來回）。"
             "影格擷取、縮圖不算在內。小分類器在快取裡的舊結果沒有時間，另見 `python -m scripts.probe time`。", ""]
    allrows = []
    for cp in final_cfg_paths:
        cfg = final.load_final_config(cp)
        rows = []
        for m in cfg["methods"]:
            rd = final.method_run_dir(cfg, m)
            rows.append({"id": m["id"], "timing": run_timing(rd), "run_dir": str(rd)})
        lines += [f"## {cfg.get('title', cp)}（`{cp}`）", ""] + timing_lines(rows) + [""]
        allrows.append({"config": str(cp), "rows": rows})
    md = out / "timing.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    (out / "timing.json").write_text(json.dumps(allrows, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"md": str(md), "rows": allrows}
