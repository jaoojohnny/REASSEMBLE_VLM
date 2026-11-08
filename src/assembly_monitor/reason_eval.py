"""第 11 包：推理方式在開發集（4 個新零件）的成績、門檻選擇與事先寫好的挑選規則。

只讀已存在的輸出（predictions.jsonl、raw.jsonl 裡存的機率），不跑模型。
- 真實答案：切分 dev 集的資料集成功／失敗標記（ref 集是看範例的範例來源，不評分）。
- 門檻 τ（「失敗分數 > τ 就判失敗」）：用「留一個零件」選——用其他 3 個零件合起來選 τ（兩類平均最高），
  在留下的零件上算成績；4 個零件輪流，合起來就是這個方法在「沒看過的零件」上的估計。
  最後給測試集用的 τ＝4 個零件合起來選的值。
- 挑選規則（2026-10-30 寫在 docs/package11_reason.md，跑之前）：候選方法中「留一個零件」的合併兩類平均最高者；
  要比第三版（寫字）高 3 個百分點以上才進測試集。
- 第二輪（2026-10-31，結合 Codex）：新增 5 新清單＋兩張裁切、6 手腕第二次觀察、7 兩次觀察合併；候選＝兩輪全部方法，規則不變。
  第二輪是看過第一輪結果之後設計、在同一批開發段上比，所以開發集成績會偏樂觀，要靠測試集確認。
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import yaml

from .evaluate import latest_predictions, mcnemar_exact
from .reason import CHECKS, decide, fail_score, mean_score
from .util import now_iso

log = logging.getLogger(__name__)

TAU_GRID = [round(1 / (1 + math.exp(-x / 4)), 6) for x in range(-24, 41)]  # logit −6 … +10，每格 0.25
MIN_GAIN = 0.03  # 要比第三版（寫字）高 3 個百分點


def load_cfg(path: str | Path) -> dict[str, Any]:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")).get("reason_dev") or {}
    if not d.get("parts"):
        raise ValueError(f"{path} 沒有 reason_dev.parts")
    return d


def _latest_raw(run_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    p = run_dir / "raw.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[str(r["segment_id"])] = r
    return out


def run_dir(exp_path: str | Path, model: str, frames: int = 4) -> Path:
    from .vlm import load_experiment

    e = load_experiment(exp_path)
    return Path(e["out_dir"]) / e["name"] / model / f"f{frames}"


def part_records(part: dict[str, Any]) -> list[dict[str, Any]]:
    """一個零件每段的真實答案與各方法的分數／答案。缺的方法記 None（算答錯）。"""
    from .split import load_split
    from .vlm import load_experiment

    e = load_experiment(part["v3"])
    segs = sorted(load_split(e["split_dir"], include=("dev",)), key=lambda r: r["segment_id"])
    v3 = latest_predictions(run_dir(part["v3"], "qwen3vl_8b_zoomfix"))
    v3raw = _latest_raw(run_dir(part["v3"], "qwen3vl_8b_zoomfix"))
    ver = _latest_raw(run_dir(part["v3"], "qwen3vl_8b_rverify"))
    few = _latest_raw(run_dir(part["v3"], "qwen3vl_8b_rfewshot"))
    con = _latest_raw(run_dir(part["contrast"], "qwen3vl_8b_rcontrast"))
    v4 = _latest_raw(run_dir(part["v4"], "qwen3vl_8b_rv4")) if part.get("v4") else {}
    out = []
    for s in segs:
        sid = s["segment_id"]
        rec: dict[str, Any] = {"segment_id": sid, "part": part["key"], "truth": s["dataset_result"]}
        p = v3.get(sid)
        rec["v3_text"] = p.get("result") if p and p.get("status") == "ok" else None
        rec["v3_text_s"] = (v3raw.get(sid) or {}).get("latency_s")
        for name, raws in (("score", ver), ("fewshot", few), ("contrast", con)):
            r = (raws.get(sid) or {}).get("reason")
            rec[f"{name}_p"] = r.get("p_yes") if r else None
            rec[f"{name}_s"] = (raws.get(sid) or {}).get("latency_s") if r else None
            if name == "score":
                rec["verify"] = r.get("result") if r else None
                rec["verify_asked"] = bool(r and r.get("verify"))
                rec["score_s1_s"] = (r.get("forward_latency_s") or [None])[0] if r else None
        r = (v4.get(sid) or {}).get("reason")
        rec["v4_p"] = r.get("p_yes") if r else None
        rec["v4w_p"] = (r.get("second") or {}).get("p_yes") if r else None
        rec["v4_s"] = (v4.get(sid) or {}).get("latency_s") if r else None
        rec["v4_s1_s"] = (r.get("forward_latency_s") or [None])[0] if r else None
        out.append(rec)
    return out


def score_of(rec: dict[str, Any], base: str) -> float | None:
    """失敗分數：一般方法＝1−Π(1−P(yes))；v4both＝兩次觀察的 logit 平均。"""
    if base == "v4both":
        a, b = rec.get("v4_p"), rec.get("v4w_p")
        return None if a is None or b is None else mean_score(fail_score(a), fail_score(b))
    p = rec.get(f"{base}_p")
    return None if p is None else fail_score(p)


def predict(rec: dict[str, Any], method: str, tau: float | None = None) -> str | None:
    """method：v3_text、verify、v4and、或 <score|contrast|fewshot|v4|v4w>_any、<…|v4both>_tau。"""
    if method in ("v3_text", "verify"):
        return rec.get(method)
    if method == "v4and":  # 兩次觀察都說失敗才算失敗（Codex 的「兩次一致」，不一致時答成功）
        a, b = rec.get("v4_p"), rec.get("v4w_p")
        return None if a is None or b is None else decide(a, "both_and", 0.5, b)
    base, rule = method.rsplit("_", 1)
    if rule == "any":
        p = rec.get(f"{base}_p")
        return None if p is None else decide(p, "any", 0.5)
    s = score_of(rec, base)
    return None if s is None else ("failure" if s > float(tau) else "success")


def metrics(recs: list[dict[str, Any]], preds: list[str | None]) -> dict[str, Any]:
    nf = sum(1 for r in recs if r["truth"] == "failure")
    ns = sum(1 for r in recs if r["truth"] == "success")
    cf = sum(1 for r, p in zip(recs, preds) if r["truth"] == "failure" and p == "failure")
    cs = sum(1 for r, p in zip(recs, preds) if r["truth"] == "success" and p == "success")
    rf, rs = (cf / nf if nf else None), (cs / ns if ns else None)
    bal = (rf + rs) / 2 if rf is not None and rs is not None else None
    return {"n": len(recs), "correct": cf + cs, "n_fail": nf, "fail_ok": cf, "n_succ": ns, "succ_ok": cs,
            "acc": (cf + cs) / len(recs) if recs else None, "bal": bal,
            "n_missing": sum(1 for p in preds if p is None)}


def best_tau(recs: list[dict[str, Any]], base: str) -> float:
    """兩類平均最高的 τ；並列時取並列區間中間那一格（結果固定、不偏向一端）。"""
    scored = [(metrics(recs, [predict(r, f"{base}_tau", t) for r in recs])["bal"] or 0.0, i) for i, t in enumerate(TAU_GRID)]
    top = max(s for s, _ in scored)
    idx = [i for s, i in scored if s >= top - 1e-12]
    return TAU_GRID[idx[len(idx) // 2]]


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    parts = cfg["parts"]
    by_part = {p["key"]: part_records(p) for p in parts}
    allrecs = [r for p in parts for r in by_part[p["key"]]]
    fixed = [("v3_text", "第三版（寫字回答）"), ("score_any", "1 機率（0.5，約等於寫字）"), ("verify", "3 答失敗再檢查一次"),
             ("contrast_any", "2 前後對照（0.5）"), ("fewshot_any", "4 看範例（0.5）")]
    tuned = [("score", "1 機率門檻"), ("contrast", "2 前後對照＋門檻"), ("fewshot", "4 看範例＋門檻")]
    if any(p.get("v4") for p in parts):  # 第二輪（結合 Codex）
        fixed += [("v4_any", "5 新清單＋兩張裁切（0.5）"), ("v4w_any", "6 手腕第二次觀察（0.5）"),
                  ("v4and", "7 兩次都說失敗才算失敗")]
        tuned += [("v4", "5 新清單＋兩張裁切＋門檻"), ("v4w", "6 手腕第二次觀察＋門檻"), ("v4both", "7 兩次觀察平均＋門檻")]
    rows: list[dict[str, Any]] = []
    per_seg: dict[str, dict[str, str | None]] = {}
    for m, label in fixed:
        preds = [predict(r, m) for r in allrecs]
        per_seg[m] = dict(zip((r["segment_id"] for r in allrecs), preds))
        rows.append({"id": m, "label": label, "pooled": metrics(allrecs, preds),
                     "parts": {k: metrics(v, [predict(r, m) for r in v]) for k, v in by_part.items()}})
    taus: dict[str, Any] = {}
    for base, label in tuned:
        lopo_preds: dict[str, str | None] = {}
        part_tau, part_m = {}, {}
        for p in parts:
            k = p["key"]
            train = [r for q in parts if q["key"] != k for r in by_part[q["key"]]]
            t = best_tau(train, base)
            part_tau[k] = t
            pr = [predict(r, f"{base}_tau", t) for r in by_part[k]]
            part_m[k] = metrics(by_part[k], pr)
            lopo_preds.update(zip((r["segment_id"] for r in by_part[k]), pr))
        preds = [lopo_preds[r["segment_id"]] for r in allrecs]
        t_all = best_tau(allrecs, base)
        taus[base] = {"lopo": part_tau, "all": t_all}
        per_seg[f"{base}_tau"] = lopo_preds
        rows.append({"id": f"{base}_tau", "label": label + "（留一個零件選 τ）", "pooled": metrics(allrecs, preds),
                     "parts": part_m, "tau_lopo": part_tau, "tau_all": t_all,
                     "in_sample": metrics(allrecs, [predict(r, f"{base}_tau", t_all) for r in allrecs])})
    base_ok = [per_seg["v3_text"][r["segment_id"]] == r["truth"] for r in allrecs]
    for row in rows:
        ok = [per_seg[row["id"]][r["segment_id"]] == r["truth"] for r in allrecs]
        row["vs_v3"] = mcnemar_exact(ok, base_ok) if row["id"] != "v3_text" else None
        ts = [r.get(_time_key(row["id"])) for r in allrecs]
        ts = [t for t in ts if t is not None]
        row["mean_s"] = sum(ts) / len(ts) if ts else None
    cands = [r for r in rows if r["id"] not in ("v3_text", "score_any")]
    best = max(cands, key=lambda r: (r["pooled"]["bal"] or 0.0, r["pooled"]["acc"] or 0.0))
    v3_bal = rows[0]["pooled"]["bal"] or 0.0
    agree = sum(1 for r in allrecs if per_seg["v3_text"][r["segment_id"]] is not None
                and per_seg["v3_text"][r["segment_id"]] == per_seg["score_any"][r["segment_id"]])
    verify_asked = sum(1 for r in allrecs if r.get("verify_asked"))
    return {"rows": rows, "taus": taus, "best": best["id"], "best_label": best["label"],
            "passes": (best["pooled"]["bal"] or 0.0) >= v3_bal + MIN_GAIN, "v3_bal": v3_bal,
            "agree_v3_score": agree, "n": len(allrecs), "verify_asked": verify_asked,
            "parts": [{"key": p["key"], "name": p["name"], **metrics(by_part[p["key"]], [r["truth"] for r in by_part[p["key"]]])}
                      for p in parts], "records": allrecs}


def _time_key(mid: str) -> str:
    if mid == "v3_text":
        return "v3_text_s"
    if mid == "verify":
        return "score_s"
    if mid.startswith("score"):
        return "score_s1_s"
    if mid.startswith("v4_"):
        return "v4_s1_s"  # 只算第一次觀察
    if mid.startswith(("v4w", "v4both", "v4and")):
        return "v4_s"     # 兩次觀察合計
    return mid.rsplit("_", 1)[0] + "_s"


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def write_report(res: dict[str, Any], out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = [p["key"] for p in res["parts"]]
    L = ["# 第 11 包：推理方式（開發集，4 個新零件）", "", f"產生時間：{now_iso()}", "",
         "真實答案＝資料集成功／失敗標記（dev 集；ref 集只當範例來源、不評分）。無效或缺的段算答錯。",
         "「門檻」方法的成績是**留一個零件**算的：τ 用其他 3 個零件選、在留下的零件上評分。", ""]
    if any(r["id"].startswith("v4") for r in res["rows"]):
        L += ["第 5–7 號是第二輪（2026-10-31，結合 Codex 的 evidence_review）：看過第一輪結果後設計、在同一批開發段上比，"
              "開發集成績偏樂觀，要靠測試集確認。", ""]
    L += [
         "| 零件 | 段數 | 失敗 | 成功 |", "|---|---|---|---|"]
    for p in res["parts"]:
        L.append(f"| {p['name']} | {p['n']} | {p['n_fail']} | {p['n_succ']} |")
    tot = res["n"]
    L += ["", f"## 合併（{tot} 段）", "",
          "| 方法 | 答對 | 失敗段答對 | 成功段答對 | 兩類平均 | 每段 | 和第三版比（只有這個對：只有第三版對） |",
          "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        m = r["pooled"]
        vs = "–" if not r["vs_v3"] else f"{r['vs_v3']['a_only']} : {r['vs_v3']['b_only']}，p＝{r['vs_v3']['p_value']:.3f}"
        sec = "–" if r["mean_s"] is None else f"{r['mean_s']:.1f} 秒"
        star = " ★" if r["id"] == res["best"] else ""
        miss = f"（缺 {m['n_missing']}）" if m["n_missing"] else ""
        L.append(f"| {r['label']}{star} | {m['correct']}/{m['n']}＝{_pct(m['acc'])}{miss} | {m['fail_ok']}/{m['n_fail']} | "
                 f"{m['succ_ok']}/{m['n_succ']} | **{_pct(m['bal'])}** | {sec} | {vs} |")
    L += ["", "## 各零件（兩類平均；答對）", "", "| 方法 | " + " | ".join(p["name"] for p in res["parts"]) + " |",
          "|---|" + "---|" * len(keys)]
    for r in res["rows"]:
        L.append(f"| {r['label']} | " + " | ".join(
            f"{_pct(r['parts'][k]['bal'])}（{r['parts'][k]['correct']}/{r['parts'][k]['n']}）" for k in keys) + " |")
    L += ["", "## 門檻 τ", "", "| 方法 | 留一個零件時各零件用的 τ | 4 個零件合起來選的 τ（給測試集） | 用 4 個零件的 τ 在開發集本身的兩類平均（偏樂觀） |",
          "|---|---|---|---|"]
    for r in res["rows"]:
        if "tau_all" in r:
            L.append(f"| {r['label']} | " + "、".join(f"{k} {v:.3f}" for k, v in r["tau_lopo"].items())
                     + f" | {r['tau_all']:.3f} | {_pct(r['in_sample']['bal'])} |")
    L += ["", "## 挑選結果（規則在跑之前寫好）", "",
          f"- 候選中兩類平均最高：**{res['best_label']}**。",
          f"- 第三版（寫字）兩類平均 {_pct(res['v3_bal'])}；規則要高 {MIN_GAIN * 100:.0f} 個百分點以上才進測試集："
          + ("**通過**。" if res["passes"] else "**沒有通過**（沒有方法明顯比第三版好，不進測試集）。"),
          f"- 檢查：機率 0.5 和第三版寫字的答案在 {res['agree_v3_score']}/{tot} 段相同。",
          f"- 答失敗再檢查一次：{res['verify_asked']}/{tot} 段有問第二階段。", ""]
    md = out / "report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({k: v for k, v in res.items() if k != "records"}, ensure_ascii=False,
                                                 indent=1), encoding="utf-8")
    with (out / "per_segment.jsonl").open("w", encoding="utf-8") as fh:
        for r in res["records"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(md)


# ---------------------------------------------------------------- 測試集：4 個零件合併（2026-10-31 凍結）


def _final_rows(cfg_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """一份凍結設定：測試段（真實答案、錄影）與每個方法的最新答案。只讀輸出。"""
    from . import final
    from .split import load_split

    cfg = final.load_final_config(cfg_path)
    segs = sorted(load_split(cfg["split_dir"], include=("test",), allow_test=True), key=lambda r: r["segment_id"])
    preds = {m["id"]: latest_predictions(final.method_run_dir(cfg, m)) for m in cfg["methods"]}
    return cfg, segs, preds


def _boot(groups: dict[str, list[int]], fn, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """以錄影為單位重抽樣（同一支錄影的段一起抽），回傳 95% 區間。"""
    import random

    rng = random.Random(seed)
    keys = sorted(groups)
    vals = []
    for _ in range(n_boot):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in groups[k]]
        v = fn(idx)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def pool_final(cfg_paths: list[str | Path], baseline: str = "qwen3vl8b_v3_eb05_f4", n_boot: int = 2000) -> dict[str, Any]:
    from . import final, timing

    parts, recs = [], []
    method_ids: list[str] | None = None
    primary = None
    times: dict[str, list[float]] = {}
    for cp in cfg_paths:
        cfg, segs, preds = _final_rows(cp)
        ids = [m["id"] for m in cfg["methods"]]
        if method_ids is None:
            method_ids, primary = ids, cfg.get("primary")
        elif ids != method_ids or cfg.get("primary") != primary:
            raise ValueError(f"{cp} 的方法清單或主方法和第一份不同")
        parts.append({"key": cfg["out_name"], "name": cfg.get("part") or cfg["out_name"], "cfg": str(cp)})
        for s in segs:
            r = {"part": cfg["out_name"], "segment_id": s["segment_id"], "recording_id": s.get("recording_id") or s["segment_id"].split("__")[0],
                 "truth": s["dataset_result"]}
            for mid in ids:
                p = preds[mid].get(s["segment_id"])
                r[mid] = p.get("result") if p and p.get("status") == "ok" else None
            recs.append(r)
        for m in cfg["methods"]:
            if m["kind"] == "majority":
                continue
            t = timing.run_timing(final.method_run_dir(cfg, m), [s["segment_id"] for s in segs])
            if t.get("mean_s") is not None and t.get("n_timed"):
                times.setdefault(m["id"], []).extend([t["mean_s"]] * t["n_timed"])
    assert method_ids is not None
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        groups.setdefault(r["recording_id"], []).append(i)

    def mt(mid: str, idx: list[int] | None = None) -> dict[str, Any]:
        rs = recs if idx is None else [recs[i] for i in idx]
        return metrics(rs, [r[mid] for r in rs])

    rows = []
    for mid in method_ids:
        m = mt(mid)
        ci_acc = _boot(groups, lambda idx, mid=mid: mt(mid, idx)["acc"], n_boot)
        ci_bal = _boot(groups, lambda idx, mid=mid: mt(mid, idx)["bal"], n_boot)
        ok = [r[mid] == r["truth"] for r in recs]
        vs_p = mcnemar_exact([r[primary] == r["truth"] for r in recs], ok) if mid != primary else None
        vs_b = mcnemar_exact(ok, [r[baseline] == r["truth"] for r in recs]) if mid != baseline else None
        ts = times.get(mid)
        rows.append({"id": mid, "pooled": m, "ci_acc": ci_acc, "ci_bal": ci_bal, "vs_primary": vs_p, "vs_baseline": vs_b,
                     "mean_s": sum(ts) / len(ts) if ts else None,
                     "parts": {p["key"]: mt(mid, [i for i, r in enumerate(recs) if r["part"] == p["key"]]) for p in parts}})
    byid = {r["id"]: r for r in rows}
    d_bal = (byid[primary]["pooled"]["bal"] or 0) - (byid[baseline]["pooled"]["bal"] or 0)
    ci_d = _boot(groups, lambda idx: (mt(primary, idx)["bal"] or 0) - (mt(baseline, idx)["bal"] or 0), n_boot)
    mc = byid[primary]["vs_baseline"]
    holds = mc["p_value"] < 0.05 and mc["a_only"] > mc["b_only"] and d_bal > 0
    return {"parts": parts, "rows": rows, "primary": primary, "baseline": baseline, "d_bal": d_bal, "ci_d_bal": ci_d,
            "holds": holds, "n": len(recs), "n_fail": sum(r["truth"] == "failure" for r in recs), "records": recs}


def write_pooled_report(res: dict[str, Any], out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {"majority": "一律答成功", "qwen3vl8b_prob_t0999": "機率門檻（τ 0.999089）",
             "qwen3vl8b_wrist_05": "手腕第二次觀察（0.5）", "qwen3vl8b_v3_eb05_f4": "Qwen＋Skill 第三版（寫字）"}
    P, B = res["primary"], res["baseline"]
    L = ["# 第 11 包測試集：4 個零件合併", "", f"產生時間：{now_iso()}", "",
         f"段數 {res['n']}（失敗 {res['n_fail']}）。真實答案＝資料集成功／失敗標記；無效或缺的段算答錯。"
         "括號是以錄影為單位重抽樣 2000 次的 95% 區間。", "",
         "| 零件 | 段數 | 失敗 |", "|---|---|---|"]
    for p in res["parts"]:
        m = res["rows"][0]["parts"][p["key"]]
        L.append(f"| {p['name']} | {m['n']} | {m['n_fail']} |")
    L += ["", "## 合併", "",
          "| 方法 | 答對 | 失敗段答對 | 成功段答對 | 兩類平均 | 每段 | 和第三版比（只有這個對：只有第三版對） | 和主方法比（只有主方法對：只有這個對） |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        m = r["pooled"]
        vb = "–" if not r["vs_baseline"] else f"{r['vs_baseline']['a_only']} : {r['vs_baseline']['b_only']}，p＝{r['vs_baseline']['p_value']:.3f}"
        vp = "–" if not r["vs_primary"] else f"{r['vs_primary']['a_only']} : {r['vs_primary']['b_only']}，p＝{r['vs_primary']['p_value']:.3f}"
        sec = "–" if r["mean_s"] is None else f"{r['mean_s']:.1f} 秒"
        star = " ★" if r["id"] == P else ""
        miss = f"（缺 {m['n_missing']}）" if m["n_missing"] else ""
        L.append(f"| {names.get(r['id'], r['id'])}{star} | {m['correct']}/{m['n']}＝{_pct(m['acc'])}（{_pct(r['ci_acc'][0])}–{_pct(r['ci_acc'][1])}）{miss} | "
                 f"{m['fail_ok']}/{m['n_fail']} | {m['succ_ok']}/{m['n_succ']} | **{_pct(m['bal'])}**（{_pct(r['ci_bal'][0])}–{_pct(r['ci_bal'][1])}） | {sec} | {vb} | {vp} |")
    L += ["", "## 各零件（兩類平均；答對）", "", "| 方法 | " + " | ".join(p["name"] for p in res["parts"]) + " |",
          "|---|" + "---|" * len(res["parts"])]
    for r in res["rows"]:
        L.append(f"| {names.get(r['id'], r['id'])} | " + " | ".join(
            f"{_pct(r['parts'][p['key']]['bal'])}（{r['parts'][p['key']]['correct']}/{r['parts'][p['key']]['n']}）" for p in res["parts"]) + " |")
    mc = {r["id"]: r for r in res["rows"]}[P]["vs_baseline"]
    L += ["", "## 事先登記的主要問題", "",
          f"- 機率門檻 vs 第三版（4 個零件合併）：McNemar {mc['a_only']} : {mc['b_only']}，p＝{mc['p_value']:.4f}；"
          f"兩類平均差 {res['d_bal'] * 100:+.1f} 個百分點（{res['ci_d_bal'][0] * 100:+.1f}～{res['ci_d_bal'][1] * 100:+.1f}）。",
          "- 規則：p < 0.05、機率門檻答對的多，而且合併兩類平均較高，才算成立。→ " + ("**成立**。" if res["holds"] else "**不成立**。"), ""]
    md = out / "report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({k: v for k, v in res.items() if k != "records"}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    with (out / "per_segment.jsonl").open("w", encoding="utf-8") as fh:
        for r in res["records"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(md)
