"""第 12 包：兩階段——本機 Qwen 機率門檻先判，判失敗才問 GPT——在開發集的成績與挑選。只讀已存在的輸出，不跑模型。

- 真實答案＝資料集標記。主要看第 11 包的乾淨開發集（Claude 盲評和標記一致的 195 段）；原本 241 段只當對照。
- 第一階段的機率來自第 11 包已跑好的輸出（qwen3vl_8b_rverify 的第一階段，和測試集主方法 qwen3vl_8b_rscore_t0999 同一個提示詞與影格）；
  第二階段＝GPT-6.1 Sol（ft-generic，和第 10 包兩階段的第二階段相同），在全部開發段都跑一次，才能事後算任何門檻。
- 規則（第一階段沒有答「成功」就用 GPT 的答案）和第 10 包的 kind: cascade 相同；第一階段無效也問 GPT。
- 挑選規則寫在 docs/package12_cascade.md（在看到任何 GPT 開發集結果之前寫好）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .clean_dev import FROZEN_TAU, load_judgments
from .evaluate import latest_predictions, mcnemar_exact
from .reason_eval import TAU_GRID, _latest_raw, _pct, load_cfg, metrics, part_records, run_dir, score_of
from .util import now_iso

GPT_MODEL = "gpt61_sol"
CLOSE = 0.02    # 兩類平均差 2 個百分點以內算差不多 → 選問 GPT 次數少的
MIN_GAIN = 0.03  # 選出的方法要比只用 Qwen 機率門檻高 3 個百分點以上才進測試集
CANDIDATES = ("casc_tau", "casc_frozen", "casc_v3", "gpt")
LABELS = {"majority": "一律答成功", "v3_text": "Qwen 第三版（寫字）", "qwen_frozen": "Qwen 機率門檻（τ 0.999089，第 11 包測試集主方法）",
          "gpt": "只用 GPT", "casc_v3": "兩階段：Qwen 第三版 → GPT（第 10 包規則）",
          "casc_frozen": "兩階段：Qwen 機率門檻 τ 0.999089 → GPT", "casc_tau": "兩階段：Qwen 機率門檻（τ 重選）→ GPT"}


def load_dev_cfg(path: str | Path) -> dict[str, Any]:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")).get("cascade_dev") or {}
    for k in ("reason_dev", "gpt_experiments", "clean_dir", "blind_dir"):
        if k not in d:
            raise ValueError(f"{path} 的 cascade_dev 缺少 {k}")
    rd = load_cfg(d["reason_dev"])
    keys = [p["key"] for p in rd["parts"]]
    if sorted(d["gpt_experiments"]) != sorted(keys):
        raise ValueError(f"gpt_experiments 的零件 {sorted(d['gpt_experiments'])} 和 reason_dev 的 {sorted(keys)} 不同")
    d["parts"] = [{**p, "gpt": d["gpt_experiments"][p["key"]]} for p in rd["parts"]]
    return d


def gpt_fields(exp_path: str | Path) -> dict[str, dict[str, Any]]:
    rd = run_dir(exp_path, GPT_MODEL)
    preds, raws = latest_predictions(rd), _latest_raw(rd)
    out = {}
    for sid, p in preds.items():
        r = raws.get(sid) or {}
        cost = (r.get("cost_usd") or 0.0) + (r.get("repair_cost_usd") or 0.0) if r.get("cost_usd") is not None else None
        out[sid] = {"gpt": p.get("result") if p.get("status") == "ok" else None, "gpt_s": r.get("latency_s"), "gpt_cost": cost}
    return out


def load_records(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """每個零件的開發段：真實答案、Qwen 機率與第三版、GPT 答案／時間／費用、是否在乾淨開發集。"""
    J = load_judgments(cfg["clean_dir"], cfg["blind_dir"])
    out = {}
    for p in cfg["parts"]:
        g = gpt_fields(p["gpt"])
        recs = part_records(p)
        for r in recs:
            r.update(g.get(r["segment_id"]) or {"gpt": None, "gpt_s": None, "gpt_cost": None})
            j = J.get(r["segment_id"])
            if j is None:
                raise ValueError(f"{r['segment_id']} 沒有盲評判斷（乾淨開發集不完整）")
            r["usable"] = j["judgment"] == r["truth"]
        out[p["key"]] = recs
    return out


def first_answer(rec: dict[str, Any], first: str | float) -> str | None:
    if first == "v3":
        return rec.get("v3_text")
    s = score_of(rec, "score")
    return None if s is None else ("failure" if s > float(first) else "success")


def cascade(rec: dict[str, Any], first: str | float) -> tuple[str | None, bool]:
    """回傳（答案、有沒有問 GPT）。第一階段答成功就用它；答失敗或無效就用 GPT 的答案。"""
    if first_answer(rec, first) == "success":
        return "success", False
    return rec.get("gpt"), True


def predict(rec: dict[str, Any], method: str, tau: float | None = None) -> tuple[str | None, bool]:
    if method == "majority":
        return "success", False
    if method == "v3_text":
        return rec.get("v3_text"), False
    if method == "qwen_frozen":
        return first_answer(rec, FROZEN_TAU), False
    if method == "gpt":
        return rec.get("gpt"), True
    if method == "casc_v3":
        return cascade(rec, "v3")
    if method == "casc_frozen":
        return cascade(rec, FROZEN_TAU)
    if method == "casc_tau":
        return cascade(rec, float(tau))
    raise ValueError(method)


def best_casc_tau(recs: list[dict[str, Any]]) -> float:
    """兩類平均最高的 τ；並列時取最大的 τ（問 GPT 最少）。"""
    scored = [(metrics(recs, [cascade(r, t)[0] for r in recs])["bal"] or 0.0, t) for t in TAU_GRID]
    top = max(s for s, _ in scored)
    return max(t for s, t in scored if s >= top - 1e-12)


def _cost(recs: list[dict[str, Any]], asked: list[bool]) -> dict[str, Any]:
    n_ask = sum(asked)
    costs = [r["gpt_cost"] for r, a in zip(recs, asked) if a and r.get("gpt_cost") is not None]
    return {"n_gpt": n_ask, "gpt_frac": n_ask / len(recs) if recs else None, "cost_usd": sum(costs) if costs else 0.0}


def score_set(by_part: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """一批段（乾淨或全部）上各方法的成績。casc_tau 用「留一個零件」選 τ（其他 3 個零件選、留下的評分）。"""
    keys = list(by_part)
    allrecs = [r for k in keys for r in by_part[k]]
    rows: dict[str, dict[str, Any]] = {}
    for m in ("majority", "v3_text", "qwen_frozen", "gpt", "casc_v3", "casc_frozen", "casc_tau"):
        pred: dict[str, tuple[str | None, bool]] = {}
        extra: dict[str, Any] = {}
        if m == "casc_tau":
            lopo = {}
            for k in keys:
                train = [r for q in keys if q != k for r in by_part[q]]
                lopo[k] = best_casc_tau(train)
                pred.update({r["segment_id"]: predict(r, m, lopo[k]) for r in by_part[k]})
            t_all = best_casc_tau(allrecs)
            fixed = [predict(r, m, t_all) for r in allrecs]
            extra = {"tau_lopo": lopo, "tau_all": t_all, "in_sample": {**metrics(allrecs, [p for p, _ in fixed]),
                                                                         **_cost(allrecs, [a for _, a in fixed])}}
        else:
            pred = {r["segment_id"]: predict(r, m) for r in allrecs}
        ans = [pred[r["segment_id"]][0] for r in allrecs]
        asked = [pred[r["segment_id"]][1] for r in allrecs]
        rows[m] = {"id": m, "label": LABELS[m], "pred": {k: v[0] for k, v in pred.items()},
                   "pooled": {**metrics(allrecs, ans), **_cost(allrecs, asked)},
                   "parts": {k: metrics(v, [pred[r["segment_id"]][0] for r in v]) for k, v in by_part.items()}, **extra}
    base = [rows["qwen_frozen"]["pred"][r["segment_id"]] == r["truth"] for r in allrecs]
    for m, row in rows.items():
        ok = [row["pred"][r["segment_id"]] == r["truth"] for r in allrecs]
        row["vs_qwen"] = mcnemar_exact(ok, base) if m != "qwen_frozen" else None
    return {"rows": rows, "n": len(allrecs), "n_fail": sum(r["truth"] == "failure" for r in allrecs),
            "n_missing_gpt": sum(r.get("gpt") is None for r in allrecs)}


def select(clean: dict[str, Any]) -> dict[str, Any]:
    """事先寫好的挑選規則（docs/package12_cascade.md）：候選中乾淨開發集兩類平均最高者；
    和最高差 2 個百分點以內的候選，選問 GPT 次數最少的；要比只用 Qwen 機率門檻高 3 個百分點以上才進測試集。"""
    rows = clean["rows"]
    bal = {m: rows[m]["pooled"]["bal"] or 0.0 for m in CANDIDATES}
    top = max(bal.values())
    near = [m for m in CANDIDATES if bal[m] >= top - CLOSE - 1e-12]
    pick = min(near, key=lambda m: (rows[m]["pooled"]["n_gpt"], -bal[m], CANDIDATES.index(m)))
    qbal = rows["qwen_frozen"]["pooled"]["bal"] or 0.0
    return {"pick": pick, "near": near, "top_bal": top, "pick_bal": bal[pick], "qwen_bal": qbal,
            "passes": bal[pick] >= qbal + MIN_GAIN, "test_tau": rows["casc_tau"]["tau_all"] if pick == "casc_tau" else None}


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    full = load_records(cfg)
    clean = {k: [r for r in v if r["usable"]] for k, v in full.items()}
    res_clean, res_full = score_set(clean), score_set(full)
    return {"parts": [{"key": p["key"], "name": p["name"]} for p in cfg["parts"]], "clean": res_clean, "full": res_full,
            "selection": select(res_clean), "records": [r for v in full.values() for r in v]}


def _row_md(r: dict[str, Any], star: bool) -> str:
    m = r["pooled"]
    vs = "–" if not r["vs_qwen"] else f"{r['vs_qwen']['a_only']} : {r['vs_qwen']['b_only']}，p＝{r['vs_qwen']['p_value']:.3f}"
    miss = f"（缺 {m['n_missing']}）" if m["n_missing"] else ""
    return (f"| {r['label']}{' ★' if star else ''} | {m['correct']}/{m['n']}＝{_pct(m['acc'])}{miss} | {m['fail_ok']}/{m['n_fail']} | "
            f"{m['succ_ok']}/{m['n_succ']} | **{_pct(m['bal'])}** | {m['n_gpt']}/{m['n']}（{_pct(m['gpt_frac'])}） | US${m['cost_usd']:.3f} | {vs} |")


def write_report(res: dict[str, Any], out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {p["key"]: p["name"] for p in res["parts"]}
    sel, C, F = res["selection"], res["clean"], res["full"]
    L = ["# 第 12 包：兩階段（Qwen 機率門檻先判，判失敗才問 GPT）——開發集", "", f"產生時間：{now_iso()}", "",
         "真實答案＝資料集標記。主要看乾淨開發集（Claude 盲評和標記一致的段）；原本全部開發段只當對照。無效或缺的段算答錯。",
         "「τ 重選」的成績是**留一個零件**算的：τ 用其他 3 個零件選、在留下的零件評分。問 GPT 的費用＝有問的段的 GPT 實際費用加總。", ""]
    if C["n_missing_gpt"] or F["n_missing_gpt"]:
        L += [f"**注意：GPT 缺 {F['n_missing_gpt']} 段（無效或沒跑），算答錯。**", ""]
    for title, R in ((f"乾淨開發集（{C['n']} 段，失敗 {C['n_fail']}）", C), (f"原本全部開發段（{F['n']} 段，失敗 {F['n_fail']}；對照）", F)):
        L += [f"## {title}", "", "| 方法 | 答對 | 失敗段答對 | 成功段答對 | 兩類平均 | 問 GPT | GPT 費用 | 和 Qwen 機率門檻比（只有這個對：只有 Qwen 對） |",
              "|---|---|---|---|---|---|---|---|"]
        L += [_row_md(r, R is C and r["id"] == sel["pick"]) for r in R["rows"].values()]
        t = R["rows"]["casc_tau"]
        L += ["", f"τ 重選：留一個零件時各零件用的 τ " + "、".join(f"{names[k]} {v:.6f}" for k, v in t["tau_lopo"].items())
              + f"；全部零件合起來選的 τ＝{t['tau_all']:.6f}（用它在這批段本身：兩類平均 {_pct(t['in_sample']['bal'])}、問 GPT "
              f"{t['in_sample']['n_gpt']} 段；偏樂觀）。", "",
              "各零件（兩類平均；答對）：", "", "| 方法 | " + " | ".join(names.values()) + " |", "|---|" + "---|" * len(names)]
        for r in R["rows"].values():
            L.append(f"| {r['label']} | " + " | ".join(
                f"{_pct(r['parts'][k]['bal'])}（{r['parts'][k]['correct']}/{r['parts'][k]['n']}）" for k in names) + " |")
        L.append("")
    L += ["## 挑選結果（規則在看到 GPT 開發集結果之前寫好）", "",
          f"- 候選：{'、'.join(LABELS[m] for m in CANDIDATES)}。乾淨開發集兩類平均最高 {_pct(sel['top_bal'])}；"
          f"差 2 個百分點以內：{'、'.join(LABELS[m] for m in sel['near'])} → 選問 GPT 最少的：**{LABELS[sel['pick']]}**"
          f"（{_pct(sel['pick_bal'])}）。",
          f"- 只用 Qwen 機率門檻 {_pct(sel['qwen_bal'])}；規則要高 {MIN_GAIN * 100:.0f} 個百分點以上才進測試集："
          + ("**通過**。" if sel["passes"] else "**沒有通過**（不進測試集）。")]
    if sel["test_tau"] is not None:
        L.append(f"- 測試集用的 τ＝{sel['test_tau']:.6f}（乾淨開發集 4 個零件合起來選；並列取最大，問 GPT 最少）。")
    L += ["- 乾淨開發集偏向「看得出結果」的段，強模型（GPT）在這裡會比在全部段好；成績不能和測試集直接比。", ""]
    md = out / "report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    slim = {"parts": res["parts"], "selection": sel}
    for k in ("clean", "full"):
        slim[k] = {**{x: v for x, v in res[k].items() if x != "rows"},
                   "rows": {m: {x: v for x, v in r.items() if x != "pred"} for m, r in res[k]["rows"].items()}}
    (out / "results.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "per_segment.jsonl").open("w", encoding="utf-8") as fh:
        for r in res["records"]:
            fh.write(json.dumps({k: r.get(k) for k in ("segment_id", "part", "truth", "usable", "v3_text", "score_p", "gpt", "gpt_s",
                                                        "gpt_cost")}, ensure_ascii=False) + "\n")
    return str(md)


def check_runs(dirs: list[str]) -> tuple[int, list[str]]:
    """小樣本檢查（API 模型）：每段都 ok 才算過。回傳（錯的資料夾數、訊息）。"""
    bad, msgs = 0, []
    for d in dirs:
        rd = Path(d)
        preds = latest_predictions(rd)
        raws = _latest_raw(rd)
        n_ok = sum(1 for p in preds.values() if p.get("status") == "ok")
        errs = [r.get("error_detail") for r in raws.values() if r.get("error_detail")]
        cost = sum((r.get("cost_usd") or 0.0) for r in raws.values())
        ok = bool(preds) and n_ok == len(preds)
        msgs.append(f"{d}：{len(preds)} 段，ok {n_ok}，費用 US${cost:.4f}" + ("" if ok else f"；錯誤：{errs[:2]}"))
        bad += 0 if ok else 1
    return bad, msgs


# ---------------------------------------------------------------- 測試集：2 個零件合併（2026-11-02 凍結）

TEST_CONFIGS = ["configs/final_test_rpeg1.yaml", "configs/final_test_lgear.yaml"]
TEST_BASELINE = "qwen3vl8b_prob_t0999"
TEST_NAMES = {"majority": "一律答成功", "gpt61_sol_ft_eb05_f4": "只用 GPT", "qwen3vl8b_prob_t0999": "Qwen 機率門檻（τ 0.999089）",
              "hybrid_prob_gpt": "兩階段：Qwen 機率門檻 → GPT（看過開發集後加入）", "hybrid_v3_gpt": "兩階段：Qwen 第三版 → GPT（第 10 包規則）",
              "qwen3vl8b_v3_eb05_f4": "Qwen＋Skill 第三版（寫字）"}


def gpt_usage(cfg_paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    """每個方法在測試段問了幾段 GPT、花多少錢（只讀 raw.jsonl）。兩階段看 escalated；只用 GPT＝每段都問。"""
    from . import final

    usage: dict[str, dict[str, Any]] = {}
    for cp in cfg_paths:
        cfg = final.load_final_config(cp)
        for m in cfg["methods"]:
            if m["kind"] == "majority":
                continue
            u = usage.setdefault(m["id"], {"n": 0, "n_gpt": 0, "cost_usd": 0.0})
            for r in _latest_raw(final.method_run_dir(cfg, m)).values():
                u["n"] += 1
                u["n_gpt"] += int(bool(r.get("escalated")) if m["kind"] == "cascade" else m.get("model") == GPT_MODEL)
                u["cost_usd"] += (r.get("cost_usd") or 0.0) + (r.get("repair_cost_usd") or 0.0)
    return usage


def pool_test(cfg_paths: list[str | Path] | None = None, n_boot: int = 2000) -> dict[str, Any]:
    from .reason_eval import pool_final

    paths = list(cfg_paths or TEST_CONFIGS)
    res = pool_final(paths, baseline=TEST_BASELINE, n_boot=n_boot)
    res["usage"] = gpt_usage(paths)
    return res


def write_test_report(res: dict[str, Any], out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P, B = res["primary"], res["baseline"]
    nm = lambda m: TEST_NAMES.get(m, m)  # noqa: E731
    L = ["# 第 12 包測試集：2 個零件合併（只用 GPT／兩階段）", "", f"產生時間：{now_iso()}", "",
         f"段數 {res['n']}（失敗 {res['n_fail']}）。真實答案＝資料集成功／失敗標記；無效或缺的段算答錯。括號是以錄影為單位重抽樣的 95% 區間。", "",
         "| 零件 | 段數 | 失敗 |", "|---|---|---|"]
    for p in res["parts"]:
        m = res["rows"][0]["parts"][p["key"]]
        L.append(f"| {p['name']} | {m['n']} | {m['n_fail']} |")
    L += ["", "## 合併", "",
          f"| 方法 | 答對 | 失敗段答對 | 成功段答對 | 兩類平均 | 問 GPT | GPT 費用 | 每段 | 和 Qwen 機率門檻比（只有這個對：只有 Qwen 對） |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        m, u = r["pooled"], res.get("usage", {}).get(r["id"])
        vb = "–" if not r["vs_baseline"] else f"{r['vs_baseline']['a_only']} : {r['vs_baseline']['b_only']}，p＝{r['vs_baseline']['p_value']:.3f}"
        sec = "–" if r["mean_s"] is None else f"{r['mean_s']:.1f} 秒"
        gq = "–" if not u else f"{u['n_gpt']}/{u['n']}"
        cost = "–" if not u else f"US${u['cost_usd']:.3f}"
        miss = f"（缺 {m['n_missing']}）" if m["n_missing"] else ""
        L.append(f"| {nm(r['id'])}{' ★' if r['id'] == P else ''} | {m['correct']}/{m['n']}＝{_pct(m['acc'])}（{_pct(r['ci_acc'][0])}–{_pct(r['ci_acc'][1])}）{miss} | "
                 f"{m['fail_ok']}/{m['n_fail']} | {m['succ_ok']}/{m['n_succ']} | **{_pct(m['bal'])}**（{_pct(r['ci_bal'][0])}–{_pct(r['ci_bal'][1])}） | "
                 f"{gq} | {cost} | {sec} | {vb} |")
    L += ["", "## 各零件（失敗段答對、成功段答對；兩類平均）", "", "| 方法 | " + " | ".join(p["name"] for p in res["parts"]) + " |",
          "|---|" + "---|" * len(res["parts"])]
    for r in res["rows"]:
        L.append(f"| {nm(r['id'])} | " + " | ".join(
            f"{r['parts'][p['key']]['fail_ok']}/{r['parts'][p['key']]['n_fail']}、{r['parts'][p['key']]['succ_ok']}/{r['parts'][p['key']]['n_succ']}；"
            f"{_pct(r['parts'][p['key']]['bal'])}" for p in res["parts"]) + " |")
    mc = {r["id"]: r for r in res["rows"]}[P]["vs_baseline"]
    L += ["", "## 事先登記的主要問題", "",
          f"- {nm(P)} vs {nm(B)}（2 個零件合併）：McNemar {mc['a_only']} : {mc['b_only']}，p＝{mc['p_value']:.4f}；"
          f"兩類平均差 {res['d_bal'] * 100:+.1f} 個百分點（{res['ci_d_bal'][0] * 100:+.1f}～{res['ci_d_bal'][1] * 100:+.1f}）。",
          "- 規則：p < 0.05、GPT 答對的多，而且合併兩類平均較高，才算成立。→ " + ("**成立**。" if res["holds"] else "**不成立**。"), ""]
    md = out / "report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({k: v for k, v in res.items() if k != "records"}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    with (out / "per_segment.jsonl").open("w", encoding="utf-8") as fh:
        for r in res["records"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(md)
