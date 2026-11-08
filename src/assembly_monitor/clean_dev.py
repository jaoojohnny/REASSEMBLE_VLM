"""第 11 包：乾淨的開發集（2026-10-31 使用者指定）。只讀已存在的輸出，不跑模型。

規則（使用者：「還是要以資料集給的正確答案為主 答案跟資料集一樣才能拿來用」）：
- 真實答案＝資料集成功／失敗標記，一律不改。
- Claude 先盲評（判斷寫好之後才讀標記），判斷和標記一樣的段＝乾淨開發集，之後用來挑方法、選門檻。
- 判斷和標記不一樣、或 Claude 看不出（uncertain）的段排除。
- 已凍結的測試集照原本標記算，不篩段（這裡不碰測試集）。

判斷來源：
- experiments/blind_review_dev30_v1：30 段，看 12 張影格（手腕＋外部相機），答案在 blind_review_dev30_answers_claude.json；
  段號只從 PRIVATE_answer_key.json 讀 segment_id 和零件。
- experiments/clean_dev_p11：其餘 211 段，每段看 3 張手腕影格（第一張、中間、最後一張），答案在 claude_judgments.jsonl。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluate import mcnemar_exact
from .reason_eval import _pct, best_tau, load_cfg, metrics, part_records, predict
from .util import now_iso

FROZEN_TAU = 0.999089  # 測試集主方法 qwen3vl_8b_rscore_t0999 用的 τ（4 個開發零件全部段選的）
JUDGE = {"成功": "success", "失敗": "failure", "success": "success", "failure": "failure"}
PROTOCOL = {"12": "12 張影格（手腕＋外部相機）", "3": "3 張手腕影格（第一張、中間、最後一張）"}
FIXED = [("v3_text", "第三版（寫字回答）"), ("score_any", "1 機率 0.5（約等於寫字）"), ("verify", "3 答失敗再檢查一次"),
         ("contrast_any", "2 前後對照 0.5"), ("fewshot_any", "4 看範例 0.5"), ("v4_any", "5 新清單＋兩張裁切 0.5"),
         ("v4w_any", "6 手腕第二次觀察 0.5"), ("v4and", "7 兩次都說失敗才算失敗")]
TUNED = [("score", "1 機率門檻"), ("contrast", "2 前後對照＋門檻"), ("fewshot", "4 看範例＋門檻"),
         ("v4", "5 新清單＋兩張裁切＋門檻"), ("v4w", "6 手腕第二次觀察＋門檻"), ("v4both", "7 兩次觀察平均＋門檻")]
TEST_ROLE = {"score_tau": "測試集主方法", "v4w_any": "測試集次要方法", "v3_text": "測試集基準"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_judgments(clean_dir: str | Path, blind_dir: str | Path) -> dict[str, dict[str, Any]]:
    """segment_id → Claude 的盲評判斷（success／failure／uncertain）。少段、多段、重複都停止。"""
    clean_dir, blind_dir = Path(clean_dir), Path(blind_dir)
    order = json.loads((clean_dir / "order.json").read_text(encoding="utf-8"))
    by_case: dict[str, dict[str, Any]] = {}
    for r in _jsonl(clean_dir / "claude_judgments.jsonl"):
        if r["case_id"] in by_case:
            raise ValueError(f"{r['case_id']} 判斷重複")
        by_case[r["case_id"]] = r
    out: dict[str, dict[str, Any]] = {}
    for o in order:
        j = by_case.get(o["case_id"])
        if j is None:
            raise ValueError(f"{o['case_id']} 還沒有判斷")
        out[o["segment_id"]] = {"case_id": o["case_id"], "part": o["part"], "judgment": JUDGE.get(j["judgment"], "uncertain"),
                                "note": j.get("note", ""), "protocol": "3"}
    extra = sorted(set(by_case) - {o["case_id"] for o in order})
    if extra:
        raise ValueError(f"判斷檔有 order.json 沒有的編號：{extra[:5]}")
    key = {k["case_id"]: k for k in json.loads((blind_dir / "PRIVATE_answer_key.json").read_text(encoding="utf-8"))}
    reviews = json.loads((blind_dir / "blind_review_dev30_answers_claude.json").read_text(encoding="utf-8"))["reviews"]
    for a in reviews:
        k = key[a["case_id"]]
        if k["segment_id"] in out:
            raise ValueError(f"{k['segment_id']} 兩邊都有判斷")
        out[k["segment_id"]] = {"case_id": a["case_id"], "part": k["part"], "judgment": JUDGE.get(a["judgment"], "uncertain"),
                                "note": a.get("evidence", ""), "protocol": "12"}
    return out


def method_rows(by_part: dict[str, list[dict[str, Any]]], tau_fixed: dict[str, float]) -> list[dict[str, Any]]:
    """各方法在這批段上的成績。門檻方法兩種算法：留一個零件選 τ（主要），和用固定 τ（tau_fixed）。"""
    keys = list(by_part)
    allrecs = [r for k in keys for r in by_part[k]]
    rows: list[dict[str, Any]] = []
    for m, label in FIXED:
        pred = {r["segment_id"]: predict(r, m) for r in allrecs}
        rows.append({"id": m, "label": label, "pred": pred})
    for base, label in TUNED:
        pred, part_tau = {}, {}
        for k in keys:
            train = [r for q in keys if q != k for r in by_part[q]]
            part_tau[k] = best_tau(train, base)
            pred.update({r["segment_id"]: predict(r, f"{base}_tau", part_tau[k]) for r in by_part[k]})
        tf = tau_fixed[base]
        rows.append({"id": f"{base}_tau", "label": label, "pred": pred, "tau_lopo": part_tau, "tau_fixed": tf,
                     "fixed": metrics(allrecs, [predict(r, f"{base}_tau", tf) for r in allrecs]),
                     "tau_here": best_tau(allrecs, base)})
    base_ok = [rows[0]["pred"][r["segment_id"]] == r["truth"] for r in allrecs]
    for row in rows:
        row["pooled"] = metrics(allrecs, [row["pred"][r["segment_id"]] for r in allrecs])
        row["parts"] = {k: metrics(v, [row["pred"][r["segment_id"]] for r in v]) for k, v in by_part.items()}
        ok = [row["pred"][r["segment_id"]] == r["truth"] for r in allrecs]
        row["vs_v3"] = mcnemar_exact(ok, base_ok) if row["id"] != "v3_text" else None
    return rows


def agreement(recs: list[dict[str, Any]]) -> dict[str, int]:
    c = {"n": len(recs), "label_fail": 0, "label_succ": 0, "ff": 0, "ss": 0, "said_s_label_f": 0, "said_f_label_s": 0,
         "unc_label_f": 0, "unc_label_s": 0}
    for r in recs:
        t, j = r["truth"], r["judgment"]
        c["label_fail" if t == "failure" else "label_succ"] += 1
        if j == "uncertain":
            c["unc_label_f" if t == "failure" else "unc_label_s"] += 1
        elif j == t:
            c["ff" if t == "failure" else "ss"] += 1
        else:
            c["said_s_label_f" if t == "failure" else "said_f_label_s"] += 1
    c["usable"] = c["ff"] + c["ss"]
    return c


def build(cfg_path: str | Path | dict[str, Any], clean_dir: str | Path, blind_dir: str | Path,
          frozen_tau: float | None = FROZEN_TAU) -> dict[str, Any]:
    cfg = cfg_path if isinstance(cfg_path, dict) else load_cfg(cfg_path)
    parts = cfg["parts"]
    full = {p["key"]: part_records(p) for p in parts}
    J = load_judgments(clean_dir, blind_dir)
    sids = {r["segment_id"] for v in full.values() for r in v}
    if set(J) != sids:
        raise ValueError(f"判斷段和開發段對不上：少 {len(sids - set(J))} 段、多 {len(set(J) - sids)} 段")
    for k, v in full.items():
        for r in v:
            j = J[r["segment_id"]]
            if j["part"] != k:
                raise ValueError(f"{r['segment_id']} 零件不同：{j['part']} vs {k}")
            r.update({"case_id": j["case_id"], "judgment": j["judgment"], "note": j["note"], "protocol": j["protocol"],
                      "usable": j["judgment"] == r["truth"]})
    clean = {k: [r for r in v if r["usable"]] for k, v in full.items()}
    allfull = [r for v in full.values() for r in v]
    tau_full = {b: best_tau(allfull, b) for b, _ in TUNED}
    if frozen_tau is not None and abs(tau_full["score"] - frozen_tau) > 1e-6:
        raise ValueError(f"全部開發段重選的機率門檻 {tau_full['score']} 和凍結的 {frozen_tau} 不同，輸出可能變了")
    return {"parts": [{"key": p["key"], "name": p["name"]} for p in parts], "tau_full": tau_full,
            "agree": {k: agreement(v) for k, v in full.items()}, "agree_all": agreement(allfull),
            "agree_by_protocol": {pr: agreement([r for r in allfull if r["protocol"] == pr]) for pr in PROTOCOL},
            "clean_rows": method_rows(clean, tau_full), "full_rows": method_rows(full, tau_full),
            "clean_counts": {k: metrics(v, [r["truth"] for r in v]) for k, v in clean.items()}, "records": allfull}


def write_outputs(res: dict[str, Any], out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {p["key"]: p["name"] for p in res["parts"]}
    keys = list(names)
    recs = res["records"]
    A, cc = res["agree_all"], res["clean_counts"]
    n_cf = sum(v["n_fail"] for v in cc.values())
    usable = {
        "name": "第 11 包乾淨開發集（Claude 盲評判斷和資料集標記一致的開發段）", "created_at": now_iso(),
        "rule": "使用者 2026-10-31 指定：以資料集給的答案為主；判斷和資料集標記一樣的段才拿來用，不一樣或看不出的段排除，標記不改。"
                "已凍結的測試集照原本標記算、不篩段。",
        "protocol": {"blind_review_dev30_v1": PROTOCOL["12"], "clean_dev_p11": PROTOCOL["3"]},
        "n_dev": len(recs), "n_usable": A["usable"], "n_usable_fail": n_cf, "n_usable_succ": A["usable"] - n_cf,
        "per_part": {k: {"name": names[k], **res["agree"][k]} for k in keys},
        "usable_segment_ids": sorted(r["segment_id"] for r in recs if r["usable"]),
        "segments": [{k: r[k] for k in ("segment_id", "part", "case_id", "protocol", "truth", "judgment", "usable", "note")}
                     for r in recs]}
    (out / "usable_dev_segments.json").write_text(json.dumps(usable, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# 第 11 包：乾淨開發集（判斷和資料集標記一致的段）", "", f"產生時間：{now_iso()}", "",
         "使用者 2026-10-31 指定：以資料集給的答案為主，判斷和資料集一樣的段才拿來用。做法：Claude 先盲評全部開發段"
         "（判斷全部寫好之後才讀標記），和標記一樣的段＝乾淨開發集；不一樣或看不出的段排除，**標記不改**。"
         "已凍結的測試集照原本標記算、不篩段。", "",
         f"- 30 段（blind_review_dev30_v1）看{PROTOCOL['12']}；其餘 211 段看{PROTOCOL['3']}。",
         "- 判斷主要看最後一張手腕影格：手指張開、零件留在位置上＝成功；零件還在指尖、被帶走、滑掉＝失敗；"
         "看不到零件、圓柱歪不歪看不出、段太短、零件碰著手指＝看不出。", "",
         "## 盲評和資料集標記比對", "",
         "| 零件 | 段數 | 標記失敗 | 標記成功 | 失敗段一致 | 成功段一致 | 我說成功、標記失敗 | 我說失敗、標記成功 | 看不出（標記失敗／成功） | 可用 |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    def arow(name: str, c: dict[str, int]) -> str:
        return (f"| {name} | {c['n']} | {c['label_fail']} | {c['label_succ']} | {c['ff']} | {c['ss']} | {c['said_s_label_f']} | "
                f"{c['said_f_label_s']} | {c['unc_label_f']}／{c['unc_label_s']} | **{c['usable']}** |")

    L += [arow(names[k], res["agree"][k]) for k in keys] + [arow("合計", A)]
    L += ["", "依看圖方式：", ""] + [f"- {PROTOCOL[pr]}：{c['n']} 段，可用 {c['usable']}（失敗段一致 {c['ff']}/{c['label_fail']}、"
                                  f"成功段一致 {c['ss']}/{c['label_succ']}）" for pr, c in res["agree_by_protocol"].items()]
    L += ["", f"乾淨開發集：**{A['usable']} 段**（失敗 {n_cf}、成功 {A['usable'] - n_cf}）；排除 {A['n'] - A['usable']} 段。", "",
          "| 零件 | 可用段數 | 失敗 | 成功 |", "|---|---|---|---|"]
    L += [f"| {names[k]} | {cc[k]['n']} | {cc[k]['n_fail']} | {cc[k]['n_succ']} |" for k in keys]

    full = {r["id"]: r for r in res["full_rows"]}
    L += ["", f"## 各方法在乾淨開發集（{A['usable']} 段）", "",
          "門檻方法：「留一個零件」＝τ 用其他 3 個零件的乾淨段選、在留下的零件評分；「全部開發段的 τ」＝原本 241 段選的 τ（機率門檻就是測試集用的 0.999089）。"
          "最右欄是原本 241 段（留一個零件）的兩類平均，給對照。", "",
          "| 方法 | 答對 | 失敗段答對 | 成功段答對 | 兩類平均 | 用全部開發段的 τ | 和第三版比（只有這個對：只有第三版對） | 原本 241 段 |",
          "|---|---|---|---|---|---|---|---|"]
    best = max((r for r in res["clean_rows"] if r["id"] not in ("v3_text", "score_any")),
               key=lambda r: (r["pooled"]["bal"] or 0.0, r["pooled"]["acc"] or 0.0))
    for r in res["clean_rows"]:
        m = r["pooled"]
        role = f"（{TEST_ROLE[r['id']]}）" if r["id"] in TEST_ROLE else ""
        star = " ★" if r is best else ""
        fx = "–" if "fixed" not in r else f"{r['fixed']['correct']}/{r['fixed']['n']}，兩類平均 {_pct(r['fixed']['bal'])}"
        vs = "–" if not r["vs_v3"] else f"{r['vs_v3']['a_only']} : {r['vs_v3']['b_only']}，p＝{r['vs_v3']['p_value']:.3f}"
        miss = f"（缺 {m['n_missing']}）" if m["n_missing"] else ""
        L.append(f"| {r['label']}{role}{star} | {m['correct']}/{m['n']}＝{_pct(m['acc'])}{miss} | {m['fail_ok']}/{m['n_fail']} | "
                 f"{m['succ_ok']}/{m['n_succ']} | **{_pct(m['bal'])}** | {fx} | {vs} | {_pct(full[r['id']]['pooled']['bal'])} |")
    L += ["", "## 各零件（乾淨段；兩類平均，答對）", "", "| 方法 | " + " | ".join(names[k] for k in keys) + " |",
          "|---|" + "---|" * len(keys)]
    for r in res["clean_rows"]:
        L.append(f"| {r['label']} | " + " | ".join(
            f"{_pct(r['parts'][k]['bal'])}（{r['parts'][k]['correct']}/{r['parts'][k]['n']}）" for k in keys) + " |")
    L += ["", "## 門檻 τ", "", "| 方法 | 原本 241 段選的 τ | 乾淨段全部選的 τ（之後新方法可用） | 乾淨段留一個零件時各零件的 τ |", "|---|---|---|---|"]
    for r in res["clean_rows"]:
        if "tau_lopo" in r:
            L.append(f"| {r['label']} | {r['tau_fixed']:.6f} | {r['tau_here']:.6f} | "
                     + "、".join(f"{names[k]} {v:.4f}" for k, v in r["tau_lopo"].items()) + " |")
    L += ["", "## 怎麼看", "",
          f"- 乾淨段上兩類平均最高（不含第三版、機率 0.5）：**{best['label']}**。這只是開發集的參考，測試集已凍結，不因此改。",
          "- 排除的段是「看得到的結果和標記不一樣」或看不出的段，通常也是模型最容易錯的段，所以乾淨段的成績會比原本 241 段高，"
          "**不能和測試集成績直接比**；它的用途是挑方法、選門檻時少受標記雜訊和看不清的段影響。",
          "- 篩選用的是 Claude 看手腕影格的判斷，Qwen 看的也是同類影格：乾淨段偏向「手腕影格看得出結果」的段。",
          "- 211 段只看 3 張手腕影格，比 30 段的 12 張少；看不出的段直接排除，不補看。", ""]
    md = out / "report.md"
    md.write_text("\n".join(L), encoding="utf-8")
    slim = {k: v for k, v in res.items() if k not in ("records", "clean_rows", "full_rows")}
    slim["clean_rows"] = [{k: v for k, v in r.items() if k != "pred"} for r in res["clean_rows"]]
    slim["full_rows"] = [{k: v for k, v in r.items() if k != "pred"} for r in res["full_rows"]]
    slim["best_clean"] = best["id"]
    (out / "results.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(md)
