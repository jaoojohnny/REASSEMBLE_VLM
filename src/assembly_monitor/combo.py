"""第 15 包：每零件分類器（第 14 包凍結的 F2S2）＋本機 Qwen 機率門檻（第 11 包，τ 0.999089）。

使用者 2026-11-09：「每個零件選擇各自的分類器，並搭配 Qwen 的機率門檻」。全部在本機，不用 GPT。

事先登記的 6 種方法（在 5 個開發零件比較、依規則選一種凍結，再套到 10 個測試零件）：
  M0 每零件分類器（只看接頭＋L1→L2；第 14 包）
  M1 Qwen 機率門檻（失敗分數 > 0.999089 判失敗；不訓練）
  C2 兩個都說失敗才判失敗（AND）
  C3 任一個說失敗就判失敗（OR）
  C1 Qwen 分數當一個特徵：L1 挑的 k 個影像特徵＋Qwen 失敗分數（logit），一起做 L2 邏輯迴歸
  C4 每個零件自己挑：每一折的內層交叉驗證在 M0、M1、C2、C3、C1 裡挑兩類平均最高的
選擇規則：5 個開發零件平均兩類平均最高；差 2 點以內選簡單的（M0 → M1 → C2 → C3 → C1 → C4）。
交叉驗證和第 14 包相同（依錄影分組、分層；外層 5 折 × 5 次，內層 4 折），M0 的分組與內層種子也相同。
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import perpart as pp
from .util import now_iso

log = logging.getLogger(__name__)

TAU = 0.999089
METHODS = ("M0", "M1", "C2", "C3", "C1", "C4")      # 「簡單」的在前
OPTIONS = ("M0", "M1", "C2", "C3", "C1")            # C4 在內層挑的範圍
NAMES = {"M0": "每零件分類器", "M1": "Qwen 機率門檻", "C2": "兩個都說失敗才失敗（AND）",
         "C3": "任一個說失敗就失敗（OR）", "C1": "Qwen 分數當一個特徵", "C4": "每個零件自己挑"}
P11_DEV = {"square peg 2": "sqpeg2", "round peg 3": "rpeg3", "D-SUB": "dsub", "medium gear": "mgear"}
FINAL_DIRS = {"round peg 1": "final_test_rpeg1", "large gear": "final_test_lgear", "round peg 2": "final_test_rpeg2",
              "square peg 3": "final_test_sqpeg3", "square peg 4": "final_test_sqpeg4", "waterproof": "final_test_wproof"}
# 之前沒跑過機率門檻的段，第 15 包補跑（同一個凍結的模型 qwen3vl_8b_rscore_t0999、提示詞 checklist-fields-v1）
NEW_RUNS = {"BNC": "bnc", "USB": "usb", "bolt 4": "bolt4", "Ethernet": "eth", "square peg 1": "sqpeg1",
            "square peg 2": "sqpeg2", "round peg 3": "rpeg3", "D-SUB": "dsub", "medium gear": "mgear"}
NEW_RUN_DIR = "experiments/p15_qwenprob"
CLOSE = 0.02


# ---------------------------------------------------------------- Qwen 失敗分數


def _fail(p_yes: dict[str, float]) -> float:
    from .reason import fail_score

    return float(fail_score(p_yes))


def raw_scores(path: str | Path) -> dict[str, float | None]:
    """raw.jsonl → {段: 失敗分數 1−Π(1−P(yes))}；沒有機率（錯誤、無效）記 None。"""
    out: dict[str, float | None] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        reason = r.get("reason")
        if isinstance(reason, str):
            try:
                reason = ast.literal_eval(reason)
            except (ValueError, SyntaxError):
                reason = None
        py = (reason or {}).get("p_yes") if isinstance(reason, dict) else None
        out[str(r["segment_id"])] = _fail(py) if py else None
    return out


def qwen_scores(part: str, base: str | Path = "experiments") -> dict[str, float]:
    base = Path(base)
    d: dict[str, float | None] = {}
    if part in P11_DEV:
        d.update(raw_scores(base / f"rdev_{P11_DEV[part]}_v3" / "qwen3vl_8b_rverify" / "f4" / "raw.jsonl"))
    if part in FINAL_DIRS:
        d.update(raw_scores(base / FINAL_DIRS[part] / "qwen3vl8b_prob_t0999" / "qwen3vl_8b_rscore_t0999" / "f4" / "raw.jsonl"))
    if part in NEW_RUNS:
        for k, v in raw_scores(Path(NEW_RUN_DIR.replace("experiments", str(base), 1)) / NEW_RUNS[part]
                               / "qwen3vl_8b_rscore_t0999" / "f4" / "raw.jsonl").items():
            if d.get(k) is None:
                d[k] = v
    return {k: v for k, v in d.items() if v is not None}


def logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6)
    return np.log(s) - np.log1p(-s)


# ---------------------------------------------------------------- 交叉驗證


def _fold_predict_stack(Xtr: np.ndarray, qtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, qte: np.ndarray,
                        keys: list[tuple]) -> dict[tuple, np.ndarray]:
    """C1：L1 從影像特徵挑 k 個，再加上 Qwen 分數（一定放），一起做 L2 邏輯迴歸。"""
    from sklearn.preprocessing import StandardScaler

    if len(set(ytr.tolist())) < 2:
        return {key: np.full(len(Xte), int(ytr[0])) for key in keys}
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    mu, sd = float(np.mean(qtr)), float(np.std(qtr)) or 1.0
    qz_tr, qz_te = ((qtr - mu) / sd)[:, None], ((qte - mu) / sd)[:, None]
    ranks = {c1: pp._rank_l1(Ztr, ytr, c1) for c1 in sorted({key[1] for key in keys})}
    out = {}
    for k, c1, c2 in keys:
        sel = ranks[c1][:k]
        A = np.hstack([Ztr[:, sel], qz_tr]) if len(sel) else qz_tr
        B = np.hstack([Zte[:, sel], qz_te]) if len(sel) else qz_te
        out[(k, c1, c2)] = pp._lr(c2).fit(A, ytr).predict(B)
    return out


def _combine(m0: np.ndarray, m1: np.ndarray, how: str) -> np.ndarray:
    return (m0 & m1) if how == "C2" else (m0 | m1)


def nested_combo(X: np.ndarray, q: np.ndarray, qpred: np.ndarray, y: np.ndarray, groups: list[str],
                 repeats: int = pp.REPEATS, outer: int = pp.OUTER, inner: int = pp.INNER,
                 seed0: int = 0) -> dict[str, Any]:
    """一次跑完 6 種方法（同一組分組）。q＝Qwen 分數 logit（缺的已補）、qpred＝門檻判斷（1＝失敗）。"""
    y = np.asarray(y, int)
    qpred = np.asarray(qpred, int)
    keys = pp.grid_keys("S2")
    res = {m: [] for m in METHODS}
    picks: Counter = Counter()
    for r in range(repeats):
        pred = {m: np.full(len(y), -1) for m in METHODS}
        for i, (tr, te) in enumerate(pp.folds(y, groups, outer, seed0 + r)):
            gtr = [groups[j] for j in tr]
            ytr = y[tr]
            p0 = {k: np.zeros(len(tr), int) for k in keys}
            p1 = {k: np.zeros(len(tr), int) for k in keys}
            for itr, ite in pp.folds(ytr, gtr, inner, 1000 * (seed0 + r + 1) + i):
                a, b = tr[itr], tr[ite]
                for k, v in pp._fold_predict(X[a], y[a], X[b], "S2", keys).items():
                    p0[k][ite] = v
                for k, v in _fold_predict_stack(X[a], q[a], y[a], X[b], q[b], keys).items():
                    p1[k][ite] = v
            b0 = pp.choose({k: pp.bal_acc(ytr, v) for k, v in p0.items()})
            b1 = pp.choose({k: pp.bal_acc(ytr, v) for k, v in p1.items()})
            inner_pred = {"M0": p0[b0], "M1": qpred[tr], "C1": p1[b1]}
            inner_pred["C2"] = _combine(inner_pred["M0"], inner_pred["M1"], "C2")
            inner_pred["C3"] = _combine(inner_pred["M0"], inner_pred["M1"], "C3")
            sc = {m: pp.bal_acc(ytr, inner_pred[m]) for m in OPTIONS}
            best = max(OPTIONS, key=lambda m: (round(sc[m], 9), -OPTIONS.index(m)))
            picks[best] += 1
            o0 = pp._fold_predict(X[tr], ytr, X[te], "S2", [b0])[b0]
            o1 = _fold_predict_stack(X[tr], q[tr], ytr, X[te], q[te], [b1])[b1]
            outp = {"M0": o0, "M1": qpred[te], "C1": o1, "C2": _combine(o0, qpred[te], "C2"),
                    "C3": _combine(o0, qpred[te], "C3")}
            outp["C4"] = outp[best]
            for m in METHODS:
                pred[m][te] = outp[m]
        for m in METHODS:
            assert (pred[m] >= 0).all()
            res[m].append({"bal": pp.bal_acc(y, pred[m]), **pp.counts(y, pred[m]), "pred": pred[m].tolist()})
    summ = {m: {"bal_mean": float(np.mean([x["bal"] for x in v])), "bal_min": float(np.min([x["bal"] for x in v])),
                "bal_max": float(np.max([x["bal"] for x in v])), "repeats": v} for m, v in res.items()}
    return {"methods": summ, "c4_picks": dict(picks)}


def select_method(dev: dict[str, dict[str, float]], close: float = CLOSE) -> dict[str, Any]:
    avg = {m: float(np.mean(list(dev[m].values()))) for m in METHODS}
    best = max(avg.values())
    pick = next(m for m in METHODS if avg[m] >= best - close - 1e-12)
    return {"method": pick, "avg": avg, "best_avg": best, "best_method": max(avg, key=avg.get)}


# ---------------------------------------------------------------- 資料


def part_data(part: str, encs: dict[str, Any], cache_root: Path, index_dir: str = "data/index",
              qbase: str | Path = "experiments") -> dict[str, Any]:
    segs = pp.part_segments(index_dir, part)
    V, keep, missing = pp.segment_vectors(segs, encs, cache_root, need=("F2",))
    qs = qwen_scores(part, qbase)
    s = np.array([qs.get(x["segment_id"], np.nan) for x in keep], float)
    have = ~np.isnan(s)
    med = float(np.nanmedian(s)) if have.any() else 0.5
    s_f = np.where(have, s, med)          # 缺的補中位數（C1 用）；門檻判斷缺的當「成功」
    qpred = np.where(have, s > TAU, False).astype(int)
    return {"X": V["F2"], "q": logit(s_f), "qpred": qpred, "y": np.array([x["label"] for x in keep]),
            "recs": [x["recording_id"] for x in keep], "ids": [x["segment_id"] for x in keep],
            "n_missing_frames": len(missing), "n_missing_qwen": int((~have).sum())}


def run_dev(out_dir: str = "experiments/p15_combo_dev", selected_path: str = "configs/combo_selected.yaml",
            cache_root: str = "experiments/_features", repeats: int = pp.REPEATS) -> dict[str, Any]:
    pp.load_selected()  # 第 14 包的做法要已凍結（M0 就是它）
    encs = pp.make_encoders()
    res: dict[str, Any] = {"generated": now_iso(), "parts": {}}
    try:
        for part in pp.DEV_PARTS:
            d = part_data(part, encs, Path(cache_root))
            r = nested_combo(d["X"], d["q"], d["qpred"], d["y"], d["recs"], repeats=repeats)
            res["parts"][part] = {"n": len(d["y"]), "n_failure": int(d["y"].sum()), "n_missing_qwen": d["n_missing_qwen"],
                                  "ids": d["ids"], "labels": d["y"].tolist(), **r}
            log.info("  開發 %-13s " + "、".join(f"{m} %.1f%%" for m in METHODS) + "（缺 Qwen %d 段）", part,
                     *[r["methods"][m]["bal_mean"] * 100 for m in METHODS], d["n_missing_qwen"])
    finally:
        for k in ("full", "crop"):
            encs[k].close()
    dev = {m: {p: res["parts"][p]["methods"][m]["bal_mean"] for p in res["parts"]} for m in METHODS}
    res["selection"] = select_method(dev)
    res["selected_config"] = str(write_selected(res, selected_path))
    write_report(res, Path(out_dir), stage="dev")
    return res


def write_selected(res: dict[str, Any], path: str | Path) -> Path:
    import yaml

    p = Path(path)
    sel = res["selection"]
    if p.exists():
        old = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("combo_selected") or {}
        if old.get("method") != sel["method"]:
            raise ValueError(f"{p} 已凍結為 {old.get('method')}，這次選到 {sel['method']}；不覆寫")
        return p
    doc = {"combo_selected": {"frozen": True, "frozen_at": now_iso(), "method": sel["method"], "name": NAMES[sel["method"]],
                              "tau": TAU, "perpart_recipe": pp.load_selected()["recipe"],
                              "dev_avg_balanced_accuracy": {k: round(v, 4) for k, v in sel["avg"].items()}}}
    p.write_text("# 第 15 包：在 5 個開發零件依事先寫好的規則選出的組合方法（自動寫入、凍結）。測試零件只用這份。\n"
                 + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def load_selected(path: str | Path = "configs/combo_selected.yaml") -> dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.exists():
        raise ValueError(f"找不到 {p}：要先跑開發階段")
    d = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("combo_selected") or {}
    if not d.get("frozen") or d.get("method") not in METHODS:
        raise ValueError(f"{p} 不是凍結的設定")
    return d


def run_test(out_dir: str = "experiments/p15_combo_test", selected_path: str = "configs/combo_selected.yaml",
             cache_root: str = "experiments/_features", repeats: int = pp.REPEATS) -> dict[str, Any]:
    sel = load_selected(selected_path)
    m = sel["method"]
    encs = pp.make_encoders()
    res: dict[str, Any] = {"generated": now_iso(), "method": m, "frozen_at": sel.get("frozen_at"), "parts": {}}
    try:
        for part in pp.TEST_PARTS:
            t0 = time.time()
            d = part_data(part, encs, Path(cache_root))
            r = nested_combo(d["X"], d["q"], d["qpred"], d["y"], d["recs"], repeats=repeats)
            days = [pp.date_of(x) for x in d["recs"]]
            rd = nested_combo(d["X"], d["q"], d["qpred"], d["y"], days, repeats=1)
            pr = {"n": len(d["y"]), "n_failure": int(d["y"].sum()), "n_missing_qwen": d["n_missing_qwen"],
                  "ids": d["ids"], "labels": d["y"].tolist(), **r,
                  "date": {k: v["bal_mean"] for k, v in rd["methods"].items()}}
            ans = pp.zero_shot_test(part)
            gpt = ans.get("gpt61_sol_ft_eb05_f4")
            if gpt:
                truth = dict(zip(d["ids"], d["y"].tolist()))
                ov = [i for i in d["ids"] if i in gpt]
                pos = {sid: j for j, sid in enumerate(d["ids"])}
                pr["gpt"] = {**pp.bal_on(ov, truth, gpt), "selected_same_segments": float(np.mean(
                    [pp.bal_acc(np.array([truth[i] for i in ov]), np.array([rep["pred"][pos[i]] for i in ov]))
                     for rep in r["methods"][m]["repeats"]]))}
            pr["seconds"] = round(time.time() - t0, 1)
            res["parts"][part] = pr
            log.info("  測試 %-13s %s %.1f%%、M0 %.1f%%、M1 %.1f%%（依日期 %s %.1f%%）", part, m,
                     r["methods"][m]["bal_mean"] * 100, r["methods"]["M0"]["bal_mean"] * 100,
                     r["methods"]["M1"]["bal_mean"] * 100, m, pr["date"][m] * 100)
    finally:
        for k in ("full", "crop"):
            encs[k].close()
    P = res["parts"]
    res["summary"] = {"sel_mean": float(np.mean([P[p]["methods"][m]["bal_mean"] for p in P])),
                      "m0_mean": float(np.mean([P[p]["methods"]["M0"]["bal_mean"] for p in P])),
                      "m1_mean": float(np.mean([P[p]["methods"]["M1"]["bal_mean"] for p in P])),
                      "date_sel_mean": float(np.mean([P[p]["date"][m] for p in P])),
                      "n_better_than_m0": int(sum(P[p]["methods"][m]["bal_mean"] > P[p]["methods"]["M0"]["bal_mean"] + 1e-9
                                                  for p in P)),
                      "means": {k: float(np.mean([P[p]["methods"][k]["bal_mean"] for p in P])) for k in METHODS}}
    write_report(res, Path(out_dir), stage="test")
    return res


def _pct(x: float | None) -> str:
    return "–" if x is None or x != x else f"{x * 100:.0f}%"


def write_report(res: dict[str, Any], out: Path, stage: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    P = res["parts"]
    parts = list(P)
    L = [f"# 第 15 包（{'開發' if stage == 'dev' else '測試'}階段）：每零件分類器＋Qwen 機率門檻", "",
         f"產生時間：{res['generated']}", "",
         "每零件分類器＝第 14 包凍結的 F2S2（只看接頭＋L1→L2）；Qwen 機率門檻＝第 11 包 τ 0.999089（不訓練）。"
         "交叉驗證和第 14 包相同（依錄影分組，外層 5 折 × 5 次、內層 4 折）。數字＝兩類平均（5 次平均）。", ""]
    if stage == "dev":
        sel = res["selection"]
        L += ["## 6 種方法（5 個開發零件）", "", "| 方法 | " + " | ".join(parts) + " | 平均 |", "|---|" + "---|" * (len(parts) + 1)]
        for m in METHODS:
            star = " ★" if m == sel["method"] else ""
            L.append(f"| {m}{star} {NAMES[m]} | " + " | ".join(_pct(P[p]["methods"][m]["bal_mean"]) for p in parts)
                     + f" | {_pct(sel['avg'][m])} |")
        L += ["", f"選出：**{sel['method']} {NAMES[sel['method']]}**（平均最高、差 2 點以內選簡單的；最高 {sel['best_method']} "
              f"{_pct(sel['best_avg'])}）。凍結在 `{res['selected_config']}`。", ""]
    else:
        m, s = res["method"], res["summary"]
        L += ["## 摘要", "",
              f"- 方法：**{m} {NAMES[m]}**（開發階段凍結於 {res['frozen_at']}）。",
              f"- 主要比較（事先登記）：10 個測試零件平均 **{_pct(s['sel_mean'])}** vs 每零件分類器單獨 **{_pct(s['m0_mean'])}**"
              f"（Qwen 機率門檻單獨 {_pct(s['m1_mean'])}）；比單獨分類器好的零件 {s['n_better_than_m0']}/10。",
              f"- 依日期分組（留一天）：{_pct(s['date_sel_mean'])}。", "",
              "## 每個零件（6 種方法都列；★＝凍結的方法）", "",
              "| 零件（失敗） | " + " | ".join(f"{k}{' ★' if k == m else ''}" for k in METHODS) + " | 依日期 ★ | GPT（同一批段） |",
              "|---|" + "---|" * (len(METHODS) + 2)]
        for p in parts:
            pr = P[p]
            g = pr.get("gpt")
            L.append(f"| {p}（{pr['n_failure']}） | " + " | ".join(_pct(pr["methods"][k]["bal_mean"]) for k in METHODS)
                     + f" | {_pct(pr['date'][m])} | " + (f"{_pct(g['bal'])}（★ {_pct(g['selected_same_segments'])}）" if g else "–") + " |")
        L += ["", "平均：" + "、".join(f"{k} {_pct(v)}" for k, v in s["means"].items()), ""]
    L += ["## 失敗段答對／成功段答對（5 次平均）", "", "| 零件 | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for p in parts:
        pr = P[p]
        cells = []
        for k in METHODS:
            rep = pr["methods"][k]["repeats"]
            cells.append(f"{np.mean([x['ok_failure'] for x in rep]):.1f}/{pr['n_failure']}、"
                         f"{np.mean([x['ok_success'] for x in rep]):.1f}/{pr['n'] - pr['n_failure']}")
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L += ["", "C4（每個零件自己挑）在各折選到的方法：", ""]
    for p in parts:
        L.append(f"- {p}：" + "、".join(f"{k} {v}" for k, v in sorted(P[p]["c4_picks"].items(), key=lambda x: -x[1])))
    L += ["", "## 注意", "",
          "- 全部在本機（Qwen 3–4 秒／段＋分類器），不用 GPT。",
          "- Qwen 分數缺的段：C1 補中位數、門檻判斷當成功（每個零件缺幾段列在 results.json 的 n_missing_qwen）。",
          "- 每零件分類器要該零件自己有標記的資料；失敗段少的零件（medium gear 8、waterproof 8、large gear 6）起伏大。", ""]
    md = out / "report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(json.loads(json.dumps(res, default=str)), ensure_ascii=False), encoding="utf-8")
    return str(md)


# ---------------------------------------------------------------- 補跑 Qwen 機率門檻的實驗設定


NEW_RUN_SPEC = {   # key → (零件, 切分, 集合, 影格資料夾, 提示詞裡的零件名稱＝之前同零件的設定)
    "bnc": ("BNC", "data/splits/bnc_alldev", "dev", "data/derived/frames/insert_BNC", "BNC connector"),
    "usb": ("USB", "data/splits/usb_test100", "test", "data/derived/frames/insert_USB", "USB connector"),
    "bolt4": ("bolt 4", "data/splits/bolt4_test100", "test", "data/derived/frames/insert_bolt_4", "bolt"),
    "eth": ("Ethernet", "data/splits/ethernet_test100", "test", "data/derived/frames/insert_Ethernet", "Ethernet connector"),
    "sqpeg1": ("square peg 1", "data/splits/sqpeg1_test100", "test", "data/derived/frames/insert_square_peg_1", "square peg"),
    "sqpeg2": ("square peg 2", "data/splits/sqpeg2_ref10_dev90", "ref", "data/derived/frames/insert_square_peg_2", "square peg"),
    "rpeg3": ("round peg 3", "data/splits/rpeg3_ref10_dev90", "ref", "data/derived/frames/insert_round_peg_3", "round peg"),
    "dsub": ("D-SUB", "data/splits/dsub_ref10_dev90", "ref", "data/derived/frames/insert_D_SUB", "D-SUB connector"),
    "mgear": ("medium gear", "data/splits/mgear_ref10_dev90", "ref", "data/derived/frames/insert_medium_gear", "gear"),
}


def experiment_for(key: str, base_path: str = "configs/experiment_eth_checklistf.yaml") -> dict[str, Any]:
    """凍結的機率門檻設定（和第 11、12 包測試集相同的提示詞與取樣），只換零件、切分、集合、影格資料夾、輸出名稱。"""
    from .vlm import load_experiment

    part, split, include, froot, pname = NEW_RUN_SPEC[key]
    e = load_experiment(base_path)
    task = dict(e["task"], part=pname)
    return dict(e, name=f"p15_qwenprob/{key}", models=["qwen3vl_8b_rscore_t0999"], split_dir=split, include=[include],
                frames_root=froot, task=task, frames=[4])


def run_qwen(models: list[str], reasons: list[str], keys: list[str] | None = None) -> list[dict[str, Any]]:
    from .vlm import load_model_configs, run_experiment

    cfgs = load_model_configs(models)
    out = []
    for key in keys or list(NEW_RUN_SPEC):
        exp = experiment_for(key)
        log.info("  Qwen 機率門檻 %s（%s，%s）……", NEW_RUN_SPEC[key][0], exp["split_dir"], exp["include"][0])
        r = run_experiment(exp=exp, model_cfgs=cfgs, model_keys=["qwen3vl_8b_rscore_t0999"], frames=[4],
                           reasons=reasons, unlock_test=exp["include"] == ["test"])
        out.append({"key": key, "n_segments": r["n_segments"]})
    return out


def extract_for_runs(keys: list[str] | None = None, index_dir: str = "data/index", slim_dir: str = "data/slim",
                     mapping_path: str = "configs/field_mapping.yaml") -> dict[str, Any]:
    """補跑 Qwen 前，確認那些段有 4 格（段前後 0.5 秒）的影格；已有的略過。"""
    from .extract import run_extraction
    from .mapping import load_mapping
    from .split import read_split_csv

    mapping = load_mapping(mapping_path)
    tot = {"n_done": 0, "n_skipped": 0, "n_missing_frames": 0, "errors": []}
    for key in keys or list(NEW_RUN_SPEC):
        part, split, include, _froot, _ = NEW_RUN_SPEC[key]
        ids = [r["segment_id"] for r in read_split_csv(Path(split) / "split.csv") if r["split"] == include]
        res = run_extraction(mapping=mapping, index_dir=index_dir, slim_dir=slim_dir, out_dir="data/derived/frames",
                             part=part, action="insert", which="p15", n_frames_list=(4,), start_buffer=0.5,
                             end_buffer=0.5, cameras=list(pp.CAMS), workers=4, segment_ids=ids)
        log.info("  影格 %-13s %3d 段：新擷取 %d、已有 %d", part, len(ids), res["n_done"], res["n_skipped"])
        for k in ("n_done", "n_skipped", "n_missing_frames"):
            tot[k] += res[k]
        tot["errors"] += res["errors"]
    return tot
