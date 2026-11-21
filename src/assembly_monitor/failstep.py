"""第 18 包：失敗在哪一步——用資料集的步驟標記當答案，第 14 包凍結的 F2S2 只換答案。

- 答案：失敗段裡第一個標成失敗的步驟（data/index/skills.csv；Approach 接近、Align 對準、Twist 旋轉…）。
- 做法：每個步驟一個是非題（L1 排名次 → 留前 k 個 → L2），三題同一組 (k, C1, C2)，答機率最高的那一步。
  參數 45 組和 F2S2 相同，內層 4 折選「三類平均」最高的；外層 5 折重複 5 次（依錄影分組、按步驟分層）。
- bolt 4（主要）只跑 F2S2；BNC（開發零件）另外跑 F1S2、F3S2 當參考。
- 另外報：打亂答案檢定、只看段長的對照。
說明：docs/package18_failstep.md；輸出：experiments/p18_failstep/
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import perpart as pp

log = logging.getLogger(__name__)

OUT = Path("experiments/p18_failstep")
PRIMARY = "bolt 4"
DEV = "BNC"
RECIPES = {PRIMARY: ("F2S2",), DEV: ("F2S2", "F1S2", "F3S2")}
MIN_CLASS = 3
N_PERM = 100
PERM_SEED = 20261116
PASS_BAL, PASS_P = 0.60, 0.05
STEP_ZH = {"Approach": "接近", "Align": "對準", "Twist": "旋轉", "Push": "推入", "Release": "放開", "Nudge": "輕推",
           "Lift": "抬起", "Grasp": "夾取", "Pull": "拉出"}
GRID = [(k, c1, c2) for k in pp.K_GRID for c1 in pp.C1_GRID for c2 in pp.C2_GRID]


class Stop(RuntimeError):
    """缺資料或檢查沒過，不計分。"""


def zh(step: str) -> str:
    return STEP_ZH.get(step, step)


# ---------------------------------------------------------------- 答案


def _sid(recording_id: str, segment_id: str) -> str:
    return f"{recording_id}__seg{int(segment_id)}"


def first_failed_steps(index_dir: str | Path) -> dict[str, dict[str, Any]]:
    """段 → {step: 第一個失敗的步驟, n_failed: 失敗步驟數, is_last: 失敗的那一步是不是最後一步}；沒有失敗步驟的段不在裡面。"""
    rows: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    with (Path(index_dir) / "skills.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rows[_sid(r["recording_id"], r["segment_id"])].append((int(r["skill_id"]), r["text"], r["success"]))
    out = {}
    for sid, ss in rows.items():
        ss.sort()
        bad = [i for i, (_, _, ok) in enumerate(ss) if ok == "False"]
        if bad:
            out[sid] = {"step": ss[bad[0]][1], "n_failed": len(bad), "is_last": bad[0] == len(ss) - 1,
                        "sequence": " > ".join(t for _, t, _ in ss)}
    return out


def durations(index_dir: str | Path) -> dict[str, float]:
    out = {}
    with (Path(index_dir) / "segments.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                out[_sid(r["recording_id"], r["segment_id"])] = float(r["duration_s"])
            except (ValueError, KeyError):
                continue
    return out


def step_dataset(index_dir: str | Path, part: str, min_class: int = MIN_CLASS) -> dict[str, Any]:
    """某零件的失敗段＋失敗步驟。類別依段數由多到少排（同數依名稱），不到 min_class 段的步驟不納入。"""
    if part in pp.RESERVED:
        raise ValueError(f"{part} 是保留零件，不碰")
    fails = [s for s in pp.part_segments(index_dir, part) if s["label"] == 1]
    steps, dur = first_failed_steps(index_dir), durations(index_dir)
    no_step = [s["segment_id"] for s in fails if s["segment_id"] not in steps]
    cnt = Counter(steps[s["segment_id"]]["step"] for s in fails if s["segment_id"] in steps)
    classes = [c for c, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_class]
    rare = [s["segment_id"] for s in fails if s["segment_id"] in steps and steps[s["segment_id"]]["step"] not in classes]
    keep = [dict(s, **steps[s["segment_id"]], duration=float(dur.get(s["segment_id"], float("nan"))))
            for s in fails if s["segment_id"] in steps and steps[s["segment_id"]]["step"] in classes]
    return {"part": part, "classes": classes, "counts": {c: cnt[c] for c in classes}, "all_counts": dict(cnt),
            "segments": keep, "n_failures": len(fails), "no_step": no_step, "rare": rare}


def restrict(ds: dict[str, Any], keep: list[dict[str, Any]], missing: list[str],
             min_class: int = MIN_CLASS) -> tuple[dict[str, Any], list[int]]:
    """去掉沒有影格／特徵的段之後重新數一次：不到 min_class 段的步驟也不納入。回傳新的資料與留下的位置。"""
    cnt = Counter(s["step"] for s in keep)
    classes = [c for c, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_class]
    idx = [i for i, s in enumerate(keep) if s["step"] in classes]
    rare = list(ds["rare"]) + [s["segment_id"] for s in keep if s["step"] not in classes]
    return dict(ds, classes=classes, counts={c: cnt[c] for c in classes}, segments=[keep[i] for i in idx],
                missing=list(missing), rare=rare), idx


def avg_name(n_classes: int) -> str:
    return {2: "兩類平均", 3: "三類平均"}.get(n_classes, f"{n_classes} 類平均")


# ---------------------------------------------------------------- 模型（F2S2 的零件，每個步驟一個是非題）


def bal_multi(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    """每個步驟答對率的平均（只算答案裡有出現的步驟）。"""
    y, p = np.asarray(y), np.asarray(p)
    rs = [float(np.mean(p[y == c] == c)) for c in range(n_classes) if np.any(y == c)]
    return float(np.mean(rs)) if rs else float("nan")


def recall_counts(y: np.ndarray, p: np.ndarray, n_classes: int) -> list[list[int]]:
    y, p = np.asarray(y), np.asarray(p)
    return [[int(np.sum((y == c) & (p == c))), int(np.sum(y == c))] for c in range(n_classes)]


def class_scores(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray, n_classes: int,
                 keys: list[tuple]) -> dict[tuple, np.ndarray]:
    """每個步驟一個是非題；回傳每組參數的 [測試段數, 步驟數] 機率。訓練折沒有的步驟給 −inf（不會被選）。"""
    out = {key: np.full((len(Zte), n_classes), -np.inf) for key in keys}
    for c in range(n_classes):
        t = (ytr == c).astype(int)
        if t.sum() == 0:
            continue
        if t.sum() == len(t):
            for key in keys:
                out[key][:, c] = np.inf
            continue
        ranks = {c1: pp._rank_l1(Ztr, t, c1) for c1 in sorted({key[1] for key in keys})}
        done: dict[tuple, np.ndarray] = {}     # L1 留下的不到 k 個時，不同 k 是同一個模型 → 只擬合一次（結果相同）
        for key in keys:
            k, c1, c2 = key
            sel = ranks[c1][:k]
            if len(sel) == 0:
                out[key][:, c] = 0.5
                continue
            ck = (tuple(sel.tolist()), c2)
            if ck not in done:
                done[ck] = pp._lr(c2).fit(Ztr[:, sel], t).predict_proba(Zte[:, sel])[:, 1]
            out[key][:, c] = done[ck]
    return out


def fold_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, n_classes: int,
                 keys: list[tuple]) -> dict[tuple, np.ndarray]:
    """標準化只用訓練折；答案＝機率最高的步驟（同分選編號小＝段數多的）。"""
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    sc_ = class_scores(sc.transform(Xtr), ytr, sc.transform(Xte), n_classes, keys)
    return {key: np.argmax(s, axis=1) for key, s in sc_.items()}


def inner_select(X: np.ndarray, y: np.ndarray, groups: list[str], n_classes: int, seed: int,
                 n_inner: int = pp.INNER) -> tuple[tuple, dict[tuple, float]]:
    preds = {key: np.zeros(len(y), int) for key in GRID}
    for tr, te in pp.folds(y, groups, n_inner, seed):
        for key, p in fold_predict(X[tr], y[tr], X[te], n_classes, GRID).items():
            preds[key][te] = p
    scores = {key: bal_multi(y, p, n_classes) for key, p in preds.items()}
    return pp.choose(scores), scores


def nested_cv(X: np.ndarray, y: np.ndarray, groups: list[str], n_classes: int, repeats: int = pp.REPEATS,
              outer: int = pp.OUTER, inner: int = pp.INNER, seed0: int = 0) -> dict[str, Any]:
    """外層估成績（重複 repeats 次），內層挑 (k, C1, C2)。"""
    y = np.asarray(y, int)
    per, chosen = [], Counter()
    votes = np.zeros((len(y), n_classes), int)
    for r in range(repeats):
        pred = np.full(len(y), -1)
        for i, (tr, te) in enumerate(pp.folds(y, groups, outer, seed0 + r)):
            gtr = [groups[j] for j in tr]
            best, _ = inner_select(X[tr], y[tr], gtr, n_classes, seed=1000 * (seed0 + r + 1) + i, n_inner=inner)
            chosen[best] += 1
            pred[te] = fold_predict(X[tr], y[tr], X[te], n_classes, [best])[best]
        assert (pred >= 0).all()
        votes[np.arange(len(y)), pred] += 1
        per.append({"bal": bal_multi(y, pred, n_classes), "acc": float(np.mean(pred == y)),
                    "recall": recall_counts(y, pred, n_classes), "pred": pred.tolist()})
    bals = [p["bal"] for p in per]
    vote = np.argmax(votes, axis=1)
    return {"bal_mean": float(np.mean(bals)), "bal_min": float(np.min(bals)), "bal_max": float(np.max(bals)),
            "acc_mean": float(np.mean([p["acc"] for p in per])), "repeats": per, "votes": votes.tolist(),
            "vote_pred": vote.tolist(), "vote_recall": recall_counts(y, vote, n_classes),
            "vote_bal": bal_multi(y, vote, n_classes),
            "chosen": {str(k): v for k, v in chosen.most_common()},
            "k_mean": float(np.mean([k[0] for k in chosen.elements()]))}


def _perm_one(X: np.ndarray, yp: np.ndarray, groups: list[str], n_classes: int) -> float:
    return float(nested_cv(X, yp, groups, n_classes, repeats=1, seed0=0)["bal_mean"])


def permutation_test(X: np.ndarray, y: np.ndarray, groups: list[str], n_classes: int, real: float,
                     n_perm: int = N_PERM, seed: int = PERM_SEED, progress: Any | None = None,
                     n_jobs: int = 1, batch: int = 20) -> dict[str, Any]:
    """把步驟答案打亂 n_perm 次，每次整個流程重跑（1 次重複）；p＝打亂後分數 ≥ 真的分數的比例（加 1 修正）。
    打亂的順序先全部決定好，所以平行或一個一個跑結果相同；平行跑不起來就改回一個一個跑。"""
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(np.asarray(y, int)) for _ in range(n_perm)]
    null: list[float] = []
    parallel = n_jobs != 1 and n_perm > 1
    for i in range(0, n_perm, batch):
        chunk = perms[i:i + batch]
        vals = None
        if parallel:
            try:
                from joblib import Parallel, delayed

                vals = Parallel(n_jobs=n_jobs)(delayed(_perm_one)(X, yp, groups, n_classes) for yp in chunk)
            except Exception as e:  # noqa: BLE001
                log.warning("平行跑不起來（%s），改成一個一個跑", e)
                parallel = False
        if vals is None:
            vals = [_perm_one(X, yp, groups, n_classes) for yp in chunk]
        null += [float(v) for v in vals]
        if progress is not None:
            progress(len(null), n_perm)
    ge = int(sum(v >= real - 1e-12 for v in null))
    return {"n_perm": n_perm, "real": float(real), "null_mean": float(np.mean(null)) if null else None,
            "null_p95": float(np.percentile(null, 95)) if null else None, "n_ge": ge,
            "p": float((ge + 1) / (n_perm + 1))}


def duration_baseline(dur: np.ndarray, y: np.ndarray, groups: list[str], n_classes: int, repeats: int = pp.REPEATS,
                      outer: int = pp.OUTER, seed0: int = 0) -> dict[str, Any]:
    """不看影像：只用段長（秒）一個數字的邏輯迴歸；外層分組和主要做法相同。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, int)
    D = np.nan_to_num(np.asarray(dur, float), nan=float(np.nanmedian(dur)))[:, None]
    bals, accs = [], []
    for r in range(repeats):
        pred = np.full(len(y), -1)
        for tr, te in pp.folds(y, groups, outer, seed0 + r):
            if len(set(y[tr].tolist())) < 2:
                pred[te] = int(y[tr][0])
                continue
            sc = StandardScaler().fit(D[tr])
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0)
            pred[te] = m.fit(sc.transform(D[tr]), y[tr]).predict(sc.transform(D[te]))
        bals.append(bal_multi(y, pred, n_classes))
        accs.append(float(np.mean(pred == y)))
    return {"bal_mean": float(np.mean(bals)), "acc_mean": float(np.mean(accs)),
            "median_s": [float(np.median(D[y == c, 0])) if np.any(y == c) else None for c in range(n_classes)]}


# ---------------------------------------------------------------- 主流程


def evaluate_part(ds: dict[str, Any], V: dict[str, np.ndarray], recipes: tuple[str, ...], n_perm: int,
                  progress: Any | None = None, n_jobs: int = 1) -> dict[str, Any]:
    segs, classes = ds["segments"], ds["classes"]
    n_classes = len(classes)
    if n_classes < 2:
        raise Stop(f"{ds['part']}：可用的失敗步驟不到 2 種")
    y = np.array([classes.index(s["step"]) for s in segs])
    groups = [s["recording_id"] for s in segs]
    res: dict[str, Any] = {"part": ds["part"], "classes": classes, "counts": ds["counts"], "all_counts": ds["all_counts"],
                           "n": len(segs), "n_failures": ds["n_failures"], "n_recordings": len(set(groups)),
                           "no_step": ds["no_step"], "rare": ds["rare"], "missing": ds.get("missing", []),
                           "n_last": int(sum(bool(s["is_last"]) for s in segs)),
                           "majority": {"acc": max(ds["counts"].values()) / len(segs), "bal": 1.0 / n_classes},
                           "recipes": {}}
    for rec in recipes:
        t0 = time.time()
        res["recipes"][rec] = nested_cv(V[rec[:2]], y, groups, n_classes)
        res["recipes"][rec]["seconds"] = round(time.time() - t0, 1)
        log.info("%s %s：%s %.0f%%（%.0f%%～%.0f%%），整體答對 %.0f%%", ds["part"], rec, avg_name(n_classes),
                 res["recipes"][rec]["bal_mean"] * 100, res["recipes"][rec]["bal_min"] * 100,
                 res["recipes"][rec]["bal_max"] * 100, res["recipes"][rec]["acc_mean"] * 100)
    main = res["recipes"][recipes[0]]
    res["duration"] = duration_baseline(np.array([s["duration"] for s in segs]), y, groups, n_classes)
    res["perm"] = (permutation_test(V[recipes[0][:2]], y, groups, n_classes, main["repeats"][0]["bal"], n_perm,
                                    progress=progress, n_jobs=n_jobs) if n_perm > 0 else None)
    res["segments"] = [{"segment_id": s["segment_id"], "step": s["step"], "duration": s["duration"],
                        "n_failed": s["n_failed"], "is_last": bool(s["is_last"]), "sequence": s["sequence"],
                        "vote": classes[main["vote_pred"][i]], "n_right": int(main["votes"][i][y[i]])}
                       for i, s in enumerate(segs)]
    return res


def run(index_dir: str = "data/index", cache_root: str | Path = "experiments/_features", out: Path = OUT,
        n_perm: int = N_PERM, parts: tuple[str, ...] = (PRIMARY, DEV), encs: dict[str, Any] | None = None,
        sheets: bool = True, n_jobs: int = -2) -> dict[str, Any]:
    t0 = time.time()
    out = Path(out)
    own = encs is None
    encs = pp.make_encoders() if own else encs
    res: dict[str, Any] = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_perm": n_perm, "parts": {},
                           "grid": {"k": list(pp.K_GRID), "c1": list(pp.C1_GRID), "c2": list(pp.C2_GRID)},
                           "cv": {"outer": pp.OUTER, "inner": pp.INNER, "repeats": pp.REPEATS}}
    try:
        for part in parts:
            ds = step_dataset(index_dir, part)
            recipes = RECIPES.get(part, ("F2S2",))
            need = tuple(sorted({r[:2] for r in recipes}))
            V, keep, missing = pp.segment_vectors(ds["segments"], encs, Path(cache_root), need=need)
            if missing:
                log.warning("%s：%d 段沒有手腕影格／特徵，不納入（%s）", part, len(missing), "、".join(missing[:5]))
            ds, idx = restrict(ds, keep, missing)
            V = {f: X[idx] for f, X in V.items()}
            if len(ds["classes"]) < 2:
                raise Stop(f"{part}：可用的失敗步驟不到 2 種")

            def progress(i: int, n: int, part: str = part) -> None:
                log.info("%s 打亂答案 %d/%d", part, i, n)

            res["parts"][part] = evaluate_part(ds, V, recipes, n_perm, progress, n_jobs)
            if sheets:
                try:
                    write_sheet(res["parts"][part], out / f"{part.replace(' ', '_')}_frames_by_step.jpg")
                except Exception as e:  # noqa: BLE001  圖只是輔助，畫不出來不影響成績
                    log.warning("%s 的影格拼圖沒畫出來：%s", part, e)
    finally:
        if own:
            for k in ("full", "crop"):
                encs[k].close()
    res["seconds"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(res, out / "report.md")
    return res


# ---------------------------------------------------------------- 圖與報告


def write_sheet(pr: dict[str, Any], path: Path, cols: int = 8, w: int = 200, h: int = 150) -> None:
    """每個步驟一列：失敗段結束後 1 秒的手腕影格（橘框＝模型看的範圍）；下方寫 5 次重複裡答對幾次。"""
    import cv2

    from . import perpart_final as pf

    rows = []
    for c in pr["classes"]:
        tiles = []
        for s in [s for s in pr["segments"] if s["step"] == c]:
            p = pf.hand_image({"part": pr["part"], "segment_id": s["segment_id"]})
            img = cv2.imread(p) if p else None
            if img is None:
                continue
            t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            b = pp.CROP_BOX
            cv2.rectangle(t, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), (46, 122, 224), 2)
            bar = np.full((22, w, 3), 255, np.uint8)
            cv2.putText(bar, f"{c} {s['n_right']}/{pp.REPEATS} -> {s['vote']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (20, 20, 20), 1, cv2.LINE_AA)
            tiles.append(np.vstack([t, bar]))
        for i in range(0, len(tiles), cols):
            line = tiles[i:i + cols]
            line += [np.full((h + 22, w, 3), 255, np.uint8)] * (cols - len(line))
            rows.append(np.hstack(line))
        rows.append(np.full((10, cols * w, 3), 200, np.uint8))
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def verdict(pr: dict[str, Any]) -> str:
    main = next(iter(pr["recipes"].values()))
    name = avg_name(len(pr.get("classes") or [0, 0, 0]))
    p = pr["perm"]["p"] if pr.get("perm") else None
    if p is None:
        return "沒有跑打亂答案檢定，不下結論"
    if main["bal_mean"] >= PASS_BAL and p < PASS_P:
        return f"達到事先的判定（{name} ≥ {PASS_BAL * 100:.0f}% 而且 p < {PASS_P}）：這個做法看得出失敗在哪一步"
    return f"沒有達到事先的判定（{name} ≥ {PASS_BAL * 100:.0f}% 而且 p < {PASS_P}）：只看結尾一格分不出失敗步驟"


def write_report(res: dict[str, Any], path: Path) -> None:
    L = ["# 第 18 包：失敗在哪一步（資料集的步驟標記當答案）", "",
         f"產生時間：{res['generated']}；做法：第 14 包凍結的 F2S2 只換答案（事先登記 `docs/package18_failstep.md`）。", ""]
    for part, pr in res["parts"].items():
        cls = pr["classes"]
        main_name = next(iter(pr["recipes"]))
        main = pr["recipes"][main_name]
        L += [f"## {part}{'（主要）' if part == PRIMARY else '（開發零件，對照）'}", "",
              f"- 失敗段 {pr['n_failures']} 段，用 {pr['n']} 段（{pr['n_recordings']} 支錄影）："
              + "、".join(f"{zh(c)} {pr['counts'][c]}" for c in cls)
              + (f"；沒有失敗步驟不納入 {len(pr['no_step'])} 段" if pr["no_step"] else "")
              + (f"；沒有手腕影格不納入 {len(pr['missing'])} 段" if pr.get("missing") else "")
              + (f"；步驟不到 {MIN_CLASS} 段不納入 {len(pr['rare'])} 段" if pr["rare"] else ""),
              f"- 失敗的那一步就是該段最後一步：{pr['n_last']}/{pr['n']} 段", "",
              f"| 做法 | {avg_name(len(cls))} | 整體答對 | " + " | ".join(f"{zh(c)}答對" for c in cls) + " |",
              "|---|---|---|" + "---|" * len(cls)]
        for name, r in pr["recipes"].items():
            rc = np.mean([[a / b if b else np.nan for a, b in rep["recall"]] for rep in r["repeats"]], axis=0)
            L.append(f"| {name}{'（主要）' if name == main_name else '（參考）'} | {_pct(r['bal_mean'])}"
                     f"（{_pct(r['bal_min'])}～{_pct(r['bal_max'])}） | {_pct(r['acc_mean'])} | "
                     + " | ".join(_pct(float(x)) for x in rc) + " |")
        d = pr["duration"]
        L += [f"| 只看段長（不看影像） | {_pct(d['bal_mean'])} | {_pct(d['acc_mean'])} | " + " | ".join("" for _ in cls) + " |",
              f"| 一律答{zh(cls[0])} | {_pct(pr['majority']['bal'])} | {_pct(pr['majority']['acc'])} | "
              + " | ".join("100%" if i == 0 else "0%" for i in range(len(cls))) + " |", "",
              f"- {avg_name(len(cls))}＝每個步驟答對率的平均；括號是 5 次重複的最低～最高。",
              "- 段長中位數：" + "、".join(f"{zh(c)} {d['median_s'][i]:.1f} 秒" for i, c in enumerate(cls)
                                       if d["median_s"][i] is not None)]
        if pr.get("perm"):
            pm = pr["perm"]
            L.append(f"- 打亂答案檢定（{main_name}，{pm['n_perm']} 次）：真的分數（第 1 次重複）{_pct(pm['real'])}；"
                     f"打亂後平均 {_pct(pm['null_mean'])}、95% 的打亂結果低於 {_pct(pm['null_p95'])}；"
                     f"打亂後 ≥ 真的分數 {pm['n_ge']} 次 → p＝{pm['p']:.3f}")
        if part == PRIMARY:
            L.append(f"- **判定**：{verdict(pr)}")
        L += [f"- 挑到的 k 平均 {main['k_mean']:.1f}；最常挑到的 (k, C1, C2)：{list(main['chosen'].items())[:3]}", "",
              f"### {part}：答案 × 模型的答案（{main_name}，每段取 5 次重複的多數）", "",
              "| 答案＼模型答 | " + " | ".join(zh(c) for c in cls) + " | 答對 |", "|---|" + "---|" * (len(cls) + 1)]
        vp = [cls.index(s["vote"]) for s in pr["segments"]]
        yt = [cls.index(s["step"]) for s in pr["segments"]]
        for i, c in enumerate(cls):
            row = [sum(1 for a, b in zip(yt, vp) if a == i and b == j) for j in range(len(cls))]
            L.append(f"| {zh(c)}（{pr['counts'][c]}） | " + " | ".join(str(v) for v in row) + f" | {row[i]}/{pr['counts'][c]} |")
        L += ["", f"### {part}：每段", "", "| 段 | 答案 | 模型（多數） | 5 次裡答對 | 段長（秒） | 步驟順序 |", "|---|---|---|---|---|---|"]
        for s in sorted(pr["segments"], key=lambda s: (cls.index(s["step"]), s["segment_id"])):
            L.append(f"| {s['segment_id']} | {zh(s['step'])} | {zh(s['vote'])} | {s['n_right']}/{pp.REPEATS} | "
                     f"{s['duration']:.1f} | {s['sequence']} |")
        L += ["", f"- 圖：`{part.replace(' ', '_')}_frames_by_step.jpg`（每個步驟一列；下方「步驟 答對次數/5 -> 模型的多數答案」）", ""]
    L += ["## 注意", "",
          "- 失敗的那一步通常就是段的最後一步，所以分類器實際上在學「流程停在哪裡」。",
          "- 段數很少（最小的一類只有幾段），1 段答對或答錯就差很多；成績只能當參考。",
          "- 打亂答案檢定只用第 1 次重複的分數，和表上的 5 次平均不完全一樣。",
          f"- 總共 {res['seconds']:.0f} 秒。", ""]
    path.write_text("\n".join(L), encoding="utf-8")
