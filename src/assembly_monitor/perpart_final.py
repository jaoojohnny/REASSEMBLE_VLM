"""每零件分類器的「最終模型」：用凍結的做法（configs/perpart_selected.yaml，F2S2）在一個零件全部的段訓練一次並存檔。

第 14 包的交叉驗證只是估成績：每折訓練一個臨時模型，算完分數就丟掉。這裡做的是之後判斷新段要用的那一個模型。
- 參數（k、L1 強度、L2 強度）用和第 14 包內層相同的方法挑：全部段依錄影分組 4 折，兩類平均最高；同分選特徵少、C 小的。
- 存成 .npz（挑中特徵的平均與標準差、特徵編號、邏輯迴歸權重與截距）＋ .json 說明；預測只要 numpy，不依賴 sklearn 版本。
- 訓練段上的答對率只是檢查（同一批段，會偏高）；預期成績看第 14 包的交叉驗證。
- 答案編碼和第 14 包相同：1＝失敗、0＝成功。
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import perpart as pp

log = logging.getLogger("perpart_final")

MODEL_DIR = Path("models/perpart_final")
REPORT_DIR = Path("experiments/perpart_final")
P14_RESULTS = (Path("experiments/p14_perpart_test/results.json"),)
CLS_DIM = 768   # DINOv2-base：前 768 維＝CLS，後 768 維＝小區塊平均


def slug(part: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", part).strip("_")


def check_selected(selected: dict[str, Any]) -> None:
    """凍結設定必須是程式支援的做法，而且參數格點和 perpart.py 相同（避免兩邊不一致）。"""
    if selected.get("recipe") != "F2S2":
        raise ValueError(f"凍結的做法是 {selected.get('recipe')}；最終模型目前只支援 F2S2")
    same = (tuple(selected.get("k_grid", ())) == tuple(pp.K_GRID)
            and tuple(float(c) for c in selected.get("c1_grid", ())) == tuple(pp.C1_GRID)
            and tuple(float(c) for c in selected.get("c2_grid", ())) == tuple(pp.C2_GRID)
            and int(selected.get("inner", -1)) == pp.INNER
            and list(selected.get("crop_box", [])) == list(pp.CROP_BOX))
    if not same:
        raise ValueError("configs/perpart_selected.yaml 的參數格點和 perpart.py 不一致")


# ---------------------------------------------------------------- 訓練與預測（只要特徵，不需要影格、GPU）


def fit_final(X: np.ndarray, y: np.ndarray, groups: list[str], seed: int = 0) -> dict[str, Any]:
    """F2S2：內層 4 折挑 (k, C1, C2) → 全部段標準化 → L1 排序取前 k 個 → L2 邏輯迴歸。"""
    from sklearn.preprocessing import StandardScaler

    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(set(y.tolist())) < 2:
        raise ValueError("只有一種答案（全是成功或全是失敗），不能訓練")
    best, scores = pp.inner_select(X, y, list(groups), "S2", seed=seed, n_inner=pp.INNER)
    k, c1, c2 = best
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    sel = pp._rank_l1(Z, y, c1)[:k]
    if len(sel) == 0:
        raise ValueError("L1 沒有留下任何特徵")
    lr = pp._lr(c2).fit(Z[:, sel], y)
    if list(lr.classes_) != [0, 1]:
        raise ValueError(f"答案類別應該是 [0, 1]，得到 {list(lr.classes_)}")
    top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return {"features": sel.astype(np.int64), "mean": sc.mean_[sel].astype(np.float64),
            "scale": sc.scale_[sel].astype(np.float64), "coef": lr.coef_[0].astype(np.float64),
            "intercept": float(lr.intercept_[0]), "k": int(k), "c1": float(c1), "c2": float(c2),
            "inner_bal": float(scores[best]), "inner_top": [[list(key), float(v)] for key, v in top],
            "dim": int(X.shape[1])}


def decision(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """邏輯迴歸的 logit：> 0 判失敗（和 sklearn 的 predict 相同）。"""
    X = np.atleast_2d(np.nan_to_num(np.asarray(X, float), nan=0.0))
    if X.shape[1] != int(model["dim"]):
        raise ValueError(f"特徵維度應該是 {model['dim']}，得到 {X.shape[1]}")
    Z = (X[:, model["features"]] - model["mean"]) / model["scale"]
    return Z @ model["coef"] + model["intercept"]


def predict_proba(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """失敗的機率（0～1）。"""
    return 1.0 / (1.0 + np.exp(-decision(model, X)))


def predict(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """1＝失敗、0＝成功。"""
    return (decision(model, X) > 0).astype(int)


ARRAYS = ("features", "mean", "scale", "coef")


def save_model(model: dict[str, Any], meta: dict[str, Any], stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz, js = stem.with_suffix(".npz"), stem.with_suffix(".json")
    np.savez(npz, **{a: model[a] for a in ARRAYS}, intercept=np.array([model["intercept"]]),
             dim=np.array([model["dim"]]))
    info = {**meta, **{k: v for k, v in model.items() if k not in ARRAYS}}
    js.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz, js


def load_model(stem: str | Path) -> dict[str, Any]:
    stem = Path(stem)
    with np.load(stem.with_suffix(".npz")) as z:
        model = {a: z[a] for a in ARRAYS}
        model["intercept"] = float(z["intercept"][0])
        model["dim"] = int(z["dim"][0])
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    return {**meta, **model}


# ---------------------------------------------------------------- 影格 → 特徵（只在使用者電腦跑：要影格、GPU）


def hand_image(seg: dict[str, Any], frames_root: str = "data/derived/frames") -> str | None:
    """這段「結束後 1 秒」的手腕影格路徑（第 14 包擷取的那一格）。"""
    from . import probe, vlm

    cfg = pp.feature_cfg(("hand",))
    cfg["frames_root"] = frames_root
    si = vlm.load_segment_frames(probe.frames_dir(cfg, seg["part"]), pp.SAMPLING_TEMPLATE.format(n=1),
                                 seg["segment_id"], ["hand"])
    if si is None or not si.frames:
        return None
    return si.frames[0].path


def image_vector(crop_encoder: Any, path: str) -> np.ndarray:
    """一張手腕影格 → 裁切中間 60% → DINOv2（CLS＋小區塊平均）＝ 1,536 維；不讀也不寫快取。"""
    return np.nan_to_num(np.asarray(crop_encoder.encode([path]), float)[0], nan=0.0)


def predict_image(model: dict[str, Any], crop_encoder: Any, path: str) -> dict[str, Any]:
    """判斷一張新的手腕影格：回傳失敗機率與答案。crop_encoder＝perpart.make_encoders()["crop"]。"""
    p = float(predict_proba(model, image_vector(crop_encoder, path)[None])[0])
    return {"p_failure": p, "pred": "失敗" if p > 0.5 else "成功"}


# ---------------------------------------------------------------- 一個零件：訓練、檢查、存檔、報告


def cv_reference(part: str, paths: tuple[Path, ...] = P14_RESULTS) -> dict[str, Any] | None:
    """第 14 包交叉驗證的成績（預期成績）。"""
    for p in paths:
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        d = (r.get("test") or {}).get(part)
        if d and d.get("rec"):
            return {"bal_mean": d["rec"]["bal_mean"], "bal_min": d["rec"]["bal_min"], "bal_max": d["rec"]["bal_max"],
                    "date_bal": (d.get("date") or {}).get("bal_mean"), "chosen": d["rec"].get("chosen"),
                    "source": str(p)}
    return None


def train_part(part: str, index_dir: str = "data/index", cache_root: str | Path = "experiments/_features",
               selected_path: str = "configs/perpart_selected.yaml", model_dir: Path = MODEL_DIR,
               report_dir: Path = REPORT_DIR, check_encode: int = 3, encs: dict[str, Any] | None = None) -> dict[str, Any]:
    if part in pp.RESERVED:
        raise ValueError(f"{part} 是保留零件，不碰")
    if part not in pp.DEV_PARTS + pp.TEST_PARTS:
        raise ValueError(f"不認得的零件：{part}")
    selected = pp.load_selected(selected_path)
    check_selected(selected)
    t0 = time.time()
    own = encs is None
    encs = pp.make_encoders() if own else encs
    try:
        segs = pp.part_segments(index_dir, part)
        V, keep, missing = pp.segment_vectors(segs, encs, Path(cache_root), need=("F2",))
        if missing:
            raise ValueError(f"{len(missing)} 段沒有影格／特徵（例如 {missing[0]}）：先跑 run_perpart_test.bat 的特徵步驟")
        X = V["F2"]
        y = np.array([s["label"] for s in keep])
        recs = [s["recording_id"] for s in keep]
        model = fit_final(X, y, recs)
        pred = predict(model, X)
        res: dict[str, Any] = {"part": part, "recipe": selected["recipe"], "frozen_at": selected.get("frozen_at"),
                               "n": len(keep), "n_failure": int(y.sum()), "n_recordings": len(set(recs)),
                               "train_bal": pp.bal_acc(y, pred), "train_counts": pp.counts(y, pred),
                               "cv": cv_reference(part), "ids": [s["segment_id"] for s in keep]}
        res["check"] = encode_check(model, X, keep, encs["crop"], check_encode) if check_encode > 0 else None
    finally:
        if own:
            for k in ("full", "crop"):
                encs[k].close()
    meta = {"part": part, "recipe": res["recipe"], "frozen_at": res["frozen_at"], "label": "1＝失敗、0＝成功",
            "input": "段結束後 1 秒的手腕影格（hand），中間 60%×60% 裁切，DINOv2-base CLS＋小區塊平均（1,536 維）",
            "encoder": selected.get("encoder"), "crop_box": list(pp.CROP_BOX), "sampling": pp.SAMPLING_TEMPLATE.format(n=1),
            "n": res["n"], "n_failure": res["n_failure"], "segment_ids": res["ids"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "train_bal": res["train_bal"], "cv": res["cv"]}
    stem = Path(model_dir) / slug(part)
    npz, js = save_model(model, meta, stem)
    res.update(model={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()},
               files=[str(npz), str(js)], seconds=round(time.time() - t0, 1))
    reloaded = load_model(stem)
    res["reload_same"] = bool(np.array_equal(predict(reloaded, X), pred))
    write_report(res, Path(report_dir) / f"{slug(part)}.md")
    (Path(report_dir) / f"{slug(part)}.json").write_text(json.dumps(res, ensure_ascii=False, default=str),
                                                          encoding="utf-8")
    return res


def encode_check(model: dict[str, Any], X: np.ndarray, keep: list[dict[str, Any]], crop_encoder: Any,
                 n: int) -> dict[str, Any]:
    """從影格重新算特徵（模擬一段新的段），和快取的特徵比：機率差多少、答案是否相同。取失敗段和成功段各一部分。"""
    fails = [i for i, s in enumerate(keep) if s["label"] == 1]
    oks = [i for i, s in enumerate(keep) if s["label"] == 0]
    pick = (fails[: (n + 1) // 2] + oks[: n // 2])[:n]
    rows = []
    for i in pick:
        path = hand_image(keep[i])
        if path is None:
            rows.append({"segment_id": keep[i]["segment_id"], "error": "找不到影格"})
            continue
        fresh = image_vector(crop_encoder, path)
        pc = float(predict_proba(model, X[i][None])[0])
        pf = float(predict_proba(model, fresh[None])[0])
        rows.append({"segment_id": keep[i]["segment_id"], "label": int(keep[i]["label"]), "p_cached": pc,
                     "p_fresh": pf, "same": (pc > 0.5) == (pf > 0.5),
                     "max_feature_diff": float(np.max(np.abs(fresh - np.nan_to_num(X[i], nan=0.0))))})
    ok = [r for r in rows if "error" not in r]
    return {"rows": rows, "all_same": bool(ok) and len(ok) == len(rows) and all(r["same"] for r in ok),
            "max_p_diff": max((abs(r["p_cached"] - r["p_fresh"]) for r in ok), default=None)}


def _feature_name(j: int) -> str:
    return f"CLS 第 {j} 維" if j < CLS_DIM else f"小區塊平均 第 {j - CLS_DIM} 維"


def write_report(res: dict[str, Any], path: Path) -> None:
    m, c, tc = res["model"], res["cv"], res["train_counts"]
    L = [f"# {res['part']}：每零件分類器的最終模型（{res['recipe']}）", "",
         f"產生時間：{time.strftime('%Y-%m-%dT%H:%M:%S%z')}；做法凍結於 {res['frozen_at']}（`configs/perpart_selected.yaml`，沒有改）。", "",
         "第 14 包的交叉驗證只估成績、模型用完就丟；這份是用全部段訓練一次、存下來給之後判斷新段用的模型。", "",
         "## 訓練資料", "",
         f"- {res['n']} 段（失敗 {res['n_failure']}、成功 {res['n'] - res['n_failure']}），{res['n_recordings']} 支錄影；全部都用來訓練。",
         "- 輸入：段結束後 1 秒的手腕影格，中間 60%×60% 裁切 → DINOv2-base（不訓練）→ 1,536 維。", "",
         "## 挑到的參數（全部段依錄影分組 4 折，和第 14 包內層相同的方法）", "",
         f"- k＝{m['k']}（L1 實際留下 {len(m['features'])} 個特徵）、L1 強度 C1＝{m['c1']}、L2 強度 C2＝{m['c2']}；"
         f"這組在 4 折的兩類平均 {m['inner_bal'] * 100:.1f}%",
         "- 4 折分數前 5 組（k, C1, C2）：" + "；".join(f"({k[0]}, {k[1]}, {k[2]}) {v * 100:.1f}%" for k, v in m["inner_top"]), ""]
    if c and c.get("chosen"):
        L += [f"- 參考：第 14 包 25 次外層訓練各自挑到的組合 {c['chosen']}（每次不一樣是正常的：段少、好幾組分數接近）", ""]
    L += ["## 挑中的特徵", "", "| 順序 | 特徵 | 權重（正＝越像失敗） |", "|---|---|---|"]
    order = np.argsort(-np.abs(np.array(m["coef"])))
    for r, i in enumerate(order, 1):
        L.append(f"| {r} | {_feature_name(int(m['features'][i]))} | {m['coef'][i]:+.3f} |")
    L += ["", "這些是 DINOv2 的抽象維度，人看不出代表什麼；所以這個模型只能說成功／失敗，不能解釋原因。", "",
          "## 結果", "",
          f"- 訓練段上（同一批段，會偏高，只是檢查）：失敗段答對 {tc['ok_failure']}/{tc['n_failure']}、"
          f"成功段答對 {tc['ok_success']}/{tc['n_success']}，兩類平均 {res['train_bal'] * 100:.1f}%"]
    if c:
        date = f"、依日期分組 {c['date_bal'] * 100:.0f}%" if c.get("date_bal") is not None else ""
        L.append(f"- 預期成績（第 14 包交叉驗證）：兩類平均 {c['bal_mean'] * 100:.0f}%（{c['bal_min'] * 100:.0f}%～"
                 f"{c['bal_max'] * 100:.0f}%）{date}")
    L.append(f"- 存檔後重新讀入，預測完全相同：{'是' if res['reload_same'] else '否（要檢查）'}")
    ch = res.get("check")
    if ch:
        L += ["", "## 從影格重新算特徵的檢查（模擬判斷一段新的段）", "",
              "| 段 | 答案 | 快取特徵的失敗機率 | 重新算的失敗機率 | 答案相同 |", "|---|---|---|---|---|"]
        for r in ch["rows"]:
            if "error" in r:
                L.append(f"| {r['segment_id']} | – | {r['error']} | – | – |")
            else:
                L.append(f"| {r['segment_id']} | {'失敗' if r['label'] else '成功'} | {r['p_cached']:.3f} | "
                         f"{r['p_fresh']:.3f} | {'是' if r['same'] else '否'} |")
        mp = ch.get("max_p_diff")
        L += ["", f"全部相同：{'是' if ch['all_same'] else '否'}；機率最大差 {mp:.4f}" if mp is not None else "沒有可比較的段"]
    L += ["", "## 檔案與用法", "",
          *[f"- `{f}`" for f in res["files"]], "",
          "```python",
          "from assembly_monitor import perpart as pp, perpart_final as pf",
          f"model = pf.load_model('models/perpart_final/{slug(res['part'])}')",
          "enc = pp.make_encoders()['crop']",
          "print(pf.predict_image(model, enc, r'<新的手腕影格.jpg>'))   # {'p_failure': 0.93, 'pred': '失敗'}",
          "```", "",
          "## 注意", "",
          f"- {res['part']} 全部的段都用來訓練了，沒有剩下沒看過的段可以測；要確認這個模型的準確度，需要新的錄影。",
          "- 新影格要和訓練時同一種：手腕相機、段結束後 1 秒、同樣的裁切；換相機位置或時間點，結果不保證。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
