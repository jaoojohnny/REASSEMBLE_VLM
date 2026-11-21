"""第 14 包：每種零件各自一個小分類器（依錄影分組的巢狀交叉驗證，特徵 10 個以內）。

使用者 2026-11-09 決定（同學的建議）：
  - 每種零件只用自己的資料訓練、自己的資料評估；資料少，所以用交叉驗證。
  - 特徵 10 個以內（怕過擬合）；可以先用 L1 讓模型自己挑特徵、刪掉不重要的，再用 L2 重新擬合。
  - 巢狀交叉驗證：外層估成績、內層挑參數；所有前處理（標準化、PCA、L1 挑選）都只在訓練的那幾折上擬合。
  - 保留零件 round peg 4、small gear 不做；做法先在 5 個開發零件（BNC、square peg 2、round peg 3、D-SUB、
    medium gear）比較並凍結，再原封不動套到其餘 10 個零件（之前用過的測試零件）跑一次。

答案：資料集的成功／失敗標記（BNC 的人工答案和標記 86/86 一致）。y＝1 是失敗、0 是成功。
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .util import now_iso

log = logging.getLogger(__name__)

DEV_PARTS = ("BNC", "square peg 2", "round peg 3", "D-SUB", "medium gear")
TEST_PARTS = ("USB", "bolt 4", "Ethernet", "square peg 1", "square peg 3", "square peg 4", "round peg 1",
              "round peg 2", "waterproof", "large gear")
RESERVED = ("round peg 4", "small gear")
CAMS = ("hama1", "hama2", "hand")
SAMPLING_TEMPLATE = "uniform_n{n}_sb0.5_eb1_jpg"   # 和第 6–10 包的小分類器相同：段結束後 1 秒的那一格
K_GRID = (2, 4, 6, 8, 10)
C2_GRID = (0.01, 0.1, 1.0)
C1_GRID = (0.05, 0.2, 1.0)
OUTER, INNER, REPEATS = 5, 4, 5
CLOSE = 0.02
FEATURES = {"F1": "全畫面：3 台相機（4,608 維）", "F2": "只看接頭：手腕相機中間 60%（1,536 維）",
            "F3": "外部全畫面＋手腕裁切（4,608 維）"}
SELECTORS = {"S1": "PCA 取前 k 個", "S2": "L1 挑 k 個 → L2 重新擬合"}
RECIPES = tuple(f + s for f in ("F2", "F1", "F3") for s in ("S1", "S2"))  # 「簡單」的在前：維度少、不看答案挑特徵
ENCODER = {"hf_id": "facebook/dinov2-base", "revision": "main", "dtype": "float16", "batch_size": 32,
           "processor_kwargs": {"do_center_crop": False, "size": {"height": 224, "width": 294}}}
CROP_BOX = (0.2, 0.2, 0.8, 0.8)   # 和 Skill 第二／三版的手腕固定裁切相同
ENCODER_CROP = dict(ENCODER, cache_tag="wristcrop_02_08", crop_box=list(CROP_BOX))


# ---------------------------------------------------------------- 段與答案


def part_segments(index_dir: str | Path, part: str) -> list[dict[str, Any]]:
    """索引裡某個零件的插入段（驗證 ok／warn、有成功／失敗標記）。"""
    out = []
    with (Path(index_dir) / "segments.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("action") != "insert" or r.get("part") != part or r.get("validation_status") not in ("ok", "warn"):
                continue
            lab = {"True": 0, "False": 1}.get(r.get("success"))
            if lab is None:
                continue
            out.append({"segment_id": f"{r['recording_id']}__seg{int(r['segment_id'])}", "recording_id": r["recording_id"],
                        "part": part, "label": lab})
    return sorted(out, key=lambda s: s["segment_id"])


def bal_acc(y: np.ndarray, p: np.ndarray) -> float:
    """兩類平均（失敗段答對率與成功段答對率的平均）；只有一類時就是那一類的答對率。"""
    y, p = np.asarray(y), np.asarray(p)
    rs = [float(np.mean(p[y == c] == c)) for c in (0, 1) if np.any(y == c)]
    return float(np.mean(rs)) if rs else float("nan")


def counts(y: np.ndarray, p: np.ndarray) -> dict[str, int]:
    y, p = np.asarray(y), np.asarray(p)
    return {"ok_failure": int(np.sum((y == 1) & (p == 1))), "n_failure": int(np.sum(y == 1)),
            "ok_success": int(np.sum((y == 0) & (p == 0))), "n_success": int(np.sum(y == 0))}


# ---------------------------------------------------------------- 交叉驗證


def folds(y: np.ndarray, groups: list[str], n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """依錄影分組、按成敗分層的 k 折（同一支錄影不會同時在訓練和測試）。"""
    import warnings

    from sklearn.model_selection import StratifiedGroupKFold

    k = max(2, min(n_splits, len(set(groups))))
    with warnings.catch_warnings():  # 失敗段比折數少時 sklearn 會警告，照樣切
        warnings.simplefilter("ignore", UserWarning)
        return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
                    .split(np.zeros(len(y)), y, groups))


def _lr(c: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=0)


def _l1_model(c1: float):
    """L1 邏輯迴歸；scikit-learn 1.8 起改用 l1_ratio=1（penalty 參數已棄用），舊版用 penalty="l1"。"""
    import sklearn
    from sklearn.linear_model import LogisticRegression

    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=c1, class_weight="balanced", max_iter=5000,
                                  random_state=0)
    return LogisticRegression(penalty="l1", solver="liblinear", C=c1, class_weight="balanced", max_iter=5000,
                              random_state=0)


def _rank_l1(Z: np.ndarray, y: np.ndarray, c1: float) -> np.ndarray:
    """L1 邏輯迴歸留下的特徵，依係數大小排序（係數全為 0 時回傳空陣列）。"""
    m = _l1_model(c1).fit(Z, y)
    w = np.abs(m.coef_[0])
    nz = np.flatnonzero(w > 0)
    return nz[np.argsort(-w[nz], kind="stable")]


def _majority(y: np.ndarray) -> int:
    return int(np.sum(y == 1) > np.sum(y == 0))


def grid_keys(selector: str) -> list[tuple]:
    if selector == "S1":
        return [(k, c2) for k in K_GRID for c2 in C2_GRID]
    return [(k, c1, c2) for k in K_GRID for c1 in C1_GRID for c2 in C2_GRID]


def _fold_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, selector: str,
                  keys: list[tuple]) -> dict[tuple, np.ndarray]:
    """在一份訓練資料上把所有參數組合都擬合一次，回傳每組對測試資料的預測。共用標準化、PCA、L1 排序。"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    out: dict[tuple, np.ndarray] = {}
    if len(set(ytr.tolist())) < 2:
        return {key: np.full(len(Xte), int(ytr[0])) for key in keys}
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    if selector == "S1":
        kmax = min(max(k for k, *_ in keys), len(ytr) - 1, Xtr.shape[1])
        pca = PCA(n_components=kmax, svd_solver="full").fit(Ztr)
        Ptr, Pte = pca.transform(Ztr), pca.transform(Zte)
        for k, c2 in keys:
            kk = min(k, kmax)
            out[(k, c2)] = _lr(c2).fit(Ptr[:, :kk], ytr).predict(Pte[:, :kk])
        return out
    ranks = {c1: _rank_l1(Ztr, ytr, c1) for c1 in sorted({key[1] for key in keys})}
    for k, c1, c2 in keys:
        sel = ranks[c1][:k]
        if len(sel) == 0:
            out[(k, c1, c2)] = np.full(len(Xte), _majority(ytr))
        else:
            out[(k, c1, c2)] = _lr(c2).fit(Ztr[:, sel], ytr).predict(Zte[:, sel])
    return out


def choose(scores: dict[tuple, float]) -> tuple:
    """分數最高；同分選特徵少、正則化強（C 小）的。"""
    return max(scores, key=lambda key: (round(scores[key], 9), -key[0], *(-v for v in key[1:])))


def inner_select(X: np.ndarray, y: np.ndarray, groups: list[str], selector: str, seed: int,
                 n_inner: int = INNER) -> tuple[tuple, dict[tuple, float]]:
    keys = grid_keys(selector)
    preds = {key: np.zeros(len(y), int) for key in keys}
    for tr, te in folds(y, groups, n_inner, seed):
        for key, p in _fold_predict(X[tr], y[tr], X[te], selector, keys).items():
            preds[key][te] = p
    scores = {key: bal_acc(y, p) for key, p in preds.items()}
    return choose(scores), scores


def nested_cv(X: np.ndarray, y: np.ndarray, groups: list[str], selector: str, repeats: int = REPEATS,
              outer: int = OUTER, inner: int = INNER, seed0: int = 0) -> dict[str, Any]:
    """外層 outer 折估成績（重複 repeats 次，每次換分組），內層 inner 折挑參數。"""
    y = np.asarray(y, int)
    per, chosen = [], Counter()
    fail_votes = np.zeros(len(y))
    for r in range(repeats):
        pred = np.full(len(y), -1)
        for i, (tr, te) in enumerate(folds(y, groups, outer, seed0 + r)):
            gtr = [groups[j] for j in tr]
            best, _ = inner_select(X[tr], y[tr], gtr, selector, seed=1000 * (seed0 + r + 1) + i, n_inner=inner)
            chosen[best] += 1
            pred[te] = _fold_predict(X[tr], y[tr], X[te], selector, [best])[best]
        assert (pred >= 0).all()
        fail_votes += pred
        per.append({"bal": bal_acc(y, pred), **counts(y, pred), "pred": pred.tolist()})
    bals = [p["bal"] for p in per]
    return {"selector": selector, "bal_mean": float(np.mean(bals)), "bal_min": float(np.min(bals)),
            "bal_max": float(np.max(bals)), "repeats": per, "fail_rate": (fail_votes / repeats).tolist(),
            "chosen": {str(k): v for k, v in chosen.most_common()},
            "k_mean": float(np.mean([k[0] for k in chosen.elements()]))}


def select_recipe(dev_scores: dict[str, dict[str, float]], order: tuple[str, ...] = RECIPES,
                  close: float = CLOSE) -> dict[str, Any]:
    """事先寫好的規則：5 個開發零件平均的兩類平均最高；和最高差 2 點以內選「簡單」的（RECIPES 的順序）。"""
    avg = {r: float(np.mean(list(dev_scores[r].values()))) for r in order if r in dev_scores}
    best = max(avg.values())
    pick = next(r for r in order if r in avg and avg[r] >= best - close - 1e-12)
    return {"recipe": pick, "avg": avg, "best_avg": best, "best_recipe": max(avg, key=avg.get)}


# ---------------------------------------------------------------- 特徵（只在使用者電腦跑：要影格、GPU）


def feature_cfg(cameras: tuple[str, ...] = CAMS) -> dict[str, Any]:
    return {"frames_root": "data/derived/frames", "action": "insert", "sampling_template": SAMPLING_TEMPLATE,
            "cameras": list(cameras)}


def make_encoders() -> dict[str, Any]:
    """F1 用原本的 DINOv2 快取（和第 6–10 包相同）；F2 是手腕裁切的新快取。需要時才載入模型。"""
    from . import probe

    class Lazy:
        def __init__(self, enc: Any):
            self.enc, self.loaded = enc, False
            self.key = enc.key

        def encode(self, paths: list[str]) -> np.ndarray:
            if not self.loaded:
                self.enc.load()
                self.loaded = True
            return self.enc.encode(paths)

        def close(self) -> None:
            if self.loaded:
                self.enc.close()

    class CropEncoder(probe.TransformersImageEncoder):
        def encode(self, paths: list[str]) -> np.ndarray:
            from PIL import Image

            torch = self.torch
            box = self.cfg["crop_box"]
            feats = []
            bs = int(self.cfg.get("batch_size", 32))
            for i in range(0, len(paths), bs):
                imgs = []
                for p in paths[i:i + bs]:
                    im = Image.open(p).convert("RGB")
                    w, h = im.size
                    imgs.append(im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))))
                inputs = self.processor(images=imgs, return_tensors="pt", **(self.cfg.get("processor_kwargs") or {}))
                inputs = inputs.to(self.device, dtype=self.dtype)
                with torch.inference_mode():
                    o = self.net(**inputs)
                h_ = o.last_hidden_state.float()
                feats.append(torch.cat([h_[:, 0], h_[:, 1:].mean(dim=1)], dim=1).cpu().numpy())
            return np.concatenate(feats).astype(np.float32)

    full = probe.make_encoder(ENCODER)
    crop = CropEncoder(ENCODER_CROP)
    return {"full": Lazy(full), "crop": Lazy(crop), "_raw": {"full": full, "crop": crop}}


def _check_crop_cache(cache_root: Path, key: str) -> None:
    ident = {"hf_id": ENCODER["hf_id"], "processor_kwargs": ENCODER["processor_kwargs"], "crop_box": list(CROP_BOX),
             "camera": "hand"}
    p = cache_root / key / "_encoder.json"
    if p.exists():
        if json.loads(p.read_text(encoding="utf-8")) != json.loads(json.dumps(ident)):
            raise ValueError(f"{p.parent} 的特徵是用不同設定做的")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ident, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def segment_vectors(segs: list[dict[str, Any]], encs: dict[str, Any], cache_root: Path,
                    need: tuple[str, ...] = ("F1", "F2", "F3")) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[str]]:
    """每段的 F1／F2／F3 向量（有快取就讀）；缺影格的段丟掉並列出。"""
    from . import probe

    probe.check_feature_cache(cache_root, encs["_raw"]["full"], ENCODER)
    if "F2" in need or "F3" in need:
        _check_crop_cache(cache_root, encs["crop"].key)
    cf, ch = feature_cfg(), feature_cfg(("hand",))
    rows: dict[str, list[np.ndarray]] = {f: [] for f in need}
    keep, missing = [], []
    for s in segs:
        full = probe.segment_features(cf, encs["full"], s, 1, cache_root) if ("F1" in need or "F3" in need) else None
        crop = probe.segment_features(ch, encs["crop"], s, 1, cache_root) if ("F2" in need or "F3" in need) else None
        if (("F1" in need or "F3" in need) and full is None) or (("F2" in need or "F3" in need) and crop is None):
            missing.append(s["segment_id"])
            continue
        full = None if full is None else np.nan_to_num(full, nan=0.0)
        crop = None if crop is None else np.nan_to_num(crop, nan=0.0)
        if "F1" in need:
            rows["F1"].append(full[0].reshape(-1))
        if "F2" in need:
            rows["F2"].append(crop[0, 0])
        if "F3" in need:
            rows["F3"].append(np.concatenate([full[0, 0], full[0, 1], crop[0, 0]]))
        keep.append(s)
    return {f: (np.vstack(v) if v else np.zeros((0, 0), np.float32)) for f, v in rows.items()}, keep, missing


def extract_frames(parts: tuple[str, ...], index_dir: str = "data/index", slim_dir: str = "data/slim",
                   frames_root: str = "data/derived/frames", mapping_path: str = "configs/field_mapping.yaml",
                   workers: int = 4) -> dict[str, Any]:
    """補擷取段結束後 1 秒那一格（3 台相機）；已有的略過。"""
    from .extract import run_extraction
    from .mapping import load_mapping

    mapping = load_mapping(mapping_path)
    tot = {"n_done": 0, "n_skipped": 0, "n_missing_frames": 0, "errors": []}
    for part in parts:
        ids = [s["segment_id"] for s in part_segments(index_dir, part)]
        res = run_extraction(mapping=mapping, index_dir=index_dir, slim_dir=slim_dir, out_dir=frames_root, part=part,
                             action="insert", which="perpart", n_frames_list=(1,), start_buffer=0.5, end_buffer=1.0,
                             cameras=list(CAMS), workers=workers, segment_ids=ids)
        log.info("  %-13s %3d 段：新擷取 %d、已有 %d", part, len(ids), res["n_done"], res["n_skipped"])
        for k in ("n_done", "n_skipped", "n_missing_frames"):
            tot[k] += res[k]
        tot["errors"] += res["errors"]
    return tot


# ---------------------------------------------------------------- 對照：一個大分類器、現成模型


def pooled_reference(target: str, index_dir: str, encs: dict[str, Any], cache_root: Path,
                     target_X: np.ndarray, target_segs: list[dict[str, Any]]) -> dict[str, Any]:
    """原本的做法（第 6–10 包）：其他零件全部混在一起訓練一個分類器（全畫面、PCA 32/64、依錄影 5 折挑參數），
    再判這個零件的全部段。排除保留零件，也排除含這個零件的錄影（避免同一場錄影洩漏）。"""
    from . import probe

    recs = {s["recording_id"] for s in target_segs}
    pool = [s for p in DEV_PARTS + TEST_PARTS if p != target for s in part_segments(index_dir, p)
            if s["recording_id"] not in recs]
    V, keep, missing = segment_vectors(pool, encs, cache_root, need=("F1",))
    y = np.array([s["label"] for s in keep])
    tr = probe.train_classifier(V["F1"], y, [s["recording_id"] for s in keep],
                                {"C_grid": [0.01, 0.1, 1.0], "pca_grid": [32, 64], "cv_folds": 5, "seed": 0})
    pred = tr["model"].predict(target_X)
    return {"pred": pred.tolist(), "n_pool": len(keep), "n_pool_failure": int(y.sum()), "missing": len(missing),
            "best": tr["best"]}


ZS_P11 = {"square peg 2": "sqpeg2", "round peg 3": "rpeg3", "D-SUB": "dsub", "medium gear": "mgear"}
ZS_NAMES = {"gpt": "GPT-6.1 Sol", "v3": "Qwen＋Skill 第三版", "prob": "Qwen 機率門檻", "qwen": "Qwen 沒有 Skill"}


def _latest_results(p: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("segment_id"):
                out[r["segment_id"]] = r.get("result") if r.get("status") == "ok" else None
    return out


def zero_shot_answers(part: str, cache: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    """開發零件上已經跑過的現成模型答案（不訓練）：BNC 用第 8 包、其他 4 個用第 11、12 包。"""
    if part == "BNC":
        base = Path("experiments")
        return {"gpt": _latest_results(base / "dev_p8_bnc/gpt61_sol/f4/predictions.jsonl"),
                "qwen": _latest_results(base / "dev_p8_bnc/qwen3vl_8b/f4/predictions.jsonl"),
                "v3": _latest_results(base / "dev_p8_checklistf_bnc/qwen3vl_8b_zoomfix/f4/predictions.jsonl")}
    if part not in ZS_P11:
        return {}
    if "p11" not in cache:
        from . import cascade_dev as cd
        from .clean_dev import FROZEN_TAU

        recs = cd.load_records(cd.load_dev_cfg("configs/cascade_dev.yaml"))
        cache["p11"] = {k: {r["segment_id"]: {"gpt": r.get("gpt"), "v3": r.get("v3_text"),
                                              "prob": cd.first_answer(r, FROZEN_TAU)} for r in v} for k, v in recs.items()}
    d = cache["p11"][ZS_P11[part]]
    return {m: {sid: v[m] for sid, v in d.items()} for m in ("gpt", "v3", "prob")}


def bal_on(ids: list[str], truth: dict[str, int], answers: dict[str, str | None]) -> dict[str, Any]:
    """只算 ids 這些段；沒答案算答錯。"""
    y = np.array([truth[i] for i in ids])
    p = np.array([{"failure": 1, "success": 0}.get(answers.get(i) or "", 1 - truth[i]) for i in ids])
    return {"bal": bal_acc(y, p), **counts(y, p), "n": len(ids)}


# ---------------------------------------------------------------- 開發階段：整體流程與報告


def run_dev(index_dir: str = "data/index", out_dir: str = "experiments/p14_perpart_dev",
            cache_root: str = "experiments/_features", selected_path: str = "configs/perpart_selected.yaml",
            parts: tuple[str, ...] = DEV_PARTS, repeats: int = REPEATS) -> dict[str, Any]:
    out, cache_root_p = Path(out_dir), Path(cache_root)
    encs = make_encoders()
    res: dict[str, Any] = {"generated": now_iso(), "parts": {}, "settings": settings()}
    zs_cache: dict[str, Any] = {}
    try:
        for part in parts:
            t0 = time.time()
            segs = part_segments(index_dir, part)
            V, keep, missing = segment_vectors(segs, encs, cache_root_p)
            y = np.array([s["label"] for s in keep])
            groups = [s["recording_id"] for s in keep]
            ids = [s["segment_id"] for s in keep]
            pr: dict[str, Any] = {"n": len(keep), "n_failure": int(y.sum()), "n_recordings": len(set(groups)),
                                  "missing": missing, "ids": ids, "labels": y.tolist(), "recipes": {}}
            for rcp in RECIPES:
                cv = nested_cv(V[rcp[:2]], y, groups, rcp[2:], repeats=repeats)
                pr["recipes"][rcp] = cv
                log.info("  %-13s %s：兩類平均 %.1f%%（%.1f～%.1f），平均特徵 %.1f 個", part, rcp, cv["bal_mean"] * 100,
                         cv["bal_min"] * 100, cv["bal_max"] * 100, cv["k_mean"])
            ref = pooled_reference(part, index_dir, encs, cache_root_p, V["F1"], keep)
            ref.update(bal=bal_acc(y, np.array(ref["pred"])), **counts(y, np.array(ref["pred"])))
            pr["pooled"] = ref
            truth = dict(zip(ids, y.tolist()))
            zs = zero_shot_answers(part, zs_cache)
            pr["zero_shot"] = {}
            for m, ans in zs.items():
                ov = [i for i in ids if i in ans]
                if not ov:
                    continue
                cls = {r: float(np.mean([bal_acc(np.array([truth[i] for i in ov]),
                                                 np.array([rep["pred"][ids.index(i)] for i in ov]))
                                         for rep in pr["recipes"][r]["repeats"]])) for r in RECIPES}
                pr["zero_shot"][m] = {**bal_on(ov, truth, ans), "classifier_same_segments": cls}
            pr["seconds"] = round(time.time() - t0, 1)
            res["parts"][part] = pr
    finally:
        for k in ("full", "crop"):
            encs[k].close()
    dev_scores = {r: {p: res["parts"][p]["recipes"][r]["bal_mean"] for p in res["parts"]} for r in RECIPES}
    res["selection"] = select_recipe(dev_scores)
    res["selected_config"] = str(write_selected(res, selected_path))
    write_dev_report(res, out)
    return res


def settings() -> dict[str, Any]:
    return {"features": FEATURES, "selectors": SELECTORS, "recipes": list(RECIPES), "k_grid": list(K_GRID),
            "c2_grid": list(C2_GRID), "c1_grid": list(C1_GRID), "outer": OUTER, "inner": INNER, "repeats": REPEATS,
            "sampling": SAMPLING_TEMPLATE.format(n=1), "cameras": list(CAMS), "encoder": ENCODER["hf_id"],
            "crop_box": list(CROP_BOX), "close": CLOSE, "dev_parts": list(DEV_PARTS), "test_parts": list(TEST_PARTS),
            "reserved": list(RESERVED)}


def write_selected(res: dict[str, Any], path: str | Path) -> Path:
    """開發階段選出的做法寫成凍結設定（測試零件只能用這份）。已存在而且做法不同就拒絕覆寫。"""
    import yaml

    p = Path(path)
    sel = res["selection"]
    doc = {"perpart_selected": {"frozen": True, "frozen_at": now_iso(), "recipe": sel["recipe"],
                                "features": FEATURES[sel["recipe"][:2]], "selector": SELECTORS[sel["recipe"][2:]],
                                "dev_avg_balanced_accuracy": {k: round(v, 4) for k, v in sel["avg"].items()},
                                **{k: v for k, v in settings().items() if k not in ("features", "selectors", "recipes")}}}
    if p.exists():
        old = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("perpart_selected") or {}
        if old.get("recipe") != sel["recipe"]:
            raise ValueError(f"{p} 已凍結為 {old.get('recipe')}，這次選到 {sel['recipe']}；不覆寫")
        return p
    head = ("# 第 14 包：在 5 個開發零件依事先寫好的規則選出的做法（自動寫入、凍結）。測試零件只用這份；不要手動改。\n")
    p.write_text(head + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def _pct(x: float | None) -> str:
    return "–" if x is None or x != x else f"{x * 100:.0f}%"


def write_dev_report(res: dict[str, Any], out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    sel = res["selection"]
    parts = list(res["parts"])
    L = ["# 第 14 包（開發階段）：每種零件各自一個小分類器", "", f"產生時間：{res['generated']}", "",
         "每個零件只用自己的段訓練與評估：依錄影分組的巢狀交叉驗證（外層 5 折估成績、內層 4 折挑參數，重複 5 次），"
         "特徵 10 個以內。數字是 5 次重複的平均兩類平均（括號＝最低～最高）。", "",
         "## 做法比較（5 個開發零件）", "",
         "| 做法 | " + " | ".join(parts) + " | 平均 |", "|---|" + "---|" * (len(parts) + 1)]
    for r in RECIPES:
        cells = []
        for p in parts:
            cv = res["parts"][p]["recipes"][r]
            cells.append(f"{_pct(cv['bal_mean'])}（{_pct(cv['bal_min'])}～{_pct(cv['bal_max'])}）")
        star = " ★" if r == sel["recipe"] else ""
        L.append(f"| {r}{star}：{FEATURES[r[:2]].split('（')[0]}＋{SELECTORS[r[2:]]} | " + " | ".join(cells)
                 + f" | {_pct(sel['avg'][r])} |")
    L += ["", f"選出：**{sel['recipe']}**（規則：平均最高；和最高差 2 點以內選比較簡單的）；最高的是 {sel['best_recipe']}"
          f"（{_pct(sel['best_avg'])}）。凍結在 `{res['selected_config']}`。", "",
          "## 每個零件：選出的做法 vs 一個大分類器 vs 現成模型", "",
          "| 零件 | 段（失敗） | 每零件分類器 失敗段答對／成功段答對（平均） | 兩類平均 | 一個大分類器（原本做法） | 一律答成功 |",
          "|---|---|---|---|---|---|"]
    for p in parts:
        pr = res["parts"][p]
        cv = pr["recipes"][sel["recipe"]]
        of = np.mean([x["ok_failure"] for x in cv["repeats"]])
        os_ = np.mean([x["ok_success"] for x in cv["repeats"]])
        ref = pr["pooled"]
        L.append(f"| {p} | {pr['n']}（{pr['n_failure']}） | {of:.1f}/{pr['n_failure']}、{os_:.1f}/{pr['n'] - pr['n_failure']} | "
                 f"{_pct(cv['bal_mean'])} | {ref['ok_failure']}/{ref['n_failure']}、{ref['ok_success']}/{ref['n_success']}"
                 f"（{_pct(ref['bal'])}） | 50% |")
    L += ["", "現成模型（不訓練）在同一批段上的兩類平均（每零件分類器也只算這些段）：", "",
          "| 零件 | 方法 | 段數 | 現成模型 | 每零件分類器（選出的做法） |", "|---|---|---|---|---|"]
    for p in parts:
        for m, z in res["parts"][p]["zero_shot"].items():
            L.append(f"| {p} | {ZS_NAMES[m]} | {z['n']} | {_pct(z['bal'])}（失敗段 {z['ok_failure']}/{z['n_failure']}、"
                     f"成功段 {z['ok_success']}/{z['n_success']}） | {_pct(z['classifier_same_segments'][sel['recipe']])} |")
    L += ["", "## 內層挑到的特徵數", ""]
    for p in parts:
        cv = res["parts"][p]["recipes"][sel["recipe"]]
        L.append(f"- {p}：平均 {cv['k_mean']:.1f} 個；最常選 " + "、".join(list(cv["chosen"])[:3]))
    L += ["", "## 注意", "",
          "- 失敗段少的零件（medium gear 8 段、D-SUB 10 段）每折只有 1–2 段失敗，成績起伏大。",
          "- 這是開發階段：用來選做法。選出的做法凍結後，才在其餘 10 個零件跑一次。",
          "- 一個大分類器的訓練資料排除了保留零件（round peg 4、small gear）和含這個零件的錄影。",
          "- 現成模型的數字來自之前各包（BNC：第 8 包；其他 4 個：第 11、12 包），每零件分類器在同一批段上重算。", ""]
    md = out / "report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    slim = json.loads(json.dumps(res, default=str))
    (out / "results.json").write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    with (out / "per_segment.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "segment_id", "label"] + [f"{r}_fail_rate" for r in RECIPES] + ["pooled"])
        for p in parts:
            pr = res["parts"][p]
            for i, sid in enumerate(pr["ids"]):
                w.writerow([p, sid, "failure" if pr["labels"][i] else "success"]
                           + [round(pr["recipes"][r]["fail_rate"][i], 2) for r in RECIPES]
                           + ["failure" if pr["pooled"]["pred"][i] else "success"])
    return str(md)


# ---------------------------------------------------------------- 測試階段：凍結的做法套到其他 10 個零件＋依日期分組的檢查


TEST_METHOD_NAMES = {
    "gpt61_sol_ft_eb05_f4": "GPT-6.1 Sol", "claude_sonnet55_ft_eb05_f4": "Claude Sonnet 5.5",
    "qwen3vl8b_v3_eb05_f4": "Qwen＋Skill 第三版", "qwen3vl8b_ft_eb05_f4": "Qwen 沒有 Skill",
    "qwen3vl8b_prob_t0999": "Qwen 機率門檻", "qwen3vl8b_wrist_05": "Qwen 手腕第二次觀察",
    "qwen3vl8b_crop_eb05_f4": "Qwen 只裁切", "qwen3vl8b_checklist_eb05_f4": "Qwen 只清單",
    "hybrid_v3_gpt": "兩階段 第三版→GPT", "hybrid_prob_gpt": "兩階段 機率門檻→GPT",
}


def method_name(mid: str) -> str:
    if mid.startswith("probe"):
        return "混合分類器（第 8–10 包原本的）"
    return TEST_METHOD_NAMES.get(mid, mid)


def date_of(recording_id: str) -> str:
    return recording_id[:10]


def load_selected(path: str | Path = "configs/perpart_selected.yaml") -> dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.exists():
        raise ValueError(f"找不到 {p}：要先跑開發階段（run_perpart_dev.bat）選出做法")
    d = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("perpart_selected") or {}
    if not d.get("frozen") or d.get("recipe") not in RECIPES:
        raise ValueError(f"{p} 不是凍結的做法設定")
    return d


def zero_shot_test(part: str, config_dir: str | Path = "configs") -> dict[str, dict[str, str | None]]:
    """這個零件之前當測試集時，各方法（不含一律答成功）的答案。"""
    from .final import load_final_config, method_run_dir

    for f in sorted(Path(config_dir).glob("final_test*.yaml")):
        cfg = load_final_config(f)
        if cfg.get("part") != part:
            continue
        return {m["id"]: _latest_results(method_run_dir(cfg, m) / "predictions.jsonl")
                for m in cfg["methods"] if m["kind"] != "majority"}
    return {}


def date_failures(segs: list[dict[str, Any]]) -> dict[str, list[int]]:
    c: dict[str, list[int]] = {}
    for s in segs:
        v = c.setdefault(date_of(s["recording_id"]), [0, 0])
        v[0] += s["label"]
        v[1] += 1
    return dict(sorted(c.items()))


def evaluate_part(part: str, recipe: str, encs: dict[str, Any], cache_root: Path, index_dir: str,
                  repeats: int = REPEATS, pooled: bool = True) -> dict[str, Any]:
    F, S = recipe[:2], recipe[2:]
    segs = part_segments(index_dir, part)
    need = tuple(sorted({F, "F1"})) if pooled else (F,)
    V, keep, missing = segment_vectors(segs, encs, cache_root, need=need)
    y = np.array([s["label"] for s in keep])
    recs = [s["recording_id"] for s in keep]
    days = [date_of(r) for r in recs]
    out: dict[str, Any] = {"n": len(keep), "n_failure": int(y.sum()), "n_recordings": len(set(recs)),
                           "n_dates": len(set(days)), "missing": missing, "ids": [s["segment_id"] for s in keep],
                           "labels": y.tolist(), "date_failures": date_failures(keep)}
    out["rec"] = nested_cv(V[F], y, recs, S, repeats=repeats)
    out["date"] = nested_cv(V[F], y, days, S, repeats=1)
    if pooled:
        ref = pooled_reference(part, index_dir, encs, cache_root, V["F1"], keep)
        ref.update(bal=bal_acc(y, np.array(ref["pred"])), **counts(y, np.array(ref["pred"])))
        out["pooled"] = ref
    return out


def compare_same_segments(pr: dict[str, Any], answers: dict[str, dict[str, str | None]]) -> dict[str, Any]:
    ids, truth = pr["ids"], dict(zip(pr["ids"], pr["labels"]))
    pos = {sid: i for i, sid in enumerate(ids)}
    out = {}
    for mid, ans in answers.items():
        ov = [i for i in ids if i in ans]
        if not ov:
            continue
        yy = np.array([truth[i] for i in ov])
        cls = float(np.mean([bal_acc(yy, np.array([rep["pred"][pos[i]] for i in ov])) for rep in pr["rec"]["repeats"]]))
        out[mid] = {**bal_on(ov, truth, ans), "classifier_same_segments": cls}
    return out


def run_test(index_dir: str = "data/index", out_dir: str = "experiments/p14_perpart_test",
             cache_root: str = "experiments/_features", selected_path: str = "configs/perpart_selected.yaml",
             repeats: int = REPEATS) -> dict[str, Any]:
    sel = load_selected(selected_path)
    recipe = sel["recipe"]
    encs = make_encoders()
    cr = Path(cache_root)
    res: dict[str, Any] = {"generated": now_iso(), "recipe": recipe, "frozen_at": sel.get("frozen_at"),
                           "dev_date_check": {}, "test": {}}
    try:
        for part in DEV_PARTS:
            pr = evaluate_part(part, recipe, encs, cr, index_dir, repeats=repeats, pooled=False)
            res["dev_date_check"][part] = {k: pr[k] for k in ("n", "n_failure", "n_dates", "date_failures")} | {
                "rec_bal": pr["rec"]["bal_mean"], "date_bal": pr["date"]["bal_mean"],
                "date_counts": {k: pr["date"]["repeats"][0][k] for k in ("ok_failure", "n_failure", "ok_success", "n_success")}}
            log.info("  開發 %-13s 依錄影 %.1f%%、依日期 %.1f%%", part, pr["rec"]["bal_mean"] * 100, pr["date"]["bal_mean"] * 100)
        for part in TEST_PARTS:
            t0 = time.time()
            pr = evaluate_part(part, recipe, encs, cr, index_dir, repeats=repeats)
            pr["zero_shot"] = compare_same_segments(pr, zero_shot_test(part))
            pr["seconds"] = round(time.time() - t0, 1)
            res["test"][part] = pr
            log.info("  測試 %-13s 依錄影 %.1f%%（%.1f～%.1f）、依日期 %.1f%%、混合分類器 %.1f%%", part,
                     pr["rec"]["bal_mean"] * 100, pr["rec"]["bal_min"] * 100, pr["rec"]["bal_max"] * 100,
                     pr["date"]["bal_mean"] * 100, pr["pooled"]["bal"] * 100)
    finally:
        for k in ("full", "crop"):
            encs[k].close()
    t = res["test"]
    res["summary"] = {
        "rec_mean": float(np.mean([t[p]["rec"]["bal_mean"] for p in t])),
        "date_mean": float(np.mean([t[p]["date"]["bal_mean"] for p in t])),
        "pooled_mean": float(np.mean([t[p]["pooled"]["bal"] for p in t])),
        "n_better_than_pooled": int(sum(t[p]["rec"]["bal_mean"] > t[p]["pooled"]["bal"] for p in t)),
        "dev_rec_mean": float(np.mean([v["rec_bal"] for v in res["dev_date_check"].values()])),
        "dev_date_mean": float(np.mean([v["date_bal"] for v in res["dev_date_check"].values()])),
    }
    write_test_report(res, Path(out_dir))
    return res


def write_test_report(res: dict[str, Any], out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    s = res["summary"]
    L = ["# 第 14 包（測試階段）：凍結的做法套到其他 10 個零件", "", f"產生時間：{res['generated']}", "",
         f"做法：**{res['recipe']}**（{FEATURES[res['recipe'][:2]]}＋{SELECTORS[res['recipe'][2:]]}），開發階段凍結於 "
         f"{res['frozen_at']}；測試零件沒有改任何設定。成績＝兩類平均。", "",
         "## 摘要", "",
         f"- 主要比較（事先登記）：10 個測試零件平均，每零件分類器（依錄影分組）**{_pct(s['rec_mean'])}** vs "
         f"一個大的混合分類器 **{_pct(s['pooled_mean'])}**；每零件分類器較好的零件 {s['n_better_than_pooled']}/10。",
         f"- 依日期分組（留一天，比較嚴格）：測試零件平均 {_pct(s['date_mean'])}；開發零件 {_pct(s['dev_date_mean'])}"
         f"（依錄影 {_pct(s['dev_rec_mean'])}）。", "",
         "## 依日期分組的檢查（開發零件）", "",
         "依錄影分組擋不住「同一天的光線、擺設」；依日期分組是拿其他天訓練、測沒看過的那一天。", "",
         "| 零件 | 段（失敗） | 依錄影 | 依日期 | 依日期 失敗段答對／成功段答對 | 各日期失敗／段數 |", "|---|---|---|---|---|---|"]
    for p, v in res["dev_date_check"].items():
        dc = v["date_counts"]
        days = "、".join(f"{d[5:]} {f}/{n}" for d, (f, n) in v["date_failures"].items())
        L.append(f"| {p} | {v['n']}（{v['n_failure']}） | {_pct(v['rec_bal'])} | {_pct(v['date_bal'])} | "
                 f"{dc['ok_failure']}/{dc['n_failure']}、{dc['ok_success']}/{dc['n_success']} | {days} |")
    L += ["", "## 測試零件", "",
          "| 零件 | 段（失敗） | 每零件分類器 依錄影（最低～最高） | 失敗段答對／成功段答對（平均） | 依日期 | 混合分類器 | 一律答成功 |",
          "|---|---|---|---|---|---|---|"]
    for p, pr in res["test"].items():
        cv = pr["rec"]
        of = np.mean([x["ok_failure"] for x in cv["repeats"]])
        os_ = np.mean([x["ok_success"] for x in cv["repeats"]])
        L.append(f"| {p} | {pr['n']}（{pr['n_failure']}） | {_pct(cv['bal_mean'])}（{_pct(cv['bal_min'])}～{_pct(cv['bal_max'])}） | "
                 f"{of:.1f}/{pr['n_failure']}、{os_:.1f}/{pr['n'] - pr['n_failure']} | {_pct(pr['date']['bal_mean'])} | "
                 f"{_pct(pr['pooled']['bal'])} | 50% |")
    L += ["", "## 和之前的方法比（同一批段；每零件分類器也只算這些段）", "",
          "| 零件 | 方法 | 段數 | 之前的方法 | 每零件分類器 |", "|---|---|---|---|---|"]
    for p, pr in res["test"].items():
        for mid, z in pr["zero_shot"].items():
            L.append(f"| {p} | {method_name(mid)} | {z['n']} | {_pct(z['bal'])}（失敗段 {z['ok_failure']}/{z['n_failure']}、"
                     f"成功段 {z['ok_success']}/{z['n_success']}） | {_pct(z['classifier_same_segments'])} |")
    L += ["", "## 注意", "",
          "- 每零件分類器的成績來自交叉驗證（每段由沒看過它的模型判一次）；之前的方法是測試集只跑一次。兩者都在同一批段上算。",
          "- 每零件分類器需要該零件自己有標記的資料；不訓練的 VLM 不需要。兩者回答的問題不同。",
          "- 失敗段少的零件（waterproof 8、large gear 6）成績起伏大。",
          "- 混合分類器＝本包重新訓練的「其他非保留零件混在一起」版本；表中「混合分類器（第 8–10 包原本的）」是當時測試集的答案。", ""]
    md = out / "report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(json.loads(json.dumps(res, default=str)), ensure_ascii=False), encoding="utf-8")
    with (out / "per_segment.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "segment_id", "label", "fail_rate_rec", "pred_date", "pooled"])
        for p, pr in res["test"].items():
            for i, sid in enumerate(pr["ids"]):
                w.writerow([p, sid, "failure" if pr["labels"][i] else "success", round(pr["rec"]["fail_rate"][i], 2),
                            "failure" if pr["date"]["repeats"][0]["pred"][i] else "success",
                            "failure" if pr["pooled"]["pred"][i] else "success"])
    return str(md)
