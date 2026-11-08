"""做法 A：有訓練的對照組。

用「其他零件」的插入片段（資料集成功／失敗標記）訓練小分類器，再到 BNC 上評分，回答：
簡單訓練能不能贏過現成 VLM、能不能套用到沒看過的零件。

- 訓練池：action=insert、零件 ≠ BNC、而且錄影不在 BNC 測試集的 48 支錄影裡（同場景不外洩）。
  只讀測試集的「錄影 ID」來排除，不讀任何測試答案。
- 特徵：凍結的影像編碼器（預設 DINOv2-base）對每張影格取 CLS＋平均 patch；每段依（時間、相機）串接。
- 分類器：標準化 → PCA → 邏輯迴歸（class_weight=balanced）；超參數用「依錄影分組」的交叉驗證挑。
- 輸出格式和第 5 包的 VLM 相同（predictions.jsonl／raw.jsonl／run_config.json），摘要會出現在同一張表。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import vlm
from .util import now_iso

log = logging.getLogger(__name__)
TOOL = "probe-v1"


class ProbeConfigError(ValueError):
    pass


def _slug(s: str) -> str:
    from .extract import _slug as s_

    return s_(s)


def load_probe_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ProbeConfigError(f"找不到 {p}")
    d = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("probe") or {}
    need = ["name", "experiment_name", "split_dir", "index_dir", "target_part", "frames_root", "sampling_template",
            "frames", "cameras", "encoder", "classifier", "out_dir"]
    miss = [k for k in need if k not in d]
    if miss:
        raise ProbeConfigError(f"{p} 缺少：{miss}")
    d.setdefault("action", "insert")
    for k, tag in (("start_buffer", "sb"), ("end_buffer", "eb")):  # 可選：改影格視窗（例如段結束後 1 秒），要和取樣資料夾名稱一致
        if d.get(k) is not None and f"_{tag}{float(d[k]):g}_" not in d["sampling_template"]:
            raise ProbeConfigError(f"{p}：{k}={d[k]} 和 sampling_template {d['sampling_template']} 不一致")
    return d


# ---------------------------------------------------------------- 資料池


def build_pools(cfg: dict[str, Any], allow_test: bool = False) -> dict[str, Any]:
    """train＝其他零件、非 BNC 測試錄影；dev＝BNC 開發集；test＝BNC 測試集（要 allow_test）。"""
    from .split import load_split, read_split_csv

    rows = read_split_csv(Path(cfg["split_dir"]) / "split.csv")
    test_recs = {r["recording_id"] for r in rows if r["split"] == "test"}  # 只用錄影 ID
    seg_csv = Path(cfg["index_dir"]) / "segments.csv"
    if not seg_csv.exists():
        raise FileNotFoundError(f"找不到 {seg_csv}")
    train = []
    with seg_csv.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("action") != cfg["action"] or r.get("validation_status") not in ("ok", "warn"):
                continue
            if r.get("part") == cfg["target_part"] or r["recording_id"] in test_recs:
                continue
            res = {"True": "success", "False": "failure"}.get(r.get("success"))
            if res is None:
                continue
            train.append({"segment_id": f"{r['recording_id']}__seg{int(r['segment_id'])}", "recording_id": r["recording_id"],
                          "part": r["part"], "dataset_result": res})
    leaked = [t for t in train if t["recording_id"] in test_recs]
    assert not leaked, "訓練池含測試錄影"  # 防呆
    dev = [dict(r, part=cfg["target_part"]) for r in load_split(cfg["split_dir"], include=("dev",))]
    test = ([dict(r, part=cfg["target_part"]) for r in load_split(cfg["split_dir"], include=("test",), allow_test=True)]
            if allow_test else [])
    return {"train": sorted(train, key=lambda x: x["segment_id"]), "dev": dev, "test": test,
            "n_test_recordings_excluded": len(test_recs)}


def pool_summary(train: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter((t["part"], t["dataset_result"]) for t in train)
    parts = sorted({t["part"] for t in train})
    return {"n": len(train), "n_failure": sum(1 for t in train if t["dataset_result"] == "failure"),
            "n_recordings": len({t["recording_id"] for t in train}),
            "per_part": {p: {"success": c[(p, "success")], "failure": c[(p, "failure")]} for p in parts}}


def frames_dir(cfg: dict[str, Any], part: str) -> Path:
    return Path(cfg["frames_root"]) / f"{_slug(cfg['action'])}_{_slug(part)}"


# ---------------------------------------------------------------- 影格擷取（沿用第 3 包）


def extract_frames(cfg: dict[str, Any], segs: list[dict[str, Any]], *, mapping: Any, slim_dir: str | Path,
                   workers: int = 4) -> dict[str, Any]:
    from .extract import run_extraction

    by_part: dict[str, list[str]] = {}
    for s in segs:
        by_part.setdefault(s["part"], []).append(s["segment_id"])
    out = {"n_done": 0, "n_skipped": 0, "n_missing_frames": 0, "errors": []}
    for part, ids in sorted(by_part.items()):
        log.info("擷取影格：%s %d 段", part, len(ids))
        res = run_extraction(mapping=mapping, index_dir=cfg["index_dir"], slim_dir=slim_dir, out_dir=cfg["frames_root"],
                             part=part, action=cfg["action"], which="probe", n_frames_list=tuple(cfg["frames"]),
                             start_buffer=cfg.get("start_buffer"), end_buffer=cfg.get("end_buffer"),
                             cameras=list(cfg["cameras"]), workers=workers, segment_ids=ids)
        for k in ("n_done", "n_skipped", "n_missing_frames"):
            out[k] += res[k]
        out["errors"] += res["errors"]
    return out


# ---------------------------------------------------------------- 編碼器


class ImageEncoder:
    key = "base"
    dim = 0

    def load(self) -> None:
        pass

    def encode(self, paths: list[str]) -> np.ndarray:  # pragma: no cover - 介面
        raise NotImplementedError

    def close(self) -> None:
        pass


class FakeEncoder(ImageEncoder):
    """測試用：用影像的平均色與亮度分布當特徵，不需要 GPU。"""

    key = "fake"
    dim = 8

    def encode(self, paths: list[str]) -> np.ndarray:
        import cv2

        out = []
        for p in paths:
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            m, s = img.mean(axis=(0, 1)), img.std(axis=(0, 1))
            out.append(np.concatenate([m, s, [img.mean(), img.std()]]))
        return np.asarray(out, dtype=np.float32)


POOLS = ("cls_mean", "pooler_mean")


class TransformersImageEncoder(ImageEncoder):
    """Hugging Face 影像模型（預設 facebook/dinov2-base）：每張圖取一個向量。
    pool=cls_mean（預設，DINOv2）：CLS token 與 patch 平均串接；pool=pooler_mean（SigLIP）：pooler 輸出與所有 token 平均串接。
    submodule：只用模型的某個分支（例如 SigLIP 的 vision_model，不載入文字分支的計算）。
    特徵快取放在 _features/<key>/；同一個模型換了輸入大小或取法時，要設 cache_tag 分開存（程式會檢查）。"""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        if cfg.get("pool", "cls_mean") not in POOLS:
            raise ProbeConfigError(f"encoder.pool 只能是 {POOLS}")
        self.key = _slug(cfg["hf_id"]).lower() + (f"_{_slug(str(cfg['cache_tag']))}" if cfg.get("cache_tag") else "")

    def load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if not torch.cuda.is_available() and self.cfg.get("device", "cuda") == "cuda":
            raise RuntimeError("找不到可用的 CUDA GPU（請先執行 setup_vlm.bat）")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(torch, self.cfg.get("dtype", "float16")) if self.device == "cuda" else torch.float32
        rev = self.cfg.get("revision") or None
        self.processor = AutoImageProcessor.from_pretrained(self.cfg["hf_id"], revision=rev)
        self.model = AutoModel.from_pretrained(self.cfg["hf_id"], revision=rev, dtype=self.dtype).to(self.device).eval()
        sub = self.cfg.get("submodule")
        self.net = getattr(self.model, sub) if sub else self.model
        cfg_h = getattr(self.model.config, "vision_config", None) if sub else None
        self.dim = 2 * int(getattr(cfg_h or self.model.config, "hidden_size"))

    def encode(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        torch = self.torch
        feats = []
        bs = int(self.cfg.get("batch_size", 32))
        for i in range(0, len(paths), bs):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + bs]]
            inputs = self.processor(images=imgs, return_tensors="pt", **(self.cfg.get("processor_kwargs") or {}))
            inputs = inputs.to(self.device, dtype=self.dtype)
            with torch.inference_mode():
                o = self.net(**inputs)
            h = o.last_hidden_state.float()
            if self.cfg.get("pool", "cls_mean") == "pooler_mean":
                v = torch.cat([o.pooler_output.float(), h.mean(dim=1)], dim=1)
            else:
                v = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
            feats.append(v.cpu().numpy())
        return np.concatenate(feats).astype(np.float32) if feats else np.zeros((0, self.dim), np.float32)

    def close(self) -> None:
        for a in ("net", "model", "processor"):
            if hasattr(self, a):
                delattr(self, a)
        try:
            self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def make_encoder(cfg: dict[str, Any]) -> ImageEncoder:
    return FakeEncoder() if cfg.get("type") == "fake" else TransformersImageEncoder(cfg)


ENCODER_ID_KEYS = ("hf_id", "revision", "processor_kwargs", "submodule", "pool", "type")


def check_feature_cache(cache_root: Path, enc: ImageEncoder, enc_cfg: dict[str, Any]) -> None:
    """同一個快取資料夾只能放同一種特徵：第一次寫下編碼器設定，之後不同就報錯（避免悄悄混用舊特徵）。"""
    ident = {k: enc_cfg.get(k) for k in ENCODER_ID_KEYS if enc_cfg.get(k) is not None}
    p = cache_root / enc.key / "_encoder.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old != ident:
            raise ProbeConfigError(f"特徵快取 {p.parent} 是用不同的編碼器設定做的（{old}）；請在 encoder 加 cache_tag 分開存")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ident, ensure_ascii=False, sort_keys=True), encoding="utf-8")


TEMPORAL = ("concat", "last_mean", "last_diff", "mean")


def model_key(cfg: dict[str, Any], temporal: str = "concat") -> str:
    """結果資料夾名稱：預設（依時間串接）沿用設定的 name；其他時間彙整方式加後綴。"""
    return cfg["name"] if temporal == "concat" else f"{cfg['name']}_{temporal}"


def temporal_modes(cfg: dict[str, Any]) -> list[str]:
    t = cfg.get("temporal") or ["concat"]
    t = [t] if isinstance(t, str) else list(t)
    bad = [x for x in t if x not in TEMPORAL]
    if bad:
        raise ProbeConfigError(f"temporal 只能是 {TEMPORAL}（目前 {bad}）")
    return t


def segment_features(cfg: dict[str, Any], enc: ImageEncoder, seg: dict[str, Any], n: int,
                     cache_root: Path, timings: dict[str, float] | None = None,
                     use_cache: bool = True) -> np.ndarray | None:
    """回傳 [n, 相機數, dim] 的特徵（缺的影格為 NaN）；有快取就直接讀。沒有影格資料夾回傳 None。
    timings：給了就記下當場編碼的秒數（第 8 包：報告執行時間；從快取讀的不記）。"""
    sampling = cfg["sampling_template"].format(n=n)
    cache = cache_root / enc.key / sampling / f"{seg['segment_id']}.npy"
    if use_cache and cache.exists():
        return np.load(cache)
    si = vlm.load_segment_frames(frames_dir(cfg, seg["part"]), sampling, seg["segment_id"], list(cfg["cameras"]))
    if si is None or not si.frames:
        return None
    t0 = time.perf_counter()
    vecs = enc.encode([f.path for f in si.frames])
    if timings is not None:
        timings[seg["segment_id"]] = time.perf_counter() - t0
    out = np.full((n, len(cfg["cameras"]), vecs.shape[1]), np.nan, dtype=np.float32)
    cam_i = {c: i for i, c in enumerate(cfg["cameras"])}
    for f, v in zip(si.frames, vecs):
        if f.k < n:
            out[f.k, cam_i[f.camera]] = v
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def _nanmean0(a: np.ndarray) -> np.ndarray:
    """沿時間平均，忽略缺的影格；整欄都缺就是 NaN（之後補 0）。"""
    with np.errstate(invalid="ignore"):
        cnt = np.sum(~np.isnan(a), axis=0)
        tot = np.nansum(a, axis=0)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def to_vector(feat: np.ndarray, temporal: str = "concat") -> np.ndarray:
    """feat：[時間, 相機, dim]。缺的影格補 0（標準化後等於「平均值」）。
    concat：依（時間、相機）串接（預設）；mean：時間平均；last_mean：最後一張＋時間平均；last_diff：最後一張＋（最後−第一張）。"""
    if temporal == "concat":
        v = feat.reshape(-1)
    elif temporal == "mean":
        v = _nanmean0(feat).reshape(-1)
    elif temporal == "last_mean":
        v = np.concatenate([feat[-1].reshape(-1), _nanmean0(feat).reshape(-1)])
    elif temporal == "last_diff":
        v = np.concatenate([feat[-1].reshape(-1), (feat[-1] - feat[0]).reshape(-1)])
    else:
        raise ProbeConfigError(f"未知的 temporal：{temporal}")
    return np.nan_to_num(v, nan=0.0)


def build_matrix(cfg: dict[str, Any], enc: ImageEncoder, segs: list[dict[str, Any]], n: int,
                 cache_root: Path, temporal: str = "concat",
                 timings: dict[str, float] | None = None) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    X, keep, missing = [], [], []
    for s in segs:
        f = segment_features(cfg, enc, s, n, cache_root, timings)
        if f is None:
            missing.append(s["segment_id"])
            continue
        X.append(to_vector(f, temporal))
        keep.append(s)
    return (np.vstack(X) if X else np.zeros((0, 0), np.float32)), keep, missing


# ---------------------------------------------------------------- 分類器


def _pipeline(c: float, n_comp: int, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=seed),
                         LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def train_classifier(X: np.ndarray, y: np.ndarray, groups: list[str], ccfg: dict[str, Any]) -> dict[str, Any]:
    """依錄影分組做交叉驗證挑 C 與 PCA 維度（分數＝平衡準確率），再用全部訓練池重訓。"""
    from sklearn.metrics import balanced_accuracy_score, recall_score
    from sklearn.model_selection import GroupKFold

    seed = int(ccfg.get("seed", 0))
    n_groups = len(set(groups))
    k = max(2, min(int(ccfg.get("cv_folds", 5)), n_groups))
    folds = list(GroupKFold(n_splits=k).split(X, y, groups))
    grid = []
    for n_comp in ccfg.get("pca_grid", [32, 64]):
        for c in ccfg.get("C_grid", [0.01, 0.1, 1.0]):
            pred = np.zeros_like(y)
            ok = True
            for tr, te in folds:
                nc = min(int(n_comp), len(tr) - 1, X.shape[1])
                if len(set(y[tr])) < 2:
                    ok = False
                    break
                m = _pipeline(float(c), nc, seed).fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            if not ok:
                continue
            grid.append({"C": float(c), "pca": int(n_comp), "cv_balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
                         "cv_failure_recall": round(float(recall_score(y, pred, pos_label=1, zero_division=0)), 4)})
    if not grid:
        raise ProbeConfigError("交叉驗證失敗：訓練池的成功／失敗類別不足")
    best = max(grid, key=lambda g: (g["cv_balanced_accuracy"], g["cv_failure_recall"], -g["C"]))
    model = _pipeline(best["C"], min(best["pca"], len(y) - 1, X.shape[1]), seed).fit(X, y)
    return {"model": model, "best": best, "grid": grid, "n_folds": k}


# ---------------------------------------------------------------- 執行


def _previous_timings(rd: Path, encoder_key: str, sampling_key: str) -> dict[str, tuple[float, str]]:
    """上一次輸出裡當場編碼的每段時間（第 8 包修正：重跑時特徵都在快取，原本會把計時清掉）。
    只在上次的編碼器與影格取樣相同時沿用；回傳 {segment_id: (encode_s, 原本量測的時間)}。"""
    rc_p, raw_p = rd / "run_config.json", rd / "raw.jsonl"
    if not (rc_p.exists() and raw_p.exists()):
        return {}
    try:
        rc = json.loads(rc_p.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    if rc.get("encoder_key") != encoder_key or rc.get("sampling_key") != sampling_key:
        return {}
    out: dict[str, tuple[float, str]] = {}
    for line in raw_p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("segment_id") and r.get("encode_s") is not None:
            out[r["segment_id"]] = (float(r["encode_s"]), r.get("timing_from") or r.get("created_at") or "")
    return out


def run_probe(cfg: dict[str, Any], *, frames: list[int] | None = None, allow_test: bool = False,
              encoder: ImageEncoder | None = None, write_summary: bool = True,
              temporal: list[str] | None = None) -> dict[str, Any]:
    """特徵 → 訓練 → 預測（dev；allow_test 時改測試集）→ 寫成和 VLM 相同格式的輸出。
    temporal：時間彙整方式（預設讀設定的 temporal，沒有就只用 concat）；1 幀時只有 concat 有意義。"""
    pools = build_pools(cfg, allow_test=allow_test)
    eval_set = pools["test"] if allow_test else pools["dev"]
    exp_name = cfg["experiment_name"] if not allow_test else cfg.get("final_experiment_name", "final_test")
    out_root = Path(cfg["out_dir"]) / exp_name
    cache_root = Path(cfg["out_dir"]) / "_features"
    modes = temporal or temporal_modes(cfg)
    enc = encoder or make_encoder(cfg["encoder"])
    check_feature_cache(cache_root, enc, cfg["encoder"])
    t0 = time.perf_counter()
    enc.load()
    load_s = time.perf_counter() - t0
    log.info("編碼器 %s 載入 %.0f 秒", enc.key, load_s)
    summary_pool = pool_summary(pools["train"])
    log.info("訓練池：%d 段（失敗 %d），%d 支錄影，零件 %d 種；已排除 BNC 測試集的 %d 支錄影",
             summary_pool["n"], summary_pool["n_failure"], summary_pool["n_recordings"], len(summary_pool["per_part"]),
             pools["n_test_recordings_excluded"])
    results = []
    try:
        for n in (frames or list(cfg["frames"])):
            for tm in modes:
                if n == 1 and tm != "concat":
                    continue  # 只有一張時各種彙整都一樣
                mk = model_key(cfg, tm)
                Xtr, keep_tr, miss_tr = build_matrix(cfg, enc, pools["train"], n, cache_root, tm)
                enc_t: dict[str, float] = {}
                Xev, keep_ev, miss_ev = build_matrix(cfg, enc, eval_set, n, cache_root, tm, timings=enc_t)
                if len(keep_tr) < 10:
                    raise ProbeConfigError(f"{n} 幀：訓練池只有 {len(keep_tr)} 段有影格（先執行 probe extract）")
                ytr = np.array([1 if s["dataset_result"] == "failure" else 0 for s in keep_tr])
                tr = train_classifier(Xtr, ytr, [s["recording_id"] for s in keep_tr], cfg["classifier"])
                log.info("[%d 幀・%s] 交叉驗證：平衡準確率 %.3f、失敗召回 %.3f（C=%g，PCA %d）", n, tm,
                         tr["best"]["cv_balanced_accuracy"], tr["best"]["cv_failure_recall"], tr["best"]["C"], tr["best"]["pca"])
                rd = out_root / mk / f"f{n}"
                rd.mkdir(parents=True, exist_ok=True)
                prev_t = _previous_timings(rd, enc.key, cfg["sampling_template"].format(n=n))
                for p in ("raw.jsonl", "predictions.jsonl"):
                    (rd / p).unlink(missing_ok=True)
                rc = {"tool": TOOL, "model_key": mk, "n_frames": n, "sampling_key": cfg["sampling_template"].format(n=n),
                      "encoder": cfg["encoder"], "encoder_key": enc.key, "classifier": tr["best"], "cv_grid": tr["grid"],
                      "cv_folds": tr["n_folds"], "train_pool": summary_pool, "train_missing_frames": miss_tr,
                      "eval_split": "test" if allow_test else "dev", "eval_missing_frames": miss_ev,
                      "excluded_test_recordings": pools["n_test_recordings_excluded"], "encoder_load_s": round(load_s, 2),
                      "gpu": _gpu_name(), "updated_at": now_iso()}
                if tm != "concat":
                    rc["temporal"] = tm
                rc["config_sha256"] = hashlib.sha256(json.dumps({k: rc[k] for k in ("model_key", "n_frames", "encoder", "classifier")},
                                                                sort_keys=True).encode()).hexdigest()
                (rd / "run_config.json").write_text(json.dumps(rc, ensure_ascii=False, indent=1), encoding="utf-8")
                tp = time.perf_counter()
                probs = tr["model"].predict_proba(Xev)[:, 1] if len(keep_ev) else np.zeros(0)
                clf_s = (time.perf_counter() - tp) / max(1, len(keep_ev))  # 分類器每段的時間（很小）
                done = set()
                for s, pf in zip(keep_ev, probs):
                    res = "failure" if pf >= 0.5 else "success"
                    base = {"segment_id": s["segment_id"], "model": mk, "n_frames": n, "prompt_version": None,
                            "sampling_key": rc["sampling_key"], "created_at": now_iso()}
                    enc_s = enc_t.get(s["segment_id"])  # 從快取讀的特徵沒有時間
                    timing_from = None
                    if enc_s is None and s["segment_id"] in prev_t:  # 重跑時沿用上次當場編碼的時間（同編碼器、同影格）
                        enc_s, timing_from = prev_t[s["segment_id"]]
                    rec = {**base, "prob_failure": round(float(pf), 4),
                           "latency_s": round(enc_s + clf_s, 4) if enc_s is not None else None,
                           "encode_s": round(enc_s, 4) if enc_s is not None else None,
                           "raw_text": json.dumps({"prob_failure": round(float(pf), 4)})}
                    if timing_from:
                        rec["timing_from"] = timing_from
                    vlm._append_jsonl(rd / "raw.jsonl", rec)
                    vlm._append_jsonl(rd / "predictions.jsonl", {
                        **base, "status": "ok", "errors": [], "warnings": [], "attempt": "first", "result": res,
                        "failure_reason": None, "evidence_start_sec": None, "evidence_end_sec": None, "best_camera": None,
                        "visible_evidence": None, "confidence": round(float(max(pf, 1 - pf)), 4),
                        "dataset_result": s.get("dataset_result")})
                    done.add(s["segment_id"])
                for s in eval_set:
                    if s["segment_id"] not in done:
                        vlm._append_jsonl(rd / "predictions.jsonl", {
                            "segment_id": s["segment_id"], "model": mk, "n_frames": n, "status": "error",
                            "errors": ["frames_missing"], "warnings": [], "result": None, "dataset_result": s.get("dataset_result"),
                            "created_at": now_iso()})
                st = vlm.summarize_run(rd, eval_set)
                results.append({"model": mk, "n_frames": n, "temporal": tm, "cv": tr["best"], "run_dir": str(rd), **st})
                log.info("[%d 幀・%s] %s：快速準確率 %s、失敗召回 %s", n, tm, "測試集" if allow_test else "開發集",
                         st["quick_accuracy"], st["quick_failure_recall"])
    finally:
        enc.close()
    report = {"model": cfg["name"], "eval_split": "test" if allow_test else "dev", "train_pool": summary_pool,
              "results": results, "generated_at": now_iso()}
    (out_root / cfg["name"]).mkdir(parents=True, exist_ok=True)
    (out_root / cfg["name"] / "probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if write_summary:
        exp = {"out_dir": cfg["out_dir"], "name": exp_name, "split_dir": cfg["split_dir"],
               "include": ["test"] if allow_test else ["dev"]}
        report["summary"] = vlm.write_summary(exp, vlm.collect_summary_rows(exp, eval_set))
    return report


def _gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:  # noqa: BLE001
        return None


def time_probe(cfg: dict[str, Any], *, n_segments: int = 20, frames: int | None = None,
               encoder: ImageEncoder | None = None) -> dict[str, Any]:
    """第 8 包：量做法 A 每段要多久（不讀快取、重新編碼；只用開發集的段，不碰測試集）。
    時間＝影像編碼器處理這段的影格＋邏輯迴歸（用訓練池的一小部分特徵訓練，只為了量預測時間）。"""
    n = int(frames or cfg["frames"][0])
    pools = build_pools(cfg)
    segs = (pools["dev"] or pools["train"])[:n_segments]
    cache_root = Path(cfg["out_dir"]) / "_features"
    enc = encoder or make_encoder(cfg["encoder"])
    t0 = time.perf_counter()
    enc.load()
    load_s = time.perf_counter() - t0
    times: dict[str, float] = {}
    try:
        feats = []
        for s in segs:
            f = segment_features(cfg, enc, s, n, cache_root, times, use_cache=False)
            if f is not None:
                feats.append(to_vector(f))
        clf_s = None
        if len(feats) >= 4:
            from sklearn.linear_model import LogisticRegression

            X = np.vstack(feats)
            y = np.arange(len(X)) % 2
            m = LogisticRegression(max_iter=200).fit(X, y)
            tp = time.perf_counter()
            m.predict_proba(X)
            clf_s = (time.perf_counter() - tp) / len(X)
    finally:
        enc.close()
    xs = sorted(times.values())
    rep = {"model": cfg["name"], "encoder": cfg["encoder"].get("hf_id"), "n_frames": n, "n_segments": len(xs),
           "encoder_load_s": round(load_s, 2), "encode_mean_s": round(sum(xs) / len(xs), 4) if xs else None,
           "encode_median_s": round(xs[len(xs) // 2], 4) if xs else None, "classifier_s": clf_s,
           "gpu": _gpu_name(), "generated_at": now_iso()}
    out = Path(cfg["out_dir"]) / "timing"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"probe_{cfg['name']}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("做法 A 執行時間：%d 段，每段編碼 %.0f ms（%s），編碼器載入 %.0f 秒", len(xs), 1000 * (rep["encode_mean_s"] or 0),
             rep["gpu"], load_s)
    return rep


# ---------------------------------------------------------------- 第 7 包：做法 A 改良（掃描＋依交叉驗證挑選）


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ProbeConfigError(f"找不到 {p}")
    d = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("probe_sweep") or {}
    miss = [k for k in ("name", "base", "experiment_name", "end_buffers", "frames", "encoders", "selected_config") if k not in d]
    if miss:
        raise ProbeConfigError(f"{p} 缺少 probe_sweep.{miss}")
    if not d["encoders"]:
        raise ProbeConfigError(f"{p}：encoders 是空的")
    d.setdefault("temporal", ["concat"])
    d.setdefault("final_experiment_name", "final_test_train")
    d.setdefault("selection_out", "experiments/probe_selection")
    d["_path"] = str(p)
    return d


def sweep_candidates(sw: dict[str, Any]) -> list[dict[str, Any]]:
    """每個（編碼器 × 最後一張的時間）一份完整的做法 A 設定；資料池與分類器沿用 base。"""
    import re

    base = load_probe_config(sw["base"])
    out = []
    for ek, ecfg in sw["encoders"].items():
        for eb in sw["end_buffers"]:
            tpl = re.sub(r"_eb[0-9.]+_", f"_eb{float(eb):g}_", base["sampling_template"])
            if f"_eb{float(eb):g}_" not in tpl:
                raise ProbeConfigError(f"base 的 sampling_template 沒有 _eb…_：{base['sampling_template']}")
            c = dict(base, name=f"{sw['name']}_{ek}_eb{float(eb):g}".replace(".", "p"), experiment_name=sw["experiment_name"],
                     final_experiment_name=sw["final_experiment_name"], sampling_template=tpl, end_buffer=float(eb),
                     frames=list(sw["frames"]), temporal=list(sw["temporal"]), encoder=dict(ecfg))
            c["_encoder_key"], c["_end_buffer"] = ek, float(eb)
            out.append(c)
    return out


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """事先決定的規則：交叉驗證平衡準確率最高；同分看失敗召回，再選幀數少的，再依設定檔裡的順序。BNC 開發集不參與。"""
    ok = [r for r in rows if r.get("cv_balanced_accuracy") is not None]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (-r["cv_balanced_accuracy"], -r["cv_failure_recall"], r["n_frames"], r["order"]))[0]


def write_selected_config(sw: dict[str, Any], cand: dict[str, Any], best: dict[str, Any]) -> Path:
    cfg = {k: v for k, v in cand.items() if not k.startswith("_")}
    cfg.update(frames=[best["n_frames"]], temporal=[best["temporal"]])
    head = (f"# 由 python -m scripts.probe sweep --config {sw['_path']} 自動產生（{now_iso()}），請不要手動修改。\n"
            f"# 挑選規則：訓練池（其他零件）依錄影分組交叉驗證的平衡準確率最高；BNC 開發集只看不挑。\n"
            f"# 選到：{best['encoder']}，最後一張 +{best['end_buffer']:g} 秒，{best['n_frames']} 幀，時間彙整 {best['temporal']}，"
            f"交叉驗證平衡準確率 {best['cv_balanced_accuracy']:.3f}。\n")
    out = Path(sw["selected_config"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + yaml.safe_dump({"probe": cfg}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def run_sweep(sw: dict[str, Any], *, encoder_factory: Any = None) -> dict[str, Any]:
    """逐一跑每個候選（只用開發集，不讀測試集），寫挑選表與選到的設定檔。一個編碼器載入失敗不影響其他候選。"""
    rows, errors = [], []
    order = 0
    for cand in sweep_candidates(sw):
        enc = encoder_factory(cand["encoder"]) if encoder_factory else None
        try:
            rep = run_probe(cand, encoder=enc, write_summary=False)
        except Exception as e:  # noqa: BLE001 - 例如模型還沒下載（離線模式）
            log.error("[%s] 失敗：%s: %s", cand["name"], type(e).__name__, e)
            errors.append({"candidate": cand["name"], "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        for r in rep["results"]:
            order += 1
            rows.append({"order": order, "candidate": cand["name"], "model": r["model"], "encoder": cand["_encoder_key"],
                         "end_buffer": cand["_end_buffer"], "n_frames": r["n_frames"], "temporal": r["temporal"],
                         "cv_balanced_accuracy": r["cv"]["cv_balanced_accuracy"], "cv_failure_recall": r["cv"]["cv_failure_recall"],
                         "C": r["cv"]["C"], "pca": r["cv"]["pca"], "dev_correct": r.get("n_correct"), "dev_n": r.get("n_segments"),
                         "dev_caught": r.get("n_caught"), "dev_failures": r.get("n_failures"),
                         "dev_false_alarms": r.get("n_false_alarms"), "dev_successes": r.get("n_successes"),
                         "dev_balanced": r.get("quick_balanced_accuracy"), "run_dir": r["run_dir"]})
    best = select_best(rows)
    sel_path = None
    if best:
        cand = next(c for c in sweep_candidates(sw) if c["name"] == best["candidate"])
        sel_path = write_selected_config(sw, cand, best)
    out = Path(sw["selection_out"])
    out.mkdir(parents=True, exist_ok=True)
    rep = {"sweep": sw["_path"], "rows": rows, "errors": errors, "best": best, "selected_config": str(sel_path) if sel_path else None,
           "rule": "cv_balanced_accuracy → cv_failure_recall → fewer frames → config order", "generated_at": now_iso()}
    (out / "selection.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    def pct(v: Any) -> str:
        return "–" if v is None else f"{100 * v:.0f}%"

    def frac(a: Any, b: Any) -> str:
        return "–" if a is None or not b else f"{a}/{b}"

    lines = ["# 做法 A 改良：候選比較（依交叉驗證排序）", "", f"產生時間：{now_iso()}　設定：`{sw['_path']}`", "",
             "挑選規則（事先決定）：訓練池（其他零件，排除 BNC 測試錄影）依錄影分組交叉驗證的**平衡準確率**最高者；"
             "同分看失敗召回，再選幀數少的。BNC 開發集（20 段）只列出來看，不參與挑選。", "",
             "| | 編碼器 | 最後一張 | 幀數 | 時間彙整 | 交叉驗證平衡準確率 | 交叉驗證失敗召回 | 開發集答對 | 開發集抓到失敗 | 開發集成功被誤判 | 開發集兩類平均 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (-r["cv_balanced_accuracy"], -r["cv_failure_recall"], r["n_frames"], r["order"])):
        mark = "★" if best and r is best else ""
        lines.append(f"| {mark} | {r['encoder']} | +{r['end_buffer']:g} s | {r['n_frames']} | {r['temporal']} | "
                     f"{pct(r['cv_balanced_accuracy'])} | {pct(r['cv_failure_recall'])} | {frac(r['dev_correct'], r['dev_n'])} | "
                     f"{frac(r['dev_caught'], r['dev_failures'])} | {frac(r['dev_false_alarms'], r['dev_successes'])} | {pct(r['dev_balanced'])} |")
    if errors:
        lines += ["", "失敗的候選："] + [f"- {e['candidate']}：{e['error']}" for e in errors]
    if sel_path:
        lines += ["", f"選到的設定已寫入 `{sel_path}`（給測試集凍結設定引用）。"]
    (out / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep["md"] = str(out / "selection.md")
    return rep


def describe_pool(cfg: dict[str, Any]) -> str:
    s = pool_summary(build_pools(cfg)["train"])
    lines = [f"訓練池 {s['n']} 段（失敗 {s['n_failure']}），{s['n_recordings']} 支錄影："]
    for p, c in s["per_part"].items():
        lines.append(f"  {p:14} 成功 {c['success']:3d}  失敗 {c['failure']:3d}")
    return "\n".join(lines)


__all__ = ["load_probe_config", "build_pools", "pool_summary", "extract_frames", "run_probe", "describe_pool",
           "FakeEncoder", "TransformersImageEncoder", "train_classifier", "to_vector", "model_key", "load_sweep_config",
           "sweep_candidates", "select_best", "run_sweep"]
