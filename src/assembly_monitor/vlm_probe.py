"""再訓練：VLM 特徵＋小分類器（只看開發集）。

LoRA（第 7 包）改模型權重後崩成常數輸出。這裡模型完全不動：讓 Qwen3-VL-8B 照 ft-generic 提示詞看完畫面，
在它「正要寫出 success／failure」的位置讀出
  - zs：log P(failure) − log P(success)（模型自己的判斷分數）
  - L50／L75／L100：語言模型第 50%／75%／最後一層在該位置的隱藏狀態
再用其他零件的插入片段（和做法 A、LoRA 同一個訓練池）訓練邏輯迴歸，依錄影分組交叉驗證挑超參數與特徵，
在 BNC 開發集看結果。程式只載入訓練池與開發集，不讀測試集。

步驟：extract（GPU，約 25 分鐘，可續跑）→ fit（CPU，幾秒）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import finetune, probe, vlm
from .util import now_iso

log = logging.getLogger(__name__)
TOOL = "vlm_probe/1"
FEATURE_ORDER = ("zs", "L50", "L75", "L100")


class VLMProbeConfigError(ValueError):
    pass


# ---------------------------------------------------------------- 設定


def load_vlm_probe_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise VLMProbeConfigError(f"找不到 {p}")
    d = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("vlm_probe") or {}
    need = ["name", "base_model", "models", "prompt_experiment", "split_dir", "index_dir", "target_part", "frames_root",
            "n_frames", "feature_dir", "layers", "features", "experiment_name", "classifier"]
    miss = [k for k in need if k not in d]
    if miss:
        raise VLMProbeConfigError(f"{p} 缺少 vlm_probe.{miss}")
    d.setdefault("action", "insert")
    d.setdefault("cache_dir", "experiments/_ft_inputs")
    d.setdefault("selection_out", "experiments/vlm_probe_selection")
    d.setdefault("out_dir", "experiments")
    names = {}
    if d.get("part_names_from"):
        names = dict(finetune.load_finetune_config(d["part_names_from"]).get("part_names") or {})
    d["part_names"] = {**names, **(d.get("part_names") or {})}
    layers = [float(x) for x in d["layers"]]
    if not layers or any(not 0 < x <= 1 for x in layers):
        raise VLMProbeConfigError(f"{p}：layers 要是 (0, 1] 之間的比例")
    d["layers"] = layers
    known = {"zs"} | {layer_name(x) for x in layers}
    bad = [f for f in d["features"] if f not in known]
    if bad:
        raise VLMProbeConfigError(f"{p}：features {bad} 不在 {sorted(known)} 裡")
    d["_path"] = str(p)
    return d


def layer_name(frac: float) -> str:
    return f"L{round(frac * 100)}"


def layer_index(frac: float, n_hidden: int) -> int:
    """hidden_states 有 n_hidden 個（第 0 個是嵌入層），比例 1.0 → 最後一個。"""
    return max(1, min(n_hidden - 1, round(frac * (n_hidden - 1))))


def feature_file(cfg: dict[str, Any], segment_id: str) -> Path:
    return Path(cfg["feature_dir"]) / f"f{cfg['n_frames']}" / f"{segment_id}.npz"


def extraction_signature(cfg: dict[str, Any], exp: dict[str, Any], model_cfg: dict[str, Any], prompt_sha: str) -> str:
    payload = {"model": model_cfg, "prompt": prompt_sha, "sampling": exp["sampling_template"], "n_frames": cfg["n_frames"],
               "layers": cfg["layers"], "part_names": cfg["part_names"], "cameras": exp["cameras"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_signature(cfg: dict[str, Any], sig: str) -> None:
    """同一個特徵資料夾只放同一種設定抽出的特徵；設定不同就報錯（請換 feature_dir）。"""
    p = Path(cfg["feature_dir"]) / f"f{cfg['n_frames']}" / "_signature.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8")).get("signature")
        if old != sig:
            raise VLMProbeConfigError(f"{p.parent} 裡的特徵是用不同設定抽的；請換 feature_dir")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"signature": sig, "created_at": now_iso()}), encoding="utf-8")


def samples(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """訓練池（其他零件）與 BNC 開發集；不讀測試集。"""
    exp = vlm.load_experiment(cfg["prompt_experiment"])
    return exp, finetune.build_samples(cfg, exp)


# ---------------------------------------------------------------- 抽特徵（GPU，只在使用者的電腦上執行）


def extract(cfg: dict[str, Any], *, limit: int | None = None) -> dict[str, Any]:  # pragma: no cover - 需要 GPU
    import torch

    exp, data = samples(cfg)
    model_cfgs = vlm.load_model_configs(cfg["models"])
    mcfg = model_cfgs[cfg["base_model"]]
    if mcfg.get("adapter_path"):
        raise VLMProbeConfigError("base_model 要用沒有 adapter 的原始模型")
    base = vlm.PromptBuilder.from_experiment(exp)
    check_signature(cfg, extraction_signature(cfg, exp, mcfg, base.sha256))
    allsamp = data["train"] + data["dev"]
    todo = [s for s in allsamp if not feature_file(cfg, s["segment_id"]).exists()]
    if limit is not None:
        todo = todo[:limit]
    log.info("特徵：訓練池 %d 段、開發集 %d 段；已抽 %d 段、這次要抽 %d 段", len(data["train"]), len(data["dev"]),
             len(allsamp) - len([s for s in allsamp if not feature_file(cfg, s["segment_id"]).exists()]), len(todo))
    if not todo:
        return {"n_new": 0, "n_total": len(allsamp)}
    ad = vlm.TransformersAdapter(cfg["base_model"], mcfg)
    ad.load()
    model, processor, tkw = ad.model, ad.processor, getattr(ad, "template_kwargs", {})
    tok = processor.tokenizer
    targets = {r: tok(finetune.answer_suffix(processor, finetune.answer_text(r), tkw), add_special_tokens=False)["input_ids"]
               for r in finetune.RESULTS}
    d = finetune.first_divergence(targets["success"], targets["failure"])
    prefix, s_tok, f_tok = targets["success"][:d], targets["success"][d], targets["failure"][d]
    builders = {p: vlm.PromptBuilder(base.template, base.version, finetune.part_task(exp["task"], p, cfg["part_names"]))
                for p in {s["part"] for s in todo}}
    max_side = mcfg.get("image_max_side")
    t0 = time.perf_counter()
    errors = []
    try:
        for i, s in enumerate(todo, 1):
            ts = time.perf_counter()
            try:
                paths = finetune.cached_images(cfg, s, max_side)
                prompt = builders[s["part"]].build(s["seg"], paths)
                pin = processor.apply_chat_template([{"role": "user", "content": prompt.content}], tokenize=True,
                                                    add_generation_prompt=True, return_dict=True, return_tensors="pt", **tkw)
                batch = finetune.build_batch(pin, prefix, model.device, ad.dtype)
                with torch.inference_mode():
                    out = model(**batch, logits_to_keep=1, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                arrays = {layer_name(f): hs[layer_index(f, len(hs))][0, -1].float().cpu().numpy().astype(np.float32)
                          for f in cfg["layers"]}
                arrays["zs"] = np.array([float(lp[f_tok] - lp[s_tok])], dtype=np.float32)
                meta = {"segment_id": s["segment_id"], "part": s["part"], "n_hidden": len(hs),
                        "layer_index": {layer_name(f): layer_index(f, len(hs)) for f in cfg["layers"]},
                        "input_tokens": int(batch["input_ids"].shape[1]), "created_at": now_iso()}
                fp = feature_file(cfg, s["segment_id"])
                fp.parent.mkdir(parents=True, exist_ok=True)
                tmp = fp.with_suffix(".tmp.npz")
                np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
                tmp.replace(fp)
                del out, hs, batch
            except torch.cuda.OutOfMemoryError:
                errors.append({"segment_id": s["segment_id"], "error": "oom"})
                torch.cuda.empty_cache()
            if i % 20 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                log.info("  %d/%d（%.1f 秒/段，預計還要 %.0f 分鐘）", i, len(todo), (time.perf_counter() - ts),
                         el / i * (len(todo) - i) / 60)
    finally:
        ad.close()
    return {"n_new": len(todo) - len(errors), "errors": errors, "minutes": round((time.perf_counter() - t0) / 60, 1)}


# ---------------------------------------------------------------- 訓練與預測（CPU）


def load_matrix(cfg: dict[str, Any], segs: list[dict[str, Any]], feature: str) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    X, keep, missing = [], [], []
    for s in segs:
        fp = feature_file(cfg, s["segment_id"])
        if not fp.exists():
            missing.append(s["segment_id"])
            continue
        with np.load(fp) as z:
            if feature not in z.files:
                raise VLMProbeConfigError(f"{fp} 沒有特徵 {feature}")
            X.append(np.asarray(z[feature], dtype=np.float32).reshape(-1))
        keep.append(s)
    return (np.vstack(X) if X else np.zeros((0, 0), np.float32)), keep, missing


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """事先決定的規則：交叉驗證平衡準確率最高 → 失敗召回 → 維度少 → 設定檔順序。"""
    ok = [r for r in rows if r.get("cv_balanced_accuracy") is not None]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (-r["cv_balanced_accuracy"], -r["cv_failure_recall"], r["dim"], r["order"]))[0]


def _write_run(rd: Path, mk: str, cfg: dict[str, Any], sampling: str, feature: str, tr: dict[str, Any],
               keep: list[dict[str, Any]], probs: np.ndarray, eval_set: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    for p in ("raw.jsonl", "predictions.jsonl"):
        (rd / p).unlink(missing_ok=True)
    rc = {"tool": TOOL, "model_key": mk, "n_frames": cfg["n_frames"], "sampling_key": sampling, "feature": feature,
          "base_model": cfg["base_model"], "classifier": tr["best"], "cv_grid": tr["grid"], "cv_folds": tr["n_folds"],
          "eval_split": "dev", "updated_at": now_iso(), **extra}
    (rd / "run_config.json").write_text(json.dumps(rc, ensure_ascii=False, indent=1), encoding="utf-8")
    done = set()
    for s, pf in zip(keep, probs):
        base = {"segment_id": s["segment_id"], "model": mk, "n_frames": cfg["n_frames"], "prompt_version": None,
                "sampling_key": sampling, "created_at": now_iso()}
        vlm._append_jsonl(rd / "raw.jsonl", {**base, "prob_failure": round(float(pf), 4), "latency_s": None,
                                             "raw_text": json.dumps({"prob_failure": round(float(pf), 4)})})
        vlm._append_jsonl(rd / "predictions.jsonl", {
            **base, "status": "ok", "errors": [], "warnings": [], "attempt": "first",
            "result": "failure" if pf >= 0.5 else "success", "failure_reason": None, "evidence_start_sec": None,
            "evidence_end_sec": None, "best_camera": None, "visible_evidence": None,
            "confidence": round(float(max(pf, 1 - pf)), 4), "dataset_result": s.get("result")})
        done.add(s["segment_id"])
    for s in eval_set:
        if s["segment_id"] not in done:
            vlm._append_jsonl(rd / "predictions.jsonl", {
                "segment_id": s["segment_id"], "model": mk, "n_frames": cfg["n_frames"], "status": "error",
                "errors": ["features_missing"], "warnings": [], "result": None, "dataset_result": s.get("result"),
                "created_at": now_iso()})


def fit(cfg: dict[str, Any], *, n_boot: int = 2000, references: list[str] | None = None) -> dict[str, Any]:
    """每個特徵：依錄影分組交叉驗證（訓練池）→ 用全部訓練池重訓 → 預測 BNC 開發集。寫挑選表與開發集報告。"""
    from . import evaluate as ev
    from .split import load_split

    exp, data = samples(cfg)
    sampling = exp["sampling_template"].format(n=cfg["n_frames"])
    train, dev = data["train"], data["dev"]
    out_root = Path(cfg["out_dir"]) / cfg["experiment_name"]
    rows, run_dirs = [], {}
    for order, feat in enumerate(cfg["features"], 1):
        Xtr, ktr, mtr = load_matrix(cfg, train, feat)
        Xdv, kdv, mdv = load_matrix(cfg, dev, feat)
        if len(ktr) < 10:
            raise VLMProbeConfigError(f"{feat}：訓練池只有 {len(ktr)} 段有特徵（先執行 python -m scripts.vlm_probe extract）")
        ytr = np.array([1 if s["result"] == "failure" else 0 for s in ktr])
        tr = probe.train_classifier(Xtr, ytr, [s["recording_id"] for s in ktr], cfg["classifier"])
        probs = tr["model"].predict_proba(Xdv)[:, 1] if len(kdv) else np.zeros(0)
        mk = f"{cfg['name']}_{feat}"
        rd = out_root / mk / f"f{cfg['n_frames']}"
        _write_run(rd, mk, cfg, sampling, feat, tr, kdv, probs, dev,
                   {"train_n": len(ktr), "train_missing": mtr, "dev_missing": mdv, "dim": int(Xtr.shape[1])})
        st = vlm.summarize_run(rd, [dict(s, dataset_result=s["result"]) for s in dev])
        run_dirs[mk] = rd
        rows.append({"order": order, "feature": feat, "model": mk, "dim": int(Xtr.shape[1]),
                     "cv_balanced_accuracy": tr["best"]["cv_balanced_accuracy"], "cv_failure_recall": tr["best"]["cv_failure_recall"],
                     "C": tr["best"]["C"], "pca": tr["best"]["pca"], "train_n": len(ktr),
                     "dev_correct": st.get("n_correct"), "dev_n": st.get("n_segments"), "dev_caught": st.get("n_caught"),
                     "dev_failures": st.get("n_failures"), "dev_false_alarms": st.get("n_false_alarms"),
                     "dev_successes": st.get("n_successes"), "dev_balanced": st.get("quick_balanced_accuracy"), "run_dir": str(rd)})
        log.info("[%s] 交叉驗證平衡準確率 %.3f、失敗召回 %.3f（C=%g，PCA %d）；開發集 %s/%s", feat,
                 tr["best"]["cv_balanced_accuracy"], tr["best"]["cv_failure_recall"], tr["best"]["C"], tr["best"]["pca"],
                 st.get("n_correct"), st.get("n_segments"))
    best = select_best(rows)
    # 開發集報告（含信賴區間）：所有特徵＋參考（同提示詞原始模型的生成結果、第 6 包做法 A），只讀開發集
    segs = load_split(cfg["split_dir"], include=("dev",))
    gold = ev.load_gold(ev.default_gold_paths())
    results = []
    for mk, rd in run_dirs.items():
        desc = ev.describe_run(rd)
        desc["model"] = f"VLM 特徵＋邏輯迴歸（{mk.rsplit('_', 1)[-1]}）"
        desc["prompt"] = "ft-generic"
        results.append({"id": mk + (" ★" if best and best["model"] == mk else ""), "desc": desc,
                        **ev.evaluate_run(rd, segs, gold, n_boot=n_boot)})
    for ref in references or []:
        rd = Path(ref)
        if (rd / "predictions.jsonl").exists():
            results.append({"id": f"參考：{rd.parent.parent.name}/{rd.parent.name}", "desc": ev.describe_run(rd),
                            **ev.evaluate_run(rd, segs, gold, n_boot=n_boot)})
    out = Path(cfg["selection_out"])
    paths = ev.write_report(results, out, title="再訓練：VLM 特徵＋小分類器（開發集，探索性）", split_name="dev",
                            notes=["特徵與挑選規則在抽特徵之前寫好（configs/vlm_probe.yaml）；依訓練池（其他零件）交叉驗證挑，開發集只看。",
                                   "開發集只有 20 段（14 支錄影），信賴區間很寬；前兩次測試集都顯示開發集會高估。",
                                   "這個方法沒有在測試集跑（使用者 2026-10-16 決定不再動測試集）。"], n_boot=n_boot)

    def pct(v: Any) -> str:
        return "–" if v is None else f"{100 * v:.0f}%"

    def frac(a: Any, b: Any) -> str:
        return "–" if a is None or not b else f"{a}/{b}"

    lines = ["# 再訓練：VLM 特徵＋小分類器——候選比較（依交叉驗證排序）", "", f"產生時間：{now_iso()}　設定：`{cfg['_path']}`", "",
             "挑選規則（事先決定）：訓練池（其他零件 368 段，排除 BNC 測試錄影）依錄影分組交叉驗證的**平衡準確率**最高；"
             "同分看失敗召回，再選維度少的。BNC 開發集只列出來看。", "",
             "| | 特徵 | 維度 | 交叉驗證平衡準確率 | 交叉驗證失敗召回 | C | PCA | 開發集答對 | 開發集抓到失敗 | 開發集成功被誤判 | 開發集兩類平均 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (-r["cv_balanced_accuracy"], -r["cv_failure_recall"], r["dim"], r["order"])):
        lines.append(f"| {'★' if best and r is best else ''} | {r['feature']} | {r['dim']} | {pct(r['cv_balanced_accuracy'])} | "
                     f"{pct(r['cv_failure_recall'])} | {r['C']:g} | {r['pca']} | {frac(r['dev_correct'], r['dev_n'])} | "
                     f"{frac(r['dev_caught'], r['dev_failures'])} | {frac(r['dev_false_alarms'], r['dev_successes'])} | {pct(r['dev_balanced'])} |")
    lines += ["", "對照（同一個訓練池的交叉驗證）：做法 A 改良的候選在 58–73%（`experiments/probe_selection/selection.md`）。",
              f"開發集完整評分（含信賴區間與參考方法）：`{paths['md']}`。"]
    (out / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = {"config": cfg["_path"], "rows": rows, "best": best, "rule": "cv_balanced_accuracy → cv_failure_recall → dim → order",
           "report": paths["md"], "generated_at": now_iso()}
    (out / "selection.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    rep["md"] = str(out / "selection.md")
    return rep


def describe_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    exp, data = samples(cfg)
    have = {n: sum(1 for s in data[n] if feature_file(cfg, s["segment_id"]).exists()) for n in ("train", "dev")}
    return {"sampling_key": data["sampling_key"], "train": finetune.sample_summary(data["train"]),
            "dev": finetune.sample_summary(data["dev"]), "train_missing_frames": data["train_missing"],
            "dev_missing_frames": data["dev_missing"], "features_done": have,
            "n_test_recordings_excluded": data["n_test_recordings_excluded"]}


__all__ = ["load_vlm_probe_config", "layer_name", "layer_index", "feature_file", "extract", "load_matrix", "select_best",
           "fit", "describe_plan"]
