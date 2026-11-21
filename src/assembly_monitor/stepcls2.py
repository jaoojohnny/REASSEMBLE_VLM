"""第 20 包：補強步驟分類器——只在 5 個開發零件上，比三種「給分類器看的數字」（bolt 4 不碰）。

- C0 原本（第 19 包凍結）：步驟結束後 1 秒、手腕、中間 60%、DINOv2 CLS＋小區塊平均（1,536）
- C1 多看幾格：結束後 0、0.5、1 秒三格的 C0 數字串起來（4,608）
- C2 空間細節：同一格，DINOv2 小區塊切成 3×3 區域各自平均＋CLS（7,680）
- C3 加外部相機：同一時間兩台外部相機整張畫面（CLS＋平均，各 1,536）＋C0（4,608）
分類器一律是第 14 包凍結的 F2S2（pp.nested_cv，"S2"）。選擇規則與預測：docs/package20_stepboost.md。
輸出：experiments/p20_stepboost_dev/
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import perpart as pp
from . import stepcls as sc

log = logging.getLogger(__name__)

OUT_DEV = Path("experiments/p20_stepboost_dev")
SELECTED = Path("configs/stepcls2_selected.yaml")
EXT_FRAMES_ROOT = "data/derived/frames_steps_ext"
EXT_CAMS = ("hama1", "hama2")
EB = 1.0                      # 第 19 包凍結的時間點
MULTI_EB = (0.0, 0.5, 1.0)
GRID = 3
CANDS = ("C0", "C1", "C2", "C3")   # C1 → C2 → C3 是「從簡單到複雜」的順序
CAND_NAME = {"C0": "原本（結束後 1 秒、手腕）", "C1": "多看幾格（0／0.5／1 秒三格）",
             "C2": "空間細節（手腕 3×3 區塊）", "C3": "加外部相機（兩台外部＋手腕）"}
MARGIN = 0.02
ENCODER_GRID = dict(pp.ENCODER_CROP, cache_tag=f"wristcrop_02_08_grid{GRID}", grid=GRID)


class Stop(RuntimeError):
    """缺資料或檢查沒過，不計分。"""


# ---------------------------------------------------------------- 3×3 區塊平均（純 numpy，可單獨測試）


def _bins(n: int, g: int) -> list[tuple[int, int]]:
    """和 torch.nn.functional.adaptive_avg_pool 相同的切法：第 i 塊 = [floor(i·n/g), ceil((i+1)·n/g))。"""
    return [((i * n) // g, -((-(i + 1) * n) // g)) for i in range(g)]


def grid_pool(h: np.ndarray, gh: int, gw: int, g: int = GRID) -> np.ndarray:
    """h：[張數, 1＋gh·gw, D]（DINOv2 最後一層：CLS＋小區塊）→ [張數, D·(1＋g·g)]：CLS 接 g×g 區域的小區塊平均（列優先）。"""
    h = np.asarray(h, np.float32)
    b, n, d = h.shape
    if n != 1 + gh * gw:
        raise ValueError(f"小區塊數 {n - 1} 和 {gh}×{gw} 不合")
    p = h[:, 1:].reshape(b, gh, gw, d)
    cells = [p[:, r0:r1, c0:c1].mean(axis=(1, 2)) for r0, r1 in _bins(gh, g) for c0, c1 in _bins(gw, g)]
    return np.concatenate([h[:, 0]] + cells, axis=1)


def make_grid_encoder() -> Any:
    """手腕中間 60% 裁切 → DINOv2 → CLS＋3×3 區塊平均。第一次 encode 才載入模型（要 GPU）。"""
    from . import probe

    class GridCropEncoder(probe.TransformersImageEncoder):
        loaded = False

        def encode(self, paths: list[str]) -> np.ndarray:
            from PIL import Image

            if not self.loaded:
                self.load()
                self.loaded = True
            torch = self.torch
            box, g = self.cfg["crop_box"], int(self.cfg["grid"])
            ps = int(getattr(self.model.config, "patch_size", 14))
            feats = []
            bs = int(self.cfg.get("batch_size", 32))
            for i in range(0, len(paths), bs):
                imgs = []
                for p in paths[i:i + bs]:
                    im = Image.open(p).convert("RGB")
                    w, hh = im.size
                    imgs.append(im.crop((int(box[0] * w), int(box[1] * hh), int(box[2] * w), int(box[3] * hh))))
                inputs = self.processor(images=imgs, return_tensors="pt", **(self.cfg.get("processor_kwargs") or {}))
                inputs = inputs.to(self.device, dtype=self.dtype)
                H, W = inputs["pixel_values"].shape[-2:]
                with torch.inference_mode():
                    o = self.net(**inputs)
                feats.append(grid_pool(o.last_hidden_state.float().cpu().numpy(), int(H) // ps, int(W) // ps, g))
            return np.concatenate(feats).astype(np.float32)

        def close(self) -> None:
            if self.loaded:
                super().close()
                self.loaded = False

    return GridCropEncoder(ENCODER_GRID)


def check_grid_cache(cache_root: Path, key: str) -> None:
    ident = {"hf_id": ENCODER_GRID["hf_id"], "processor_kwargs": ENCODER_GRID["processor_kwargs"],
             "crop_box": list(pp.CROP_BOX), "camera": "hand", "grid": GRID}
    p = Path(cache_root) / key / "_encoder.json"
    if p.exists():
        if json.loads(p.read_text(encoding="utf-8")) != json.loads(json.dumps(ident)):
            raise ValueError(f"{p.parent} 的特徵是用不同設定做的")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ident, ensure_ascii=False, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------- 各種改法的數字


def step_vectors(task: dict[str, Any], enc: Any, eb: float, cameras: tuple[str, ...], frames_root: str,
                 cache_root: Path) -> dict[str, np.ndarray]:
    """題目裡每個步驟的向量（cameras 依序串接）；有任何一台相機缺影格就不在回傳裡。"""
    from . import probe

    cfg = {"frames_root": frames_root, "action": "insert", "sampling_template": "uniform_n{n}_sb0.5_eb%g_jpg" % eb,
           "cameras": list(cameras)}
    out = {}
    for s in task["instances"]:
        f = probe.segment_features(cfg, enc, {"segment_id": s["segment_id"], "part": task["name"]}, 1, Path(cache_root))
        if f is None:
            continue
        v = np.asarray(f, float)[0]
        if not np.isfinite(v).all():
            continue
        out[s["segment_id"]] = v.reshape(-1)
    return out


def candidate_features(task: dict[str, Any], encs: dict[str, Any], cache_root: Path = sc.CACHE_ROOT,
                       frames_root: str = sc.FRAMES_ROOT, ext_root: str = EXT_FRAMES_ROOT) -> dict[str, dict[str, np.ndarray]]:
    """回傳 {改法: {步驟: 向量}}。"""
    c0 = sc.task_features(task, EB, encs["crop"], Path(cache_root), frames_root)
    multi = [sc.task_features(task, eb, encs["crop"], Path(cache_root), frames_root) for eb in MULTI_EB]
    c1 = {sid: np.concatenate([m[sid] for m in multi]) for sid in multi[0] if all(sid in m for m in multi)}
    c2 = step_vectors(task, encs["grid"], EB, ("hand",), frames_root, Path(cache_root))
    ext = step_vectors(task, encs["full"], EB, EXT_CAMS, ext_root, Path(cache_root))
    c3 = {sid: np.concatenate([ext[sid], c0[sid]]) for sid in ext if sid in c0}
    return {"C0": c0, "C1": c1, "C2": c2, "C3": c3}


# ---------------------------------------------------------------- 評分


def last_flags(instances: list[dict[str, Any]]) -> dict[str, bool]:
    """每個步驟是不是它那一段的最後一步（instances 要是整個零件的所有步驟）。"""
    mx: dict[str, int] = defaultdict(int)
    for s in instances:
        mx[s["parent"]] = max(mx[s["parent"]], s["order"])
    return {s["segment_id"]: s["order"] == mx[s["parent"]] for s in instances}


def evaluate_task(task: dict[str, Any], feats: dict[str, dict[str, np.ndarray]], is_last: dict[str, bool],
                  cands: tuple[str, ...] = CANDS, min_each: int = sc.MIN_EACH) -> dict[str, Any]:
    """同一批步驟（每種改法都有數字的）在每種改法各跑一次凍結的 F2S2 巢狀交叉驗證。"""
    inst = [s for s in task["instances"] if all(s["segment_id"] in feats[c] for c in cands)]
    y = np.array([s["label"] for s in inst], int)
    groups = [s["recording_id"] for s in inst]
    hard = np.array([bool(s["label"]) and not is_last.get(s["segment_id"], True) for s in inst])
    lastf = np.array([bool(s["label"]) and is_last.get(s["segment_id"], True) for s in inst])
    res: dict[str, Any] = {"name": task["name"], "part": task["part"], "step": task["step"], "n": len(inst),
                           "n_failure": int(y.sum()), "n_success": int(len(y) - y.sum()), "n_hard": int(hard.sum()),
                           "n_last_fail": int(lastf.sum()), "n_dropped": len(task["instances"]) - len(inst),
                           "ids": [s["segment_id"] for s in inst], "labels": y.tolist(), "cands": {}}
    if res["n_failure"] < min_each or res["n_success"] < min_each:
        res["skipped"] = f"四種改法都有數字的步驟裡，成功或失敗不到 {min_each} 筆"
        return res
    for c in cands:
        X = np.vstack([feats[c][s["segment_id"]] for s in inst])
        t0 = time.time()
        r = pp.nested_cv(X, y, groups, "S2")
        judged = np.asarray(r["fail_rate"], float) > 0.5
        res["cands"][c] = {"bal_mean": r["bal_mean"], "bal_min": r["bal_min"], "bal_max": r["bal_max"],
                           "dim": int(X.shape[1]), "k_mean": r["k_mean"], "fail_rate": r["fail_rate"],
                           "judged": judged.astype(int).tolist(),
                           "hard_hit": int(np.sum(judged & hard)), "last_hit": int(np.sum(judged & lastf)),
                           "ok_hit": int(np.sum(~judged & (y == 0))), "seconds": round(time.time() - t0, 1)}
        log.info("  %-20s %s：兩類平均 %.0f%%（%.0f%%～%.0f%%）；難題抓到 %d/%d", task["name"], c, r["bal_mean"] * 100,
                 r["bal_min"] * 100, r["bal_max"] * 100, res["cands"][c]["hard_hit"], res["n_hard"])
    return res


def which_step_for(cand: str, instances: list[dict[str, Any]], task_results: list[dict[str, Any]]) -> dict[str, Any]:
    tr = [{"name": t["name"], "step": t["step"], "ids": t["ids"], "judged": t["cands"][cand]["judged"]}
          for t in task_results if not t.get("skipped")]
    return sc.which_step(instances, tr)


def summarize(task_results: list[dict[str, Any]], cands: tuple[str, ...] = CANDS, margin: float = MARGIN) -> dict[str, Any]:
    done = [t for t in task_results if not t.get("skipped")]
    if not done:
        raise Stop("沒有可以計分的題目")
    avg = {c: float(np.mean([t["cands"][c]["bal_mean"] for t in done])) for c in cands}
    hard = {c: int(sum(t["cands"][c]["hard_hit"] for t in done)) for c in cands}
    last = {c: int(sum(t["cands"][c]["last_hit"] for t in done)) for c in cands}
    ok = {c: int(sum(t["cands"][c]["ok_hit"] for t in done)) for c in cands}
    n_hard = int(sum(t["n_hard"] for t in done))
    eligible = [c for c in cands[1:] if avg[c] >= avg["C0"] + margin - 1e-12 and hard[c] >= hard["C0"]]
    if eligible:
        best = max(avg[c] for c in eligible)
        pick = next(c for c in cands[1:] if c in eligible and avg[c] >= best - margin - 1e-12)
    else:
        pick = "C0"
    return {"avg": avg, "hard_hit": hard, "n_hard": n_hard, "last_hit": last,
            "n_last_fail": int(sum(t["n_last_fail"] for t in done)), "ok_hit": ok,
            "n_success": int(sum(t["n_success"] for t in done)), "n_tasks": len(done), "eligible": eligible,
            "pick": pick, "improved": pick != "C0"}


def write_selected(summary: dict[str, Any], path: Path = SELECTED) -> Path | None:
    """選出 C1–C3 才凍結；已經凍結過就不覆蓋。內容是 JSON（也是合法的 YAML）。"""
    if not summary["improved"]:
        return None
    path = Path(path)
    if path.exists():
        log.warning("%s 已經存在，不覆蓋（做法已凍結）", path)
        return path
    body = {"package": 20, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "recipe": "F2S2", "candidate": summary["pick"],
            "candidate_name": CAND_NAME[summary["pick"]], "eb": EB, "multi_eb": list(MULTI_EB), "grid": GRID,
            "ext_cams": list(EXT_CAMS), "dev_avg": summary["avg"], "dev_hard_hit": summary["hard_hit"],
            "min_each": sc.MIN_EACH,
            "f2s2_grid": {"k": list(pp.K_GRID), "c1": list(pp.C1_GRID), "c2": list(pp.C2_GRID)},
            "cv": {"outer": pp.OUTER, "inner": pp.INNER, "repeats": pp.REPEATS}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# 第 20 包凍結的做法（docs/package20_stepboost.md）；不要手動改\n"
                    + json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- 開發階段


def extract_ext(tasks: list[dict[str, Any]], index_dir: Path = sc.STEP_INDEX, frames_root: str = EXT_FRAMES_ROOT,
                slim_dir: str = "data/slim", mapping_path: str = "configs/field_mapping.yaml", workers: int = 4) -> dict[str, Any]:
    """兩台外部相機在「步驟結束後 1 秒」的影格（另一個資料夾，不動第 19 包的手腕影格）。"""
    from .extract import run_extraction
    from .mapping import load_mapping

    mapping = load_mapping(mapping_path)
    tot: dict[str, Any] = {"n_done": 0, "n_skipped": 0, "n_missing_frames": 0, "errors": []}
    for t in tasks:
        ids = [s["segment_id"] for s in t["instances"]]
        res = run_extraction(mapping=mapping, index_dir=index_dir, slim_dir=slim_dir, out_dir=frames_root, part=t["name"],
                             action="insert", which="steps", n_frames_list=(1,), start_buffer=0.5, end_buffer=EB,
                             cameras=list(EXT_CAMS), workers=workers, segment_ids=ids)
        log.info("  %-20s 外部相機：%3d 筆，新擷取 %d、已有 %d", t["name"], len(ids), res["n_done"], res["n_skipped"])
        for k in ("n_done", "n_skipped", "n_missing_frames"):
            tot[k] += res[k]
        tot["errors"] += res["errors"]
    return tot


def run_dev(index_dir: str = "data/index", out: Path = OUT_DEV, parts: tuple[str, ...] = pp.DEV_PARTS,
            cache_root: Path = sc.CACHE_ROOT, frames_root: str = sc.FRAMES_ROOT, ext_root: str = EXT_FRAMES_ROOT,
            step_index: Path = sc.STEP_INDEX, selected_path: Path = SELECTED, encs: dict[str, Any] | None = None,
            do_extract: bool = True, features: Callable[..., dict[str, dict[str, np.ndarray]]] | None = None) -> dict[str, Any]:
    if sc.TEST_PART in parts or any(p in pp.RESERVED for p in parts):
        raise ValueError("開發階段只用開發零件")
    t0 = time.time()
    out = Path(out)
    tasks = sc.build_tasks(index_dir, parts)
    if not tasks:
        raise Stop("開發零件裡沒有成功、失敗都夠多的步驟")
    log.info("題目 %d 個：%s", len(tasks), "、".join(t["name"] for t in tasks))
    sc.write_step_index(tasks, step_index)
    ext = None
    if do_extract:
        w = sc.extract(tasks, MULTI_EB, index_dir=step_index, frames_root=frames_root)   # 第 19 包已經擷取過 → 都會略過
        e = extract_ext(tasks, index_dir=step_index, frames_root=ext_root)
        ext = {k: w[k] + e[k] for k in ("n_done", "n_skipped", "n_missing_frames")}
        ext["errors"] = w["errors"] + e["errors"]
        log.info("影格：新擷取 %d、已有 %d、缺格 %d；錯誤 %d", ext["n_done"], ext["n_skipped"], ext["n_missing_frames"],
                 len(ext["errors"]))
        for er in ext["errors"][:10]:
            log.error("  %s", er)
    own = encs is None
    if own:
        base = pp.make_encoders()
        encs = {"crop": base["crop"], "full": base["full"], "grid": make_grid_encoder(), "_base": base}
    feats_all = []
    try:
        if features is None:
            from . import probe

            pp._check_crop_cache(Path(cache_root), encs["crop"].key)
            check_grid_cache(Path(cache_root), encs["grid"].key)
            probe.check_feature_cache(Path(cache_root), encs["full"], pp.ENCODER)
        for t in tasks:
            feats_all.append((features or candidate_features)(t, encs, Path(cache_root), frames_root, ext_root))
    finally:
        if own:
            for k in ("full", "crop"):
                encs["_base"][k].close()
            encs["grid"].close()
    results = []
    lasts: dict[str, dict[str, bool]] = {}
    for t, f in zip(tasks, feats_all):
        if t["part"] not in lasts:
            lasts[t["part"]] = last_flags(sc.step_instances(index_dir, t["part"]))
        results.append(evaluate_task(t, f, lasts[t["part"]]))
    summary = summarize(results)
    ws = {}
    for part in sorted({t["part"] for t in results if not t.get("skipped")}):
        mine = [t for t in results if t["part"] == part and not t.get("skipped")]
        if len(mine) >= 2:
            inst = sc.step_instances(index_dir, part)
            ws[part] = {c: {k: v for k, v in which_step_for(c, inst, mine).items() if k != "rows"} for c in CANDS}
    frozen = write_selected(summary, selected_path)
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tasks": results, "summary": summary, "which_step": ws,
           "selected_file": str(frozen) if frozen else None,
           "extract": {k: v for k, v in (ext or {}).items() if k != "errors"} if ext else None,
           "extract_errors": (ext or {}).get("errors", [])[:20], "seconds": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_dev_report(res, out / "report.md")
    return res


# ---------------------------------------------------------------- 報告


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def write_dev_report(res: dict[str, Any], path: Path) -> None:
    s = res["summary"]
    L = ["# 第 20 包（開發階段）：補強步驟分類器——三種改法比一比", "",
         f"產生時間：{res['generated']}；分類器一律是第 14 包凍結的 F2S2，只換給它看的數字（事先登記 `docs/package20_stepboost.md`）。"
         "bolt 4 這裡沒有碰。", "",
         "## 兩類平均（每題；同一批步驟比四種）", "",
         "| 題目 | 成功／失敗（筆） | 難題（筆） | " + " | ".join(f"{c} {CAND_NAME[c]}" for c in CANDS) + " |",
         "|---|---|---|" + "---|" * len(CANDS)]
    for t in res["tasks"]:
        head = f"| {t['part']} {sc.zh(t['step'])} | {t['n_success']}／{t['n_failure']} | {t['n_hard']} | "
        if t.get("skipped"):
            L.append(head + " | ".join("沒做" for _ in CANDS) + " |")
            continue
        L.append(head + " | ".join(f"{_pct(t['cands'][c]['bal_mean'])}（難題 {t['cands'][c]['hard_hit']}）" for c in CANDS) + " |")
    L += [f"| **{s['n_tasks']} 題平均** | | {s['n_hard']} | " + " | ".join(f"**{_pct(s['avg'][c])}**" for c in CANDS) + " |", "",
          "- 兩類平均＝失敗的步驟答對率和成功的步驟答對率的平均；一律答成功＝50%。",
          "- 難題＝失敗的步驟裡，後面還有別的步驟的那些；括號是抓到幾筆（5 次重複裡過半數判失敗）。", "",
          "## 合起來", "",
          "| 改法 | 平均 | 難題抓到 | 失敗而且是最後一步：抓到 | 成功的步驟答對 |", "|---|---|---|---|---|"]
    for c in CANDS:
        L.append(f"| {c} {CAND_NAME[c]} | {_pct(s['avg'][c])} | {s['hard_hit'][c]}/{s['n_hard']} | "
                 f"{s['last_hit'][c]}/{s['n_last_fail']} | {s['ok_hit'][c]}/{s['n_success']} |")
    if res.get("which_step"):
        L += ["", "## 失敗在哪一步（有兩個以上步驟分類器的零件）", ""]
        for part, d in res["which_step"].items():
            L.append(f"- {part}：" + "、".join(f"{c} 答對 {d[c]['n_right']}/{d[c]['n_failed']}" for c in CANDS)
                     + f"（一律答最多的那一步 {d['C0']['always_most']}/{d['C0']['n_failed']}）")
    L += ["", "## 選擇（規則事先寫好）", "",
          f"- 能取代 C0 的條件：平均 ≥ C0（{_pct(s['avg']['C0'])}）＋ 2 點，而且難題抓到的不比 C0（{s['hard_hit']['C0']} 筆）少。",
          "- 符合條件的：" + ("、".join(s["eligible"]) if s["eligible"] else "沒有"),
          (f"- **選 {s['pick']} {CAND_NAME[s['pick']]}**，已凍結（`configs/stepcls2_selected.yaml`）；下一步才套到 bolt 4。"
           if s["improved"] else "- **保留 C0**：這三種改法都沒有讓步驟分類器變好；bolt 4 不重跑。"), "",
          "## 注意", "",
          "- 失敗的步驟每題只有 7–18 筆，難題更少；差 1–2 筆可能只是運氣。",
          "- 每種改法的數字個數："
          + "、".join(f"{c} {next((t['cands'][c]['dim'] for t in res['tasks'] if not t.get('skipped')), 0):,}" for c in CANDS)
          + "；分類器都只留 10 個以內。"]
    drop = [(t["name"], t["n_dropped"]) for t in res["tasks"] if t.get("n_dropped")]
    if drop:
        L.append("- 有改法缺數字、不納入的步驟：" + "、".join(f"{n} {k} 筆" for n, k in drop))
    if res.get("extract"):
        e = res["extract"]
        L.append(f"- 影格：新擷取 {e['n_done']}、已有 {e['n_skipped']}、缺格 {e['n_missing_frames']}；"
                 f"擷取錯誤 {len(res.get('extract_errors') or [])} 筆")
    L += [f"- 總共 {res['seconds']:.0f} 秒。", ""]
    path.write_text("\n".join(L), encoding="utf-8")
