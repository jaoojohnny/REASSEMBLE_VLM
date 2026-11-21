"""第 19 包：每個步驟自己的小分類器——在每一步剛做完時看一張手腕影格，問「這一步成功了嗎」。

- 一筆資料＝一個步驟（data/index/skills.csv；成功段裡的步驟也算），答案＝這一步自己的成功／失敗標記。
- 一個題目＝某零件的某種步驟（成功、失敗都 ≥ 7 筆）。每一題都用第 14 包凍結的 F2S2（pp.nested_cv，"S2"）。
- 開發階段（這個檔的 run_dev）：5 個開發零件的題目，比「步驟結束後 0／0.5／1 秒」三個時間點；
  平均最高、差 2 點以內選早的；平均 ≥ 65% 才凍結（configs/stepcls_selected.yaml）。bolt 4 這裡不碰。
- 測試階段（run_test）：凍結的時間點套到 bolt 4 的旋轉、對準兩題，各跑一次；另外算「失敗在哪一步」
  （每段依序看每一步的判斷，第一個被判失敗的步驟）和打亂答案檢定。
- 步驟的影格用第 3 包的擷取程式：先把每個步驟寫成一列「段」（data/index_steps/segments.csv），
  段編號＝原段編號×100＋步驟編號。
說明：docs/package19_stepcls.md；輸出：experiments/p19_stepcls_dev/、experiments/p19_stepcls_test/
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import perpart as pp

log = logging.getLogger(__name__)

STEP_INDEX = Path("data/index_steps")
FRAMES_ROOT = "data/derived/frames_steps"
CACHE_ROOT = Path("experiments/_features_steps")
OUT_DEV = Path("experiments/p19_stepcls_dev")
OUT_TEST = Path("experiments/p19_stepcls_test")
N_PERM = 100
PERM_SEED = 20261117
PASS_BAL, PASS_P = 0.65, 0.05
SELECTED = Path("configs/stepcls_selected.yaml")
OFFSETS = (0.0, 0.5, 1.0)          # 步驟結束後幾秒；同分時選前面（早）的
MIN_EACH = 7
CLOSE = 0.02
GATE = 0.65
TEST_PART = "bolt 4"
STEP_ZH = {"Approach": "接近", "Align": "對準", "Twist": "旋轉", "Push": "推入", "Release": "放開", "Nudge": "輕推",
           "Lift": "抬起", "Grasp": "夾取", "Pull": "拉出"}
INDEX_COLS = ["recording_id", "segment_id", "action", "part", "validation_status", "start_unix", "end_unix", "text",
              "success", "error_code"]


class Stop(RuntimeError):
    """缺資料或檢查沒過，不計分。"""


def zh(step: str) -> str:
    return STEP_ZH.get(step, step)


def task_name(part: str, step: str) -> str:
    return f"{part} {step}"


def sampling_key(eb: float) -> str:
    """和第 3 包擷取程式的資料夾名稱相同（n＝1、段前 0.5 秒、結束後 eb 秒）。"""
    return f"uniform_n1_sb0.5_eb{eb:g}_jpg"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_") or "x"


# ---------------------------------------------------------------- 步驟與題目


def step_instances(index_dir: str | Path, part: str) -> list[dict[str, Any]]:
    """某零件所有插入段（驗證 ok／warn、有成敗標記）裡的每一個步驟。label：1＝這一步失敗、0＝成功。"""
    if part in pp.RESERVED:
        raise ValueError(f"{part} 是保留零件，不碰")
    segs = {}
    with (Path(index_dir) / "segments.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if (r.get("action") == "insert" and r.get("part") == part and r.get("validation_status") in ("ok", "warn")
                    and r.get("success") in ("True", "False")):
                segs[(r["recording_id"], r["segment_id"])] = r
    out = []
    with (Path(index_dir) / "skills.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            k = (r["recording_id"], r["segment_id"])
            if k not in segs or r.get("success") not in ("True", "False"):
                continue
            if (r.get("validation_status") or "ok") not in ("ok", "warn"):
                continue
            order = int(r["skill_id"])
            if not 0 <= order < 100:
                raise ValueError(f"步驟編號超出範圍：{k} {order}")
            num = int(r["segment_id"]) * 100 + order
            out.append({"segment_id": f"{r['recording_id']}__seg{num}", "num": num, "recording_id": r["recording_id"],
                        "part": part, "step": r["text"], "label": 0 if r["success"] == "True" else 1,
                        "start_unix": float(r["start_unix"]), "end_unix": float(r["end_unix"]), "order": order,
                        "parent": f"{r['recording_id']}__seg{int(r['segment_id'])}",
                        "parent_failed": segs[k]["success"] == "False"})
    return sorted(out, key=lambda s: (s["recording_id"], s["num"]))


def build_tasks(index_dir: str | Path, parts: tuple[str, ...], min_each: int = MIN_EACH) -> list[dict[str, Any]]:
    """題目＝（零件, 步驟），成功、失敗都 ≥ min_each 筆。依零件順序、步驟名稱排。"""
    tasks = []
    for part in parts:
        by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in step_instances(index_dir, part):
            by_step[s["step"]].append(s)
        for step in sorted(by_step):
            inst = by_step[step]
            n_fail = sum(s["label"] for s in inst)
            if n_fail >= min_each and len(inst) - n_fail >= min_each:
                tasks.append({"name": task_name(part, step), "part": part, "step": step, "instances": inst})
    return tasks


def write_step_index(tasks: list[dict[str, Any]], index_dir: str | Path = STEP_INDEX) -> Path:
    """把每個步驟寫成第 3 包擷取程式看得懂的一列「段」（零件欄放題目名稱）。已有的其他題目保留。"""
    p = Path(index_dir) / "segments.csv"
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    if p.exists():
        with p.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rows[(r["part"], r["recording_id"], r["segment_id"])] = {c: r.get(c, "") for c in INDEX_COLS}
    for t in tasks:
        for k in [k for k in rows if k[0] == t["name"]]:
            del rows[k]
        for s in t["instances"]:
            rows[(t["name"], s["recording_id"], str(s["num"]))] = {
                "recording_id": s["recording_id"], "segment_id": str(s["num"]), "action": "insert", "part": t["name"],
                "validation_status": "ok", "start_unix": f"{s['start_unix']:.6f}", "end_unix": f"{s['end_unix']:.6f}",
                "text": f"{t['step']} ({t['part']})", "success": "False" if s["label"] else "True", "error_code": ""}
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    return p


# ---------------------------------------------------------------- 影格與特徵（只在使用者電腦跑：要精簡版 h5、GPU）


def extract(tasks: list[dict[str, Any]], offsets: tuple[float, ...] = OFFSETS, index_dir: str | Path = STEP_INDEX,
            slim_dir: str = "data/slim", frames_root: str = FRAMES_ROOT,
            mapping_path: str = "configs/field_mapping.yaml", workers: int = 4) -> dict[str, Any]:
    """每一步「結束後 eb 秒」的手腕影格（和 pp.extract_frames 同樣的呼叫方式；已有的略過）。"""
    from .extract import run_extraction
    from .mapping import load_mapping

    mapping = load_mapping(mapping_path)
    tot: dict[str, Any] = {"n_done": 0, "n_skipped": 0, "n_missing_frames": 0, "errors": []}
    for t in tasks:
        ids = [s["segment_id"] for s in t["instances"]]
        for eb in offsets:
            res = run_extraction(mapping=mapping, index_dir=index_dir, slim_dir=slim_dir, out_dir=frames_root,
                                 part=t["name"], action="insert", which="steps", n_frames_list=(1,), start_buffer=0.5,
                                 end_buffer=float(eb), cameras=["hand"], workers=workers, segment_ids=ids)
            log.info("  %-20s 結束後 %g 秒：%3d 筆，新擷取 %d、已有 %d", t["name"], eb, len(ids), res["n_done"],
                     res["n_skipped"])
            for k in ("n_done", "n_skipped", "n_missing_frames"):
                tot[k] += res[k]
            tot["errors"] += res["errors"]
    return tot


def task_features(task: dict[str, Any], eb: float, enc: Any, cache_root: Path = CACHE_ROOT,
                  frames_root: str = FRAMES_ROOT) -> dict[str, np.ndarray]:
    """題目裡每個步驟的 F2 向量（手腕、中間 60%、DINOv2 1,536 維）；沒有影格的不在回傳裡。"""
    from . import probe

    cfg = {"frames_root": frames_root, "action": "insert", "sampling_template": "uniform_n{n}_sb0.5_eb%g_jpg" % eb,
           "cameras": ["hand"]}
    out = {}
    for s in task["instances"]:
        f = probe.segment_features(cfg, enc, {"segment_id": s["segment_id"], "part": task["name"]}, 1, Path(cache_root))
        if f is None:
            continue
        v = np.asarray(f, float)[0, 0]
        if not np.isfinite(v).any():
            continue
        out[s["segment_id"]] = np.nan_to_num(v, nan=0.0)
    return out


def frame_path(task_name_: str, eb: float, segment_id: str, frames_root: str = FRAMES_ROOT) -> str | None:
    d = Path(frames_root) / f"insert_{_slug(task_name_)}" / sampling_key(eb) / "segments" / segment_id
    g = sorted(glob.glob(str(d / "hand__00__*.jpg")))
    return g[0] if g else None


# ---------------------------------------------------------------- 評分與選擇


def evaluate_task(task: dict[str, Any], feats: dict[float, dict[str, np.ndarray]],
                  offsets: tuple[float, ...] = OFFSETS, min_each: int = MIN_EACH) -> dict[str, Any]:
    """同一批步驟（每個時間點都有影格的）在每個時間點各跑一次凍結的 F2S2 巢狀交叉驗證。"""
    inst = [s for s in task["instances"] if all(s["segment_id"] in feats[eb] for eb in offsets)]
    y = np.array([s["label"] for s in inst], int)
    groups = [s["recording_id"] for s in inst]
    res: dict[str, Any] = {"name": task["name"], "part": task["part"], "step": task["step"], "n": len(inst),
                           "n_failure": int(y.sum()), "n_success": int(len(y) - y.sum()),
                           "n_dropped": len(task["instances"]) - len(inst), "n_recordings": len(set(groups)),
                           "ids": [s["segment_id"] for s in inst], "labels": y.tolist(), "offsets": {}}
    if res["n_failure"] < min_each or res["n_success"] < min_each:
        res["skipped"] = f"去掉沒有影格的步驟後，成功或失敗不到 {min_each} 筆"
        return res
    for eb in offsets:
        X = np.vstack([feats[eb][s["segment_id"]] for s in inst])
        t0 = time.time()
        r = pp.nested_cv(X, y, groups, "S2")
        res["offsets"][f"{eb:g}"] = {
            "bal_mean": r["bal_mean"], "bal_min": r["bal_min"], "bal_max": r["bal_max"],
            "fail_recall": float(np.mean([p["ok_failure"] / p["n_failure"] for p in r["repeats"]])),
            "ok_recall": float(np.mean([p["ok_success"] / p["n_success"] for p in r["repeats"]])),
            "k_mean": r["k_mean"], "chosen": dict(list(r["chosen"].items())[:3]), "fail_rate": r["fail_rate"],
            "seconds": round(time.time() - t0, 1)}
        log.info("  %-20s 結束後 %g 秒：兩類平均 %.0f%%（%.0f%%～%.0f%%）", task["name"], eb, r["bal_mean"] * 100,
                 r["bal_min"] * 100, r["bal_max"] * 100)
    return res


def select_offset(avg: dict[float, float], offsets: tuple[float, ...] = OFFSETS, close: float = CLOSE) -> dict[str, Any]:
    """事先寫好的規則：平均最高；和最高差 close 以內，選比較早的。"""
    best = max(avg.values())
    pick = next(eb for eb in sorted(offsets) if avg[eb] >= best - close - 1e-12)
    return {"offset": pick, "best_offset": max(sorted(offsets), key=lambda eb: avg[eb]), "best_avg": best,
            "avg": {f"{eb:g}": avg[eb] for eb in sorted(offsets)}}


def summarize(task_results: list[dict[str, Any]], offsets: tuple[float, ...] = OFFSETS, gate: float = GATE) -> dict[str, Any]:
    done = [t for t in task_results if not t.get("skipped")]
    if not done:
        raise Stop("沒有可以計分的題目")
    avg = {eb: float(np.mean([t["offsets"][f"{eb:g}"]["bal_mean"] for t in done])) for eb in offsets}
    sel = select_offset(avg, offsets)
    key = f"{sel['offset']:g}"
    by_step: dict[str, list[float]] = defaultdict(list)
    for t in done:
        by_step[t["step"]].append(t["offsets"][key]["bal_mean"])
    sel.update(n_tasks=len(done), selected_avg=avg[sel["offset"]], gate=gate, gate_passed=bool(avg[sel["offset"]] >= gate),
               by_step={s: float(np.mean(v)) for s, v in by_step.items()})
    return sel


def write_selected(summary: dict[str, Any], tasks: list[dict[str, Any]], path: Path = SELECTED) -> Path | None:
    """過關才凍結；已經凍結過就不覆蓋。內容是 JSON（也是合法的 YAML）。"""
    if not summary["gate_passed"]:
        return None
    path = Path(path)
    if path.exists():
        log.warning("%s 已經存在，不覆蓋（做法已凍結）", path)
        return path
    body = {"package": 19, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "recipe": "F2S2",
            "offset_s": summary["offset"], "dev_avg": summary["avg"], "gate": summary["gate"],
            "dev_tasks": [t["name"] for t in tasks if not t.get("skipped")], "min_each": MIN_EACH,
            "grid": {"k": list(pp.K_GRID), "c1": list(pp.C1_GRID), "c2": list(pp.C2_GRID)},
            "cv": {"outer": pp.OUTER, "inner": pp.INNER, "repeats": pp.REPEATS}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# 第 19 包凍結的做法（docs/package19_stepcls.md）；不要手動改\n"
                    + json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- 開發階段


def run_dev(index_dir: str = "data/index", out: Path = OUT_DEV, parts: tuple[str, ...] = pp.DEV_PARTS,
            offsets: tuple[float, ...] = OFFSETS, cache_root: Path = CACHE_ROOT, frames_root: str = FRAMES_ROOT,
            step_index: Path = STEP_INDEX, selected_path: Path = SELECTED, encs: dict[str, Any] | None = None,
            do_extract: bool = True, sheets: bool = True) -> dict[str, Any]:
    if TEST_PART in parts or any(p in pp.RESERVED for p in parts):
        raise ValueError("開發階段只用開發零件")
    t0 = time.time()
    out = Path(out)
    tasks = build_tasks(index_dir, parts)
    if not tasks:
        raise Stop("開發零件裡沒有成功、失敗都夠多的步驟")
    log.info("題目 %d 個：%s", len(tasks), "、".join(
        f"{t['name']}（成功 {sum(1 - s['label'] for s in t['instances'])}／失敗 {sum(s['label'] for s in t['instances'])}）"
        for t in tasks))
    write_step_index(tasks, step_index)
    ext = None
    if do_extract:
        ext = extract(tasks, offsets, index_dir=step_index, frames_root=frames_root)
        log.info("影格：新擷取 %d、已有 %d、缺格 %d；錯誤 %d", ext["n_done"], ext["n_skipped"], ext["n_missing_frames"],
                 len(ext["errors"]))
        for e in ext["errors"][:10]:
            log.error("  %s", e)
    own = encs is None
    encs = pp.make_encoders() if own else encs
    results = []
    try:
        pp._check_crop_cache(Path(cache_root), encs["crop"].key)
        for t in tasks:
            feats = {eb: task_features(t, eb, encs["crop"], Path(cache_root), frames_root) for eb in offsets}
            results.append(evaluate_task(t, feats, offsets))
    finally:
        if own:
            for k in ("full", "crop"):
                encs[k].close()
    summary = summarize(results, offsets)
    frozen = write_selected(summary, results, selected_path)
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "offsets": list(offsets), "tasks": results,
           "summary": summary, "selected_file": str(frozen) if frozen else None,
           "extract": {k: v for k, v in (ext or {}).items() if k != "errors"} if ext else None,
           "extract_errors": (ext or {}).get("errors", [])[:20], "seconds": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if sheets:
        for t, r in zip(tasks, results):
            try:
                write_sheet(t, r, summary["offset"], out / f"{_slug(t['name'])}_frames.jpg", frames_root)
            except Exception as e:  # noqa: BLE001  圖只是輔助
                log.warning("%s 的影格拼圖沒畫出來：%s", t["name"], e)
    write_dev_report(res, out / "report.md")
    return res


# ---------------------------------------------------------------- 圖與報告


def write_sheet(task: dict[str, Any], r: dict[str, Any], eb: float, path: Path, frames_root: str = FRAMES_ROOT,
                cols: int = 8, per_class: int = 16, w: int = 200, h: int = 150) -> None:
    """上面：失敗的步驟（最多 16 張）；下面：成功的步驟（前 16 張）。字＝F／S 和 5 次重複裡被判失敗的比例。"""
    import cv2

    rate = dict(zip(r.get("ids", []), (r.get("offsets", {}).get(f"{eb:g}") or {}).get("fail_rate", [])))
    rows = []
    for lab in (1, 0):
        tiles = []
        for s in [s for s in task["instances"] if s["label"] == lab][:per_class]:
            p = frame_path(task["name"], eb, s["segment_id"], frames_root)
            img = cv2.imread(p) if p else None
            if img is None:
                continue
            t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            b = pp.CROP_BOX
            cv2.rectangle(t, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), (46, 122, 224), 2)
            bar = np.full((22, w, 3), 255, np.uint8)
            fr = rate.get(s["segment_id"])
            txt = f"{'FAIL' if lab else 'OK'} {task['step']}" + ("" if fr is None else f"  p_fail {fr:.1f}")
            cv2.putText(bar, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
            tiles.append(np.vstack([t, bar]))
        for i in range(0, len(tiles), cols):
            line = tiles[i:i + cols]
            line += [np.full((h + 22, w, 3), 255, np.uint8)] * (cols - len(line))
            rows.append(np.hstack(line))
        rows.append(np.full((10, cols * w, 3), 200, np.uint8))
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def write_dev_report(res: dict[str, Any], path: Path) -> None:
    s, offs = res["summary"], [f"{eb:g}" for eb in res["offsets"]]
    pick = f"{s['offset']:g}"
    L = ["# 第 19 包（開發階段）：每個步驟自己的小分類器——這一步成功了嗎", "",
         f"產生時間：{res['generated']}；做法：第 14 包凍結的 F2S2，只比「步驟結束後幾秒的那一格」"
         "（事先登記 `docs/package19_stepcls.md`）。bolt 4 這裡沒有碰。", "",
         "## 兩類平均（每題；同一批步驟比三個時間點）", "",
         "| 題目 | 成功／失敗（筆） | " + " | ".join(f"結束後 {o} 秒" for o in offs) + " |",
         "|---|---|" + "---|" * len(offs)]
    for t in res["tasks"]:
        if t.get("skipped"):
            L.append(f"| {t['part']} {zh(t['step'])} | {t['n_success']}／{t['n_failure']} | " + " | ".join("沒做" for _ in offs) + " |")
            continue
        L.append(f"| {t['part']} {zh(t['step'])} | {t['n_success']}／{t['n_failure']} | "
                 + " | ".join(_pct(t["offsets"][o]["bal_mean"]) for o in offs) + " |")
    L += [f"| **{s['n_tasks']} 題平均** | | " + " | ".join(f"**{_pct(s['avg'][o])}**" for o in offs) + " |", "",
          "- 兩類平均＝失敗的步驟答對率和成功的步驟答對率的平均；一律答成功＝50%。",
          f"- 規則：平均最高的是結束後 {s['best_offset']:g} 秒（{_pct(s['best_avg'])}）；差 2 點以內選早的 → "
          f"**選結束後 {pick} 秒（{_pct(s['selected_avg'])}）**",
          f"- 過關門檻 {_pct(s['gate'])}：" + ("**過關**，做法已凍結（`configs/stepcls_selected.yaml`），可以套到 bolt 4。"
                                             if s["gate_passed"] else "**沒過**，先留在開發零件上改做法，不碰 bolt 4。"),
          "- 依步驟種類（選出的時間點）：" + "、".join(f"{zh(k)} {_pct(v)}" for k, v in s["by_step"].items()), "",
          f"## 選出的時間點（結束後 {pick} 秒）每題的細節", "",
          "| 題目 | 兩類平均（5 次最低～最高） | 失敗的步驟答對 | 成功的步驟答對 | 平均用幾個數字 | 錄影數 |", "|---|---|---|---|---|---|"]
    for t in res["tasks"]:
        if t.get("skipped"):
            L.append(f"| {t['part']} {zh(t['step'])} | 沒做：{t['skipped']} | | | | |")
            continue
        o = t["offsets"][pick]
        L.append(f"| {t['part']} {zh(t['step'])} | {_pct(o['bal_mean'])}（{_pct(o['bal_min'])}～{_pct(o['bal_max'])}） | "
                 f"{_pct(o['fail_recall'])} | {_pct(o['ok_recall'])} | {o['k_mean']:.1f} | {t['n_recordings']} |")
    drop = [(t["name"], t["n_dropped"]) for t in res["tasks"] if t.get("n_dropped")]
    L += ["", "## 注意", "",
          "- 失敗的步驟每題只有 7–18 筆，1 筆答對或答錯就差很多；成績只能當參考。",
          "- 圖：每題一張 `<題目>_frames.jpg`（上面是失敗的步驟、下面是成功的步驟；橘框＝模型看的範圍；"
          "p_fail＝5 次重複裡被判失敗的比例）。"]
    if drop:
        L.append("- 沒有影格、不納入的步驟：" + "、".join(f"{n} {k} 筆" for n, k in drop))
    if res.get("extract"):
        e = res["extract"]
        L.append(f"- 影格：新擷取 {e['n_done']}、已有 {e['n_skipped']}、缺格 {e['n_missing_frames']}；"
                 f"擷取錯誤 {len(res.get('extract_errors') or [])} 筆")
    L += [f"- 總共 {res['seconds']:.0f} 秒。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- 測試階段：凍結的做法 → bolt 4


def load_selected(path: Path = SELECTED) -> dict[str, Any]:
    """讀凍結的做法；沒有（開發沒過關）或和現在的程式設定不同就停。"""
    path = Path(path)
    if not path.exists():
        raise Stop(f"找不到 {path}：開發階段要先過關（run_stepcls_dev.bat）")
    body = json.loads("\n".join(x for x in path.read_text(encoding="utf-8").splitlines() if not x.startswith("#")))
    same = (body.get("recipe") == "F2S2" and body.get("grid") == {"k": list(pp.K_GRID), "c1": list(pp.C1_GRID), "c2": list(pp.C2_GRID)}
            and body.get("cv") == {"outer": pp.OUTER, "inner": pp.INNER, "repeats": pp.REPEATS}
            and body.get("min_each") == MIN_EACH and float(body.get("offset_s", -1)) in OFFSETS)
    if not same:
        raise Stop(f"{path} 的設定和現在的程式不一樣，不能用")
    return body


def _perm_one(X: np.ndarray, yp: np.ndarray, groups: list[str]) -> float:
    return float(pp.nested_cv(X, yp, groups, "S2", repeats=1)["bal_mean"])


def permutation_test(X: np.ndarray, y: np.ndarray, groups: list[str], real: float, n_perm: int = N_PERM,
                     seed: int = PERM_SEED, n_jobs: int = 1, progress: Any | None = None, batch: int = 20) -> dict[str, Any]:
    """把成功／失敗的答案打亂 n_perm 次，每次整個流程重跑（1 次重複）；p＝打亂後 ≥ 真的分數的比例（加 1 修正）。
    打亂的順序先全部決定好，平行或一個一個跑結果相同；平行跑不起來就改回一個一個跑。"""
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

                vals = Parallel(n_jobs=n_jobs)(delayed(_perm_one)(X, yp, groups) for yp in chunk)
            except Exception as e:  # noqa: BLE001
                log.warning("平行跑不起來（%s），改成一個一個跑", e)
                parallel = False
        if vals is None:
            vals = [_perm_one(X, yp, groups) for yp in chunk]
        null += [float(v) for v in vals]
        if progress is not None:
            progress(len(null), n_perm)
    ge = int(sum(v >= real - 1e-12 for v in null))
    return {"n_perm": n_perm, "real": float(real), "null_mean": float(np.mean(null)) if null else None,
            "null_p95": float(np.percentile(null, 95)) if null else None, "n_ge": ge,
            "p": float((ge + 1) / (n_perm + 1))}


def evaluate_test_task(task: dict[str, Any], feats: dict[str, np.ndarray], n_perm: int = N_PERM, n_jobs: int = 1,
                       min_each: int = MIN_EACH) -> dict[str, Any]:
    """一題（零件, 步驟）：凍結的 F2S2 巢狀交叉驗證一次＋打亂答案檢定。judged＝每個步驟 5 次重複裡過半數判失敗。"""
    inst = [s for s in task["instances"] if s["segment_id"] in feats]
    y = np.array([s["label"] for s in inst], int)
    groups = [s["recording_id"] for s in inst]
    res: dict[str, Any] = {"name": task["name"], "part": task["part"], "step": task["step"], "n": len(inst),
                           "n_failure": int(y.sum()), "n_success": int(len(y) - y.sum()),
                           "n_dropped": len(task["instances"]) - len(inst), "n_recordings": len(set(groups)),
                           "ids": [s["segment_id"] for s in inst], "labels": y.tolist()}
    if res["n_failure"] < min_each or res["n_success"] < min_each:
        res["skipped"] = f"去掉沒有影格的步驟後，成功或失敗不到 {min_each} 筆"
        return res
    X = np.vstack([feats[s["segment_id"]] for s in inst])
    r = pp.nested_cv(X, y, groups, "S2")
    fr = np.asarray(r["fail_rate"], float)
    judged = fr > 0.5
    res.update(bal_mean=r["bal_mean"], bal_min=r["bal_min"], bal_max=r["bal_max"],
               fail_recall=float(np.mean([p["ok_failure"] / p["n_failure"] for p in r["repeats"]])),
               ok_recall=float(np.mean([p["ok_success"] / p["n_success"] for p in r["repeats"]])),
               k_mean=r["k_mean"], chosen=dict(list(r["chosen"].items())[:3]), fail_rate=fr.tolist(),
               judged=judged.astype(int).tolist(),
               vote={"ok_failure": int(np.sum(judged & (y == 1))), "ok_success": int(np.sum(~judged & (y == 0)))})

    def progress(i: int, n: int) -> None:
        log.info("  %s 打亂答案 %d/%d", task["name"], i, n)

    res["perm"] = (permutation_test(X, y, groups, r["repeats"][0]["bal"], n_perm, n_jobs=n_jobs, progress=progress)
                   if n_perm > 0 else None)
    return res


def task_verdict(tr: dict[str, Any]) -> str:
    if tr.get("skipped"):
        return "沒做"
    if not tr.get("perm"):
        return "沒有跑打亂答案檢定，不下結論"
    ok = tr["bal_mean"] >= PASS_BAL and tr["perm"]["p"] < PASS_P
    return ("達到事先的判定" if ok else "沒有達到事先的判定") + f"（兩類平均 ≥ {PASS_BAL * 100:.0f}% 而且 p < {PASS_P}）"


def which_step(instances: list[dict[str, Any]], task_results: list[dict[str, Any]]) -> dict[str, Any]:
    """每一段依步驟順序看有分類器的步驟，第一個被判失敗的就是「失敗在哪一步」的答案；都沒有＝None（沒判到）。
    失敗段和資料集第一個失敗的步驟比；成功段看是不是每一步都沒被判失敗。"""
    judged: dict[str, bool] = {}
    steps_with_model = []
    for tr in task_results:
        if tr.get("skipped"):
            continue
        steps_with_model.append(tr["step"])
        judged.update({sid: bool(j) for sid, j in zip(tr["ids"], tr["judged"])})
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in instances:
        by_parent[s["parent"]].append(s)
    rows = []
    for parent, ss in sorted(by_parent.items()):
        ss.sort(key=lambda s: s["order"])
        true = next((s["step"] for s in ss if s["label"] == 1), None)
        pred = next((s["step"] for s in ss if judged.get(s["segment_id"], False)), None)
        rows.append({"segment_id": parent, "failed": bool(ss[0]["parent_failed"]), "true": true, "pred": pred,
                     "sequence": " > ".join(s["step"] + ("✗" if s["label"] else "") for s in ss),
                     "judged": " > ".join(s["step"] + ("✗" if judged.get(s["segment_id"], False) else "")
                                          for s in ss if s["step"] in steps_with_model)})
    fails = [r for r in rows if r["failed"]]
    oks = [r for r in rows if not r["failed"]]
    by_true: dict[str, dict[str, int]] = {}
    for r in fails:
        d = by_true.setdefault(r["true"] or "沒有失敗步驟", {"n": 0, "right": 0})
        d["n"] += 1
        d["right"] += int(r["true"] is not None and r["pred"] == r["true"])
    confusion: dict[str, dict[str, int]] = {}
    for r in fails:
        c = confusion.setdefault(r["true"] or "沒有失敗步驟", {})
        c[r["pred"] or "沒判到"] = c.get(r["pred"] or "沒判到", 0) + 1
    n_right = sum(d["right"] for d in by_true.values())
    most = max((d["n"] for k, d in by_true.items() if k != "沒有失敗步驟"), default=0)
    return {"steps_with_model": steps_with_model, "rows": rows, "n_failed": len(fails), "n_right": n_right,
            "by_true": by_true, "confusion": confusion, "always_most": most, "n_ok": len(oks),
            "ok_clean": int(sum(r["pred"] is None for r in oks)),
            "failed_flagged": int(sum(r["pred"] is not None for r in fails))}


def run_test(index_dir: str = "data/index", out: Path = OUT_TEST, part: str = TEST_PART, cache_root: Path = CACHE_ROOT,
             frames_root: str = FRAMES_ROOT, step_index: Path = STEP_INDEX, selected_path: Path = SELECTED,
             n_perm: int = N_PERM, n_jobs: int = -2, encs: dict[str, Any] | None = None, do_extract: bool = True,
             sheets: bool = True, force: bool = False) -> dict[str, Any]:
    if part in pp.RESERVED:
        raise ValueError(f"{part} 是保留零件，不碰")
    out = Path(out)
    if (out / "result.json").exists() and not force:
        raise Stop(f"{out / 'result.json'} 已經有結果：凍結的做法只套一次，不重跑")
    sel = load_selected(selected_path)
    eb = float(sel["offset_s"])
    t0 = time.time()
    tasks = build_tasks(index_dir, (part,))
    if not tasks:
        raise Stop(f"{part} 沒有成功、失敗都夠多的步驟")
    log.info("凍結的時間點：步驟結束後 %g 秒；題目：%s", eb, "、".join(t["name"] for t in tasks))
    write_step_index(tasks, step_index)
    ext = None
    if do_extract:
        ext = extract(tasks, (eb,), index_dir=step_index, frames_root=frames_root)
        log.info("影格：新擷取 %d、已有 %d、缺格 %d；錯誤 %d", ext["n_done"], ext["n_skipped"], ext["n_missing_frames"],
                 len(ext["errors"]))
        for e in ext["errors"][:10]:
            log.error("  %s", e)
    own = encs is None
    encs = pp.make_encoders() if own else encs
    results = []
    try:
        pp._check_crop_cache(Path(cache_root), encs["crop"].key)
        feats = [task_features(t, eb, encs["crop"], Path(cache_root), frames_root) for t in tasks]
    finally:
        if own:
            for k in ("full", "crop"):
                encs[k].close()
    for t, f in zip(tasks, feats):
        r = evaluate_test_task(t, f, n_perm, n_jobs)
        results.append(r)
        if not r.get("skipped"):
            log.info("%s：兩類平均 %.0f%%（%.0f%%～%.0f%%）；%s", t["name"], r["bal_mean"] * 100, r["bal_min"] * 100,
                     r["bal_max"] * 100, task_verdict(r))
    ws = which_step(step_instances(index_dir, part), results)
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "part": part, "offset_s": eb, "frozen_at": sel.get("frozen_at"),
           "dev_avg": sel.get("dev_avg"), "tasks": results, "which_step": ws, "n_perm": n_perm,
           "extract": {k: v for k, v in (ext or {}).items() if k != "errors"} if ext else None,
           "extract_errors": (ext or {}).get("errors", [])[:20], "seconds": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if sheets:
        for t, r in zip(tasks, results):
            try:
                rr = dict(r, offsets={f"{eb:g}": {"fail_rate": r.get("fail_rate", [])}})
                write_sheet(t, rr, eb, out / f"{_slug(t['name'])}_frames.jpg", frames_root)
            except Exception as e:  # noqa: BLE001
                log.warning("%s 的影格拼圖沒畫出來：%s", t["name"], e)
    write_test_report(res, out / "report.md")
    return res


def write_test_report(res: dict[str, Any], path: Path) -> None:
    part, eb, ws = res["part"], res["offset_s"], res["which_step"]
    L = [f"# 第 19 包（測試階段）：{part} 每個步驟自己的小分類器", "",
         f"產生時間：{res['generated']}；凍結的做法（{res.get('frozen_at')}）：F2S2、步驟結束後 {eb:g} 秒的手腕影格，"
         "原封不動套到 bolt 4 一次（事先登記 `docs/package19_stepcls.md`）。", "",
         "## 這一步成功了嗎（每題）", "",
         "| 題目 | 成功／失敗（筆） | 兩類平均（5 次最低～最高） | 失敗的步驟答對 | 成功的步驟答對 | 打亂答案 p | 判定 |",
         "|---|---|---|---|---|---|---|"]
    for t in res["tasks"]:
        if t.get("skipped"):
            L.append(f"| {t['part']} {zh(t['step'])} | {t['n_success']}／{t['n_failure']} | 沒做：{t['skipped']} | | | | |")
            continue
        pm = t.get("perm")
        L.append(f"| {t['part']} {zh(t['step'])} | {t['n_success']}／{t['n_failure']} | {_pct(t['bal_mean'])}"
                 f"（{_pct(t['bal_min'])}～{_pct(t['bal_max'])}） | {_pct(t['fail_recall'])} | {_pct(t['ok_recall'])} | "
                 f"{'–' if not pm else format(pm['p'], '.3f')} | {task_verdict(t)} |")
    L += ["", "- 兩類平均＝失敗的步驟答對率和成功的步驟答對率的平均；一律答成功＝50%。"]
    for t in res["tasks"]:
        if t.get("skipped"):
            continue
        pm = t.get("perm")
        L.append(f"- {t['part']} {zh(t['step'])}：過半數判斷下，失敗的步驟答對 {t['vote']['ok_failure']}/{t['n_failure']}、"
                 f"成功的步驟答對 {t['vote']['ok_success']}/{t['n_success']}；平均用 {t['k_mean']:.1f} 個數字"
                 + (f"；打亂答案 {pm['n_perm']} 次：真的（第 1 次重複）{_pct(pm['real'])}、打亂後平均 {_pct(pm['null_mean'])}、"
                    f"≥ 真的 {pm['n_ge']} 次" if pm else "")
                 + (f"；沒有影格不納入 {t['n_dropped']} 筆" if t.get("n_dropped") else ""))
    if res.get("dev_avg"):
        L.append("- 參考：開發 5 題平均 " + "、".join(f"結束後 {k} 秒 {_pct(v)}" for k, v in res["dev_avg"].items()))
    order = [s for s in ("Approach", "Align", "Twist", "Push", "Release") if s in ws["by_true"]]
    order += [s for s in ws["by_true"] if s not in order]
    preds = [s for s in ("Approach", "Align", "Twist", "Push", "Release") if any(s in c for c in ws["confusion"].values())]
    preds += ["沒判到"]
    L += ["", f"## 失敗在哪一步（{ws['n_failed']} 段失敗段）", "",
          "- 算法：每一段依步驟順序看" + "、".join(zh(s) for s in ws["steps_with_model"])
          + "，第一個被判失敗的步驟就是答案；都沒被判失敗＝沒判到。",
          f"- **答對 {ws['n_right']}/{ws['n_failed']} 段**（對照：一律答最多的那一步 {ws['always_most']}/{ws['n_failed']}"
          + ("；第 18 包只看整段結尾 11/30）" if part == TEST_PART else "）"),
          f"- 失敗段裡至少有一步被判失敗：{ws['failed_flagged']}/{ws['n_failed']} 段；"
          f"成功段每一步都沒被判失敗：{ws['ok_clean']}/{ws['n_ok']} 段", "",
          "| 資料集第一個失敗的步驟 | 段數 | 答對 | " + " | ".join(("沒判到" if p_ == "沒判到" else f"模型答{zh(p_)}") for p_ in preds) + " |",
          "|---|---|---|" + "---|" * len(preds)]
    for s in order:
        d, c = ws["by_true"][s], ws["confusion"].get(s, {})
        note = "（沒有分類器）" if s not in ws["steps_with_model"] and s != "沒有失敗步驟" else ""
        L.append(f"| {zh(s)}{note} | {d['n']} | {d['right']}/{d['n']} | " + " | ".join(str(c.get(p_, 0)) for p_ in preds) + " |")
    L += ["", "### 每段（失敗段）", "", "| 段 | 資料集的步驟（✗＝失敗） | 模型的判斷（✗＝判失敗） | 資料集第一個失敗 | 模型的答案 |", "|---|---|---|---|---|"]
    for r in sorted((r for r in ws["rows"] if r["failed"]), key=lambda r: (str(r["true"]), r["segment_id"])):
        L.append(f"| {r['segment_id']} | {r['sequence']} | {r['judged']} | {zh(r['true']) if r['true'] else '–'} | "
                 f"{zh(r['pred']) if r['pred'] else '沒判到'} |")
    bad_ok = [r for r in ws["rows"] if not r["failed"] and r["pred"] is not None]
    if bad_ok:
        L += ["", "### 成功段裡有步驟被判失敗的段", "", "| 段 | 模型的判斷（✗＝判失敗） |", "|---|---|"]
        L += [f"| {r['segment_id']} | {r['judged']} |" for r in bad_ok]
    L += ["", "## 注意", "",
          "- 步驟結束後 1 秒，成功的步驟多半已經放開（零件留在板上），失敗的步驟零件還在手上；分類器主要看的是這個。",
          "- 接近這一步失敗只有 6 筆，沒有分類器；那幾段在「失敗在哪一步」一定算沒答對。",
          "- 每題的判斷都是交叉驗證（判某個步驟時，同一支錄影的步驟沒有拿來訓練）。",
          "- 圖：每題一張 `<題目>_frames.jpg`（上面是失敗的步驟、下面是成功的步驟；p_fail＝5 次重複裡被判失敗的比例）。"]
    if res.get("extract"):
        e = res["extract"]
        L.append(f"- 影格：新擷取 {e['n_done']}、已有 {e['n_skipped']}、缺格 {e['n_missing_frames']}；"
                 f"擷取錯誤 {len(res.get('extract_errors') or [])} 筆")
    L += [f"- 總共 {res['seconds']:.0f} 秒。", ""]
    path.write_text("\n".join(L), encoding="utf-8")
