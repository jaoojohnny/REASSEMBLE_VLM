"""第 4 包：標註任務（規格 P3-3）——選段、密集影格、離線 HTML 標註頁、Claude 用的檢視圖、匯出檔收集。

    data/derived/annotation/<任務>/
        index.html            標註頁（雙擊開啟；?mode=adjudicate＝裁決模式）
        adjudicate.html       轉到裁決模式
        data.js               段落、時間軸、低階技能與設定（標註頁讀這份）
        frames/<段>/<相機>__0000.jpg   每秒 fps 張的密集影格（依各相機時間戳挑最近幀）
        review/<段>__overview.jpg、<段>__<相機>.jpg   給 Claude 獨立標註用的檢視圖
        selection.csv、task.json
    data/annotations/<任務>/
        human__<標註者>.jsonl、ai__claude.jsonl、reference__dataset_lowlevel.jsonl、adjudication__<標註者>.jsonl
        gold.jsonl / gold.csv、inbox/（收進來的原始匯出檔）、reports/（比較報告）
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import logging
import os
import random
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .agreement import compare_sources, load_skills, merge_gold, reference_from_lowlevel, skill_sequence
from .extract import Sampling, _contact_sheet, _put, _write_img, fingerprint, pick_frames, target_times
from .schema import (AnnotationConfig, SegmentInfo, merge_annotations, normalize_record, read_annotations,
                     validate_records, write_csv, write_jsonl)
from .util import now_iso

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"
EXPORT_RE = re.compile(r"^(?P<task>[A-Za-z0-9-]+(?:_[A-Za-z0-9-]+)*)__(?:(?P<adj>adjudication)__)?(?P<annot>.+?)__(?P<ts>\d{8}-\d{6})\.(?P<ext>jsonl|csv)$")


# ---------------------------------------------------------------- 選段


def select_task_segments(segs: dict[str, SegmentInfo], selection_csv: str | Path | None, n_fail: int, n_ok: int,
                         seed: int = 0) -> list[SegmentInfo]:
    """從第 3 包已核對的 pilot 清單（selection_pilot.csv）取：失敗全收（最多 n_fail）、成功依 seed 抽 n_ok 段。"""
    pool: list[SegmentInfo] = []
    if selection_csv and Path(selection_csv).exists():
        with open(selection_csv, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                key = f"{r['recording_id']}__seg{int(r['segment_id'])}"
                if key in segs:
                    pool.append(segs[key])
    else:
        pool = list(segs.values())
    pool.sort(key=lambda s: (s.recording_id, s.segment_index))
    rng = random.Random(seed)
    fails = [s for s in pool if s.dataset_result == "failure"]
    oks = [s for s in pool if s.dataset_result == "success"]
    if len(fails) > n_fail:
        fails = sorted(rng.sample(fails, n_fail), key=lambda s: (s.recording_id, s.segment_index))
    oks = sorted(rng.sample(oks, min(n_ok, len(oks))), key=lambda s: (s.recording_id, s.segment_index))
    return fails + oks


def rest_segments(segs: dict[str, SegmentInfo], exclude_task_dirs: list[str | Path]) -> list[SegmentInfo]:
    """其餘的段：全部有效段扣掉已經有標註任務的段（例如 pilot30），依錄影與段序排列。"""
    taken: set[str] = set()
    for d in exclude_task_dirs:
        p = Path(d) / "task.json"
        if not p.exists():
            raise FileNotFoundError(f"找不到 {p}：要排除的任務還沒建立")
        taken |= set(json.loads(p.read_text(encoding="utf-8"))["segments"])
    return sorted((s for k, s in segs.items() if k not in taken), key=lambda s: (s.recording_id, s.segment_index))


def dense_targets(seg: SegmentInfo, sb: float, eb: float, fps: float) -> list[float]:
    w0, w1 = seg.start_unix - sb, seg.end_unix + eb
    n = int(np.floor((w1 - w0) * fps + 1e-9))
    ts = [w0 + i / fps for i in range(n + 1)]
    if w1 - ts[-1] > 1e-6:
        ts.append(w1)
    return ts


def _task_key(seg: SegmentInfo, settings: dict, cams: list[dict], fp: str) -> str:
    payload = json.dumps({"seg": [seg.segment_id, seg.start_unix, seg.end_unix], "settings": settings,
                          "cams": [c["name"] for c in cams], "src": fp, "tool": "annotate-v1"}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _resize(img: Any, max_w: int) -> Any:
    import cv2

    h, w = img.shape[:2]
    if max_w and w > max_w:
        img = cv2.resize(img, (max_w, int(round(h * max_w / w))), interpolation=cv2.INTER_AREA)
    return img


def _detail_sheet(seg: SegmentInfo, cam: str, items: list[tuple[float, Any]], skills: list[dict]) -> Any:
    """單一相機 12 格（4 欄），格內標相對秒數與當下的低階技能。"""
    import cv2

    tw, th, cols = 320, 240, 4
    rows = (len(items) + cols - 1) // cols
    sheet = np.zeros((44 + rows * th, cols * tw, 3), np.uint8)
    lab = {"failure": "FAIL", "success": "OK"}.get(seg.dataset_result or "", "NA")
    _put(sheet, f"{seg.segment_id}  {seg.text}  label={lab}  len={seg.duration:.2f}s  camera={cam}", (8, 18), 0.5)
    _put(sheet, "skills: " + " > ".join(f"{s['text']}[{s['start_rel']:+.1f},{s['end_rel']:+.1f}]{'X' if s['success'] is False else ''}"
                                         for s in skills), (8, 38), 0.42, (200, 200, 200))
    for i, (t_rel, img) in enumerate(items):
        y0, x0 = 44 + (i // cols) * th, (i % cols) * tw
        if img is None:
            cv2.rectangle(sheet, (x0, y0), (x0 + tw - 1, y0 + th - 1), (0, 0, 120), -1)
            _put(sheet, "MISSING", (x0 + 8, y0 + 40), 0.55)
            continue
        sheet[y0 : y0 + th, x0 : x0 + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        phase = next((s["text"] + ("X" if s["success"] is False else "") for s in skills if s["start_rel"] <= t_rel <= s["end_rel"]),
                     "before" if t_rel < 0 else ("after" if t_rel > seg.duration else "-"))
        _put(sheet, f"#{i} t={t_rel:+.2f}s {phase}", (x0 + 6, y0 + 18), 0.48)
    return sheet


def build_recording(job: dict) -> dict:
    """一個錄製檔的所有任務段：密集影格、frames.json、檢視圖。可在子行程執行。"""
    import h5py

    from .check_clips import _CamReader

    slim = Path(job["slim_path"])
    rid = job["recording_id"]
    cams: list[dict] = job["cams"]
    task_dir = Path(job["task_dir"])
    st = job["settings"]
    force = bool(job.get("force"))
    summary: dict[str, Any] = {"recording_id": rid, "done": 0, "skipped": 0, "frames": 0, "missing": 0, "errors": [], "segments": {}}
    if not slim.exists():
        summary["errors"].append(f"找不到精簡版 {slim}")
        return summary
    fp = fingerprint(slim)
    segs = [SegmentInfo(**d) for d in job["segments"]]
    todo = []
    for seg in segs:
        key = _task_key(seg, st, cams, fp)
        meta_p = task_dir / "frames" / seg.segment_id / "frames.json"
        if not force and meta_p.exists():
            try:
                m = json.loads(meta_p.read_text(encoding="utf-8"))
                if m.get("cache_key") == key:
                    summary["skipped"] += 1
                    summary["segments"][seg.segment_id] = m
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        todo.append((seg, key))
    if not todo:
        return summary
    sb, eb, fps = st["start_buffer"], st["end_buffer"], st["fps"]
    ov = Sampling(n_frames=int(st["overview_n"]), start_buffer=sb, end_buffer=eb, fmt="jpg", jpeg_quality=90)
    with h5py.File(os.fspath(slim), "r") as f:
        ts = {c["name"]: (np.asarray(f[c["timestamps"]][()], dtype=np.float64).ravel() if c["timestamps"] in f and c["video"] in f else None)
              for c in cams}
        plans = []
        need: dict[str, set[int]] = {c["name"]: set() for c in cams}
        for seg, key in todo:
            w0, w1 = seg.start_unix - sb, seg.end_unix + eb
            dense = dense_targets(seg, sb, eb, fps)
            plan: dict[str, Any] = {"seg": seg, "key": key, "dense": {}, "overview": {}, "detail": []}
            for c in cams:
                t = ts[c["name"]]
                if t is None:
                    plan["dense"][c["name"]] = []
                    plan["overview"][c["name"]] = [{"target_unix": x, "frame_idx": None, "source_unix": None, "missing": "camera_missing"}
                                                   for x in target_times(seg.start_unix, seg.end_unix, ov)]
                    continue
                picks = [p for p in pick_frames(t, dense, w0, w1) if p["frame_idx"] is not None]
                uniq, last = [], None
                for p in picks:
                    if p["frame_idx"] != last:
                        uniq.append(p)
                        last = p["frame_idx"]
                plan["dense"][c["name"]] = uniq
                need[c["name"]].update(p["frame_idx"] for p in uniq)
                o = pick_frames(t, target_times(seg.start_unix, seg.end_unix, ov), w0, w1)
                plan["overview"][c["name"]] = o
                need[c["name"]].update(p["frame_idx"] for p in o if p["frame_idx"] is not None)
            dc = st["detail_camera"]
            if ts.get(dc) is not None:
                n = int(st["detail_n"])
                det_t = [w0 + (w1 - w0) * i / (n - 1) for i in range(n)] if n > 1 else [w1]
                plan["detail"] = pick_frames(ts[dc], det_t, w0, w1)
                need[dc].update(p["frame_idx"] for p in plan["detail"] if p["frame_idx"] is not None)
            plans.append(plan)
        # 密集影格解碼後立刻寫檔（不全部留在記憶體）；只保留檢視圖需要的幀
        dense_s = Sampling(fmt="jpg", jpeg_quality=int(st["jpeg_quality"]))
        keep: dict[str, set[int]] = {c["name"]: set() for c in cams}
        for plan in plans:
            for cname, es in plan["overview"].items():
                keep[cname].update(e["frame_idx"] for e in es if e["frame_idx"] is not None)
            keep[st["detail_camera"]].update(e["frame_idx"] for e in plan["detail"] if e["frame_idx"] is not None)
            d = task_dir / "frames" / plan["seg"].segment_id
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*.jpg"):
                old.unlink()  # 設定改變時清掉舊影格，避免殘留
            plan["out"] = {c["name"]: {"t": [], "frame_idx": []} for c in cams}
        images: dict[tuple[str, int], Any] = {}
        for c in cams:
            cname = c["name"]
            users: dict[int, list[tuple[dict, dict]]] = {}
            for plan in plans:
                for p in plan["dense"][cname]:
                    users.setdefault(p["frame_idx"], []).append((plan, p))
            idxs = sorted(need[cname])
            if not idxs:
                continue
            try:
                reader = _CamReader(f, {"name": cname, "path": c["video"], "timestamps": c["timestamps"]}, os.fspath(slim), 64 << 20)
            except Exception as e:  # noqa: BLE001
                summary["errors"].append(f"{cname}: {e}")
                continue
            try:
                for i in idxs:
                    img = reader.read_index(i)
                    if img is None:
                        summary["missing"] += len(users.get(i, []))
                        continue
                    if i in keep[cname]:
                        images[(cname, i)] = img
                    if i in users:
                        small = _resize(img, int(st["max_width"]))
                        for plan, p in users[i]:
                            out = plan["out"][cname]
                            k = len(out["t"])
                            _write_img(task_dir / "frames" / plan["seg"].segment_id / f"{cname}__{k:04d}.jpg", small, dense_s)
                            out["t"].append(round(p["source_unix"] - plan["seg"].start_unix, 3))
                            out["frame_idx"].append(int(i))
            finally:
                reader.close()
    for plan in plans:
        seg: SegmentInfo = plan["seg"]
        d = task_dir / "frames" / seg.segment_id
        cam_meta = plan["out"]
        summary["frames"] += sum(len(v["t"]) for v in cam_meta.values())
        rv = task_dir / "review"
        rv.mkdir(parents=True, exist_ok=True)
        seg_d = {"recording_id": seg.recording_id, "segment_id": seg.segment_index, "text": seg.text,
                 "success": {"success": True, "failure": False}.get(seg.dataset_result or ""), "start": seg.start_unix, "end": seg.end_unix}
        sheet = _contact_sheet(seg_d, ov, cams, plan["overview"], images)
        _write_img(rv / f"{seg.segment_id}__overview.jpg", sheet, ov)
        if plan["detail"]:
            skills = [{**s, "start_rel": s["start_unix"] - seg.start_unix, "end_rel": s["end_unix"] - seg.start_unix}
                      for s in job["skills"].get(seg.segment_id, [])]
            items = [((p["source_unix"] - seg.start_unix) if p["frame_idx"] is not None else (p["target_unix"] - seg.start_unix),
                      images.get((st["detail_camera"], p["frame_idx"])) if p["frame_idx"] is not None else None) for p in plan["detail"]]
            _write_img(rv / f"{seg.segment_id}__{st['detail_camera']}.jpg", _detail_sheet(seg, st["detail_camera"], items, skills), ov)
        meta = {"segment_id": seg.segment_id, "cache_key": plan["key"], "cams": cam_meta, "created_at": now_iso(),
                "source": {"slim": slim.name, "fingerprint": fp}, "settings": st}
        tmp = d / "frames.json.part"
        tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, d / "frames.json")
        summary["segments"][seg.segment_id] = meta
        summary["done"] += 1
    return summary


def write_page(task_dir: Path, data: dict[str, Any]) -> dict[str, str]:
    (task_dir / "data.js").write_text("window.ANNOT = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    shutil.copyfile(WEB_DIR / "annotate.html", task_dir / "index.html")
    (task_dir / "adjudicate.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>裁決模式</title>'
        '<script>location.replace("index.html?mode=adjudicate")</script><a href="index.html?mode=adjudicate">開啟裁決模式</a>',
        encoding="utf-8")
    return {"index": str(task_dir / "index.html"), "data": str(task_dir / "data.js")}


def refresh_page(task_dir: str | Path, cfg: AnnotationConfig) -> dict[str, str]:
    """改了 annotation.yaml（類別、畫面難度、提示）後：只重寫 data.js 的設定與 index.html，不重新解碼影片。"""
    td = Path(task_dir)
    p = td / "data.js"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，請先執行 annotate build")
    text = p.read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    data["config"] = cfg.public()
    data["refreshed_at"] = now_iso()
    return write_page(td, data)


def build_task(*, mapping: Any, cfg: AnnotationConfig, segs: dict[str, SegmentInfo], chosen: list[SegmentInfo],
               index_dir: str | Path, slim_dir: str | Path, out_root: str | Path, task: str, title: str,
               workers: int = 4, force: bool = False, fps: float | None = None) -> dict[str, Any]:
    cams = [{"name": c, "video": mapping.cameras[c]["video"], "timestamps": mapping.cameras[c]["timestamps"]} for c in cfg.cameras]
    settings = {"start_buffer": cfg.start_buffer, "end_buffer": cfg.end_buffer, "fps": float(fps or cfg.dense.get("fps", 5)),
                "max_width": int(cfg.dense.get("max_width", 640)), "jpeg_quality": int(cfg.dense.get("jpeg_quality", 82)),
                "overview_n": int(cfg.review.get("overview_n", 8)), "detail_camera": cfg.review.get("detail_camera", cfg.cameras[-1]),
                "detail_n": int(cfg.review.get("detail_n", 12))}
    if settings["detail_camera"] not in cfg.cameras:
        settings["detail_camera"] = cfg.cameras[-1]
    task_dir = Path(out_root) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    skills = load_skills(index_dir)
    by_rec: dict[str, list[SegmentInfo]] = {}
    for s in chosen:
        by_rec.setdefault(s.recording_id, []).append(s)
    jobs = [{"slim_path": str(Path(slim_dir) / f"{rid}.h5"), "recording_id": rid, "cams": cams, "task_dir": str(task_dir),
             "segments": [asdict(s) for s in ss], "settings": settings, "force": force,
             "skills": {s.segment_id: skills.get(s.segment_id, []) for s in ss}} for rid, ss in sorted(by_rec.items())]
    log.info("標註任務 %s：%d 段（失敗 %d / 成功 %d），%d 個錄製檔，%g fps", task, len(chosen),
             sum(1 for s in chosen if s.dataset_result == "failure"), sum(1 for s in chosen if s.dataset_result == "success"),
             len(jobs), settings["fps"])
    results: list[dict] = []
    if workers > 1 and len(jobs) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                futs = [pool.submit(build_recording, j) for j in jobs]
                for k, fut in enumerate(as_completed(futs), 1):
                    r = fut.result()
                    results.append(r)
                    log.info("  [%d/%d] %s：新做 %d、沿用 %d、影格 %d%s", k, len(jobs), r["recording_id"], r["done"], r["skipped"],
                             r["frames"], f"、錯誤 {r['errors']}" if r["errors"] else "")
        except Exception as e:  # noqa: BLE001
            log.warning("平行處理失敗（%s），改為逐一處理", e)
            done = {r["recording_id"] for r in results}
            jobs = [j for j in jobs if j["recording_id"] not in done]
            workers = 1
    if workers <= 1 or len(jobs) <= 1:
        for k, j in enumerate(jobs, 1):
            r = build_recording(j)
            results.append(r)
            log.info("  [%d/%d] %s：新做 %d、沿用 %d、影格 %d%s", k, len(jobs), r["recording_id"], r["done"], r["skipped"],
                     r["frames"], f"、錯誤 {r['errors']}" if r["errors"] else "")
    metas = {sid: m for r in results for sid, m in r["segments"].items()}
    seg_items = []
    for n, s in enumerate(chosen, 1):
        m = metas.get(s.segment_id) or {}
        sk = skills.get(s.segment_id, [])
        seg_items.append({
            "id": s.segment_id, "n": n, "recording_id": s.recording_id, "segment_index": s.segment_index, "text": s.text,
            "dataset_result": s.dataset_result, "duration": round(s.duration, 3),
            "window": [round(-cfg.start_buffer, 3), round(s.duration + cfg.end_buffer, 3)],
            "skills": [{"text": k["text"], "start": round(k["start_unix"] - s.start_unix, 3), "end": round(k["end_unix"] - s.start_unix, 3),
                        "success": k["success"]} for k in sk],
            "cams": {c: {"t": (m.get("cams", {}).get(c) or {}).get("t", [])} for c in cfg.cameras},
        })
    data = {"task": task, "title": title, "action": segs[chosen[0].segment_id].action if chosen else "", "part": chosen[0].part if chosen else "",
            "created_at": now_iso(), "fps": settings["fps"], "show_skills": True, "config": cfg.public(), "segments": seg_items}
    pages = write_page(task_dir, data)
    with (task_dir / "selection.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "segment_id", "recording_id", "segment_index", "dataset_result", "duration_s", "lowlevel"])
        w.writeheader()
        for it, s in zip(seg_items, chosen):
            w.writerow({"n": it["n"], "segment_id": s.segment_id, "recording_id": s.recording_id, "segment_index": s.segment_index,
                        "dataset_result": s.dataset_result, "duration_s": round(s.duration, 3),
                        "lowlevel": skill_sequence(skills.get(s.segment_id, []))})
    task_meta = {"task": task, "title": title, "created_at": now_iso(), "tool": f"assembly_monitor.annotate {__version__}",
                 "settings": settings, "n_segments": len(chosen), "n_frames": sum(r["frames"] for r in results),
                 "n_missing": sum(r["missing"] for r in results), "errors": [e for r in results for e in r["errors"]],
                 "segments": [s.segment_id for s in chosen]}
    (task_dir / "task.json").write_text(json.dumps(task_meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"task_dir": str(task_dir), **pages, "results": results, "n_done": sum(r["done"] for r in results),
            "n_skipped": sum(r["skipped"] for r in results), "n_frames": task_meta["n_frames"], "errors": task_meta["errors"]}


# ---------------------------------------------------------------- 收檔、參考標籤、比較、合併


def task_segments(task_dir: str | Path, segs: dict[str, SegmentInfo]) -> dict[str, SegmentInfo]:
    p = Path(task_dir) / "task.json"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}：請先執行 annotate build")
    ids = json.loads(p.read_text(encoding="utf-8"))["segments"]
    return {i: segs[i] for i in ids if i in segs}


def collect_exports(src_dirs: list[str | Path], task: str, ann_dir: str | Path, cfg: AnnotationConfig,
                    segs: dict[str, SegmentInfo]) -> dict[str, Any]:
    """把標註頁匯出的檔（<任務>__<標註者>__<時間>.jsonl / csv）收進 data/annotations/<任務>/，依標註者合併成一份。"""
    ann = Path(ann_dir)
    inbox = ann / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    found: list[tuple[str, Path, re.Match]] = []
    for d in src_dirs:
        dp = Path(d).expanduser()
        if not dp.is_dir():
            continue
        for p in dp.iterdir():
            m = EXPORT_RE.match(p.name)
            if p.is_file() and m and m.group("task") == task:
                found.append((m.group("ts"), p, m))
    found.sort(key=lambda x: (x[0], x[1].name))
    report: dict[str, Any] = {"collected": [], "problems": []}
    for ts, p, m in found:
        dst = inbox / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        kind = "adjudication" if m.group("adj") else "human"
        annot = re.sub(r"[^\w一-鿿-]+", "_", m.group("annot"))
        target = ann / f"{kind}__{annot}.jsonl"
        try:
            incoming = read_annotations(dst, cfg)
        except Exception as e:  # noqa: BLE001
            report["problems"].append({"file": p.name, "error": str(e)})
            continue
        for r in incoming:
            r.setdefault("annotator_type", "human")
        base = [normalize_record(r, cfg) for r in read_annotations(target, cfg)] if target.exists() else []
        merged, rep = merge_annotations(base, incoming, cfg, segs)
        if rep["conflicts"]:
            report["problems"].append({"file": p.name, "error": f"原始標籤衝突 {rep['conflicts'][:3]}"})
        write_jsonl(merged, target, cfg)
        v = validate_records(merged, cfg, segs)
        report["collected"].append({"file": p.name, "target": target.name, "kind": kind, "added": rep["added"],
                                    "updated": rep["updated"], "n": len(merged), "problems": v["problems"][:20]})
    return report


def write_reference(task_segs: dict[str, SegmentInfo], index_dir: str | Path, ann_dir: str | Path, cfg: AnnotationConfig) -> Path:
    recs = reference_from_lowlevel(task_segs, load_skills(index_dir), cfg)
    return write_jsonl(recs, Path(ann_dir) / f"reference__{cfg.reference.get('annotator', 'dataset_lowlevel')}.jsonl", cfg)


def _load(p: Path, cfg: AnnotationConfig) -> list[dict[str, Any]]:
    return [normalize_record(r, cfg) for r in read_annotations(p, cfg)]


def discover_sources(ann_dir: str | Path, cfg: AnnotationConfig, primary: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """主標註者（human）＝A，其次是第二位人類或 AI（B），最後是資料集參考（C）。"""
    ann = Path(ann_dir)
    humans = sorted(ann.glob("human__*.jsonl"))
    if primary:
        humans.sort(key=lambda p: p.stem != f"human__{primary}")
    ais = sorted(ann.glob("ai__*.jsonl"))
    refs = sorted(ann.glob("reference__*.jsonl"))
    files = humans + ais + refs
    labels = "ABCDEFGH"
    return {labels[i]: _load(p, cfg) for i, p in enumerate(files[: len(labels)])}


def run_compare(ann_dir: str | Path, task_dir: str | Path, cfg: AnnotationConfig, task: str,
                primary: str | None = None) -> dict[str, Any]:
    sources = discover_sources(ann_dir, cfg, primary)
    if len(sources) < 2:
        raise FileNotFoundError(f"{ann_dir} 至少要有兩份標註（human__*.jsonl 與 ai__*.jsonl 或 reference__*.jsonl）")
    res = compare_sources(sources, cfg, Path(ann_dir) / "reports", task)
    shutil.copyfile(res["compare_js"], Path(task_dir) / "compare_data.js")
    res["agreement_html"] = str(render_markdown_html(Path(res["agreement_md"])))
    return res


def run_merge(ann_dir: str | Path, cfg: AnnotationConfig, primary: str | None = None) -> dict[str, Any]:
    ann = Path(ann_dir)
    humans = sorted(ann.glob("human__*.jsonl"))
    if primary:
        humans.sort(key=lambda p: p.stem != f"human__{primary}")
    if not humans:
        raise FileNotFoundError("找不到主標註者的檔（human__*.jsonl）")
    prim = _load(humans[0], cfg)
    adj = [r for p in sorted(ann.glob("adjudication__*.jsonl")) for r in _load(p, cfg)]
    disputes: list[str] = []
    dp = ann / "reports" / "disputes.csv"
    if dp.exists():
        with dp.open(encoding="utf-8-sig") as fh:
            disputes = [r["segment_id"] for r in csv.DictReader(fh)]
    gold, report = merge_gold(prim, adj, disputes, cfg)
    write_jsonl(gold, ann / "gold.jsonl", cfg)
    write_csv(gold, ann / "gold.csv", cfg)
    report["gold"] = str(ann / "gold.jsonl")
    report["primary"] = humans[0].name
    report["n_adjudication_files"] = len(list(ann.glob("adjudication__*.jsonl")))
    (ann / "reports").mkdir(parents=True, exist_ok=True)
    (ann / "reports" / "gold_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    flag = ann / "reports" / "needs_adjudication.flag"
    if report["disputed_unresolved"]:
        flag.write_text("\n".join(report["disputed_unresolved"]) + "\n", encoding="utf-8")
    elif flag.exists():
        flag.unlink()
    return report


def render_markdown_html(md_path: Path) -> Path:
    """把報告的 Markdown（標題、段落、清單、表格）轉成可直接在瀏覽器看的 HTML。"""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    out, i = [], 0

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-+", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr>" +
                           "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
        elif ln.startswith("- "):
            out.append(f"<li>{inline(ln[2:])}</li>")
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    page = ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>' + html.escape(md_path.stem) + '</title>'
            '<style>body{font-family:system-ui,"Microsoft JhengHei",sans-serif;max-width:1100px;margin:20px auto;padding:0 16px;color:#1f2328}'
            'table{border-collapse:collapse;margin:8px 0}th,td{border:1px solid #d0d7de;padding:4px 8px;font-size:13px}th{background:#f6f8fa}'
            'code{background:#f6f8fa;padding:0 4px}</style></head><body>' + "\n".join(out) + "</body></html>")
    hp = md_path.with_suffix(".html")
    hp.write_text(page, encoding="utf-8")
    return hp
