"""第 3 包：依真實時間戳擷取選定零件各段的影格、產生多視角預覽與 metadata（P2-1～P2-5）。

P2-1 由 segment ID 找到精簡版檔案，依「各相機自己的時間戳」取 [start − start_buffer, end + end_buffer]
P2-2 均勻取樣 n 幀（含首尾，n=1 取視窗最後一幀＝標註結束 + end_buffer 的最後狀態）；沒有隨機性，同設定永遠同一組 frame ID
P2-3 每個視角各自輸出影格；另產生同時間多視角 contact sheet
P2-4 輸出 JPEG/PNG、contact sheet、metadata JSON；缺幀標記 missing 並寫原因
P2-5 快取鍵＝來源指紋 + 取樣設定 + 相機；同設定第二次直接略過，換設定寫到另一個資料夾，永不改動原檔
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .util import now_iso

log = logging.getLogger(__name__)

TILE_W, TILE_H = 320, 240
HEADER_H, TITLE_H = 26, 44
FAR_DELTA_S = 0.25  # 取到的幀與目標時間差超過這個值就標記 far


@dataclass(frozen=True)
class Sampling:
    n_frames: int = 8
    start_buffer: float = 0.5
    end_buffer: float = 2.0
    fmt: str = "jpg"
    jpeg_quality: int = 95
    strategy: str = "uniform"
    seed: int = 0

    def key(self) -> str:
        return f"{self.strategy}_n{self.n_frames}_sb{self.start_buffer:g}_eb{self.end_buffer:g}_{self.fmt}"


def window(start: float, end: float, s: Sampling) -> tuple[float, float]:
    return start - s.start_buffer, end + s.end_buffer


def target_times(start: float, end: float, s: Sampling) -> list[float]:
    """n=1 → [視窗終點]；n≥2 → 從視窗起點到終點均勻 n 點（含首尾）。"""
    w0, w1 = window(start, end, s)
    if s.n_frames <= 1:
        return [w1]
    return [w0 + (w1 - w0) * i / (s.n_frames - 1) for i in range(s.n_frames)]


def pick_frames(ts: np.ndarray, targets: list[float], w0: float, w1: float) -> list[dict[str, Any]]:
    """每個目標時間取視窗內時間戳最接近的那一幀；視窗內沒有幀就標 missing。"""
    inside = np.nonzero((ts >= w0) & (ts <= w1))[0]
    out = []
    for t in targets:
        if inside.size == 0:
            out.append({"target_unix": t, "frame_idx": None, "source_unix": None, "missing": "no_frame_in_window"})
            continue
        j = int(inside[np.argmin(np.abs(ts[inside] - t))])
        d = float(ts[j] - t)
        out.append({"target_unix": t, "frame_idx": j, "source_unix": float(ts[j]), "delta_s": d,
                    "far": abs(d) > FAR_DELTA_S, "missing": None})
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_") or "x"


def fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def cache_key(seg: dict, s: Sampling, cams: list[dict], fp: str) -> str:
    payload = json.dumps({"seg": [seg["recording_id"], seg["segment_id"], seg["start"], seg["end"]],
                          "sampling": asdict(s), "cams": [c["name"] for c in cams], "src": fp,
                          "tool": "extract-v1"}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- 選段


def load_part_segments(index_dir: str | Path, action: str, part: str) -> list[dict]:
    p = Path(index_dir) / "segments.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，請先執行 index-segments")
    with p.open(encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("action") == action and r.get("part") == part and r.get("validation_status") in ("ok", "warn")]
    segs = []
    for r in rows:
        segs.append({
            "recording_id": r["recording_id"],
            "segment_id": r["segment_id"],
            "start": float(r["start_unix"]),
            "end": float(r["end_unix"]),
            "text": r.get("text") or "",
            "success": {"True": True, "False": False}.get(r.get("success")),
            "validation_status": r.get("validation_status"),
            "error_code": r.get("error_code") or "",
        })
    return segs


def select_pilot(segs: list[dict], n_fail: int = 20, n_ok: int = 20, seed: int = 0) -> list[dict]:
    """失敗 / 成功各取若干段，先打散再輪流從不同錄製檔取，結果可重現。"""
    rng = random.Random(seed)

    def spread(items: list[dict], k: int) -> list[dict]:
        items = sorted(items, key=lambda r: (r["recording_id"], int(r["segment_id"]) if r["segment_id"].isdigit() else 0))
        rng.shuffle(items)
        by_rec: dict[str, list[dict]] = {}
        for it in items:
            by_rec.setdefault(it["recording_id"], []).append(it)
        lists = list(by_rec.values())
        out: list[dict] = []
        while len(out) < k and any(lists):
            for lst in lists:
                if lst and len(out) < k:
                    out.append(lst.pop(0))
        return out

    ok_valid = [s for s in segs if s["validation_status"] == "ok"]
    fails = spread([s for s in ok_valid if s["success"] is False], n_fail)
    oks = spread([s for s in ok_valid if s["success"] is True], n_ok)
    return fails + oks


# ---------------------------------------------------------------- 單一錄製檔（可在子行程執行）


def _put(img: Any, text: str, org: tuple[int, int], scale: float = 0.5, color=(255, 255, 255), box: bool = True) -> None:
    """白字＋黑底框（粗細不同的描邊會讓 Hershey 字距錯位，所以改用底框）。"""
    import cv2

    t = str(text).encode("ascii", "replace").decode("ascii")
    if box:
        (w, h), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x, y = org
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 1), (0, 0, 0), -1)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _write_img(path: Path, img: Any, s: Sampling) -> None:
    import cv2

    if s.fmt == "png":
        ok, buf = cv2.imencode(".png", img)
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(s.jpeg_quality)])
    if not ok:
        raise OSError(f"影像編碼失敗：{path}")
    buf.tofile(os.fspath(path))


def _contact_sheet(seg: dict, s: Sampling, cams: list[dict], entries: dict[str, list[dict]],
                   images: dict[tuple[str, int], Any]) -> Any:
    import cv2

    n = s.n_frames
    width = max(n * TILE_W, 3 * TILE_W)
    height = TITLE_H + HEADER_H + len(cams) * TILE_H
    sheet = np.zeros((height, width, 3), np.uint8)
    lab = "FAIL" if seg["success"] is False else ("OK" if seg["success"] else "NA")
    _put(sheet, f"{seg['recording_id']}  seg {seg['segment_id']}  \"{seg['text']}\"  label={lab}", (8, 18), 0.5)
    _put(sheet, f"len={seg['end'] - seg['start']:.2f}s  window=[start-{s.start_buffer:g}s, end+{s.end_buffer:g}s]  "
                f"n={n} ({s.strategy})", (8, 38), 0.45, (200, 200, 200))
    first_cam = entries[cams[0]["name"]]
    for k, e in enumerate(first_cam):
        t_rel = e["target_unix"] - seg["start"]
        inside = seg["start"] <= e["target_unix"] <= seg["end"]
        x0 = k * TILE_W
        color = (0, 140, 0) if inside else (90, 90, 90)
        cv2.rectangle(sheet, (x0, TITLE_H), (x0 + TILE_W - 1, TITLE_H + HEADER_H - 1), color, -1)
        tag = "in segment" if inside else ("after end" if e["target_unix"] > seg["end"] else "before start")
        _put(sheet, f"#{k}  t={t_rel:+.2f}s  {tag}", (x0 + 6, TITLE_H + 18), 0.48, box=False)
    for r, cam in enumerate(cams):
        y0 = TITLE_H + HEADER_H + r * TILE_H
        for k, e in enumerate(entries[cam["name"]]):
            x0 = k * TILE_W
            img = images.get((cam["name"], e["frame_idx"])) if e.get("frame_idx") is not None else None
            if img is None:
                cv2.rectangle(sheet, (x0, y0), (x0 + TILE_W - 1, y0 + TILE_H - 1), (0, 0, 120), -1)
                _put(sheet, f"{cam['name']}: MISSING", (x0 + 8, y0 + 40), 0.55)
                _put(sheet, str(e.get("missing") or ""), (x0 + 8, y0 + 70), 0.45)
                continue
            tile = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
            sheet[y0 : y0 + TILE_H, x0 : x0 + TILE_W] = tile
            _put(sheet, f"{cam['name']} f{e['frame_idx']}", (x0 + 6, y0 + 18), 0.45)
            _put(sheet, f"t={e['source_unix'] - seg['start']:+.3f}s" + (" FAR" if e.get("far") else ""),
                 (x0 + 6, y0 + TILE_H - 8), 0.45, (0, 255, 255) if e.get("far") else (255, 255, 255))
    return sheet


def extract_recording(job: dict) -> dict:
    """處理一個錄製檔的所有段與取樣設定。可在子行程呼叫（參數與回傳都是純資料）。"""
    import h5py

    from .check_clips import _CamReader

    slim = Path(job["slim_path"])
    rid = job["recording_id"]
    cams: list[dict] = job["cams"]
    samplings = [Sampling(**d) for d in job["samplings"]]
    out_root = Path(job["out_root"])
    force = bool(job.get("force"))
    summary: dict[str, Any] = {"recording_id": rid, "done": 0, "skipped": 0, "missing_frames": 0, "errors": []}
    if not slim.exists():
        summary["errors"].append(f"找不到精簡版 {slim}")
        return summary
    fp = fingerprint(slim)
    # 哪些 (段, 設定) 需要重做
    todo: list[tuple[dict, Sampling, Path, str]] = []
    for seg in job["segments"]:
        for s in samplings:
            d = out_root / s.key() / "segments" / f"{rid}__seg{seg['segment_id']}"
            key = cache_key(seg, s, cams, fp)
            meta = d / "meta.json"
            if not force and meta.exists():
                try:
                    if json.loads(meta.read_text(encoding="utf-8")).get("cache_key") == key:
                        summary["skipped"] += 1
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            todo.append((seg, s, d, key))
    if not todo:
        return summary
    with h5py.File(os.fspath(slim), "r") as f:
        ts: dict[str, np.ndarray | None] = {}
        for c in cams:
            ts[c["name"]] = (np.asarray(f[c["timestamps"]][()], dtype=np.float64).ravel()
                             if c["timestamps"] in f and c["video"] in f else None)
        plans = []
        need: dict[str, set[int]] = {c["name"]: set() for c in cams}
        for seg, s, d, key in todo:
            w0, w1 = window(seg["start"], seg["end"], s)
            targets = target_times(seg["start"], seg["end"], s)
            entries = {}
            for c in cams:
                t = ts[c["name"]]
                if t is None:
                    entries[c["name"]] = [{"target_unix": x, "frame_idx": None, "source_unix": None,
                                           "missing": "camera_missing"} for x in targets]
                    continue
                es = pick_frames(t, targets, w0, w1)
                entries[c["name"]] = es
                need[c["name"]].update(e["frame_idx"] for e in es if e["frame_idx"] is not None)
            plans.append((seg, s, d, key, entries, (w0, w1)))
        images: dict[tuple[str, int], Any] = {}
        for c in cams:
            idxs = sorted(need[c["name"]])
            if not idxs:
                continue
            try:
                reader = _CamReader(f, {"name": c["name"], "path": c["video"], "timestamps": c["timestamps"]},
                                    os.fspath(slim), 64 << 20)
            except Exception as e:  # noqa: BLE001
                summary["errors"].append(f"{c['name']}: {e}")
                continue
            try:
                for i in idxs:
                    img = reader.read_index(i)
                    if img is not None:
                        images[(c["name"], i)] = img
            finally:
                reader.close()
    for seg, s, d, key, entries, (w0, w1) in plans:
        d.mkdir(parents=True, exist_ok=True)
        cam_meta = []
        for c in cams:
            frame_paths, times_rel = [], []
            for k, e in enumerate(entries[c["name"]]):
                if e.get("frame_idx") is not None and (c["name"], e["frame_idx"]) not in images and not e.get("missing"):
                    e["missing"] = "decode_failed"
                if e.get("missing"):
                    summary["missing_frames"] += 1
                    frame_paths.append(None)
                    times_rel.append(None)
                    continue
                rel = e["source_unix"] - seg["start"]
                name = f"{c['name']}__{k:02d}__t{rel:+08.3f}s__f{e['frame_idx']:06d}.{s.fmt}"
                _write_img(d / name, images[(c["name"], e["frame_idx"])], s)
                e["path"] = name
                frame_paths.append(name)
                times_rel.append(round(rel, 4))
            cam_meta.append({
                "segment_id": f"{seg['recording_id']}__seg{seg['segment_id']}",
                "camera": c["name"],
                "source_times_sec": times_rel,
                "frame_paths": frame_paths,
                "sampling": s.strategy,
                "n_frames": s.n_frames,
                "frames": entries[c["name"]],
            })
        sheet = _contact_sheet(seg, s, cams, entries, images)
        _write_img(d / f"sheet.{s.fmt}", sheet, s)
        meta = {
            "recording_id": seg["recording_id"],
            "segment_id": seg["segment_id"],
            "text": seg["text"],
            "label_success": seg["success"],
            "start_unix": seg["start"],
            "end_unix": seg["end"],
            "duration_s": seg["end"] - seg["start"],
            "window_unix": [w0, w1],
            "sampling": asdict(s),
            "cameras": cam_meta,
            "source": {"slim": slim.name, "fingerprint": fp},
            "cache_key": key,
            "created_at": now_iso(),
            "tool": f"assembly_monitor.extract {__version__}",
        }
        tmp = d / "meta.json.part"
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, d / "meta.json")
        summary["done"] += 1
    return summary


# ---------------------------------------------------------------- 整批與彙整


def rebuild_set_outputs(set_dir: Path, title: str, pilot_ids: set[str] | None = None,
                        html_name: str = "index.html") -> dict[str, str]:
    """從 segments/*/meta.json 重建 samples.csv、manifest.json 與 HTML 瀏覽頁。"""
    metas = []
    for p in sorted((set_dir / "segments").glob("*/meta.json")):
        try:
            metas.append((p.parent, json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    rows = []
    n_missing = 0
    for d, m in metas:
        for cm in m["cameras"]:
            for k, e in enumerate(cm["frames"]):
                n_missing += 1 if e.get("missing") else 0
                rows.append({
                    "recording_id": m["recording_id"], "segment_id": m["segment_id"], "label_success": m["label_success"],
                    "camera": cm["camera"], "k": k,
                    "target_sec": round(e["target_unix"] - m["start_unix"], 4),
                    "source_sec": round(e["source_unix"] - m["start_unix"], 4) if e.get("source_unix") is not None else "",
                    "source_unix": e.get("source_unix") if e.get("source_unix") is not None else "",
                    "frame_idx": e.get("frame_idx") if e.get("frame_idx") is not None else "",
                    "delta_s": round(e["delta_s"], 4) if e.get("delta_s") is not None else "",
                    "far": e.get("far", ""),
                    "path": f"segments/{d.name}/{e['path']}" if e.get("path") else "",
                    "missing": e.get("missing") or "",
                })
    csv_path = set_dir / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        cols = ["recording_id", "segment_id", "label_success", "camera", "k", "target_sec", "source_sec", "source_unix",
                "frame_idx", "delta_s", "far", "path", "missing"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    sampling = metas[0][1]["sampling"] if metas else {}
    manifest = {
        "generated_at": now_iso(),
        "sampling": sampling,
        "n_segments": len(metas),
        "n_failure": sum(1 for _, m in metas if m["label_success"] is False),
        "n_success": sum(1 for _, m in metas if m["label_success"] is True),
        "n_frames_written": sum(1 for r in rows if r["path"]),
        "n_missing_frames": n_missing,
        "n_far_frames": sum(1 for r in rows if r["far"] is True),
        "segments": [{"dir": f"segments/{d.name}", "recording_id": m["recording_id"], "segment_id": m["segment_id"],
                      "label_success": m["label_success"], "cache_key": m["cache_key"]} for d, m in metas],
    }
    man_path = set_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    # HTML 瀏覽頁（本機開啟即可；失敗在前）
    show = [(d, m) for d, m in metas if pilot_ids is None or f"{m['recording_id']}__seg{m['segment_id']}" in pilot_ids]
    show.sort(key=lambda dm: (dm[1]["label_success"] is not False, dm[1]["recording_id"], int(dm[1]["segment_id"]) if str(dm[1]["segment_id"]).isdigit() else 0))
    fmt = sampling.get("fmt", "jpg")
    cards = []
    for i, (d, m) in enumerate(show, 1):
        lab = "失敗" if m["label_success"] is False else "成功"
        cls = "fail" if m["label_success"] is False else "ok"
        cards.append(
            f'<section class="card {cls}"><h2>#{i} {html.escape(m["recording_id"])} · seg {html.escape(str(m["segment_id"]))} · '
            f'標註{lab} · {m["duration_s"]:.1f} 秒</h2>'
            f'<a href="segments/{d.name}/sheet.{fmt}"><img loading="lazy" src="segments/{d.name}/sheet.{fmt}" alt="{html.escape(d.name)}"></a>'
            f'<p>☐ 接觸區看得到　☐ 時間對齊　☐ 畫面完整　備註：</p></section>')
    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,"Microsoft JhengHei",sans-serif;margin:16px;background:#f6f6f6;color:#222}}
.card{{background:#fff;border-left:6px solid #888;margin:0 0 18px;padding:8px 12px;border-radius:6px}}
.card.fail{{border-color:#c0392b}}.card.ok{{border-color:#27ae60}}h1{{font-size:20px}}h2{{font-size:15px;margin:4px 0 8px}}
img{{max-width:100%;height:auto;display:block}}p{{margin:6px 0 2px;font-size:14px}}</style></head><body>
<h1>{html.escape(title)}</h1>
<p>每張圖：上排標頭是取樣時間（相對標註起點；綠＝標註區間內、灰＝前後緩衝），三列依序為各相機。
請檢查：接觸區（接頭與插座）是否看得到、時間是否對齊（綠色區間內應該在插、最後一欄是否看得到最後狀態）、影格是否完整（有無 MISSING／FAR）。</p>
<p>共 {len(show)} 段：失敗 {sum(1 for _, m in show if m['label_success'] is False)}、成功 {sum(1 for _, m in show if m['label_success'] is True)}；
缺幀 {n_missing}、時間差過大 {manifest['n_far_frames']}。</p>
{''.join(cards)}</body></html>"""
    html_path = set_dir / html_name
    html_path.write_text(page, encoding="utf-8")
    return {"samples": str(csv_path), "manifest": str(man_path), "html": str(html_path)}


def run_extraction(
    *,
    mapping: Any,
    index_dir: str | Path,
    slim_dir: str | Path,
    out_dir: str | Path,
    part: str | None = None,
    action: str | None = None,
    which: str = "pilot",
    n_fail: int = 20,
    n_ok: int = 20,
    seed: int = 0,
    n_frames_list: tuple[int, ...] = (1, 4, 8),
    start_buffer: float | None = None,
    end_buffer: float | None = None,
    fmt: str = "jpg",
    jpeg_quality: int = 95,
    cameras: list[str] | None = None,
    workers: int = 4,
    force: bool = False,
    segment_ids: list[str] | None = None,
) -> dict[str, Any]:
    action = action or mapping.target_action or "insert"
    part = part or mapping.target_part
    if not part:
        raise ValueError("沒有指定零件：請用 --part 或在 field_mapping.yaml 設定 target_part")
    cam_names = cameras or list(mapping.best_candidates)
    unknown = [c for c in cam_names if c not in mapping.cameras]
    if unknown:
        raise ValueError(f"未定義的相機：{unknown}")
    cams = [{"name": c, "video": mapping.cameras[c]["video"], "timestamps": mapping.cameras[c]["timestamps"]} for c in cam_names]
    sb = mapping.start_buffer_sec if start_buffer is None else start_buffer
    eb = mapping.end_buffer_sec if end_buffer is None else end_buffer
    samplings = [Sampling(n_frames=n, start_buffer=sb, end_buffer=eb, fmt=fmt, jpeg_quality=jpeg_quality, seed=seed)
                 for n in n_frames_list]
    segs = load_part_segments(index_dir, action, part)
    if not segs:
        raise ValueError(f"索引裡沒有 {action} / {part} 的段")
    if segment_ids is not None:  # 例如 --set dev：只做切分 dev 的段
        wanted = set(segment_ids)
        chosen = [s for s in segs if f"{s['recording_id']}__seg{s['segment_id']}" in wanted]
        missing = wanted - {f"{s['recording_id']}__seg{s['segment_id']}" for s in chosen}
        if missing:
            log.warning("索引裡找不到 %d 段：%s", len(missing), sorted(missing)[:5])
    else:
        chosen = select_pilot(segs, n_fail, n_ok, seed) if which == "pilot" else segs
    root = Path(out_dir) / f"{_slug(action)}_{_slug(part)}"
    root.mkdir(parents=True, exist_ok=True)
    sel_path = root / f"selection_{which}.csv"
    with sel_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["recording_id", "segment_id", "text", "success", "start", "end", "validation_status"])
        w.writeheader()
        for s in chosen:
            w.writerow({k: s[k] for k in ("recording_id", "segment_id", "text", "success", "start", "end", "validation_status")})
    by_rec: dict[str, list[dict]] = {}
    for s in chosen:
        by_rec.setdefault(s["recording_id"], []).append(s)
    jobs = [{"slim_path": str(Path(slim_dir) / f"{rid}.h5"), "recording_id": rid, "segments": ss, "cams": cams,
             "samplings": [asdict(x) for x in samplings], "out_root": str(root), "force": force}
            for rid, ss in sorted(by_rec.items())]
    log.info("第 3 包影格擷取：%s %s，%s %d 段（失敗 %d / 成功 %d），%d 個錄製檔，設定 %s",
             action, part, which, len(chosen), sum(1 for s in chosen if s["success"] is False),
             sum(1 for s in chosen if s["success"] is True), len(jobs), [x.key() for x in samplings])
    results: list[dict] = []
    if workers > 1 and len(jobs) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                futs = {pool.submit(extract_recording, j): j["recording_id"] for j in jobs}
                for k, fut in enumerate(as_completed(futs), 1):
                    r = fut.result()
                    results.append(r)
                    log.info("  [%d/%d] %s：新做 %d、沿用 %d%s", k, len(jobs), r["recording_id"], r["done"], r["skipped"],
                             f"、錯誤 {r['errors']}" if r["errors"] else "")
        except Exception as e:  # noqa: BLE001 - 平行失敗就改逐一處理
            log.warning("平行處理失敗（%s），改為逐一處理", e)
            done_ids = {r["recording_id"] for r in results}
            jobs = [j for j in jobs if j["recording_id"] not in done_ids]
            workers = 1
    if workers <= 1 or len(jobs) <= 1:
        for k, j in enumerate(jobs, 1):
            r = extract_recording(j)
            results.append(r)
            log.info("  [%d/%d] %s：新做 %d、沿用 %d%s", k, len(jobs), r["recording_id"], r["done"], r["skipped"],
                     f"、錯誤 {r['errors']}" if r["errors"] else "")
    pilot_ids = {f"{s['recording_id']}__seg{s['segment_id']}" for s in chosen} if which != "all" else None
    html_name = "index.html" if which in ("pilot", "all") else f"index_{_slug(which)}.html"
    sets = {}
    for x in samplings:
        title = f"{action.capitalize()} {part}：{'' if which == 'all' else which + ' '}影格預覽（{x.key()}）"
        sets[x.key()] = rebuild_set_outputs(root / x.key(), title, pilot_ids, html_name)
    largest = max(samplings, key=lambda x: x.n_frames)
    gallery = Path(sets[largest.key()]["html"])
    top = Path(out_dir) / f"{which}_index.html"
    rel = os.path.relpath(gallery, top.parent).replace(os.sep, "/")
    top.write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={html.escape(rel)}">'
                   f'<a href="{html.escape(rel)}">開啟 {html.escape(largest.key())} 預覽</a>', encoding="utf-8")
    return {"selection": str(sel_path), "results": results, "sets": sets, "gallery": str(gallery), "top_index": str(top),
            "errors": [e for r in results for e in r["errors"]],
            "n_done": sum(r["done"] for r in results), "n_skipped": sum(r["skipped"] for r in results),
            "n_missing_frames": sum(r["missing_frames"] for r in results)}
