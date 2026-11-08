"""人工核對用的多視角分段短片（第 1 包驗收：至少 5 段 segment 的起訖與 success 要人工看過）。

從 inspect-h5 產生的 candidates.json 挑 segment（優先 Insert，失敗與成功都有），
依各相機自己的時間戳把多個視角對齊成一支影片：綠框＝標註區間內，灰框＝前後緩衝。
秒數一律來自時間戳（第 i 幀 ↔ 第 i 個時間戳），不用 frame_idx / fps。
相機與欄位全部讀 candidates.json，不寫死任何 HDF5 key。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .candidates import VERB_RE
from .media import make_blob_ref

log = logging.getLogger(__name__)

BANNER_H = 70
GREEN = (0, 200, 0)
GREY = (140, 140, 140)


def _ascii(s: str) -> str:
    return str(s).encode("ascii", "replace").decode("ascii")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")[:40] or "x"


def _split_text(text: str) -> tuple[str, str]:
    m = VERB_RE.match(text or "")
    if not m:
        return "unknown", ""
    return m.group(1).lower(), m.group(2).rstrip(".").strip()


def load_candidates(inspect_dir: str | Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(Path(inspect_dir).glob("*/candidates.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("略過無法讀取的 %s：%s", p, e)
    return out


def _round_robin(items: list[dict]) -> list[dict]:
    """輪流從不同檔案取，讓樣本分散在各錄製檔。"""
    by_file: dict[str, list[dict]] = {}
    for it in items:
        by_file.setdefault(it["file_stem"], []).append(it)
    lists = [by_file[k] for k in sorted(by_file)]
    order: list[dict] = []
    while any(lists):
        for lst in lists:
            if lst:
                order.append(lst.pop(0))
    return order


def select_segments(cands: list[dict], n: int = 5, action: str = "insert") -> list[dict]:
    """優先選目標動作：失敗最多約一半、其餘為成功；不足再用其他動作補。"""
    fails, oks, others = [], [], []
    for c in cands:
        for r in c["segments"]["rows"]:
            if r["level"] != 0 or r["start"] is None or r["end"] is None or r["start"] >= r["end"]:
                continue
            text = (r.get("text") or "").strip()
            item = dict(r, file_stem=c["file"]["stem"], file_name=c["file"]["name"])
            if text.lower().startswith(action):
                (fails if r.get("success") is False else oks).append(item)
            elif not text.lower().startswith("no action"):
                others.append(item)
    fails, oks, others = _round_robin(fails), _round_robin(oks), _round_robin(others)
    n_fail = min(len(fails), math.ceil(n / 2) if oks else n)
    chosen = fails[:n_fail]
    chosen += oks[: max(0, n - len(chosen))]
    if len(chosen) < n:
        chosen += fails[n_fail : n_fail + (n - len(chosen))]
    if len(chosen) < n:
        chosen += others[: n - len(chosen)]
    return chosen


def camera_plan(cand: dict, wanted: list[str] | None = None) -> list[dict]:
    cams = [cm for cm in cand["streams"]["cameras"] if cm.get("timestamps")]
    if wanted:
        by_name = {cm["name"]: cm for cm in cams}
        return [by_name[w] for w in wanted if w in by_name]
    return sorted(cams, key=lambda cm: (-((cm.get("width") or 0) * (cm.get("height") or 0)), cm["name"]))


class _CamReader:
    """把內嵌 mp4 複製到暫存檔交給 OpenCV，依時間戳逐幀前進（略過的幀只 grab 不轉換）。"""

    def __init__(self, f: Any, cam: dict, file_path: str, max_read_bytes: int):
        import cv2

        self.name = cam["name"]
        blob = make_blob_ref(f[cam["path"]], file_path, max_read_bytes)
        if blob is None or blob.access == "unavailable":
            raise RuntimeError(f"{cam['path']} 無法讀取內嵌影片")
        fd, self.tmp = tempfile.mkstemp(prefix="am_clip_", suffix=".mp4")
        with os.fdopen(fd, "wb") as out, blob.open() as src:
            shutil.copyfileobj(src, out, 4 << 20)
        self.cap = cv2.VideoCapture(self.tmp)
        if not self.cap.isOpened():
            self.close()
            raise RuntimeError(f"OpenCV 無法開啟 {cam['path']}")
        self.ts = np.asarray(f[cam["timestamps"]][()], dtype=np.float64).ravel()
        self.i = -1
        self.frame = None
        self.ended = False

    def advance_to(self, t: float) -> tuple[Any, int]:
        moved = False
        while not self.ended and self.i + 1 < len(self.ts) and self.ts[self.i + 1] <= t:
            if not self.cap.grab():
                self.ended = True
                break
            self.i += 1
            moved = True
        if moved:
            ok, img = self.cap.retrieve()
            if ok:
                self.frame = img
        return self.frame, self.i

    def read_index(self, idx: int) -> Any:
        """依序讀到第 idx 幀並解碼（只能往後讀；中間的幀只 grab 不轉換）。讀不到回傳 None。"""
        if idx < self.i:
            raise ValueError(f"只能往後讀：目前在第 {self.i} 幀，要求第 {idx} 幀")
        while self.i < idx:
            if self.ended or not self.cap.grab():
                self.ended = True
                return None
            self.i += 1
        ok, img = self.cap.retrieve()
        return img if ok else None

    def close(self) -> None:
        try:
            if getattr(self, "cap", None) is not None:
                self.cap.release()
        finally:
            try:
                os.remove(self.tmp)
            except OSError:
                pass


def _put(img: Any, text: str, org: tuple[int, int], scale: float = 0.6, color=(255, 255, 255)) -> None:
    """白字＋黑底框（粗細不同的描邊會讓 Hershey 字距錯位，所以改用底框）。"""
    import cv2

    t = _ascii(text)
    (w, h), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 1), (0, 0, 0), -1)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _tile(img: Any, tw: int, th: int) -> np.ndarray:
    import cv2

    canvas = np.zeros((th, tw, 3), np.uint8)
    if img is None:
        _put(canvas, "(no frame yet)", (10, th // 2))
        return canvas
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def export_segment(
    h5_path: str | Path,
    seg: dict,
    cams: list[dict],
    out_dir: str | Path,
    k: int,
    total: int,
    pad: float = 1.5,
    tile: tuple[int, int] = (640, 480),
    fps: float = 30.0,
    max_read_bytes: int = 64 << 20,
) -> dict[str, Any]:
    import cv2
    import h5py

    t0, t1 = float(seg["start"]), float(seg["end"])
    text = seg.get("text") or ""
    action, part = _split_text(text)
    success = seg.get("success")
    tag = "FAIL" if success is False else ("OK" if success is True else "NA")
    name = f"{k:02d}_{seg['file_stem']}_seg{seg['id']}_{_slug(action)}_{_slug(part)}_{tag}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clip_path = out / f"{name}.mp4"
    sheet_path = out / f"{name}_sheet.png"
    readers: list[_CamReader] = []
    skipped: dict[str, str] = {}
    with h5py.File(os.fspath(h5_path), "r") as f:
        try:
            for cam in cams:
                try:
                    readers.append(_CamReader(f, cam, os.fspath(h5_path), max_read_bytes))
                except Exception as e:  # noqa: BLE001 - 單一相機失敗不影響其他視角
                    skipped[cam["name"]] = str(e)
                    log.warning("    略過相機 %s：%s", cam["name"], e)
            if not readers:
                raise RuntimeError(f"沒有可讀的相機：{skipped}")
            ref = readers[0].ts
            times = ref[(ref >= t0 - pad) & (ref <= t1 + pad)]
            if times.size == 0:
                raise RuntimeError("這段時間內參考相機沒有影格")
            n = len(readers)
            cols = 1 if n == 1 else 2
            rows = math.ceil(n / cols)
            tw, th = tile
            width, height = cols * tw, rows * th + BANNER_H
            writer = cv2.VideoWriter(os.fspath(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"無法建立影片檔 {clip_path}")
            targets = [("before", t0 - pad / 2), ("start", t0), ("middle", (t0 + t1) / 2), ("end", t1), ("after", t1 + pad / 2)]
            snaps: dict[str, np.ndarray] = {}
            try:
                for t in times:
                    tiles = []
                    for r in readers:
                        img, idx = r.advance_to(float(t))
                        tl = _tile(img, tw, th)
                        label = f"{r.name}  frame {idx}" + (f"  t-start={r.ts[idx] - t0:+.3f}s" if idx >= 0 else "")
                        _put(tl, label, (10, 24))
                        tiles.append(tl)
                    while len(tiles) < rows * cols:
                        tiles.append(np.zeros((th, tw, 3), np.uint8))
                    grid = np.vstack([np.hstack(tiles[i * cols : (i + 1) * cols]) for i in range(rows)])
                    frame = np.zeros((height, width, 3), np.uint8)
                    frame[BANNER_H:] = grid
                    inside = t0 <= t <= t1
                    _put(frame, f"[{k}/{total}] {seg['file_stem']}  seg {seg['id']}  \"{text}\"  label success={success}", (12, 28), 0.7)
                    _put(
                        frame,
                        f"t={t:.3f}  ({t - t0:+.2f}s from annotated start, length {t1 - t0:.2f}s)  "
                        + ("INSIDE annotated segment" if inside else "padding (outside segment)"),
                        (12, 58), 0.6, GREEN if inside else (200, 200, 200),
                    )
                    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), GREEN if inside else GREY, 8)
                    writer.write(frame)
                    small = None
                    for key, tt in targets:
                        if key == "end":
                            take = t <= t1  # END＝區間內最後一幀（持續更新）
                        else:
                            take = key not in snaps and t >= tt
                        if take:
                            if small is None:
                                small = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                            snaps[key] = small
            finally:
                writer.release()
            if snaps:
                rows_img = []
                for key, _ in targets:
                    if key in snaps:
                        im = snaps[key].copy()
                        _put(im, key.upper(), (12, im.shape[0] - 14), 0.8, (0, 255, 255))
                        rows_img.append(im)
                ok, buf = cv2.imencode(".png", np.vstack(rows_img))
                if ok:
                    buf.tofile(os.fspath(sheet_path))
        finally:
            for r in readers:
                r.close()
    return {
        "no": k,
        "file": seg["file_name"],
        "segment": seg["path"],
        "text": text,
        "action": action,
        "part": part,
        "label_success": success,
        "start_unix": t0,
        "end_unix": t1,
        "duration_s": round(t1 - t0, 3),
        "clip": clip_path.name,
        "sheet": sheet_path.name if sheet_path.exists() else None,
        "frames": int(times.size),
        "cameras": [r.name for r in readers],
        "skipped_cameras": skipped,
    }


def write_checklist(results: list[dict], failures: list[dict], out_dir: str | Path, pad: float,
                    title: str = "人工核對清單（第 1 包驗收）") -> Path:
    out = Path(out_dir)
    cols = ["no", "clip", "file", "segment", "text", "label_success", "duration_s", "start_unix", "end_unix",
            "verdict_start_end", "verdict_success", "note"]
    with (out / "checklist.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({**r, "verdict_start_end": "", "verdict_success": "", "note": ""})
    lines = [
        f"# {title}",
        "",
        "每支短片把各相機依時間戳對齊播放。**綠框＝標註的 segment 區間**，灰框＝前後各約 "
        f"{pad:g} 秒的緩衝。`*_sheet.png` 是同一段的五個時間點（BEFORE / START / MIDDLE / END / AFTER）縮圖。",
        "",
        "每段請確認兩件事：",
        "",
        "1. **起訖**：綠框一出現時動作是否剛開始、綠框消失時動作是否剛結束（差半秒內算正確）。",
        "2. **success**：標籤的成功 / 失敗是否和畫面一致（Insert 成功＝零件確實插到定位）。",
        "",
        "看完直接在對話裡告訴 Claude 結果即可（例如「5 段都對」或「第 3 段結束太早」）；也可以填 checklist.csv。",
        "",
        "| # | 短片 | 動作描述 | 標註 success | 長度 (s) | 起訖正確？ | success 正確？ |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['no']} | `{r['clip']}` | {r['text']} | {r['label_success']} | {r['duration_s']:.2f} | ☐ 對 ☐ 錯 | ☐ 對 ☐ 錯 |")
    if failures:
        lines += ["", "## 無法產生的段落", ""]
        lines += [f"- {fl['segment']}（{fl['file']}）：{fl['error']}" for fl in failures]
    lines.append("")
    p = out / "checklist.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _export_all(segs: list[dict], cams_for: Any, h5_for: Any, out_dir: str | Path, pad: float,
                tile: tuple[int, int], fps: float, title: str) -> dict[str, Any]:
    log.info("挑出 %d 段：", len(segs))
    for i, s in enumerate(segs, 1):
        log.info("  %d. %s %s「%s」success=%s（%.1f s）", i, s["file_stem"], s["path"], s.get("text"), s.get("success"), s["end"] - s["start"])
    results, failures = [], []
    for i, s in enumerate(segs, 1):
        h5 = h5_for(s)
        log.info("[%d/%d] 產生短片 %s %s …", i, len(segs), s["file_stem"], s["path"])
        try:
            if not Path(h5).exists():
                raise FileNotFoundError(str(h5))
            res = export_segment(h5, s, cams_for(s), out_dir, i, len(segs), pad=pad, tile=tile, fps=fps)
            log.info("    → %s（%d 幀，相機 %s）", res["clip"], res["frames"], ", ".join(res["cameras"]))
            results.append(res)
        except Exception as e:  # noqa: BLE001 - 單段失敗不中止
            log.error("    失敗：%s: %s", type(e).__name__, e)
            failures.append({"file": s["file_name"], "segment": s["path"], "error": f"{type(e).__name__}: {e}"})
    checklist = write_checklist(results, failures, out_dir, pad, title)
    (Path(out_dir) / "clips.json").write_text(
        json.dumps({"results": results, "failures": failures}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"results": results, "failures": failures, "checklist": str(checklist)}


def export_check_clips(
    inspect_dir: str | Path,
    raw_dir: str | Path,
    out_dir: str | Path,
    n: int = 5,
    pad: float = 1.5,
    cameras: list[str] | None = None,
    tile: tuple[int, int] = (640, 480),
    fps: float = 30.0,
) -> dict[str, Any]:
    """第 1 包：從 inspect-h5 的 candidates.json 挑段（原始樣本檔）。"""
    cands = load_candidates(inspect_dir)
    if not cands:
        raise FileNotFoundError(f"{inspect_dir} 裡沒有 candidates.json，請先執行 inspect-h5")
    by_stem = {c["file"]["stem"]: c for c in cands}
    segs = select_segments(cands, n)
    if not segs:
        raise ValueError("找不到可核對的 segment")
    return _export_all(
        segs,
        lambda s: camera_plan(by_stem[s["file_stem"]], cameras),
        lambda s: Path(raw_dir) / s["file_name"],
        out_dir, pad, tile, fps, "人工核對清單（第 1 包驗收）",
    )


def select_from_index(
    index_dir: str | Path,
    action: str,
    part: str,
    n: int = 5,
    seed: int = 0,
    seg_container: str = "/segments_info",
) -> list[dict]:
    """第 2 包：從 segments.csv 挑某個動作 + 零件的段，失敗約一半、各段來自不同錄製檔、只取驗證 ok 的段。"""
    import random

    path = Path(index_dir) / "segments.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，請先執行 index-segments")
    with path.open(encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("action") == action and r.get("part") == part and r.get("validation_status") == "ok"]
    if not rows:
        raise ValueError(f"索引裡沒有 {action} / {part} 的有效段")
    rng = random.Random(seed)
    fails = sorted((r for r in rows if r["success"] == "False"), key=lambda r: (r["recording_id"], natural_int(r["segment_id"])))
    oks = sorted((r for r in rows if r["success"] == "True"), key=lambda r: (r["recording_id"], natural_int(r["segment_id"])))
    rng.shuffle(fails)
    rng.shuffle(oks)
    n_fail = min(len(fails), math.ceil(n / 2))
    chosen: list[dict] = []
    used: set[str] = set()
    for pool, limit in ((fails, n_fail), (oks, n)):
        for r in pool:
            if len(chosen) >= limit:
                break
            if r["recording_id"] not in used:
                chosen.append(r)
                used.add(r["recording_id"])
    for r in fails + oks:  # 不足時允許同一錄製檔
        if len(chosen) >= n:
            break
        if r not in chosen:
            chosen.append(r)
    segs = []
    for r in chosen:
        segs.append({
            "file_stem": r["recording_id"],
            "file_name": f"{r['recording_id']}.h5",
            "path": f"{seg_container.rstrip('/')}/{r['segment_id']}",
            "id": r["segment_id"],
            "start": float(r["start_unix"]),
            "end": float(r["end_unix"]),
            "text": r.get("text") or "",
            "success": {"True": True, "False": False}.get(r["success"]),
            "level": 0,
        })
    return sorted(segs, key=lambda s: (s["success"] is not False, s["file_stem"]))


def natural_int(s: str) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


def export_check_clips_from_index(
    index_dir: str | Path,
    slim_dir: str | Path,
    out_dir: str | Path,
    mapping: Any,
    part: str | None = None,
    action: str | None = None,
    n: int = 5,
    seed: int = 0,
    pad: float = 1.5,
    tile: tuple[int, int] = (640, 480),
    fps: float = 30.0,
) -> dict[str, Any]:
    """第 2 包：依確認過的映射與索引，對選定零件抽段做多視角短片（讀 data/slim 精簡版）。"""
    action = action or mapping.target_action or "insert"
    part = part or mapping.target_part
    if not part:
        raise ValueError("沒有指定零件：請用 --part，或在 field_mapping.yaml 設定 action_parse.target_part")
    segs = select_from_index(index_dir, action, part, n, seed, mapping.seg_container)
    order = list(mapping.best_candidates) + [c for c in mapping.cameras if c not in mapping.best_candidates]
    cams = [{"name": c, "path": mapping.cameras[c]["video"], "timestamps": mapping.cameras[c]["timestamps"]} for c in order]
    return _export_all(
        segs,
        lambda s: cams,
        lambda s: Path(slim_dir) / s["file_name"],
        out_dir, pad, tile, fps, f"人工核對清單（第 2 包：{action.capitalize()} {part}，從索引抽 {len(segs)} 段）",
    )
