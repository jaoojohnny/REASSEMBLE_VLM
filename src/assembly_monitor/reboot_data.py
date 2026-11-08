"""第 13 包：REBOOT 16 mm 圓柱（開發集）——下載彩色影片、建索引、做檢查用的拼圖。

為什麼先用圓柱：4 種插頭零件沒有公開失敗時間點；只有官方範例（16 mm 圓柱救援段）附 meta/phase.json。
圓柱只拿來設計流程（怎麼切段、用哪台相機、取哪幾格），插頭零件留到流程凍結後只測一次。

只下載：README、meta/、3 台相機（俯視 cam_high、左手腕、右手腕）的彩色影片。
不下載 data/（深度＋關節，約 34 GB）和低角度前視 cam_low。
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .reboot_meta import hf_downloader, hf_lister, local_name
from .util import now_iso

log = logging.getLogger(__name__)

CYL = {"recovery": "REBOOT26/sample_recovery-demonstration", "expert": "REBOOT26/16mm-cylinder_install"}
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
PHASES = ("Align (pick)", "Engage (pick)", "Transport", "Align (place)", "Engage (place)")
PLACE_PHASES = (4, 5)  # 1 起算：4 = Align (place)、5 = Engage (place)
MAX_GB = 6.0
VIDEO_TMPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
TILE_W, TILE_H, TITLE_H, HEAD_H = 320, 240, 48, 24


# ---------------------------------------------------------------- 下載


def download_plan(entries: list[tuple[str, int]], cams: tuple[str, ...] = CAMS) -> list[tuple[str, int]]:
    """從檔案清單挑出要下載的：README、meta/、指定相機的影片。"""
    keep = []
    vids = tuple(f"videos/observation.images.{c}/" for c in cams)
    for path, size in entries:
        if path == "README.md" or path.startswith("meta/") or path.startswith(vids):
            keep.append((path, size))
    return sorted(keep)


def plan_all(lister: Callable[[str], list[tuple[str, int]]] = hf_lister) -> dict[str, list[tuple[str, int]]]:
    return {kind: download_plan(lister(repo)) for kind, repo in CYL.items()}


def download(plan: dict[str, list[tuple[str, int]]], dest_root: str | Path,
             downloader: Callable[[str, str, Path], Any] = hf_downloader) -> dict[str, Any]:
    """逐檔下載；本機已有同樣大小的檔案就跳過（可中斷後重跑）。"""
    stats = {"files": 0, "skipped": 0, "bytes": 0, "failed": []}
    for kind, files in plan.items():
        repo = CYL[kind]
        dest = Path(dest_root) / local_name(repo)
        for i, (f, size) in enumerate(files, 1):
            p = dest / f
            if p.exists() and p.stat().st_size == size:
                stats["skipped"] += 1
                continue
            log.info("  [%s %d/%d] %s（%.0f MB）", kind, i, len(files), f, size / 1e6)
            try:
                downloader(repo, f, dest)
                stats["files"] += 1
                stats["bytes"] += size
            except Exception as e:  # noqa: BLE001
                stats["failed"].append(f"{repo}/{f}: {type(e).__name__}: {e}"[:300])
                log.warning("    下載失敗：%s", e)
    return stats


# ---------------------------------------------------------------- 索引


def episode_index(rows: list[dict[str, Any]], info: dict[str, Any], cams: tuple[str, ...] = CAMS) -> list[dict[str, Any]]:
    """meta/episodes 表的列 → 每段的影片檔與起訖時間（LeRobot v3：一個 mp4 裡接了很多段）。"""
    tmpl = info.get("video_path") or VIDEO_TMPL
    out = []
    for r in sorted(rows, key=lambda r: int(r["episode_index"])):
        e: dict[str, Any] = {"episode_index": int(r["episode_index"]), "length": int(r["length"]), "videos": {}}
        for c in cams:
            k = f"observation.images.{c}"
            ci, fi = r.get(f"videos/{k}/chunk_index"), r.get(f"videos/{k}/file_index")
            if ci is None or fi is None:
                continue
            e["videos"][c] = {"path": tmpl.format(video_key=k, chunk_index=int(ci), file_index=int(fi)),
                              "from": float(r[f"videos/{k}/from_timestamp"]),
                              "to": float(r[f"videos/{k}/to_timestamp"])}
        out.append(e)
    return out


def read_episodes(local: Path, cams: tuple[str, ...] = CAMS) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    info = json.loads((local / "meta" / "info.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for p in sorted((local / "meta" / "episodes").rglob("*.parquet")):
        cols = [c for c in pq.read_schema(p).names if not c.startswith("stats/")]
        rows += pq.read_table(p, columns=cols).to_pylist()
    return episode_index(rows, info, cams)


def failure_info(ann: dict[str, Any]) -> dict[str, Any]:
    """phase.json 一段的失敗標註，加上檢查旗標（官方範例的標註有不少筆誤）。"""
    f = ann.get("failure") or {}
    k, ind, rec = f.get("originating_phase"), f.get("induced_at_frame"), f.get("recovery_started_at_frame")
    pb = ann.get("phase_boundaries") or {}
    taus = [pb.get(f"tau_{i}") for i in range(6)]
    flags = []
    ok_phase = isinstance(k, int) and 1 <= k <= 5
    if not ok_phase:
        flags.append("bad_phase")
    if not (isinstance(ind, int) and isinstance(rec, int)) or rec <= ind:
        flags.append("bad_order")
    if "lucky" in str(f.get("language_description", "")).lower():
        flags.append("lucky")
    if any(not isinstance(t, int) for t in taus) or any(taus[i] > taus[i + 1] for i in range(5)):
        flags.append("bad_bounds")
    elif ok_phase and isinstance(ind, int) and not taus[k - 1] <= ind <= taus[k]:
        flags.append("phase_out")
    return {"phase": k, "phase_name": PHASES[k - 1] if ok_phase else None, "mode": f.get("failure_mode"),
            "induced": ind, "recovery": rec, "taus": taus, "flags": flags, "place": k in PLACE_PHASES,
            "text": f.get("language_description", ""), "length": ann.get("duration_frames")}


def read_phase(local: Path) -> dict[int, dict[str, Any]]:
    p = local / "meta" / "phase.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {int(e["episode_index"]): failure_info(e) for e in d.get("episodes", [])}


# ---------------------------------------------------------------- 取格與拼圖


def _clamp_points(pts: list[tuple[int, str]], length: int, n_max: int = 8) -> list[tuple[int, str]]:
    last = length - 1
    seen: dict[int, str] = {}
    for f, lab in pts:
        f = min(max(int(f), 0), last)
        if f not in seen:
            seen[f] = lab
    return sorted(seen.items())[:n_max]


def recovery_points(length: int, fi: dict[str, Any]) -> list[tuple[int, str]]:
    """救援段要看的格：放置階段開始、失敗開始、中間、救援前一格、救援後 1.5 秒、之後、最後一格。"""
    last = length - 1
    taus, ind, rec = fi["taus"], fi["induced"], fi["recovery"]
    pts: list[tuple[int, str]] = []
    if "bad_bounds" not in fi["flags"]:
        pts += [(taus[3], "align-place"), (taus[4], "engage-place")]
    if isinstance(ind, int):
        pts.append((ind, "induced"))
    if isinstance(ind, int) and isinstance(rec, int) and rec > ind:
        pts.append(((ind + rec) // 2, "mid"))
    if isinstance(rec, int):
        pts += [(rec - 1, "recovery-1"), (rec + 45, "recovery+1.5s"), ((max(rec, 0) + last) // 2, "after")]
    pts.append((last, "last"))
    return _clamp_points(pts, length)


def even_points(length: int, n: int = 8) -> list[tuple[int, str]]:
    return _clamp_points([(round(i * (length - 1) / (n - 1)), f"{i}/{n - 1}") for i in range(n)], length, n)


class FrameReader:
    """依秒數讀 mp4 的影格（PyAV；AV1 用 libdav1d）。同一檔往後讀時不重新 seek。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._h: dict[str, dict[str, Any]] = {}

    def _get(self, rel: str) -> dict[str, Any]:
        if rel not in self._h:
            import av  # type: ignore[import-not-found]

            c = av.open(str(self.root / rel))
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            self._h[rel] = {"c": c, "s": s, "gen": None, "last": None}
        return self._h[rel]

    def read_at(self, rel: str, t: float, fps: float) -> tuple[Any, float] | tuple[None, None]:
        h = self._get(rel)
        eps = 0.5 / fps
        last = h["last"]
        if last is not None and abs(t - last[0]) <= eps and last[1] is not None:
            return last[1], last[0]
        if h["gen"] is None or last is None or t < last[0] - eps or t - last[0] > 1.0:
            s = h["s"]
            h["c"].seek(int(max(t - 0.05, 0.0) / s.time_base), stream=s, backward=True, any_frame=False)
            h["gen"] = h["c"].decode(s)
            h["last"] = None
        for fr in h["gen"]:
            if fr.time is None:
                continue
            ft = float(fr.time)
            if ft >= t - eps:
                img = fr.to_ndarray(format="bgr24")
                h["last"] = (ft, img)
                return img, ft
            h["last"] = (ft, None)
        h["gen"], h["last"] = None, None
        return None, None

    def close(self) -> None:
        for h in self._h.values():
            try:
                h["c"].close()
            except Exception:  # noqa: BLE001
                pass
        self._h.clear()


def _put(img: Any, text: str, org: tuple[int, int], scale: float = 0.45, color=(255, 255, 255)) -> None:
    import cv2

    t = str(text).encode("ascii", "replace").decode("ascii")
    (w, h), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 1), (0, 0, 0), -1)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def make_sheet(title: list[str], points: list[tuple[int, str]], fps: float,
               grid: dict[tuple[str, int], Any], cams: tuple[str, ...] = CAMS) -> Any:
    import cv2
    import numpy as np

    W = TILE_W * max(len(points), 4)
    sheet = np.zeros((TITLE_H + HEAD_H + TILE_H * len(cams), W, 3), np.uint8)
    for i, line in enumerate(title[:2]):
        _put(sheet, line, (6, 18 + 22 * i), 0.47 if i == 0 else 0.4)
    for j, (f, lab) in enumerate(points):
        x0 = j * TILE_W
        cv2.rectangle(sheet, (x0, TITLE_H), (x0 + TILE_W - 2, TITLE_H + HEAD_H - 2), (90, 60, 0), -1)
        _put(sheet, f"f{f} {f / fps:.1f}s {lab}", (x0 + 6, TITLE_H + 17), 0.45)
    for r, cam in enumerate(cams):
        y0 = TITLE_H + HEAD_H + r * TILE_H
        for j, (f, _) in enumerate(points):
            x0 = j * TILE_W
            img = grid.get((cam, f))
            if img is None:
                cv2.rectangle(sheet, (x0, y0), (x0 + TILE_W - 1, y0 + TILE_H - 1), (0, 0, 110), -1)
                _put(sheet, f"{cam}: MISSING", (x0 + 8, y0 + 40))
                continue
            sheet[y0:y0 + TILE_H, x0:x0 + TILE_W] = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
            if j == 0:
                _put(sheet, cam, (x0 + 6, y0 + 18))
    return sheet


def _write_jpg(path: Path, img: Any, q: int = 85) -> None:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise OSError(f"影像編碼失敗：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def check(dest_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """建索引、對每段做拼圖（3 台相機 × 最多 8 格），寫報告。"""
    out = Path(out_dir)
    res: dict[str, Any] = {"generated": now_iso(), "kinds": {}, "decode": {"asked": 0, "ok": 0, "max_err_s": 0.0}}
    for kind, repo in CYL.items():
        local = Path(dest_root) / local_name(repo)
        info = json.loads((local / "meta" / "info.json").read_text(encoding="utf-8"))
        fps = float(info["fps"])
        eps = read_episodes(local)
        phase = read_phase(local)
        reader = FrameReader(local)
        t0 = time.time()
        rows = []
        try:
            for e in eps:
                fi = phase.get(e["episode_index"])
                pts = recovery_points(e["length"], fi) if fi else even_points(e["length"])
                grid: dict[tuple[str, int], Any] = {}
                for cam in CAMS:
                    v = e["videos"].get(cam)
                    if not v or not (local / v["path"]).exists():
                        continue
                    for f, _ in pts:
                        res["decode"]["asked"] += 1
                        t = v["from"] + f / fps
                        try:
                            img, ft = reader.read_at(v["path"], t, fps)
                        except Exception as ex:  # noqa: BLE001
                            log.warning("  解碼失敗 %s ep%d f%d：%s", cam, e["episode_index"], f, ex)
                            img, ft = None, None
                        if img is not None:
                            grid[(cam, f)] = img
                            res["decode"]["ok"] += 1
                            res["decode"]["max_err_s"] = max(res["decode"]["max_err_s"], abs(ft - t))
                head = f"{kind} ep{e['episode_index']:02d}  len={e['length']} ({e['length'] / fps:.1f}s)"
                sub = ""
                if fi:
                    head += (f"  phase={fi['phase']} {fi['phase_name'] or '?'}  mode={fi['mode']}  "
                             f"induced={fi['induced']} recovery={fi['recovery']}  flags={','.join(fi['flags']) or '-'}")
                    sub = f"taus={fi['taus']}  \"{fi['text'][:150]}\""
                sheet = make_sheet([head, sub], pts, fps, grid)
                name = f"{kind}_ep{e['episode_index']:02d}.jpg"
                _write_jpg(out / "sheets" / name, sheet)
                rows.append({"episode_index": e["episode_index"], "length": e["length"], "sheet": name,
                             "points": pts, "failure": fi, "videos": e["videos"]})
        finally:
            reader.close()
        res["kinds"][kind] = {"repo": repo, "fps": fps, "episodes": rows, "has_phase": bool(phase),
                              "seconds": round(time.time() - t0, 1)}
        log.info("  %s：%d 段，拼圖完成（%.0f 秒）", kind, len(rows), time.time() - t0)
    write_report(res, out)
    return res


def summarize_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fis = [r["failure"] for r in rows if r.get("failure")]
    by_phase: dict[str, int] = {}
    flags: dict[str, int] = {}
    for fi in fis:
        key = f"{fi['phase']} {fi['phase_name'] or '?'} / {fi['mode']}"
        by_phase[key] = by_phase.get(key, 0) + 1
        for fl in fi["flags"]:
            flags[fl] = flags.get(fl, 0) + 1
    place = [fi for fi in fis if fi["place"]]
    clean = [fi for fi in place if not set(fi["flags"]) & {"bad_order", "lucky", "bad_phase"}]
    return {"n": len(fis), "by_phase": dict(sorted(by_phase.items())), "flags": flags,
            "place": len(place), "place_clean": len(clean)}


def write_report(res: dict[str, Any], out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    rec = res["kinds"].get("recovery", {})
    exp = res["kinds"].get("expert", {})
    s = summarize_failures(rec.get("episodes", []))
    d = res["decode"]
    L = ["# 第 13 包：REBOOT 16 mm 圓柱（開發集）檢查", "", f"產生時間：{res['generated']}", "",
         "## 摘要", "",
         f"- 救援段 {len(rec.get('episodes', []))} 段（有 phase.json：{'是' if rec.get('has_phase') else '否'}）；"
         f"專家示範 {len(exp.get('episodes', []))} 段（有 phase.json：{'是' if exp.get('has_phase') else '否'}）",
         f"- 解碼：要 {d['asked']} 格，成功 {d['ok']} 格；取到的格和要的時間最多差 {d['max_err_s'] * 1000:.0f} 毫秒",
         f"- 失敗發生在放置階段（Align/Engage place）：{s['place']} 段；去掉筆誤和 lucky success 後 {s['place_clean']} 段",
         "", "## 失敗標註（階段 / 模式）", "", "| 階段 / 模式 | 段數 |", "|---|---|"]
    L += [f"| {k} | {n} |" for k, n in s["by_phase"].items()]
    L += ["", "旗標：" + ("、".join(f"{k} {n}" for k, n in s["flags"].items()) or "無"),
          "（bad_order＝救援開始不晚於失敗開始；lucky＝描述寫 lucky success；bad_phase＝階段編號不在 1～5；"
          "phase_out＝失敗影格不在標的階段內）", "", "## 放置階段的失敗段", "",
          "| 段 | 階段 | 模式 | 失敗開始 | 救援開始 | 旗標 | 描述 |", "|---|---|---|---|---|---|---|"]
    for r in rec.get("episodes", []):
        fi = r.get("failure")
        if fi and fi["place"]:
            L.append(f"| {r['episode_index']:02d} | {fi['phase']} | {fi['mode']} | {fi['induced']} | {fi['recovery']} | "
                     f"{','.join(fi['flags']) or '–'} | {fi['text'][:80]} |")
    L += ["", "拼圖：`experiments/p13_cyl_check/sheets/`（recovery_epNN.jpg、expert_epNN.jpg；3 台相機 × 最多 8 格）", ""]
    md = out / "report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "index.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return str(md)


def disk_ok(dest_root: str | Path, need: int, margin: int = 2 * 10**9) -> tuple[bool, int]:
    p = Path(dest_root)
    while not p.exists():
        p = p.parent
    free = shutil.disk_usage(p).free
    return free >= need + margin, free
