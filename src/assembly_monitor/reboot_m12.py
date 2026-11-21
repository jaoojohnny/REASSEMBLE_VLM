"""第 16 包：把 bolt 4 的最終模型（models/perpart_final/bolt_4）直接套到外部資料 REBOOT 的 M12 螺絲件，不重新訓練。

資料：REBOOT26/M12-fastener_install（專家示範＝成功）、REBOOT26/m12_recovery_install（人為失敗再救回＝失敗）。
- 只下載 README、meta/、俯視 cam_high＋做事那隻手的手腕相機彩色影片；data/（深度＋關節）只部分讀取夾爪欄位。
- 成功段的影格：專家「最後一次放開」後 1 秒（和 bolt 4 的「段結束後 1 秒」對應），手腕相機。
- 失敗段的影格：REBOOT 沒公開失敗時間點 → Claude 看每 2 秒一張的拼圖，標「失敗看得出來、救援還沒開始」的那一格；
  標完存檔（failure_annotations.json，blind_to_model＝true）之後才套模型。
- 模型和門檻都不改：失敗機率 > 0.5 判失敗。另外報 AUC（不看門檻、只看排序）。

兩個步驟：prepare（下載、夾爪、拼圖）→ Claude 標註 → apply（取影格、DINOv2、套模型、計分）。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import reboot_data as rd
from . import reboot_seg as rs
from .reboot_meta import hf_downloader, hf_lister, local_name

log = logging.getLogger(__name__)

REPOS = {"expert": "REBOOT26/M12-fastener_install", "recovery": "REBOOT26/m12_recovery_install"}
WRIST = {"left": "cam_left_wrist", "right": "cam_right_wrist"}
DEST = "data/raw/reboot"
OUT = Path("experiments/p16_m12")
FRAMES = Path("data/derived/frames/reboot_m12")
ANN_NAME = "failure_annotations.json"
MODEL = "models/perpart_final/bolt_4"
MAX_GB = 5.0
STEP_S = 2.0
AFTER_RELEASE_S = 1.0
COLS = 8
CELL_W, CELL_H, LABEL_H, STRIP_H, TITLE_H = 240, 180, 20, 90, 34


class Stop(RuntimeError):
    """停止且不繼續（超過下載上限、磁碟不足、標註不完整……）。"""


def local_dir(dest_root: str | Path, kind: str) -> Path:
    return Path(dest_root) / local_name(REPOS[kind])


# ---------------------------------------------------------------- 下載


def meta_files(entries: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return sorted((p, s) for p, s in entries if p == "README.md" or p.startswith("meta/"))


def video_files(entries: list[tuple[str, int]], cams: tuple[str, ...]) -> list[tuple[str, int]]:
    pre = tuple(f"videos/observation.images.{c}/" for c in cams)
    return sorted((p, s) for p, s in entries if p.startswith(pre))


def missing(files: list[tuple[str, int]], local: Path) -> list[tuple[str, int]]:
    return [(p, s) for p, s in files if not ((local / p).exists() and (local / p).stat().st_size == s)]


def fetch_files(repo: str, files: list[tuple[str, int]], local: Path,
                downloader: Callable[[str, str, Path], Any] = hf_downloader) -> dict[str, Any]:
    st: dict[str, Any] = {"files": 0, "bytes": 0, "failed": []}
    todo = missing(files, local)
    for i, (f, size) in enumerate(todo, 1):
        log.info("  [%d/%d] %s（%.0f MB）", i, len(todo), f, size / 1e6)
        try:
            downloader(repo, f, local)
            st["files"] += 1
            st["bytes"] += size
        except Exception as e:  # noqa: BLE001
            st["failed"].append(f"{repo}/{f}: {type(e).__name__}: {e}"[:300])
            log.warning("    下載失敗：%s", e)
    return st


# ---------------------------------------------------------------- 夾爪與取樣


def sample_points(length: int, fps: float, step_s: float = STEP_S) -> list[int]:
    """每 step_s 秒一格（從第 0 格開始）；最後一格和上一格差超過半秒就補上最後一格。"""
    step = max(1, int(round(step_s * fps)))
    pts = list(range(0, max(length, 1), step))
    if length - 1 - pts[-1] > 0.5 * fps:
        pts.append(length - 1)
    return pts


def expert_frame(rel: dict[str, Any], length: int, fps: float) -> int | None:
    """專家最後一次放開（cross）後 1 秒那一格；超出結尾就用最後一格。找不到放開就回傳 None。"""
    if not rel.get("ok"):
        return None
    return int(min(int(rel["cross"]) + int(round(AFTER_RELEASE_S * fps)), length - 1))


def analyse(grip: dict[str, Any], kind: str) -> dict[str, dict[str, Any]]:
    """每段：做事的手臂（夾爪動得多的那隻）、長度；專家另找最後一次放開。"""
    fps = float(grip["fps"])
    out: dict[str, dict[str, Any]] = {}
    for ep, e in sorted(grip["episodes"].items(), key=lambda kv: int(kv[0])):
        r = rs.detect_episode(e, fps)
        d: dict[str, Any] = {"arm": r["arm"], "length": int(r["length"]), "range": r["range"]}
        if kind == "expert":
            d["release"] = {k: r.get(k) for k in ("ok", "why", "onset", "cross", "flags", "n_holds")}
            d["model_frame"] = expert_frame(r, d["length"], fps)
            d["excluded"] = None if d["model_frame"] is not None else f"找不到放開（{r.get('why')}）"
        else:
            d["points"] = sample_points(d["length"], fps)
        out[str(int(ep))] = d
    return out


def cams_needed(eps: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, ...]:
    arms = sorted({d["arm"] for kind in eps.values() for d in kind.values()})
    return ("cam_high",) + tuple(WRIST[a] for a in arms)


# ---------------------------------------------------------------- 拼圖（只畫 ASCII 字）


def _blank(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), np.uint8)


def _fit(img: Any, w: int, h: int) -> np.ndarray:
    import cv2

    if img is None:
        t = _blank(h, w)
        t[:] = (0, 0, 110)
        rd._put(t, "MISSING", (8, 30))
        return t
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def grip_strip(states: dict[str, list[float]], active: str, marks: list[int], width: int,
               height: int = STRIP_H) -> np.ndarray:
    """兩隻手臂的夾爪曲線（做事的手臂畫粗）＋取樣格的位置（編號＝拼圖上的 #）。"""
    import cv2

    img = _blank(height, width)
    img[:] = (30, 30, 30)
    n = max(len(states.get(active) or []), 1)
    allv = np.concatenate([np.asarray(states.get(a) or [], float) for a in ("left", "right")])
    lo, hi = (float(np.min(allv)), float(np.max(allv))) if len(allv) else (0.0, 1.0)  # 兩隻手臂同一個尺度
    for arm, color in (("left", (255, 160, 60)), ("right", (60, 160, 255))):
        g = np.asarray(states.get(arm) or [], float)
        if len(g) < 2:
            continue
        yy = (height - 14) - (g - lo) / ((hi - lo) or 1.0) * (height - 24)
        xx = np.arange(len(g)) * (width - 1) / max(len(g) - 1, 1)
        pts = np.stack([xx, yy], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, color, 3 if arm == active else 1, cv2.LINE_AA)
    for k, f in enumerate(marks):
        x = int(f * (width - 1) / max(n - 1, 1))
        cv2.line(img, (x, 0), (x, height - 12), (200, 200, 200), 1)
        rd._put(img, str(k), (min(x + 2, width - 20), height - 2), 0.35)
    rd._put(img, f"gripper position, both arms on one scale (thick = {active} arm; blue = left, "
                 f"orange = right); numbers = tile #", (6, 14), 0.38)
    return img


def recovery_sheet(title: str, points: list[int], fps: float, frames: dict[tuple[str, int], Any],
                   cams: tuple[str, str], states: dict[str, list[float]], active: str,
                   cols: int = COLS) -> np.ndarray:
    """失敗段的標註拼圖：每格上＝cam_high、下＝手腕；上方夾爪曲線。"""
    rows = (len(points) + cols - 1) // cols
    W = cols * CELL_W
    cell_h = LABEL_H + 2 * CELL_H
    sheet = _blank(TITLE_H + STRIP_H + rows * cell_h, W)
    rd._put(sheet, title, (6, 22), 0.5)
    sheet[TITLE_H:TITLE_H + STRIP_H] = grip_strip(states, active, points, W)
    for k, f in enumerate(points):
        r, c = divmod(k, cols)
        x0, y0 = c * CELL_W, TITLE_H + STRIP_H + r * cell_h
        sheet[y0:y0 + LABEL_H, x0:x0 + CELL_W - 2] = (90, 60, 0)
        rd._put(sheet, f"#{k} f{f} {f / fps:.1f}s", (x0 + 6, y0 + 15), 0.45)
        for j, cam in enumerate(cams):
            yy = y0 + LABEL_H + j * CELL_H
            sheet[yy:yy + CELL_H, x0:x0 + CELL_W] = _fit(frames.get((cam, f)), CELL_W, CELL_H)
        if c == 0:
            rd._put(sheet, cams[0], (x0 + 6, y0 + LABEL_H + 16), 0.4)
            rd._put(sheet, cams[1], (x0 + 6, y0 + LABEL_H + CELL_H + 16), 0.4)
    return sheet


def grid_sheet(title: str, items: list[tuple[str, Any]], cols: int = 10, w: int = 192, h: int = 144,
               box: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """很多段各一張（例如模型看的那一格）；box＝在圖上畫出裁切範圍（相對座標）。"""
    import cv2

    rows = max(1, (len(items) + cols - 1) // cols)
    sheet = _blank(TITLE_H + rows * (h + LABEL_H), cols * w)
    rd._put(sheet, title, (6, 22), 0.5)
    for k, (lab, img) in enumerate(items):
        r, c = divmod(k, cols)
        x0, y0 = c * w, TITLE_H + r * (h + LABEL_H)
        rd._put(sheet, lab, (x0 + 4, y0 + 14), 0.4)
        t = _fit(img, w, h)
        if box is not None and img is not None:
            cv2.rectangle(t, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)), (0, 200, 255), 1)
        sheet[y0 + LABEL_H:y0 + LABEL_H + h, x0:x0 + w] = t
    return sheet


def read_frames(reader: Any, vids: dict[str, dict[str, Any]], cams: tuple[str, ...], frames: list[int],
                fps: float) -> dict[tuple[str, int], Any]:
    out: dict[tuple[str, int], Any] = {}
    for cam in cams:
        v = vids.get(cam)
        if v is None:
            continue
        for f in frames:
            img, _ = reader.read_at(v["path"], v["from"] + f / fps, fps)
            out[(cam, f)] = img
    return out


# ---------------------------------------------------------------- 第一步：準備


def prepare(dest_root: str | Path = DEST, out: Path = OUT, max_gb: float = MAX_GB,
            lister: Callable[[str], list[tuple[str, int]]] = hf_lister,
            downloader: Callable[[str, str, Path], Any] = hf_downloader,
            opener: Callable[[str, str], Any] = rs.hf_open) -> dict[str, Any]:
    out = Path(out)
    t0 = time.time()
    res: dict[str, Any] = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repos": REPOS, "kinds": {}}
    listing = {kind: lister(repo) for kind, repo in REPOS.items()}
    log.info("1/4 說明檔（meta）……")
    for kind, repo in REPOS.items():
        st = fetch_files(repo, meta_files(listing[kind]), local_dir(dest_root, kind), downloader)
        if st["failed"]:
            raise Stop(f"{repo} 的說明檔下載失敗：{st['failed'][0]}")
    log.info("2/4 夾爪（只讀關節欄位，不讀深度）……")
    grips, eps = {}, {}
    for kind, repo in REPOS.items():
        grips[kind] = rs.fetch_gripper(repo, local_dir(dest_root, kind), opener)
        eps[kind] = analyse(grips[kind], kind)
    fps = float(grips["expert"]["fps"])
    cams = cams_needed(eps)
    res.update(fps=fps, cams=list(cams))
    plans = {kind: video_files(listing[kind], cams) for kind in REPOS}
    todo = sum(s for kind in REPOS for _, s in missing(plans[kind], local_dir(dest_root, kind)))
    total = sum(s for kind in REPOS for _, s in plans[kind])
    res["download"] = {"cams": list(cams), "total_gb": round(total / 1e9, 2), "todo_gb": round(todo / 1e9, 2)}
    log.info("3/4 影片：%s，合計 %.2f GB（還要下載 %.2f GB）", "、".join(cams), total / 1e9, todo / 1e9)
    if total > max_gb * 1e9:
        raise Stop(f"影片合計 {total / 1e9:.2f} GB，超過上限 {max_gb:.1f} GB；沒有下載影片")
    ok, free = rd.disk_ok(dest_root, todo)
    if not ok:
        raise Stop(f"磁碟空間不足：剩 {free / 1e9:.1f} GB，需要 {todo / 1e9:.1f} GB＋2 GB")
    for kind, repo in REPOS.items():
        st = fetch_files(repo, plans[kind], local_dir(dest_root, kind), downloader)
        if st["failed"]:
            raise Stop(f"{repo} 的影片下載失敗：{st['failed'][0]}")
    log.info("4/4 拼圖……")
    sheets = out / "sheets"
    for kind in REPOS:
        local = local_dir(dest_root, kind)
        vids = {e["episode_index"]: e["videos"] for e in rd.read_episodes(local, cams)}
        reader = rd.FrameReader(local)
        try:
            if kind == "recovery":
                for ep, d in eps[kind].items():
                    wcam = WRIST[d["arm"]]
                    g = grips[kind]["episodes"][ep]
                    fr = read_frames(reader, vids.get(int(ep), {}), ("cam_high", wcam), d["points"], fps)
                    d["decoded"] = sum(v is not None for v in fr.values())
                    img = recovery_sheet(f"M12 recovery ep {ep} | {d['length'] / fps:.1f}s | {d['arm']} arm | every "
                                         f"{STEP_S:g}s", d["points"], fps, fr, ("cam_high", wcam),
                                         {"left": g["state_left"], "right": g["state_right"]}, d["arm"])
                    p = sheets / f"recovery_ep{int(ep):03d}.jpg"
                    rd._write_jpg(p, img)
                    d["sheet"] = str(p)
            else:
                items = []
                for ep, d in eps[kind].items():
                    f = d["model_frame"]
                    img = None
                    if f is not None:
                        img = read_frames(reader, vids.get(int(ep), {}), (WRIST[d["arm"]],), [f], fps).get((WRIST[d["arm"]], f))
                    d["decoded"] = img is not None
                    items.append((f"ep{ep} {'' if f is not None else 'NO RELEASE'}", img))
                rd._write_jpg(out / "expert_model_frames.jpg",
                              grid_sheet("M12 expert: wrist frame 1 s after the last release (model input)", items,
                                         box=(0.2, 0.2, 0.8, 0.8)))
        finally:
            reader.close()
    res["kinds"] = {kind: {"repo": REPOS[kind], "n": len(eps[kind]), "episodes": eps[kind]} for kind in REPOS}
    res["seconds"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prepare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_prepare_report(res, out / "prepare_report.md")
    return res


def write_prepare_report(res: dict[str, Any], path: Path) -> None:
    ex, rc = res["kinds"]["expert"], res["kinds"]["recovery"]
    rel_ok = sum(d["model_frame"] is not None for d in ex["episodes"].values())
    flags: dict[str, int] = {}
    for d in ex["episodes"].values():
        for f in d["release"].get("flags") or []:
            flags[f] = flags.get(f, 0) + 1
    arms: dict[str, int] = {}
    for k in (ex, rc):
        for d in k["episodes"].values():
            arms[d["arm"]] = arms.get(d["arm"], 0) + 1
    fps = res["fps"]
    L = ["# 第 16 包（準備）：REBOOT M12 螺絲件", "", f"產生時間：{res['generated']}（{res['seconds']:.0f} 秒）", "",
         f"- 專家示範（成功）`{ex['repo']}`：{ex['n']} 段；找到最後一次放開 {rel_ok} 段" +
         (f"；旗標 {flags}" if flags else ""),
         f"- 失敗再救回 `{rc['repo']}`：{rc['n']} 段；每 {STEP_S:g} 秒一格做成標註拼圖（`sheets/`）",
         f"- 做事的手臂：{arms}；下載的相機：{'、'.join(res['cams'])}；影片合計 {res['download']['total_gb']} GB",
         f"- 每段長度：成功 {_len_range(ex, fps)}、失敗 {_len_range(rc, fps)}", "",
         "## 下一步", "",
         "- Claude 看 `sheets/recovery_ep*.jpg`，標每段「失敗看得出來、救援還沒開始」的那一格，存成 "
         f"`{ANN_NAME}`（標完才套模型）。",
         "- `expert_model_frames.jpg`：成功段模型會看的那一格（最後一次放開後 1 秒，手腕相機；黃框＝裁切範圍）。"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _len_range(k: dict[str, Any], fps: float) -> str:
    ls = [d["length"] / fps for d in k["episodes"].values()]
    return f"{min(ls):.1f}～{max(ls):.1f} 秒" if ls else "–"


# ---------------------------------------------------------------- 標註


def load_annotations(path: Path, prep: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """檢查標註：完整、標註時沒看模型答案、每段的格子是拼圖上的取樣格。回傳可用的段。"""
    if not path.exists():
        raise Stop(f"找不到 {path}：要先讓 Claude 標失敗時刻")
    a = json.loads(path.read_text(encoding="utf-8"))
    if not a.get("complete") or not a.get("blind_to_model"):
        raise Stop(f"{path} 沒有標成 complete＝true、blind_to_model＝true")
    eps = prep["kinds"]["recovery"]["episodes"]
    got = a.get("episodes") or {}
    if set(got) != set(eps):
        raise Stop(f"標註的段和資料不一致：少 {sorted(set(eps) - set(got))[:5]}、多 {sorted(set(got) - set(eps))[:5]}")
    out = {}
    for ep, d in got.items():
        if not d.get("usable"):
            if not d.get("note"):
                raise Stop(f"第 {ep} 段標成不能用，但沒寫原因")
            continue
        if int(d["frame"]) not in eps[ep]["points"]:
            raise Stop(f"第 {ep} 段標的格子 {d['frame']} 不是拼圖上的取樣格")
        out[int(ep)] = d
    return out


# ---------------------------------------------------------------- 第二步：套模型


def auc(p_fail: list[float], p_ok: list[float]) -> float | None:
    """失敗段的失敗機率高於成功段的比例（同分算一半）＝ AUC；0.5＝分不出來。"""
    if not p_fail or not p_ok:
        return None
    a, b = np.asarray(p_fail)[:, None], np.asarray(p_ok)[None, :]
    return float(np.mean((a > b) + 0.5 * (a == b)))


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.array([r["label"] for r in rows])
    p = np.array([r["p_failure"] for r in rows])
    pred = (p > 0.5).astype(int)
    n_f, n_s = int((y == 1).sum()), int((y == 0).sum())
    ok_f, ok_s = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 0)).sum())
    bal = np.mean([ok_f / n_f if n_f else np.nan, ok_s / n_s if n_s else np.nan])
    return {"n_failure": n_f, "n_success": n_s, "ok_failure": ok_f, "ok_success": ok_s, "bal": float(bal),
            "auc": auc(p[y == 1].tolist(), p[y == 0].tolist()),
            "median_p_failure": {"失敗段": float(np.median(p[y == 1])) if n_f else None,
                                 "成功段": float(np.median(p[y == 0])) if n_s else None}}


def apply(dest_root: str | Path = DEST, out: Path = OUT, frames_root: Path = FRAMES, model_stem: str = MODEL,
          encoder: Any | None = None) -> dict[str, Any]:
    from . import perpart as pp
    from . import perpart_final as pf

    out = Path(out)
    prep = json.loads((out / "prepare.json").read_text(encoding="utf-8"))
    ann = load_annotations(out / ANN_NAME, prep)
    model = pf.load_model(model_stem)
    fps, cams = float(prep["fps"]), tuple(prep["cams"])
    own = encoder is None
    enc = pp.make_encoders()["crop"] if own else encoder
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    t0 = time.time()
    try:
        for kind in ("expert", "recovery"):
            local = local_dir(dest_root, kind)
            vids = {e["episode_index"]: e["videos"] for e in rd.read_episodes(local, cams)}
            reader = rd.FrameReader(local)
            try:
                for ep, d in sorted(prep["kinds"][kind]["episodes"].items(), key=lambda kv: int(kv[0])):
                    if kind == "expert":
                        f = d.get("model_frame")
                        if f is None:
                            excluded.append({"kind": kind, "episode": int(ep), "why": d.get("excluded")})
                            continue
                    else:
                        if int(ep) not in ann:
                            excluded.append({"kind": kind, "episode": int(ep),
                                             "why": "標註不能用：" + str(prep_note(out, ep))})
                            continue
                        f = int(ann[int(ep)]["frame"])
                    cam = WRIST[d["arm"]]
                    img = read_frames(reader, vids.get(int(ep), {}), (cam,), [f], fps).get((cam, f))
                    if img is None:
                        excluded.append({"kind": kind, "episode": int(ep), "why": "影格讀不到"})
                        continue
                    p = Path(frames_root) / kind / f"ep{int(ep):03d}_f{f}.jpg"
                    rd._write_jpg(p, img, q=95)
                    pf_ = float(pf.predict_proba(model, pf.image_vector(enc, str(p))[None])[0])
                    rows.append({"kind": kind, "episode": int(ep), "label": 1 if kind == "recovery" else 0,
                                 "frame": f, "t": round(f / fps, 2), "arm": d["arm"], "camera": cam,
                                 "p_failure": pf_, "pred": "失敗" if pf_ > 0.5 else "成功", "image": str(p),
                                 **({"type": ann[int(ep)].get("type"), "in_gripper": ann[int(ep)].get("bolt_in_gripper")}
                                    if kind == "recovery" else {})})
            finally:
                reader.close()
    finally:
        if own:
            enc.close()
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": model_stem, "model_part": model.get("part"),
           "model_cv": model.get("cv"), "score": score(rows) if rows else None, "rows": rows, "excluded": excluded,
           "seconds": round(time.time() - t0, 1)}
    (out / "apply.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    import cv2

    items = [(f"{'F' if r['label'] else 'S'} ep{r['episode']} p={r['p_failure']:.2f}", cv2.imread(r["image"]))
             for r in sorted(rows, key=lambda r: (-r["label"], r["episode"]))]
    rd._write_jpg(out / "model_inputs.jpg", grid_sheet("model inputs (F = failure, S = success; p = P(failure))",
                                                       items, box=(0.2, 0.2, 0.8, 0.8)))
    write_apply_report(res, out / "report.md")
    return res


def prep_note(out: Path, ep: str) -> str:
    try:
        a = json.loads((Path(out) / ANN_NAME).read_text(encoding="utf-8"))
        return str(a["episodes"][ep].get("note") or "")
    except Exception:  # noqa: BLE001
        return ""


def write_apply_report(res: dict[str, Any], path: Path) -> None:
    s, cv = res["score"], res.get("model_cv") or {}
    L = ["# 第 16 包：bolt 4 模型直接套到 REBOOT M12 螺絲件", "",
         f"產生時間：{res['generated']}；模型 `{res['model']}`（{res['model_part']}，F2S2，沒有重新訓練、門檻 0.5 不改）。", ""]
    if not s:
        L.append("沒有可計分的段。")
    else:
        L += ["## 結果", "",
              "| | 失敗段答對 | 成功段答對 | 兩類平均 | AUC |", "|---|---|---|---|---|",
              f"| bolt 4 模型 → M12 | {s['ok_failure']}/{s['n_failure']} | {s['ok_success']}/{s['n_success']} | "
              f"{s['bal'] * 100:.0f}% | {s['auc']:.2f} |" if s["auc"] is not None else
              f"| bolt 4 模型 → M12 | {s['ok_failure']}/{s['n_failure']} | {s['ok_success']}/{s['n_success']} | "
              f"{s['bal'] * 100:.0f}% | – |",
              f"| 一律答成功（基準） | 0/{s['n_failure']} | {s['n_success']}/{s['n_success']} | 50% | 0.50 |", "",
              f"- 失敗機率中位數：失敗段 {_f(s['median_p_failure']['失敗段'])}、成功段 {_f(s['median_p_failure']['成功段'])}",
              f"- 參考：同一個模型在 REASSEMBLE bolt 4 的交叉驗證 {cv.get('bal_mean', float('nan')) * 100:.0f}%" if cv else "",
              "- AUC＝隨便挑一段失敗、一段成功，模型給失敗段較高失敗機率的比例；0.5＝分不出來，1＝完全分得開（不受門檻影響）。", ""]
    L += ["## 每段", "", "| 段 | 答案 | 影格（秒） | 失敗機率 | 模型 | 失敗型態（Claude 標） |", "|---|---|---|---|---|---|"]
    for r in sorted(res["rows"], key=lambda r: (-r["label"], r["episode"])):
        L.append(f"| {'失敗' if r['label'] else '成功'} ep{r['episode']} | {'失敗' if r['label'] else '成功'} | "
                 f"{r['t']} | {r['p_failure']:.3f} | {r['pred']} | {r.get('type') or ''} |")
    if res["excluded"]:
        L += ["", "## 沒有計分的段", ""] + [f"- {e['kind']} ep{e['episode']}：{e['why']}" for e in res["excluded"]]
    L += ["", "## 注意", "",
          "- 換了機器人（WidowX）、相機、光線和零件，模型只靠 DINOv2 的 2 個數字；這裡測的主要是「換環境還準不準」。",
          "- 成功段的影格在放開後、失敗段在救援前（多半還夾著），所以就算分數好，也可能只是在分夾爪有沒有放開。",
          "- 失敗時刻是 Claude 看拼圖標的（每 2 秒一格），標的時候還沒有模型答案。",
          "- 圖：`model_inputs.jpg`（模型看的每一格，黃框＝裁切範圍）。"]
    path.write_text("\n".join(x for x in L if x is not None) + "\n", encoding="utf-8")


def _f(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"
