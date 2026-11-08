"""第 13 包第三步：REBOOT 16 mm 圓柱（開發集）——用夾爪開合找專家示範「放開圓柱」的時刻，切出成功段與失敗段。

- 夾爪資料在 data/*.parquet，和深度影像放在同一個檔（每檔約 550 MB）。這裡用部分讀取：
  先讀檔尾的欄位目錄，只抓 observation.state、action、episode_index、frame_index、timestamp 這幾欄的位元組，
  深度欄位完全不讀。讀之前先算好要抓多少位元組，超過上限就停。
- 成功段（專家示範）：結尾＝夾爪開始放開的那一格（onset）；失敗段（救援段，官方標註）：結尾＝救援開始的那一格。
  兩邊都停在「插入動作剛結束、夾爪還在孔旁」，長度都固定 3 秒，避免用手臂位置或段長就能分出答案。
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import reboot_data as rd
from .reboot_meta import local_name
from .util import now_iso

log = logging.getLogger(__name__)

COLUMNS = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
GRIP = {"left": "left_left_carriage_joint.pos", "right": "right_left_carriage_joint.pos"}
FILE_CAP = 20 * 2**20       # 一個檔最多讀 20 MB（只讀關節欄位，正常應遠小於此）
TOTAL_CAP = 300 * 2**20     # 全部最多 300 MB
MIN_RANGE = 1e-3            # 夾爪幾乎不動的門檻（關節單位）
WIN_S = 3.0                 # 每段固定 3 秒
SHEET_OFFSETS = (-3.0, -2.0, -1.0, 0.0, 0.5, 1.5)


class TooBig(RuntimeError):
    pass


class CountingFile(io.RawIOBase):
    """包住檔案物件，記錄實際讀了多少位元組。"""

    def __init__(self, f: Any):
        self.f, self.n = f, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, off: int, whence: int = 0) -> int:
        return self.f.seek(off, whence)

    def tell(self) -> int:
        return self.f.tell()

    def read(self, size: int = -1) -> bytes:
        b = self.f.read(size)
        self.n += len(b)
        return b

    def readinto(self, buf: Any) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)


def planned_bytes(md: Any, columns: list[str]) -> int:
    """依 parquet 檔尾目錄，算出只讀這些欄位要抓的位元組。"""
    total = 0
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        for j in range(rg.num_columns):
            c = rg.column(j)
            path = c.path_in_schema
            if any(path == col or path.startswith(col + ".") for col in columns):
                total += c.total_compressed_size
    return total


def read_columns(fobj: Any, columns: list[str] = COLUMNS, cap: int = FILE_CAP) -> tuple[Any, int, int]:
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    cf = CountingFile(fobj)
    pf = pq.ParquetFile(cf)
    need = planned_bytes(pf.metadata, columns)
    if need > cap:
        raise TooBig(f"要讀 {need / 2**20:.1f} MB，超過單檔上限 {cap / 2**20:.0f} MB")
    t = pf.read(columns=columns)
    return t, cf.n, need


def hf_open(repo: str, path: str, block_size: int = 1 << 16) -> Any:
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(f"datasets/{repo}/{path}", "rb", block_size=block_size)


def data_files(local: Path) -> list[str]:
    """episodes 表裡列的 data 檔（不重複）。"""
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    info = json.loads((local / "meta" / "info.json").read_text(encoding="utf-8"))
    tmpl = info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    out: set[str] = set()
    for p in sorted((local / "meta" / "episodes").rglob("*.parquet")):
        for r in pq.read_table(p, columns=["data/chunk_index", "data/file_index"]).to_pylist():
            out.add(tmpl.format(chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])))
    return sorted(out)


def grip_index(info: dict[str, Any]) -> dict[str, int]:
    names = info["features"]["observation.state"]["names"]
    return {arm: names.index(n) for arm, n in GRIP.items()}


def fetch_gripper(repo: str, local: Path, opener: Callable[[str, str], Any] = hf_open,
                  total_cap: int = TOTAL_CAP, retries: int = 3) -> dict[str, Any]:
    """讀每段的夾爪（state 與 action），存到 local/gripper.json；已存在就直接讀。"""
    out_p = local / "gripper.json"
    if out_p.exists():
        return json.loads(out_p.read_text(encoding="utf-8"))
    info = json.loads((local / "meta" / "info.json").read_text(encoding="utf-8"))
    gi = grip_index(info)
    eps: dict[str, dict[str, list]] = {}
    stats = {"files": 0, "bytes_read": 0, "bytes_planned": 0, "seconds": 0.0}
    t0 = time.time()
    files = data_files(local)
    for k, path in enumerate(files, 1):
        for attempt in range(retries):
            try:
                with opener(repo, path) as f:
                    t, nread, need = read_columns(f)
                break
            except TooBig:
                raise
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise
                log.warning("  %s 讀取失敗（%s），重試……", path, e)
                time.sleep(3 * (attempt + 1))
        stats["files"] += 1
        stats["bytes_read"] += nread
        stats["bytes_planned"] += need
        if stats["bytes_read"] > total_cap:
            raise TooBig(f"累計讀了 {stats['bytes_read'] / 2**20:.0f} MB，超過總上限 {total_cap / 2**20:.0f} MB")
        d = t.to_pydict()
        for i in range(len(d["episode_index"])):
            e = eps.setdefault(str(int(d["episode_index"][i])), {"frame_index": [], "timestamp": [], "state_left": [],
                                                                 "state_right": [], "action_left": [], "action_right": []})
            st, ac = d["observation.state"][i], d["action"][i]
            e["frame_index"].append(int(d["frame_index"][i]))
            e["timestamp"].append(float(d["timestamp"][i]))
            for arm in ("left", "right"):
                e[f"state_{arm}"].append(float(st[gi[arm]]))
                e[f"action_{arm}"].append(float(ac[gi[arm]]))
        log.info("  [%d/%d] %s：讀 %.2f MB（%d 列）", k, len(files), path, nread / 2**20, len(d["episode_index"]))
    for e in eps.values():  # 依影格排序
        order = np.argsort(e["frame_index"])
        for key in list(e):
            e[key] = [e[key][i] for i in order]
    stats["seconds"] = round(time.time() - t0, 1)
    res = {"repo": repo, "fps": info["fps"], "stats": stats, "episodes": eps}
    out_p.write_text(json.dumps(res), encoding="utf-8")
    return res


# ---------------------------------------------------------------- 找放開的時刻


def _median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def _runs(b: np.ndarray) -> list[tuple[int, int, bool]]:
    runs, s = [], 0
    for i in range(1, len(b) + 1):
        if i == len(b) or b[i] != b[s]:
            runs.append((s, i, bool(b[s])))
            s = i
    return runs


def _drop_short(b: np.ndarray, k: int) -> np.ndarray:
    b = b.copy()
    for _ in range(5):
        runs = _runs(b)
        short = [r for r in runs[1:-1] if r[1] - r[0] < k]
        if not short:
            break
        for s, e, v in short:
            b[s:e] = not v
    return b


def detect_release(g: list[float] | np.ndarray, fps: float, min_hold_s: float = 1.0) -> dict[str, Any]:
    """夾爪位置序列 → 最後一次「夾住 → 放開」的時刻。

    夾住＝和結尾狀態相反、持續至少 1 秒的那段；cross＝放開後第一格；onset＝夾爪開始動（走完 10%）的那一格。
    """
    g = np.asarray(g, dtype=float)
    n = len(g)
    res: dict[str, Any] = {"ok": False, "flags": []}
    if n < 2 * fps:
        res["why"] = "too_short"
        return res
    sm = _median_filter(g)
    lo, hi = float(np.percentile(sm, 5)), float(np.percentile(sm, 95))
    res.update(lo=lo, hi=hi)
    if hi - lo < MIN_RANGE:
        res["why"] = "no_motion"
        return res
    b = _drop_short(sm > (lo + hi) / 2, max(3, int(0.2 * fps)))
    final = bool(b[-int(fps):].mean() > 0.5)
    cand = [r for r in _runs(b) if r[2] != final and r[1] - r[0] >= min_hold_s * fps]
    # 從第 0 格開始的那段是手臂在原位的狀態（REBOOT 原位時夾爪是閉合的），不算夾住
    holds = [r for r in cand if r[0] > 0]
    if not holds:
        res["why"] = "no_hold" if not cand else "only_initial_state"
        return res
    h0, h1, _ = holds[-1]
    cross = h1
    if cross >= n:
        res["why"] = "no_release"
        return res
    hold_level = float(np.median(sm[h0:h1]))
    open_level = float(np.median(sm[cross:min(n, cross + int(fps))]))
    span = open_level - hold_level
    frac = (sm[h0:cross + 1] - hold_level) / span if span else np.zeros(cross + 1 - h0)
    idx = np.where(frac <= 0.1)[0]
    onset = h0 + int(idx[-1]) if len(idx) else cross
    if len(holds) > 1:
        res["flags"].append("multi_hold")
    if n - cross < 0.5 * fps:
        res["flags"].append("late")
    res.update(ok=True, onset=int(onset), cross=int(cross), hold=[int(h0), int(h1)], n_holds=len(holds),
               hold_level=hold_level, open_level=open_level, final_open=final)
    return res


def detect_episode(e: dict[str, list], fps: float) -> dict[str, Any]:
    """選動作較大的那隻手臂，找放開時刻。"""
    rng = {arm: float(np.ptp(e[f"state_{arm}"])) if e[f"state_{arm}"] else 0.0 for arm in ("left", "right")}
    arm = max(rng, key=rng.get)
    r = detect_release(e[f"state_{arm}"], fps)
    r.update(arm=arm, range=rng, length=len(e[f"state_{arm}"]))
    return r


# ---------------------------------------------------------------- 切段


def build_segments(releases: dict[int, dict[str, Any]], failures: dict[int, dict[str, Any]], fps: float,
                   win_s: float = WIN_S) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    win = int(round(win_s * fps))
    segs, excluded = [], []
    for ep, r in sorted(releases.items()):
        if not r.get("ok"):
            excluded.append({"id": f"exp{ep:02d}", "why": r.get("why", "?")})
            continue
        if set(r["flags"]) & {"multi_hold", "late"}:
            excluded.append({"id": f"exp{ep:02d}", "why": ",".join(r["flags"])})
            continue
        end = r["onset"]
        if end < win:
            excluded.append({"id": f"exp{ep:02d}", "why": "release_before_3s"})
            continue
        segs.append({"id": f"exp{ep:02d}", "source": "expert", "label": "success", "episode": ep,
                     "start_frame": end - win, "end_frame": end, "arm": r["arm"]})
    for ep, fi in sorted(failures.items()):
        if not fi["place"]:
            continue
        bad = set(fi["flags"]) & {"bad_order", "lucky", "bad_phase"}
        if bad:
            excluded.append({"id": f"rec{ep:02d}", "why": ",".join(sorted(bad))})
            continue
        end = int(fi["recovery"])
        segs.append({"id": f"rec{ep:02d}", "source": "recovery", "label": "failure", "episode": ep,
                     "start_frame": max(0, end - win), "end_frame": end, "arm": "right",
                     "mode": fi["mode"], "phase": fi["phase"]})
    return segs, excluded


# ---------------------------------------------------------------- 圖


def gripper_plot(gr: dict[str, Any], releases: dict[int, dict[str, Any]], path: Path, cols: int = 6) -> None:
    import cv2

    W, H = 360, 120
    eps = sorted(int(k) for k in gr["episodes"])
    rows = (len(eps) + cols - 1) // cols
    img = np.zeros((rows * H, cols * W, 3), np.uint8)
    fps = float(gr["fps"])
    for n, ep in enumerate(eps):
        e, r = gr["episodes"][str(ep)], releases[ep]
        x0, y0 = (n % cols) * W, (n // cols) * H
        cv2.rectangle(img, (x0, y0), (x0 + W - 2, y0 + H - 2), (40, 40, 40), -1)
        L = len(e["state_right"])
        for arm, color in (("left", (120, 120, 120)), ("right", (255, 255, 255))):
            s = np.asarray(e[f"state_{arm}"], float)
            if not len(s):
                continue
            lo, hi = float(s.min()), float(s.max())
            yy = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
            pts = np.stack([x0 + 4 + np.arange(len(s)) * (W - 10) / max(L - 1, 1),
                            y0 + H - 22 - yy * (H - 40)], 1).astype(np.int32)
            cv2.polylines(img, [pts], False, color if arm == r.get("arm") else (90, 90, 90), 1, cv2.LINE_AA)
        if r.get("ok"):
            for f, c in ((r["onset"], (0, 220, 0)), (r["cross"], (0, 220, 255))):
                x = int(x0 + 4 + f * (W - 10) / max(L - 1, 1))
                cv2.line(img, (x, y0 + 14), (x, y0 + H - 18), c, 1)
        txt = (f"ep{ep:02d} {r.get('arm', '?')[0].upper()} onset {r['onset'] / fps:.1f}s {','.join(r['flags'])}"
               if r.get("ok") else f"ep{ep:02d} FAIL {r.get('why')}")
        rd._put(img, txt, (x0 + 6, y0 + 14), 0.4, (255, 255, 255) if r.get("ok") else (80, 80, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    rd._write_jpg(path, img, 88)


def segment_sheets(segs: list[dict[str, Any]], dest_root: Path, out: Path, fps: float) -> dict[str, int]:
    """每段一張：手臂那隻手腕＋俯視，結尾前 3、2、1 秒、結尾、結尾後 0.5、1.5 秒。"""
    stats = {"asked": 0, "ok": 0}
    videos: dict[str, dict[int, dict[str, Any]]] = {}
    for kind, repo in rd.CYL.items():
        local = dest_root / local_name(repo)
        videos[kind] = {e["episode_index"]: e for e in rd.read_episodes(local)}
    readers = {kind: rd.FrameReader(dest_root / local_name(repo)) for kind, repo in rd.CYL.items()}
    try:
        for s in segs:
            kind = "expert" if s["source"] == "expert" else "recovery"
            e = videos[kind][s["episode"]]
            cams = (f"cam_{s['arm']}_wrist", "cam_high")
            pts = rd._clamp_points([(s["end_frame"] + round(o * fps), f"end{o:+g}s") for o in SHEET_OFFSETS],
                                   e["length"])
            grid = {}
            for cam in cams:
                v = e["videos"][cam]
                for f, _ in pts:
                    stats["asked"] += 1
                    img, _ft = readers[kind].read_at(v["path"], v["from"] + f / fps, fps)
                    if img is not None:
                        grid[(cam, f)] = img
                        stats["ok"] += 1
            head = f"{s['id']}  label={s['label']}  end=f{s['end_frame']} ({s['end_frame'] / fps:.1f}s)  arm={s['arm']}"
            if s.get("mode"):
                head += f"  mode={s['mode']}  phase={s['phase']}"
            sheet = rd.make_sheet([head, "window = [end-3s, end]; columns after 'end+0' are outside the segment"],
                                  pts, fps, grid, cams)
            rd._write_jpg(out / "sheets" / f"{s['label']}_{s['id']}.jpg", sheet)
    finally:
        for r in readers.values():
            r.close()
    return stats


def write_report(res: dict[str, Any], out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fps = res["fps"]
    rel = res["releases"]
    ok = [r for r in rel.values() if r.get("ok")]
    on = sorted(r["onset"] / fps for r in ok)
    segs, exc = res["segments"], res["excluded"]
    st = res["gripper_stats"]
    L = ["# 第 13 包第三步：REBOOT 16 mm 圓柱——找放開時刻、切段", "", f"產生時間：{res['generated']}", "",
         "## 摘要", "",
         f"- 夾爪資料：讀 {st['files']} 個 data 檔，只抓關節欄位，共 {st['bytes_read'] / 2**20:.1f} MB"
         f"（預估 {st['bytes_planned'] / 2**20:.1f} MB；深度沒讀）",
         f"- 專家示範 {len(rel)} 段：找到放開時刻 {len(ok)} 段" +
         (f"，onset {on[0]:.1f}～{on[-1]:.1f} 秒（中位 {on[len(on) // 2]:.1f}）" if on else ""),
         f"- 動作的手臂：" + "、".join(f"{a} {sum(1 for r in ok if r['arm'] == a)}" for a in ("right", "left")),
         f"- 切出：成功段 {sum(1 for s in segs if s['label'] == 'success')}、失敗段 {sum(1 for s in segs if s['label'] == 'failure')}"
         f"（每段固定 {WIN_S:g} 秒）；排除 {len(exc)} 段",
         f"- 檢查圖解碼：{res['sheet_stats']['ok']}/{res['sheet_stats']['asked']} 格", "",
         "## 排除", "", "| 段 | 原因 |", "|---|---|"]
    L += [f"| {x['id']} | {x['why']} |" for x in exc] or ["| – | – |"]
    L += ["", "## 失敗段", "", "| 段 | 模式 | 階段 | 結尾（救援開始） |", "|---|---|---|---|"]
    L += [f"| {s['id']} | {s['mode']} | {s['phase']} | f{s['end_frame']}（{s['end_frame'] / fps:.1f}s） |"
          for s in segs if s["label"] == "failure"]
    L += ["", "## 檔案", "", "- `gripper_plots.jpg`：每段專家示範的夾爪曲線（白＝動作的手臂，灰＝另一隻；綠線＝onset，黃線＝放開後第一格）",
          "- `sheets/`：每段一張（手腕＋俯視；end+0 之後的欄在段外）", "- `segments.json`：段的定義", ""]
    md = out / "report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    return str(md)


def run(dest_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    dest_root, out = Path(dest_root), Path(out_dir)
    exp_local = dest_root / local_name(rd.CYL["expert"])
    rec_local = dest_root / local_name(rd.CYL["recovery"])
    gr = fetch_gripper(rd.CYL["expert"], exp_local)
    fps = float(gr["fps"])
    releases = {int(ep): detect_episode(e, fps) for ep, e in gr["episodes"].items()}
    failures = rd.read_phase(rec_local)
    segs, excluded = build_segments(releases, failures, fps)
    gripper_plot(gr, releases, out / "gripper_plots.jpg")
    sheet_stats = segment_sheets(segs, dest_root, out, fps)
    res = {"generated": now_iso(), "fps": fps, "gripper_stats": gr["stats"], "releases": releases,
           "segments": segs, "excluded": excluded, "sheet_stats": sheet_stats}
    out.mkdir(parents=True, exist_ok=True)
    (out / "segments.json").write_text(json.dumps({"generated": res["generated"], "fps": fps, "window_s": WIN_S,
                                                   "segments": segs, "excluded": excluded},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "releases.json").write_text(json.dumps({str(k): v for k, v in releases.items()}, ensure_ascii=False,
                                                  indent=1), encoding="utf-8")
    write_report(res, out)
    return res
