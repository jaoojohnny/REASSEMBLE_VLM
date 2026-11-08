"""第 13 包第一步：外部插頭資料 REBOOT（NIST 組裝任務板）——只下載說明檔，檢查有沒有「失敗時間點」標註。

REBOOT 每個任務約 37 GB，其中約 34 GB 是 data/ 裡的深度資料，彩色影片約 2.8 GB。
這一步完全不碰 data/ 和 videos/，只抓：
  - README.md、meta/ 底下的小檔（info.json、stats.json、tasks.parquet、episodes/*.parquet）
  - data/、videos/ 以外任何 20 MB 以下的檔案（可能是另外放的標註檔）
並列出每個任務每台相機的影片大小，供下一步估算下載量。

要找的東西（論文寫有、但資料卡沒寫）：救援段的 t_f（失敗開始）、t_r（開始救援）、t_s（救回成功）、
五個階段的邊界、失敗模式（misalignment、slip、premature release、jamming、off-axis collision）。
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

from .util import now_iso

log = logging.getLogger(__name__)

ORG = "REBOOT26"
# 插頭類 4 種零件的「插入」任務：專家示範（成功）＋故意製造失敗再救回（救援段）
PARTS: dict[str, dict[str, Any]] = {
    "usb_a": {"part": "USB-A", "expert": ["REBOOT26/usbA-install"],
              "recovery": ["REBOOT26/USB-A_recovery_install"]},
    "rj45": {"part": "RJ45", "expert": ["REBOOT26/rj45-install"],
             "recovery": ["REBOOT26/rj45_recovery_install"]},
    "usb_c": {"part": "USB-C", "expert": ["REBOOT26/usb-c_install"],
              "recovery": ["REBOOT26/usbc_recovery_install", "REBOOT26/usbc_recovery_install2"]},
    "nema": {"part": "NEMA 1-15P", "expert": ["REBOOT26/nema1-15-plug_install"],
             "recovery": ["REBOOT26/nema1-15-plug_recovery_install"]},
}
SAMPLE = ["REBOOT26/sample_recovery-demonstration"]  # 官方示範用的救援段，可能附標註範例
KIND_ZH = {"expert": "專家示範", "recovery": "救援段", "sample": "官方範例"}
SMALL_MAX = 20 * 2**20
STD_FILES = {"README.md", ".gitattributes", "meta/info.json", "meta/stats.json", "meta/tasks.parquet",
             "meta/tasks.jsonl", "meta/episodes.jsonl", "meta/episodes_stats.jsonl"}
STD_FEATURES = {"action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"}
STD_EP_COLS = {"episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index", "__index_level_0__"}
STD_EP_PREFIX = ("data/", "videos/", "meta/", "stats/")
KEYWORDS = ("t_f", "t_r", "t_s", "onset", "recover", "fail", "phase", "jam", "misalign", "slip",
            "premature", "collision", "annotat", "engage", "transport")
TEXT_SUFFIX = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml"}
CAM_ZH = {"cam_high": "俯視", "cam_left_wrist": "左手腕", "cam_right_wrist": "右手腕", "cam_low": "低角度前視"}

Lister = Callable[[str], list[tuple[str, int]]]
Downloader = Callable[[str, str, Path], Any]


def repo_list(parts: Iterable[str] | None = None, sample: bool = True) -> list[tuple[str, str, str]]:
    """[(零件 key, 種類, repo_id)]；種類 = expert / recovery / sample。"""
    out: list[tuple[str, str, str]] = []
    for k in (list(parts) if parts else list(PARTS)):
        for kind in ("expert", "recovery"):
            out += [(k, kind, r) for r in PARTS[k][kind]]
    if sample:
        out += [("sample", "sample", r) for r in SAMPLE]
    return out


def local_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _video_key(path: str) -> str:
    parts = path.split("/")
    for p in parts[1:-1]:
        if p.startswith("observation"):
            return p
    return parts[1] if len(parts) > 2 else "?"


def classify_tree(entries: list[tuple[str, int]]) -> dict[str, Any]:
    """entries = [(路徑, 位元組)]（只含檔案）。分出要下載的小檔、非標準檔、各群組大小。"""
    small, extra, big_skip = [], [], []
    groups = {"data": 0, "videos": 0, "meta": 0, "other": 0}
    video_keys: dict[str, int] = {}
    for path, size in entries:
        top = path.split("/", 1)[0]
        if top == "videos":
            groups["videos"] += size
            k = _video_key(path)
            video_keys[k] = video_keys.get(k, 0) + size
            continue
        if top == "data":
            groups["data"] += size
            continue
        groups["meta" if top == "meta" else "other"] += size
        if not (path in STD_FILES or path.startswith("meta/episodes/")):
            extra.append(path)
        (small if size <= SMALL_MAX else big_skip).append(path)
    return {"small": sorted(small), "extra": sorted(extra), "big_skip": sorted(big_skip), "groups": groups,
            "video_keys": dict(sorted(video_keys.items())), "total": sum(s for _, s in entries)}


def _hit(k: str, low: str) -> bool:
    if k.startswith("t_"):  # t_f / t_r / t_s 要是獨立的字（避免 dataset_from 這種誤中）
        return re.search(rf"(?<![a-z0-9]){k}(?![a-z0-9])", low) is not None
    return k in low


def keyword_hits(text: str, max_lines: int = 12) -> dict[str, Any]:
    counts: dict[str, int] = {}
    lines: list[str] = []
    for line in text.splitlines():
        low = line.lower()
        found = [k for k in KEYWORDS if _hit(k, low)]
        for k in found:
            counts[k] = counts.get(k, 0) + 1
        if found and len(lines) < max_lines:
            lines.append(line.strip()[:200])
    return {"counts": counts, "lines": lines}


def info_findings(info: dict[str, Any]) -> dict[str, Any]:
    feats = info.get("features", {}) or {}
    cams = [k for k, v in feats.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    codecs = sorted({str((feats[k].get("info") or {}).get("video.codec", "?")) for k in cams})
    nonstd = [k for k in feats if k not in STD_FEATURES and not k.startswith("observation.images.")]
    return {"codebase_version": info.get("codebase_version"), "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"), "total_frames": info.get("total_frames"),
            "total_tasks": info.get("total_tasks"), "cameras": cams, "codecs": codecs,
            "nonstandard_features": nonstd}


def _short(v: Any, n: int = 120) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 1] + "…"


def rows_findings(rows: list[dict[str, Any]], kind: str, rel: str, fps: float | None = None) -> dict[str, Any]:
    """表格（parquet 或 jsonl 讀成的 rows）的檢查結果。kind = episodes / tasks / other。"""
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    strings: set[str] = set()
    for r in rows:
        for v in r.values():
            if isinstance(v, str):
                strings.add(v)
            elif isinstance(v, list):
                strings.update(x for x in v if isinstance(x, str))
    res: dict[str, Any] = {"file": rel, "kind": kind, "rows": len(rows), "n_columns": len(cols)}
    if kind == "episodes":
        nonstd = [c for c in cols if c not in STD_EP_COLS and not c.startswith(STD_EP_PREFIX)]
    elif kind == "tasks":
        nonstd = [c for c in cols if c not in {"task", "task_index", "tasks", "__index_level_0__"}]
    else:
        nonstd = cols
    res["nonstandard_columns"] = nonstd
    res["samples"] = {c: [_short(r.get(c)) for r in rows[:3]] for c in nonstd[:20]}
    lengths = [r["length"] for r in rows if isinstance(r.get("length"), (int, float))]
    if lengths and fps:
        sec = sorted(x / fps for x in lengths)
        res["seconds"] = {"min": round(sec[0], 1), "median": round(statistics.median(sec), 1), "max": round(sec[-1], 1)}
    uniq = sorted(strings)
    res["n_strings"] = len(uniq)
    res["strings"] = [_short(s) for s in uniq[:20]]
    res["keyword_hits"] = keyword_hits("\n".join(uniq))
    return res


def _pq():
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
        return pq
    except ImportError:
        return None


def table_findings(path: Path, kind: str, rel: str, fps: float | None = None) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        try:
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        except (OSError, ValueError) as e:
            return {"file": rel, "kind": kind, "error": f"{type(e).__name__}: {e}"[:200]}
        return rows_findings(rows, kind, rel, fps)
    pq = _pq()
    if pq is None:
        return {"file": rel, "kind": kind, "skipped": "pyarrow 未安裝，沒讀這個表格"}
    try:
        rows = pq.read_table(path).to_pylist()
    except Exception as e:  # noqa: BLE001
        return {"file": rel, "kind": kind, "error": f"{type(e).__name__}: {e}"[:200]}
    return rows_findings(rows, kind, rel, fps)


def hf_lister(repo_id: str) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    items = HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    return [(it.path, int(it.size or 0)) for it in items if isinstance(it, RepoFile)]


def hf_downloader(repo_id: str, filename: str, dest: Path) -> Any:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=str(dest))


def fetch(repos: list[tuple[str, str, str]], dest_root: str | Path,
          lister: Lister = hf_lister, downloader: Downloader = hf_downloader) -> list[dict[str, Any]]:
    """列出每個資料集的檔案（不下載），只下載小檔。"""
    out = []
    for key, kind, repo in repos:
        rec: dict[str, Any] = {"key": key, "kind": kind, "repo": repo,
                               "local": str(Path(dest_root) / local_name(repo))}
        try:
            entries = lister(repo)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"列不出檔案：{type(e).__name__}: {e}"[:300]
            log.warning("  %s：%s", repo, rec["error"])
            out.append(rec)
            continue
        cls = classify_tree(entries)
        rec["tree"] = cls
        got, failed = [], []
        for f in cls["small"]:
            try:
                downloader(repo, f, Path(rec["local"]))
                got.append(f)
            except Exception as e:  # noqa: BLE001
                failed.append(f"{f}: {type(e).__name__}: {e}"[:200])
        rec["downloaded"], rec["download_failed"] = got, failed
        log.info("  %-45s 檔案 %4d 個，下載小檔 %d 個（%s）", repo, len(entries), len(got),
                 f"失敗 {len(failed)}" if failed else "全部成功")
        out.append(rec)
    return out


def analyze(rec: dict[str, Any]) -> dict[str, Any]:
    """讀下載的小檔，找失敗時間點的線索。回傳 rec 加上 analysis 與 signals。"""
    if "error" in rec:
        return rec
    local = Path(rec["local"])
    a: dict[str, Any] = {}
    fps = None
    info_p = local / "meta" / "info.json"
    if info_p.exists():
        try:
            a["info"] = info_findings(json.loads(info_p.read_text(encoding="utf-8")))
            fps = a["info"].get("fps")
        except (OSError, ValueError) as e:
            a["info_error"] = f"{type(e).__name__}: {e}"[:200]
    readme = local / "README.md"
    if readme.exists():
        a["readme_hits"] = keyword_hits(readme.read_text(encoding="utf-8", errors="replace"))
    tables, texts = [], []
    for f in rec.get("downloaded", []):
        p = local / f
        if f.endswith((".parquet", ".jsonl")) and (f.startswith("meta/") or f in rec["tree"]["extra"]):
            if f.startswith("meta/episodes_stats"):
                continue
            if f.startswith("meta/episodes"):
                kind = "episodes"
            elif f.startswith("meta/tasks"):
                kind = "tasks"
            else:
                kind = "other"
            tables.append(table_findings(p, kind, f, fps))
        elif f in rec["tree"]["extra"] and p.suffix.lower() in TEXT_SUFFIX:
            try:
                t = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                texts.append({"file": f, "error": str(e)[:200]})
                continue
            texts.append({"file": f, "head": t[:300], "keyword_hits": keyword_hits(t)})
    a["tables"], a["texts"] = tables, texts
    rec["analysis"] = a
    rec["signals"] = signals(rec)
    rec["unread"] = [t["file"] for t in tables if "skipped" in t or "error" in t]
    return rec


def signals(rec: dict[str, Any]) -> list[str]:
    """失敗時間點標註的線索（結構上不屬於標準 LeRobot 的東西）。"""
    a = rec.get("analysis", {})
    sig = []
    if rec["tree"]["extra"]:
        sig.append("非標準檔案：" + "、".join(rec["tree"]["extra"][:8]))
    info = a.get("info") or {}
    if info.get("nonstandard_features"):
        sig.append("info.json 非標準欄位：" + "、".join(info["nonstandard_features"][:8]))
    if isinstance(info.get("total_tasks"), int) and info["total_tasks"] > 1:
        sig.append(f"任務數 total_tasks = {info['total_tasks']}（可能用任務編號標階段）")
    for t in a.get("tables", []):
        if t.get("kind") == "episodes" and t.get("nonstandard_columns"):
            sig.append(f"{t['file']} 非標準欄位：" + "、".join(t["nonstandard_columns"][:8]))
        if t.get("kind") == "tasks" and t.get("n_strings", 0) > 1:
            sig.append(f"{t['file']} 有 {t['n_strings']} 筆任務文字")
    return sig


def verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"state": "error", "zh": "全部連線失敗，沒檢查到"}
    found = [r["repo"] for r in ok if r.get("signals") and r["kind"] in ("recovery", "sample")]
    unread = [r["repo"] for r in ok if r.get("unread") and r["kind"] in ("recovery", "sample")]
    if found:
        return {"state": "found", "zh": "有線索（見下方各資料集）", "repos": found}
    if unread:
        return {"state": "unknown", "zh": "說明檔已下載，但有表格沒讀到（請 Claude 讀）", "repos": unread}
    return {"state": "none", "zh": "沒有：檔案和欄位都是標準 LeRobot 格式，沒有失敗時間點", "repos": []}


def av1_check() -> dict[str, Any]:
    """這台電腦的 PyAV 能不能解 AV1（REBOOT 影片的編碼）。"""
    try:
        import av  # type: ignore[import-not-found]
    except ImportError:
        return {"pyav": None, "decoders": [], "ok": False, "zh": "沒有 PyAV"}
    decs = []
    for name in ("libdav1d", "libaom-av1", "av1"):
        try:
            av.codec.Codec(name, "r")
            decs.append(name)
        except Exception:  # noqa: BLE001
            pass
    sw = [d for d in decs if d != "av1"]
    zh = f"可以（{sw[0]}）" if sw else ("只有硬體解碼器 av1，可能不行" if decs else "不行：沒有 AV1 解碼器")
    return {"pyav": getattr(av, "__version__", "?"), "decoders": decs, "ok": bool(sw), "zh": zh}


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f}"


def size_plan(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每個零件（專家＋救援）只抓彩色影片的大小。"""
    rows = []
    for key, meta in PARTS.items():
        recs = [r for r in results if r["key"] == key and "tree" in r]
        if not recs:
            continue
        cams: dict[str, int] = {}
        depth = 0
        for r in recs:
            depth += r["tree"]["groups"]["data"]
            for k, v in r["tree"]["video_keys"].items():
                short = k.split(".")[-1]
                cams[short] = cams.get(short, 0) + v
        rows.append({"key": key, "part": meta["part"], "cams": cams, "rgb_total": sum(cams.values()),
                     "depth_skipped": depth, "repos": len(recs)})
    return rows


def write_report(results: list[dict[str, Any]], av1: dict[str, Any], out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    v = verdict(results)
    plan = size_plan(results)
    L = ["# 第 13 包第一步：REBOOT 說明檔檢查", "", f"產生時間：{now_iso()}", "",
         "只下載了 README 和 meta/ 小檔；沒有下載影片和深度資料。", "", "## 結論", "",
         f"- 失敗時間點標註：**{v['zh']}**", f"- AV1 影片解碼：**{av1['zh']}**（PyAV {av1.get('pyav')}）", ""]
    if v.get("repos"):
        L += ["  相關資料集：" + "、".join(v["repos"]), ""]
    L += ["## 各資料集", "", "| 零件 | 種類 | 資料集 | 段數 | 影格 | fps | 任務數 | 線索 |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        part = PARTS.get(r["key"], {}).get("part", r["key"])
        if "error" in r:
            L.append(f"| {part} | {KIND_ZH[r['kind']]} | {r['repo']} | – | – | – | – | 連線失敗 |")
            continue
        info = (r.get("analysis") or {}).get("info") or {}
        L.append(f"| {part} | {KIND_ZH[r['kind']]} | {r['repo']} | {info.get('total_episodes', '?')} | "
                 f"{info.get('total_frames', '?')} | {info.get('fps', '?')} | {info.get('total_tasks', '?')} | "
                 f"{len(r.get('signals', []))} |")
    L += ["", "## 每個資料集的細節", ""]
    for r in results:
        L += [f"### {r['repo']}（{KIND_ZH[r['kind']]}）", ""]
        if "error" in r:
            L += [f"- {r['error']}", ""]
            continue
        a = r.get("analysis") or {}
        info = a.get("info") or {}
        L.append(f"- 相機：{'、'.join(info.get('cameras', [])) or '?'}；編碼：{'、'.join(info.get('codecs', [])) or '?'}")
        if r["tree"]["big_skip"]:
            L.append("- 沒下載（超過 20 MB）：" + "、".join(r["tree"]["big_skip"]))
        if r.get("download_failed"):
            L.append("- 下載失敗：" + "；".join(r["download_failed"][:5]))
        for s in r.get("signals", []):
            L.append(f"- **線索**：{s}")
        for t in a.get("tables", []):
            if "skipped" in t or "error" in t:
                L.append(f"- {t['file']}：{t.get('skipped') or t.get('error')}")
                continue
            line = f"- {t['file']}：{t['rows']} 列、{t['n_columns']} 欄"
            if "seconds" in t:
                s = t["seconds"]
                line += f"；每段長度 {s['min']}～{s['max']} 秒（中位 {s['median']}）"
            L.append(line)
            if t.get("kind") == "tasks" or (t.get("strings") and t.get("kind") == "other"):
                L.append("  - 文字：" + " ｜ ".join(t["strings"][:6]))
            for c, vals in t.get("samples", {}).items():
                L.append(f"  - 欄位 `{c}` 範例：" + " ｜ ".join(vals))
            if t["keyword_hits"]["counts"]:
                L.append("  - 關鍵字：" + "、".join(f"{k}×{n}" for k, n in t["keyword_hits"]["counts"].items()))
        for t in a.get("texts", []):
            L.append(f"- 額外檔案 {t['file']}：" + ("、".join(f"{k}×{n}" for k, n in t.get("keyword_hits", {}).get("counts", {}).items()) or "沒有關鍵字"))
        rh = a.get("readme_hits") or {}
        if rh.get("counts"):
            L.append("- README 關鍵字（多半只是資料集名稱裡的 recovery）：" + "、".join(f"{k}×{n}" for k, n in rh["counts"].items()))
        L.append("")
    if plan:
        keys = ["cam_high", "cam_left_wrist", "cam_right_wrist", "cam_low"]
        L += ["## 下載量估算（專家＋救援，只抓彩色影片，GB）", "",
              "| 零件 | 資料集數 | " + " | ".join(CAM_ZH[k] for k in keys) + " | 彩色合計 | 深度（不抓） |",
              "|---|---|" + "---|" * len(keys) + "---|---|"]
        for p in plan:
            L.append(f"| {p['part']} | {p['repos']} | " + " | ".join(_gb(p['cams'].get(k, 0)) for k in keys)
                     + f" | {_gb(p['rgb_total'])} | {_gb(p['depth_skipped'])} |")
        tot = sum(p["rgb_total"] for p in plan)
        L += ["", f"4 種零件彩色影片合計 {_gb(tot)} GB；只抓俯視＋兩個手腕約 "
              f"{_gb(sum(sum(p['cams'].get(k, 0) for k in keys[:3]) for p in plan))} GB。", ""]
    md = out / "report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"generated": now_iso(), "verdict": v, "av1": av1,
                                                  "size_plan": plan, "repos": results},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    return str(md)
