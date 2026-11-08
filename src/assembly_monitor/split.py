"""第 4 包：資料切分與洩漏檢查（規格 P4-1～P4-5）。

P4-1 以 group key（預設 recording_id＝完整錄製檔）分組，整組進同一集合
P4-2 同段的不同相機、幀數、裁切與前後緩衝版本（data/derived 底下以 <錄製檔>__seg<編號> 命名的衍生檔）都跟著該段的 group
P4-3 固定 seed 與比例：依「錄製檔有沒有失敗段」分層，各層打散後依缺額指派；輸出 split.csv 與 group／result／reason 分布
P4-4 稽核：群組跨集合、衍生檔歸屬、同錄製檔視窗重疊、錄製檔時間範圍重疊、來源檔雜湊重複、
     影格 hash（位元組完全相同＝錯誤；近似影格以「段配對」彙總，同一對段在每台相機都有大比例影格近似
     ＝疑似重複序列＝警告。合成圖（sheet.jpg、review/）不比近似，因為它們由同段影格拼成）
P4-5 鎖定：manifest.json 記錄設定與各集合清單的 SHA-256；verify_split 檢查是否被改；load_split 未明確允許就拒讀 test
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import __version__
from .schema import SEG_ID_RE, dataset_result_of, segment_key
from .util import now_iso

log = logging.getLogger(__name__)

DEFAULT_RATIOS = {"dev": 0.25, "test": 0.75}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
SPLIT_COLS = ["segment_id", "recording_id", "segment_index", "group", "split", "dataset_result", "duration_s",
              "date", "start_unix", "end_unix"]
_SEG_IN_NAME = re.compile(r"(?P<key>[^/\\]+?__seg\d+)(?=__|\.|$)")


class SplitError(RuntimeError):
    pass


class LockedTestSetError(PermissionError):
    pass


# ---------------------------------------------------------------- 設定與分組


def parse_ratios(text: str | dict[str, float] | None) -> dict[str, float]:
    """"dev=0.25,test=0.75" → {"dev": 0.25, "test": 0.75}（依序，總和正規化為 1）。"""
    if text is None:
        return dict(DEFAULT_RATIOS)
    if isinstance(text, dict):
        items = list(text.items())
    else:
        items = []
        for part in str(text).split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise SplitError(f"比例格式錯誤：{part!r}（要像 dev=0.25,test=0.75）")
            k, v = part.split("=", 1)
            items.append((k.strip(), float(v)))
    if not items:
        raise SplitError("至少要一個集合")  # 第 8 包：可以只有一個集合（例如 USB 全部當測試集、BNC 全部當開發集）
    names = [k for k, _ in items]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", k) for k in names):
        raise SplitError(f"集合名稱重複或不合法：{names}")
    if any(v <= 0 for _, v in items):
        raise SplitError("比例必須大於 0")
    tot = sum(v for _, v in items)
    return {k: v / tot for k, v in items}


def load_units(index_dir: str | Path, action: str, part: str) -> list[dict[str, Any]]:
    """目標動作／零件的有效段（validation ok / warn）。"""
    p = Path(index_dir) / "segments.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}")
    units = []
    with p.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("action") != action or r.get("part") != part or r.get("validation_status") not in ("ok", "warn"):
                continue
            rid = r["recording_id"]
            m = re.match(r"(\d{4}-\d{2}-\d{2})", rid)
            units.append({
                "segment_id": segment_key(rid, r["segment_id"]),
                "recording_id": rid,
                "segment_index": int(r["segment_id"]),
                "dataset_result": dataset_result_of(r.get("success")),
                "duration_s": round(float(r["end_unix"]) - float(r["start_unix"]), 4),
                "date": m.group(1) if m else "",
                "start_unix": float(r["start_unix"]),
                "end_unix": float(r["end_unix"]),
            })
    units.sort(key=lambda u: (u["recording_id"], u["segment_index"]))
    return units


def _group_value(u: dict[str, Any], group_key: str) -> str:
    if group_key not in u:
        raise SplitError(f"group key {group_key!r} 不存在（可用 recording_id、date、segment_id）")
    return str(u[group_key])


def assign_groups(units: list[dict[str, Any]], ratios: dict[str, float], seed: int = 0,
                  group_key: str = "recording_id", stratify: str = "has_failure") -> dict[str, str]:
    """回傳 {group: split}。同 seed、同輸入 → 同結果。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for u in units:
        groups[_group_value(u, group_key)].append(u)

    def stratum(g: str) -> str:
        if stratify == "has_failure":
            return "has_failure" if any(u["dataset_result"] == "failure" for u in groups[g]) else "success_only"
        if stratify in ("none", "", None):
            return "all"
        raise SplitError(f"未知的分層方式 {stratify!r}（has_failure 或 none）")

    rng = random.Random(seed)
    out: dict[str, str] = {}
    names = list(ratios)
    for st in sorted({stratum(g) for g in groups}):
        items = sorted(g for g in groups if stratum(g) == st)
        rng.shuffle(items)

        def weight(g: str) -> int:
            if st == "has_failure":
                return sum(1 for u in groups[g] if u["dataset_result"] == "failure")
            return len(groups[g])

        total = sum(weight(g) for g in items)
        cur = dict.fromkeys(names, 0.0)
        for g in items:
            best = max(names, key=lambda s: (ratios[s] * total - cur[s], ratios[s], -names.index(s)))
            out[g] = best
            cur[best] += weight(g)
    return out


# ---------------------------------------------------------------- 鎖定（P4-5）


def _sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def lock_info(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        per[r["split"]].append(r["segment_id"])
    return {
        "segments_sha256": _sha256_lines(sorted(f"{r['segment_id']}\t{r['split']}" for r in rows)),
        "per_split": {s: {"n_segments": len(v), "sha256": _sha256_lines(sorted(v))} for s, v in sorted(per.items())},
    }


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_split_csv(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def verify_split(split_dir: str | Path) -> list[str]:
    """重新計算 split.csv 的雜湊並與 manifest 比對；回傳問題清單（空＝未被改動）。"""
    d = Path(split_dir)
    man_p, csv_p = d / "manifest.json", d / "split.csv"
    if not man_p.exists() or not csv_p.exists():
        return [f"{d} 缺少 manifest.json 或 split.csv"]
    man = json.loads(man_p.read_text(encoding="utf-8"))
    rows = read_split_csv(csv_p)
    info = lock_info(rows)
    problems = []
    if info["segments_sha256"] != man.get("lock", {}).get("segments_sha256"):
        problems.append("split.csv 的段與集合清單和 manifest 記錄的不同（可能被改過）")
    for s, v in (man.get("lock", {}).get("per_split") or {}).items():
        if info["per_split"].get(s, {}).get("sha256") != v.get("sha256"):
            problems.append(f"集合 {s} 的清單雜湊不符")
    if man.get("split_csv_sha256") and file_sha256(csv_p) != man["split_csv_sha256"]:
        problems.append("split.csv 檔案內容與 manifest 記錄的 SHA-256 不同")
    return problems


def load_split(split_dir: str | Path, include: Iterable[str] = ("dev",), allow_test: bool = False) -> list[dict[str, Any]]:
    """開發時只讀 dev。要讀 test 必須 allow_test=True（避免在開發階段看到測試答案）。"""
    inc = list(include)
    if "test" in inc and not allow_test:
        raise LockedTestSetError("測試集已鎖定：最終評估時才用 allow_test=True（命令列 --unlock-test）讀取")
    problems = verify_split(split_dir)
    if problems:
        raise SplitError("切分檢查失敗：" + "；".join(problems))
    return [r for r in read_split_csv(Path(split_dir) / "split.csv") if r["split"] in inc]


# ---------------------------------------------------------------- 衍生檔與影格 hash（P4-2、P4-4）


def segment_of_path(rel: str) -> str | None:
    """從相對路徑找出 <錄製檔>__seg<編號>（資料夾名稱或檔名開頭）。"""
    for comp in re.split(r"[/\\]", rel):
        if SEG_ID_RE.match(comp):
            return comp
        m = _SEG_IN_NAME.match(comp)
        if m and SEG_ID_RE.match(m.group("key")):
            return m.group("key")
    return None


def discover_derived(roots: Iterable[str | Path]) -> list[dict[str, Any]]:
    out = []
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        for dirpath, _dirs, files in os.walk(r):
            for name in files:
                if name.endswith(".part"):
                    continue
                p = Path(dirpath) / name
                rel = p.relative_to(r).as_posix()
                seg = segment_of_path(rel)
                ext = p.suffix.lower()
                kind = "image" if ext in IMAGE_EXT else ("meta" if ext == ".json" else ("video" if ext == ".mp4" else "other"))
                out.append({"root": r.as_posix(), "path": rel, "abs": str(p), "segment_id": seg, "kind": kind})
    return out


def dhash64(path: str | Path) -> int | None:
    """差異雜湊：灰階縮成 9×8，比較左右相鄰像素 → 64 位元。讀不到回傳 None。"""
    import cv2

    data = np.fromfile(os.fspath(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def _min_hamming_pairs(a: np.ndarray, b: np.ndarray, threshold: int, chunk: int = 256) -> list[tuple[int, int, int]]:
    """a、b 為 uint64 陣列；回傳漢明距離 ≤ threshold 的 (i, j, d)。"""
    hits: list[tuple[int, int, int]] = []
    if a.size == 0 or b.size == 0:
        return hits
    for s in range(0, a.size, chunk):
        x = np.bitwise_xor(a[s : s + chunk, None], b[None, :])
        d = _POP16[x.view(np.uint16).reshape(x.shape + (4,))].sum(axis=-1, dtype=np.int32)
        ii, jj = np.nonzero(d <= threshold)
        hits.extend((int(s + i), int(j), int(d[i, j])) for i, j in zip(ii, jj))
    return hits


COMPOSITE_DIRS = {"review"}


def frame_camera(rel: str) -> str | None:
    """單一相機的影格（檔名 <相機>__…）回傳相機名；合成圖（sheet.jpg、review/…、檔名沒有相機前綴）回傳 None。"""
    parts = [p for p in re.split(r"[/\\]", rel) if p]
    if not parts or any(p in COMPOSITE_DIRS for p in parts[:-1]):
        return None
    name = parts[-1]
    if "__" not in name or _SEG_IN_NAME.match(name):  # 以段代號開頭的檔名＝整段的拼圖（review/<段>__hand.jpg 之類）
        return None
    cam = name.split("__", 1)[0]
    if not cam or SEG_ID_RE.match(cam) or cam.lower().startswith(("sheet", "overview", "contact")):
        return None
    return cam


def near_duplicate_sequences(frames: list[dict[str, Any]], seg_split: dict[str, str], threshold: int = 4,
                             seq_coverage: float = 0.5, min_frames: int = 3) -> dict[str, Any]:
    """跨集合近似影格，依「段配對 × 相機」彙總。

    frames：每張單相機影格 {segment_id, cam, dhash, path}。同一台相機、不同集合的兩段，
    某段影格在另一段找得到漢明距離 ≤ threshold 的影格，就算「被覆蓋」。
    覆蓋率＝兩段中較高的「被覆蓋影格數／該段該相機影格數」。
    兩段共有的每台相機覆蓋率都 ≥ seq_coverage（且被覆蓋影格數 ≥ min_frames，或該相機影格本來就更少）
    ＝疑似重複序列（同一段畫面出現在兩個集合）。只有部分相機或零星影格相似，通常是相同場景。
    """
    by_cam: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    totals: Counter = Counter()
    seg_cams: dict[str, set[str]] = defaultdict(set)
    for f in frames:
        if f.get("dhash") is None or f.get("cam") is None or f["segment_id"] not in seg_split:
            continue
        by_cam[f["cam"]][seg_split[f["segment_id"]]].append(f)
        totals[(f["segment_id"], f["cam"])] += 1
        seg_cams[f["segment_id"]].add(f["cam"])
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    per_cam: Counter = Counter()
    hit_frames: dict[str, set[tuple[str, int]]] = defaultdict(set)
    n_pairs = 0
    for cam in sorted(by_cam):
        names = sorted(by_cam[cam])
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = by_cam[cam][names[i]], by_cam[cam][names[j]]
                a = np.array([f["dhash"] for f in A], dtype=np.uint64)
                b = np.array([f["dhash"] for f in B], dtype=np.uint64)
                for ia, jb, dist in _min_hamming_pairs(a, b, threshold):
                    fa, fb = A[ia], B[jb]
                    key = (fa["segment_id"], fb["segment_id"])
                    m = pairs.setdefault(key, {"splits": (names[i], names[j]), "cams": {}, "n": 0, "min_d": 64,
                                               "example": None})
                    sa, sb = m["cams"].setdefault(cam, (set(), set()))
                    sa.add(ia)
                    sb.add(jb)
                    hit_frames[cam].add((names[i], ia))
                    hit_frames[cam].add((names[j], jb))
                    m["n"] += 1
                    if dist < m["min_d"]:
                        m["min_d"] = dist
                        m["example"] = [fa.get("path"), fb.get("path")]
                    n_pairs += 1
                    per_cam[cam] += 1
    rows: list[dict[str, Any]] = []
    for (a_id, b_id), m in pairs.items():
        common = sorted(seg_cams[a_id] & seg_cams[b_id])
        cov: dict[str, float] = {}
        enough = True
        for cam in common:
            sa, sb = m["cams"].get(cam, (set(), set()))
            ta, tb = totals[(a_id, cam)], totals[(b_id, cam)]
            cov[cam] = round(max(len(sa) / ta if ta else 0.0, len(sb) / tb if tb else 0.0), 3)
            if max(len(sa), len(sb)) < min(min_frames, ta, tb):
                enough = False
        min_cov = min(cov.values()) if cov else 0.0
        rows.append({"a": a_id, "a_split": m["splits"][0], "b": b_id, "b_split": m["splits"][1],
                     "min_coverage": round(min_cov, 3), "coverage": cov, "near_pairs": m["n"],
                     "min_distance": m["min_d"], "example": m["example"],
                     "suspect": bool(common) and enough and min_cov >= seq_coverage})
    rows.sort(key=lambda r: (-r["suspect"], -r["min_coverage"], -r["near_pairs"]))
    cam_total: Counter = Counter()
    for (_seg, cam), n in totals.items():
        cam_total[cam] += n
    return {"n_frames": sum(totals.values()), "n_pairs": n_pairs, "per_camera": dict(per_cam),
            "frames_with_match": {c: len(hit_frames.get(c, ())) for c in sorted(cam_total)},
            "frames_per_camera": dict(sorted(cam_total.items())),
            "n_segment_pairs": len(rows), "suspects": [r for r in rows if r["suspect"]], "top": rows[:10]}


def hash_images(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for f in files:
        try:
            raw = Path(f["abs"]).read_bytes()
        except OSError:
            out.append({**f, "sha1": None, "dhash": None})
            continue
        out.append({**f, "sha1": hashlib.sha1(raw).hexdigest(), "dhash": dhash64(f["abs"])})
    return out


# ---------------------------------------------------------------- 稽核（P4-4）


def _recording_ranges(index_dir: Path, rec_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for rid in rec_ids:
        p = index_dir / "recordings" / f"{rid}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        firsts = [c.get("t_first") for c in (d.get("cameras") or {}).values() if isinstance(c, dict) and c.get("t_first")]
        lasts = [c.get("t_last") for c in (d.get("cameras") or {}).values() if isinstance(c, dict) and c.get("t_last")]
        out[rid] = {"t0": min(firsts) if firsts else None, "t1": max(lasts) if lasts else None,
                    "sha256": d.get("source_sha256"), "crc32": d.get("source_crc32"),
                    "size": d.get("source_size_bytes")}
    return out


def audit_split(rows: list[dict[str, Any]], index_dir: str | Path, derived_roots: Iterable[str | Path] = (),
                start_buffer: float = 0.5, end_buffer: float = 2.0, hash_frames: bool = True,
                near_threshold: int = 4, max_examples: int = 20, seq_coverage: float = 0.5) -> dict[str, Any]:
    idx = Path(index_dir)
    seg_split = {r["segment_id"]: r["split"] for r in rows}
    seg_group = {r["segment_id"]: r["group"] for r in rows}
    checks: list[dict[str, Any]] = []

    def add(cid: str, title: str, status: str, detail: str, examples: list | None = None, **extra: Any) -> None:
        checks.append({"id": cid, "title": title, "status": status, "detail": detail,
                       "examples": (examples or [])[:max_examples], **extra})

    # 1. 段唯一、群組不跨集合
    dup = [k for k, n in Counter(r["segment_id"] for r in rows).items() if n > 1]
    add("segment_unique", "每段只出現一次", "fail" if dup else "pass", f"重複 {len(dup)} 段", dup)
    g_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        g_splits[r["group"]].add(r["split"])
    cross = sorted(g for g, s in g_splits.items() if len(s) > 1)
    add("group_crossing", "同一 group 不跨集合（P4-1）", "fail" if cross else "pass",
        f"{len(g_splits)} 個 group，跨集合 {len(cross)} 個", [{"group": g, "splits": sorted(g_splits[g])} for g in cross])
    # 2. 同錄製檔、不同集合的段：取影格視窗是否重疊
    by_rec: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_rec[r["recording_id"]].append(r)
    overlap, mixed = [], []
    for rid, rs in by_rec.items():
        if len({r["split"] for r in rs}) > 1:
            mixed.append(rid)
        rs = sorted(rs, key=lambda r: float(r["start_unix"]))
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a["split"] == b["split"]:
                    continue
                a0, a1 = float(a["start_unix"]) - start_buffer, float(a["end_unix"]) + end_buffer
                b0, b1 = float(b["start_unix"]) - start_buffer, float(b["end_unix"]) + end_buffer
                if min(a1, b1) > max(a0, b0):
                    overlap.append({"a": a["segment_id"], "a_split": a["split"], "b": b["segment_id"],
                                    "b_split": b["split"], "overlap_s": round(min(a1, b1) - max(a0, b0), 3)})
    add("time_overlap", "同錄製檔不同集合的段，取影格視窗不重疊", "fail" if overlap else "pass",
        f"錄製檔分在多個集合 {len(mixed)} 個；視窗重疊 {len(overlap)} 對", overlap)
    # 3. 錄製檔時間範圍重疊、來源雜湊重複
    ranges = _recording_ranges(idx, by_rec)
    rec_split = {rid: sorted({r["split"] for r in rs}) for rid, rs in by_rec.items()}
    items = sorted((v["t0"], v["t1"], rid) for rid, v in ranges.items() if v["t0"] is not None and v["t1"] is not None)
    r_overlap = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[j][0] >= items[i][1]:
                break
            a, b = items[i][2], items[j][2]
            if set(rec_split[a]) != set(rec_split[b]):
                r_overlap.append({"a": a, "b": b, "overlap_s": round(min(items[i][1], items[j][1]) - items[j][0], 3)})
    add("recording_time_overlap", "不同集合的錄製檔，錄影時間不重疊", "fail" if r_overlap else "pass",
        f"有時間範圍的錄製檔 {len(items)} 個；不同集合且重疊 {len(r_overlap)} 對", r_overlap)
    src: dict[str, list[str]] = defaultdict(list)
    for rid, v in ranges.items():
        key = v["sha256"] or (f"{v['size']}:{v['crc32']}" if v["size"] and v["crc32"] else None)
        if key:
            src[key].append(rid)
    s_dup = [{"recordings": rids, "splits": sorted({s for r in rids for s in rec_split[r]})}
             for rids in src.values() if len(rids) > 1]
    s_bad = [x for x in s_dup if len(x["splits"]) > 1]
    add("source_duplicates", "來源檔雜湊不重複出現在不同集合", "fail" if s_bad else ("warn" if s_dup else "pass"),
        f"比對 {len(ranges)} 個錄製檔的來源雜湊；重複 {len(s_dup)} 組，其中跨集合 {len(s_bad)} 組", s_dup)
    # 4. 衍生檔歸屬（P4-2）
    files = discover_derived(derived_roots)
    in_split = [f for f in files if f["segment_id"] in seg_split]
    per_split = Counter(seg_split[f["segment_id"]] for f in in_split)
    groups_per_split: dict[str, set[str]] = defaultdict(set)
    for f in in_split:
        groups_per_split[seg_split[f["segment_id"]]].add(seg_group[f["segment_id"]])
    g_count = Counter(g for gs in groups_per_split.values() for g in gs)
    shared = sorted(g for g, n in g_count.items() if n > 1)
    kinds = Counter(f["kind"] for f in in_split)
    add("derived_samples", "衍生檔（各相機、1/4/8 幀、緩衝版本、標註影格）都跟著所屬段的集合（P4-2）",
        "fail" if shared else "pass",
        f"掃描 {len(files)} 個衍生檔，屬於本切分 {len(in_split)} 個（{dict(per_split)}；{dict(kinds)}），"
        f"其他零件或無法對應 {len(files) - len(in_split)} 個；同一 group 出現在多個集合的衍生檔 {len(shared)} 個",
        shared, n_files=len(in_split), per_split=dict(per_split))
    # 5. 影格 hash
    if hash_frames:
        imgs = hash_images([f for f in in_split if f["kind"] == "image"])
        by_sha: dict[str, list[dict]] = defaultdict(list)
        for f in imgs:
            if f["sha1"]:
                by_sha[f["sha1"]].append(f)
        exact = []
        for sha, fs in by_sha.items():
            if len({seg_split[f["segment_id"]] for f in fs}) > 1:
                exact.append({"sha1": sha, "files": [f"{f['path']} ({seg_split[f['segment_id']]})" for f in fs[:6]]})
        frames = []
        for f in imgs:
            cam = frame_camera(f["path"])
            if cam is not None:
                frames.append({**f, "cam": cam})
        nd = near_duplicate_sequences(frames, seg_split, near_threshold, seq_coverage)
        unreadable = sum(1 for f in imgs if f["dhash"] is None)
        add("image_hash_exact", "影格位元組完全相同的檔案不跨集合（P4-4）", "fail" if exact else "pass",
            f"雜湊 {len(imgs)} 張影像；跨集合完全相同 {len(exact)} 組；讀不到 {unreadable} 張", exact)
        cams = "、".join(f"{c} {nd['frames_with_match'].get(c, 0)}／{n}" for c, n in nd["frames_per_camera"].items()) or "無"
        if nd["suspects"]:
            tail = "；請打開 examples 裡的影格確認是不是同一段畫面"
        elif nd["n_pairs"]:
            tail = "；沒有疑似重複序列（近似只集中在部分相機，例如背景固定的外部相機，屬於相同場景，不算洩漏）"
        else:
            tail = ""
        add("image_hash_near",
            f"近似影格跨集合（dHash 漢明距離 ≤ {near_threshold}；同一對段每台相機都有 ≥ {seq_coverage:.0%} 影格近似＝疑似重複序列）",
            "warn" if nd["suspects"] else "pass",
            f"比對 {nd['n_frames']} 張單相機影格（合成圖 {len(imgs) - len(frames)} 張不比）；在另一集合找得到近似影格的張數"
            f"（依相機）：{cams}，分布在 {nd['n_segment_pairs']} 組段配對；疑似重複序列 {len(nd['suspects'])} 組{tail}",
            nd["suspects"], n_pairs=nd["n_pairs"], n_segment_pairs=nd["n_segment_pairs"], per_camera=nd["per_camera"],
            frames_with_match=nd["frames_with_match"], frames_per_camera=nd["frames_per_camera"], top_pairs=nd["top"])
    status = "fail" if any(c["status"] == "fail" for c in checks) else ("warn" if any(c["status"] == "warn" for c in checks) else "pass")
    return {"status": status, "generated_at": now_iso(), "checks": checks}


def write_leakage_report(report: dict[str, Any], out_dir: Path, title: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "leakage_report.json"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    mark = {"pass": "通過", "warn": "警告", "fail": "**未通過**"}
    lines = [f"# 洩漏檢查報告：{title}", "", f"產生時間：{report['generated_at']}　總結：{mark[report['status']]}", "",
             "| 檢查 | 結果 | 說明 |", "|---|---|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['title']} | {mark[c['status']]} | {c['detail']} |")
    for c in report["checks"]:
        if c["examples"] and c["status"] != "pass":
            lines += ["", f"## {c['title']}：前 {len(c['examples'])} 例", ""]
            lines += [f"- `{json.dumps(e, ensure_ascii=False)}`" for e in c["examples"]]
    for c in report["checks"]:
        top = c.get("top_pairs") or []
        if top:
            lines += ["", "## 近似影格最多的段配對（前 5 組，供人工確認）", "",
                      "覆蓋率＝這對段中，某段影格在另一段找得到近似影格的比例（各相機取兩段較高者）。",
                      "疑似重複序列要每台相機都高；只有外部相機高，通常只是背景相同。", "",
                      "| 段 A（集合） | 段 B（集合） | 各相機覆蓋率 | 近似對數 | 最小距離 | 疑似重複 |", "|---|---|---|---|---|---|"]
            for t in top[:5]:
                cov = "、".join(f"{k} {v:.0%}" for k, v in t["coverage"].items())
                lines.append(f"| {t['a']}（{t['a_split']}） | {t['b']}（{t['b_split']}） | {cov} | {t['near_pairs']} | "
                             f"{t['min_distance']} | {'是' if t['suspect'] else '否'} |")
    mp = out_dir / "leakage_report.md"
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(jp), "md": str(mp)}


# ---------------------------------------------------------------- 主流程（P4-3）


def summarize(rows: list[dict[str, Any]], names: list[str], gold: dict[str, dict] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}}
    for s in names + ["all"]:
        rs = [r for r in rows if s == "all" or r["split"] == s]
        fails = [r for r in rs if r["dataset_result"] == "failure"]
        d = {
            "n_groups": len({r["group"] for r in rs}),
            "n_segments": len(rs),
            "n_failure": len(fails),
            "n_success": sum(1 for r in rs if r["dataset_result"] == "success"),
            "failure_rate": round(len(fails) / len(rs), 3) if rs else None,
            "by_date": dict(sorted(Counter(r["date"] for r in rs).items())),
            "failures_by_date": dict(sorted(Counter(r["date"] for r in fails).items())),
            "groups_with_failure": len({r["group"] for r in fails}),
        }
        if gold is not None:
            reasons = Counter((gold.get(r["segment_id"]) or {}).get("failure_reason") or "（未標註）" for r in fails)
            d["failure_reasons"] = dict(sorted(reasons.items()))
            d["n_gold"] = sum(1 for r in rs if r["segment_id"] in gold)
        out["splits"][s] = d
    return out


def write_summary_md(summary: dict[str, Any], spec: dict[str, Any], path: Path) -> None:
    names = [s for s in summary["splits"] if s != "all"]
    lines = [f"# 切分摘要：{spec['name']}", "",
             f"{spec['action']} {spec['part']}｜group key：`{spec['group_key']}`｜分層：`{spec['stratify']}`｜seed {spec['seed']}｜"
             f"比例 {', '.join(f'{k} {v:.0%}' for k, v in spec['ratios'].items())}", "",
             "| 集合 | group 數 | 段數 | 失敗 | 成功 | 失敗率 | 含失敗的 group |", "|---|---|---|---|---|---|---|"]
    for s in names + ["all"]:
        d = summary["splits"][s]
        lines.append(f"| {s} | {d['n_groups']} | {d['n_segments']} | {d['n_failure']} | {d['n_success']} | "
                     f"{(d['failure_rate'] or 0):.0%} | {d['groups_with_failure']} |")
    lines += ["", "## 依日期（段數／失敗）", "", "| 日期 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    dates = sorted({k for s in names for k in summary["splits"][s]["by_date"]})
    for dt in dates:
        cells = [f"{summary['splits'][s]['by_date'].get(dt, 0)}／{summary['splits'][s]['failures_by_date'].get(dt, 0)}" for s in names]
        lines.append(f"| {dt} | " + " | ".join(cells) + " |")
    if "failure_reasons" in summary["splits"][names[0]]:
        lines += ["", "## 失敗原因分布（gold 標註）", "", "| 原因 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        rs = sorted({k for s in names for k in summary["splits"][s]["failure_reasons"]})
        for k in rs:
            lines.append(f"| {k} | " + " | ".join(str(summary["splits"][s]["failure_reasons"].get(k, 0)) for s in names) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_split(*, index_dir: str | Path, out_root: str | Path, action: str, part: str,
               ratios: dict[str, float] | None = None, seed: int = 0, group_key: str = "recording_id",
               stratify: str = "has_failure", name: str | None = None, derived_roots: Iterable[str | Path] = (),
               gold: dict[str, dict] | None = None, start_buffer: float = 0.5, end_buffer: float = 2.0,
               hash_frames: bool = True, near_threshold: int = 4, seq_coverage: float = 0.5,
               force: bool = False) -> dict[str, Any]:
    ratios = parse_ratios(ratios)
    units = load_units(index_dir, action, part)
    if not units:
        raise SplitError(f"索引裡沒有 {action} / {part} 的有效段")
    name = name or f"{re.sub(r'[^A-Za-z0-9]+', '', part).lower()}_" + "_".join(f"{k}{round(v * 100)}" for k, v in ratios.items()) + f"_s{seed}"
    out_dir = Path(out_root) / name
    assign = assign_groups(units, ratios, seed, group_key, stratify)
    rows = []
    for u in units:
        g = _group_value(u, group_key)
        rows.append({**{k: u[k] for k in SPLIT_COLS if k in u}, "group": g, "split": assign[g]})
    info = lock_info(rows)
    man_p = out_dir / "manifest.json"
    if man_p.exists() and not force:
        old = json.loads(man_p.read_text(encoding="utf-8"))
        if old.get("locked") and old.get("lock", {}).get("segments_sha256") != info["segments_sha256"]:
            raise SplitError(f"{out_dir} 已鎖定且內容不同；確定要重切請加 --force（舊的 test 清單會失效）")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / "split.csv"
    with csv_p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=SPLIT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SPLIT_COLS})
    spec = {"name": name, "action": action, "part": part, "ratios": ratios, "seed": seed, "group_key": group_key,
            "stratify": stratify, "unit": "segment", "window": {"start_buffer": start_buffer, "end_buffer": end_buffer}}
    summary = summarize(rows, list(ratios), gold)
    (out_dir / "split_summary.json").write_text(json.dumps({"spec": spec, **summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    write_summary_md(summary, spec, out_dir / "split_summary.md")
    report = audit_split(rows, index_dir, derived_roots, start_buffer, end_buffer, hash_frames, near_threshold,
                         seq_coverage=seq_coverage)
    rep_paths = write_leakage_report(report, out_dir, name)
    derived = discover_derived(derived_roots)
    seg_split = {r["segment_id"]: r["split"] for r in rows}
    with (out_dir / "derived_manifest.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "segment_id", "kind", "root", "path"])
        w.writeheader()
        for f in sorted(derived, key=lambda f: (seg_split.get(f["segment_id"], "~"), f["root"], f["path"])):
            if f["segment_id"] in seg_split:
                w.writerow({"split": seg_split[f["segment_id"]], "segment_id": f["segment_id"], "kind": f["kind"],
                            "root": f["root"], "path": f["path"]})
    manifest = {
        "name": name, "created_at": now_iso(), "tool": f"assembly_monitor.split {__version__}", "spec": spec,
        "counts": {s: {k: summary["splits"][s][k] for k in ("n_groups", "n_segments", "n_failure", "n_success")}
                   for s in summary["splits"]},
        "lock": info, "split_csv_sha256": file_sha256(csv_p), "locked": True,
        "leakage_status": report["status"],
        "note": "test 清單已鎖定：開發階段只用 dev；最終評估才讀 test（load_split(..., allow_test=True)）。",
    }
    man_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("切分 %s：%s；洩漏檢查 %s", name,
             "、".join(f"{s} {summary['splits'][s]['n_segments']} 段（失敗 {summary['splits'][s]['n_failure']}）" for s in ratios),
             report["status"])
    return {"name": name, "dir": str(out_dir), "split_csv": str(csv_p), "manifest": str(man_p), "summary": summary,
            "leakage": report, "reports": rep_paths}
