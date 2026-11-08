"""第 2 包：依確認過的欄位映射建立 segment 索引與驗證（P1-3、P1-4）。

每個錄製檔先寫成 data/index/recordings/<id>.json（單檔是唯一真相、可續跑），
再彙整成 segments.csv、skills.csv、recordings.csv、validation.json 與 Insert 候選報表。
錯誤碼：E_* 會讓該段 validation_status=error；W_* 為 warn；被排除的段（例如 "No action."）標 excluded。
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .mapping import Mapping
from .media import make_blob_ref, probe_blob
from .util import natural_key, normalize_id, now_iso

log = logging.getLogger(__name__)

RANGE_TOL_S = 0.5
SHORT_SEGMENT_S = 0.3


def _scalar(g: Any, key: str | None) -> Any:
    """讀群組內的純量（或長度 1 的陣列）；bytes 轉成字串，numpy 轉成 Python 型別。"""
    import h5py

    if not key or key not in g:
        return None
    obj = g[key]
    if not isinstance(obj, h5py.Dataset) or obj.shape not in ((), (1,)):
        return None
    v = obj[()] if obj.shape == () else obj[0]
    if isinstance(v, (bytes, np.bytes_)):
        return bytes(v).decode("utf-8", "replace").rstrip("\x00")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def _camera_info(f: Any, mapping: Mapping, file_path: str, probe_video: bool) -> tuple[dict, dict]:
    cams: dict[str, dict] = {}
    ts_arrays: dict[str, np.ndarray] = {}
    for name, spec in mapping.cameras.items():
        info: dict[str, Any] = {"name": name, "role": spec.get("role"), "video": spec["video"],
                                "timestamps": spec["timestamps"], "has_video": spec["video"] in f,
                                "has_timestamps": spec["timestamps"] in f}
        if info["has_timestamps"]:
            t = np.asarray(f[spec["timestamps"]][()], dtype=np.float64).ravel()
            ts_arrays[name] = t
            d = np.diff(t) if t.size > 1 else np.array([])
            pos = d[d > 0]
            info.update(
                n_timestamps=int(t.size),
                t_first=float(t[0]) if t.size else None,
                t_last=float(t[-1]) if t.size else None,
                monotonic=bool((d >= 0).all()) if d.size else True,
                rate_hz=float(1.0 / np.median(pos)) if pos.size else None,
            )
        if info["has_video"] and probe_video:
            try:
                blob = make_blob_ref(f[spec["video"]], file_path, 64 << 20)
                if blob is not None and blob.access != "unavailable":
                    res = probe_blob(blob, "mp4", count_frames=False)
                    if res.get("ok"):
                        s = res["summary"]
                        info.update(frames=s.get("frames"), fps_nominal=s.get("fps_nominal"),
                                    width=s.get("width"), height=s.get("height"), codec=s.get("codec"))
                    else:
                        info["probe_error"] = str(res.get("errors"))[:300]
            except Exception as e:  # noqa: BLE001
                info["probe_error"] = f"{type(e).__name__}: {e}"
        info["present"] = info["has_video"] and info["has_timestamps"]
        cams[name] = info
    return cams, ts_arrays


def index_h5(
    path: str | Path,
    mapping: Mapping,
    recording_id: str | None = None,
    source: str | None = None,
    known_issue_files: set[str] | None = None,
    probe_video: bool = True,
) -> dict[str, Any]:
    """索引單一錄製檔。只讀映射裡指定的路徑；任何單段問題都寫進錯誤碼，不拋例外。"""
    import h5py

    p = Path(path)
    rid = recording_id or p.stem
    out: dict[str, Any] = {
        "recording_id": rid,
        "source": source or str(p),
        "file_name": p.name,
        "size_bytes": p.stat().st_size,
        "indexed_at": now_iso(),
        "tool": f"assembly_monitor.index {__version__}",
        "mapping": {"path": mapping.path, "confirmed_at": mapping.confirmed_at},
        "status": "ok",
        "recording_issues": [],
        "cameras": {},
        "segments": [],
        "skills": [],
    }
    rec_issues: list[str] = out["recording_issues"]
    if known_issue_files and p.name in known_issue_files:
        rec_issues.append("W_KNOWN_ISSUE")
    with h5py.File(os.fspath(p), "r") as f:
        cams, ts = _camera_info(f, mapping, os.fspath(p), probe_video)
        out["cameras"] = cams
        for name, c in cams.items():
            if not c["present"]:
                rec_issues.append(f"W_CAMERA_MISSING:{name}")
            if c.get("frames") and c.get("n_timestamps") and c["frames"] != c["n_timestamps"]:
                rec_issues.append(f"W_FRAMES_TS_MISMATCH:{name}")
            if c.get("has_timestamps") and not c.get("monotonic", True):
                rec_issues.append(f"W_TS_NON_MONOTONIC:{name}")
        firsts = [c["t_first"] for c in cams.values() if c.get("t_first") is not None]
        if mapping.seg_container not in f:
            out["status"] = "error"
            rec_issues.append("E_NO_SEGMENTS")
            return out
        cont = f[mapping.seg_container]
        fields = mapping.seg_fields
        rows: list[dict] = []
        skills: list[dict] = []
        for key in sorted(cont.keys(), key=natural_key):
            g = cont[key]
            if not isinstance(g, h5py.Group):
                continue
            start, end = _num(_scalar(g, fields["start"])), _num(_scalar(g, fields["end"]))
            succ_raw = _scalar(g, fields.get("success"))
            text = _scalar(g, fields.get("text"))
            text = str(text) if text is not None else None
            action, part = mapping.parse_text(text)
            row: dict[str, Any] = {
                "recording_id": rid,
                "segment_id": key,
                "order": _scalar(g, fields.get("order")),
                "source_path": out["source"],
                "text": text,
                "action": action,
                "part": part,
                "excluded": mapping.is_excluded(text),
                "start_unix": start,
                "end_unix": end,
                "duration_s": (end - start) if start is not None and end is not None else None,
                "success": _bool(succ_raw),
                "codes": [],
            }
            codes: list[str] = row["codes"]
            if start is None or end is None:
                codes.append("E_MISSING_TIME")
            elif start >= end:
                codes.append("E_START_GE_END")
            elif end - start < SHORT_SEGMENT_S:
                codes.append("W_SHORT")
            if row["success"] is None:
                codes.append("E_SUCCESS_NOT_BOOL")
            if text and action is None and not row["excluded"]:
                codes.append("W_TEXT_UNPARSED")
            frames: dict[str, int] = {}
            in_range_any = False
            for cam in mapping.cameras:
                t = ts.get(cam)
                if t is None or not cams[cam]["present"] or start is None or end is None:
                    continue
                i0 = int(np.searchsorted(t, start, side="left"))
                i1 = int(np.searchsorted(t, end, side="right"))
                frames[cam] = max(0, i1 - i0)
                if cam in mapping.best_candidates:
                    if t.size and start >= t[0] - RANGE_TOL_S and end <= t[-1] + RANGE_TOL_S:
                        in_range_any = True
                    if frames[cam] == 0:
                        codes.append(f"W_NO_FRAMES:{cam}")
            row["frames"] = frames
            best_frames = [frames[c] for c in mapping.best_candidates if c in frames]
            row["frame_count"] = min(best_frames) if best_frames else 0
            row["camera_ids"] = [c for c in mapping.best_candidates if frames.get(c)]
            if start is not None and end is not None and start < end:
                if not best_frames or max(best_frames) == 0:
                    codes.append("E_NO_FRAMES")
                elif not in_range_any:
                    codes.append("E_OUT_OF_RANGE")
            # 低階技能
            n_sk = 0
            if mapping.low_container and mapping.low_container in g and isinstance(g[mapping.low_container], h5py.Group):
                lg = g[mapping.low_container]
                lf = mapping.low_fields
                for sk in sorted(lg.keys(), key=natural_key):
                    sg = lg[sk]
                    if not isinstance(sg, h5py.Group):
                        continue
                    s0, s1 = _num(_scalar(sg, lf["start"])), _num(_scalar(sg, lf["end"]))
                    st = _scalar(sg, lf.get("text"))
                    srow = {
                        "recording_id": rid, "segment_id": key, "skill_id": sk,
                        "text": str(st) if st is not None else None,
                        "start_unix": s0, "end_unix": s1,
                        "duration_s": (s1 - s0) if s0 is not None and s1 is not None else None,
                        "success": _bool(_scalar(sg, lf.get("success"))), "codes": [],
                    }
                    if s0 is None or s1 is None:
                        srow["codes"].append("E_MISSING_TIME")
                    elif s0 >= s1:
                        srow["codes"].append("E_START_GE_END")
                    elif start is not None and end is not None and start < end:
                        eps = 1e-3 * (end - start)
                        if s0 < start - eps or s1 > end + eps:
                            srow["codes"].append("W_SKILL_OUTSIDE")
                    skills.append(srow)
                    n_sk += 1
            row["n_skills"] = n_sk
            rows.append(row)
    # 重複 ID、前後間隔、相對秒數
    ids = Counter(normalize_id(r["segment_id"]) for r in rows)
    for r in rows:
        if ids[normalize_id(r["segment_id"])] > 1:
            r["codes"].append("E_DUP_ID")
    timed = sorted((r for r in rows if r["start_unix"] is not None and r["end_unix"] is not None),
                   key=lambda r: r["start_unix"])
    for a, b in zip(timed, timed[1:]):
        gap = b["start_unix"] - a["end_unix"]
        a["next_gap_s"] = gap
        b["prev_gap_s"] = gap
    t0 = min(firsts) if firsts else (timed[0]["start_unix"] if timed else None)
    out["t0_unix"] = t0
    for r in rows + skills:
        r["start_sec"] = (r["start_unix"] - t0) if t0 is not None and r["start_unix"] is not None else None
        r["end_sec"] = (r["end_unix"] - t0) if t0 is not None and r["end_unix"] is not None else None
    for r in rows:
        r["codes"] = sorted(set(r["codes"]))
        if r["excluded"]:
            r["validation_status"] = "excluded"
        elif any(c.startswith("E_") for c in r["codes"]):
            r["validation_status"] = "error"
        elif r["codes"]:
            r["validation_status"] = "warn"
        else:
            r["validation_status"] = "ok"
    for s in skills:
        s["validation_status"] = "error" if any(c.startswith("E_") for c in s["codes"]) else ("warn" if s["codes"] else "ok")
    out["segments"] = rows
    out["skills"] = skills
    if any(c.startswith("E_") for c in rec_issues):
        out["status"] = "error"
    return out


# ---------------------------------------------------------------- 彙整輸出


def save_recording(result: dict[str, Any], index_dir: str | Path) -> Path:
    d = Path(index_dir) / "recordings"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{result['recording_id']}.json"
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    err = d / f"{result['recording_id']}.error.json"
    if err.exists():
        err.unlink()
    return p


def save_recording_error(recording_id: str, source: str, error: str, index_dir: str | Path) -> Path:
    d = Path(index_dir) / "recordings"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{recording_id}.error.json"
    p.write_text(json.dumps({"recording_id": recording_id, "source": source, "error": error, "at": now_iso()},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_recordings(index_dir: str | Path) -> tuple[list[dict], list[dict]]:
    d = Path(index_dir) / "recordings"
    recs, errs = [], []
    if not d.exists():
        return recs, errs
    for p in sorted(d.glob("*.json"), key=lambda q: natural_key(q.name)):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            errs.append({"recording_id": p.stem, "source": str(p), "error": f"無法讀取：{e}"})
            continue
        (errs if p.name.endswith(".error.json") else recs).append(data)
    return recs, errs


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return f"{v:.6f}" if abs(v) >= 1e5 else f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return "" if v is None else v


def write_index_outputs(index_dir: str | Path, mapping: Mapping) -> dict[str, str]:
    """從 recordings/*.json 重建所有彙整檔（可隨時重跑）。"""
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs, errs = load_recordings(out)
    cams = list(mapping.cameras)
    seg_cols = ["recording_id", "segment_id", "order", "source_path", "text", "action", "part", "excluded",
                "start_unix", "end_unix", "start_sec", "end_sec", "duration_s", "success", "prev_gap_s",
                "next_gap_s", "camera_ids", "fps_or_timestamps", "frame_count"] + [f"frames_{c}" for c in cams] + [
                "n_skills", "validation_status", "error_code"]
    sk_cols = ["recording_id", "segment_id", "skill_id", "text", "start_unix", "end_unix", "start_sec", "end_sec",
               "duration_s", "success", "validation_status", "error_code"]
    rec_cols = ["recording_id", "source", "size_bytes", "n_segments", "n_excluded", "n_insert", "status",
                "recording_issues"] + [f"{c}_{k}" for c in cams for k in ("frames", "n_timestamps", "rate_hz")]
    seg_path, sk_path, rec_path = out / "segments.csv", out / "skills.csv", out / "recordings.csv"
    items: list[dict] = []
    code_counts: Counter = Counter()
    status_counts: Counter = Counter()
    with seg_path.open("w", newline="", encoding="utf-8-sig") as fs, \
            sk_path.open("w", newline="", encoding="utf-8-sig") as fk, \
            rec_path.open("w", newline="", encoding="utf-8-sig") as fr:
        ws = csv.DictWriter(fs, fieldnames=seg_cols)
        wk = csv.DictWriter(fk, fieldnames=sk_cols)
        wr = csv.DictWriter(fr, fieldnames=rec_cols)
        ws.writeheader()
        wk.writeheader()
        wr.writeheader()
        seen_ids: Counter = Counter()
        for rec in recs:
            for r in rec["segments"]:
                seen_ids[(r["recording_id"], normalize_id(r["segment_id"]))] += 1
        for rec in recs:
            rid = rec["recording_id"]
            for r in rec["segments"]:
                codes = list(r.get("codes") or [])
                if seen_ids[(rid, normalize_id(r["segment_id"]))] > 1 and "E_DUP_ID" not in codes:
                    codes.append("E_DUP_ID")  # 跨檔重複（例如同名錄製檔出現兩次）
                status = r.get("validation_status")
                if "E_DUP_ID" in codes and status in ("ok", "warn"):
                    status = "error"
                row = {k: _fmt(r.get(k)) for k in seg_cols if k in r}
                row.update({f"frames_{c}": (r.get("frames") or {}).get(c, "") for c in cams})
                row["fps_or_timestamps"] = "timestamps"
                row["validation_status"] = status
                row["error_code"] = ";".join(codes)
                ws.writerow(row)
                status_counts[status] += 1
                for c in codes:
                    code_counts[c] += 1
                if codes and status != "excluded":
                    items.append({"recording_id": rid, "segment_id": r["segment_id"], "source": r.get("source_path"),
                                  "text": r.get("text"), "status": status, "codes": codes})
            for s in rec.get("skills", []):
                srow = {k: _fmt(s.get(k)) for k in sk_cols if k in s}
                srow["error_code"] = ";".join(s.get("codes") or [])
                wk.writerow(srow)
            cinfo = rec.get("cameras") or {}
            rrow = {
                "recording_id": rid,
                "source": rec.get("source"),
                "size_bytes": rec.get("size_bytes"),
                "n_segments": len(rec["segments"]),
                "n_excluded": sum(1 for r in rec["segments"] if r.get("excluded")),
                "n_insert": sum(1 for r in rec["segments"] if r.get("action") == (mapping.target_action or "insert")),
                "status": rec.get("status"),
                "recording_issues": ";".join(rec.get("recording_issues") or []),
            }
            for c in cams:
                ci = cinfo.get(c) or {}
                rrow[f"{c}_frames"] = _fmt(ci.get("frames"))
                rrow[f"{c}_n_timestamps"] = _fmt(ci.get("n_timestamps"))
                rrow[f"{c}_rate_hz"] = _fmt(ci.get("rate_hz"))
            wr.writerow(rrow)
    rec_issue_counts: Counter = Counter(i.split(":")[0] for rec in recs for i in rec.get("recording_issues") or [])
    validation = {
        "generated_at": now_iso(),
        "mapping": {"path": mapping.path, "confirmed_at": mapping.confirmed_at},
        "n_recordings": len(recs),
        "n_failed_recordings": len(errs),
        "segment_status_counts": dict(status_counts),
        "code_counts": dict(code_counts.most_common()),
        "recording_issue_counts": dict(rec_issue_counts.most_common()),
        "failed_recordings": errs,
        "recordings_with_issues": [{"recording_id": r["recording_id"], "source": r.get("source"),
                                    "issues": r.get("recording_issues")} for r in recs if r.get("recording_issues")],
        "items": items,
    }
    val_path = out / "validation.json"
    val_path.write_text(json.dumps(validation, ensure_ascii=False, indent=1), encoding="utf-8")
    rep = write_action_report(recs, errs, out, mapping)
    return {"segments": str(seg_path), "skills": str(sk_path), "recordings": str(rec_path),
            "validation": str(val_path), **rep}


def write_action_report(recs: list[dict], errs: list[dict], out: Path, mapping: Mapping) -> dict[str, str]:
    """動作 × 成敗統計，與目標動作（Insert）各零件的候選數量報表。"""
    target = mapping.target_action or "insert"
    act: dict[str, Counter] = defaultdict(Counter)
    parts: dict[str, dict[str, Any]] = {}
    for rec in recs:
        for r in rec["segments"]:
            if r.get("excluded") or not r.get("action"):
                continue
            key = {True: "success", False: "failure"}.get(r.get("success"), "unknown")
            act[r["action"]][key] += 1
            act[r["action"]]["total"] += 1
            if r["action"] != target:
                continue
            pk = r.get("part") or "(未解析)"
            d = parts.setdefault(pk, {"part": pk, "n_segments": 0, "n_success": 0, "n_failure": 0, "n_unknown": 0,
                                      "recordings": set(), "durations": [], "n_error": 0, "n_warn": 0})
            d["n_segments"] += 1
            d[f"n_{key}"] += 1
            d["recordings"].add(rec["recording_id"])
            if r.get("duration_s") is not None and r["duration_s"] > 0 and r.get("validation_status") != "error":
                d["durations"].append(r["duration_s"])
            if r.get("validation_status") == "error":
                d["n_error"] += 1
            elif r.get("validation_status") == "warn":
                d["n_warn"] += 1
    rows = []
    for d in parts.values():
        n = d["n_segments"]
        rows.append({
            "part": d["part"],
            "n_segments": n,
            "n_success": d["n_success"],
            "n_failure": d["n_failure"],
            "failure_rate": round(d["n_failure"] / n, 3) if n else 0.0,
            "n_recordings": len(d["recordings"]),
            "mean_duration_s": round(float(np.mean(d["durations"])), 2) if d["durations"] else "",
            "test_failures_if_15pct": round(d["n_failure"] * 0.15, 1),
            "n_validation_error": d["n_error"],
            "n_validation_warn": d["n_warn"],
        })
    rows.sort(key=lambda x: (-x["n_failure"], -x["n_segments"], x["part"]))
    csv_path = out / f"{target}_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        cols = ["part", "n_segments", "n_success", "n_failure", "failure_rate", "n_recordings", "mean_duration_s",
                "test_failures_if_15pct", "n_validation_error", "n_validation_warn"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    L = [f"# 動作與 {target.capitalize()} 候選統計（{now_iso()}）", "",
         f"已索引錄製檔：{len(recs)}；失敗：{len(errs)}。被排除的段（{', '.join(sorted(mapping.exclude_text)) or '無'}）不計。", "",
         "## 動作 × 成敗", "", "| 動作 | 段數 | 成功 | 失敗 | 未知 |", "|---|---|---|---|---|"]
    for a, c in sorted(act.items(), key=lambda kv: -kv[1]["total"]):
        L.append(f"| {a} | {c['total']} | {c['success']} | {c['failure']} | {c['unknown']} |")
    L += ["", f"## {target.capitalize()} 各零件（依失敗數排序）", "",
          "規格建議 70/15/15 切分；「test 預估失敗」= 失敗數 × 0.15，太少就要調整或縮小研究問題。", "",
          "| 零件 | 段數 | 成功 | 失敗 | 失敗率 | 錄製檔數 | 平均長度 (s) | test 預估失敗 | 驗證 error / warn |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['part']} | {r['n_segments']} | {r['n_success']} | {r['n_failure']} | {r['failure_rate']:.0%} | "
                 f"{r['n_recordings']} | {r['mean_duration_s']} | {r['test_failures_if_15pct']} | "
                 f"{r['n_validation_error']} / {r['n_validation_warn']} |")
    if errs:
        L += ["", "## 索引失敗的錄製檔", ""] + [f"- {e['recording_id']}：{e.get('error')}" for e in errs]
    L.append("")
    md_path = out / "action_summary.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    return {"candidates": str(csv_path), "action_summary": str(md_path)}
