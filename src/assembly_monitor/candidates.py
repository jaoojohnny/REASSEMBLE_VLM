"""依結構與名稱推測欄位角色，並列出異常與待確認問題（第 1 包 P1-2）。

輸入是 inspect_h5 產生的節點清單（純 dict），不需要 h5py，方便單元測試。
這裡所有結論都只是「候選」：報告會附上證據，交給使用者核對後才寫進正式映射。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .media import AUDIO_TYPES, VIDEO_TYPES
from .util import as_number, fmt_num, is_int_name, natural_key, normalize_id

START_RE = re.compile(r"^(start|begin|t_?start|start_?t(ime)?|start_?ts|onset|t0)$", re.IGNORECASE)
END_RE = re.compile(r"^(end|stop|finish|t_?end|end_?t(ime)?|end_?ts|t1)$", re.IGNORECASE)
SUCCESS_RE = re.compile(r"^(success|succeeded|successful|is_?success|result|outcome|ok|label)$", re.IGNORECASE)
TEXT_RE = re.compile(
    r"^(text|description|desc|caption|instruction|task|name|action|label|language|lang|skill)$", re.IGNORECASE
)
REASON_RE = re.compile(r"(reason|failure|fail_?type|error|evidence|cause)", re.IGNORECASE)
VERB_RE = re.compile(r"^\s*([A-Za-z][A-Za-z_\-]*)\s*(.*?)\s*$")

SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}


def _bool_like(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, float) and v in (0.0, 1.0):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "success", "succeeded", "yes", "ok", "1"):
            return True
        if t in ("false", "failure", "failed", "fail", "no", "0"):
            return False
    return None


def _pattern(path: str) -> str:
    return "/".join("*" if is_int_name(p) else p for p in path.split("/")) or "/"


def _index(nodes: list[dict]) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    by_path = {n["path"]: n for n in nodes}
    children: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        if n["path"] == "/":
            continue
        parent = n["path"].rsplit("/", 1)[0] or "/"
        children[parent].append(n)
    for k in children:
        children[k].sort(key=lambda c: natural_key(c["name"]))
    return by_path, children


def _scalars(children: dict[str, list[dict]], gpath: str) -> dict[str, dict]:
    return {
        c["name"]: c
        for c in children.get(gpath, [])
        if c["kind"] == "dataset" and c.get("shape") == [] and c.get("media") is None
    }


def _anomaly(code: str, severity: str, message: str, path: str | None = None, **extra: Any) -> dict[str, Any]:
    d = {"code": code, "severity": severity, "message": message}
    if path:
        d["path"] = path
    d.update(extra)
    return d


# ---------------------------------------------------------------- segment 容器


def find_segment_containers(by_path: dict, children: dict) -> list[dict[str, Any]]:
    """數字命名子群組 + 子群組內有起訖純量 → segment 容器（不看容器本身叫什麼）。"""
    containers = []
    for g in by_path.values():
        if g["kind"] != "group":
            continue
        kids = children.get(g["path"], [])
        int_groups = [c for c in kids if c["kind"] == "group" and is_int_name(c["name"])]
        if not int_groups or len(int_groups) < 0.6 * len(kids):
            continue
        start_votes: Counter = Counter()
        end_votes: Counter = Counter()
        n_both = 0
        for s in int_groups:
            sc = _scalars(children, s["path"])
            sk = sorted(k for k in sc if START_RE.match(k))
            ek = sorted(k for k in sc if END_RE.match(k))
            if sk and ek:
                n_both += 1
                start_votes[sk[0]] += 1
                end_votes[ek[0]] += 1
        if n_both == 0 or n_both < 0.5 * len(int_groups):
            continue
        start_key = start_votes.most_common(1)[0][0]
        end_key = end_votes.most_common(1)[0][0]
        succ_votes: Counter = Counter()
        text_votes: Counter = Counter()
        bool_any: Counter = Counter()
        str_any: Counter = Counter()
        reason_like: Counter = Counter()
        other: Counter = Counter()
        for s in int_groups:
            for k, c in _scalars(children, s["path"]).items():
                if k in (start_key, end_key):
                    continue
                v = c.get("preview")
                if REASON_RE.search(k):
                    reason_like[k] += 1
                if SUCCESS_RE.match(k) and _bool_like(v) is not None:
                    succ_votes[k] += 1
                elif TEXT_RE.match(k) and isinstance(v, str):
                    text_votes[k] += 1
                elif isinstance(v, bool):
                    bool_any[k] += 1
                elif isinstance(v, str):
                    str_any[k] += 1
                else:
                    other[k] += 1
        success_key = (succ_votes or bool_any).most_common(1)[0][0] if (succ_votes or bool_any) else None
        text_key = (text_votes or str_any).most_common(1)[0][0] if (text_votes or str_any) else None
        used = {start_key, end_key, success_key, text_key}
        extra_keys = sorted(
            {k for cnt in (succ_votes, text_votes, bool_any, str_any, other) for k in cnt} - used, key=natural_key
        )
        sub_groups: Counter = Counter(
            c["name"] for s in int_groups for c in children.get(s["path"], []) if c["kind"] == "group"
        )
        containers.append(
            {
                "path": g["path"],
                "pattern": _pattern(g["path"]),
                "n_segments": len(int_groups),
                "n_with_start_end": n_both,
                "keys": {
                    "start": start_key,
                    "end": end_key,
                    "success": success_key,
                    "text": text_key,
                    "extra": extra_keys,
                    "success_by_name": bool(succ_votes),
                    "text_by_name": bool(text_votes),
                },
                "reason_like_keys": sorted(reason_like, key=natural_key),
                "sub_groups": dict(sub_groups),
                "non_segment_children": [c["name"] for c in kids if not (c["kind"] == "group" and is_int_name(c["name"]))],
            }
        )
    paths = [c["path"] for c in containers]
    for c in containers:
        ancestors = [q for q in paths if q != c["path"] and c["path"].startswith(q.rstrip("/") + "/")]
        c["level"] = len(ancestors)
        if ancestors:
            nearest = max(ancestors, key=len)
            rest = c["path"][len(nearest.rstrip("/")) + 1 :].split("/")
            c["parent_segment_path"] = f"{nearest.rstrip('/')}/{rest[0]}"
        else:
            c["parent_segment_path"] = None
    containers.sort(key=lambda c: (c["level"], natural_key(c["path"])))
    return containers


def summarize_patterns(containers: list[dict]) -> list[dict[str, Any]]:
    """把同一路徑樣式（例如 /segments_info/*/Low_level）的多個容器合併成一筆。"""
    pats: dict[str, dict[str, Any]] = {}
    for c in containers:
        p = pats.setdefault(c["pattern"], {
            "pattern": c["pattern"], "level": c["level"], "n_containers": 0, "n_segments": 0,
            "votes": Counter(), "sub_groups": Counter(), "non_segment_children": set(), "reason_like_keys": set(),
        })
        p["n_containers"] += 1
        p["n_segments"] += c["n_segments"]
        k = c["keys"]
        p["votes"][(k["start"], k["end"], k["success"], k["text"], tuple(k["extra"]))] += 1
        p["sub_groups"].update(c["sub_groups"])
        p["non_segment_children"].update(c["non_segment_children"])
        p["reason_like_keys"].update(c["reason_like_keys"])
    out = []
    for p in pats.values():
        (st, en, su, tx, ex), _ = p["votes"].most_common(1)[0]
        out.append({
            "pattern": p["pattern"],
            "level": p["level"],
            "n_containers": p["n_containers"],
            "n_segments": p["n_segments"],
            "keys": {"start": st, "end": en, "success": su, "text": tx, "extra": list(ex)},
            "keys_consistent": len(p["votes"]) == 1,
            "sub_groups": dict(p["sub_groups"]),
            "non_segment_children": sorted(p["non_segment_children"], key=natural_key),
            "reason_like_keys": sorted(p["reason_like_keys"], key=natural_key),
        })
    out.sort(key=lambda x: (x["level"], natural_key(x["pattern"])))
    return out


def extract_segment_rows(containers: list[dict], children: dict) -> list[dict[str, Any]]:
    rows = []
    for c in containers:
        keys = c["keys"]
        for s in children.get(c["path"], []):
            if not (s["kind"] == "group" and is_int_name(s["name"])):
                continue
            sc = _scalars(children, s["path"])
            start = as_number(sc[keys["start"]]["preview"]) if keys["start"] in sc else None
            end = as_number(sc[keys["end"]]["preview"]) if keys["end"] in sc else None
            succ_raw = sc[keys["success"]]["preview"] if keys["success"] and keys["success"] in sc else None
            text = sc[keys["text"]]["preview"] if keys["text"] and keys["text"] in sc else None
            if text is not None and not isinstance(text, str):
                text = str(text)
            rows.append(
                {
                    "container": c["path"],
                    "pattern": c["pattern"],
                    "level": c["level"],
                    "id": s["name"],
                    "path": s["path"],
                    "parent_path": c["parent_segment_path"],
                    "start": start,
                    "end": end,
                    "duration": (end - start) if start is not None and end is not None else None,
                    "success": _bool_like(succ_raw),
                    "success_raw": succ_raw,
                    "text": text,
                    "extra": {k: sc[k]["preview"] for k in keys["extra"][:12] if k in sc},
                }
            )
    return rows


def action_stats(rows: list[dict]) -> dict[str, Any]:
    """text 第一個詞當動作候選、其餘當零件候選，分層統計成功 / 失敗。"""
    out: dict[str, Any] = {}
    for r in rows:
        if not r.get("text"):
            continue
        m = VERB_RE.match(r["text"])
        if not m:
            continue
        verb, obj = m.group(1).lower(), m.group(2)
        lvl = out.setdefault(f"level{r['level']}", {})
        v = lvl.setdefault(verb, {"count": 0, "success_true": 0, "success_false": 0, "success_unknown": 0, "objects": {}})
        o = v["objects"].setdefault(obj or "(無)", {"count": 0, "success_true": 0, "success_false": 0, "success_unknown": 0})
        key = {True: "success_true", False: "success_false"}.get(r.get("success"), "success_unknown")
        for d in (v, o):
            d["count"] += 1
            d[key] += 1
    return out


# ---------------------------------------------------------------- 時間軸與相機


def _pair_score(stream_name: str, time_name: str) -> int:
    s, t = stream_name.lower(), time_name.lower()
    if s == t:
        return 3
    if t.startswith(s + "_") or t.endswith("_" + s) or t.startswith(s + "-"):
        return 2
    if min(len(s), len(t)) >= 3 and (s in t or t in s):
        return 1
    return 0


def _best_time_pair(stream: dict, time_nodes: list[dict]) -> tuple[dict | None, int]:
    best, best_score = None, 0
    for t in time_nodes:
        if t["path"] == stream["path"]:
            continue
        sc = _pair_score(stream["name"], t["name"])
        if sc == 0:
            continue
        if sc > best_score or (sc == best_score and best is not None and len(t["path"]) < len(best["path"])):
            best, best_score = t, sc
    return best, best_score


def _ts_brief(t: dict | None) -> dict[str, Any]:
    if not t:
        return {}
    st = t["time_stats"]
    unit = st.get("unit_guess") or {}
    return {
        "timestamps": t["path"],
        "ts_n": st.get("n"),
        "ts_first": st.get("first"),
        "ts_last": st.get("last"),
        "ts_rate_hz": st.get("rate_hz"),
        "ts_span_s": st.get("span_s"),
        "ts_unit_guess": unit.get("unit"),
        "ts_base_guess": unit.get("base"),
        "ts_monotonic": st.get("monotonic_non_decreasing"),
        "ts_sampled": st.get("sampled"),
    }


def analyze_streams(nodes: list[dict], expected_cameras: list[str]) -> tuple[dict[str, Any], list[dict]]:
    anomalies: list[dict] = []
    time_nodes = [
        n for n in nodes if n.get("time_stats") and not n["time_stats"].get("error") and n["time_stats"].get("n")
    ]
    time_paths = {t["path"] for t in time_nodes}
    video = [n for n in nodes if (n.get("media") or {}).get("type") in VIDEO_TYPES]
    audio = [n for n in nodes if (n.get("media") or {}).get("type") in AUDIO_TYPES]
    unsniffed = [n for n in nodes if (n.get("media") or {}).get("access") == "unavailable"]
    image_arrays = [
        n
        for n in nodes
        if n["kind"] == "dataset"
        and n.get("dtype") == "uint8"
        and n.get("shape")
        and len(n["shape"]) >= 3
        and n.get("media") is None
    ]

    cameras = []
    for v in video:
        probe = v["media"].get("probe") or {}
        summ = probe.get("summary") or {}
        t, score = _best_time_pair(v, time_nodes)
        e: dict[str, Any] = {
            "name": v["name"],
            "path": v["path"],
            "encoding": v["media"]["type"],
            "bytes": v["media"].get("bytes"),
            "access": v["media"].get("access"),
            "probe_ok": probe.get("ok"),
            "probe_backend": summ.get("backend"),
            "codec": summ.get("codec"),
            "width": summ.get("width"),
            "height": summ.get("height"),
            "fps_nominal": summ.get("fps_nominal"),
            "frames": summ.get("frames"),
            "frames_source": summ.get("frames_source"),
            "duration_s": summ.get("duration_s"),
            "timestamps_match_score": score,
        }
        e.update(_ts_brief(t))
        if probe and not probe.get("ok", False):
            e["probe_errors"] = probe.get("errors")
            anomalies.append(
                _anomaly("CAM_PROBE_FAILED", "warn", f"影片 {v['path']} 無法讀取中繼資料：{probe.get('errors')}", v["path"])
            )
        if t is None:
            anomalies.append(
                _anomaly("CAM_NO_TIMESTAMPS", "warn", f"影片 {v['path']} 找不到名稱對應的時間陣列", v["path"])
            )
        frames, ts_n = e.get("frames"), e.get("ts_n")
        if frames and ts_n:
            e["frames_minus_ts"] = int(frames) - int(ts_n)
            if frames != ts_n:
                anomalies.append(
                    _anomaly(
                        "CAM_FRAMES_TS_MISMATCH",
                        "warn",
                        f"{v['name']}：影片 {frames} 幀，但 {e['timestamps']} 有 {ts_n} 筆時間",
                        v["path"],
                    )
                )
        fps, rate = e.get("fps_nominal"), e.get("ts_rate_hz")
        if fps and rate and abs(fps - rate) / max(fps, rate) > 0.1:
            anomalies.append(
                _anomaly(
                    "CAM_FPS_VS_TS_RATE",
                    "info",
                    f"{v['name']}：影片標稱 {fps:.3f} fps，但時間戳實測約 {rate:.3f} Hz；秒數必須用時間戳換算，不能用 frame_idx/fps",
                    v["path"],
                )
            )
        cameras.append(e)

    audio_entries = []
    for a in audio:
        probe = a["media"].get("probe") or {}
        summ = probe.get("summary") or {}
        t, score = _best_time_pair(a, time_nodes)
        if score < 2:  # 例如 hama1_audio 不應配到影片的 /timestamps/hama1
            t, score = None, 0
        e = {
            "name": a["name"],
            "path": a["path"],
            "encoding": a["media"]["type"],
            "bytes": a["media"].get("bytes"),
            "probe_ok": probe.get("ok"),
            "sample_rate": summ.get("sample_rate"),
            "channels": summ.get("channels"),
            "duration_s": summ.get("audio_duration_s"),
            "timestamps_match_score": score,
        }
        e.update(_ts_brief(t))
        if a["media"].get("note"):
            e["storage_note"] = a["media"]["note"]
        audio_entries.append(e)

    for n in unsniffed:
        anomalies.append(
            _anomaly("MEDIA_NOT_SNIFFED", "info", f"{n['path']}：{(n.get('media') or {}).get('note')}", n["path"])
        )

    expected_missing = []
    names_lower = {c["name"].lower() for c in cameras}
    for cam in expected_cameras or []:
        if cam.lower() not in names_lower:
            expected_missing.append(cam)
            anomalies.append(
                _anomaly("CAM_EXPECTED_MISSING", "error", f"設定的預期相機 {cam} 不在這個檔案的影片清單中", None, camera=cam)
            )

    # 其他資料（機器人狀態、事件相機等）與時間軸的名稱配對
    others = []
    data_nodes = [
        n
        for n in nodes
        if n["kind"] == "dataset" and n["path"] not in time_paths and n.get("shape") and n.get("media") is None
    ]
    for d in data_nodes:
        t, score = _best_time_pair(d, time_nodes)
        if t is None or score < 3:
            continue
        e = {"name": d["name"], "path": d["path"], "shape": d["shape"], "dtype": d["dtype"]}
        e.update(_ts_brief(t))
        e["length_match"] = bool(d["shape"][0] == t["time_stats"].get("n"))
        others.append(e)

    # 時間陣列也可能對應整個群組（例如 timestamps/robot_state ↔ /robot_state）
    group_nodes = [n for n in nodes if n["kind"] == "group" and n["path"] != "/"]
    for t in time_nodes:
        for g in group_nodes:
            if g["name"].lower() != t["name"].lower() or t["path"].startswith(g["path"].rstrip("/") + "/"):
                continue
            kids = [
                n for n in nodes
                if n["kind"] == "dataset" and n["path"].startswith(g["path"] + "/") and n.get("shape")
            ]
            lens = {k["shape"][0] for k in kids}
            e = {"name": g["name"], "path": g["path"], "kind": "group", "n_datasets": len(kids)}
            e.update(_ts_brief(t))
            e["length_match"] = bool(kids) and lens == {t["time_stats"].get("n")}
            others.append(e)

    paired_ts = {c.get("timestamps") for c in cameras + audio_entries + others}
    stream_names = {n["name"].lower() for n in video + audio + data_nodes + group_nodes}
    unpaired = []
    for t in time_nodes:
        if t["path"] in paired_ts:
            continue
        if t["name"].lower() in stream_names:
            continue
        unpaired.append(t["path"])
        anomalies.append(
            _anomaly(
                "TS_UNPAIRED",
                "warn",
                f"時間陣列 {t['path']} 找不到同名的影片或資料（可能缺少視角或資料）",
                t["path"],
            )
        )

    time_arrays = []
    for t in time_nodes:
        st = t["time_stats"]
        time_arrays.append({"path": t["path"], **{k: st.get(k) for k in (
            "n", "first", "last", "min", "max", "monotonic_non_decreasing", "n_backward_steps",
            "n_repeated", "median_dt", "median_dt_s", "rate_hz", "span_s", "n_gaps_gt_3x_median",
            "max_dt_s", "first_utc", "last_utc", "sampled", "note")}, "unit_guess": st.get("unit_guess")})
        if st.get("n_backward_steps"):
            anomalies.append(
                _anomaly(
                    "TS_NON_MONOTONIC",
                    "error",
                    f"{t['path']} 有 {st['n_backward_steps']} 處時間倒退",
                    t["path"],
                )
            )
        if st.get("n_nonfinite"):
            anomalies.append(_anomaly("TS_NONFINITE", "warn", f"{t['path']} 有 {st['n_nonfinite']} 個 NaN/Inf", t["path"]))
        if st.get("n_gaps_gt_3x_median"):
            anomalies.append(
                _anomaly(
                    "TS_GAPS",
                    "info",
                    f"{t['path']} 有 {st['n_gaps_gt_3x_median']} 處間隔大於中位數 3 倍（最大 {fmt_num(st.get('max_dt_s'))} 秒）",
                    t["path"],
                )
            )

    result = {
        "cameras": cameras,
        "audio": audio_entries,
        "image_arrays": [{"path": n["path"], "shape": n["shape"]} for n in image_arrays],
        "other_streams": others,
        "time_arrays": time_arrays,
        "unpaired_time_arrays": unpaired,
        "expected_cameras_missing": expected_missing,
    }
    return result, anomalies


# ---------------------------------------------------------------- 時間基準


def analyze_time_base(rows: list[dict], time_arrays: list[dict], preferred: set[str] | None = None) -> dict[str, Any]:
    seg = [r for r in rows if r["level"] == 0 and r["start"] is not None and r["end"] is not None]
    if not seg:
        return {"relation": "unknown", "reason": "沒有可用的 segment 起訖數值"}
    if not time_arrays:
        return {"relation": "unknown", "reason": "沒有時間陣列可比對"}
    smin = min(r["start"] for r in seg)
    smax = max(r["end"] for r in seg)
    candidates = []
    for t in time_arrays:
        if t.get("min") is None or t.get("max") is None:
            continue
        tmin, tmax = t["min"], t["max"]
        tol = 0.01 * (tmax - tmin) + 1e-9
        same = sum(1 for r in seg if r["start"] >= tmin - tol and r["end"] <= tmax + tol) / len(seg)
        off = t.get("first") or tmin
        rel = sum(1 for r in seg if r["start"] + off >= tmin - tol and r["end"] + off <= tmax + tol) / len(seg)
        candidates.append({"timestamps": t["path"], "fraction_inside_same": round(same, 3),
                           "fraction_inside_relative": round(rel, 3), "ts_min": tmin, "ts_max": tmax,
                           "unit_guess": (t.get("unit_guess") or {}).get("unit")})
    if not candidates:
        return {"relation": "unknown", "reason": "時間陣列沒有有效範圍"}
    pref = preferred or set()
    best_same = max(candidates, key=lambda c: (c["fraction_inside_same"], c["timestamps"] in pref, -len(c["timestamps"])))
    best_rel = max(candidates, key=lambda c: (c["fraction_inside_relative"], c["timestamps"] in pref, -len(c["timestamps"])))
    base: dict[str, Any] = {"segment_min": smin, "segment_max": smax, "n_segments": len(seg), "candidates": candidates}
    if best_same["fraction_inside_same"] >= 0.6:
        base.update(relation="same_as", reference=best_same["timestamps"],
                    fraction_inside=best_same["fraction_inside_same"], unit_guess=best_same["unit_guess"])
    elif best_rel["fraction_inside_relative"] >= 0.6:
        base.update(relation="relative_to_start_of", reference=best_rel["timestamps"],
                    fraction_inside=best_rel["fraction_inside_relative"], unit_guess=best_rel["unit_guess"])
    else:
        base.update(relation="unknown", reason="segment 範圍與任何時間陣列都對不上")
    return base


def segment_anomalies(rows: list[dict], containers: list[dict], time_base: dict, time_arrays: list[dict]) -> list[dict]:
    out: list[dict] = []
    for c in containers:
        mine = [r for r in rows if r["container"] == c["path"]]
        ids = Counter(normalize_id(r["id"]) for r in mine)
        for nid, cnt in ids.items():
            if cnt > 1:
                raw = [r["id"] for r in mine if normalize_id(r["id"]) == nid]
                out.append(_anomaly("SEG_DUPLICATE_ID", "error", f"{c['path']} 內 ID {raw} 正規化後重複", c["path"], ids=raw))
        seen: dict[tuple, str] = {}
        for r in mine:
            key = (r["start"], r["end"], r["text"])
            if r["start"] is None or r["end"] is None:
                continue
            if key in seen:
                out.append(_anomaly("SEG_DUPLICATE_INTERVAL", "warn",
                                    f"{r['path']} 與 {seen[key]} 的起訖與文字完全相同", r["path"]))
            else:
                seen[key] = r["path"]
        if not c["keys"].get("success"):
            out.append(_anomaly("SEG_NO_SUCCESS_FIELD", "info", f"{c['path']} 找不到像 success 的布林欄位", c["path"]))
        if not c["keys"].get("text"):
            out.append(_anomaly("SEG_NO_TEXT_FIELD", "info", f"{c['path']} 找不到動作描述文字欄位", c["path"]))
    by_path = {r["path"]: r for r in rows}
    ref = None
    if time_base.get("relation") == "same_as":
        ref = next((t for t in time_arrays if t["path"] == time_base.get("reference")), None)
    for r in rows:
        if r["start"] is None or r["end"] is None:
            out.append(_anomaly("SEG_MISSING_TIME", "error", f"{r['path']} 缺少可讀的起訖數值", r["path"]))
            continue
        if r["start"] >= r["end"]:
            out.append(_anomaly("SEG_START_GE_END", "error",
                                f"{r['path']} 起點 {fmt_num(r['start'])} ≥ 終點 {fmt_num(r['end'])}", r["path"]))
        if ref is not None:
            tol = 0.01 * (ref["max"] - ref["min"]) + 1e-9
            if r["start"] < ref["min"] - tol or r["end"] > ref["max"] + tol:
                out.append(_anomaly("SEG_OUT_OF_RANGE", "warn",
                                    f"{r['path']} 區間超出 {ref['path']} 的時間範圍", r["path"]))
        if r["success_raw"] is not None and r["success"] is None:
            out.append(_anomaly("SEG_SUCCESS_NOT_BOOL", "warn", f"{r['path']} 的 success 值不是布林：{r['success_raw']!r}", r["path"]))
        parent = by_path.get(r["parent_path"]) if r.get("parent_path") else None
        if parent and parent["start"] is not None and parent["end"] is not None:
            # 容差取上層長度的 0.1%，避免 Unix 時間這種大數值放大誤差
            eps = 1e-3 * max(abs(parent["end"] - parent["start"]), 1e-12)
            if r["start"] < parent["start"] - eps or r["end"] > parent["end"] + eps:
                out.append(_anomaly("SEG_CHILD_OUTSIDE_PARENT", "warn",
                                    f"{r['path']} 不在上層 segment {parent['path']} 的起訖內", r["path"]))
    top = sorted((r for r in rows if r["level"] == 0 and r["start"] is not None and r["end"] is not None),
                 key=lambda r: (r["container"], r["start"]))
    for a, b in zip(top, top[1:]):
        eps = 1e-4 * max(min(abs(a["end"] - a["start"]), abs(b["end"] - b["start"])), 1e-12)
        if a["container"] == b["container"] and b["start"] < a["end"] - eps and a["start"] < a["end"]:
            out.append(_anomaly("SEG_OVERLAP", "info", f"{a['path']} 與 {b['path']} 時間重疊", b["path"]))
    return out


# ---------------------------------------------------------------- 人工核對樣本與問題


def pick_manual_check(rows: list[dict], k: int = 5) -> list[dict]:
    """挑 k 段高層 segment 給人工核對：優先 Insert，其次平均分散。"""
    top = [r for r in rows if r["level"] == 0]
    if not top:
        return []
    ins = [r for r in top if (r.get("text") or "").strip().lower().startswith("insert")]
    picked: list[dict] = []
    for r in ins:
        if len(picked) >= k:
            break
        picked.append(r)
    rest = [r for r in top if r not in picked]
    need = k - len(picked)
    if need > 0 and rest:
        step = max(1, len(rest) // need)
        picked.extend(rest[::step][:need])
    return sorted(picked, key=lambda r: (r["start"] is None, r["start"] or 0))


def build_questions(c: dict[str, Any]) -> list[dict[str, Any]]:
    qs: list[dict[str, Any]] = []

    def add(topic: str, question: str, evidence: Any = None) -> None:
        qs.append({"id": f"Q{len(qs) + 1}", "topic": topic, "question": question, "evidence": evidence})

    conts = c["segments"]["patterns"]
    if conts:
        for ct in conts:
            k = ct["keys"]
            where = f"共 {ct['n_segments']} 段" + (f"，分散在 {ct['n_containers']} 個容器" if ct["n_containers"] > 1 else "")
            note = "" if ct["keys_consistent"] else "（各容器鍵名不完全一致，見 candidates.json）"
            add(
                "segment 欄位",
                f"`{ct['pattern']}`（第 {ct['level']} 層，{where}）：start=`{k['start']}`、end=`{k['end']}`、"
                f"success=`{k['success']}`、text=`{k['text']}`，其他欄位 {k['extra'] or '無'}{note}。"
                "這些是否就是 segment 起訖、成功標記與動作描述？",
                {"sub_groups": ct["sub_groups"], "non_segment_children": ct["non_segment_children"]},
            )
    else:
        add("segment 欄位", "沒有自動找到 segment 容器（數字子群組 + 起訖純量）。segment 資訊實際存在哪裡？", None)

    tb = c["time_base"]
    if tb.get("relation") == "same_as":
        add(
            "時間單位與基準",
            f"segment 起訖範圍 {fmt_num(tb.get('segment_min'))}–{fmt_num(tb.get('segment_max'))} 落在 `{tb['reference']}` "
            f"的範圍內（{tb['fraction_inside']:.0%}），推測兩者同一時間基準、單位 {tb.get('unit_guess')}。請確認單位與是否為 Unix 時間。",
            tb.get("candidates"),
        )
    elif tb.get("relation") == "relative_to_start_of":
        add(
            "時間單位與基準",
            f"segment 起訖像是相對於 `{tb['reference']}` 起點的偏移（{tb['fraction_inside']:.0%} 吻合）。請確認單位與起點定義。",
            tb.get("candidates"),
        )
    else:
        add("時間單位與基準", f"無法判斷 segment 時間與相機時間戳的關係：{tb.get('reason')}。請說明時間單位與基準。", tb.get("candidates"))

    acts = c["segments"]["actions"]
    lvl0 = acts.get("level0") or {}
    if lvl0:
        dist = {v: d["count"] for v, d in sorted(lvl0.items(), key=lambda kv: -kv[1]["count"])}
        add("action / part 解析", f"高層 text 第一個詞的分佈：{dist}。是否以第一個詞當 action、其餘字串當零件（part）？", None)
        ins = lvl0.get("insert")
        if ins:
            objs = {o: f"{d['count']} 段（成功 {d['success_true']} / 失敗 {d['success_false']}）"
                    for o, d in sorted(ins["objects"].items(), key=lambda kv: -kv[1]["count"])}
            add("Insert 零件候選", f"這些樣本中的 Insert 零件：{objs}。研究要選哪一種？（失敗數需等第 2 包全量索引才準）", None)
    reason_keys = sorted({k for ct in conts for k in ct.get("reason_like_keys", [])})
    if reason_keys:
        add("失敗原因欄位", f"發現名稱像失敗原因或證據的欄位 {reason_keys}，請確認其意義。", None)
    else:
        add("失敗原因與證據時間", "資料中沒有失敗原因、證據時間欄位（符合規格預期）。這些將在第 4 包走人工標註流程，請確認。", None)
    if any(ct["level"] >= 1 for ct in conts):
        low = acts.get("level1") or {}
        add("低階 segment", f"第 1 層（如 Low_level）動作分佈：{ {v: d['count'] for v, d in low.items()} }。要不要拿來當證據時間的候選區間？", None)

    cams = c["streams"]["cameras"]
    if cams:
        brief = {cm["name"]: f"{cm.get('width')}x{cm.get('height')}, {fmt_num(cm.get('fps_nominal'))} fps, {cm.get('frames')} 幀" for cm in cams}
        add("相機清單", f"偵測到的內嵌影片：{brief}。哪些是外部視角、哪個是腕部相機？best_camera 的候選集合為何？", None)
        for cm in cams:
            if cm.get("timestamps"):
                add(
                    "影格與時間戳",
                    f"{cm['name']}：影片 {cm.get('frames')} 幀、標稱 {fmt_num(cm.get('fps_nominal'))} fps；`{cm['timestamps']}` 有 {cm.get('ts_n')} 筆，"
                    f"實測約 {fmt_num(cm.get('ts_rate_hz'))} Hz。第 i 幀是否對應第 i 個時間戳？",
                    {"frames_minus_ts": cm.get("frames_minus_ts")},
                )
    else:
        add("相機清單", "沒有偵測到內嵌影片（mp4 等）。影像存在哪裡、什麼格式？", c["streams"].get("image_arrays"))
    if c["streams"].get("unpaired_time_arrays"):
        add("未配對時間軸", f"這些時間陣列找不到同名資料：{c['streams']['unpaired_time_arrays']}。是否代表缺少視角或資料？", None)

    errs = [a for a in c["anomalies"] if a["severity"] in ("error", "warn")]
    if errs:
        add("異常需確認", f"共 {len(errs)} 項 error/warn（見 candidates.json 的 anomalies），請確認是資料問題還是解讀錯誤。",
            [a["code"] for a in errs[:20]])
    return qs


def build_candidates(nodes: list[dict], opts: Any) -> dict[str, Any]:
    by_path, children = _index(nodes)
    containers = find_segment_containers(by_path, children)
    rows = extract_segment_rows(containers, children)
    streams, stream_anoms = analyze_streams(nodes, list(getattr(opts, "expected_cameras", []) or []))
    cam_ts = {cm.get("timestamps") for cm in streams["cameras"] if cm.get("timestamps")}
    time_base = analyze_time_base(rows, streams["time_arrays"], cam_ts)
    seg_anoms = segment_anomalies(rows, containers, time_base, streams["time_arrays"])
    anomalies = sorted(stream_anoms + seg_anoms, key=lambda a: (SEVERITY_ORDER.get(a["severity"], 9), a["code"]))
    result: dict[str, Any] = {
        "segments": {
            "patterns": summarize_patterns(containers),
            "containers": containers,
            "rows": rows,
            "n_level0": sum(1 for r in rows if r["level"] == 0),
            "n_level1plus": sum(1 for r in rows if r["level"] >= 1),
            "actions": action_stats(rows),
        },
        "streams": streams,
        "time_base": time_base,
        "anomalies": anomalies,
        "manual_check": pick_manual_check(rows),
    }
    result["questions"] = build_questions(result)
    return result
