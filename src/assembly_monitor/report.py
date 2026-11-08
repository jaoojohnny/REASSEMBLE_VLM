"""把檢查結果寫成人可讀與機器可讀的檔案。

每個 HDF5 一個資料夾：tree.txt、nodes.json、candidates.json、questions.md、
field_mapping.proposed.yaml；多檔時另寫 _summary.md / _summary.json。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .candidates import _index
from .util import fmt_num, human_bytes, is_int_name, now_iso

UNIT_TO_S = {"s": 1.0, "s?": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def _compact(v: Any) -> Any:
    """顯示用：浮點數縮短（小數字 6 位有效數字；Unix 時間這類大數字保留到微秒）。"""
    if isinstance(v, float):
        return round(v, 6) if abs(v) >= 1e5 else float(f"{v:.6g}")
    if isinstance(v, list):
        return [_compact(x) for x in v]
    if isinstance(v, dict):
        return {k: _compact(x) for k, x in v.items()}
    return v


def _short(v: Any, limit: int = 160) -> str:
    s = json.dumps(_compact(v), ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _shape_str(shape: list[int] | None) -> str:
    if shape is None:
        return "null"
    if not shape:
        return "()"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(s) for s in shape) + ")"


# ---------------------------------------------------------------- tree.txt


def _node_line(n: dict) -> str:
    kind = n["kind"]
    if kind in ("group", "softlink"):
        s = f"{n['name']} [G] {n.get('n_members', 0)} 項"
        if n.get("link_target"):
            s += f"（軟連結 → {n['link_target']}）"
        return s
    if kind == "extlink":
        return f"{n['name']} [L] 外部連結 → {n.get('link_target')}"
    if kind in ("unresolved", "datatype"):
        return f"{n['name']} [?] {kind} {n.get('error') or ''}".rstrip()
    parts = [f"{n['name']} [D] shape={_shape_str(n.get('shape'))} {n.get('dtype')}"]
    if n.get("layout") and n["layout"] != "contiguous":
        parts.append(f"layout={n['layout']}")
    if n.get("chunks"):
        parts.append(f"chunks={_shape_str(n['chunks'])}")
    if n.get("compression"):
        opt = n.get("compression_opts")
        parts.append(f"{n['compression']}({opt})" if opt is not None else str(n["compression"]))
    sb = n.get("storage_bytes")
    if sb is not None and (sb >= 1 << 20 or n.get("compression")):
        parts.append(f"儲存 {human_bytes(sb)}")
    if n.get("link_target"):
        parts.append(f"（軟連結 → {n['link_target']}）")
    if n.get("shape") == [] and n.get("preview") is not None and n.get("media") is None:
        parts.append(f"= {_short(n['preview'], 120)}")
    return "  ".join(parts)


def _media_line(m: dict) -> str:
    t = m.get("type")
    if t is None:
        extra = m.get("note") or m.get("error") or (f"檔頭 {m.get('head_hex')}" if m.get("head_hex") else "")
        return f"內嵌位元組 {human_bytes(m.get('bytes'))}，未辨識格式 {extra}".rstrip()
    probe = m.get("probe")
    head = f"偵測為 {t}（{human_bytes(m.get('bytes'))}，讀取方式 {m.get('access')}）"
    if m.get("note"):
        head += f"［{m['note']}］"
    if not probe:
        return head
    if not probe.get("ok"):
        return f"{head}；探測失敗：{probe.get('errors')}"
    s = probe.get("summary") or {}
    bits = []
    if s.get("codec") or s.get("width"):
        bits.append(f"{s.get('codec')} {s.get('width')}x{s.get('height')}")
    if s.get("fps_nominal"):
        bits.append(f"標稱 {fmt_num(s['fps_nominal'])} fps")
    if s.get("frames"):
        bits.append(f"{s['frames']} 幀（{s.get('frames_source')}）")
    if s.get("duration_s"):
        bits.append(f"{fmt_num(s['duration_s'])} s")
    if s.get("sample_rate"):
        bits.append(f"{s['sample_rate']} Hz, {s.get('channels')} ch")
    if s.get("audio_duration_s") and not s.get("duration_s"):
        bits.append(f"{fmt_num(s['audio_duration_s'])} s")
    bits.append(f"[{s.get('backend')}]")
    return f"{head}：" + "，".join(bits)


def _time_line(st: dict) -> str:
    if st.get("error"):
        return f"時間軸統計失敗：{st['error']}"
    if st.get("empty") or st.get("all_nonfinite"):
        return "時間軸：空或全部非有限值"
    u = st.get("unit_guess") or {}
    base = {"unix_epoch": "Unix 時間", "relative?": "相對時間？", "unknown": "未知基準"}.get(u.get("base"), u.get("base"))
    s = f"時間軸：n={st.get('n')}，{fmt_num(st.get('first'))} → {fmt_num(st.get('last'))}（推測單位 {u.get('unit')}、{base}"
    if st.get("first_utc"):
        s += f"，{st['first_utc']} 起"
    s += "）"
    if st.get("median_dt_s"):
        s += f"，中位間隔 {st['median_dt_s'] * 1000:.3f} ms ≈ {st['rate_hz']:.3f} Hz"
    s += "，單調不減" if st.get("monotonic_non_decreasing") else f"，倒退 {st.get('n_backward_steps')} 次"
    if st.get("n_gaps_gt_3x_median"):
        s += f"，大間隔 {st['n_gaps_gt_3x_median']} 處"
    if st.get("sampled"):
        s += "（抽樣統計）"
    return s


def _detail_lines(n: dict) -> list[str]:
    out = []
    if n.get("media"):
        out.append(_media_line(n["media"]))
    if n.get("time_stats"):
        out.append(_time_line(n["time_stats"]))
    pv = n.get("preview")
    if n["kind"] == "dataset" and n.get("shape") and pv is not None and not n.get("time_stats"):
        if isinstance(pv, dict) and "head" in pv:
            s = f"前 {len(pv['head'])} 筆 {_short(pv['head'], 110)}"
            if pv.get("tail"):
                s += f" … 末 {len(pv['tail'])} 筆 {_short(pv['tail'], 80)}"
            out.append(s)
        elif isinstance(pv, dict) and "corner" in pv:
            out.append(f"角落 [{', '.join(pv['corner_index'])}] {_short(pv['corner'], 140)}")
    if n.get("preview_note") and not n.get("media"):
        out.append(n["preview_note"])
    if n.get("attrs"):
        out.append(f"attrs {_short(n['attrs'], 160)}")
    if n.get("error"):
        out.append(f"錯誤：{n['error']}")
    return out


def _collapse(kids: list[dict], children: dict, threshold: int) -> list[Any]:
    if threshold <= 0 or len(kids) <= threshold:
        return list(kids)

    def sig(n: dict) -> tuple:
        if n["kind"] != "group":
            return ("<leaf>",)
        return tuple(sorted(c["name"] for c in children.get(n["path"], [])))

    groups = [k for k in kids if k["kind"] == "group"]
    if not groups:
        return list(kids)
    common, cnt = Counter(sig(g) for g in groups).most_common(1)[0]
    if cnt < 0.8 * len(kids):
        return list(kids)
    same = [k for k in kids if k["kind"] == "group" and sig(k) == common]
    same_ids = {id(k) for k in same}
    n_head = max(1, min(3, threshold))
    keep = {id(k) for k in same[:n_head] + same[-1:]}
    hidden = len(same) - len(keep)
    out: list[Any] = []
    placed = False
    for k in kids:
        if id(k) in same_ids and id(k) not in keep:
            if not placed:
                out.append({"collapsed": hidden, "signature": list(common)})
                placed = True
            continue
        out.append(k)
    return out


def render_tree(result: dict[str, Any], collapse_threshold: int = 8) -> str:
    meta = result["file"]
    nodes = result["nodes"]
    by_path, children = _index(nodes)
    lines = [
        f"{meta['tool']} | {meta['inspected_at']}",
        f"檔案：{meta['path']}（{human_bytes(meta['size_bytes'])}）",
        f"h5py {meta.get('h5py')} / HDF5 {meta.get('hdf5')}",
        "圖例：[G] 群組  [D] 資料集  [L] 外部連結；shape=() 是純量；opaque 是不透明位元組（常見於內嵌 mp4/mp3）",
        "注意：欄位意義全是推測，請看 questions.md 核對。",
        "",
    ]
    root = by_path.get("/", {"n_members": 0, "attrs": {}})
    head = f"/ [G] {root.get('n_members')} 項"
    if root.get("attrs"):
        head += f"  attrs {_short(root['attrs'], 160)}"
    lines.append(head)

    def rec(gpath: str, prefix: str) -> None:
        items = _collapse(children.get(gpath, []), children, collapse_threshold)
        for i, it in enumerate(items):
            last = i == len(items) - 1
            branch, cont = ("└── ", "    ") if last else ("├── ", "│   ")
            if isinstance(it, dict) and "collapsed" in it:
                lines.append(f"{prefix}{branch}… 另有 {it['collapsed']} 個同構子群組省略（子鍵：{', '.join(it['signature'])}）；完整清單見 nodes.json")
                continue
            lines.append(prefix + branch + _node_line(it))
            for d in _detail_lines(it):
                lines.append(f"{prefix}{cont}  ↳ {d}")
            if it["kind"] == "group":
                rec(it["path"], prefix + cont)

    rec("/", "")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- questions.md


def _dur_s(r: dict, unit: str | None) -> str:
    if r.get("duration") is None:
        return "?"
    scale = UNIT_TO_S.get(unit or "")
    return f"{r['duration'] * scale:.3f}" if scale else f"{fmt_num(r['duration'])}（原單位）"


def render_questions(result: dict[str, Any]) -> str:
    meta = result["file"]
    c = result["candidates"]
    unit = c["time_base"].get("unit_guess")
    L = [
        f"# 待確認問題：{meta['name']}",
        "",
        f"由 inspect-h5 依實際檔案自動產生（{meta['inspected_at']}）。所有欄位意義都只是推測，"
        "請逐項確認或更正；確認後再整理成 `configs/field_mapping.yaml`（第 2 包 indexer 會讀取）。",
        "",
        "## 1. 問題清單",
        "",
        "| # | 主題 | 問題 |",
        "|---|---|---|",
    ]
    for q in c["questions"]:
        L.append(f"| {q['id']} | {q['topic']} | {q['question'].replace('|', '/')} |")
    L += ["", "## 2. 人工核對 5 段 segment", "",
          "規格要求在真實樣本中人工核對至少 5 段的起訖與 success。請對照影片（第 3 包會產生預覽圖）。", "",
          "| # | segment | text | start | end | 長度 (s) | success | 核對 | 備註 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(c.get("manual_check") or [], 1):
        L.append(
            f"| {i} | `{r['path']}` | {r.get('text') or ''} | {fmt_num(r.get('start'))} | {fmt_num(r.get('end'))} | "
            f"{_dur_s(r, unit)} | {r.get('success') if r.get('success') is not None else r.get('success_raw')} | ☐ 正確 ☐ 有誤 | |"
        )
    if not c.get("manual_check"):
        L.append("| - | （沒有偵測到 segment） | | | | | | | |")
    anoms = [a for a in c["anomalies"] if a["severity"] in ("error", "warn")]
    L += ["", "## 3. 異常（error / warn）", ""]
    if anoms:
        for a in anoms[:60]:
            L.append(f"- **{a['severity']}** `{a['code']}` {a['message']}")
        if len(anoms) > 60:
            L.append(f"- …另有 {len(anoms) - 60} 項，見 candidates.json")
    else:
        L.append("- 無")
    info = [a for a in c["anomalies"] if a["severity"] == "info"]
    if info:
        L += ["", "資訊類（info）：" + "、".join(sorted({a["code"] for a in info}))]
    L += ["", "## 4. 偵測摘要", ""]
    for ct in c["segments"]["patterns"]:
        k = ct["keys"]
        L.append(
            f"- segment 容器 `{ct['pattern']}`：第 {ct['level']} 層，{ct['n_segments']} 段（{ct['n_containers']} 個容器），"
            f"start=`{k['start']}` end=`{k['end']}` success=`{k['success']}` text=`{k['text']}`"
        )
    for cm in c["streams"]["cameras"]:
        L.append(
            f"- 影片 `{cm['path']}`：{cm.get('codec')} {cm.get('width')}x{cm.get('height')}，標稱 {fmt_num(cm.get('fps_nominal'))} fps，"
            f"{cm.get('frames')} 幀；時間戳 `{cm.get('timestamps')}`（{cm.get('ts_n')} 筆，約 {fmt_num(cm.get('ts_rate_hz'))} Hz）"
        )
    for au in c["streams"]["audio"]:
        L.append(
            f"- 音訊 `{au['path']}`：{au.get('sample_rate') or '?'} Hz、{au.get('channels') or '?'} 聲道、"
            f"{fmt_num(au.get('duration_s'))} s；時間戳 `{au.get('timestamps')}`"
            + (f"（{au['storage_note']}）" if au.get("storage_note") else "")
        )
    L.append(f"- 時間陣列 {len(c['streams']['time_arrays'])} 個；未配對：{c['streams']['unpaired_time_arrays'] or '無'}")
    tb = c["time_base"]
    L.append(f"- 時間基準推測：{tb.get('relation')}（參考 `{tb.get('reference')}`，單位 {tb.get('unit_guess')}）")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- field_mapping.proposed.yaml


def _y(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def render_mapping(result: dict[str, Any]) -> str:
    meta = result["file"]
    c = result["candidates"]
    tb = c["time_base"]
    conts = c["segments"]["containers"]
    top = next((ct for ct in conts if ct["level"] == 0), None)
    L = [
        f"# 欄位映射「提案」：由 inspect-h5 依 {meta['name']} 自動產生（{meta['inspected_at']}）",
        "# 未經使用者核對前不可當正式設定使用。核對後另存為 configs/field_mapping.yaml，",
        "# 並把 status 改成 confirmed（第 2 包 indexer 只接受 confirmed）。",
        "version: 0",
        "status: proposed",
        f"source_file: {_y(meta['name'])}",
        "segments:",
    ]
    if top:
        k = top["keys"]
        L += [
            f"  container: {_y(top['path'])}   # 偵測到 {top['n_segments']} 段",
            "  id: group_name                 # 以子群組名稱當 segment_id",
            f"  start: {_y(k['start'])}",
            f"  end: {_y(k['end'])}",
            f"  success: {_y(k['success'])}   # 推測 True=成功、False=失敗（待確認）",
            f"  text: {_y(k['text'])}",
            f"  time_unit: {_y(tb.get('unit_guess'))}   # 推測，待確認",
            f"  time_base: {_y(tb.get('relation'))}   # 參考 {tb.get('reference')}",
            f"  time_reference: {_y(tb.get('reference'))}",
        ]
        subs = sorted({ct["pattern"] for ct in conts if ct["level"] >= 1})
        if subs:
            L.append("  children:")
            for s in subs:
                lvl = next(ct["level"] for ct in conts if ct["pattern"] == s)
                L.append(f"    - container_pattern: {_y(s)}   # 第 {lvl} 層")
    else:
        L.append("  container: null   # 未偵測到，需人工指定")
    acts = (c["segments"]["actions"].get("level0") or {})
    ins_objs = sorted((acts.get("insert") or {}).get("objects", {}).keys())
    L += [
        "action_parse:",
        "  source: text",
        "  regex: '^\\s*(?P<action>[A-Za-z][A-Za-z_\\-]*)\\s*(?P<part>.*?)\\s*$'   # 第一個詞當 action（待確認）",
        "  target_action: insert   # 規格：只取一種 Insert 操作",
        f"  target_part: null       # 候選：{_y(ins_objs)}",
        "cameras:",
    ]
    cams = c["streams"]["cameras"]
    if not cams:
        L.append("  {}   # 未偵測到內嵌影片")
    for cm in cams:
        L += [
            f"  {_y(cm['name'])}:",
            f"    video: {_y(cm['path'])}",
            f"    encoding: {_y(cm['encoding'])}",
            f"    timestamps: {_y(cm.get('timestamps'))}",
            f"    frame_to_time: index   # 第 i 幀 ↔ 第 i 個時間戳（影片 {cm.get('frames')} 幀 / 時間 {cm.get('ts_n')} 筆，待確認）",
            "    role: null              # external / wrist / event，待確認",
        ]
    L += [
        "missing_fields:   # 規格輸出需要、但資料中沒有：第 4 包人工標註",
        "  - failure_reason",
        "  - evidence_start_sec",
        "  - evidence_end_sec",
        "  - best_camera",
        "  - visible_evidence",
        "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------- 寫檔


def write_file_outputs(result: dict[str, Any], out_dir: str | Path, collapse_threshold: int = 8) -> dict[str, str]:
    d = Path(out_dir) / result["file"]["stem"]
    d.mkdir(parents=True, exist_ok=True)
    files = {
        "tree": d / "tree.txt",
        "nodes": d / "nodes.json",
        "candidates": d / "candidates.json",
        "questions": d / "questions.md",
        "mapping": d / "field_mapping.proposed.yaml",
    }
    files["tree"].write_text(render_tree(result, collapse_threshold), encoding="utf-8")
    files["nodes"].write_text(json.dumps({"file": result["file"], "nodes": result["nodes"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    files["candidates"].write_text(json.dumps({"file": result["file"], **result["candidates"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    files["questions"].write_text(render_questions(result), encoding="utf-8")
    files["mapping"].write_text(render_mapping(result), encoding="utf-8")
    return {k: str(v) for k, v in files.items()}


def _norm_path(p: str) -> str:
    return "/".join("*" if is_int_name(x) else x for x in p.split("/")) or "/"


def write_summary(results: list[dict[str, Any]], failures: list[dict[str, str]], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    pattern_files: dict[str, set] = defaultdict(set)
    cam_presence: dict[str, set] = defaultdict(set)
    verbs: Counter = Counter()
    insert_objs: dict[str, Counter] = defaultdict(Counter)
    codes: Counter = Counter()
    for r in results:
        name = r["file"]["name"]
        c = r["candidates"]
        for n in r["nodes"]:
            pattern_files[_norm_path(n["path"])].add(name)
        for cm in c["streams"]["cameras"]:
            cam_presence[cm["name"]].add(name)
        lvl0 = c["segments"]["actions"].get("level0") or {}
        for v, d in lvl0.items():
            verbs[v] += d["count"]
        for o, d in (lvl0.get("insert") or {}).get("objects", {}).items():
            insert_objs[o].update({"count": d["count"], "success": d["success_true"], "failure": d["success_false"]})
        for a in c["anomalies"]:
            codes[(a["severity"], a["code"])] += 1
        ins = lvl0.get("insert") or {}
        rows.append({
            "file": name,
            "size": human_bytes(r["file"]["size_bytes"]),
            "cameras": [cm["name"] for cm in c["streams"]["cameras"]],
            "n_time_arrays": len(c["streams"]["time_arrays"]),
            "n_level0": c["segments"]["n_level0"],
            "n_level1plus": c["segments"]["n_level1plus"],
            "insert": f"{ins.get('count', 0)}（{ins.get('success_true', 0)}/{ins.get('success_false', 0)}）",
            "errors": sum(1 for a in c["anomalies"] if a["severity"] == "error"),
            "warns": sum(1 for a in c["anomalies"] if a["severity"] == "warn"),
        })
    names = [r["file"]["name"] for r in results]
    not_everywhere = {p: sorted(fs) for p, fs in pattern_files.items() if len(fs) < len(names)}
    summary = {
        "generated_at": now_iso(),
        "files": rows,
        "failures": failures,
        "camera_presence": {k: sorted(v) for k, v in sorted(cam_presence.items())},
        "paths_not_in_all_files": not_everywhere,
        "level0_verbs": dict(verbs.most_common()),
        "insert_objects": {o: dict(cn) for o, cn in sorted(insert_objs.items())},
        "anomaly_counts": [{"severity": s, "code": k, "count": n} for (s, k), n in sorted(codes.items())],
    }
    jp = out / "_summary.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    L = [f"# HDF5 檢查總表（{summary['generated_at']}）", "",
         "| 檔案 | 大小 | 影片 | 時間陣列 | 高層段數 | 低層段數 | Insert（成/敗） | error | warn |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['file']} | {r['size']} | {', '.join(r['cameras']) or '-'} | {r['n_time_arrays']} | {r['n_level0']} | "
                 f"{r['n_level1plus']} | {r['insert']} | {r['errors']} | {r['warns']} |")
    for fl in failures:
        L.append(f"| {fl['file']} | 失敗 | {fl['error']} | | | | | | |")
    L += ["", "## 各檔共有 / 不一致的結構", ""]
    if len(names) <= 1:
        L.append("- 只有 1 個檔案，無法比較。")
    elif not_everywhere:
        items = sorted(not_everywhere.items())
        for p, fs in items[:40]:
            L.append(f"- `{p}` 只出現在 {len(fs)}/{len(names)} 個檔：{', '.join(fs)}")
        if len(items) > 40:
            L.append(f"- …另有 {len(items) - 40} 項，見 _summary.json")
    else:
        L.append("- 所有檔案結構一致（數字編號以 * 表示）。")
    L += ["", "## 相機出現次數", ""]
    for cam, fs in summary["camera_presence"].items():
        L.append(f"- {cam}：{len(fs)}/{len(names)}")
    L += ["", "## 高層動作（text 第一個詞）", "", f"- {summary['level0_verbs'] or '無'}"]
    L += ["", "## Insert 零件（樣本內）", ""]
    for o, cn in summary["insert_objects"].items():
        L.append(f"- {o}：{cn.get('count', 0)} 段（成功 {cn.get('success', 0)} / 失敗 {cn.get('failure', 0)}）")
    if not summary["insert_objects"]:
        L.append("- 無")
    L += ["", "## 異常統計", ""]
    for a in summary["anomaly_counts"]:
        L.append(f"- {a['severity']} `{a['code']}`：{a['count']}")
    L.append("")
    mp = out / "_summary.md"
    mp.write_text("\n".join(L), encoding="utf-8")
    return {"summary_md": str(mp), "summary_json": str(jp)}
