"""第 13 包第四步：REBOOT 16 mm 圓柱（開發集）——把切好的段做成和 REASSEMBLE 相同格式的影格與切分，
用第 12 包凍結的方法原封不動地跑，再計分。

使用者 2026-11-06 選的：
  - 成功段的結尾兩種都跑：A＝放開前（最後一格＝夾爪開始放開的那一格），B＝放開後（最後一格＝放開後 1 秒）。
    失敗段兩種共用（最後一格＝官方標註的救援開始）。
  - 多抓低角度前視相機 cam_low：相機對應 hama1＝俯視 cam_high、hama2＝低角度前視 cam_low、hand＝動作那隻手的手腕。

取樣和凍結方法相同（uniform_n4_sb0.5_eb0.5_jpg：段前後各加 0.5 秒，均勻取 4 格，含首尾）。
每段長 3 秒；段的結尾放在「最後一格往前 0.5 秒」，所以取樣窗＝[最後一格−4 秒, 最後一格]。
retention_failure（夾爪不放開；圓柱可能已插好）照跑，但主要成績排除、另外列。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from . import reboot_data as rd
from .reboot_meta import hf_lister, local_name
from .split import SPLIT_COLS, file_sha256, lock_info
from .util import now_iso

log = logging.getLogger(__name__)

SEG_DIR = "experiments/p13_cyl_segments"
FRAMES_ROOT = "data/derived/frames/reboot_cyl"
SPLIT_DIR = "data/splits/reboot_cyl_dev"
OUT = "experiments/p13_reboot_cyl"
SAMPLING_KEY = "uniform_n4_sb0.5_eb0.5_jpg"
N_FRAMES, SB, EB = 4, 0.5, 0.5
SEG_S = 3.0
AFTER_RELEASE_S = 1.0
EXCLUDE_MAIN = ("retention_failure",)
SEG_NO = {"A": 1, "B": 2, "F": 0}
VARIANT_ZH = {"A": "放開前", "B": "放開後"}
CAM_LOW_MAX_GB = 1.5
# 方法：和第 12 包測試集相同（凍結）。run 目錄＝OUT/<實驗>/<模型>/f4
RUNS = {"gpt": ("gpt", "gpt61_sol"), "v3": ("v3", "qwen3vl_8b_zoomfix"), "prob": ("v3", "qwen3vl_8b_rscore_t0999")}
METHODS = [("majority", "一律答成功（基準）"), ("gpt", "只用 GPT-6.1 Sol ★"), ("prob", "只用 Qwen 機率門檻 τ 0.999089"),
           ("prob_gpt", "兩階段：Qwen 機率門檻 → GPT"), ("v3_gpt", "兩階段：Qwen 第三版 → GPT"), ("v3", "Qwen＋Skill 第三版")]


def cam_source(cam: str, arm: str) -> str:
    return {"hama1": "cam_high", "hama2": "cam_low", "hand": f"cam_{arm}_wrist"}[cam]


# ---------------------------------------------------------------- 段


def model_segments(step3: dict[str, Any], releases: dict[str, Any], fps: float) -> list[dict[str, Any]]:
    """第三步的段 → 要給模型的段（A、B 兩種成功段＋失敗段）。"""
    out = []
    eb = int(round(EB * fps))

    def seg(kind: str, ep: int, variant: str, label: str, last: int, **extra: Any) -> dict[str, Any]:
        end_s = (last - eb) / fps
        rid = f"reboot-cyl-{kind}{ep:02d}"
        return {"segment_id": f"{rid}__seg{SEG_NO[variant]}", "recording_id": rid, "variant": variant, "label": label,
                "episode": ep, "last_frame": int(last), "start_s": round(end_s - SEG_S, 6), "end_s": round(end_s, 6),
                **extra}

    for s in step3["segments"]:
        if s["label"] == "success":
            r = releases[str(s["episode"])]
            out.append(seg("exp", s["episode"], "A", "success", r["onset"], source="expert", arm=s["arm"], main=True))
            out.append(seg("exp", s["episode"], "B", "success", r["cross"] + int(round(AFTER_RELEASE_S * fps)),
                           source="expert", arm=s["arm"], main=True))
        else:
            out.append(seg("rec", s["episode"], "F", "failure", s["end_frame"], source="recovery", arm=s["arm"],
                           mode=s.get("mode"), phase=s.get("phase"), main=s.get("mode") not in EXCLUDE_MAIN))
    return sorted(out, key=lambda x: x["segment_id"])


def frame_plan(seg: dict[str, Any], length: int, fps: float) -> list[dict[str, Any]]:
    """和 extract.target_times 相同：窗＝[start−SB, end+EB]，均勻 N 點（含首尾），取最近的影格。"""
    w0, w1 = seg["start_s"] - SB, seg["end_s"] + EB
    out = []
    for k in range(N_FRAMES):
        t = w0 + (w1 - w0) * k / (N_FRAMES - 1)
        f = min(max(int(round(t * fps)), 0), length - 1)
        out.append({"k": k, "target_s": t, "frame": f, "source_s": f / fps})
    return out


def write_split(segs: list[dict[str, Any]], split_dir: str | Path) -> dict[str, Any]:
    """REASSEMBLE 格式的切分（全部是 dev）。內容不同才重寫。"""
    d = Path(split_dir)
    rows = [{"segment_id": s["segment_id"], "recording_id": s["recording_id"], "segment_index": SEG_NO[s["variant"]],
             "group": s["recording_id"], "split": "dev", "dataset_result": s["label"], "duration_s": SEG_S,
             "date": "reboot", "start_unix": s["start_s"], "end_unix": s["end_s"]} for s in segs]
    info = lock_info(rows)
    man_p = d / "manifest.json"
    if man_p.exists():
        old = json.loads(man_p.read_text(encoding="utf-8"))
        if old.get("lock", {}).get("segments_sha256") == info["segments_sha256"]:
            return old
    d.mkdir(parents=True, exist_ok=True)
    csv_p = d / "split.csv"
    with csv_p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=SPLIT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SPLIT_COLS})
    n_f = sum(1 for r in rows if r["dataset_result"] == "failure")
    man = {"name": d.name, "created_at": now_iso(), "tool": "assembly_monitor.reboot_run",
           "spec": {"name": d.name, "source": "REBOOT 16 mm cylinder (dev)", "unit": "segment",
                    "window": {"start_buffer": SB, "end_buffer": EB}, "segment_s": SEG_S},
           "counts": {"dev": {"n_groups": len({r["group"] for r in rows}), "n_segments": len(rows), "n_failure": n_f,
                              "n_success": len(rows) - n_f}},
           "lock": info, "split_csv_sha256": file_sha256(csv_p), "locked": True, "leakage_status": "not_checked",
           "note": "REBOOT 圓柱開發集：全部是 dev。成功段有兩種結尾（seg1＝放開前、seg2＝放開後），失敗段 seg0。"}
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def _cache_key(seg: dict[str, Any], plan: list[dict[str, Any]]) -> str:
    s = json.dumps({"seg": {k: seg[k] for k in ("segment_id", "start_s", "end_s", "arm", "episode")},
                    "frames": [p["frame"] for p in plan], "cams": ["hama1", "hama2", "hand"]}, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def extract(segs: list[dict[str, Any]], dest_root: str | Path, frames_root: str | Path = FRAMES_ROOT,
            fps: float = 30.0) -> dict[str, Any]:
    """依 frame_plan 從 mp4 取格，寫成和第 3 包相同的資料夾與 meta.json（vlm.load_segment_frames 讀得到）。"""
    import cv2  # noqa: F401

    dest_root = Path(dest_root)
    eps = {kind: {e["episode_index"]: e for e in rd.read_episodes(dest_root / local_name(repo), cams=rd.CAMS + ("cam_low",))}
           for kind, repo in rd.CYL.items()}
    readers = {kind: rd.FrameReader(dest_root / local_name(repo)) for kind, repo in rd.CYL.items()}
    st = {"done": 0, "skipped": 0, "missing": 0}
    try:
        for s in segs:
            kind = "expert" if s["source"] == "expert" else "recovery"
            e = eps[kind][s["episode"]]
            plan = frame_plan(s, e["length"], fps)
            d = Path(frames_root) / SAMPLING_KEY / "segments" / s["segment_id"]
            key = _cache_key(s, plan)
            meta_p = d / "meta.json"
            if meta_p.exists():
                try:
                    if json.loads(meta_p.read_text(encoding="utf-8")).get("cache_key") == key:
                        st["skipped"] += 1
                        continue
                except (OSError, ValueError):
                    pass
            d.mkdir(parents=True, exist_ok=True)
            cams = []
            for cam in ("hama1", "hama2", "hand"):
                src = cam_source(cam, s["arm"])
                v = e["videos"].get(src)
                paths, times = [], []
                for p in plan:
                    img = None
                    if v and (dest_root / local_name(rd.CYL[kind]) / v["path"]).exists():
                        img, _ft = readers[kind].read_at(v["path"], v["from"] + p["frame"] / fps, fps)
                    if img is None:
                        st["missing"] += 1
                        paths.append(None)
                        times.append(None)
                        continue
                    rel = p["source_s"] - s["start_s"]
                    name = f"{cam}__{p['k']:02d}__t{rel:+08.3f}s__f{p['frame']:06d}.jpg"
                    rd._write_jpg(d / name, img, 95)
                    paths.append(name)
                    times.append(round(rel, 4))
                cams.append({"segment_id": s["segment_id"], "camera": cam, "source_camera": src,
                             "source_times_sec": times, "frame_paths": paths, "sampling": "uniform",
                             "n_frames": N_FRAMES})
            meta = {"recording_id": s["recording_id"], "segment_id": SEG_NO[s["variant"]], "text": "Insert round peg",
                    "label_success": s["label"] == "success", "start_unix": s["start_s"], "end_unix": s["end_s"],
                    "duration_s": SEG_S, "window_unix": [s["start_s"] - SB, s["end_s"] + EB],
                    "sampling": {"n_frames": N_FRAMES, "start_buffer": SB, "end_buffer": EB, "fmt": "jpg",
                                 "jpeg_quality": 95, "strategy": "uniform"},
                    "cameras": cams, "source": {"dataset": rd.CYL[kind], "episode": s["episode"], "variant": s["variant"]},
                    "cache_key": key, "created_at": now_iso(), "tool": "assembly_monitor.reboot_run"}
            tmp = d / "meta.json.part"
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, meta_p)
            st["done"] += 1
    finally:
        for r in readers.values():
            r.close()
    return st


def input_sheets(segs: list[dict[str, Any]], frames_root: str | Path, out: Path) -> int:
    """每段一張：模型實際看到的 3 台相機 × 4 格。"""
    import cv2

    n = 0
    for s in segs:
        d = Path(frames_root) / SAMPLING_KEY / "segments" / s["segment_id"]
        meta_p = d / "meta.json"
        if not meta_p.exists():
            continue
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        grid, pts = {}, []
        for c in m["cameras"]:
            for k, (p, t) in enumerate(zip(c["frame_paths"], c["source_times_sec"])):
                if p:
                    img = cv2.imread(str(d / p))
                    if img is not None:
                        grid[(c["camera"], k)] = img
                if c["camera"] == "hama1":
                    pts.append((k, f"#{k} t={t:+.2f}s" if t is not None else f"#{k}"))
        head = f"{s['segment_id']}  label={s['label']}  variant={s['variant']}" + (f"  mode={s.get('mode')}" if s.get("mode") else "")
        sheet = rd.make_sheet([head, "model input: hama1=cam_high, hama2=cam_low, hand=wrist; last column = last frame"],
                              pts, 1.0, grid, ("hama1", "hama2", "hand"))
        rd._write_jpg(out / "inputs" / f"{s['segment_id']}.jpg", sheet, 85)
        n += 1
    return n


def download_cam_low(dest_root: str | Path, lister=hf_lister, downloader=None, max_gb: float = CAM_LOW_MAX_GB) -> dict[str, Any]:
    plan = {kind: rd.download_plan(lister(repo), cams=("cam_low",)) for kind, repo in rd.CYL.items()}
    vids = sum(sz for files in plan.values() for p, sz in files if p.startswith("videos/"))
    if vids > max_gb * 1e9:
        raise RuntimeError(f"cam_low 影片 {vids / 1e9:.2f} GB，超過上限 {max_gb} GB")
    log.info("cam_low：%.2f GB（README、meta 已有就略過）", vids / 1e9)
    st = rd.download(plan, dest_root, downloader) if downloader else rd.download(plan, dest_root)
    st["video_bytes"] = vids
    return st


def prepare(dest_root: str | Path = "data/raw/reboot", seg_dir: str | Path = SEG_DIR, out: str | Path = OUT) -> dict[str, Any]:
    seg_dir, out = Path(seg_dir), Path(out)
    step3 = json.loads((seg_dir / "segments.json").read_text(encoding="utf-8"))
    releases = json.loads((seg_dir / "releases.json").read_text(encoding="utf-8"))
    fps = float(step3["fps"])
    segs = model_segments(step3, releases, fps)
    man = write_split(segs, SPLIT_DIR)
    st = extract(segs, dest_root, FRAMES_ROOT, fps)
    n_sheets = input_sheets(segs, FRAMES_ROOT, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "segments.json").write_text(json.dumps({"generated": now_iso(), "fps": fps, "segments": segs},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return {"segments": segs, "split": man, "extract": st, "sheets": n_sheets}


# ---------------------------------------------------------------- 計分


def _latest(p: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("segment_id"):
                out[r["segment_id"]] = r
    return out


def load_runs(out: str | Path = OUT) -> dict[str, dict[str, Any]]:
    res = {}
    for key, (exp, model) in RUNS.items():
        d = Path(out) / exp / model / "f4"
        res[key] = {"pred": _latest(d / "predictions.jsonl"), "raw": _latest(d / "raw.jsonl")}
    return res


def answer(runs: dict[str, Any], method: str, sid: str) -> tuple[str | None, bool]:
    """(答案, 有沒有問 GPT)。無效、錯誤、缺都算 None（計分時算答錯）。"""
    def ok(key: str) -> str | None:
        p = runs[key]["pred"].get(sid)
        return p.get("result") if p and p.get("status") == "ok" and p.get("result") in ("success", "failure") else None

    if method == "majority":
        return "success", False
    if method in ("gpt", "v3", "prob"):
        return ok(method), method == "gpt"
    first = "prob" if method == "prob_gpt" else "v3"
    a = ok(first)
    if a == "success":
        return a, False
    return ok("gpt"), True


def score(segs: list[dict[str, Any]], runs: dict[str, Any], method: str) -> dict[str, Any]:
    n = {"success": 0, "failure": 0}
    c = {"success": 0, "failure": 0}
    none = asked = 0
    cost = 0.0
    for s in segs:
        a, used_gpt = answer(runs, method, s["segment_id"])
        n[s["label"]] += 1
        c[s["label"]] += int(a == s["label"])
        none += int(a is None)
        if used_gpt:
            asked += 1
            cost += float((runs["gpt"]["raw"].get(s["segment_id"]) or {}).get("cost_usd") or 0.0)
    rs = c["success"] / n["success"] if n["success"] else None
    rf = c["failure"] / n["failure"] if n["failure"] else None
    bal = (rs + rf) / 2 if rs is not None and rf is not None else None
    return {"n_success": n["success"], "n_failure": n["failure"], "ok_success": c["success"], "ok_failure": c["failure"],
            "bal": bal, "no_answer": none, "gpt_calls": asked, "gpt_cost": round(cost, 4)}


def evaluate(out: str | Path = OUT) -> dict[str, Any]:
    out = Path(out)
    segs = json.loads((out / "segments.json").read_text(encoding="utf-8"))["segments"]
    runs = load_runs(out)
    fails = [s for s in segs if s["label"] == "failure"]
    main_f = [s for s in fails if s["main"]]
    res: dict[str, Any] = {"generated": now_iso(), "variants": {}, "by_mode": {}, "excluded": [],
                           "coverage": {k: len(v["pred"]) for k, v in runs.items()}}
    for v in ("A", "B"):
        sv = [s for s in segs if s["variant"] == v] + main_f
        res["variants"][v] = {m: score(sv, runs, m) for m, _ in METHODS}
    modes = sorted({s.get("mode") for s in fails})
    for mode in modes:
        ss = [s for s in fails if s.get("mode") == mode]
        res["by_mode"][mode] = {m: score(ss, runs, m)["ok_failure"] for m, _ in METHODS} | {"n": len(ss)}
    for s in fails:
        if not s["main"]:
            res["excluded"].append({"segment_id": s["segment_id"], "mode": s["mode"],
                                    **{m: answer(runs, m, s["segment_id"])[0] for m, _ in METHODS}})
    # 成功段：A 和 B 的答案是否不同（看「放開」這件事影響多大）
    flips = {}
    for m, _ in METHODS:
        a = {s["episode"]: answer(runs, m, s["segment_id"])[0] for s in segs if s["variant"] == "A"}
        b = {s["episode"]: answer(runs, m, s["segment_id"])[0] for s in segs if s["variant"] == "B"}
        flips[m] = {"A_fail_B_ok": sum(1 for e in a if a[e] != "success" and b.get(e) == "success"),
                    "A_ok_B_fail": sum(1 for e in a if a[e] == "success" and b.get(e) != "success")}
    res["flips"] = flips
    write_report(res, segs, runs, out)
    return res


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def write_report(res: dict[str, Any], segs: list[dict[str, Any]], runs: dict[str, Any], out: Path) -> str:
    names = dict(METHODS)
    cov = res["coverage"]
    n_seg = len(segs)
    L = ["# 第 13 包第四步：REBOOT 16 mm 圓柱（開發集）× 第 12 包凍結的方法", "", f"產生時間：{res['generated']}", "",
         "方法、提示詞、取樣都和第 12 包測試集相同，沒有為 REBOOT 調整（零件名稱也是 round peg）。"
         "相機對應：hama1＝俯視、hama2＝低角度前視、hand＝右手腕。每段 3 秒、4 格。", "",
         f"跑完的段數（共 {n_seg} 段）：GPT {cov['gpt']}、Qwen 第三版 {cov['v3']}、Qwen 機率門檻 {cov['prob']}", ""]
    for v in ("A", "B"):
        sc = res["variants"][v]
        any_ = next(iter(sc.values()))
        L += [f"## 成功段結尾＝{VARIANT_ZH[v]}（成功 {any_['n_success']} 段、失敗 {any_['n_failure']} 段）", "",
              "| 方法 | 失敗段答對 | 成功段答對 | 兩類平均 | 沒答案 | 問 GPT | GPT 費用 |", "|---|---|---|---|---|---|---|"]
        for m, _ in METHODS:
            r = sc[m]
            L.append(f"| {names[m]} | {r['ok_failure']}/{r['n_failure']} | {r['ok_success']}/{r['n_success']} | "
                     f"{_pct(r['bal'])} | {r['no_answer']} | {r['gpt_calls']} | US${r['gpt_cost']:.2f} |")
        L.append("")
    L += ["## 成功段：放開前 vs 放開後，答案改變的段數", "", "| 方法 | 放開前答錯→放開後答對 | 放開前答對→放開後答錯 |", "|---|---|---|"]
    for m, _ in METHODS:
        f = res["flips"][m]
        L.append(f"| {names[m]} | {f['A_fail_B_ok']} | {f['A_ok_B_fail']} |")
    L += ["", "## 失敗段：依失敗模式（答對段數）", "",
          "| 模式 | 段數 | " + " | ".join(names[m] for m, _ in METHODS if m != "majority") + " |",
          "|---|---|" + "---|" * (len(METHODS) - 1)]
    for mode, r in res["by_mode"].items():
        L.append(f"| {mode} | {r['n']} | " + " | ".join(str(r[m]) for m, _ in METHODS if m != "majority") + " |")
    L += ["", "## 不算在主要成績的失敗段（retention_failure：夾爪不放開，圓柱可能已插好）", "",
          "| 段 | " + " | ".join(names[m] for m, _ in METHODS if m != "majority") + " |",
          "|---|" + "---|" * (len(METHODS) - 1)]
    for x in res["excluded"]:
        L.append(f"| {x['segment_id']} | " + " | ".join(str(x[m] or "沒答案") for m, _ in METHODS if m != "majority") + " |")
    L += ["", "## 注意", "",
          "- 這是開發集：用來決定 REBOOT 流程（成功段結尾、相機），插頭零件的影格都沒看過。",
          "- 失敗是操作員故意做出來的；成功段全部來自專家示範、失敗段全部來自救援段（不同錄製時段，背景擺設略有不同）。",
          "- 段長固定 3 秒，比 REASSEMBLE 的插入段（約 15 秒）短很多；4 格只涵蓋最後 4 秒。",
          "- 每段答案在 `per_segment.csv`；模型輸入的拼圖在 `inputs/`。", ""]
    md = out / "report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    with (out / "per_segment.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "variant", "label", "mode", "main"] + [m for m, _ in METHODS])
        for s in segs:
            w.writerow([s["segment_id"], s["variant"], s["label"], s.get("mode") or "", s["main"]]
                       + [answer(runs, m, s["segment_id"])[0] or "" for m, _ in METHODS])
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(md)
