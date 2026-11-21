"""第 20 包：補強步驟分類器——3×3 區塊平均、最後一步旗標、四種改法評分、選擇規則、凍結檔、多相機讀檔、開發流程（合成資料，不需要 GPU、h5）。"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from assembly_monitor import perpart as pp
from assembly_monitor import stepcls as sc
from assembly_monitor import stepcls2 as s2

SEG_COLS = ["recording_id", "segment_id", "action", "part", "success", "validation_status", "start_unix", "end_unix"]
SK_COLS = ["recording_id", "segment_id", "skill_id", "text", "start_unix", "end_unix", "success", "validation_status"]


# ---------------------------------------------------------------- 3×3 區塊平均


def test_bins_match_adaptive_pool():
    assert s2._bins(16, 3) == [(0, 6), (5, 11), (10, 16)]
    assert s2._bins(21, 3) == [(0, 7), (7, 14), (14, 21)]
    assert s2._bins(4, 2) == [(0, 2), (2, 4)]


def test_grid_pool_values_and_shape():
    b, gh, gw, d = 2, 4, 6, 3
    rng = np.random.default_rng(0)
    h = rng.normal(size=(b, 1 + gh * gw, d)).astype(np.float32)
    out = s2.grid_pool(h, gh, gw, 2)
    assert out.shape == (b, d * 5)
    assert np.allclose(out[:, :d], h[:, 0])                                     # 先放 CLS
    p = h[:, 1:].reshape(b, gh, gw, d)
    assert np.allclose(out[:, d:2 * d], p[:, :2, :3].mean(axis=(1, 2)), atol=1e-6)   # 左上
    assert np.allclose(out[:, 4 * d:], p[:, 2:, 3:].mean(axis=(1, 2)), atol=1e-6)    # 右下
    full = s2.grid_pool(h, gh, gw, 1)
    assert np.allclose(full[:, d:], h[:, 1:].mean(axis=1), atol=1e-6)            # 1×1 ＝ 原本的小區塊平均
    with pytest.raises(ValueError):
        s2.grid_pool(h, 5, 5, 3)
    assert s2.ENCODER_GRID["cache_tag"] != pp.ENCODER_CROP["cache_tag"]           # 不和原本的快取混在一起


def test_check_grid_cache(tmp_path):
    s2.check_grid_cache(tmp_path, "k")
    s2.check_grid_cache(tmp_path, "k")
    p = tmp_path / "k" / "_encoder.json"
    p.write_text(p.read_text(encoding="utf-8").replace('"grid": 3', '"grid": 4'), encoding="utf-8")
    with pytest.raises(ValueError):
        s2.check_grid_cache(tmp_path, "k")


# ---------------------------------------------------------------- 資料


def _index(tmp_path, part="BNC", n_ok=14, n_twist_bad=8, n_align_last=5, n_align_mid=3):
    """成功段：Approach > Align > Twist（都成功）；
    旋轉失敗段：Approach > Align > Twist✗；對準失敗（最後一步）：Approach > Align✗；
    對準失敗（後面還有）：Approach > Align✗ > Twist✗。每支錄影 2 段。"""
    d = tmp_path / "index"
    d.mkdir()
    plans = ([[("Approach", 1), ("Align", 1), ("Twist", 1)]] * n_ok + [[("Approach", 1), ("Align", 1), ("Twist", 0)]] * n_twist_bad
             + [[("Approach", 1), ("Align", 0)]] * n_align_last + [[("Approach", 1), ("Align", 0), ("Twist", 0)]] * n_align_mid)
    segs, skills = [], []
    for i, steps in enumerate(plans):
        rec, sid, t0 = f"2025-01-{9 + i // 10:02d}-10-{(i // 2) % 60:02d}-00", 3 + i % 2, 1000.0 + 100 * i
        ok = all(g for _, g in steps)
        segs.append({"recording_id": rec, "segment_id": sid, "action": "insert", "part": part, "success": str(ok),
                     "validation_status": "ok", "start_unix": t0, "end_unix": t0 + 5 * len(steps)})
        for k, (name, good) in enumerate(steps):
            skills.append({"recording_id": rec, "segment_id": sid, "skill_id": k, "text": name, "start_unix": t0 + 5 * k,
                           "end_unix": t0 + 5 * k + 5, "success": str(bool(good)), "validation_status": "ok"})
    for name, rows, cols in (("segments.csv", segs, SEG_COLS), ("skills.csv", skills, SK_COLS)):
        with (d / name).open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)
    return d


def test_last_flags(tmp_path):
    inst = sc.step_instances(_index(tmp_path), "BNC")
    last = s2.last_flags(inst)
    al = [s for s in inst if s["step"] == "Align" and s["label"] == 1]
    assert sum(last[s["segment_id"]] for s in al) == 5 and sum(not last[s["segment_id"]] for s in al) == 3
    assert all(last[s["segment_id"]] for s in inst if s["step"] == "Twist")
    assert not any(last[s["segment_id"]] for s in inst if s["step"] == "Approach")


def _cand_feats(task, seps, seed=0, d=24, drop=None):
    """seps：{改法: (一般失敗的分開程度, 難題的分開程度)}；難題＝後面還有步驟的失敗。"""
    rng = np.random.default_rng(seed)
    out = {}
    for c, (sep, sep_hard) in seps.items():
        f = {}
        for s in task["instances"]:
            if drop and s["segment_id"] in drop.get(c, ()):
                continue
            v = rng.normal(size=d)
            if s["label"]:
                v[0] += sep_hard if s.get("_hard") else sep
            f[s["segment_id"]] = v
        out[c] = f
    return out


def _mark_hard(task, last):
    for s in task["instances"]:
        s["_hard"] = bool(s["label"]) and not last[s["segment_id"]]


def test_evaluate_task_counts_hard(tmp_path):
    idx = _index(tmp_path, n_align_last=5, n_align_mid=4)
    task = next(t for t in sc.build_tasks(idx, ("BNC",)) if t["step"] == "Align")
    last = s2.last_flags(sc.step_instances(idx, "BNC"))
    _mark_hard(task, last)
    gone = task["instances"][0]["segment_id"]
    feats = _cand_feats(task, {"C0": (6, 0), "C1": (6, 6), "C2": (6, 0), "C3": (0, 0)}, drop={"C2": (gone,)})
    r = s2.evaluate_task(task, feats, last)
    assert r["n_dropped"] == 1 and gone not in r["ids"] and r["n_hard"] == 4 and r["n_last_fail"] == 5
    assert r["cands"]["C1"]["hard_hit"] >= 3 and r["cands"]["C0"]["hard_hit"] <= 1
    assert r["cands"]["C0"]["last_hit"] >= 4 and r["cands"]["C3"]["bal_mean"] < r["cands"]["C1"]["bal_mean"]
    assert len(r["cands"]["C0"]["judged"]) == r["n"]


def _tr(name, step, bals, hard, n_hard=4):
    return {"name": name, "part": name.split()[0], "step": step, "n": 30, "n_failure": 10, "n_success": 20, "n_hard": n_hard,
            "n_last_fail": 10 - n_hard, "n_dropped": 0, "ids": [], "labels": [],
            "cands": {c: {"bal_mean": b, "bal_min": b, "bal_max": b, "dim": 1536, "k_mean": 3.0, "fail_rate": [], "judged": [],
                          "hard_hit": h, "last_hit": 5, "ok_hit": 18} for c, b, h in zip(s2.CANDS, bals, hard)}}


def test_summarize_rule():
    # C2 最高、C1 差 1 點 → 選比較簡單的 C1
    s = s2.summarize([_tr("BNC Align", "Align", (0.70, 0.75, 0.76, 0.74), (1, 2, 2, 1))])
    assert s["eligible"] == ["C1", "C2", "C3"] and s["pick"] == "C1" and s["improved"]
    # C2 高很多 → 選 C2
    assert s2.summarize([_tr("BNC Align", "Align", (0.70, 0.73, 0.80, 0.74), (1, 1, 1, 1))])["pick"] == "C2"
    # 平均高但難題變少 → 不能選
    s = s2.summarize([_tr("BNC Align", "Align", (0.70, 0.80, 0.71, 0.71), (2, 1, 2, 2))])
    assert "C1" not in s["eligible"] and s["pick"] == "C0" and not s["improved"]
    # 只高 1 點 → 不夠
    assert s2.summarize([_tr("BNC Align", "Align", (0.70, 0.71, 0.71, 0.715), (1, 1, 1, 1))])["pick"] == "C0"
    with pytest.raises(s2.Stop):
        s2.summarize([dict(_tr("X Y", "Y", (0.5,) * 4, (0,) * 4), skipped="太少")])


def test_write_selected(tmp_path):
    s = s2.summarize([_tr("BNC Align", "Align", (0.70, 0.73, 0.80, 0.74), (1, 1, 1, 1))])
    p = s2.write_selected(s, tmp_path / "sel.yaml")
    body = json.loads("\n".join(x for x in p.read_text(encoding="utf-8").splitlines() if not x.startswith("#")))
    assert body["candidate"] == "C2" and body["grid"] == 3 and body["eb"] == 1.0
    before = p.read_text(encoding="utf-8")
    s2.write_selected(dict(s, pick="C1"), p)
    assert p.read_text(encoding="utf-8") == before
    keep = s2.summarize([_tr("BNC Align", "Align", (0.70, 0.70, 0.70, 0.70), (1, 1, 1, 1))])
    assert s2.write_selected(keep, tmp_path / "none.yaml") is None and not (tmp_path / "none.yaml").exists()


class FakeEnc:
    def __init__(self, key, dim=3):
        self.key, self.dim, self.calls = key, dim, 0

    def encode(self, paths):
        self.calls += 1
        return np.array([[float(Path(p).stat().st_size)] + [float(i) for i in range(self.dim - 1)] for p in paths], np.float32)


def _frames(root, task_name, eb, sid, cams):
    d = Path(root) / f"insert_{sc._slug(task_name)}" / sc.sampling_key(eb) / "segments" / sid
    d.mkdir(parents=True)
    names = []
    for k, c in enumerate(cams):
        n = f"{c}__00__t+005.000s__f{100 + k:06d}.jpg"
        (d / n).write_bytes(b"x" * (10 + k))
        names.append(n)
    (d / "meta.json").write_text(json.dumps({"start_unix": 0.0, "end_unix": 5.0, "duration_s": 5.0, "window_unix": [-0.5, 5 + eb],
                                             "cameras": [{"camera": c, "frame_paths": [n], "source_times_sec": [5 + eb]}
                                                         for c, n in zip(cams, names)]}), encoding="utf-8")


def test_step_vectors_two_cameras(tmp_path):
    pytest.importorskip("assembly_monitor.probe")
    task = sc.build_tasks(_index(tmp_path), ("BNC",))[0]
    root = tmp_path / "ext"
    for s in task["instances"][:-1]:
        _frames(root, task["name"], 1.0, s["segment_id"], s2.EXT_CAMS)
    _frames(root, task["name"], 1.0, task["instances"][-1]["segment_id"], ("hama1",))      # 少一台 → 不納入
    enc = FakeEnc("full_fake", dim=4)
    v = s2.step_vectors(task, enc, 1.0, s2.EXT_CAMS, str(root), tmp_path / "cache")
    assert len(v) == len(task["instances"]) - 1
    x = v[task["instances"][0]["segment_id"]]
    assert x.shape == (8,) and x[0] == 10 and x[4] == 11                                      # hama1 在前、hama2 在後


def test_run_dev_end_to_end(tmp_path, monkeypatch):
    idx = _index(tmp_path, n_ok=16, n_twist_bad=9, n_align_last=6, n_align_mid=4)
    last = s2.last_flags(sc.step_instances(idx, "BNC"))

    def fake_features(task, encs, cache_root, frames_root, ext_root):
        _mark_hard(task, last)
        return _cand_feats(task, {"C0": (6, 0), "C1": (6, 0), "C2": (6, 6), "C3": (6, 0)}, seed=len(task["name"]))

    out = tmp_path / "out"
    res = s2.run_dev(index_dir=str(idx), out=out, parts=("BNC",), cache_root=tmp_path / "c", step_index=tmp_path / "steps",
                     selected_path=tmp_path / "sel.yaml", encs={"crop": None, "full": None, "grid": None},
                     do_extract=False, features=fake_features)
    s = res["summary"]
    assert [t["name"] for t in res["tasks"]] == ["BNC Align", "BNC Twist"] and s["n_hard"] == 4
    assert s["hard_hit"]["C2"] > s["hard_hit"]["C0"]
    ws = res["which_step"]["BNC"]
    assert set(ws) == set(s2.CANDS) and ws["C0"]["n_failed"] == 19
    assert ws["C2"]["n_right"] >= ws["C0"]["n_right"]
    rep = (out / "report.md").read_text(encoding="utf-8")
    assert "BNC 對準" in rep and "難題" in rep and "誤判" not in rep and ("保留 C0" in rep or "已凍結" in rep)
    json.loads((out / "result.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        s2.run_dev(index_dir=str(idx), parts=("bolt 4",), do_extract=False, features=fake_features)
    with pytest.raises(ValueError):
        s2.run_dev(index_dir=str(idx), parts=(pp.RESERVED[0],), do_extract=False, features=fake_features)
