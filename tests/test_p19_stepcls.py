"""第 19 包：步驟分類器——步驟清單、題目、步驟索引、讀影格算特徵、選時間點、門檻、開發流程、測試流程、失敗在哪一步（合成資料，不需要 GPU、h5）。"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from assembly_monitor import perpart as pp
from assembly_monitor import stepcls as sc

SEG_COLS = ["recording_id", "segment_id", "action", "part", "success", "validation_status", "start_unix", "end_unix"]
SK_COLS = ["recording_id", "segment_id", "skill_id", "text", "start_unix", "end_unix", "success", "validation_status"]


def _index(tmp_path, n_ok=10, n_bad=8, part="BNC"):
    """每段：Approach（都成功）→ Align（都成功）→ Twist（前 n_ok 段成功、後 n_bad 段失敗）；每支錄影 2 段。"""
    d = tmp_path / "index"
    d.mkdir()
    segs, skills = [], []
    for i in range(n_ok + n_bad):
        rec, sid, t0 = f"2025-01-{9 + i // 2:02d}-10-00-00", 3 + i % 2, 1000.0 + 100 * i
        ok = i < n_ok
        segs.append({"recording_id": rec, "segment_id": sid, "action": "insert", "part": part, "success": str(ok),
                     "validation_status": "ok", "start_unix": t0, "end_unix": t0 + 15})
        for k, (name, a, b, good) in enumerate([("Approach", 0, 7, True), ("Align", 7, 10, True), ("Twist", 10, 15, ok)]):
            skills.append({"recording_id": rec, "segment_id": sid, "skill_id": k, "text": name, "start_unix": t0 + a,
                           "end_unix": t0 + b, "success": str(good), "validation_status": "ok"})
    segs.append({"recording_id": "zz", "segment_id": 1, "action": "pick", "part": part, "success": "True",
                 "validation_status": "ok", "start_unix": 1, "end_unix": 2})          # 不是插入段 → 不看
    skills.append({"recording_id": "zz", "segment_id": 1, "skill_id": 0, "text": "Twist", "start_unix": 1, "end_unix": 2,
                   "success": "False", "validation_status": "ok"})
    for name, rows, cols in (("segments.csv", segs, SEG_COLS), ("skills.csv", skills, SK_COLS)):
        with (d / name).open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)
    return d


def test_step_instances_and_tasks(tmp_path):
    idx = _index(tmp_path)
    inst = sc.step_instances(idx, "BNC")
    assert len(inst) == 18 * 3 and {s["step"] for s in inst} == {"Approach", "Align", "Twist"}
    tw = [s for s in inst if s["step"] == "Twist"]
    assert sum(s["label"] for s in tw) == 8 and all(s["num"] % 100 == 2 for s in tw)
    s0 = tw[0]
    assert s0["segment_id"] == f"{s0['recording_id']}__seg{s0['num']}" and s0["num"] in (302, 402)
    assert s0["end_unix"] - s0["start_unix"] == pytest.approx(5.0) and s0["parent"].endswith(("__seg3", "__seg4"))
    tasks = sc.build_tasks(idx, ("BNC",))
    assert [t["name"] for t in tasks] == ["BNC Twist"]            # Approach、Align 沒有失敗 → 不是題目
    assert sc.build_tasks(idx, ("BNC",), min_each=9) == []
    with pytest.raises(ValueError):
        sc.step_instances(idx, pp.RESERVED[0])


def test_step_index_is_readable_by_extraction(tmp_path):
    extract = pytest.importorskip("assembly_monitor.extract")
    idx = _index(tmp_path)
    tasks = sc.build_tasks(idx, ("BNC",))
    p = sc.write_step_index(tasks, tmp_path / "index_steps")
    segs = extract.load_part_segments(tmp_path / "index_steps", "insert", "BNC Twist")
    assert len(segs) == 18 and sum(s["success"] is False for s in segs) == 8
    ids = {f"{s['recording_id']}__seg{s['segment_id']}" for s in segs}
    assert ids == {s["segment_id"] for s in tasks[0]["instances"]}
    a = next(s for s in segs if s["success"] is False)
    assert a["end"] - a["start"] == pytest.approx(5.0)
    assert extract.Sampling(n_frames=1, start_buffer=0.5, end_buffer=0.5).key() == sc.sampling_key(0.5)
    assert extract.Sampling(n_frames=1, start_buffer=0.5, end_buffer=0.0).key() == sc.sampling_key(0.0)
    assert extract.target_times(a["start"], a["end"], extract.Sampling(n_frames=1, start_buffer=0.5, end_buffer=1.0)) == [a["end"] + 1.0]
    # 再寫一次別的題目：原本的題目保留，同名題目整批換新
    other = [{"name": "X Align", "part": "X", "step": "Align", "instances": tasks[0]["instances"][:3]}]
    sc.write_step_index(other, tmp_path / "index_steps")
    sc.write_step_index(tasks, tmp_path / "index_steps")
    with p.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 18 + 3 and {r["part"] for r in rows} == {"BNC Twist", "X Align"}


class FakeEnc:
    key = "fake_enc"

    def __init__(self):
        self.calls = 0

    def encode(self, paths):
        self.calls += 1
        return np.array([[float(Path(p).stat().st_size), 1.0, 2.0] for p in paths], np.float32)


def _fake_frames(root, task, eb, segment_id):
    d = Path(root) / f"insert_{sc._slug(task)}" / sc.sampling_key(eb) / "segments" / segment_id
    d.mkdir(parents=True)
    name = "hand__00__t+005.000s__f000100.jpg"
    (d / name).write_bytes(b"x" * (10 + len(segment_id)))
    (d / "meta.json").write_text(json.dumps({"start_unix": 0.0, "end_unix": 5.0, "duration_s": 5.0, "window_unix": [-0.5, 5.0 + eb],
                                             "cameras": [{"camera": "hand", "frame_paths": [name], "source_times_sec": [5.0 + eb]}]}),
                                 encoding="utf-8")


def test_task_features_reads_extracted_frames_and_caches(tmp_path):
    pytest.importorskip("assembly_monitor.probe")
    idx = _index(tmp_path)
    task = sc.build_tasks(idx, ("BNC",))[0]
    fr, cache = tmp_path / "frames", tmp_path / "cache"
    for s in task["instances"][:-1]:                              # 最後一筆沒有影格
        _fake_frames(fr, task["name"], 0.5, s["segment_id"])
    enc = FakeEnc()
    f = sc.task_features(task, 0.5, enc, cache, str(fr))
    assert len(f) == 17 and task["instances"][-1]["segment_id"] not in f
    v = f[task["instances"][0]["segment_id"]]
    assert v.shape == (3,) and v[1] == 1.0
    assert (cache / "fake_enc" / sc.sampling_key(0.5) / f"{task['instances'][0]['segment_id']}.npy").exists()
    n = enc.calls
    sc.task_features(task, 0.5, enc, cache, str(fr))               # 第二次讀快取，不再編碼
    assert enc.calls == n
    assert sc.frame_path(task["name"], 0.5, task["instances"][0]["segment_id"], str(fr)).endswith(".jpg")
    assert sc.frame_path(task["name"], 0.5, task["instances"][-1]["segment_id"], str(fr)) is None
    assert sc.task_features(task, 1.0, enc, cache, str(fr)) == {}  # 這個時間點沒有擷取


def _feats(task, sep, seed, d=30, drop=()):
    rng = np.random.default_rng(seed)
    out = {}
    for s in task["instances"]:
        if s["segment_id"] in drop:
            continue
        v = rng.normal(size=d)
        v[0] += sep * s["label"]
        out[s["segment_id"]] = v
    return out


def test_evaluate_task_uses_common_instances(tmp_path):
    task = sc.build_tasks(_index(tmp_path, 14, 10), ("BNC",))[0]
    gone = task["instances"][0]["segment_id"]
    feats = {0.0: _feats(task, 0.0, 1), 0.5: _feats(task, 6.0, 2, drop=(gone,)), 1.0: _feats(task, 6.0, 3)}
    r = sc.evaluate_task(task, feats)
    assert r["n"] == 23 and r["n_dropped"] == 1 and gone not in r["ids"]
    assert r["offsets"]["0.5"]["bal_mean"] > 0.9 and r["offsets"]["0"]["bal_mean"] < 0.75
    assert len(r["offsets"]["1"]["fail_rate"]) == 23 and 0 <= r["offsets"]["1"]["fail_recall"] <= 1
    few = sc.evaluate_task(task, {eb: _feats(task, 6.0, 4, drop=tuple(s["segment_id"] for s in task["instances"] if s["label"])[:5])
                                  for eb in sc.OFFSETS})
    assert "skipped" in few and few["offsets"] == {}


def test_select_offset_rule():
    assert sc.select_offset({0.0: 0.70, 0.5: 0.715, 1.0: 0.72})["offset"] == 0.0      # 差 2 點以內 → 選早的
    assert sc.select_offset({0.0: 0.60, 0.5: 0.71, 1.0: 0.72})["offset"] == 0.5
    s = sc.select_offset({0.0: 0.60, 0.5: 0.65, 1.0: 0.80})
    assert s["offset"] == 1.0 and s["best_offset"] == 1.0 and s["avg"]["0.5"] == 0.65


def _task_result(name, step, bals):
    return {"name": name, "part": name.split()[0], "step": step, "n": 30, "n_failure": 10, "n_success": 20, "n_dropped": 0,
            "n_recordings": 9, "offsets": {k: {"bal_mean": v, "bal_min": v - 0.05, "bal_max": v + 0.05, "fail_recall": v,
                                               "ok_recall": v, "k_mean": 4.0, "chosen": {}, "fail_rate": []}
                                           for k, v in zip(("0", "0.5", "1"), bals)}}


def test_summarize_gate_and_selected_file(tmp_path):
    tasks = [_task_result("BNC Twist", "Twist", (0.6, 0.7, 0.9)), _task_result("BNC Align", "Align", (0.5, 0.52, 0.5)),
             dict(_task_result("D-SUB Align", "Align", (0, 0, 0)), skipped="太少", offsets={})]
    s = sc.summarize(tasks)
    assert s["n_tasks"] == 2 and s["offset"] == 1.0 and s["selected_avg"] == pytest.approx(0.7) and s["gate_passed"]
    assert s["by_step"] == {"Twist": pytest.approx(0.9), "Align": pytest.approx(0.5)}
    p = sc.write_selected(s, tasks, tmp_path / "sel.yaml")
    body = json.loads("\n".join(p.read_text(encoding="utf-8").splitlines()[1:]))
    assert body["offset_s"] == 1.0 and body["dev_tasks"] == ["BNC Twist", "BNC Align"] and body["recipe"] == "F2S2"
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["offset_s"] == 1.0          # 也是合法的 YAML
    before = p.read_text(encoding="utf-8")
    sc.write_selected(dict(s, offset=0.0), tasks, p)                                  # 已凍結 → 不覆蓋
    assert p.read_text(encoding="utf-8") == before
    low = sc.summarize([_task_result("BNC Align", "Align", (0.5, 0.52, 0.5))])
    assert not low["gate_passed"] and sc.write_selected(low, tasks, tmp_path / "no.yaml") is None
    assert not (tmp_path / "no.yaml").exists()
    with pytest.raises(sc.Stop):
        sc.summarize([tasks[2]])


def test_run_dev_end_to_end(tmp_path, monkeypatch):
    idx = _index(tmp_path, 14, 10)

    def fake_features(task, eb, enc, cache_root=None, frames_root=None):
        return _feats(task, {0.0: 0.5, 0.5: 6.0, 1.0: 6.0}[eb], int(eb * 10))

    monkeypatch.setattr(sc, "task_features", fake_features)
    monkeypatch.setattr(pp, "_check_crop_cache", lambda root, key: None)
    out = tmp_path / "out"
    res = sc.run_dev(index_dir=str(idx), out=out, parts=("BNC",), cache_root=tmp_path / "c", frames_root=str(tmp_path / "f"),
                     step_index=tmp_path / "steps", selected_path=tmp_path / "sel.yaml",
                     encs={"crop": FakeEnc(), "full": None}, do_extract=False, sheets=False)
    assert res["summary"]["offset"] == 0.5 and res["summary"]["gate_passed"] and (tmp_path / "sel.yaml").exists()
    assert (tmp_path / "steps" / "segments.csv").exists()
    rep = (out / "report.md").read_text(encoding="utf-8")
    assert "選結束後 0.5 秒" in rep and "過關" in rep and "BNC 旋轉" in rep and "誤判" not in rep
    json.loads((out / "result.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        sc.run_dev(index_dir=str(idx), parts=("bolt 4",), do_extract=False)
    with pytest.raises(ValueError):
        sc.run_dev(index_dir=str(idx), parts=(pp.RESERVED[0],), do_extract=False)


# ---------------------------------------------------------------- 測試階段


def _selected(tmp_path, offset=1.0):
    s = sc.summarize([_task_result("BNC Twist", "Twist", (0.6, 0.7, 0.9) if offset == 1.0 else (0.9, 0.7, 0.6))])
    return sc.write_selected(s, [_task_result("BNC Twist", "Twist", (0.6, 0.7, 0.9))], tmp_path / "sel.yaml")


def test_load_selected(tmp_path):
    with pytest.raises(sc.Stop):
        sc.load_selected(tmp_path / "none.yaml")
    p = _selected(tmp_path)
    body = sc.load_selected(p)
    assert body["offset_s"] == 1.0 and body["recipe"] == "F2S2"
    p.write_text(p.read_text(encoding="utf-8").replace('"offset_s": 1.0', '"offset_s": 3.0'), encoding="utf-8")
    with pytest.raises(sc.Stop):
        sc.load_selected(p)


def _inst(parent, failed, steps):
    """steps：[(步驟, 這一步失敗嗎)]"""
    return [{"segment_id": f"{parent}0{k}", "parent": parent, "order": k, "step": name, "label": int(bad), "parent_failed": failed}
            for k, (name, bad) in enumerate(steps)]


def test_which_step():
    inst = (_inst("a", True, [("Approach", 0), ("Align", 0), ("Twist", 1)])        # 旋轉失敗，模型判旋轉 → 對
            + _inst("b", True, [("Approach", 0), ("Align", 1), ("Twist", 1)])       # 對準先失敗，模型只判到旋轉 → 不對
            + _inst("c", True, [("Approach", 1), ("Align", 1)])                     # 接近失敗（沒有分類器），模型判對準 → 不對
            + _inst("d", True, [("Approach", 0), ("Align", 1)])                     # 對準失敗，模型判對準 → 對
            + _inst("e", True, [("Approach", 0), ("Align", 0), ("Twist", 1)])       # 旋轉失敗，模型都沒判到 → 不對
            + _inst("f", False, [("Approach", 0), ("Align", 0), ("Twist", 0)])      # 成功段，沒有被判失敗
            + _inst("g", False, [("Approach", 0), ("Align", 0), ("Twist", 0)]))     # 成功段，旋轉被判失敗
    judged = {"a02": 1, "b01": 0, "b02": 1, "c01": 1, "d01": 1, "e01": 0, "e02": 0, "f01": 0, "f02": 0, "g01": 0, "g02": 1, "a01": 0}
    tasks = []
    for step in ("Align", "Twist"):
        ids = [s["segment_id"] for s in inst if s["step"] == step]
        tasks.append({"name": f"X {step}", "step": step, "ids": ids, "judged": [judged[i] for i in ids]})
    tasks.append({"name": "X Push", "step": "Push", "skipped": "太少", "ids": [], "judged": []})
    ws = sc.which_step(inst, tasks)
    assert ws["steps_with_model"] == ["Align", "Twist"] and ws["n_failed"] == 5 and ws["n_right"] == 2
    assert ws["by_true"] == {"Twist": {"n": 2, "right": 1}, "Align": {"n": 2, "right": 1}, "Approach": {"n": 1, "right": 0}}
    assert ws["confusion"]["Align"] == {"Twist": 1, "Align": 1} and ws["confusion"]["Twist"] == {"Twist": 1, "沒判到": 1}
    assert ws["confusion"]["Approach"] == {"Align": 1}
    assert ws["always_most"] == 2 and ws["n_ok"] == 2 and ws["ok_clean"] == 1 and ws["failed_flagged"] == 4
    row = next(r for r in ws["rows"] if r["segment_id"] == "b")
    assert row["sequence"] == "Approach > Align✗ > Twist✗" and row["judged"] == "Align > Twist✗" and row["pred"] == "Twist"


def test_evaluate_test_task_and_verdict(tmp_path):
    task = sc.build_tasks(_index(tmp_path, 14, 10), ("BNC",))[0]
    f = _feats(task, 6.0, 5)
    gone = task["instances"][0]["segment_id"]
    del f[gone]
    r = sc.evaluate_test_task(task, f, n_perm=2, n_jobs=1)
    assert r["n"] == 23 and r["n_dropped"] == 1 and len(r["judged"]) == 23 and r["bal_mean"] > 0.9
    assert r["vote"]["ok_failure"] == 10 and r["perm"]["n_perm"] == 2 and r["perm"]["n_ge"] == 0
    assert sc.task_verdict(r).startswith("沒有達到")                     # 只打亂 2 次，p 最小 0.33
    assert sc.task_verdict(dict(r, perm={"p": 0.01})).startswith("達到")
    assert sc.task_verdict(dict(r, bal_mean=0.6, perm={"p": 0.01})).startswith("沒有達到")
    assert sc.task_verdict({"skipped": "x"}) == "沒做"
    few = sc.evaluate_test_task(task, {k: v for k, v in list(f.items())[:6]}, n_perm=0)
    assert "skipped" in few


def test_run_test_end_to_end(tmp_path, monkeypatch):
    idx = _index(tmp_path, 14, 10, part="bolt 4")
    sel = _selected(tmp_path)
    seen = []

    def fake_features(task, eb, enc, cache_root=None, frames_root=None):
        seen.append(eb)
        return _feats(task, 6.0, 7)

    monkeypatch.setattr(sc, "task_features", fake_features)
    monkeypatch.setattr(pp, "_check_crop_cache", lambda root, key: None)
    out = tmp_path / "out"
    kw = dict(index_dir=str(idx), out=out, cache_root=tmp_path / "c", frames_root=str(tmp_path / "f"),
              step_index=tmp_path / "steps", selected_path=sel, n_perm=2, n_jobs=1,
              encs={"crop": FakeEnc(), "full": None}, do_extract=False, sheets=False)
    res = sc.run_test(**kw)
    assert seen == [1.0] and [t["name"] for t in res["tasks"]] == ["bolt 4 Twist"]
    ws = res["which_step"]
    assert ws["n_failed"] == 10 and ws["n_right"] == 10 and ws["ok_clean"] == 14 and ws["n_ok"] == 14
    rep = (out / "report.md").read_text(encoding="utf-8")
    assert "答對 10/10 段" in rep and "bolt 4 旋轉" in rep and "成功段每一步都沒被判失敗：14/14" in rep and "誤判" not in rep
    json.loads((out / "result.json").read_text(encoding="utf-8"))
    with pytest.raises(sc.Stop):                                         # 已經有結果 → 不重跑
        sc.run_test(**kw)
    sc.run_test(**dict(kw, force=True))
    with pytest.raises(sc.Stop):                                         # 沒有凍結檔 → 停
        sc.run_test(**dict(kw, out=tmp_path / "o2", selected_path=tmp_path / "none.yaml"))
    with pytest.raises(ValueError):
        sc.run_test(**dict(kw, out=tmp_path / "o3", part=pp.RESERVED[0]))
