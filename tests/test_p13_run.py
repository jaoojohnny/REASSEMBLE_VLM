"""第 13 包第四步：REBOOT 圓柱的段、取格位置、切分、計分（不連網、不跑模型）。"""

import json
from pathlib import Path

from assembly_monitor import reboot_run as rr
from assembly_monitor.split import verify_split, load_split

ROOT = Path(__file__).resolve().parent.parent
FPS = 30.0


def _step3():
    segs = [{"id": "exp00", "source": "expert", "label": "success", "episode": 0, "start_frame": 328, "end_frame": 418, "arm": "right"},
            {"id": "rec56", "source": "recovery", "label": "failure", "episode": 56, "start_frame": 450, "end_frame": 540,
             "arm": "right", "mode": "sub_mm_misalignment", "phase": 5},
            {"id": "rec47", "source": "recovery", "label": "failure", "episode": 47, "start_frame": 360, "end_frame": 450,
             "arm": "right", "mode": "retention_failure", "phase": 5}]
    rel = {"0": {"ok": True, "onset": 418, "cross": 424, "arm": "right", "flags": []}}
    return {"fps": FPS, "segments": segs}, rel


def test_model_segments():
    step3, rel = _step3()
    segs = {s["segment_id"]: s for s in rr.model_segments(step3, rel, FPS)}
    assert set(segs) == {"reboot-cyl-exp00__seg1", "reboot-cyl-exp00__seg2", "reboot-cyl-rec56__seg0", "reboot-cyl-rec47__seg0"}
    a, b, f = segs["reboot-cyl-exp00__seg1"], segs["reboot-cyl-exp00__seg2"], segs["reboot-cyl-rec56__seg0"]
    assert a["last_frame"] == 418 and b["last_frame"] == 424 + 30 and f["last_frame"] == 540
    assert abs(a["end_s"] - (418 - 15) / FPS) < 1e-6 and abs(a["end_s"] - a["start_s"] - 3.0) < 1e-6
    assert f["main"] and not segs["reboot-cyl-rec47__seg0"]["main"]


def test_frame_plan_last_frame_is_decision_moment():
    step3, rel = _step3()
    segs = {s["segment_id"]: s for s in rr.model_segments(step3, rel, FPS)}
    for sid, last in (("reboot-cyl-exp00__seg1", 418), ("reboot-cyl-rec56__seg0", 540), ("reboot-cyl-exp00__seg2", 454)):
        plan = rr.frame_plan(segs[sid], 898, FPS)
        assert len(plan) == 4 and plan[-1]["frame"] == last and plan[0]["frame"] == last - 120
        assert [p["frame"] for p in plan] == sorted(p["frame"] for p in plan)


def test_write_split_verifies(tmp_path):
    step3, rel = _step3()
    segs = rr.model_segments(step3, rel, FPS)
    d = tmp_path / "split"
    man = rr.write_split(segs, d)
    assert verify_split(d) == [] and man["counts"]["dev"]["n_segments"] == 4
    rows = load_split(d, include=["dev"])
    assert {r["dataset_result"] for r in rows} == {"success", "failure"}
    man2 = rr.write_split(segs, d)  # 內容相同就不重寫
    assert man2["created_at"] == man["created_at"]


def _runs(gpt, v3, prob):
    def pred(d):
        return {sid: {"segment_id": sid, "status": "ok" if r else "invalid", "result": r} for sid, r in d.items()}
    return {"gpt": {"pred": pred(gpt), "raw": {sid: {"cost_usd": 0.01} for sid in gpt}},
            "v3": {"pred": pred(v3), "raw": {}}, "prob": {"pred": pred(prob), "raw": {}}}


def test_answer_and_score():
    A, F = "reboot-cyl-exp00__seg1", "reboot-cyl-rec56__seg0"
    runs = _runs({A: "success", F: "failure"}, {A: "success", F: None}, {A: "failure", F: "success"})
    assert rr.answer(runs, "v3_gpt", A) == ("success", False)
    assert rr.answer(runs, "v3_gpt", F) == ("failure", True)   # 第三版沒答案 → 問 GPT
    assert rr.answer(runs, "prob_gpt", F) == ("success", False)  # 門檻答成功就停
    segs = [{"segment_id": A, "label": "success"}, {"segment_id": F, "label": "failure"}]
    s = rr.score(segs, runs, "gpt")
    assert s["ok_success"] == 1 and s["ok_failure"] == 1 and s["bal"] == 1.0 and s["gpt_calls"] == 2
    assert rr.score(segs, runs, "majority")["bal"] == 0.5
    assert rr.score(segs, runs, "v3")["no_answer"] == 1


def test_evaluate_report(tmp_path):
    step3, rel = _step3()
    segs = rr.model_segments(step3, rel, FPS)
    out = tmp_path / "out"
    out.mkdir()
    (out / "segments.json").write_text(json.dumps({"segments": segs}), encoding="utf-8")
    for key, (exp, model) in rr.RUNS.items():
        d = out / exp / model / "f4"
        d.mkdir(parents=True, exist_ok=True)
        with (d / "predictions.jsonl").open("a", encoding="utf-8") as fh:
            for s in segs:
                fh.write(json.dumps({"segment_id": s["segment_id"], "status": "ok", "result": s["label"]}) + "\n")
    res = rr.evaluate(out)
    assert res["variants"]["A"]["gpt"]["bal"] == 1.0 and res["variants"]["A"]["gpt"]["n_failure"] == 1
    assert res["excluded"][0]["segment_id"] == "reboot-cyl-rec47__seg0"
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "放開前" in text and "放開後" in text and "retention_failure" in text
    assert (out / "per_segment.csv").exists()


def test_experiment_configs_match_frozen():
    """REBOOT 的實驗設定只能改名稱、切分、集合與影格資料夾，其他和第 12 包凍結的完全相同。"""
    import yaml

    for kind in ("gpt", "v3"):
        new = yaml.safe_load((ROOT / f"configs/experiment_reboot_cyl_{kind}.yaml").read_text(encoding="utf-8"))["experiment"]
        old = yaml.safe_load((ROOT / f"configs/experiment_test_rpeg1_{kind}.yaml").read_text(encoding="utf-8"))["experiment"]
        diff = {k for k in set(new) | set(old) if new.get(k) != old.get(k)}
        assert diff == {"name", "split_dir", "include", "frames_root"}, diff
        assert new["split_dir"] == rr.SPLIT_DIR and new["frames_root"] == rr.FRAMES_ROOT
        assert new["sampling_template"].format(n=4) == rr.SAMPLING_KEY and new["frames"] == [4]
        assert f"experiments/{new['name']}" == rr.OUT + "/" + kind


def test_bat_ascii_crlf():
    b = (ROOT / "run_reboot_cyl_models.bat").read_bytes()
    assert all(x < 128 for x in b)
    assert b.count(b"\n") == b.count(b"\r\n") > 10
    for s in (b"scripts.reboot_run cam-low", b"scripts.reboot_run prepare", b"experiment_reboot_cyl_gpt.yaml",
              b"experiment_reboot_cyl_v3.yaml", b"scripts.reboot_run evaluate", b"test_p13_run.py"):
        assert s in b, s


def test_extract_meta_readable_by_vlm(tmp_path, monkeypatch):
    """extract 寫出的 meta.json 要讓 vlm.load_segment_frames 讀得到 12 格（3 台相機 × 4 格），時間相對段起點。"""
    import numpy as np

    from assembly_monitor import reboot_data as rd
    from assembly_monitor.vlm import load_segment_frames

    step3, rel = _step3()
    segs = rr.model_segments(step3, rel, FPS)
    vids = {c: {"path": f"videos/{c}.mp4", "from": 0.0, "to": 29.9} for c in rd.CAMS + ("cam_low",)}

    def fake_eps(local, cams=rd.CAMS):
        return [{"episode_index": e, "length": 898, "videos": vids} for e in (0, 47, 56)]

    class FakeReader:
        def __init__(self, root):
            self.root = root

        def read_at(self, rel_path, t, fps):
            return np.full((48, 64, 3), 100, np.uint8), t

        def close(self):
            pass

    monkeypatch.setattr(rd, "read_episodes", fake_eps)
    monkeypatch.setattr(rd, "FrameReader", FakeReader)
    dest = tmp_path / "raw"
    for repo in rd.CYL.values():
        for c in vids:
            p = dest / rd.local_name(repo) / "videos" / f"{c}.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    froot = tmp_path / "frames"
    st = rr.extract(segs, dest, froot, FPS)
    assert st["done"] == 4 and st["missing"] == 0
    assert rr.extract(segs, dest, froot, FPS)["skipped"] == 4
    si = load_segment_frames(froot, rr.SAMPLING_KEY, "reboot-cyl-rec56__seg0", ["hama1", "hama2", "hand"])
    assert si is not None and len(si.frames) == 12 and si.n_missing == 0
    assert abs(si.window[0] + 0.5) < 1e-6 and abs(si.window[1] - 3.5) < 1e-6
    assert abs(si.frames[-1].time_sec - 3.5) < 0.02 and si.frames[-1].camera == "hand"
    meta = json.loads((froot / rr.SAMPLING_KEY / "segments" / "reboot-cyl-rec56__seg0" / "meta.json").read_text(encoding="utf-8"))
    assert [c["source_camera"] for c in meta["cameras"]] == ["cam_high", "cam_low", "cam_right_wrist"]
