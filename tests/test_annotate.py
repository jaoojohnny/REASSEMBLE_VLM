"""第 4 包 P3-3：標註任務選段、密集影格與標註頁、收檔、參考標籤、比較與合併的整條流程。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assembly_monitor.annotate import (EXPORT_RE, build_task, collect_exports, dense_targets, run_compare, run_merge,
                                       select_task_segments, task_segments, write_reference)
from assembly_monitor.schema import SegmentInfo, load_segment_infos, normalize_record, read_annotations, write_jsonl


def _segs(n_fail=6, n_ok=6):
    out = {}
    for i in range(n_fail + n_ok):
        sid = f"2025-01-09-10-{i:02d}-00__seg{i}"
        out[sid] = SegmentInfo(segment_id=sid, recording_id=sid.split("__")[0], segment_index=i, action="insert", part="BNC",
                               text="Insert BNC.", start_unix=100.0 * i, end_unix=100.0 * i + 10.0,
                               dataset_result="failure" if i < n_fail else "success", validation_status="ok")
    return out


def test_select_task_segments(tmp_path):
    segs = _segs()
    sel = tmp_path / "selection_pilot.csv"
    sel.write_text("recording_id,segment_id\n" + "".join(f"{s.recording_id},{s.segment_index}\n" for s in segs.values()), encoding="utf-8")
    a = select_task_segments(segs, sel, n_fail=4, n_ok=3, seed=0)
    b = select_task_segments(segs, sel, n_fail=4, n_ok=3, seed=0)
    assert [s.segment_id for s in a] == [s.segment_id for s in b]
    assert [s.dataset_result for s in a] == ["failure"] * 4 + ["success"] * 3
    assert len(select_task_segments(segs, sel, n_fail=20, n_ok=10, seed=0)) == 12  # 不足時全收


def test_dense_targets_cover_window():
    seg = _segs()[next(iter(_segs()))]
    ts = dense_targets(seg, 0.5, 2.0, 5.0)
    assert ts[0] == pytest.approx(seg.start_unix - 0.5) and ts[-1] == pytest.approx(seg.end_unix + 2.0)
    assert len(ts) == 64 and max(b - a for a, b in zip(ts, ts[1:])) <= 0.2 + 1e-9


def test_export_filename_pattern():
    m = EXPORT_RE.match("pilot30__小明__20261005-153000.jsonl")
    assert m and m.group("annot") == "小明" and not m.group("adj")
    m = EXPORT_RE.match("pilot30__adjudication__A__20261005-153000.csv")
    assert m and m.group("adj") and m.group("annot") == "A"
    assert EXPORT_RE.match("other__A__20261005-153000.jsonl").group("task") == "other"


@pytest.fixture
def prepared(make_synth, synth_mapping, tmp_path):
    pytest.importorskip("cv2")
    from assembly_monitor.index import index_h5, save_recording, write_index_outputs
    from assembly_monitor.slim import make_slim

    idx, slim = tmp_path / "index", tmp_path / "slim"
    for case, rid in (("normal", "rec_a"), ("missing_view", "rec_b")):
        info = make_synth(case, name=f"{rid}.h5")
        if not info["real_video"]:
            pytest.skip("沒有 OpenCV 影片編碼器")
        save_recording(index_h5(info["path"], synth_mapping, recording_id=rid), idx)
        make_slim(info["path"], slim / f"{rid}.h5", synth_mapping)
    write_index_outputs(idx, synth_mapping)
    return idx, slim


def test_build_task_end_to_end(prepared, synth_mapping, ann_cfg, tmp_path):
    idx, slim = prepared
    segs = {**load_segment_infos(idx, "insert", "round peg 1"), **load_segment_infos(idx, "insert", "square peg 2")}
    chosen = sorted(segs.values(), key=lambda s: (s.dataset_result != "failure", s.segment_id))
    out = tmp_path / "pages"
    res = build_task(mapping=synth_mapping, cfg=ann_cfg, segs=segs, chosen=chosen, index_dir=idx, slim_dir=slim, out_root=out,
                     task="t1", title="測試", workers=1)
    assert res["errors"] == [] and res["n_done"] == len(chosen) == 4
    td = out / "t1"
    assert (td / "index.html").exists() and (td / "adjudicate.html").exists() and (td / "task.json").exists()
    js = (td / "data.js").read_text(encoding="utf-8")
    data = json.loads(js[js.index("=") + 1 : js.rstrip().rindex(";")])
    assert [s["id"] for s in data["segments"]] == [s.segment_id for s in chosen]
    assert data["config"]["cameras"] == ["cam_left", "cam_right", "cam_wrist"] and "unknown" in data["config"]["reasons"]
    for s in data["segments"]:
        w0, w1 = s["window"]
        assert s["skills"] and all(k["text"] in ("Approach", "Align", "Push") for k in s["skills"])
        for cam, c in s["cams"].items():
            if s["recording_id"] == "rec_b" and cam == "cam_right":
                assert c["t"] == []  # 缺少的視角沒有影格
                continue
            assert c["t"], (s["id"], cam)
            assert all(w0 - 0.04 <= t <= w1 + 0.04 for t in c["t"]) and c["t"] == sorted(c["t"])
            for k in range(len(c["t"])):
                assert (td / "frames" / s["id"] / f"{cam}__{k:04d}.jpg").exists()
        assert (td / "review" / f"{s['id']}__overview.jpg").exists() and (td / "review" / f"{s['id']}__cam_wrist.jpg").exists()
    # 快取：再跑一次全部沿用
    res2 = build_task(mapping=synth_mapping, cfg=ann_cfg, segs=segs, chosen=chosen, index_dir=idx, slim_dir=slim, out_root=out,
                      task="t1", title="測試", workers=1)
    assert res2["n_done"] == 0 and res2["n_skipped"] == 4


def test_collect_compare_merge_flow(fake_index, ann_cfg, tmp_path):
    idx = fake_index([(f"2025-01-09-1{i}-00-00", [(10.0, 30.0, i % 2 == 1, [("Approach", 0, 5, True), ("Align", 5, 12, i % 2 == 1),
                                                                            ("Twist", 12, 18, True)])]) for i in range(6)])
    segs = load_segment_infos(idx, "insert", "BNC")
    task_dir = tmp_path / "pages" / "pilot30"
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(json.dumps({"segments": list(segs)}), encoding="utf-8")
    tsegs = task_segments(task_dir, segs)
    ann = tmp_path / "ann" / "pilot30"

    def page_record(seg, annot, **kw):
        r = {"segment_id": seg.segment_id, "recording_id": seg.recording_id, "segment_index": seg.segment_index, "part": "BNC",
             "dataset_result": seg.dataset_result, "result": seg.dataset_result,
             "failure_reason": "misalignment" if seg.dataset_result == "failure" else None,
             "evidence_start_sec": 5.0, "evidence_end_sec": 12.0, "best_camera": "cam_wrist", "visible_evidence": "看得到",
             "notes": "", "camera_visibility": {"cam_wrist": "clear"}, "annotator": annot, "annotator_type": "human",
             "label_version": "v1", "reason_set": "bnc-pilot-v1", "review_status": "submitted", "source": "annotate-page"}
        r.update(kw)
        return r

    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    first = list(tsegs.values())
    (downloads / "pilot30__小明__20261005-120000.jsonl").write_text(
        "\n".join(json.dumps(page_record(s, "小明")) for s in first[:3]) + "\n", encoding="utf-8")
    (downloads / "pilot30__小明__20261005-130000.jsonl").write_text(
        "\n".join(json.dumps(page_record(s, "小明", notes="第二次匯出")) for s in first) + "\n", encoding="utf-8")
    (downloads / "other__小明__20261005-130000.jsonl").write_text("{}\n", encoding="utf-8")
    rep = collect_exports([downloads], "pilot30", ann, ann_cfg, tsegs)
    assert [c["file"] for c in rep["collected"]] == ["pilot30__小明__20261005-120000.jsonl", "pilot30__小明__20261005-130000.jsonl"]
    human = read_annotations(ann / "human__小明.jsonl", ann_cfg)
    assert len(human) == 6 and all(r["notes"] == "第二次匯出" for r in human)
    assert (ann / "inbox" / "pilot30__小明__20261005-120000.jsonl").exists()
    # Claude 與資料集參考
    ai = []
    for s in first:
        kw = {"annotator_type": "ai"}
        if s.dataset_result == "failure" and s.recording_id == "2025-01-09-10-00-00":
            kw["failure_reason"] = "incomplete_insert"   # 只有這一段和主標註者不同
        ai.append(normalize_record(page_record(s, "claude", **kw), ann_cfg))
    write_jsonl(ai, ann / "ai__claude.jsonl", ann_cfg)
    write_reference(tsegs, idx, ann, ann_cfg)
    res = run_compare(ann, task_dir, ann_cfg, "pilot30")
    assert res["primary"] == "A" and res["second"] == "B" and res["reference"] == "C"
    assert len(res["disputes"]) == 1 and (task_dir / "compare_data.js").exists() and Path(res["agreement_html"]).exists()
    m = res["pairs"]["A|C"]["metrics"]
    assert m["failure_reason"]["n"] == 3 and m["failure_reason"]["agree"] == 3  # 參考標籤：Align 失敗 → misalignment
    # 尚未裁決 → disputed；收進裁決檔 → adjudicated
    rep_m = run_merge(ann, ann_cfg)
    assert rep_m["approved"] == 5 and len(rep_m["disputed_unresolved"]) == 1
    sid = res["disputes"][0]["segment_id"]
    (downloads / "pilot30__adjudication__小明__20261005-140000.jsonl").write_text(
        json.dumps(page_record(tsegs[sid], "小明", review_status="adjudicated", source="adjudication-page")) + "\n", encoding="utf-8")
    collect_exports([downloads], "pilot30", ann, ann_cfg, tsegs)
    rep_m2 = run_merge(ann, ann_cfg)
    assert rep_m2["adjudicated"] == 1 and rep_m2["approved"] == 5 and not rep_m2["disputed_unresolved"]
    gold = read_annotations(ann / "gold.jsonl", ann_cfg)
    assert len(gold) == 6 and {g["review_status"] for g in gold} == {"approved", "adjudicated"}
    assert (ann / "gold.csv").exists()


def test_refresh_page_updates_config_only(tmp_path, ann_cfg):
    from assembly_monitor.annotate import refresh_page

    td = tmp_path / "pilot30"
    (td / "frames" / "x").mkdir(parents=True)
    (td / "frames" / "x" / "hand__0000.jpg").write_bytes(b"jpg")
    old = {"task": "pilot30", "title": "t", "config": {"label_version": "v1", "reasons": {"unknown": {}}}, "segments": [{"id": "x"}]}
    (td / "data.js").write_text("window.ANNOT = " + json.dumps(old, ensure_ascii=False) + ";\n", encoding="utf-8")
    pages = refresh_page(td, ann_cfg)
    text = Path(pages["data"]).read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    assert data["config"]["label_version"] == ann_cfg.label_version and "board_lifted" in data["config"]["reasons"]
    assert data["config"]["difficulty"] and data["segments"] == [{"id": "x"}] and "refreshed_at" in data
    html = (td / "index.html").read_text(encoding="utf-8")
    assert "fDiff" in html and "visual_difficulty" in html
    assert (td / "frames" / "x" / "hand__0000.jpg").read_bytes() == b"jpg"
    with pytest.raises(FileNotFoundError):
        refresh_page(tmp_path / "missing", ann_cfg)


def test_rest_segments_excludes_existing_tasks(fake_index, tmp_path):
    from assembly_monitor.annotate import rest_segments

    idx = fake_index([(f"2025-01-09-1{i}-00-00", [(10.0, 30.0, i % 2 == 1, [("Approach", 0, 5, True)]),
                                                  (40.0, 60.0, True, [("Approach", 0, 5, True)])]) for i in range(3)])
    segs = load_segment_infos(idx, "insert", "BNC")
    ids = sorted(segs)
    pilot = tmp_path / "pages" / "pilot30"
    pilot.mkdir(parents=True)
    (pilot / "task.json").write_text(json.dumps({"segments": ids[:2]}), encoding="utf-8")
    rest = rest_segments(segs, [pilot])
    assert [s.segment_id for s in rest] == sorted(ids[2:], key=lambda k: (segs[k].recording_id, segs[k].segment_index))
    assert not set(s.segment_id for s in rest) & set(ids[:2]) and len(rest) == len(segs) - 2
    with pytest.raises(FileNotFoundError):
        rest_segments(segs, [tmp_path / "pages" / "nope"])
