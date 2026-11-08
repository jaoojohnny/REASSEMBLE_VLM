"""第 2 包 indexer / validation：讀已確認映射，規格的四種情境要給出對應錯誤碼。"""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from assembly_monitor.index import index_h5, save_recording, write_index_outputs
from assembly_monitor.mapping import MappingError, load_mapping
from assembly_monitor.synthetic import SYNTHETIC_MAPPING as SYN_MAPPING


def test_mapping_must_be_confirmed(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(dict(SYN_MAPPING, status="proposed")), encoding="utf-8")
    with pytest.raises(MappingError):
        load_mapping(p)
    p.write_text(yaml.safe_dump(dict(SYN_MAPPING, best_camera_candidates=["nope"])), encoding="utf-8")
    with pytest.raises(MappingError):
        load_mapping(p)
    with pytest.raises(MappingError):
        load_mapping(tmp_path / "missing.yaml")


def test_parse_text_rules(synth_mapping):
    assert synth_mapping.parse_text("Insert USB.") == ("insert", "USB")
    assert synth_mapping.parse_text("Insert square peg 1.") == ("insert", "square peg 1")
    assert synth_mapping.parse_text("???") == (None, None)
    assert synth_mapping.is_excluded("Pick square peg 2")


def test_index_normal(make_synth, synth_mapping):
    info = make_synth("normal")
    res = index_h5(info["path"], synth_mapping)
    by = {s["segment_id"]: s for s in res["segments"]}
    assert len(by) == 4
    assert (by["1"]["action"], by["1"]["part"]) == ("insert", "round peg 1")
    assert by["2"]["excluded"] and by["2"]["validation_status"] == "excluded"
    assert by["3"]["success"] is False
    assert all(s["validation_status"] in ("ok", "excluded") for s in res["segments"]), [s["codes"] for s in res["segments"]]
    assert 18 <= by["1"]["frames"]["cam_left"] <= 23  # 0.7 s × 30 Hz
    assert by["1"]["camera_ids"] == ["cam_left", "cam_right", "cam_wrist"]
    assert abs(by["0"]["next_gap_s"]) < 1e-6
    assert by["1"]["start_sec"] == pytest.approx(0.7, abs=0.01)
    assert len(res["skills"]) == 12 and all(s["validation_status"] == "ok" for s in res["skills"])
    assert res["recording_issues"] == []
    if info["real_video"]:
        cam = res["cameras"]["cam_left"]
        assert cam.get("frames") in (None, 90) and cam["n_timestamps"] == 90


def test_index_flags_the_four_cases(make_synth, synth_mapping):
    bad = index_h5(make_synth("bad_time")["path"], synth_mapping)
    codes = {c for s in bad["segments"] for c in s["codes"]}
    assert {"E_START_GE_END", "E_OUT_OF_RANGE"} <= codes
    assert "W_TS_NON_MONOTONIC:cam_left" in bad["recording_issues"]
    assert any("W_SKILL_OUTSIDE" in s["codes"] for s in bad["skills"])
    dup = index_h5(make_synth("dup_id")["path"], synth_mapping)
    assert sum("E_DUP_ID" in s["codes"] for s in dup["segments"]) == 2
    miss = index_h5(make_synth("missing_view")["path"], synth_mapping)
    assert "W_CAMERA_MISSING:cam_right" in miss["recording_issues"]
    assert all("cam_right" not in s["frames"] for s in miss["segments"])
    assert all(s["validation_status"] in ("ok", "excluded") for s in miss["segments"])


def test_outputs_and_insert_report(make_synth, synth_mapping, tmp_path):
    out = tmp_path / "index"
    for case in ("normal", "dup_id"):
        save_recording(index_h5(make_synth(case)["path"], synth_mapping, recording_id=case), out)
    outs = write_index_outputs(out, synth_mapping)
    rows = list(csv.DictReader(open(outs["segments"], encoding="utf-8-sig")))
    assert len(rows) == 9
    need = {"recording_id", "segment_id", "source_path", "action", "part", "start_sec", "end_sec", "success",
            "camera_ids", "fps_or_timestamps", "frame_count", "validation_status", "error_code"}
    assert need <= set(rows[0])
    val = json.load(open(outs["validation"], encoding="utf-8"))
    assert val["code_counts"]["E_DUP_ID"] == 2
    assert val["n_recordings"] == 2
    cand = {r["part"]: r for r in csv.DictReader(open(outs["candidates"], encoding="utf-8-sig"))}
    assert (int(cand["square peg 2"]["n_segments"]), int(cand["square peg 2"]["n_failure"])) == (2, 2)
    assert int(cand["round peg 1"]["n_segments"]) == 3
    assert "Insert" in open(outs["action_summary"], encoding="utf-8").read()
    skills = list(csv.DictReader(open(outs["skills"], encoding="utf-8-sig")))
    assert len(skills) == 24
