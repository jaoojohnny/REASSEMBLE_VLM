"""inspect_h5 + candidates：規格要求的四種合成情境（正常、多視角缺失、非法時間、重複 ID）與改名情境。"""

from __future__ import annotations

import json

import pytest
import yaml

from assembly_monitor.inspect_h5 import InspectOptions, inspect_file
from assembly_monitor.report import render_tree, write_file_outputs


def _codes(res):
    return {a["code"] for a in res["candidates"]["anomalies"]}


def _node(res, path):
    return next(n for n in res["nodes"] if n["path"] == path)


def test_normal_detects_segments_cameras_and_time(make_synth):
    info = make_synth("normal")
    res = inspect_file(info["path"], InspectOptions())
    c = res["candidates"]

    top = [ct for ct in c["segments"]["containers"] if ct["level"] == 0]
    assert [ct["path"] for ct in top] == ["/segments_info"]
    keys = top[0]["keys"]
    assert (keys["start"], keys["end"], keys["success"], keys["text"]) == ("start", "end", "success", "text")
    assert c["segments"]["n_level0"] == 4
    assert c["segments"]["n_level1plus"] == 12
    low = [ct for ct in c["segments"]["containers"] if ct["level"] == 1]
    assert {ct["pattern"] for ct in low} == {"/segments_info/*/Low_level"}

    cams = {cm["name"]: cm for cm in c["streams"]["cameras"]}
    assert set(cams) == {"cam_left", "cam_right", "cam_wrist"}
    assert cams["cam_left"]["timestamps"] == "/timestamps/cam_left"
    assert cams["cam_left"]["ts_n"] == 90
    assert cams["cam_left"]["access"] == "file-offset"
    if info["real_video"] and cams["cam_left"]["probe_ok"]:
        assert cams["cam_left"]["frames"] == 90
        assert abs(cams["cam_left"]["fps_nominal"] - 30.0) < 0.5
        assert cams["cam_left"]["frames_minus_ts"] == 0
    assert [a["name"] for a in c["streams"]["audio"]] == ["mic_wrist"]
    others = {o["path"] for o in c["streams"]["other_streams"]}
    assert {"/robot/joint_pos", "/robot/force", "/events"} <= others
    assert c["streams"]["unpaired_time_arrays"] == []

    tb = c["time_base"]
    assert tb["relation"] == "same_as"
    assert tb["unit_guess"] == "s"

    assert [a for a in c["anomalies"] if a["severity"] == "error"] == []
    lvl0 = c["segments"]["actions"]["level0"]
    assert lvl0["insert"]["count"] == 2
    assert lvl0["insert"]["success_false"] == 1
    assert set(lvl0["insert"]["objects"]) == {"round peg 1", "square peg 2"}
    assert len(c["manual_check"]) == 4
    assert c["questions"]


def test_missing_view_is_reported(make_synth):
    info = make_synth("missing_view")
    res = inspect_file(info["path"], InspectOptions(expected_cameras=info["all_video_names"]))
    codes = _codes(res)
    assert "CAM_EXPECTED_MISSING" in codes
    assert "TS_UNPAIRED" in codes
    assert res["candidates"]["streams"]["expected_cameras_missing"] == ["cam_right"]
    assert "/timestamps/cam_right" in res["candidates"]["streams"]["unpaired_time_arrays"]


def test_bad_time_is_reported(make_synth):
    res = inspect_file(make_synth("bad_time")["path"], InspectOptions())
    codes = _codes(res)
    for code in ("SEG_START_GE_END", "SEG_OUT_OF_RANGE", "TS_NON_MONOTONIC", "SEG_CHILD_OUTSIDE_PARENT"):
        assert code in codes, code
    ts = next(t for t in res["candidates"]["streams"]["time_arrays"] if t["path"] == "/timestamps/cam_left")
    assert ts["monotonic_non_decreasing"] is False


def test_duplicate_id_is_reported(make_synth):
    res = inspect_file(make_synth("dup_id")["path"], InspectOptions())
    codes = _codes(res)
    assert "SEG_DUPLICATE_ID" in codes
    assert "SEG_DUPLICATE_INTERVAL" in codes
    dup = next(a for a in res["candidates"]["anomalies"] if a["code"] == "SEG_DUPLICATE_ID")
    assert sorted(dup["ids"]) == ["01", "1"]


def test_renamed_keys_still_detected(make_synth):
    """完全不同的名稱也要找得到：證明沒有寫死 REASSEMBLE 的 key。"""
    res = inspect_file(make_synth("renamed")["path"], InspectOptions())
    c = res["candidates"]
    top = [ct for ct in c["segments"]["containers"] if ct["level"] == 0]
    assert [ct["path"] for ct in top] == ["/annotations"]
    k = top[0]["keys"]
    assert (k["start"], k["end"], k["success"], k["text"]) == ("begin", "stop", "ok", "description")
    assert {cm["name"] for cm in c["streams"]["cameras"]} == {"view_a", "view_b", "view_c"}
    assert all(cm["timestamps"].startswith("/clock/") for cm in c["streams"]["cameras"])
    link = _node(res, "/latest")
    assert link["kind"] == "softlink" and link["link_target"] == "/annotations/0"


def test_large_arrays_are_only_previewed(make_synth):
    res = inspect_file(make_synth("normal")["path"], InspectOptions(preview_n=3, max_read_mb=0.01))
    jp = _node(res, "/robot/joint_pos")
    assert jp["shape"] == [3000, 7]
    assert jp["compression"] == "gzip" and jp["chunks"] == [500, 7]
    corner = jp["preview"]["corner"]
    assert len(corner) == 3 and len(corner[0]) == 3
    ts = _node(res, "/timestamps/joint_pos")["time_stats"]
    assert ts["sampled"] is True and ts["n"] == 3000
    assert abs(ts["rate_hz"] - 1000.0) < 1.0


def test_outputs_are_written(make_synth, tmp_path):
    res = inspect_file(make_synth("normal")["path"], InspectOptions())
    outs = write_file_outputs(res, tmp_path / "out")
    tree = open(outs["tree"], encoding="utf-8").read()
    assert "[G]" in tree and "cam_left" in tree and "segments_info" in tree
    cands = json.load(open(outs["candidates"], encoding="utf-8"))
    assert cands["segments"]["n_level0"] == 4
    nodes = json.load(open(outs["nodes"], encoding="utf-8"))
    assert any(n["path"] == "/segments_info/0/text" for n in nodes["nodes"])
    q = open(outs["questions"], encoding="utf-8").read()
    assert "人工核對" in q and "Insert round peg 1" in q
    mapping = yaml.safe_load(open(outs["mapping"], encoding="utf-8"))
    assert mapping["status"] == "proposed"
    assert mapping["segments"]["container"] == "/segments_info"
    assert mapping["cameras"]["cam_left"]["timestamps"] == "/timestamps/cam_left"


def test_tree_collapses_many_homogeneous_groups(make_synth):
    res = inspect_file(make_synth("normal")["path"], InspectOptions())
    tree = render_tree(res, collapse_threshold=2)
    assert "同構子群組省略" in tree


@pytest.mark.parametrize("case", ["normal", "missing_view", "bad_time", "dup_id", "renamed"])
def test_every_case_runs_without_crash_on_fake_video(make_synth, case):
    info = make_synth(case, real_video=False, name=f"{case}_fake.h5")
    res = inspect_file(info["path"], InspectOptions())
    for cm in res["candidates"]["streams"]["cameras"]:
        assert cm["encoding"] == "mp4"
