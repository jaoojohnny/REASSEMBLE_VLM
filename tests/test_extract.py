"""第 3 包影格擷取：時間落在範圍、含首尾、同設定同幀、快取不重做、換設定重建、缺幀標記。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from assembly_monitor.extract import Sampling, pick_frames, run_extraction, select_pilot, target_times


def test_target_times_include_first_and_last():
    s1 = Sampling(n_frames=1, start_buffer=0.5, end_buffer=2.0)
    assert target_times(10.0, 12.0, s1) == [14.0]
    s4 = Sampling(n_frames=4, start_buffer=0.5, end_buffer=2.0)
    assert target_times(10.0, 12.0, s4) == pytest.approx([9.5, 11.0, 12.5, 14.0])
    assert Sampling(n_frames=8).key() == "uniform_n8_sb0.5_eb2_jpg"


def test_pick_frames_window_and_missing():
    ts = np.arange(0, 20, 0.1)
    picks = pick_frames(ts, [9.5, 14.0], 9.5, 14.0)
    assert [p["frame_idx"] for p in picks] == [95, 140]
    assert all(9.5 - 1e-9 <= p["source_unix"] <= 14.0 + 1e-9 for p in picks)
    none = pick_frames(ts, [30.0], 25.0, 30.0)
    assert none[0]["missing"] == "no_frame_in_window" and none[0]["frame_idx"] is None


def test_select_pilot_counts_and_spread():
    segs = [{"recording_id": f"r{i % 5}", "segment_id": str(i), "success": i % 3 != 0, "validation_status": "ok",
             "start": 0.0, "end": 1.0, "text": "Insert X."} for i in range(30)]
    p1 = select_pilot(segs, n_fail=4, n_ok=5, seed=3)
    p2 = select_pilot(segs, n_fail=4, n_ok=5, seed=3)
    assert p1 == p2
    assert sum(1 for s in p1 if s["success"] is False) == 4 and sum(1 for s in p1 if s["success"]) == 5
    assert len({s["recording_id"] for s in p1[:4]}) == 4  # 失敗樣本來自不同錄製檔


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


def _run(synth_mapping, idx, slim, out, **kw):
    args = dict(mapping=synth_mapping, index_dir=idx, slim_dir=slim, out_dir=out, part="round peg 1",
                which="all", n_frames_list=(1, 4), workers=1)
    args.update(kw)
    return run_extraction(**args)


def test_extract_end_to_end(prepared, synth_mapping, tmp_path):
    idx, slim = prepared
    out = tmp_path / "frames"
    res = _run(synth_mapping, idx, slim, out)
    assert res["errors"] == [] and res["n_done"] == 4 and res["n_skipped"] == 0
    root = out / "insert_round_peg_1"
    metas = {}
    for key in ("uniform_n1_sb0.5_eb2_jpg", "uniform_n4_sb0.5_eb2_jpg"):
        d = root / key
        assert (d / "samples.csv").exists() and (d / "manifest.json").exists() and (d / "index.html").exists()
        for mp in sorted((d / "segments").glob("*/meta.json")):
            m = json.loads(mp.read_text(encoding="utf-8"))
            metas[(key, m["recording_id"])] = m
            w0, w1 = m["window_unix"]
            assert (mp.parent / "sheet.jpg").exists()
            assert {c["camera"] for c in m["cameras"]} == {"cam_left", "cam_right", "cam_wrist"}
            for cm in m["cameras"]:
                assert set(cm) >= {"segment_id", "camera", "source_times_sec", "frame_paths", "sampling", "n_frames"}
                for e, fpth in zip(cm["frames"], cm["frame_paths"]):
                    if e.get("missing"):
                        assert fpth is None
                        continue
                    assert w0 - 1e-6 <= e["source_unix"] <= w1 + 1e-6
                    assert (mp.parent / fpth).exists()
                    assert cm["camera"] in fpth and "__t" in fpth  # 檔名標記相機與時間
    # rec_b 少了 cam_right → 全部標 camera_missing
    mb = metas[("uniform_n4_sb0.5_eb2_jpg", "rec_b")]
    right = next(c for c in mb["cameras"] if c["camera"] == "cam_right")
    assert all(e["missing"] == "camera_missing" for e in right["frames"])
    # n=4 首尾：第一幀靠近視窗起點，最後一幀是視窗內最後一幀
    ma = metas[("uniform_n4_sb0.5_eb2_jpg", "rec_a")]
    left = next(c for c in ma["cameras"] if c["camera"] == "cam_left")
    assert left["frames"][0]["source_unix"] - ma["window_unix"][0] < 0.05
    m1 = metas[("uniform_n1_sb0.5_eb2_jpg", "rec_a")]
    one = next(c for c in m1["cameras"] if c["camera"] == "cam_left")["frames"][0]
    assert one["frame_idx"] == left["frames"][-1]["frame_idx"]
    rows = list(csv.DictReader(open(root / "uniform_n4_sb0.5_eb2_jpg" / "samples.csv", encoding="utf-8-sig")))
    assert len(rows) == 2 * 3 * 4

    # 快取：同設定第二次不重做
    res2 = _run(synth_mapping, idx, slim, out)
    assert res2["n_done"] == 0 and res2["n_skipped"] == 4
    # 同設定強制重做 → 同一組 frame ID
    res3 = _run(synth_mapping, idx, slim, out, force=True)
    assert res3["n_done"] == 4
    again = json.loads((root / "uniform_n4_sb0.5_eb2_jpg" / "segments" / "rec_a__seg1" / "meta.json").read_text(encoding="utf-8"))
    assert [e["frame_idx"] for e in next(c for c in again["cameras"] if c["camera"] == "cam_left")["frames"]] == \
        [e["frame_idx"] for e in left["frames"]]
    # 換設定 → 另一個資料夾
    res4 = _run(synth_mapping, idx, slim, out, n_frames_list=(8,))
    assert res4["n_done"] == 2 and (root / "uniform_n8_sb0.5_eb2_jpg").exists()
    assert Path(res4["top_index"]).exists()


def test_extract_pilot_mode(prepared, synth_mapping, tmp_path):
    idx, slim = prepared
    res = _run(synth_mapping, idx, slim, tmp_path / "f", part="square peg 2", which="pilot", n_fail=2, n_ok=2,
               n_frames_list=(4,))
    assert res["n_done"] == 2  # 只有 2 段失敗、沒有成功
    html_text = Path(res["gallery"]).read_text(encoding="utf-8")
    assert "失敗" in html_text and "sheet.jpg" in html_text
    sel = list(csv.DictReader(open(res["selection"], encoding="utf-8-sig")))
    assert len(sel) == 2


def test_extract_selected_segments_keep_pilot_gallery(prepared, synth_mapping, tmp_path):
    """--set dev：只做指定段；圖庫另存 index_dev.html，不覆蓋既有的 index.html。"""
    idx, slim = prepared
    out = tmp_path / "f"
    _run(synth_mapping, idx, slim, out, n_frames_list=(4,))
    set_dir = out / "insert_round_peg_1" / "uniform_n4_sb0.5_eb2_jpg"
    before = (set_dir / "index.html").read_text(encoding="utf-8")
    res = _run(synth_mapping, idx, slim, out, which="dev", segment_ids=["rec_a__seg1", "rec_x__seg9"], n_frames_list=(4,))
    assert res["n_done"] == 0 and res["n_skipped"] == 1  # 已做過的段沿用快取；不存在的段只警告
    assert (set_dir / "index_dev.html").exists() and (set_dir / "index.html").read_text(encoding="utf-8") == before
    sel = list(csv.DictReader(open(res["selection"], encoding="utf-8-sig")))
    assert [r["recording_id"] for r in sel] == ["rec_a"]
