"""人工核對短片：從合成檔的 inspect 結果挑段、對齊多視角、輸出影片與清單。"""

from __future__ import annotations

import json

import pytest

from assembly_monitor.check_clips import export_check_clips, select_segments
from assembly_monitor.inspect_h5 import InspectOptions, inspect_file
from assembly_monitor.report import write_file_outputs


def test_select_prefers_failures_then_successes():
    def cand(stem, rows):
        return {"file": {"stem": stem, "name": stem + ".h5"}, "segments": {"rows": rows}}

    def row(i, text, ok):
        return {"level": 0, "start": float(i), "end": float(i) + 1, "text": text, "success": ok, "path": f"/s/{i}", "id": str(i)}

    cands = [
        cand("a", [row(0, "No action.", True), row(1, "Insert USB.", False), row(2, "Insert USB.", True)]),
        cand("b", [row(0, "Insert BNC.", False), row(1, "Pick BNC.", True), row(2, "Insert gear.", True)]),
    ]
    chosen = select_segments(cands, n=4)
    assert [c["success"] for c in chosen].count(False) == 2
    assert all(c["text"].startswith("Insert") for c in chosen)
    assert {c["file_stem"] for c in chosen} == {"a", "b"}
    more = select_segments(cands, n=5)
    assert len(more) == 5 and more[-1]["text"] == "Pick BNC."


def test_export_clips_from_synthetic(make_synth, tmp_path):
    cv2 = pytest.importorskip("cv2")
    info = make_synth("normal")
    if not info["real_video"]:
        pytest.skip("沒有 OpenCV 影片編碼器")
    raw = tmp_path  # make_synth 寫在 tmp_path
    res = inspect_file(info["path"], InspectOptions())
    write_file_outputs(res, tmp_path / "inspect")
    out = tmp_path / "clips"
    r = export_check_clips(tmp_path / "inspect", raw, out, n=2, pad=0.2, tile=(160, 120))
    assert r["failures"] == []
    assert len(r["results"]) == 2
    assert [x["label_success"] for x in r["results"]] == [False, True]  # 先失敗、再成功
    for x in r["results"]:
        cap = cv2.VideoCapture(str(out / x["clip"]))
        assert cap.isOpened()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        assert n == x["frames"] > 10
        assert w == 320  # 3 台相機 → 2 欄 x 160
        assert (out / x["sheet"]).exists()
        assert x["cameras"] == ["cam_left", "cam_right", "cam_wrist"]
    assert "人工核對清單" in (out / "checklist.md").read_text(encoding="utf-8")
    assert (out / "checklist.csv").exists()
    assert len(json.loads((out / "clips.json").read_text(encoding="utf-8"))["results"]) == 2


def test_select_and_export_from_index(make_synth, synth_mapping, tmp_path):
    """第 2 包：從索引抽選定零件的段，讀精簡版產生短片。"""
    pytest.importorskip("cv2")
    from assembly_monitor.check_clips import export_check_clips_from_index, select_from_index
    from assembly_monitor.index import index_h5, save_recording, write_index_outputs
    from assembly_monitor.slim import make_slim

    idx, slim = tmp_path / "index", tmp_path / "slim"
    for case, rid in (("normal", "rec_a"), ("dup_id", "rec_b")):
        info = make_synth(case, name=f"{rid}.h5")
        if not info["real_video"]:
            pytest.skip("沒有 OpenCV 影片編碼器")
        save_recording(index_h5(info["path"], synth_mapping, recording_id=rid), idx)
        make_slim(info["path"], slim / f"{rid}.h5", synth_mapping)
    write_index_outputs(idx, synth_mapping)

    segs = select_from_index(idx, "insert", "round peg 1", n=3, seed=1)
    # rec_b 的兩段 round peg 1 有 E_DUP_ID（驗證 error），不能拿來核對 → 只剩 rec_a 的 1 段
    assert [(s["file_stem"], s["id"]) for s in segs] == [("rec_a", "1")]
    with pytest.raises(ValueError):
        select_from_index(idx, "insert", "no such part")

    r = export_check_clips_from_index(idx, slim, tmp_path / "clips", synth_mapping, part="square peg 2",
                                      n=2, pad=0.2, tile=(160, 120))
    assert r["failures"] == [] and len(r["results"]) == 2
    assert all(x["label_success"] is False for x in r["results"])
    assert "square peg 2" in (tmp_path / "clips" / "checklist.md").read_text(encoding="utf-8")
