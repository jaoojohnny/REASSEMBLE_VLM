"""第 13 包：REBOOT 16 mm 圓柱（開發集）——下載清單、索引、標註旗標、取格（不連網）。"""

import json
from pathlib import Path

from assembly_monitor import reboot_data as rd

ROOT = Path(__file__).resolve().parent.parent


def _ann(phase=5, ind=450, rec=540, taus=(0, 120, 150, 300, 420, 897), text="Gripper exerts excessive force"):
    return {"episode_index": "07", "duration_frames": 897, "phase_boundaries": {f"tau_{i}": t for i, t in enumerate(taus)},
            "failure": {"originating_phase": phase, "failure_mode": "excessive_force", "induced_at_frame": ind,
                        "recovery_started_at_frame": rec, "language_description": text}}


def test_download_plan():
    entries = [("README.md", 7), (".gitattributes", 2), ("meta/info.json", 6), ("meta/phase.json", 60),
               ("meta/episodes/chunk-000/file-000.parquet", 400), ("data/chunk-000/file-000.parquet", 10**9),
               ("videos/observation.images.cam_high/chunk-000/file-000.mp4", 100),
               ("videos/observation.images.cam_low/chunk-000/file-000.mp4", 100),
               ("videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4", 100),
               ("videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4", 100)]
    plan = dict(rd.download_plan(entries))
    assert "data/chunk-000/file-000.parquet" not in plan and ".gitattributes" not in plan
    assert "videos/observation.images.cam_low/chunk-000/file-000.mp4" not in plan
    assert "meta/phase.json" in plan and sum(plan.values()) == 7 + 6 + 60 + 400 + 300


def test_download_skips_existing(tmp_path):
    calls = []

    def dl(repo, f, dest):
        calls.append(f)
        p = Path(dest) / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 5)

    plan = {"recovery": [("meta/info.json", 5), ("README.md", 5)], "expert": [("meta/info.json", 5)]}
    st = rd.download(plan, tmp_path, dl)
    assert st["files"] == 3 and st["skipped"] == 0 and len(calls) == 3
    st2 = rd.download(plan, tmp_path, dl)
    assert st2["files"] == 0 and st2["skipped"] == 3 and len(calls) == 3


def test_episode_index():
    info = {"fps": 30, "video_path": rd.VIDEO_TMPL}
    rows = []
    for i in (1, 0):
        r = {"episode_index": i, "length": 897}
        for c in rd.CAMS:
            k = f"observation.images.{c}"
            r.update({f"videos/{k}/chunk_index": 0, f"videos/{k}/file_index": i,
                      f"videos/{k}/from_timestamp": 29.9 * i, f"videos/{k}/to_timestamp": 29.9 * (i + 1)})
        rows.append(r)
    eps = rd.episode_index(rows, info)
    assert [e["episode_index"] for e in eps] == [0, 1]
    v = eps[1]["videos"]["cam_left_wrist"]
    assert v["path"] == "videos/observation.images.cam_left_wrist/chunk-000/file-001.mp4" and v["from"] == 29.9


def test_failure_info_flags():
    ok = rd.failure_info(_ann())
    assert ok["flags"] == [] and ok["place"] and ok["phase_name"] == "Engage (place)"
    assert rd.failure_info(_ann(ind=450, rec=51))["flags"] == ["bad_order"]
    assert "lucky" in rd.failure_info(_ann(text="Misalignment with lucky success"))["flags"]
    f21 = rd.failure_info(_ann(phase=21))
    assert "bad_phase" in f21["flags"] and not f21["place"]
    assert rd.failure_info(_ann(phase=4, ind=360, taus=(0, 180, 330, 450, 780, 897)))["flags"] == ["phase_out"]
    assert "bad_bounds" in rd.failure_info(_ann(taus=(0, 180, 360, 540, 897, 800)))["flags"]


def test_points():
    fi = rd.failure_info(_ann())
    pts = rd.recovery_points(897, fi)
    frames = [f for f, _ in pts]
    assert frames == sorted(frames) and frames[-1] == 896 and len(pts) <= 8
    assert dict(pts)[450] == "induced" and dict(pts)[539] == "recovery-1" and 300 in dict(pts)
    bad = rd.failure_info(_ann(ind=900, rec=2000))
    assert all(0 <= f <= 896 for f, _ in rd.recovery_points(897, bad))
    ev = rd.even_points(897)
    assert [f for f, _ in ev][0] == 0 and ev[-1][0] == 896 and len(ev) == 8


def test_summarize_failures():
    rows = [{"failure": rd.failure_info(_ann())}, {"failure": rd.failure_info(_ann(rec=51))},
            {"failure": rd.failure_info(_ann(phase=2, ind=200, rec=300, taus=(0, 150, 330, 450, 780, 897)))},
            {"failure": None}]
    s = rd.summarize_failures(rows)
    assert s["n"] == 3 and s["place"] == 2 and s["place_clean"] == 1 and s["flags"] == {"bad_order": 1}


def test_frame_reader_synthetic(tmp_path):
    """用 PyAV 做一支 2 秒的測試影片（第 i 格亮度 = 4i），檢查依秒數讀格（往後讀、往回 seek）。沒有 PyAV 就略過。"""
    try:
        import av
        import numpy as np
    except ImportError:
        return
    p = tmp_path / "v.mp4"
    with av.open(str(p), "w") as c:
        s = c.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.codec_context.gop_size = 10
        for i in range(60):
            fr = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 4 * i, np.uint8), format="rgb24")
            for pk in s.encode(fr):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    r = rd.FrameReader(tmp_path)
    try:
        for i in (5, 6, 30, 59, 12, 0):  # 往後、跳遠、往回
            img, ft = r.read_at("v.mp4", i / 30, 30)
            assert img is not None and abs(ft - i / 30) < 0.5 / 30, (i, ft)
            assert abs(float(img.mean()) - 4 * i) < 6, (i, float(img.mean()))
    finally:
        r.close()


def test_make_sheet():
    import numpy as np

    pts = [(0, "a"), (10, "b")]
    grid = {("cam_high", 0): np.zeros((480, 640, 3), np.uint8), ("cam_high", 10): np.ones((480, 640, 3), np.uint8)}
    sheet = rd.make_sheet(["title", "sub"], pts, 30.0, grid)
    assert sheet.shape == (rd.TITLE_H + rd.HEAD_H + 3 * rd.TILE_H, 4 * rd.TILE_W, 3)


def test_bat_ascii_crlf():
    b = (ROOT / "run_reboot_cyl.bat").read_bytes()
    assert all(x < 128 for x in b)
    assert b.count(b"\n") == b.count(b"\r\n") > 10
    assert b"scripts.reboot_cyl" in b and b"test_p13_cyl.py" in b
