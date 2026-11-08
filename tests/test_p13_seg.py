"""第 13 包第三步：找放開時刻、切段、只讀部分欄位（不連網）。"""

from pathlib import Path

import numpy as np

from assembly_monitor import reboot_data as rd
from assembly_monitor import reboot_seg as rs

ROOT = Path(__file__).resolve().parent.parent
FPS = 30.0


def _signal(open_v=0.04, closed_v=0.01, grasp=4.0, release=15.0, n=897, ramp=9, noise=0.0, regrasp=None):
    t = np.arange(n) / FPS
    g = np.full(n, open_v)
    g[(t >= grasp)] = closed_v
    g[(t >= release)] = open_v
    if regrasp:
        a, b = regrasp
        g[(t >= a) & (t < b)] = open_v
    k = np.ones(ramp) / ramp  # 夾爪不是瞬間動
    g = np.convolve(np.pad(g, (ramp, ramp), mode="edge"), k, mode="same")[ramp:-ramp]
    if noise:
        g = g + np.random.default_rng(0).normal(0, noise, n)
    return g


def test_detect_release_basic():
    r = rs.detect_release(_signal(), FPS)
    assert r["ok"] and r["flags"] == [] and r["final_open"]
    assert abs(r["onset"] / FPS - 15.0) < 0.25 and r["cross"] >= r["onset"]
    assert r["hold"][0] / FPS < 4.5


def test_detect_release_inverted_and_noise():
    r = rs.detect_release(_signal(open_v=0.0, closed_v=0.03, noise=0.0005), FPS)  # 方向相反也要找得到
    assert r["ok"] and abs(r["onset"] / FPS - 15.0) < 0.3


def test_detect_release_flags():
    assert rs.detect_release(np.full(897, 0.02), FPS)["why"] == "no_motion"
    assert rs.detect_release(np.zeros(20), FPS)["why"] == "too_short"
    r = rs.detect_release(_signal(regrasp=(8.0, 9.5)), FPS)
    assert r["ok"] and "multi_hold" in r["flags"] and abs(r["onset"] / FPS - 15.0) < 0.25
    late = rs.detect_release(_signal(release=29.7), FPS)
    assert not late["ok"] or "late" in late["flags"]


def test_initial_closed_state_is_not_a_hold():
    """REBOOT 的手臂在原位時夾爪閉合：閉合（原位）→ 張開 → 夾住圓柱（半開）→ 張開。不該算成夾了兩次。"""
    t = np.arange(897) / FPS
    g = np.where(t < 2.0, 0.0, np.where(t < 5.0, 1.0, np.where(t < 17.0, 0.4, 1.0)))
    r = rs.detect_release(g, FPS)
    assert r["ok"] and r["flags"] == [] and abs(r["onset"] / FPS - 17.0) < 0.2 and r["hold"][0] > 0


def test_detect_episode_picks_moving_arm():
    e = {"state_left": list(np.full(897, 0.02)), "state_right": list(_signal())}
    r = rs.detect_episode(e, FPS)
    assert r["arm"] == "right" and r["ok"]


def test_build_segments():
    rel = {0: {"ok": True, "flags": [], "onset": 450, "arm": "right"},
           1: {"ok": False, "why": "no_hold"},
           2: {"ok": True, "flags": ["multi_hold"], "onset": 450, "arm": "right"},
           3: {"ok": True, "flags": [], "onset": 60, "arm": "right"}}
    ann = lambda **k: rd.failure_info({"duration_frames": 897, "phase_boundaries": {f"tau_{i}": t for i, t in enumerate((0, 150, 180, 300, 420, 897))},
                                       "failure": {"originating_phase": k.get("phase", 5), "failure_mode": "sub_mm_misalignment",
                                                   "induced_at_frame": 450, "recovery_started_at_frame": k.get("rec", 540),
                                                   "language_description": k.get("text", "")}})
    fails = {56: ann(), 37: ann(rec=51), 10: ann(phase=2), 43: ann(text="lucky success")}
    segs, exc = rs.build_segments(rel, fails, FPS)
    ids = {s["id"]: s for s in segs}
    assert set(ids) == {"exp00", "rec56"}
    assert ids["exp00"]["start_frame"] == 360 and ids["exp00"]["end_frame"] == 450 and ids["exp00"]["label"] == "success"
    assert ids["rec56"]["end_frame"] == 540 and ids["rec56"]["start_frame"] == 450 and ids["rec56"]["label"] == "failure"
    why = {x["id"]: x["why"] for x in exc}
    assert why == {"exp01": "no_hold", "exp02": "multi_hold", "exp03": "release_before_3s",
                   "rec37": "bad_order", "rec43": "lucky"}


def test_counting_file_and_column_read(tmp_path):
    """只讀關節欄位：實際讀的位元組要遠小於整個檔（深度欄很大）。沒有 pyarrow 就略過。"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return
    n = 300
    rng = np.random.default_rng(0)
    t = pa.table({
        "observation.images.cam_high_depth": pa.array([rng.bytes(20000) for _ in range(n)], pa.binary()),
        "observation.state": pa.array([list(map(float, rng.random(14))) for _ in range(n)], pa.list_(pa.float32(), 14)),
        "action": pa.array([list(map(float, rng.random(14))) for _ in range(n)], pa.list_(pa.float32(), 14)),
        "timestamp": pa.array(np.arange(n) / 30.0, pa.float32()),
        "frame_index": pa.array(np.arange(n), pa.int64()),
        "episode_index": pa.array(np.zeros(n, int), pa.int64()),
    })
    p = tmp_path / "file-000.parquet"
    pq.write_table(t, p, row_group_size=100)
    size = p.stat().st_size
    with open(p, "rb") as f:
        tab, nread, need = rs.read_columns(f)
    assert tab.num_rows == n and set(tab.column_names) == set(rs.COLUMNS)
    assert nread < size / 10 and need < size / 10, (nread, need, size)
    with open(p, "rb") as f:
        try:
            rs.read_columns(f, rs.COLUMNS + ["observation.images.cam_high_depth"], cap=1 << 20)
            raise AssertionError("應該超過上限")
        except rs.TooBig:
            pass


def test_bat_ascii_crlf():
    b = (ROOT / "run_reboot_seg.bat").read_bytes()
    assert all(x < 128 for x in b)
    assert b.count(b"\n") == b.count(b"\r\n") > 10
    assert b"scripts.reboot_seg" in b and b"test_p13_seg.py" in b
