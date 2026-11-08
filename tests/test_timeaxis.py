"""時間軸統計（純 numpy，不需要 h5py）。"""

from __future__ import annotations

import numpy as np

from assembly_monitor.timeaxis import guess_time_unit, is_time_like_path, summarize_time_blocks


def test_guess_unit_epoch_scales():
    t = 1736420000.0
    assert guess_time_unit(t, t + 10)["unit"] == "s"
    assert guess_time_unit(t * 1e3, t * 1e3 + 1)["unit"] == "ms"
    assert guess_time_unit(t * 1e6, t * 1e6 + 1)["unit"] == "us"
    assert guess_time_unit(t * 1e9, t * 1e9 + 1)["unit"] == "ns"
    rel = guess_time_unit(0.0, 120.0, 1 / 30)
    assert rel["base"] == "relative?" and rel["confidence"] == "low"
    assert guess_time_unit(0.0, 5e12, 1e6)["unit"] == "unknown"


def test_summary_rate_monotonic_and_gaps():
    t = 1736420000.0 + np.arange(300) / 30.0
    t[150:] += 1.0  # 一個 1 秒缺口
    s = summarize_time_blocks([t], len(t), sampled=False)
    assert s["monotonic_non_decreasing"] is True
    assert abs(s["rate_hz"] - 30.0) < 0.01
    assert s["n_gaps_gt_3x_median"] == 1
    assert s["first_utc"].startswith("2025-01-09")


def test_summary_backward_steps_and_nan():
    t = np.array([1.0, 2.0, 1.5, 3.0, np.nan, 4.0])
    s = summarize_time_blocks([t], len(t), sampled=False)
    assert s["n_backward_steps"] == 1
    assert s["n_nonfinite"] == 1
    assert s["monotonic_non_decreasing"] is False


def test_summary_blocks_do_not_diff_across_boundaries():
    a = np.arange(0, 10, dtype=float)
    b = np.arange(100, 110, dtype=float)
    s = summarize_time_blocks([b, a], 1000, sampled=True)  # 第二段比第一段小，但不算倒退
    assert s["n_backward_steps"] == 0 and s["sampled"] is True and s["n"] == 1000


def test_time_like_path():
    assert is_time_like_path("/timestamps/hama1")
    assert is_time_like_path("/robot_state/time")
    assert not is_time_like_path("/segments_info/0/start")
    assert not is_time_like_path("/robot_state/joint_positions")
