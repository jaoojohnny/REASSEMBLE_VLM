"""合成 HDF5（只供測試與示範）：模仿「內嵌 mp4 + 時間戳 + 分段標註」的常見結構。

刻意不用 REASSEMBLE 的真實 key 名稱，確保檢查工具沒有把欄位寫死。
規格要求的四種情境：normal、missing_view、bad_time、dup_id；另加 renamed（所有名稱都換掉）。

    python -m assembly_monitor.synthetic --out data/derived/synthetic
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

CASES = ("normal", "missing_view", "bad_time", "dup_id", "renamed")
T0 = 1736420000.0  # 2025-01-09（Unix 秒）
N_FRAMES = 90
FPS = 30.0
FAKE_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + bytes(512)
FAKE_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(2048)

# 高層 (相對秒 start, end, success, text)；低層每段 3 個子動作
SEGMENTS = [
    (0.10, 0.70, True, "Pick round peg 1"),
    (0.70, 1.40, True, "Insert round peg 1"),
    (1.40, 2.10, True, "Pick square peg 2"),
    (2.10, 2.90, False, "Insert square peg 2"),
]
LOW = ["Approach", "Align", "Push"]

# 與 build() 產生的鍵名對應的「已確認」欄位映射（測試與示範用）
SYNTHETIC_MAPPING = {
    "version": 1,
    "status": "confirmed",
    "confirmed_at": "synthetic",
    "segments": {
        "container": "/segments_info",
        "id": "group_name",
        "fields": {"start": "start", "end": "end", "success": "success", "text": "text"},
        "exclude_text": ["Pick square peg 2"],
        "low_level": {"container": "Low_level",
                      "fields": {"start": "start", "end": "end", "success": "success", "text": "text"}},
    },
    "action_parse": {"regex": r"^\s*(?P<action>[A-Za-z]+)\s+(?P<part>.+?)\.?\s*$",
                     "action_lowercase": True, "target_action": "insert"},
    "cameras": {
        "cam_left": {"video": "/cam_left", "timestamps": "/timestamps/cam_left", "role": "external"},
        "cam_right": {"video": "/cam_right", "timestamps": "/timestamps/cam_right", "role": "external"},
        "cam_wrist": {"video": "/cam_wrist", "timestamps": "/timestamps/cam_wrist", "role": "wrist"},
    },
    "best_camera_candidates": ["cam_left", "cam_right", "cam_wrist"],
    "extraction": {"start_buffer_sec": 0.5, "end_buffer_sec": 2.0},
}


def make_tiny_mp4(n_frames: int = N_FRAMES, width: int = 32, height: int = 24, fps: float = FPS) -> bytes | None:
    """用 OpenCV 產生極小的真 mp4（MPEG-4 Part 2）；沒有 OpenCV 時回傳 None。"""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    fd, path = tempfile.mkstemp(prefix="am_synth_", suffix=".mp4")
    os.close(fd)
    try:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not vw.isOpened():
            return None
        for i in range(n_frames):
            frame = np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
            frame[:, : (i % width) + 1, 1] = 255
            vw.write(frame)
        vw.release()
        data = Path(path).read_bytes()
        return data if len(data) > 12 and data[4:8] == b"ftyp" else None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def build(path: str | Path, case: str = "normal", real_video: bool = True, seed: int = 0) -> dict[str, Any]:
    """寫出一個合成檔，回傳寫入內容摘要（測試用來比對）。"""
    import h5py

    if case not in CASES:
        raise ValueError(f"未知情境 {case}；可用 {CASES}")
    rng = np.random.default_rng(seed)
    renamed = case == "renamed"
    names = {
        "videos": ["view_a", "view_b", "view_c"] if renamed else ["cam_left", "cam_right", "cam_wrist"],
        "audio": "mic_c" if renamed else "mic_wrist",
        "time": "clock" if renamed else "timestamps",
        "robot": "state" if renamed else "robot",
        "seg": "annotations" if renamed else "segments_info",
        "low": "sub" if renamed else "Low_level",
        "start": "begin" if renamed else "start",
        "end": "stop" if renamed else "end",
        "success": "ok" if renamed else "success",
        "text": "description" if renamed else "text",
    }
    mp4 = make_tiny_mp4() if real_video else None
    video_bytes = mp4 if mp4 is not None else FAKE_MP4
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written_videos = []
    with h5py.File(os.fspath(p), "w") as f:
        f.attrs["note"] = "synthetic test file (not REASSEMBLE data)"
        f.attrs["schema"] = "synthetic-v1"
        for i, v in enumerate(names["videos"]):
            if case == "missing_view" and i == 1:
                continue
            f.create_dataset(v, data=np.void(video_bytes))
            written_videos.append(v)
        f.create_dataset(names["audio"], data=np.frombuffer(FAKE_MP3, dtype=np.uint8))
        f.create_dataset("events", data=rng.integers(0, 300, size=(4000, 3)).astype(np.int16), chunks=True, compression="gzip")
        g_robot = f.create_group(names["robot"])
        g_robot.create_dataset("joint_pos", data=rng.normal(size=(3000, 7)), chunks=(500, 7), compression="gzip", compression_opts=4)
        g_robot.create_dataset("force", data=rng.normal(size=(3000, 3)).astype(np.float32))
        g_t = f.create_group(names["time"])
        for v in names["videos"]:
            ts = T0 + np.arange(N_FRAMES) / FPS + rng.uniform(0, 0.002, N_FRAMES)
            if case == "bad_time" and v == names["videos"][0]:
                ts[40], ts[41] = ts[41] + 0.05, ts[40]
            g_t.create_dataset(v, data=ts)
        g_t.create_dataset(names["audio"], data=T0 + np.arange(150) / 50.0)
        g_t.create_dataset("joint_pos", data=T0 + np.arange(3000) / 1000.0)
        g_t.create_dataset("force", data=T0 + np.arange(3000) / 1000.0)
        g_t.create_dataset("events", data=np.sort(T0 + rng.uniform(0, 3.0, 4000)))
        g_s = f.create_group(names["seg"])
        n_low = 0
        for k, (s, e, ok, text) in enumerate(SEGMENTS):
            start, end = T0 + s, T0 + e
            if case == "bad_time" and k == 1:
                start, end = end, start
            if case == "bad_time" and k == 2:
                end = end + 100.0
            g = g_s.create_group(str(k))
            g.create_dataset(names["start"], data=start)
            g.create_dataset(names["end"], data=end)
            g.create_dataset(names["success"], data=np.bool_(ok))
            g.create_dataset(names["text"], data=text)
            low = g.create_group(names["low"])
            lo, hi = min(start, end), max(start, end)
            step = (hi - lo) / len(LOW)
            for j, skill in enumerate(LOW):
                c = low.create_group(str(j))
                ls, le = lo + j * step, lo + (j + 1) * step
                if case == "bad_time" and k == 0 and j == 2:
                    le = hi + 0.5
                c.create_dataset(names["start"], data=ls)
                c.create_dataset(names["end"], data=le)
                c.create_dataset(names["success"], data=np.bool_(ok or j < 2))
                c.create_dataset(names["text"], data=skill)
                n_low += 1
        n_top = len(SEGMENTS)
        if case == "dup_id":
            s, e, ok, text = SEGMENTS[1]
            g = g_s.create_group("01")
            g.create_dataset(names["start"], data=T0 + s)
            g.create_dataset(names["end"], data=T0 + e)
            g.create_dataset(names["success"], data=np.bool_(ok))
            g.create_dataset(names["text"], data=text)
            n_top += 1
        if renamed:
            f["latest"] = h5py.SoftLink(f"/{names['seg']}/0")
    return {
        "path": str(p),
        "case": case,
        "videos": written_videos,
        "all_video_names": names["videos"],
        "names": names,
        "real_video": mp4 is not None,
        "n_frames": N_FRAMES,
        "n_top": n_top,
        "n_low": n_low,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="產生合成 HDF5（測試 / 示範用，不是真實資料）")
    ap.add_argument("--out", default="data/derived/synthetic", help="輸出資料夾")
    ap.add_argument("--case", default="all", choices=("all",) + CASES)
    ap.add_argument("--fake-video", action="store_true", help="不用 OpenCV，只放假的 mp4 檔頭")
    args = ap.parse_args(argv)
    cases = CASES if args.case == "all" else (args.case,)
    for c in cases:
        info = build(Path(args.out) / f"synthetic_{c}.h5", c, real_video=not args.fake_video)
        print(f"{info['path']}  videos={info['videos']}  real_video={info['real_video']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
