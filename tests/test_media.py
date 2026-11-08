"""內嵌媒體辨識、檔案視窗與探測後端。"""

from __future__ import annotations

import io

import numpy as np
import pytest

from assembly_monitor.media import BlobRef, FileWindow, make_blob_ref, probe_blob, sniff_magic
from assembly_monitor.synthetic import FAKE_MP3, FAKE_MP4, make_tiny_mp4


def test_sniff_magic():
    assert sniff_magic(FAKE_MP4[:64]) == "mp4"
    assert sniff_magic(FAKE_MP3[:64]) == "mp3"
    assert sniff_magic(b"\x89PNG\r\n\x1a\n" + bytes(8)) == "png"
    assert sniff_magic(b"\xff\xd8\xff\xe0" + bytes(8)) == "jpeg"
    assert sniff_magic(b"RIFF\x00\x00\x00\x00WAVE") == "wav"
    assert sniff_magic(b"hello world, plain text") is None
    assert sniff_magic(b"") is None


def test_file_window_reads_only_its_range(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"HEADER" + b"0123456789" + b"TRAILER")
    with io.BufferedReader(FileWindow(str(p), 6, 10)) as fh:
        assert fh.read(4) == b"0123"
        fh.seek(-2, io.SEEK_END)
        assert fh.read() == b"89"
        fh.seek(0)
        assert fh.read() == b"0123456789"
        assert fh.read(5) == b""


def test_blob_ref_unavailable_has_no_head():
    b = BlobRef(length=10, access="unavailable", note="x")
    assert b.head() == b""
    with pytest.raises(OSError):
        b.open()


def test_probe_real_mp4_in_memory():
    data = make_tiny_mp4(n_frames=45, fps=15.0)
    if data is None:
        pytest.skip("沒有 OpenCV，無法產生測試影片")
    res = probe_blob(BlobRef(length=len(data), access="in-memory", data=data), "mp4")
    if not res["ok"]:
        pytest.skip(f"沒有可用的探測後端：{res['errors']}")
    s = res["summary"]
    assert s["frames"] == 45
    assert abs(s["fps_nominal"] - 15.0) < 0.5
    assert (s["width"], s["height"]) == (32, 24)


def test_probe_fake_mp4_fails_gracefully():
    res = probe_blob(BlobRef(length=len(FAKE_MP4), access="in-memory", data=FAKE_MP4), "mp4")
    assert res["ok"] is False
    assert res["errors"]


class _ArrayDataset:
    """模擬 h5py Dataset 的最小介面（shape / dtype / 切片）。"""

    def __init__(self, arr):
        self._a = arr
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.chunks = None
        self.compression = None

    def __getitem__(self, key):
        return self._a[key]


def test_int64_byte_array_is_detected_as_blob():
    """REASSEMBLE 的 *_audio：每個位元組存成 int64。"""
    data = np.frombuffer(FAKE_MP3, dtype=np.uint8).astype(np.int64)
    ref = make_blob_ref(_ArrayDataset(data), "unused.h5", 1 << 20)
    assert ref is not None and ref.access == "dataset-slices"
    assert sniff_magic(ref.head(16)) == "mp3"
    assert "int64" in ref.note


def test_int64_timestamps_are_not_blobs():
    ts = (1736420000 + np.arange(1000, dtype=np.int64)) * 10**9
    assert make_blob_ref(_ArrayDataset(ts), "unused.h5", 1 << 20) is None
    small = np.arange(200, dtype=np.int64) % 7  # 值在 0–255 但沒有媒體檔頭
    assert make_blob_ref(_ArrayDataset(small), "unused.h5", 1 << 20) is None
