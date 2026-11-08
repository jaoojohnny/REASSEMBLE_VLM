"""全量串流索引：本地 Range 伺服器上的 zip → 索引、精簡版、可續跑、壞檔隔離、沿用本機檔。"""

from __future__ import annotations

import csv
import shutil
import zipfile

import pytest

from assembly_monitor.stream import stream_index

NAMES = {"normal": "2025-01-01-00-00-00.h5", "dup_id": "2025-01-02-00-00-00.h5", "missing_view": "2025-01-03-00-00-00.h5"}


@pytest.fixture
def zip_site(make_synth, tmp_path):
    srv = tmp_path / "srv"
    srv.mkdir()
    built = {case: make_synth(case, name=name) for case, name in NAMES.items()}
    zp = srv / "data.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for case, name in NAMES.items():
            zf.write(built[case]["path"], f"data/{name}")
        zf.writestr("data/2025-01-04-00-00-00.h5", b"this is not an hdf5 file" * 100)
    return srv, built


def test_stream_index_end_to_end(zip_site, synth_mapping, range_server, local_session_factory, tmp_path):
    h5py = pytest.importorskip("h5py")
    srv, built = zip_site
    base, counter = range_server(srv)
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(built["missing_view"]["path"], raw / NAMES["missing_view"])  # 已有本機檔 → 不重新下載
    kw = dict(mapping=synth_mapping, index_dir=tmp_path / "index", zip_url=f"{base}/data.zip", raw_dir=raw,
              slim_dir=tmp_path / "slim", tmp_dir=raw / "_stream", workers=2, retries=0,
              session_factory=local_session_factory)
    res = stream_index(**kw)
    assert res["ok"] == 3 and len(res["failed"]) == 1
    assert res["failed"][0]["rid"] == "2025-01-04-00-00-00"
    rec = tmp_path / "index" / "recordings"
    assert sorted(p.name for p in rec.glob("*.json")) == [
        "2025-01-01-00-00-00.json", "2025-01-02-00-00-00.json", "2025-01-03-00-00-00.json", "2025-01-04-00-00-00.error.json"]
    assert not list((raw / "_stream").glob("*.h5*"))  # 暫存原檔已刪
    assert (raw / NAMES["missing_view"]).exists()  # 本機檔保留
    rows = list(csv.DictReader(open(tmp_path / "index" / "segments.csv", encoding="utf-8-sig")))
    assert len(rows) == 4 + 5 + 4
    slim = tmp_path / "slim" / "2025-01-01-00-00-00.h5"
    with h5py.File(str(slim), "r") as f:
        assert "cam_left" in f and "timestamps" in f and "segments_info" in f
        assert "events" not in f and "robot" not in f and "mic_wrist" not in f
        assert f["timestamps"]["cam_left"].shape == (90,)
    from pathlib import Path

    assert slim.stat().st_size < Path(built["normal"]["path"]).stat().st_size
    progress = (tmp_path / "index" / "stream_progress.json").read_text(encoding="utf-8")
    assert '"finished": true' in progress

    before = counter.get("requests", 0)
    res2 = stream_index(**kw)  # 續跑：只重試失敗的那一個
    assert res2["processed"] == 1 and res2["already_done"] == 3
    assert counter["requests"] - before < 15


def test_stream_limit_and_local_zip(zip_site, synth_mapping, tmp_path):
    srv, _ = zip_site
    res = stream_index(mapping=synth_mapping, index_dir=tmp_path / "idx", zip_path=str(srv / "data.zip"),
                       raw_dir=tmp_path / "raw", slim_dir=None, tmp_dir=tmp_path / "raw" / "_s", workers=1,
                       limit=2, retries=0)
    assert res["processed"] == 2 and res["ok"] == 2
    assert not (tmp_path / "slim").exists()
