"""最小樣本取得：本機 zip 與支援 / 不支援 Range 的本地 HTTP 伺服器。"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

from assembly_monitor.fetch import (  # noqa: E402
    FetchCancelled,
    HttpRangeReader,
    RangeNotSupported,
    fetch_samples,
    select_members,
)

NAMES = [f"data/2025-01-{d:02d}-10-00-00.h5" for d in (9, 10, 11, 12, 13)]


def _local_session():
    """不讀系統 / 環境的 proxy 設定，確保連到本機測試伺服器。"""
    s = requests.Session()
    s.trust_env = False
    return s


@pytest.fixture
def sample_zip(tmp_path):
    """5 個假 .h5（大小不同）+ 1 個 json；混用 stored / deflated，並強制一個 zip64。"""
    contents = {}
    zp = tmp_path / "srv" / "data.zip"
    zp.parent.mkdir()
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("data/", b"")
        for k, name in enumerate(NAMES):
            data = os.urandom(20_000 + 15_000 * k)
            contents[name] = data
            method = zipfile.ZIP_STORED if k % 2 == 0 else zipfile.ZIP_DEFLATED
            if k == 3:
                with zf.open(zipfile.ZipInfo(name), "w", force_zip64=True) as w:
                    w.write(data)
            else:
                zf.writestr(name, data, compress_type=method)
        zf.writestr("poses/2025-01-09-10-00-00_poses.json", b"{}")
    return zp, contents


class _RangeHandler(SimpleHTTPRequestHandler):
    support_range = True
    counter: dict = {}

    def log_message(self, *args):  # 安靜
        pass

    def do_GET(self):  # noqa: N802
        path = Path(self.translate_path(self.path))
        data = path.read_bytes()
        self.counter["requests"] = self.counter.get("requests", 0) + 1
        rng = self.headers.get("Range")
        if self.support_range and rng and rng.startswith("bytes="):
            a, b = rng[6:].split("-")
            a = int(a)
            b = int(b) if b else len(data) - 1
            b = min(b, len(data) - 1)
            chunk = data[a : b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            self.counter["bytes"] = self.counter.get("bytes", 0) + len(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass


def _serve(directory: Path, support_range: bool):
    handler_cls = type("H", (_RangeHandler,), {"support_range": support_range, "counter": {}})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler_cls, directory=str(directory)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, handler_cls.counter


@pytest.fixture
def range_server(sample_zip):
    zp, _ = sample_zip
    srv, counter = _serve(zp.parent, True)
    yield f"http://127.0.0.1:{srv.server_address[1]}/data.zip", counter
    srv.shutdown()


@pytest.fixture
def plain_server(sample_zip):
    zp, _ = sample_zip
    srv, counter = _serve(zp.parent, False)
    yield f"http://127.0.0.1:{srv.server_address[1]}/data.zip", counter
    srv.shutdown()


def _infos(zp):
    with zipfile.ZipFile(zp) as zf:
        return [i for i in zf.infolist() if i.filename.endswith(".h5")]


def test_select_strategies_are_deterministic(sample_zip):
    zp, _ = sample_zip
    infos = _infos(zp)
    spread = [i.filename for i in select_members(infos, 3, "spread")]
    assert spread == [NAMES[0], NAMES[2], NAMES[4]]
    smallest = [i.filename for i in select_members(infos, 2, "smallest")]
    assert smallest == [NAMES[0], NAMES[1]]
    r1 = [i.filename for i in select_members(infos, 2, "random", seed=7)]
    r2 = [i.filename for i in select_members(infos, 2, "random", seed=7)]
    assert r1 == r2
    excl = [i.filename for i in select_members(infos, 3, "spread", exclude_names=["2025-01-09-10-00-00.h5"])]
    assert NAMES[0] not in excl


def test_http_range_reader_matches_file(sample_zip, range_server):
    zp, _ = sample_zip
    url, _ = range_server
    raw = zp.read_bytes()
    r = HttpRangeReader(url, session=_local_session())
    assert r.size == len(raw)
    r.seek(100)
    buf = bytearray(50)
    n = r.readinto(buf)
    assert bytes(buf[:n]) == raw[100 : 100 + n]
    r.seek(-10, 2)
    assert r.read() == raw[-10:]


def test_remote_fetch_downloads_only_selected(tmp_path, sample_zip, range_server):
    zp, contents = sample_zip
    url, counter = range_server
    out = tmp_path / "raw"
    res = fetch_samples(zip_url=url, zip_path=None, out_dir=out, index_dir=tmp_path / "index",
                        n=2, strategy="smallest", assume_yes=True, session=_local_session())
    assert res["status"] == "ok"
    got = sorted(p.name for p in out.glob("*.h5"))
    assert got == sorted(Path(n).name for n in NAMES[:2])
    for name in NAMES[:2]:
        assert (out / Path(name).name).read_bytes() == contents[name]
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entry = man["files"][Path(NAMES[0]).name]
    assert entry["sha256"] == hashlib.sha256(contents[NAMES[0]]).hexdigest()
    assert counter["bytes"] < zp.stat().st_size  # 沒有整包下載
    assert (tmp_path / "index" / "zip_listing.csv").exists()

    before = counter["requests"]
    res2 = fetch_samples(zip_url=url, zip_path=None, out_dir=out, index_dir=tmp_path / "index",
                         n=2, strategy="smallest", assume_yes=True, session=_local_session())
    assert len(res2["skipped"]) == 2 and res2["downloaded"] == []
    assert counter["requests"] - before < 10  # 只讀目錄，不重抓檔案


def test_zip64_member_and_local_zip(tmp_path, sample_zip):
    zp, contents = sample_zip
    res = fetch_samples(zip_url=None, zip_path=str(zp), out_dir=tmp_path / "raw", index_dir=tmp_path / "idx",
                        members=["2025-01-12-10-00-00.h5"], assume_yes=True)
    assert res["status"] == "ok"
    assert (tmp_path / "raw" / "2025-01-12-10-00-00.h5").read_bytes() == contents[NAMES[3]]


def test_server_without_range_is_refused(tmp_path, plain_server):
    url, _ = plain_server
    with pytest.raises(RangeNotSupported):
        fetch_samples(zip_url=url, zip_path=None, out_dir=tmp_path / "raw", index_dir=tmp_path / "idx",
                      n=1, assume_yes=True, session=_local_session())
    assert not (tmp_path / "raw").exists() or not list((tmp_path / "raw").glob("*.h5"))


def test_cancel_and_size_limit(tmp_path, sample_zip):
    zp, _ = sample_zip
    with pytest.raises(FetchCancelled):
        fetch_samples(zip_url=None, zip_path=str(zp), out_dir=tmp_path / "raw", index_dir=tmp_path / "idx",
                      n=1, confirm=lambda msg: False)
    with pytest.raises(ValueError):
        fetch_samples(zip_url=None, zip_path=str(zp), out_dir=tmp_path / "raw", index_dir=tmp_path / "idx",
                      n=3, max_total_bytes=1000, assume_yes=True)
    assert not list((tmp_path / "raw").glob("*.h5")) if (tmp_path / "raw").exists() else True


def test_disk_space_guard(tmp_path, sample_zip, monkeypatch):
    zp, _ = sample_zip
    import assembly_monitor.fetch as fetch_mod

    monkeypatch.setattr(fetch_mod, "disk_free_bytes", lambda path: 10)
    with pytest.raises(ValueError, match="磁碟"):
        fetch_samples(zip_url=None, zip_path=str(zp), out_dir=tmp_path / "raw", index_dir=tmp_path / "idx",
                      n=1, assume_yes=True)


def test_size_cap_uses_download_size(tmp_path):
    """上限比的是壓縮後的下載量，不是解壓後大小。"""
    zp = tmp_path / "z.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data/2025-01-01-00-00-00.h5", b"\0" * 2_000_000)  # 壓縮後只剩幾 KB
    res = fetch_samples(zip_url=None, zip_path=str(zp), out_dir=tmp_path / "raw", index_dir=tmp_path / "idx",
                        n=1, assume_yes=True, max_total_bytes=500_000)
    assert res["status"] == "ok"
    assert (tmp_path / "raw" / "2025-01-01-00-00-00.h5").stat().st_size == 2_000_000
