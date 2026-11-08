"""測試共用設定：未安裝套件時也能從 src/ 匯入；提供合成 HDF5 工廠。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
try:
    import assembly_monitor  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def make_synth(tmp_path):
    """make_synth(case, real_video=True) → build() 的摘要 dict；沒有 h5py 時整個測試略過。"""
    pytest.importorskip("h5py")
    from assembly_monitor.synthetic import build

    def _make(case: str = "normal", real_video: bool = True, name: str | None = None):
        return build(tmp_path / (name or f"{case}.h5"), case, real_video=real_video)

    return _make


@pytest.fixture
def synth_mapping(tmp_path):
    """合成檔用的已確認映射（鍵名與 synthetic.py 相同）。"""
    import yaml

    from assembly_monitor.mapping import load_mapping
    from assembly_monitor.synthetic import SYNTHETIC_MAPPING

    p = tmp_path / "field_mapping.yaml"
    p.write_text(yaml.safe_dump(SYNTHETIC_MAPPING, allow_unicode=True), encoding="utf-8")
    return load_mapping(p)


@pytest.fixture
def range_server():
    """serve(directory, support_range=True) → (url_base, counter)；測試結束自動關閉。"""
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path as _P

    servers = []

    def serve(directory, support_range=True):
        counter: dict = {}

        class H(SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                data = _P(self.translate_path(self.path)).read_bytes()
                counter["requests"] = counter.get("requests", 0) + 1
                rng = self.headers.get("Range")
                if support_range and rng and rng.startswith("bytes="):
                    a, b = rng[6:].split("-")
                    a = int(a)
                    b = min(int(b) if b else len(data) - 1, len(data) - 1)
                    chunk = data[a : b + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    counter["bytes"] = counter.get("bytes", 0) + len(chunk)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    try:
                        self.wfile.write(data)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(H, directory=str(directory)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}", counter

    yield serve
    for s in servers:
        s.shutdown()


@pytest.fixture
def local_session_factory():
    requests = pytest.importorskip("requests")

    def make():
        s = requests.Session()
        s.trust_env = False
        return s

    return make


@pytest.fixture
def ann_cfg(synth_mapping):
    """第 4 包標註設定（configs/annotation.yaml＋合成映射的相機）。"""
    from assembly_monitor.schema import load_annotation_config

    return load_annotation_config(ROOT / "configs" / "annotation.yaml", synth_mapping)


@pytest.fixture
def fake_index(tmp_path):
    """fake_index(recordings, part="BNC") → index 資料夾。

    recordings＝[(recording_id, [(start_rel, end_rel, success, [(skill, s_rel, e_rel, ok), ...]), ...]), ...]；
    每個錄製檔的時間原點錯開，並寫 recordings/<id>.json（相機時間範圍與來源雜湊）。
    """
    import csv as _csv
    import json as _json

    def build(recordings, part="BNC", action="insert", name="index"):
        idx = tmp_path / name
        (idx / "recordings").mkdir(parents=True, exist_ok=True)
        seg_cols = ["recording_id", "segment_id", "text", "action", "part", "start_unix", "end_unix", "success", "validation_status"]
        sk_cols = ["recording_id", "segment_id", "skill_id", "text", "start_unix", "end_unix", "success"]
        with open(idx / "segments.csv", "w", newline="", encoding="utf-8-sig") as fs, \
                open(idx / "skills.csv", "w", newline="", encoding="utf-8-sig") as fk:
            ws, wk = _csv.DictWriter(fs, fieldnames=seg_cols), _csv.DictWriter(fk, fieldnames=sk_cols)
            ws.writeheader()
            wk.writeheader()
            for r_i, (rid, segs) in enumerate(recordings):
                t0 = 1736420000.0 + r_i * 1000.0
                for s_i, seg in enumerate(segs):
                    start, end, ok = seg[0], seg[1], seg[2]
                    skills = seg[3] if len(seg) > 3 else []
                    ws.writerow({"recording_id": rid, "segment_id": s_i, "text": f"{action.capitalize()} {part}.", "action": action,
                                 "part": part, "start_unix": t0 + start, "end_unix": t0 + end, "success": ok, "validation_status": "ok"})
                    for k_i, (txt, a, b, kok) in enumerate(skills):
                        wk.writerow({"recording_id": rid, "segment_id": s_i, "skill_id": k_i, "text": txt,
                                     "start_unix": t0 + start + a, "end_unix": t0 + start + b, "success": kok})
                last = max((s[1] for s in segs), default=0.0)
                (idx / "recordings" / f"{rid}.json").write_text(_json.dumps({
                    "recording_id": rid, "source_sha256": f"sha-{rid}", "source_size_bytes": 1000 + r_i, "source_crc32": r_i,
                    "cameras": {"cam_left": {"t_first": t0 - 1.0, "t_last": t0 + last + 3.0}}}), encoding="utf-8")
        return idx

    return build
