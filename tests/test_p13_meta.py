"""第 13 包第一步：REBOOT 說明檔檢查（不連網，用假資料）。"""

import json
from pathlib import Path

from assembly_monitor import reboot_meta as rm

ROOT = Path(__file__).resolve().parent.parent

STD_INFO = {
    "codebase_version": "v3.0", "fps": 30, "total_episodes": 2, "total_frames": 600, "total_tasks": 1,
    "features": {
        "action": {"dtype": "float32"}, "observation.state": {"dtype": "float32"},
        "observation.images.cam_high": {"dtype": "video", "info": {"video.codec": "av1"}},
        "observation.images.cam_high_depth": {"dtype": "depth"},
        "timestamp": {}, "frame_index": {}, "episode_index": {}, "index": {}, "task_index": {},
    },
}


def test_repo_list():
    repos = rm.repo_list()
    assert len(repos) == 10 and all(r.startswith("REBOOT26/") for _, _, r in repos)
    assert repos[-1] == ("sample", "sample", "REBOOT26/sample_recovery-demonstration")
    assert [r for k, kind, r in repos if k == "usb_a"] == ["REBOOT26/usbA-install", "REBOOT26/USB-A_recovery_install"]
    assert len(rm.repo_list(["rj45"], sample=False)) == 2


def test_classify_tree():
    big = 600 * 2**20
    c = rm.classify_tree([
        ("README.md", 7000), (".gitattributes", 2500), ("meta/info.json", 6400),
        ("meta/episodes/chunk-000/file-000.parquet", 494145), ("data/chunk-000/file-000.parquet", big),
        ("videos/observation.images.cam_high/chunk-000/file-000.mp4", 200),
        ("videos/observation.images.cam_high/chunk-000/file-001.mp4", 100),
        ("videos/chunk-000/observation.images.cam_low/episode_000000.mp4", 50),  # v2 的路徑
        ("annotations/failures.json", 3000), ("extra/huge.bin", big),
    ])
    assert "data/chunk-000/file-000.parquet" not in c["small"]
    assert not any(p.startswith("videos/") for p in c["small"])
    assert "annotations/failures.json" in c["small"] and c["big_skip"] == ["extra/huge.bin"]
    assert c["extra"] == ["annotations/failures.json", "extra/huge.bin"]
    assert c["video_keys"] == {"observation.images.cam_high": 300, "observation.images.cam_low": 50}
    assert c["groups"]["data"] == big


def test_keyword_hits():
    h = rm.keyword_hits("dataset_from_index\nt_f = 210, t_r = 420\nJamming at seating")
    assert h["counts"]["t_f"] == 1 and h["counts"]["t_r"] == 1 and h["counts"]["jam"] == 1
    assert len(h["lines"]) == 2  # dataset_from 不算 t_f


def test_info_findings():
    info = json.loads(json.dumps(STD_INFO))
    assert rm.info_findings(info)["nonstandard_features"] == []
    info["features"]["failure_onset"] = {"dtype": "int64"}
    f = rm.info_findings(info)
    assert f["nonstandard_features"] == ["failure_onset"]
    assert f["cameras"] == ["observation.images.cam_high"] and f["codecs"] == ["av1"]


def test_rows_findings_episodes():
    rows = [{"episode_index": i, "tasks": ["insert the usb"], "length": 300 + 30 * i, "stats/action/min": [0.0],
             "videos/observation.images.cam_high/from_timestamp": 0.0, "failure_mode": "jamming"} for i in range(3)]
    t = rm.rows_findings(rows, "episodes", "meta/episodes/x.parquet", fps=30)
    assert t["nonstandard_columns"] == ["failure_mode"]
    assert t["seconds"] == {"min": 10.0, "median": 11.0, "max": 12.0}
    assert t["keyword_hits"]["counts"]["jam"] == 1


def _fake_repo(files):
    def lister(repo):
        if repo.endswith("down"):
            raise ConnectionError("no network")
        return [(p, len(b)) for p, b in files[repo].items()]

    def downloader(repo, f, dest):
        p = Path(dest) / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(files[repo][f])
        return p

    return lister, downloader


def _std_files(extra=None):
    files = {"README.md": b"recovery install dataset", "meta/info.json": json.dumps(STD_INFO).encode(),
             "meta/episodes.jsonl": b'{"episode_index": 0, "tasks": ["t"], "length": 300}\n',
             "meta/tasks.jsonl": b'{"task_index": 0, "task": "t"}\n',
             "data/chunk-000/file-000.parquet": b"x" * 100,
             "videos/observation.images.cam_high/chunk-000/file-000.mp4": b"v" * 40}
    files.update(extra or {})
    return files


def test_fetch_analyze_verdict(tmp_path):
    files = {"A/expert": _std_files(), "A/recovery": _std_files(),
             "A/recovery2": _std_files({"annotations.json": b'[{"episode": 3, "t_f": 210, "t_r": 420}]'})}
    lister, downloader = _fake_repo(files)
    repos = [("usb_a", "expert", "A/expert"), ("usb_a", "recovery", "A/recovery"), ("usb_a", "recovery", "A/down")]
    res = [rm.analyze(r) for r in rm.fetch(repos, tmp_path, lister, downloader)]
    assert "error" in res[2] and res[0]["downloaded"] and "data/chunk-000/file-000.parquet" not in res[0]["downloaded"]
    assert res[1]["signals"] == [] and rm.verdict(res)["state"] == "none"
    res2 = [rm.analyze(r) for r in rm.fetch([("usb_a", "recovery", "A/recovery2")], tmp_path, lister, downloader)]
    assert rm.verdict(res2)["state"] == "found" and "annotations.json" in res2[0]["signals"][0]
    assert res2[0]["analysis"]["texts"][0]["keyword_hits"]["counts"]["t_f"] == 1
    assert rm.verdict([res[2]])["state"] == "error"


def test_unread_parquet_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "_pq", lambda: None)
    files = {"B/rec": _std_files({"meta/episodes/chunk-000/file-000.parquet": b"PAR1"})}
    lister, downloader = _fake_repo(files)
    res = [rm.analyze(r) for r in rm.fetch([("rj45", "recovery", "B/rec")], tmp_path, lister, downloader)]
    assert res[0]["unread"] == ["meta/episodes/chunk-000/file-000.parquet"]
    assert rm.verdict(res)["state"] == "unknown"


def test_write_report(tmp_path):
    files = {"A/expert": _std_files(), "A/rec": _std_files()}
    lister, downloader = _fake_repo(files)
    res = [rm.analyze(r) for r in rm.fetch([("usb_a", "expert", "A/expert"), ("usb_a", "recovery", "A/rec")],
                                            tmp_path / "raw", lister, downloader)]
    md = rm.write_report(res, {"pyav": "x", "decoders": [], "ok": False, "zh": "不行"}, tmp_path / "out")
    text = Path(md).read_text(encoding="utf-8")
    assert "## 結論" in text and "沒有：" in text and "下載量估算" in text
    s = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert s["verdict"]["state"] == "none" and s["size_plan"][0]["rgb_total"] == 80


def test_av1_check_keys():
    a = rm.av1_check()
    assert set(a) >= {"pyav", "decoders", "ok", "zh"}


def test_bat_ascii_crlf():
    b = (ROOT / "run_reboot_meta.bat").read_bytes()
    assert all(x < 128 for x in b)
    assert b.count(b"\n") == b.count(b"\r\n") > 10
    assert b"reboot_meta_check" in b and b"pyarrow" in b
