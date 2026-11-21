"""第 16 包：REBOOT M12——檔案篩選、取樣格、放開後 1 秒、標註檢查、計分、下載上限（不需要網路、影片、GPU）。"""

import json
from pathlib import Path

import numpy as np
import pytest

from assembly_monitor import reboot_m12 as rm

ROOT = Path(__file__).resolve().parent.parent
FPS = 30.0


def _grip_expert(n=450):
    g = np.zeros(n)
    g[30:90] = 1.0      # 原位閉合 → 張開
    g[90:300] = 0.2     # 夾住
    g[300:] = 1.0       # 最後放開（cross＝300）
    return g


def _grip():
    rng = np.random.default_rng(0)
    e = {"state_right": _grip_expert().tolist(), "state_left": (rng.normal(size=450) * 1e-5).tolist()}
    r = {"state_right": (rng.normal(size=900) * 1e-5).tolist(), "state_left": _grip_expert(900).tolist()}
    return ({"fps": FPS, "episodes": {"0": e}}, {"fps": FPS, "episodes": {"3": r}})


def test_file_filters(tmp_path):
    entries = [("README.md", 5), ("meta/info.json", 10), ("meta/episodes/chunk-000/file-000.parquet", 20),
               ("data/chunk-000/file-000.parquet", 10**9),
               ("videos/observation.images.cam_high/chunk-000/file-000.mp4", 100),
               ("videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4", 200),
               ("videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4", 300),
               ("videos/observation.images.cam_high_depth/chunk-000/file-000.mp4", 400)]
    assert [p for p, _ in rm.meta_files(entries)] == ["README.md", "meta/episodes/chunk-000/file-000.parquet",
                                                     "meta/info.json"]
    v = rm.video_files(entries, ("cam_high", "cam_right_wrist"))
    assert [p.split("/")[1] for p, _ in v] == ["observation.images.cam_high", "observation.images.cam_right_wrist"]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_bytes(b"x" * 10)
    assert ("meta/info.json", 10) not in rm.missing(rm.meta_files(entries), tmp_path)


def test_fetch_files_skips_existing(tmp_path):
    calls = []

    def dl(repo, f, dest):
        calls.append(f)
        p = Path(dest) / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"y" * 3)

    files = [("meta/a.json", 3), ("meta/b.json", 3)]
    st = rm.fetch_files("r", files, tmp_path, dl)
    assert st["files"] == 2 and not st["failed"]
    st = rm.fetch_files("r", files, tmp_path, dl)
    assert st["files"] == 0 and calls == ["meta/a.json", "meta/b.json"]


def test_sample_points():
    p = rm.sample_points(900, FPS)
    assert p[:3] == [0, 60, 120] and p[-2:] == [840, 899] and len(p) == 16
    assert rm.sample_points(61, FPS) == [0, 60]
    assert rm.sample_points(1, FPS) == [0]


def test_expert_frame():
    assert rm.expert_frame({"ok": True, "cross": 300}, 450, FPS) == 330
    assert rm.expert_frame({"ok": True, "cross": 440}, 450, FPS) == 449
    assert rm.expert_frame({"ok": False, "why": "no_hold"}, 450, FPS) is None


def test_analyse_and_cams():
    ge, gr = _grip()
    ex = rm.analyse(ge, "expert")
    assert ex["0"]["arm"] == "right" and ex["0"]["release"]["ok"] and ex["0"]["release"]["cross"] == 300
    assert ex["0"]["model_frame"] == 330 and ex["0"]["excluded"] is None
    rc = rm.analyse(gr, "recovery")
    assert rc["3"]["arm"] == "left" and rc["3"]["points"][0] == 0 and "release" not in rc["3"]
    assert rm.cams_needed({"expert": ex, "recovery": rc}) == ("cam_high", "cam_left_wrist", "cam_right_wrist")
    assert rm.cams_needed({"expert": ex}) == ("cam_high", "cam_right_wrist")


def test_sheets_shapes():
    pts = rm.sample_points(900, FPS)
    img = np.full((480, 640, 3), 120, np.uint8)
    frames = {("cam_high", f): img for f in pts}  # 手腕全缺 → MISSING 格
    sheet = rm.recovery_sheet("t", pts, FPS, frames, ("cam_high", "cam_right_wrist"),
                              {"left": [0.0] * 900, "right": _grip_expert(900).tolist()}, "right")
    rows = 2
    assert sheet.shape == (rm.TITLE_H + rm.STRIP_H + rows * (rm.LABEL_H + 2 * rm.CELL_H), rm.COLS * rm.CELL_W, 3)
    g = rm.grid_sheet("g", [("a", img), ("b", None)] * 6, cols=10, box=(0.2, 0.2, 0.8, 0.8))
    assert g.shape[1] == 10 * 192 and g.shape[0] == rm.TITLE_H + 2 * (144 + rm.LABEL_H)


def _prep():
    return {"kinds": {"recovery": {"episodes": {"0": {"points": [0, 60, 120]}, "1": {"points": [0, 60]},
                                                "2": {"points": [0, 60]}}}}}


def _write(p, **kw):
    a = {"complete": True, "blind_to_model": True,
         "episodes": {"0": {"usable": True, "frame": 60, "type": "misalignment"},
                      "1": {"usable": False, "note": "看不出失敗"}, "2": {"usable": True, "frame": 0}}}
    a.update(kw)
    p.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")


def test_load_annotations(tmp_path):
    p = tmp_path / "a.json"
    with pytest.raises(rm.Stop):
        rm.load_annotations(p, _prep())                      # 沒有檔案
    _write(p)
    got = rm.load_annotations(p, _prep())
    assert sorted(got) == [0, 2] and got[0]["frame"] == 60
    for bad in ({"complete": False}, {"blind_to_model": False},
                {"episodes": {"0": {"usable": True, "frame": 60}}},                         # 少段
                {"episodes": {"0": {"usable": True, "frame": 61}, "1": {"usable": False, "note": "x"},
                              "2": {"usable": True, "frame": 0}}},                          # 不是取樣格
                {"episodes": {"0": {"usable": True, "frame": 60}, "1": {"usable": False},
                              "2": {"usable": True, "frame": 0}}}):                         # 不能用但沒寫原因
        _write(p, **bad)
        with pytest.raises(rm.Stop):
            rm.load_annotations(p, _prep())


def test_auc_and_score():
    assert rm.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert rm.auc([0.1], [0.9]) == 0.0
    assert rm.auc([0.5], [0.5]) == 0.5
    assert rm.auc([], [0.1]) is None
    rows = [{"label": 1, "p_failure": 0.7}, {"label": 1, "p_failure": 0.4},
            {"label": 0, "p_failure": 0.3}, {"label": 0, "p_failure": 0.6}]
    s = rm.score(rows)
    assert (s["ok_failure"], s["n_failure"], s["ok_success"], s["n_success"]) == (1, 2, 1, 2)
    assert s["bal"] == pytest.approx(0.5) and s["auc"] == pytest.approx(0.75)


def test_prepare_stops_before_videos_when_too_big(tmp_path, monkeypatch):
    ge, gr = _grip()
    calls = []

    def lister(repo):
        return [("meta/info.json", 3), ("videos/observation.images.cam_high/chunk-000/file-000.mp4", 4 * 10**9),
                ("videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4", 4 * 10**9)]

    def dl(repo, f, dest):
        calls.append(f)
        p = Path(dest) / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"z" * 3)

    monkeypatch.setattr(rm.rs, "fetch_gripper", lambda repo, local, opener=None: ge if "install" in repo and "recovery" not in repo else gr)
    with pytest.raises(rm.Stop):
        rm.prepare(dest_root=tmp_path / "raw", out=tmp_path / "out", max_gb=5.0, lister=lister, downloader=dl,
                   opener=None)
    assert calls and all(c.startswith("meta/") for c in calls)   # 只下載了說明檔


def test_reports(tmp_path):
    prep = {"generated": "g", "seconds": 1.0, "fps": FPS, "cams": ["cam_high", "cam_right_wrist"],
            "download": {"total_gb": 1.2},
            "kinds": {"expert": {"repo": rm.REPOS["expert"], "n": 2, "episodes": {
                "0": {"arm": "right", "length": 450, "release": {"flags": ["multi_hold"]}, "model_frame": 330},
                "1": {"arm": "right", "length": 400, "release": {"flags": []}, "model_frame": None}}},
                "recovery": {"repo": rm.REPOS["recovery"], "n": 1, "episodes": {"3": {"arm": "right", "length": 900}}}}}
    rm.write_prepare_report(prep, tmp_path / "p.md")
    t = (tmp_path / "p.md").read_text(encoding="utf-8")
    assert "找到最後一次放開 1 段" in t and "multi_hold" in t
    res = {"generated": "g", "model": rm.MODEL, "model_part": "bolt 4", "model_cv": {"bal_mean": 0.935},
           "score": rm.score([{"label": 1, "p_failure": 0.7}, {"label": 0, "p_failure": 0.3}]),
           "rows": [{"kind": "recovery", "episode": 3, "label": 1, "t": 14.0, "p_failure": 0.7, "pred": "失敗",
                     "type": "misalignment"},
                    {"kind": "expert", "episode": 0, "label": 0, "t": 11.0, "p_failure": 0.3, "pred": "成功"}],
           "excluded": [{"kind": "expert", "episode": 1, "why": "找不到放開"}]}
    rm.write_apply_report(res, tmp_path / "r.md")
    t = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "1/1" in t and "100%" in t and "94%" in t and "找不到放開" in t and "misalignment" in t


@pytest.mark.parametrize("name", ["run_m12_prepare.bat", "run_m12_apply.bat"])
def test_bat_ascii_crlf(name):
    p = ROOT / name
    if not p.exists():
        pytest.skip("沒有 .bat")
    b = p.read_bytes()
    assert all(c < 128 for c in b) and b"\r\n" in b and b.count(b"\n") == b.count(b"\r\n")
