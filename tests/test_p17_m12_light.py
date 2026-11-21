"""第 17 包：OpenCV 調光線——統計、整批／每張調整、截斷、讀第 16 包、計分流程（假的編碼器，不需要 GPU）。"""

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from assembly_monitor import m12_light as ml


def _img(level, noise=10, seed=0, h=120, w=160):
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(level, noise, size=(h, w, 3)), 0, 255).astype(np.uint8)
    return a


def test_crop_is_center_60_percent():
    im = np.zeros((100, 200, 3), np.uint8)
    assert ml.crop(im).shape == (60, 120, 3)


def test_lab_stats_pooled():
    a = np.zeros((2, 2, 3))
    b = np.full((2, 2, 3), 10.0)
    s = ml.lab_stats([a, b])
    assert s["L_mean"] == pytest.approx(5.0) and s["L_std"] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        ml.lab_stats([])


def test_adjust_matches_target_stats():
    im = _img(90, 20)
    src = ml.crop_stats(im)
    tgt = {"L_mean": 150.0, "L_std": 30.0, "a_mean": 126.0, "b_mean": 131.0}
    out = ml.adjust(im, ml.lab_stats([ml.to_lab(im)]), tgt)
    st = ml.lab_stats([ml.to_lab(out)])
    assert st["L_mean"] == pytest.approx(150, abs=2.5)
    assert st["L_std"] == pytest.approx(30, abs=2.5)
    assert out.shape == im.shape and out.dtype == np.uint8
    assert src["L_mean"] < 120


def test_adjust_clips_instead_of_wrapping():
    im = np.full((20, 20, 3), 250, np.uint8)
    out = ml.adjust(im, {"L_mean": 100.0, "L_std": 1.0, "a_mean": 128.0, "b_mean": 128.0},
                    {"L_mean": 200.0, "L_std": 50.0, "a_mean": 128.0, "b_mean": 128.0})
    assert out.min() >= 200          # 很亮的像素被截在最亮，不會繞回變暗


def test_l1_keeps_order_l2_flattens():
    dark, light = _img(80, 5, 1), _img(130, 5, 2)
    tgt = {"L_mean": 150.0, "L_std": 30.0, "a_mean": 128.0, "b_mean": 128.0}
    r = ml.adjust_all([dark, light], tgt)
    m1 = [ml.crop_stats(x)["L_mean"] for x in r["L1"]]
    m2 = [ml.crop_stats(x)["L_mean"] for x in r["L2"]]
    assert m1[0] < m1[1] - 20                       # 整批：原本較暗的還是較暗
    assert abs(m2[0] - m2[1]) < 3                   # 每張：都拉到同一個亮度
    assert np.mean(m1) == pytest.approx(150, abs=4)


def test_path_of_windows_separators():
    assert ml.path_of(r"data\derived\x.jpg") == Path("data/derived/x.jpg")


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fr = Path("data/derived/frames/reboot_m12")
    rows = []
    for k, (kind, lab, lev) in enumerate([("expert", 0, 110), ("expert", 0, 120), ("recovery", 1, 125)]):
        p = fr / kind / f"ep{k:03d}_f10.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), _img(lev, 8, k))
        rows.append({"kind": kind, "episode": k, "label": lab, "image": str(p).replace("/", "\\"), "p_failure": 0.0})
    Path("experiments/p16_m12").mkdir(parents=True)
    ids = ["r1__seg1", "r2__seg2"]
    for i, sid in enumerate(ids):
        d = ml.B4_FRAMES / sid
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "hand__00__t+001.000s__f000010.jpg"), _img(155, 15, 10 + i))
    Path("models/perpart_final").mkdir(parents=True)
    Path("models/perpart_final/bolt_4.json").write_text(json.dumps({"segment_ids": ids}), encoding="utf-8")
    return rows


class FakeEnc:
    def encode(self, paths):
        return np.array([[cv2.imread(p).mean()] for p in paths])

    def close(self):
        pass


def test_run_end_to_end_with_fake_model(tmp_path, monkeypatch):
    rows = _setup(tmp_path, monkeypatch)
    enc = FakeEnc()
    for r in rows:   # 第 16 包的答案＝同一個假模型在原始影格上的值
        r["p_failure"] = float(1 / (1 + np.exp(-(enc.encode([r["image"].replace("\\", "/")])[0, 0] - 120) / 10)))
    Path("experiments/p16_m12/apply.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")

    import types
    import sys

    fake_pf = types.SimpleNamespace(
        load_model=lambda stem: {"stem": stem},
        image_vector=lambda e, p: e.encode([p])[0],
        predict_proba=lambda m, X: 1 / (1 + np.exp(-(np.asarray(X)[:, 0] - 120) / 10)))

    def score(vr):
        y = np.array([r["label"] for r in vr])
        p = np.array([r["p_failure"] for r in vr])
        return {"n_failure": int(y.sum()), "n_success": int((1 - y).sum()), "ok_failure": int(((p > .5) & (y == 1)).sum()),
                "ok_success": int(((p <= .5) & (y == 0)).sum()), "bal": 0.5, "auc": 0.5,
                "median_p_failure": {"失敗段": float(np.median(p[y == 1])), "成功段": float(np.median(p[y == 0]))}}

    monkeypatch.setitem(sys.modules, "assembly_monitor.perpart_final", fake_pf)
    monkeypatch.setitem(sys.modules, "assembly_monitor.perpart", types.SimpleNamespace(make_encoders=lambda: {"crop": enc}))
    monkeypatch.setitem(sys.modules, "assembly_monitor.reboot_m12", types.SimpleNamespace(score=score))
    import assembly_monitor
    monkeypatch.setattr(assembly_monitor, "perpart_final", fake_pf, raising=False)
    monkeypatch.setattr(assembly_monitor, "perpart", sys.modules["assembly_monitor.perpart"], raising=False)
    monkeypatch.setattr(assembly_monitor, "reboot_m12", sys.modules["assembly_monitor.reboot_m12"], raising=False)

    res = ml.run(encoder=enc)
    assert res["l0_same_as_p16"]
    assert set(res["variants"]) == {"L0", "L1", "L2"}
    assert (ml.FRAMES / "L1" / "expert" / "ep000_f10.jpg").exists()
    assert (ml.FRAMES / "L2" / "recovery" / "ep002_f10.jpg").exists()
    # 調亮之後，假模型（越亮越像失敗）給的失敗機率變高
    p0 = [r["p_failure"] for r in res["variants"]["L0"]["rows"]]
    p1 = [r["p_failure"] for r in res["variants"]["L1"]["rows"]]
    assert np.mean(p1) > np.mean(p0)
    rep = (ml.OUT / "report.md").read_text(encoding="utf-8")
    assert "L1 整批調整（主要）" in rep and "一樣（流程沒問題）" in rep
    assert (ml.OUT / "before_after.jpg").exists()


def test_missing_p16_stops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ml.Stop):
        ml.load_rows()
