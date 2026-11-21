"""每零件分類器的最終模型：訓練、numpy 預測和 sklearn 相同、存檔讀回、保留零件不碰（不需要影格、GPU）。"""

from pathlib import Path

import numpy as np
import pytest

from assembly_monitor import perpart as pp
from assembly_monitor import perpart_final as pf

ROOT = Path(__file__).resolve().parent.parent


def _data(n=83, n_fail=30, d=1536, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, int)
    y[:n_fail] = 1
    rng.shuffle(y)
    groups = [f"r{i // 3}" for i in range(n)]  # 每支錄影約 3 段
    X = rng.normal(size=(n, d))
    X[:, 5] += 1.8 * y
    X[:, 900] -= 1.4 * y
    return X, y, groups


def _selected(**kw):
    d = {"recipe": "F2S2", "k_grid": list(pp.K_GRID), "c1_grid": list(pp.C1_GRID), "c2_grid": list(pp.C2_GRID),
         "inner": pp.INNER, "crop_box": list(pp.CROP_BOX)}
    d.update(kw)
    return d


def test_fit_matches_sklearn_path():
    X, y, g = _data()
    m = pf.fit_final(X, y, g)
    assert 1 <= len(m["features"]) <= m["k"] <= 10
    assert m["k"] in pp.K_GRID and m["c1"] in pp.C1_GRID and m["c2"] in pp.C2_GRID
    best = (m["k"], m["c1"], m["c2"])
    ref = pp._fold_predict(X, y, X, "S2", [best])[best]   # 第 14 包同一條路徑：在全部段訓練、預測全部段
    assert np.array_equal(pf.predict(m, X), ref)
    assert {5, 900} & set(m["features"].tolist())        # 有訊號的維度被挑中
    p = pf.predict_proba(m, X)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.array_equal(pf.predict(m, X), (pf.decision(m, X) > 0).astype(int))
    assert pp.bal_acc(y, pf.predict(m, X)) > 0.8


def test_same_choice_as_inner_select():
    X, y, g = _data(seed=1)
    m = pf.fit_final(X, y, g)
    best, scores = pp.inner_select(X, y, g, "S2", seed=0, n_inner=pp.INNER)
    assert (m["k"], m["c1"], m["c2"]) == best and m["inner_bal"] == pytest.approx(scores[best])


def test_save_load_roundtrip(tmp_path):
    X, y, g = _data(seed=2)
    m = pf.fit_final(X, y, g)
    npz, js = pf.save_model(m, {"part": "bolt 4"}, tmp_path / "bolt_4")
    assert npz.exists() and js.exists()
    r = pf.load_model(tmp_path / "bolt_4")
    assert r["part"] == "bolt 4" and r["k"] == m["k"]
    assert np.allclose(pf.predict_proba(r, X), pf.predict_proba(m, X))
    assert np.array_equal(pf.predict(r, X), pf.predict(m, X))


def test_single_vector_and_dim_check():
    X, y, g = _data(seed=3)
    m = pf.fit_final(X, y, g)
    assert pf.predict_proba(m, X[0]).shape == (1,)
    with pytest.raises(ValueError):
        pf.predict(m, X[:, :100])


def test_nan_features_become_zero():
    X, y, g = _data(seed=4)
    m = pf.fit_final(X, y, g)
    Xn = X.copy()
    Xn[0, m["features"][0]] = np.nan
    X0 = X.copy()
    X0[0, m["features"][0]] = 0.0
    assert pf.predict_proba(m, Xn)[0] == pytest.approx(pf.predict_proba(m, X0)[0])


def test_one_class_refused():
    X, y, g = _data()
    with pytest.raises(ValueError):
        pf.fit_final(X, np.zeros(len(y), int), g)


def test_check_selected():
    pf.check_selected(_selected())
    with pytest.raises(ValueError):
        pf.check_selected(_selected(recipe="F1S2"))
    with pytest.raises(ValueError):
        pf.check_selected(_selected(k_grid=[2, 4, 6, 8, 10, 12]))
    with pytest.raises(ValueError):
        pf.check_selected(_selected(crop_box=[0.1, 0.1, 0.9, 0.9]))


def test_reserved_and_unknown_parts_refused(tmp_path):
    for part in pp.RESERVED + ("not a part",):
        with pytest.raises(ValueError):
            pf.train_part(part, index_dir=str(tmp_path), selected_path=str(tmp_path / "none.yaml"),
                          model_dir=tmp_path, report_dir=tmp_path, check_encode=0, encs={})


def test_frozen_config_is_supported():
    p = ROOT / "configs" / "perpart_selected.yaml"
    if not p.exists():
        pytest.skip("沒有凍結設定（只在使用者電腦上有）")
    pf.check_selected(pp.load_selected(p))


def test_slug_and_report(tmp_path):
    assert pf.slug("bolt 4") == "bolt_4" and pf.slug("D-SUB") == "D_SUB"
    X, y, g = _data(seed=5)
    m = pf.fit_final(X, y, g)
    pred = pf.predict(m, X)
    res = {"part": "bolt 4", "recipe": "F2S2", "frozen_at": "x", "n": len(y), "n_failure": int(y.sum()),
           "n_recordings": len(set(g)), "train_bal": pp.bal_acc(y, pred), "train_counts": pp.counts(y, pred),
           "cv": {"bal_mean": 0.935, "bal_min": 0.914, "bal_max": 0.948, "date_bal": 0.9, "chosen": {"(8, 1.0, 0.01)": 5}},
           "model": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()},
           "files": ["a.npz", "a.json"], "reload_same": True,
           "check": {"rows": [{"segment_id": "s1", "label": 1, "p_cached": 0.9, "p_fresh": 0.91, "same": True,
                               "max_feature_diff": 0.01}, {"segment_id": "s2", "error": "找不到影格"}],
                     "all_same": False, "max_p_diff": 0.01}}
    pf.write_report(res, tmp_path / "r.md")
    t = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "最終模型" in t and "k＝" in t and "94%" in t and "找不到影格" in t


def test_bat_ascii_crlf():
    p = ROOT / "run_perpart_final.bat"
    if not p.exists():
        pytest.skip("沒有 .bat")
    b = p.read_bytes()
    assert all(c < 128 for c in b) and b"\r\n" in b and b.count(b"\n") == b.count(b"\r\n")
