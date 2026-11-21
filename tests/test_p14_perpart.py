"""第 14 包：每種零件各自一個小分類器——巢狀交叉驗證、挑特徵、選做法的規則（不需要影格、GPU）。"""

from pathlib import Path

import numpy as np

from assembly_monitor import perpart as pp

ROOT = Path(__file__).resolve().parent.parent


def _data(n=72, n_fail=18, d=1536, signal=True, seed=0, strength=1.0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, int)
    y[:n_fail] = 1
    rng.shuffle(y)
    groups = [f"r{i}" if i % 7 else f"r{i - 1}" for i in range(n)]  # 少數錄影有兩段
    X = rng.normal(size=(n, d))
    if signal:
        X[:, 5] += 1.6 * strength * y
        X[:, 9] -= 1.2 * strength * y
    return X, y, groups


def test_part_lists():
    assert len(pp.DEV_PARTS) == 5 and len(pp.TEST_PARTS) == 10
    assert not set(pp.DEV_PARTS) & set(pp.TEST_PARTS)
    assert not (set(pp.DEV_PARTS) | set(pp.TEST_PARTS)) & set(pp.RESERVED)
    assert max(pp.K_GRID) <= 10 and pp.RECIPES[0] == "F2S1"


def test_part_segments(tmp_path):
    rows = ["recording_id,segment_id,action,part,validation_status,success",
            "r1,3,insert,BNC,ok,True", "r1,4,insert,BNC,warn,False", "r2,1,insert,BNC,error,True",
            "r2,2,insert,USB,ok,True", "r3,5,pick,BNC,ok,True", "r4,6,insert,BNC,ok,"]
    (tmp_path / "segments.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    segs = pp.part_segments(tmp_path, "BNC")
    assert [(s["segment_id"], s["label"]) for s in segs] == [("r1__seg3", 0), ("r1__seg4", 1)]


def test_bal_acc_counts():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([1, 0, 0, 0, 0, 1])
    assert abs(pp.bal_acc(y, p) - (0.5 + 0.75) / 2) < 1e-9
    assert pp.counts(y, p) == {"ok_failure": 1, "n_failure": 2, "ok_success": 3, "n_success": 4}


def test_folds_keep_recordings_apart():
    X, y, g = _data()
    for tr, te in pp.folds(y, g, 5, 0):
        assert not ({g[i] for i in tr} & {g[i] for i in te})
        assert y[te].sum() >= 1


def test_nested_cv_signal_noise_and_reproducible():
    X, y, g = _data()
    r = pp.nested_cv(X, y, g, "S2", repeats=2)
    assert r["bal_mean"] > 0.7 and r["k_mean"] <= 10
    assert pp.nested_cv(X, y, g, "S2", repeats=2)["repeats"][0]["pred"] == r["repeats"][0]["pred"]
    Xn, yn, gn = _data(signal=False, seed=1)
    assert pp.nested_cv(Xn, yn, gn, "S2", repeats=2)["bal_mean"] < 0.68   # 沒有訊號就不該高（沒有洩漏）
    s1 = pp.nested_cv(X[:, :40], y, g, "S1", repeats=1)
    assert len(s1["repeats"][0]["pred"]) == len(y) and set(s1["repeats"][0]["pred"]) <= {0, 1}


def test_l1_selects_nothing_predicts_majority():
    y = np.array([0] * 12 + [1] * 5)
    X = np.ones((17, 30))
    out = pp._fold_predict(X[:12], y[:12], X[12:], "S2", [(4, 0.05, 0.1)])  # 只有一類 → 常數
    assert (out[(4, 0.05, 0.1)] == 0).all()
    out = pp._fold_predict(X, y, X[:3], "S2", [(4, 0.05, 0.1)])  # 特徵都一樣 → L1 什麼都不留 → 多數類
    assert (out[(4, 0.05, 0.1)] == 0).all()


def test_choose_and_selection_rule():
    assert pp.choose({(2, 0.1): 0.8, (4, 0.01): 0.8, (2, 0.01): 0.8}) == (2, 0.01)
    assert pp.choose({(2, 0.05, 1.0): 0.7, (6, 0.2, 0.1): 0.75}) == (6, 0.2, 0.1)
    sc = {r: {"a": v, "b": v} for r, v in zip(pp.RECIPES, [0.70, 0.72, 0.715, 0.73, 0.60, 0.61])}
    sel = pp.select_recipe(sc)
    assert sel["best_recipe"] == "F1S2" and sel["recipe"] == "F2S2"   # 差 2 點以內選比較簡單的
    sc2 = {r: {"a": v} for r, v in zip(pp.RECIPES, [0.60, 0.60, 0.60, 0.80, 0.60, 0.60])}
    assert pp.select_recipe(sc2)["recipe"] == "F1S2"


def test_write_selected_refuses_change(tmp_path):
    res = {"selection": {"recipe": "F2S2", "avg": {r: 0.7 for r in pp.RECIPES}}}
    p = tmp_path / "sel.yaml"
    pp.write_selected(res, p)
    assert "F2S2" in p.read_text(encoding="utf-8")
    pp.write_selected(res, p)   # 一樣：沒事
    res2 = {"selection": {"recipe": "F1S1", "avg": {r: 0.7 for r in pp.RECIPES}}}
    try:
        pp.write_selected(res2, p)
        raise AssertionError("應該拒絕覆寫")
    except ValueError:
        pass


def test_bat_ascii_crlf():
    b = (ROOT / "run_perpart_dev.bat").read_bytes()
    assert all(x < 128 for x in b)
    assert b.count(b"\n") == b.count(b"\r\n") > 10
    assert b"scripts.perpart prepare" in b and b"scripts.perpart dev" in b and b"test_p14_perpart.py" in b


def test_date_grouped_cv_is_leave_one_day_out():
    X, y, _ = _data(strength=2.0)
    days = [f"2025-01-{9 + i % 5:02d}" for i in range(len(y))]
    for tr, te in pp.folds(y, days, 5, 0):
        assert len({days[i] for i in te}) == 1 and not ({days[i] for i in tr} & {days[i] for i in te})
    r = pp.nested_cv(X, y, days, "S2", repeats=1)
    assert len(r["repeats"]) == 1 and r["bal_mean"] > 0.7


def test_load_selected(tmp_path):
    p = tmp_path / "sel.yaml"
    try:
        pp.load_selected(p)
        raise AssertionError("沒有檔案應該報錯")
    except ValueError:
        pass
    pp.write_selected({"selection": {"recipe": "F2S2", "avg": {r: 0.7 for r in pp.RECIPES}}}, p)
    assert pp.load_selected(p)["recipe"] == "F2S2"
    assert pp.date_of("2025-01-13-11-07-43") == "2025-01-13"
    assert pp.method_name("probe_usb_eb1_f1").startswith("混合分類器") and pp.method_name("gpt61_sol_ft_eb05_f4") == "GPT-6.1 Sol"


def test_compare_same_segments():
    pr = {"ids": ["a", "b", "c", "d"], "labels": [1, 1, 0, 0],
          "rec": {"repeats": [{"pred": [1, 0, 0, 0]}, {"pred": [1, 1, 0, 1]}]}}
    out = pp.compare_same_segments(pr, {"gpt": {"a": "failure", "b": "success", "c": "success"}, "none": {}})
    assert set(out) == {"gpt"} and out["gpt"]["n"] == 3
    assert abs(out["gpt"]["bal"] - 0.75) < 1e-9          # a 對、b 錯、c 對
    assert abs(out["gpt"]["classifier_same_segments"] - (0.75 + 1.0) / 2) < 1e-9


def test_test_bat_ascii_crlf():
    b = (ROOT / "run_perpart_test.bat").read_bytes()
    assert all(x < 128 for x in b)
    assert b.count(b"\n") == b.count(b"\r\n") > 10
    assert b"scripts.perpart test" in b and b"test_p14_perpart.py" in b
