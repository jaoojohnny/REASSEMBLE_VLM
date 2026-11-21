"""第 15 包：每零件分類器＋Qwen 機率門檻的組合（不需要影格、GPU、模型）。"""

import json
from pathlib import Path

import numpy as np

from assembly_monitor import combo as cb
from assembly_monitor import perpart as pp

ROOT = Path(__file__).resolve().parent.parent


def _data(n=72, n_fail=18, seed=0, img=1.6, qsig=2.0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, int)
    y[:n_fail] = 1
    rng.shuffle(y)
    g = [f"r{i}" if i % 7 else f"r{i - 1}" for i in range(n)]
    X = rng.normal(size=(n, 300))
    X[:, 3] += img * y
    s = 1 / (1 + np.exp(-(rng.normal(size=n) + qsig * (2 * y - 1) + 6)))   # 偏高的失敗分數（像 Qwen）
    return X, s, y, g


def test_methods_and_rule():
    assert cb.METHODS[0] == "M0" and set(cb.OPTIONS) == set(cb.METHODS) - {"C4"}
    dev = {m: {"a": v} for m, v in zip(cb.METHODS, [0.78, 0.70, 0.75, 0.74, 0.795, 0.80])}
    sel = cb.select_method(dev)
    assert sel["best_method"] == "C4" and sel["method"] == "M0"   # 差 2 點以內選簡單的
    dev2 = {m: {"a": v} for m, v in zip(cb.METHODS, [0.70, 0.60, 0.65, 0.66, 0.75, 0.76])}
    assert cb.select_method(dev2)["method"] == "C1"


def test_nested_combo_consistent_with_perpart():
    X, s, y, g = _data()
    q, qp = cb.logit(s), (s > cb.TAU).astype(int)
    r = cb.nested_combo(X, q, qp, y, g, repeats=2)
    m0 = pp.nested_cv(X, y, g, "S2", repeats=2)
    assert r["methods"]["M0"]["repeats"][0]["pred"] == m0["repeats"][0]["pred"]   # M0＝第 14 包的做法（同分組、同種子）
    m1 = r["methods"]["M1"]["repeats"][0]["pred"]
    assert m1 == qp.tolist()
    for k in range(2):
        a = np.array(r["methods"]["M0"]["repeats"][k]["pred"])
        assert r["methods"]["C2"]["repeats"][k]["pred"] == (a & qp).tolist()
        assert r["methods"]["C3"]["repeats"][k]["pred"] == (a | qp).tolist()
    assert sum(r["c4_picks"].values()) == 2 * 5
    assert r["methods"]["C1"]["bal_mean"] > 0.6


def test_stack_uses_qwen_when_images_useless():
    X, s, y, g = _data(img=0.0, qsig=2.5, seed=3)
    q, qp = cb.logit(s), (s > cb.TAU).astype(int)
    r = cb.nested_combo(X, q, qp, y, g, repeats=1)
    assert r["methods"]["C1"]["bal_mean"] > r["methods"]["M0"]["bal_mean"] + 0.1


def test_raw_scores_and_qwen_scores(tmp_path):
    d = tmp_path / "final_test_rpeg1" / "qwen3vl8b_prob_t0999" / "qwen3vl_8b_rscore_t0999" / "f4"
    d.mkdir(parents=True)
    recs = [{"segment_id": "a__seg1", "reason": {"p_yes": {"held_by_gripper": 0.5, "place_empty_or_part_misplaced": 0.5,
                                                            "board_moved": 0.0}}},
            {"segment_id": "a__seg2", "reason": str({"p_yes": {"held_by_gripper": 0.0, "place_empty_or_part_misplaced": 0.0,
                                                                "board_moved": 0.0}})},
            {"segment_id": "a__seg3", "error": "timeout"}]
    (d / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    sc = cb.raw_scores(d / "raw.jsonl")
    assert abs(sc["a__seg1"] - 0.75) < 1e-9 and sc["a__seg2"] == 0.0 and sc["a__seg3"] is None
    q = cb.qwen_scores("round peg 1", tmp_path)
    assert set(q) == {"a__seg1", "a__seg2"}
    assert abs(cb.logit(np.array([1.0]))[0] - np.log((1 - 1e-6) / 1e-6)) < 1e-6


def test_new_run_spec_covers_missing_parts():
    have = set(cb.P11_DEV) | set(cb.FINAL_DIRS)
    for part in pp.DEV_PARTS + pp.TEST_PARTS:
        assert part in have or part in {v[0] for v in cb.NEW_RUN_SPEC.values()}, part
    assert not {v[0] for v in cb.NEW_RUN_SPEC.values()} & set(pp.RESERVED)


def test_write_and_load_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "load_selected", lambda *a, **k: {"recipe": "F2S2"})
    p = tmp_path / "c.yaml"
    res = {"selection": {"method": "C1", "avg": {m: 0.7 for m in cb.METHODS}}}
    cb.write_selected(res, p)
    assert cb.load_selected(p)["method"] == "C1"
    try:
        cb.write_selected({"selection": {"method": "M0", "avg": {m: 0.7 for m in cb.METHODS}}}, p)
        raise AssertionError("應該拒絕覆寫")
    except ValueError:
        pass


def test_bat_ascii_crlf():
    b = (ROOT / "run_combo.bat").read_bytes()
    assert all(x < 128 for x in b)
    assert b.count(b"\n") == b.count(b"\r\n") > 10
    for s in (b"scripts.combo qwen", b"scripts.combo dev", b"scripts.combo test", b"test_p15_combo.py"):
        assert s in b, s
