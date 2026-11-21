"""第 18 包：失敗步驟——讀步驟標記、類別篩選、三類平均、三個是非題、巢狀交叉驗證、打亂答案、報告（合成資料，不需要 GPU）。"""

import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from assembly_monitor import failstep as fs
from assembly_monitor import perpart as pp

SEG_COLS = ["recording_id", "segment_id", "action", "part", "success", "validation_status", "duration_s"]
SK_COLS = ["recording_id", "segment_id", "skill_id", "text", "success"]


def _index(tmp_path, segs, skills):
    d = tmp_path / "index"
    d.mkdir()
    with (d / "segments.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, SEG_COLS)
        w.writeheader()
        w.writerows(segs)
    with (d / "skills.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, SK_COLS)
        w.writeheader()
        w.writerows(skills)
    return d


def _seg(rec, sid, ok, part="bolt 4", dur=10.0):
    return {"recording_id": rec, "segment_id": sid, "action": "insert", "part": part, "success": str(ok),
            "validation_status": "ok", "duration_s": dur}


def _sk(rec, sid, steps):
    return [{"recording_id": rec, "segment_id": sid, "skill_id": i, "text": t, "success": str(ok)}
            for i, (t, ok) in enumerate(steps)]


def test_first_failed_step_and_dataset(tmp_path):
    segs, skills = [], []
    plan = [("Twist", 4), ("Align", 3), ("Approach", 2)]          # Approach 只有 2 段 → 不納入
    n = 0
    for step, k in plan:
        for _ in range(k):
            n += 1
            segs.append(_seg(f"r{n}", n, False, dur=5.0 + n))
            seq = []
            for t in ("Approach", "Align", "Twist"):
                seq.append((t, t != step))
                if t == step:
                    break
            skills += _sk(f"r{n}", n, seq)
    segs.append(_seg("r50", 50, False))                            # 失敗段但沒有失敗步驟
    skills += _sk("r50", 50, [("Approach", True), ("Align", True)])
    segs.append(_seg("r60", 60, True))                             # 成功段不看
    skills += _sk("r60", 60, [("Approach", True), ("Align", False)])
    segs.append(_seg("r70", 70, False))                            # 兩個失敗步驟 → 取第一個
    skills += _sk("r70", 70, [("Approach", True), ("Align", False), ("Twist", False)])
    idx = _index(tmp_path, segs, skills)

    st = fs.first_failed_steps(idx)
    assert st["r70__seg70"]["step"] == "Align" and st["r70__seg70"]["n_failed"] == 2 and not st["r70__seg70"]["is_last"]
    assert st["r1__seg1"]["is_last"] and "r50__seg50" not in st

    ds = fs.step_dataset(idx, "bolt 4")
    assert ds["classes"] == ["Align", "Twist"]                     # 各 4 段，同數依名稱
    assert ds["counts"] == {"Align": 4, "Twist": 4} and ds["all_counts"]["Approach"] == 2
    assert ds["no_step"] == ["r50__seg50"] and len(ds["rare"]) == 2 and len(ds["segments"]) == 8
    assert ds["n_failures"] == 11
    assert all(s["label"] == 1 and "duration" in s for s in ds["segments"])


def test_restrict_drops_missing_and_small_classes():
    keep = ([{"segment_id": f"t{i}", "step": "Twist"} for i in range(4)] + [{"segment_id": f"a{i}", "step": "Align"} for i in range(3)]
            + [{"segment_id": f"p{i}", "step": "Approach"} for i in range(2)])       # 接近原本 3 段，少 1 段沒有影格
    ds = {"part": "BNC", "classes": ["Twist", "Align", "Approach"], "counts": {}, "segments": [], "rare": ["x"], "no_step": []}
    out, idx = fs.restrict(ds, keep, ["p_missing"])
    assert out["classes"] == ["Twist", "Align"] and out["counts"] == {"Twist": 4, "Align": 3}
    assert idx == list(range(7)) and len(out["segments"]) == 7
    assert out["missing"] == ["p_missing"] and out["rare"] == ["x", "p0", "p1"]
    assert fs.avg_name(2) == "兩類平均" and fs.avg_name(3) == "三類平均"


def test_reserved_part_refused(tmp_path):
    idx = _index(tmp_path, [], [])
    with pytest.raises(ValueError):
        fs.step_dataset(idx, pp.RESERVED[0])


def test_bal_multi_and_counts():
    y = np.array([0, 0, 0, 0, 1, 1, 2])
    p = np.array([0, 0, 0, 1, 1, 0, 0])
    assert fs.bal_multi(y, p, 3) == pytest.approx((0.75 + 0.5 + 0.0) / 3)
    assert fs.recall_counts(y, p, 3) == [[3, 4], [1, 2], [0, 1]]
    assert fs.bal_multi(np.array([0, 0]), np.array([0, 1]), 3) == pytest.approx(0.5)   # 沒出現的步驟不算


def _data(n_per=(14, 9, 7), d=40, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for c, n in enumerate(n_per):
        for i in range(n):
            v = rng.normal(size=d)
            v[c] += sep
            X.append(v)
            y.append(c)
            g.append(f"rec{c}_{i // 2}")                           # 每支錄影 2 段
    return np.array(X), np.array(y), g


def test_fold_predict_separable_and_missing_class():
    X, y, _ = _data()
    key = (2, 1.0, 1.0)
    p = fs.fold_predict(X, y, X, 3, [key])[key]
    assert fs.bal_multi(y, p, 3) > 0.9
    keep = y != 2                                                  # 訓練折沒有第 3 個步驟 → 不會被選
    p2 = fs.fold_predict(X[keep], y[keep], X, 3, [key])[key]
    assert set(p2.tolist()) <= {0, 1}


def test_class_scores_no_feature_gives_half():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(12, 6))
    y = np.array([0] * 6 + [1] * 6)
    s = fs.class_scores(Z, y, Z, 2, [(2, 1e-6, 1.0)])[(2, 1e-6, 1.0)]   # L1 罰到全部是 0
    assert np.allclose(s, 0.5)


def test_nested_cv_and_permutation():
    X, y, g = _data()
    r = fs.nested_cv(X, y, g, 3, repeats=1)
    assert r["bal_mean"] > 0.8 and len(r["repeats"]) == 1
    assert np.array(r["votes"]).sum() == len(y)
    assert sum(r["chosen"].values()) == pp.OUTER
    pm = fs.permutation_test(X, y, g, 3, r["repeats"][0]["bal"], n_perm=4)
    assert pm["n_ge"] == 0 and pm["p"] == pytest.approx(1 / 5) and pm["null_mean"] < 0.6
    pm2 = fs.permutation_test(X, y, g, 3, r["repeats"][0]["bal"], n_perm=4, n_jobs=2)     # 平行和一個一個跑結果相同
    assert pm2["null_mean"] == pytest.approx(pm["null_mean"]) and pm2["p"] == pm["p"]


def test_permutation_falls_back_to_serial(monkeypatch):
    import joblib

    def boom(*a, **k):
        raise RuntimeError("no pool")

    monkeypatch.setattr(joblib, "Parallel", boom)
    X, y, g = _data(n_per=(6, 5, 5), d=12)
    pm = fs.permutation_test(X, y, g, 3, 1.0, n_perm=2, n_jobs=4)
    assert pm["n_perm"] == 2 and pm["null_mean"] is not None


def test_duration_baseline_uses_duration():
    _, y, g = _data()
    dur = np.array([5.0, 10.0, 15.0])[y] + np.random.default_rng(0).normal(scale=0.3, size=len(y))
    d = fs.duration_baseline(dur, y, g, 3, repeats=2)
    assert d["bal_mean"] > 0.9 and d["median_s"][0] < d["median_s"][2]


def test_evaluate_and_report(tmp_path):
    X, y, g = _data()
    classes = ["Twist", "Align", "Approach"]
    segs = [{"segment_id": f"{g[i]}__seg{i}", "recording_id": g[i], "part": "bolt 4", "label": 1, "step": classes[y[i]],
             "duration": 5.0 + 3 * y[i], "n_failed": 1, "is_last": True, "sequence": "Approach > Align"} for i in range(len(y))]
    ds = {"part": "bolt 4", "classes": classes, "counts": {"Twist": 14, "Align": 9, "Approach": 7},
          "all_counts": {"Twist": 14, "Align": 9, "Approach": 7}, "segments": segs, "n_failures": 30, "no_step": [], "rare": []}
    pr = fs.evaluate_part(ds, {"F2": X}, ("F2S2",), n_perm=2)
    assert pr["majority"]["acc"] == pytest.approx(14 / 30) and pr["perm"]["n_perm"] == 2
    assert len(pr["segments"]) == 30 and all(0 <= s["n_right"] <= pp.REPEATS for s in pr["segments"])
    res = {"generated": "now", "parts": {"bolt 4": pr}, "seconds": 1.0}
    fs.write_report(res, tmp_path / "report.md")
    rep = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "三類平均" in rep and "旋轉" in rep and "判定" in rep and "打亂答案檢定" in rep
    assert "誤判" not in rep
    assert fs.verdict(pr).startswith("沒有達到")                    # 只打亂 2 次，p 最小 0.33


def test_verdict_fails_when_low():
    pr = {"recipes": {"F2S2": {"bal_mean": 0.45}}, "perm": {"p": 0.2}}
    assert fs.verdict(pr).startswith("沒有達到")
    assert fs.verdict({"recipes": {"F2S2": {"bal_mean": 0.7}}, "perm": {"p": 0.01}}).startswith("達到")
    assert fs.verdict({"recipes": {"F2S2": {"bal_mean": 0.7}}, "perm": {"p": 0.2}}).startswith("沒有達到")
    assert "不下結論" in fs.verdict({"recipes": {"F2S2": {"bal_mean": 0.9}}, "perm": None})
