"""第 6 包：評分指標、信賴區間、McNemar、gold 指標、開發集報告、測試集凍結鎖。不需要 GPU。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
yaml = pytest.importorskip("yaml")

from assembly_monitor import evaluate as ev  # noqa: E402
from assembly_monitor import final as fin  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))   # 共用 test_vlm.py 的合成資料工具


def _rows(pairs, rec_of=lambda i: f"R{i // 2}"):
    return [{"segment_id": f"s{i}", "recording_id": rec_of(i), "truth": t, "pred": p, "status": "ok" if p else "invalid",
             "correct": p == t, "difficulty": "clear", "has_gold": False} for i, (t, p) in enumerate(pairs)]


def test_result_metrics_count_uncertain_and_invalid_as_wrong():
    rows = _rows([("failure", "failure"), ("failure", "uncertain"), ("failure", None),
                  ("success", "success"), ("success", "failure"), ("success", "success")])
    m = ev.result_metrics(rows)
    assert (m["n"], m["n_correct"], m["n_caught"], m["n_false_alarms"], m["n_uncertain"], m["n_invalid_or_error"]) == (6, 3, 1, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(0.5) and m["balanced_accuracy"] == pytest.approx((1 / 3 + 2 / 3) / 2)
    # failure：precision 1/2、recall 1/3 → F1 0.4；success：precision 2/2、recall 2/3 → 0.8
    assert m["macro_f1"] == pytest.approx((0.4 + 0.8) / 2)
    assert m["answered_accuracy"] == pytest.approx(3 / 4)
    always_success = ev.result_metrics(_rows([("failure", "success")] * 3 + [("success", "success")] * 7))
    assert always_success["accuracy"] == pytest.approx(0.7) and always_success["balanced_accuracy"] == pytest.approx(0.5)


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    rows = _rows([("failure", "failure"), ("success", "success"), ("success", "failure"), ("failure", "failure")] * 5)
    a, b = ev.bootstrap_ci(rows, n_boot=300, seed=1), ev.bootstrap_ci(rows, n_boot=300, seed=1)
    assert a == b
    lo, hi = a["accuracy"]
    assert 0 <= lo <= ev.result_metrics(rows)["accuracy"] <= hi <= 1


def test_mcnemar_exact():
    assert ev.mcnemar_exact([True] * 5 + [False] * 5, [False] * 5 + [False] * 5) == {"a_only": 5, "b_only": 0, "p_value": 0.0625}
    assert ev.mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
    with pytest.raises(ValueError):
        ev.mcnemar_exact([True], [True, False])


def test_gold_overrides_truth_and_reason_evidence_metrics():
    segs = [{"segment_id": "a", "recording_id": "A", "dataset_result": "failure"},
            {"segment_id": "b", "recording_id": "B", "dataset_result": "failure"},
            {"segment_id": "c", "recording_id": "C", "dataset_result": "success"}]
    gold = {"a": {"result": "failure", "failure_reason": "misalignment", "visual_difficulty": "subtle",
                  "evidence_start_sec": 5.0, "evidence_end_sec": 9.0, "review_status": "adjudicated"},
            "c": {"result": "failure", "failure_reason": "unknown", "visual_difficulty": "not_visible",
                  "evidence_start_sec": 1.0, "evidence_end_sec": 2.0, "review_status": "approved"}}
    preds = {"a": {"status": "ok", "result": "failure", "failure_reason": "misalignment", "evidence_start_sec": 8.0, "evidence_end_sec": 12.0},
             "b": {"status": "ok", "result": "success"},
             "c": {"status": "ok", "result": "success"}}
    rows = ev.segment_rows(preds, segs, gold)
    assert [r["truth"] for r in rows] == ["failure", "failure", "failure"]   # c 以 gold 為準
    assert [r["difficulty"] for r in rows] == ["subtle", "（未標）", "not_visible"]
    g = ev.gold_metrics(preds, ["a", "b", "c"], gold)
    assert (g["n_gold"], g["n_gold_failures"], g["n_pred_failure_on_gold_failures"], g["reason_correct"]) == (2, 2, 1, 1)
    assert g["evidence_hit_rate"] == 1.0 and g["evidence_mean_iou"] == pytest.approx(1 / 7)
    assert set(ev.by_difficulty(rows)) == {"subtle", "not_visible", "（未標）"}


def test_load_gold_keeps_official_only(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [{"segment_id": "a", "result": "failure", "review_status": "approved"},
                                                  {"segment_id": "b", "result": "failure", "review_status": "disputed"}]),
                 encoding="utf-8")
    assert set(ev.load_gold([p])) == {"a"}


# ---------------------------------------------------------------- 用 mock 模型實跑一輪


def _setup_runs(tmp_path, monkeypatch, split_name="dev"):
    """合成 4 段＋影格，回傳 (exp yaml 路徑, model yaml 路徑)。load_split 改成回傳合成段。"""
    import assembly_monitor.split as split_mod
    from test_vlm import CAMS, SAMPLING, _task, _write_segment

    segs = []
    for i, res in enumerate(["failure", "success", "success", "failure"]):
        sid = f"2025-01-0{i + 1}-10-00-00__seg{i}"
        for n in (1, 8):
            _write_segment(tmp_path / "frames", sid, n)
        segs.append({"segment_id": sid, "recording_id": sid.split("__")[0], "dataset_result": res, "split": split_name})

    def fake_load_split(split_dir, include=("dev",), allow_test=False):
        if "test" in include and not allow_test:
            from assembly_monitor.split import LockedTestSetError
            raise LockedTestSetError("locked")
        return [s for s in segs if s["split"] in include]

    monkeypatch.setattr(split_mod, "load_split", fake_load_split)
    tpl = tmp_path / "tpl.txt"
    tpl.write_text((ROOT / "configs/prompts/bnc_p1.txt").read_text(encoding="utf-8"), encoding="utf-8")
    exp = {"name": "devx", "split_dir": str(tmp_path / "split"), "include": ["dev"], "frames_root": str(tmp_path / "frames"),
           "sampling_template": SAMPLING, "frames": [1, 8], "cameras": CAMS, "out_dir": str(tmp_path / "exp"),
           "timeout_sec": 30, "repair_retry": True, "evidence_tolerance_sec": 0.05,
           "prompt": {"version": "t1", "template": str(tpl)}, "task": _task()}
    exp_p = tmp_path / "experiment.yaml"
    exp_p.write_text(yaml.safe_dump({"experiment": exp}, allow_unicode=True), encoding="utf-8")
    models_p = tmp_path / "model.yaml"
    models_p.write_text(yaml.safe_dump({"models": {"mock": {"type": "mock"}, "majority": {"type": "majority", "result": "success"}}}),
                        encoding="utf-8")
    return exp_p, models_p, segs


def test_evaluate_dev_report(tmp_path, monkeypatch):
    from assembly_monitor import vlm
    from test_vlm import REASONS

    exp_p, models_p, _ = _setup_runs(tmp_path, monkeypatch)
    exp = vlm.load_experiment(exp_p)
    vlm.run_experiment(exp=exp, model_cfgs=vlm.load_model_configs(models_p), model_keys=["majority", "mock"], frames=[1, 8],
                       reasons=REASONS)
    rep = ev.evaluate_dev([exp_p], tmp_path / "dev_eval", gold_paths=[], n_boot=50)
    md = Path(rep["md"]).read_text(encoding="utf-8")
    assert "任務判斷成功率" in md and "majority" in md and "mock·f8·t1·+2 s" in md
    assert (tmp_path / "dev_eval" / "metrics.csv").exists() and (tmp_path / "dev_eval" / "per_segment.csv").exists()
    maj = next(r for r in rep["results"] if r["desc"]["model"] == "majority")
    assert maj["metrics"]["n_correct"] == 2 and maj["metrics"]["balanced_accuracy"] == 0.5


def _final_cfg(tmp_path, exp_p, models_p, frozen=True):
    d = {"final": {"frozen": frozen, "frozen_at": "2026-10-13", "split_dir": str(tmp_path / "split"), "out_dir": str(tmp_path / "exp"),
                   "out_name": "final_test", "models": str(models_p), "primary": "mock_f8",
                   "methods": [{"id": "majority", "kind": "majority", "experiment": str(exp_p)},
                               {"id": "mock_f8", "kind": "vlm", "experiment": str(exp_p), "model": "mock", "frames": 8}]}}
    p = tmp_path / "final_test.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return p


def test_final_requires_freeze_and_locks_settings(tmp_path, monkeypatch):
    from test_vlm import REASONS

    exp_p, models_p, _ = _setup_runs(tmp_path, monkeypatch, split_name="test")
    cfg = fin.load_final_config(_final_cfg(tmp_path, exp_p, models_p, frozen=False))
    with pytest.raises(fin.FinalLockError):
        fin.check_or_write_lock(cfg)
    cfg = fin.load_final_config(_final_cfg(tmp_path, exp_p, models_p))
    assert fin.needed_extractions(cfg) == [{"end_buffer": 2.0, "frames": [8]}]
    fin.run_final(cfg, reasons=REASONS)
    lock = tmp_path / "exp" / "final_test" / "lock.json"
    assert lock.exists()
    for m in cfg["methods"]:
        assert (fin.method_run_dir(cfg, m) / "predictions.jsonl").exists(), m["id"]
    rep = fin.evaluate_final(cfg, gold_paths=[], n_boot=50)
    md = Path(rep["md"]).read_text(encoding="utf-8")
    assert "測試集最終評估" in md and "mock_f8 ★" in md and "McNemar" in md
    fin.run_final(cfg, reasons=REASONS)   # 同一份設定可以續跑
    assert str(tmp_path / "tpl.txt") in json.loads(lock.read_text(encoding="utf-8"))["files"]   # 提示詞內容也被凍結
    (tmp_path / "tpl.txt").write_text("changed {part}", encoding="utf-8")   # 改提示詞 → 雜湊變 → 拒絕
    with pytest.raises(fin.FinalLockError):
        fin.check_or_write_lock(cfg)


def test_final_config_validation(tmp_path):
    p = tmp_path / "f.yaml"
    p.write_text(yaml.safe_dump({"final": {"split_dir": "x", "out_name": "y", "methods": [{"id": "a", "kind": "vlm"}]}}), encoding="utf-8")
    with pytest.raises(fin.FinalConfigError):
        fin.load_final_config(p)
    p.write_text(yaml.safe_dump({"final": {"split_dir": "x", "out_name": "y", "primary": "zz",
                                           "methods": [{"id": "a", "kind": "majority"}]}}), encoding="utf-8")
    with pytest.raises(fin.FinalConfigError):
        fin.load_final_config(p)


def test_repo_final_config_loads():
    cfg = fin.load_final_config(ROOT / "configs/final_test.yaml")
    assert cfg["split_dir"] == "data/splits/bnc_dev25_test75_s0" and any(m["kind"] == "majority" for m in cfg["methods"])
    ex = fin.needed_extractions(cfg)
    assert all(x["end_buffer"] > 0 for x in ex)
