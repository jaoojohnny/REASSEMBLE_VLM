"""第 12 包：兩階段（Qwen 機率門檻先判，判失敗才問 GPT）——開發集設定與計分。"""

import json
from pathlib import Path

import pytest
import yaml

from assembly_monitor import cascade_dev
from assembly_monitor.vlm import load_experiment

KEYS = ("sqpeg2", "rpeg3", "dsub", "mgear")


@pytest.mark.parametrize("key", KEYS)
def test_cdev_gpt_experiments(key):
    e = load_experiment(f"configs/experiment_cdev_{key}_gpt.yaml")
    v3 = load_experiment(f"configs/experiment_rdev_{key}_v3.yaml")
    eth = load_experiment("configs/experiment_eth_ft.yaml")
    assert e["name"] == f"cdev_{key}_gpt" and e["models"] == ["gpt61_sol"] and e["include"] == ["dev"]
    assert e["prompt"] == eth["prompt"]  # 和第 10 包兩階段的第二階段同一個提示詞
    for k in ("split_dir", "frames_root", "sampling_template", "frames", "cameras", "task"):
        assert e[k] == v3[k], k
    assert "model_overrides" not in yaml.safe_load(Path(f"configs/experiment_cdev_{key}_gpt.yaml").read_text(encoding="utf-8"))["experiment"]


def test_cascade_dev_config():
    cfg = cascade_dev.load_dev_cfg("configs/cascade_dev.yaml")
    assert [p["key"] for p in cfg["parts"]] == list(KEYS)
    assert all(p["gpt"] == f"configs/experiment_cdev_{p['key']}_gpt.yaml" and p.get("v3") for p in cfg["parts"])


def _recs():
    """4 個零件 × 10 段（i<3 失敗）。Qwen：失敗 P(held)＝0.97；成功奇數 0.8、偶數 0.1。
    GPT：失敗 i＝0 說成功、成功 i＝9 說失敗，其他都對。i＝9 不在乾淨開發集。"""
    out = []
    for k in "abcd":
        for i in range(10):
            fail = i < 3
            ph = 0.97 if fail else (0.8 if i % 2 else 0.1)
            gpt = ("success" if i == 0 else "failure") if fail else ("failure" if i == 9 else "success")
            out.append({"segment_id": f"{k}{i}", "part": k, "truth": "failure" if fail else "success",
                        "v3_text": "failure" if ph > 0.5 else "success",
                        "score_p": {"held_by_gripper": ph, "place_empty_or_part_misplaced": 0.05, "board_moved": 0.0},
                        "gpt": gpt, "gpt_s": 2.0, "gpt_cost": 0.01, "usable": i != 9})
    return out


def test_cascade_rules():
    r = _recs()[3]  # 成功段，P(held)＝0.8：第三版說失敗 → 問 GPT
    assert cascade_dev.cascade(r, "v3") == ("success", True)
    assert cascade_dev.cascade(r, 0.9) == ("success", False) and cascade_dev.cascade(r, 0.5) == ("success", True)
    bad = dict(r, score_p=None, gpt=None)  # 第一階段無效 → 問 GPT；GPT 也無效 → 答案缺
    assert cascade_dev.cascade(bad, 0.9) == (None, True)
    assert cascade_dev.predict(r, "qwen_frozen") == ("success", False) and cascade_dev.predict(r, "gpt") == ("success", True)
    with pytest.raises(ValueError):
        cascade_dev.predict(r, "nope")


def test_best_casc_tau_ties_take_largest():
    recs = [r for r in _recs() if r["usable"]]
    # τ 在 [0.145, 0.9715) 之間兩類平均都一樣（成功段全對、失敗段 8/12）→ 取最大的格點（問 GPT 最少）
    assert cascade_dev.best_casc_tau(recs) == pytest.approx(0.970688)


def test_evaluate_and_select(monkeypatch, tmp_path):
    recs = _recs()
    monkeypatch.setattr(cascade_dev, "load_records", lambda cfg: {k: [r for r in recs if r["part"] == k] for k in "abcd"})
    res = cascade_dev.evaluate({"parts": [{"key": k, "name": k.upper()} for k in "abcd"]})
    C, F = res["clean"]["rows"], res["full"]["rows"]
    assert res["clean"]["n"] == 36 and res["full"]["n"] == 40
    # 全部段：第 10 包規則 32/40、問 GPT 28 段；τ 重選 → 成功段全對、只問失敗段的 GPT
    assert F["casc_v3"]["pooled"]["correct"] == 32 and F["casc_v3"]["pooled"]["n_gpt"] == 28
    assert F["casc_tau"]["pooled"]["correct"] == 36 and F["casc_tau"]["pooled"]["n_gpt"] == 12
    assert F["casc_tau"]["pooled"]["cost_usd"] == pytest.approx(0.12) and F["gpt"]["pooled"]["n_gpt"] == 40
    assert F["qwen_frozen"]["pooled"]["fail_ok"] == 0 and F["casc_frozen"]["pooled"]["n_gpt"] == 0
    # 乾淨段：τ 重選、第 10 包規則、只用 GPT 兩類平均一樣 → 選問 GPT 最少的
    b = C["casc_tau"]["pooled"]["bal"]
    assert b == pytest.approx((8 / 12 + 1) / 2) and C["casc_v3"]["pooled"]["bal"] == pytest.approx(b) and C["gpt"]["pooled"]["bal"] == pytest.approx(b)
    s = res["selection"]
    assert s["pick"] == "casc_tau" and set(s["near"]) == {"casc_tau", "casc_v3", "gpt"} and s["passes"]
    assert s["test_tau"] == pytest.approx(0.970688)
    assert C["casc_tau"]["vs_qwen"]["a_only"] == 8 and C["casc_tau"]["vs_qwen"]["b_only"] == 0
    md = cascade_dev.write_report(res, tmp_path)
    text = Path(md).read_text(encoding="utf-8")
    assert "通過" in text and "留一個零件" in text and (tmp_path / "per_segment.jsonl").exists()
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["selection"]["pick"] == "casc_tau"


def test_select_fails_when_no_gain(monkeypatch):
    recs = _recs()
    for r in recs:
        r["gpt"] = "success"  # GPT 一律答成功 → 兩階段不會比只用 Qwen 好
        r["score_p"] = {"held_by_gripper": 0.97 if r["truth"] == "failure" else 0.1, "place_empty_or_part_misplaced": 0.0,
                        "board_moved": 0.0}
    monkeypatch.setattr(cascade_dev, "load_records", lambda cfg: {k: [r for r in recs if r["part"] == k] for k in "abcd"})
    res = cascade_dev.evaluate({"parts": [{"key": k, "name": k} for k in "abcd"]})
    assert not res["selection"]["passes"]


def test_check_runs(tmp_path):
    good, bad = tmp_path / "g", tmp_path / "b"
    for d, st in ((good, "ok"), (bad, "invalid")):
        d.mkdir()
        (d / "predictions.jsonl").write_text("".join(json.dumps({"segment_id": f"s{i}", "status": "ok" if i else st, "result": "success"})
                                                     + "\n" for i in range(2)), encoding="utf-8")
        (d / "raw.jsonl").write_text("".join(json.dumps({"segment_id": f"s{i}", "cost_usd": 0.005}) + "\n" for i in range(2)),
                                     encoding="utf-8")
    n_bad, msgs = cascade_dev.check_runs([str(good), str(bad)])
    assert n_bad == 1 and "US$0.0100" in msgs[0]
    assert cascade_dev.check_runs([str(tmp_path / "missing")])[0] == 1


# ---------------------------------------------------------------- 測試集（2026-11-02 凍結）

TEST_PARTS = [("rpeg1", "round peg 1", "round peg", "round_peg_1"), ("lgear", "large gear", "gear", "large_gear")]


@pytest.mark.parametrize("key,part,phrase,froot", TEST_PARTS)
def test_final_test_p12_configs(key, part, phrase, froot):
    from assembly_monitor import final

    cfg = final.load_final_config(f"configs/final_test_{key}.yaml")
    assert cfg["frozen"] and cfg["part"] == part and cfg["split_dir"] == f"data/splits/{key}_test100"
    assert cfg["out_name"] == f"final_test_{key}" and cfg["primary"] == "gpt61_sol_ft_eb05_f4"
    m = {x["id"]: x for x in cfg["methods"]}
    assert list(m) == ["majority", "gpt61_sol_ft_eb05_f4", "qwen3vl8b_prob_t0999", "hybrid_prob_gpt", "hybrid_v3_gpt",
                       "qwen3vl8b_v3_eb05_f4"]
    assert (m["hybrid_prob_gpt"]["first"], m["hybrid_prob_gpt"]["second"]) == ("qwen3vl8b_prob_t0999", "gpt61_sol_ft_eb05_f4")
    assert (m["hybrid_v3_gpt"]["first"], m["hybrid_v3_gpt"]["second"]) == ("qwen3vl8b_v3_eb05_f4", "gpt61_sol_ft_eb05_f4")
    assert (m["gpt61_sol_ft_eb05_f4"]["model"], m["qwen3vl8b_prob_t0999"]["model"], m["qwen3vl8b_v3_eb05_f4"]["model"]) == \
        ("gpt61_sol", "qwen3vl_8b_rscore_t0999", "qwen3vl_8b_zoomfix")
    pairs = ((f"configs/experiment_test_{key}_gpt.yaml", "configs/experiment_cdev_sqpeg2_gpt.yaml"),
             (f"configs/experiment_test_{key}_v3.yaml", "configs/experiment_test_sqpeg4_v3.yaml"))
    for path, ref in pairs:
        e, d = load_experiment(path), load_experiment(ref)
        assert e["split_dir"] == cfg["split_dir"] and e["include"] == ["test"] and e["frames_root"] == f"data/derived/frames/insert_{froot}"
        assert e["task"]["part"] == phrase and e["models"] == d["models"]
        for f in ("prompt", "sampling_template", "frames", "cameras"):
            assert e[f] == d[f], f
        assert {k: v for k, v in e["task"].items() if k != "part"} == {k: v for k, v in d["task"].items() if k != "part"}
    for x in cfg["methods"]:
        if x["kind"] == "vlm":
            final.experiment_for_test(cfg, x)
    files = final.locked_files(cfg)
    for f in ("configs/model_reason.yaml", "configs/model_p8.yaml", f"configs/experiment_test_{key}_gpt.yaml",
              f"configs/experiment_test_{key}_v3.yaml", "configs/prompts/ft_generic.txt", "configs/prompts/checklist_fields_v1.txt"):
        assert f in files
    assert final.needed_extractions(cfg) == [{"end_buffer": 0.5, "frames": [4]}]


def test_pool_test_report(monkeypatch, tmp_path):
    from assembly_monitor import final, reason_eval

    ids = ["majority", "gpt61_sol_ft_eb05_f4", "qwen3vl8b_prob_t0999", "hybrid_prob_gpt", "hybrid_v3_gpt", "qwen3vl8b_v3_eb05_f4"]

    def fake_rows(cp):
        k = Path(cp).stem
        segs = [{"segment_id": f"{k}R{i // 2}__seg{i}", "recording_id": f"{k}R{i // 2}",
                 "dataset_result": "failure" if i < 6 else "success"} for i in range(30)]
        res = {"majority": lambda i: "success", "gpt61_sol_ft_eb05_f4": lambda i: "failure" if i < 5 else "success",
               "qwen3vl8b_prob_t0999": lambda i: "failure" if i < 3 or i >= 24 else "success",
               "hybrid_prob_gpt": lambda i: "failure" if i < 3 else "success", "hybrid_v3_gpt": lambda i: "failure" if i < 5 else "success",
               "qwen3vl8b_v3_eb05_f4": lambda i: "failure" if i < 20 else "success"}
        preds = {m: {s["segment_id"]: {"status": "ok", "result": f(i)} for i, s in enumerate(segs)} for m, f in res.items()}
        cfg = {"out_name": k, "part": k, "primary": "gpt61_sol_ft_eb05_f4",
               "methods": [{"id": x, "kind": "majority" if x == "majority" else "vlm"} for x in ids]}
        return cfg, segs, preds

    monkeypatch.setattr(reason_eval, "_final_rows", fake_rows)
    monkeypatch.setattr(final, "method_run_dir", lambda cfg, m: tmp_path / "none")
    monkeypatch.setattr(cascade_dev, "gpt_usage", lambda paths: {"gpt61_sol_ft_eb05_f4": {"n": 60, "n_gpt": 60, "cost_usd": 0.4},
                                                                "hybrid_prob_gpt": {"n": 60, "n_gpt": 18, "cost_usd": 0.12}})
    res = cascade_dev.pool_test(["a.yaml", "b.yaml"], n_boot=200)
    rows = {r["id"]: r for r in res["rows"]}
    g = rows["gpt61_sol_ft_eb05_f4"]
    assert g["pooled"]["correct"] == 58 and g["vs_baseline"] == {"a_only": 16, "b_only": 0, "p_value": pytest.approx(2 / 2 ** 16)}
    assert res["baseline"] == "qwen3vl8b_prob_t0999" and res["holds"]
    md = cascade_dev.write_test_report(res, tmp_path / "out")
    text = Path(md).read_text(encoding="utf-8")
    assert "成立" in text and "18/60" in text and "US$0.120" in text


def test_gpt_usage_reads_raw(monkeypatch, tmp_path):
    from assembly_monitor import final

    cfg = {"methods": [{"id": "majority", "kind": "majority"}, {"id": "g", "kind": "vlm", "model": "gpt61_sol"},
                       {"id": "h", "kind": "cascade"}, {"id": "q", "kind": "vlm", "model": "qwen"}]}
    for mid, rows in (("g", [{"segment_id": "s1", "cost_usd": 0.01}, {"segment_id": "s2", "cost_usd": 0.02}]),
                      ("h", [{"segment_id": "s1", "escalated": True, "cost_usd": 0.01}, {"segment_id": "s2", "escalated": False, "cost_usd": 0.0}]),
                      ("q", [{"segment_id": "s1"}, {"segment_id": "s2"}])):
        (tmp_path / mid).mkdir()
        (tmp_path / mid / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(final, "load_final_config", lambda cp: cfg)
    monkeypatch.setattr(final, "method_run_dir", lambda c, m: tmp_path / m["id"])
    u = cascade_dev.gpt_usage(["x.yaml"])
    assert u["g"] == {"n": 2, "n_gpt": 2, "cost_usd": pytest.approx(0.03)} and u["h"]["n_gpt"] == 1 and u["q"]["n_gpt"] == 0
    assert "majority" not in u
