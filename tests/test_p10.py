"""第 10 包：兩階段（本地 Qwen＋Skill 第三版先判，沒答成功才用 GPT）與 Ethernet、square peg 1 新測試集設定。
不需要 GPU、網路或 API key。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from assembly_monitor import final, probe, timing, vlm

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("key,part,split", [("eth", "Ethernet", "ethernet_test100"), ("sqpeg1", "square peg 1", "sqpeg1_test100")])
def test_final_test_p10_configs(key, part, split):
    cfg = final.load_final_config(ROOT / f"configs/final_test_{key}.yaml")
    assert cfg["frozen"] and cfg["part"] == part and cfg["split_dir"] == f"data/splits/{split}"
    assert cfg["primary"] == "hybrid_v3_gpt"
    m = {x["id"]: x for x in cfg["methods"]}
    assert list(m) == ["majority", "hybrid_v3_gpt", "qwen3vl8b_v3_eb05_f4", "gpt61_sol_ft_eb05_f4", "qwen3vl8b_ft_eb05_f4", f"probe_{key}_eb1_f1"]
    h = m["hybrid_v3_gpt"]
    assert h["kind"] == "cascade" and (h["first"], h["second"]) == ("qwen3vl8b_v3_eb05_f4", "gpt61_sol_ft_eb05_f4") and h["frames"] == 0
    assert (m["qwen3vl8b_v3_eb05_f4"]["model"], m["qwen3vl8b_v3_eb05_f4"]["experiment"]) == ("qwen3vl_8b_zoomfix", f"configs/experiment_{key}_checklistf.yaml")
    files = final.locked_files(cfg)
    for f in ("configs/model_skill2.yaml", f"configs/experiment_{key}_ft.yaml", f"configs/experiment_{key}_checklistf.yaml",
              "configs/prompts/checklist_fields_v1.txt", "configs/prompts/ft_generic.txt", f"configs/probe_{key}.yaml"):
        assert f in files
    assert final.needed_extractions(cfg) == [{"end_buffer": 0.5, "frames": [4]}, {"end_buffer": 1.0, "frames": [1]}]
    assert final.method_run_dir(cfg, h) == Path("experiments") / f"final_test_{key}" / "hybrid_v3_gpt" / "cascade" / "f0"
    # 實驗設定：和 bolt 4 相同（只換零件名稱、切分、影格資料夾）；清單提示詞沒改
    b_ft = vlm.load_experiment(ROOT / "configs/experiment_bolt4_ft.yaml")
    b_cl = vlm.load_experiment(ROOT / "configs/experiment_bolt4_checklistf.yaml")
    for e, b in ((vlm.load_experiment(ROOT / f"configs/experiment_{key}_ft.yaml"), b_ft),
                 (vlm.load_experiment(ROOT / f"configs/experiment_{key}_checklistf.yaml"), b_cl)):
        assert e["prompt"] == b["prompt"] and e["sampling_template"] == b["sampling_template"] and e["cameras"] == b["cameras"]
        assert {k: v for k, v in e["task"].items() if k != "part"} == {k: v for k, v in b["task"].items() if k != "part"}
        assert e["split_dir"] == cfg["split_dir"] and e["include"] == ["test"]
    for x in cfg["methods"]:
        if x["kind"] == "vlm":
            final.experiment_for_test(cfg, x)
    pc, pb = probe.load_probe_config(ROOT / f"configs/probe_{key}.yaml"), probe.load_probe_config(ROOT / "configs/probe_bolt4.yaml")
    assert pc["target_part"] == part and pc["final_experiment_name"] == f"final_test_{key}"
    for k in ("encoder", "classifier", "sampling_template", "frames", "cameras"):
        assert pc[k] == pb[k]


def test_bad_cascade_configs(tmp_path):
    p = tmp_path / "f.yaml"
    base = """final:
  split_dir: data/splits/x
  out_name: t
  methods:
    - {id: a, kind: vlm, experiment: e.yaml, model: m, frames: 4}
    - {id: b, kind: majority}
    - %s
"""
    for bad in ("{id: h, kind: cascade, first: a}", "{id: h, kind: cascade, first: a, second: b}",
                "{id: h, kind: cascade, first: a, second: a}", "{id: h, kind: cascade, first: a, second: zz}"):
        p.write_text(base % bad, encoding="utf-8")
        with pytest.raises(final.FinalConfigError):
            final.load_final_config(p)


def _write_run(d: Path, preds: dict[str, tuple[str, str | None]], lat: float, cost: float | None, rc: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "predictions.jsonl", "w", encoding="utf-8") as f:
        for sid, (status, res) in preds.items():
            f.write(json.dumps({"segment_id": sid, "status": status, "errors": [], "warnings": [], "result": res}) + "\n")
    with open(d / "raw.jsonl", "w", encoding="utf-8") as f:
        for sid in preds:
            r = {"segment_id": sid, "latency_s": lat}
            if cost is not None:
                r["cost_usd"] = cost
            f.write(json.dumps(r) + "\n")
    (d / "run_config.json").write_text(json.dumps(rc), encoding="utf-8")


def test_build_cascade(tmp_path, monkeypatch):
    """第一階段答成功就用它；失敗、無效、缺的段都用第二階段；時間相加、費用只算第二階段有用到的段。"""
    monkeypatch.chdir(tmp_path)
    segs = [{"segment_id": f"s{i}", "split": "test"} for i in range(5)]
    from assembly_monitor import split

    monkeypatch.setattr(split, "load_split", lambda *a, **k: segs)
    cfg = {"split_dir": "x", "out_dir": "experiments", "out_name": "t"}
    m1 = {"id": "q", "kind": "vlm", "model": "qwen", "frames": 4}
    m2 = {"id": "g", "kind": "vlm", "model": "gpt", "frames": 4}
    h = {"id": "h", "kind": "cascade", "first": "q", "second": "g", "model": "cascade", "frames": 0}
    _write_run(final.method_run_dir(cfg, m1), {"s0": ("ok", "success"), "s1": ("ok", "failure"), "s2": ("invalid", None),
                                                "s3": ("ok", "success")}, 3.0, None,
               {"n_frames": 4, "sampling_key": "uniform_n4_sb0.5_eb0.5_jpg", "prompt_version": "checklist-fields-v1",
                "metadata": {"load_s": 20.0}})
    _write_run(final.method_run_dir(cfg, m2), {f"s{i}": ("ok", "success" if i % 2 else "failure") for i in range(5)}, 5.0, 0.01,
               {"n_frames": 4, "prompt_version": "ft-generic"})
    st = final.build_cascade(cfg, h, {"q": m1, "g": m2, "h": h})
    assert st == {"n_segments": 5, "n_escalated": 3, "n_missing": 0}  # s1（失敗）、s2（無效）、s4（第一階段缺）
    rd = final.method_run_dir(cfg, h)
    preds = {r["segment_id"]: r for r in map(json.loads, (rd / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    assert preds["s0"]["result"] == "success" and preds["s0"]["stage"] == 1
    assert preds["s1"]["result"] == "success" and preds["s1"]["stage"] == 2 and preds["s1"]["first_result"] == "failure"
    assert preds["s2"]["result"] == "failure" and preds["s4"]["result"] == "failure"
    t = timing.run_timing(rd)
    assert t["model_type"] == "cascade" and t["n_escalated"] == 3 and t["load_s"] == 20.0
    assert t["cost_usd"] == pytest.approx(0.03)  # 費用：3 段用到 GPT（s4 第一階段缺，也算）
    assert t["n_timed"] == 4 and t["total_s"] == pytest.approx(3 + 8 + 8 + 3)  # s4 沒有第一階段時間 → 不計時
    line = timing.timing_lines([{"id": "h", "timing": t}])[2]
    assert "第二階段 3/5 段" in line and "兩階段" in line
    desc = __import__("assembly_monitor.evaluate", fromlist=["describe_run"]).describe_run(rd)
    assert desc["n_frames"] == 4 and desc["prompt"] == "checklist-fields-v1 → ft-generic" and desc["end_buffer"] == "+0.5 s"
