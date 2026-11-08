"""第 9 包：bolt 4 新測試集的凍結設定。不需要 GPU、網路或 API key。"""

from __future__ import annotations

from pathlib import Path

from assembly_monitor import final, probe, vlm

ROOT = Path(__file__).resolve().parent.parent


def test_final_test_bolt4_config():
    cfg = final.load_final_config(ROOT / "configs/final_test_bolt4.yaml")
    assert cfg["frozen"] and cfg["part"] == "bolt 4" and cfg["out_name"] == "final_test_bolt4"
    assert cfg["split_dir"] == "data/splits/bolt4_test100"
    ids = [m["id"] for m in cfg["methods"]]
    assert ids == ["majority", "qwen3vl8b_v3_eb05_f4", "qwen3vl8b_ft_eb05_f4", "qwen3vl8b_crop_eb05_f4",
                   "qwen3vl8b_checklist_eb05_f4", "probe_bolt4_eb1_f1", "gpt61_sol_ft_eb05_f4", "claude_sonnet55_ft_eb05_f4"]
    assert cfg["primary"] == "qwen3vl8b_v3_eb05_f4"
    m = {x["id"]: x for x in cfg["methods"]}
    # 主方法＝Skill 第三版：清單（逐項回答）＋手腕固定裁切；2x2 的另外三格也在
    assert (m["qwen3vl8b_v3_eb05_f4"]["model"], m["qwen3vl8b_v3_eb05_f4"]["experiment"]) == ("qwen3vl_8b_zoomfix", "configs/experiment_bolt4_checklistf.yaml")
    assert (m["qwen3vl8b_ft_eb05_f4"]["model"], m["qwen3vl8b_ft_eb05_f4"]["experiment"]) == ("qwen3vl_8b", "configs/experiment_bolt4_ft.yaml")
    assert (m["qwen3vl8b_crop_eb05_f4"]["model"], m["qwen3vl8b_crop_eb05_f4"]["experiment"]) == ("qwen3vl_8b_zoomfix", "configs/experiment_bolt4_ft.yaml")
    assert (m["qwen3vl8b_checklist_eb05_f4"]["model"], m["qwen3vl8b_checklist_eb05_f4"]["experiment"]) == ("qwen3vl_8b", "configs/experiment_bolt4_checklistf.yaml")
    files = final.locked_files(cfg)
    for f in ("configs/model_skill2.yaml", "configs/model_p9.yaml", "configs/experiment_bolt4_ft.yaml",
              "configs/experiment_bolt4_checklistf.yaml", "configs/prompts/checklist_fields_v1.txt",
              "configs/prompts/ft_generic.txt", "configs/probe_bolt4.yaml"):
        assert f in files
    assert final.needed_extractions(cfg) == [{"end_buffer": 0.5, "frames": [4]}, {"end_buffer": 1.0, "frames": [1]}]
    for x in cfg["methods"]:
        if x["kind"] == "vlm":
            assert final.experiment_for_test(cfg, x)["split_dir"] == cfg["split_dir"]  # 切分一致（不同就拒絕）


def test_bolt4_experiments_match_usb_and_bnc():
    """bolt 4 的實驗設定：影格、任務文字和 USB 測試集相同（只換零件名稱）；清單提示詞和 BNC 開發集的第三版相同。"""
    usb = vlm.load_experiment(ROOT / "configs/experiment_usb_ft.yaml")
    ft = vlm.load_experiment(ROOT / "configs/experiment_bolt4_ft.yaml")
    cl = vlm.load_experiment(ROOT / "configs/experiment_bolt4_checklistf.yaml")
    bnc_v3 = vlm.load_experiment(ROOT / "configs/experiment_bnc_checklist_fields.yaml")
    for e in (ft, cl):
        assert e["split_dir"].endswith("bolt4_test100") and e["frames_root"].endswith("insert_bolt_4") and e["include"] == ["test"]
        assert e["sampling_template"] == usb["sampling_template"] and e["frames"] == [4] and e["cameras"] == usb["cameras"]
        assert e["task"]["part"] == "bolt"
        assert {k: v for k, v in e["task"].items() if k != "part"} == {k: v for k, v in usb["task"].items() if k != "part"}
    assert ft["prompt"] == usb["prompt"]
    assert cl["prompt"] == bnc_v3["prompt"]  # Skill 第三版的提示詞沒有改
    mc = vlm.load_model_configs([ROOT / "configs/model.yaml", ROOT / "configs/model_p8.yaml",
                                 ROOT / "configs/model_skill2.yaml", ROOT / "configs/model_p9.yaml"])
    c8, c = mc["claude_sonnet55"], mc["claude_sonnet55_8k"]
    assert c["max_tokens"] == 8000 and {k: v for k, v in c.items() if k not in ("max_tokens", "request_timeout")} == \
        {k: v for k, v in c8.items() if k not in ("max_tokens", "request_timeout")}
    for e in (ft, cl):
        assert all(k in mc for k in e["models"])


def test_probe_bolt4_config():
    pc = probe.load_probe_config(ROOT / "configs/probe_bolt4.yaml")
    pu = probe.load_probe_config(ROOT / "configs/probe_usb.yaml")
    assert pc["target_part"] == "bolt 4" and pc["final_experiment_name"] == "final_test_bolt4"
    assert pc["split_dir"].endswith("bolt4_test100")
    for k in ("encoder", "classifier", "sampling_template", "frames", "cameras", "end_buffer"):
        assert pc[k] == pu[k], k
