"""第 7 包：LoRA 微調的資料與設定（不需要 GPU）、多個模型設定檔、adapter 進凍結鎖、做法 A 改良（彙整方式與掃描）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from assembly_monitor import final, finetune, probe, vlm  # noqa: E402
from test_probe import SAMPLING, fake_load_split, world  # noqa: E402,F401


# ---------------------------------------------------------------- 設定檔


def test_repo_finetune_config_and_generic_prompt():
    cfg = finetune.load_finetune_config(ROOT / "configs/finetune_qwen3vl8b.yaml")
    assert cfg["base_model"] == "qwen3vl_8b" and cfg["models"] == "configs/model.yaml" and cfg["n_frames"] == 4
    assert cfg["train"]["epochs"] == 2 and cfg["lora"]["r"] == 16 and cfg["target_part"] == "BNC"
    assert "visual" not in cfg["lora"]["target_modules"] and "language_model" in cfg["lora"]["target_modules"]
    exp = vlm.load_experiment(ROOT / cfg["prompt_experiment"])
    assert exp["sampling_template"] == "uniform_n{n}_sb0.5_eb0.5_jpg" and exp["include"] == ["dev"]
    assert exp["models"] == ["qwen3vl_8b_ft", "qwen3vl_8b"] and exp["frames"] == [4]
    from assembly_monitor.mapping import load_mapping
    from assembly_monitor.schema import load_annotation_config

    acfg = load_annotation_config(ROOT / "configs/annotation.yaml", load_mapping(ROOT / "configs/field_mapping.yaml"))
    vlm.check_task_against_annotation(exp["task"], acfg.reasons, exp["cameras"])
    b = vlm.PromptBuilder.from_experiment(exp, ROOT)
    usb = vlm.PromptBuilder(b.template, b.version, finetune.part_task(exp["task"], "USB", cfg["part_names"]))
    text = usb.instruction(-0.5, 10.0, 12)
    assert "insert the USB connector into its place" in text
    assert "BNC" not in text and "socket" not in text and "twist" not in text  # 不含零件專屬的細節
    assert '{"result": "success"} or {"result": "failure"}' in text and "uncertain" not in text
    # 訓練樣本和 BNC 推論只差零件名稱
    bnc = vlm.PromptBuilder(b.template, b.version, finetune.part_task(exp["task"], "BNC", cfg["part_names"]))
    assert bnc.instruction(-0.5, 10.0, 12).replace("BNC connector", "USB connector") == text


def test_model_train_yaml_merges_without_touching_model_yaml():
    mc = vlm.load_model_configs([ROOT / "configs/model.yaml", ROOT / "configs/model_train.yaml"])
    base, ft = mc["qwen3vl_8b"], mc["qwen3vl_8b_ft"]
    for k in ("hf_id", "quantization", "quant_skip_modules", "dtype", "image_max_side"):
        assert base[k] == ft[k], k  # 推論時和訓練時的影像與量化設定相同
    assert ft["adapter_path"].endswith("qwen3vl8b_lora_r16/final")
    assert mc["dinov2_large"]["type"] == "encoder" and mc["siglip2_b384"]["allow_patterns"]
    with pytest.raises(vlm.VLMConfigError):
        vlm.load_model_configs([ROOT / "configs/model.yaml", ROOT / "configs/model.yaml"])  # key 重複
    assert "qwen3vl_8b_ft" not in vlm.load_model_configs(ROOT / "configs/model.yaml")


def test_adapter_hash_goes_into_run_config(tmp_path):
    cfg = {"type": "transformers", "hf_id": "x", "adapter_path": str(tmp_path / "final")}
    with pytest.raises(vlm.VLMConfigError):
        vlm.resolve_adapter(cfg)
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "final" / "adapter_model.safetensors").write_bytes(b"abc")
    r1 = vlm.resolve_adapter(cfg)
    (tmp_path / "final" / "adapter_model.safetensors").write_bytes(b"abd")
    r2 = vlm.resolve_adapter(cfg)
    assert r1["adapter_sha256"] != r2["adapter_sha256"]  # 重新訓練 → 舊結果不會被當成已跑完
    assert vlm.resolve_adapter({"type": "majority"}) == {"type": "majority"}


# ---------------------------------------------------------------- 小工具


def test_part_labels_and_answers():
    names = finetune.load_finetune_config(ROOT / "configs/finetune_qwen3vl8b.yaml")["part_names"]
    # 索引裡的零件名稱（用空白）
    for raw, label in (("round peg 2", "round peg"), ("square peg 4", "square peg"), ("large gear", "large gear"),
                       ("bolt 4", "bolt"), ("D-SUB", "D-SUB connector"), ("Ethernet", "Ethernet connector"),
                       ("waterproof", "waterproof connector"), ("BNC", "BNC connector"), ("USB", "USB connector")):
        assert finetune.part_label(raw, names) == label, raw
    assert finetune.part_label("round_peg_2", {}) == "round peg"  # 底線寫法也可以
    assert json.loads(finetune.answer_text("failure")) == {"result": "failure"}
    with pytest.raises(ValueError):
        finetune.answer_text("uncertain")


def test_class_weights_schedule_and_smoke_pick():
    s = [{"segment_id": f"s{i}", "result": "failure" if i < 2 else "success"} for i in range(10)]
    w = finetune.class_weights(s)
    assert w == {"success": 10 / 16, "failure": 2.5}
    assert w["failure"] * 2 == pytest.approx(w["success"] * 8)  # 兩類總權重相同
    assert finetune.class_weights(s, "none") == {"success": 1.0, "failure": 1.0}
    assert finetune.n_optimizer_steps(368, 2, 8) == 92
    f = [finetune.lr_factor(i, 10, 2) for i in range(10)]
    assert f[0] == 0.5 and f[1] == 1.0 and f[2] == 1.0 and f[-1] == pytest.approx(0.125) and all(x > 0 for x in f)
    pick = finetune.pick_balanced(s, 4)
    assert sum(1 for x in pick if x["result"] == "failure") == 2 and len(pick) == 4
    assert finetune.pick_balanced(s, 4) == pick  # 固定 seed


def test_first_divergence_and_seq_fill():
    assert finetune.first_divergence([1, 2, 3, 9], [1, 2, 4, 9]) == 2
    with pytest.raises(ValueError):
        finetune.first_divergence([1, 2], [1, 2, 3])
    assert finetune.seq_fill("attention_mask") == 1 and finetune.seq_fill("mm_token_type_ids") == 0


class FakeProcessor:
    """模仿 Qwen 的 chat template（只做文字）。"""

    def __init__(self, think: bool = False, broken: bool = False):
        self.think, self.broken = think, broken

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
        out = ""
        for m in msgs:
            text = "".join(c["text"] for c in m["content"] if c["type"] == "text")
            if m["role"] == "assistant":
                pre = "<think>\n\n</think>\n\n" if self.think else ""
                out += f"<|im_start|>assistant\n{pre}{'X' if self.broken else text}<|im_end|>\n"
            else:
                out += f"<|im_start|>{m['role']}\n{text}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out


def test_answer_suffix_follows_chat_template():
    ans = finetune.answer_text("success")
    assert finetune.answer_suffix(FakeProcessor(), ans) == ans + "<|im_end|>"
    assert finetune.answer_suffix(FakeProcessor(think=True), ans) == "<think>\n\n</think>\n\n" + ans + "<|im_end|>"
    with pytest.raises(finetune.FinetuneConfigError):
        finetune.answer_suffix(FakeProcessor(broken=True), ans)


# ---------------------------------------------------------------- 訓練資料（用做法 A 測試的假資料）


def _ft_cfg(world_cfg, tmp_path):
    exp = {"experiment": {"name": "dev_ft_t", "split_dir": world_cfg["split_dir"], "include": ["dev"],
                          "frames_root": str(Path(world_cfg["frames_root"]) / "insert_BNC"), "sampling_template": SAMPLING,
                          "frames": [4], "cameras": world_cfg["cameras"], "out_dir": str(tmp_path / "exp"),
                          "prompt": {"version": "ft-generic", "template": str(ROOT / "configs/prompts/ft_generic.txt")},
                          "task": vlm.load_experiment(ROOT / "configs/experiment_ft.yaml")["task"]}}
    (tmp_path / "exp_ft.yaml").write_text(yaml.safe_dump(exp, allow_unicode=True), encoding="utf-8")
    cfg = yaml.safe_load((ROOT / "configs/finetune_qwen3vl8b.yaml").read_text(encoding="utf-8"))["finetune"]
    cfg.update(prompt_experiment=str(tmp_path / "exp_ft.yaml"), split_dir=world_cfg["split_dir"], index_dir=world_cfg["index_dir"],
               frames_root=world_cfg["frames_root"], out_dir=str(tmp_path / "ft"), cache_dir=str(tmp_path / "cache"))
    p = tmp_path / "ft.yaml"
    p.write_text(yaml.safe_dump({"finetune": cfg}, allow_unicode=True), encoding="utf-8")
    return finetune.load_finetune_config(p)


def test_build_samples_uses_other_parts_only(world, fake_load_split, tmp_path):
    wcfg, split_rows = world
    cfg = _ft_cfg(wcfg, tmp_path)
    exp = vlm.load_experiment(cfg["prompt_experiment"])
    data = finetune.build_samples(cfg, exp)
    test_recs = {r["recording_id"] for r in split_rows if r["split"] == "test"}
    assert len(data["train"]) == 16 and not data["train_missing"]
    assert all(s["part"] != "BNC" for s in data["train"]) and not ({s["recording_id"] for s in data["train"]} & test_recs)
    assert {s["part"] for s in data["dev"]} == {"BNC"} and len(data["dev"]) == 2
    assert all(len(s["seg"].frames) == 12 for s in data["train"])  # 4 幀 × 3 相機
    assert fake_load_split == [(["dev"], False)]  # 從來沒有讀測試集
    pl = finetune.describe_plan(cfg)
    assert pl["optimizer_steps"] == 2 * 2 and pl["part_labels"] == {"USB": "USB connector", "bolt 4": "bolt"}
    assert pl["example"]["target"] in ('{"result": "success"}', '{"result": "failure"}')
    s0 = data["train"][0]
    paths = finetune.cached_images(cfg, s0, 32)
    assert len(paths) == 12 and all(Path(p).exists() for p in paths)
    img = cv2.imread(paths[0])
    assert max(img.shape[:2]) == 32
    assert finetune.cached_images(cfg, s0, 32) == paths  # 第二次沿用
    sha = finetune.train_sha(cfg, exp, {"hf_id": "x"}, "p", [s["segment_id"] for s in data["train"]])
    assert sha == finetune.train_sha(cfg, exp, {"hf_id": "x"}, "p", [s["segment_id"] for s in reversed(data["train"])])
    assert sha != finetune.train_sha(dict(cfg, train=dict(cfg["train"], lr=2e-4)), exp, {"hf_id": "x"}, "p",
                                     [s["segment_id"] for s in data["train"]])


def test_finetune_config_validation(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("finetune: {name: x}\n", encoding="utf-8")
    with pytest.raises(finetune.FinetuneConfigError):
        finetune.load_finetune_config(p)


# ---------------------------------------------------------------- 做法 A 改良


def test_temporal_modes():
    f = np.zeros((3, 2, 2), np.float32)
    f[0], f[1], f[2] = 1, 2, 4
    f[1, 1] = np.nan  # 缺一張
    assert probe.to_vector(f).shape == (12,)
    lm = probe.to_vector(f, "last_mean")
    assert lm.shape == (8,) and lm[:4].tolist() == [4, 4, 4, 4]
    assert lm[4:6].tolist() == pytest.approx([7 / 3, 7 / 3]) and lm[6:].tolist() == pytest.approx([2.5, 2.5])
    ld = probe.to_vector(f, "last_diff")
    assert ld[4:].tolist() == [3, 3, 3, 3]
    with pytest.raises(probe.ProbeConfigError):
        probe.to_vector(f, "max")


def test_feature_cache_refuses_mixed_encoders(tmp_path):
    enc = probe.FakeEncoder()
    probe.check_feature_cache(tmp_path, enc, {"type": "fake"})
    probe.check_feature_cache(tmp_path, enc, {"type": "fake", "batch_size": 8})  # batch_size 不影響特徵
    with pytest.raises(probe.ProbeConfigError):
        probe.check_feature_cache(tmp_path, enc, {"type": "fake", "pool": "pooler_mean"})
    e = probe.TransformersImageEncoder({"hf_id": "facebook/dinov2-base", "cache_tag": "hr448"})
    assert e.key == "facebook_dinov2_base_hr448"
    assert probe.TransformersImageEncoder({"hf_id": "facebook/dinov2-base"}).key == "facebook_dinov2_base"
    with pytest.raises(probe.ProbeConfigError):
        probe.TransformersImageEncoder({"hf_id": "x", "pool": "max"})


def test_run_probe_temporal_dirs(world, fake_load_split):
    cfg, _ = world
    rep = probe.run_probe(dict(cfg, temporal=["concat", "last_mean"]), write_summary=False)
    got = [(r["model"], r["n_frames"]) for r in rep["results"]]
    assert got == [("probe_fake", 1), ("probe_fake", 4), ("probe_fake_last_mean", 4)]  # 1 幀只有 concat
    rd = Path(cfg["out_dir"]) / "dev_t" / "probe_fake_last_mean" / "f4"
    rc = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
    assert rc["temporal"] == "last_mean" and rc["model_key"] == "probe_fake_last_mean"
    assert "temporal" not in json.loads((Path(cfg["out_dir"]) / "dev_t" / "probe_fake" / "f4" / "run_config.json").read_text())


def test_sweep_selects_by_cross_validation(world, fake_load_split, tmp_path):
    cfg, _ = world
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"probe": cfg}, allow_unicode=True), encoding="utf-8")
    sw = {"probe_sweep": {"name": "pv2", "base": str(tmp_path / "base.yaml"), "experiment_name": "dev_pv2",
                          "end_buffers": [2], "frames": [1, 4], "temporal": ["concat", "last_mean"],
                          "encoders": {"fa": {"type": "fake"}, "fb": {"type": "fake"}},
                          "selected_config": str(tmp_path / "selected.yaml"), "selection_out": str(tmp_path / "sel")}}
    (tmp_path / "sw.yaml").write_text(yaml.safe_dump(sw), encoding="utf-8")
    s = probe.load_sweep_config(tmp_path / "sw.yaml")
    cands = probe.sweep_candidates(s)
    assert [c["name"] for c in cands] == ["pv2_fa_eb2", "pv2_fb_eb2"] and all(c["experiment_name"] == "dev_pv2" for c in cands)
    rep = probe.run_sweep(s)
    assert len(rep["rows"]) == 6 and not rep["errors"]
    b = rep["best"]
    assert b["cv_balanced_accuracy"] == max(r["cv_balanced_accuracy"] for r in rep["rows"])
    sel = probe.load_probe_config(rep["selected_config"])
    assert sel["frames"] == [b["n_frames"]] and sel["temporal"] == [b["temporal"]] and sel["final_experiment_name"] == "final_test_train"
    assert "★" in Path(rep["md"]).read_text(encoding="utf-8")
    assert fake_load_split and all(call == (["dev"], False) for call in fake_load_split)  # 掃描不讀測試集


def test_select_best_tie_breaks():
    rows = [{"order": 1, "cv_balanced_accuracy": 0.8, "cv_failure_recall": 0.7, "n_frames": 8},
            {"order": 2, "cv_balanced_accuracy": 0.8, "cv_failure_recall": 0.7, "n_frames": 4},
            {"order": 3, "cv_balanced_accuracy": 0.8, "cv_failure_recall": 0.6, "n_frames": 1},
            {"order": 4, "cv_balanced_accuracy": 0.79, "cv_failure_recall": 0.9, "n_frames": 1}]
    assert probe.select_best(rows)["order"] == 2
    assert probe.select_best([]) is None


def test_repo_sweep_config():
    s = probe.load_sweep_config(ROOT / "configs/probe_sweep_v2.yaml")
    cands = probe.sweep_candidates(s)
    assert len(cands) == 6
    names = {c["name"] for c in cands}
    assert "probe_v2_dinov2_base_eb0p5" in names and "probe_v2_siglip2_b384_eb1" in names
    for c in cands:
        assert c["target_part"] == "BNC" and f"_eb{c['end_buffer']:g}_" in c["sampling_template"]
    base = probe.load_probe_config(ROOT / "configs/probe_eb05.yaml")
    d2 = next(c for c in cands if c["_encoder_key"] == "dinov2_base")
    assert d2["encoder"]["processor_kwargs"] == base["encoder"]["processor_kwargs"]  # 沿用已算好的特徵


# ---------------------------------------------------------------- 凍結設定（第 7 包的新設定 vs. 已凍結的第 6 包）


def test_frozen_package6_lock_file_list_unchanged(monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = final.load_final_config("configs/final_test.yaml")
    assert cfg["frozen"] is True
    assert [f.replace("\\", "/") for f in final.locked_files(cfg)] == [  # Windows 的路徑分隔是反斜線
        "configs/final_test.yaml", "configs/model.yaml", "configs/experiment.yaml", "configs/prompts/bnc_p1.txt",
        "configs/probe_eb1.yaml", "configs/experiment_e05.yaml", "configs/experiment_r3.yaml", "configs/prompts/bnc_r3.txt"]


def test_final_config_with_adapter_and_probe_temporal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    for f in ("model.yaml", "model_train.yaml", "experiment_ft.yaml", "probe_eb05.yaml"):
        (tmp_path / "configs" / f).write_text((ROOT / "configs" / f).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "configs/prompts").mkdir()
    (tmp_path / "configs/prompts/ft_generic.txt").write_text("x", encoding="utf-8")
    ad = tmp_path / "models/ft/qwen3vl8b_lora_r16/final"
    ad.mkdir(parents=True)
    (ad / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ad / "adapter_model.safetensors").write_bytes(b"w1")
    fc = {"final": {"split_dir": "s", "out_name": "final_test_train", "frozen": True, "primary": "ft",
                    "models": ["configs/model.yaml", "configs/model_train.yaml"],
                    "methods": [{"id": "ft", "kind": "vlm", "experiment": "configs/experiment_ft.yaml", "model": "qwen3vl_8b_ft", "frames": 4},
                                {"id": "pv2", "kind": "probe", "config": "configs/probe_eb05.yaml", "frames": 4, "temporal": "last_mean"}]}}
    (tmp_path / "configs/final_train.yaml").write_text(yaml.safe_dump(fc), encoding="utf-8")
    cfg = final.load_final_config("configs/final_train.yaml")
    files = final.locked_files(cfg)
    assert "configs/model_train.yaml" in files
    assert any(f.endswith("adapter_model.safetensors") for f in files) and any(f.endswith("adapter_config.json") for f in files)
    final.check_or_write_lock(cfg)
    (ad / "adapter_model.safetensors").write_bytes(b"w2")  # 凍結後換掉 adapter → 拒絕
    with pytest.raises(final.FinalLockError):
        final.check_or_write_lock(cfg)
    rd = final.method_run_dir(cfg, cfg["methods"][1])
    assert rd.parts[-3:] == ("final_test_eb05", "probe_dinov2_eb05_last_mean", "f4")
    bad = dict(fc["final"], methods=[dict(fc["final"]["methods"][0], temporal="mean")])
    (tmp_path / "configs/bad.yaml").write_text(yaml.safe_dump({"final": bad}), encoding="utf-8")
    with pytest.raises(final.FinalConfigError):
        final.load_final_config("configs/bad.yaml")


def test_package7_test_config(monkeypatch):
    """第 7 包的測試集設定：不碰第 6 包的輸出、鎖住 adapter 與選到的做法 A 設定。"""
    monkeypatch.chdir(ROOT)
    cfg = final.load_final_config("configs/final_test_train.yaml")
    assert cfg["out_name"] == "final_test_train" and cfg["primary"] == "qwen3vl8b_ft_eb05_f4"
    ids = [m["id"] for m in cfg["methods"]]
    assert ids == ["majority", "qwen3vl8b_ft_eb05_f4", "qwen3vl8b_generic_eb05_f4", "qwen3vl8b_p1_eb05_f4",
                   "probe_eb1_f1", "probe_v2_siglip2_eb05_f8"]
    dirs = {m["id"]: final.method_run_dir(cfg, m) for m in cfg["methods"]}
    for d in dirs.values():
        assert d.parts[:2] == ("experiments", "final_test_train"), d  # 第 6 包的 experiments/final_test* 不會被覆蓋
    assert dirs["probe_eb1_f1"].parts[-2:] == ("probe_dinov2_eb1", "f1")
    assert dirs["probe_v2_siglip2_eb05_f8"].parts[-2:] == ("probe_v2_siglip2_b384_eb0p5_last_mean", "f8")
    assert final.needed_extractions(cfg) == [{"end_buffer": 0.5, "frames": [4, 8]}, {"end_buffer": 1.0, "frames": [1]}]
    files = [f.replace("\\", "/") for f in final.locked_files(cfg)]
    for f in ("configs/final_test_train.yaml", "configs/model.yaml", "configs/model_train.yaml", "configs/experiment_ft.yaml",
              "configs/prompts/ft_generic.txt", "configs/experiment_e05.yaml", "configs/probe_eb1.yaml",
              "configs/probe_v2_selected.yaml", "models/ft/qwen3vl8b_lora_r16/final/adapter_config.json",
              "models/ft/qwen3vl8b_lora_r16/final/adapter_model.safetensors"):
        assert f in files, f
    exp = final.experiment_for_test(cfg, cfg["methods"][1])
    assert exp["include"] == ["test"] and exp["name"] == "final_test_train/qwen3vl8b_ft_eb05_f4"
    sel = probe.load_probe_config("configs/probe_v2_selected.yaml")
    assert sel["frames"] == [8] and sel["temporal"] == ["last_mean"] and sel["encoder"]["hf_id"] == "google/siglip2-base-patch16-384"
