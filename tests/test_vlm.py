"""第 5 包：PromptBuilder、Parser、Runner（續跑、錯誤隔離、格式修正）與設定一致性。不需要 GPU。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from assembly_monitor import vlm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLING = "uniform_n{n}_sb0.5_eb2_jpg"
CAMS = ["hama1", "hama2", "hand"]
REASONS = ["misalignment", "incomplete_insert", "slip_or_drop", "board_lifted", "wrong_contact", "other", "unknown"]


def _task():
    return {"part": "BNC connector", "part_note": "push and twist", "success_definition": "locked",
            "failure_definition": "not locked", "cameras": {c: f"cam {c}" for c in CAMS},
            "failure_reasons": {r: f"def {r}" for r in REASONS}}


def _write_segment(root: Path, sid: str, n: int, duration: float = 10.0, size=(480, 640)) -> None:
    d = root / SAMPLING.format(n=n) / "segments" / sid
    d.mkdir(parents=True, exist_ok=True)
    times = [duration + 2.0] if n == 1 else [-0.5 + (duration + 2.5) * i / (n - 1) for i in range(n)]
    cams = []
    for c in CAMS:
        paths = []
        for k, t in enumerate(times):
            name = f"{c}__{k:02d}__t{t:+08.3f}s.jpg"
            cv2.imwrite(str(d / name), np.full((*size, 3), 40 * (k + 1) % 255, np.uint8))
            paths.append(name)
        cams.append({"camera": c, "frame_paths": paths, "source_times_sec": [round(t, 4) for t in times]})
    start = 1000.0
    meta = {"start_unix": start, "end_unix": start + duration, "duration_s": duration,
            "window_unix": [start - 0.5, start + duration + 2.0], "cameras": cams}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _exp(tmp_path: Path, **kw) -> dict:
    tpl = tmp_path / "tpl.txt"
    tpl.write_text((ROOT / "configs/prompts/bnc_p1.txt").read_text(encoding="utf-8"), encoding="utf-8")
    e = {"name": "t", "split_dir": str(tmp_path / "split"), "include": ["dev"], "frames_root": str(tmp_path / "frames"),
         "sampling_template": SAMPLING, "frames": [1, 4], "cameras": CAMS, "out_dir": str(tmp_path / "exp"),
         "timeout_sec": 30, "repair_retry": True, "evidence_tolerance_sec": 0.05,
         "prompt": {"version": "t1", "template": str(tpl)}, "task": _task()}
    e.update(kw)
    return e


# ---------------------------------------------------------------- parser


def test_extract_json_text_variants():
    assert json.loads(vlm.extract_json_text('```json\n{"result": "success"}\n```')) == {"result": "success"}
    assert json.loads(vlm.extract_json_text('Sure! {"a": "x}y", "b": {"c": 1}} done')) == {"a": "x}y", "b": {"c": 1}}
    assert vlm.extract_json_text("no json here") is None
    assert vlm.extract_json_text('{"unclosed": 1') is None


def test_parse_valid_and_normalized():
    raw = json.dumps({"result": " Failure ", "failure_reason": "Misalignment", "evidence_start_sec": 1.0,
                      "evidence_end_sec": 3.5, "best_camera": "HAND", "visible_evidence": "offset", "confidence": 0.7})
    p = vlm.parse_prediction(raw, segment_id="s", reasons=REASONS, cameras=CAMS, t_range=(-0.5, 12.0))
    assert p["status"] == "ok" and p["warnings"] == []
    assert (p["result"], p["failure_reason"], p["best_camera"], p["confidence"]) == ("failure", "misalignment", "hand", 0.7)
    assert (p["evidence_start_sec"], p["evidence_end_sec"]) == (1.0, 3.5)


def test_parse_never_fills_answers():
    raw = json.dumps({"result": "success", "failure_reason": "misalignment", "evidence_start_sec": -9,
                      "evidence_end_sec": 3, "best_camera": "external_1", "confidence": 80})
    p = vlm.parse_prediction(raw, segment_id="s", reasons=REASONS, cameras=CAMS, t_range=(-0.5, 12.0))
    assert p["status"] == "ok"
    assert p["failure_reason"] is None and p["best_camera"] is None and p["confidence"] is None
    assert p["evidence_start_sec"] is None and p["evidence_end_sec"] is None
    assert set(p["warnings"]) == {"reason_given_for_non_failure", "evidence_out_of_range", "bad_best_camera", "bad_confidence"}
    f = vlm.parse_prediction('{"result": "failure", "failure_reason": "gremlins"}', segment_id="s", reasons=REASONS,
                             cameras=CAMS, t_range=None)
    assert f["status"] == "ok" and f["failure_reason"] is None and "bad_failure_reason" in f["warnings"]


@pytest.mark.parametrize("raw,err", [("I think it failed.", "json_not_found"), ('{"result": success}', "json_decode_error"),
                                     ('{"result": "maybe"}', "bad_result"), ('{"failure_reason": null}', "bad_result")])
def test_parse_invalid(raw, err):
    p = vlm.parse_prediction(raw, segment_id="s", reasons=REASONS, cameras=CAMS, t_range=None)
    assert p["status"] == "invalid" and p["errors"] == [err] and p["result"] is None


# ---------------------------------------------------------------- prompt


def test_prompt_builder_labels_cameras_times_and_uncertain(tmp_path):
    _write_segment(tmp_path / "frames", "R__seg1", 4)
    seg = vlm.load_segment_frames(tmp_path / "frames", SAMPLING.format(n=4), "R__seg1", CAMS)
    assert len(seg.frames) == 12 and [f.camera for f in seg.frames[:3]] == CAMS  # 依時間、再依相機
    assert seg.window == (-0.5, 12.0)
    b = vlm.PromptBuilder((ROOT / "configs/prompts/bnc_p1.txt").read_text(encoding="utf-8"), "p1", _task())
    pr = b.build(seg)
    assert "BNC connector" in pr.text and '"uncertain"' in pr.text and "There are 12 images" in pr.text
    assert '"board_lifted"' in pr.text and '"hand"' in pr.text
    imgs = [c for c in pr.content if c["type"] == "image"]
    labels = [c["text"].strip() for c in pr.content if c["type"] == "text" and c["text"].strip().startswith("Image ")]
    assert len(imgs) == 12 and labels[0] == "Image 1: camera hama1, t = -0.50 s" and labels[-1].endswith("t = +12.00 s")
    assert pr.t_range == (-0.5, 12.0)
    assert b.sha256 == vlm.PromptBuilder(b.template, "p1", _task()).sha256
    assert b.sha256 != vlm.PromptBuilder(b.template, "p2", _task()).sha256


def test_prepare_images_resizes(tmp_path):
    _write_segment(tmp_path / "frames", "R__seg1", 1)
    seg = vlm.load_segment_frames(tmp_path / "frames", SAMPLING.format(n=1), "R__seg1", CAMS)
    paths = vlm.prepare_images(seg.frames, 320, tmp_path / "in")
    assert cv2.imread(paths[0]).shape[:2] == (240, 320)


def test_task_must_match_annotation():
    with pytest.raises(vlm.VLMConfigError):
        vlm.check_task_against_annotation(_task(), REASONS[:-1], CAMS)
    with pytest.raises(vlm.VLMConfigError):
        vlm.check_task_against_annotation(_task(), REASONS, CAMS[:2])


def test_repo_configs_are_consistent():
    from assembly_monitor.mapping import load_mapping
    from assembly_monitor.schema import load_annotation_config

    exp = vlm.load_experiment(ROOT / "configs/experiment.yaml")
    models = vlm.load_model_configs(ROOT / "configs/model.yaml")
    acfg = load_annotation_config(ROOT / "configs/annotation.yaml", load_mapping(ROOT / "configs/field_mapping.yaml"))
    vlm.check_task_against_annotation(exp["task"], list(acfg.reasons), exp["cameras"])
    assert exp["include"] == ["dev"] and {"qwen3vl_8b", "qwen3vl_4b", "smolvlm2", "majority"} <= set(models)
    b = vlm.PromptBuilder.from_experiment(exp, ROOT)
    assert "{" not in b.instruction(-0.5, 12.0, 24).replace('{"result"', "")  # 沒有漏填的佔位符


# ---------------------------------------------------------------- runner


def _segments(*ids):
    return [{"segment_id": i, "dataset_result": "failure" if k % 2 == 0 else "success"} for k, i in enumerate(ids)]


def test_run_one_isolates_errors_repairs_and_resumes(tmp_path):
    for sid in ("A__seg1", "B__seg2", "C__seg3"):
        _write_segment(tmp_path / "frames", sid, 4)
    exp = _exp(tmp_path)
    builder = vlm.PromptBuilder.from_experiment(exp)
    good = json.dumps({"result": "success", "failure_reason": None, "evidence_start_sec": None, "evidence_end_sec": None,
                       "best_camera": "hama1", "visible_evidence": "seated", "confidence": 0.9})
    cfg = {"type": "mock", "script": {"B__seg2": "result: success", "C__seg3": "oom"}, "repair": {"result: success": good}}
    ad = vlm.MockAdapter("mock", cfg)
    spec = vlm.RunSpec(exp=exp, model_key="mock", model_cfg=cfg, n_frames=4)
    segs = _segments("A__seg1", "B__seg2", "C__seg3", "D__seg4")  # D 沒有影格
    st = vlm.run_one(spec, ad, builder, segs, reasons=REASONS)
    rd = spec.run_dir()
    preds = vlm.latest_by_segment(vlm._read_jsonl(rd / "predictions.jsonl"))
    assert preds["A__seg1"]["status"] == "ok" and preds["A__seg1"]["result"] == "failure"
    assert preds["B__seg2"]["status"] == "ok" and preds["B__seg2"]["attempt"] == "repair"
    assert preds["B__seg2"]["first_errors"] == ["json_not_found"]
    assert preds["C__seg3"]["status"] == "error" and preds["C__seg3"]["errors"] == ["oom"]
    assert preds["D__seg4"]["errors"] == ["frames_missing"]
    raws = vlm.latest_by_segment(vlm._read_jsonl(rd / "raw.jsonl"))
    assert raws["B__seg2"]["raw_text"] == "result: success" and raws["B__seg2"]["repair_text"] == good
    assert raws["A__seg1"]["n_images"] == 12 and raws["A__seg1"]["images"][0]["camera"] == "hama1"
    assert (rd / "errors.csv").read_text(encoding="utf-8-sig").count("\n") == 3  # 表頭 + C + D
    assert st["n_ok"] == 2 and st["n_error"] == 2 and st["quick_accuracy"] == 0.5 and st["quick_failure_recall"] == 0.5
    # 續跑：只重試 error 的段；ok 的不再呼叫模型
    ad2 = vlm.MockAdapter("mock", cfg)
    st2 = vlm.run_one(spec, ad2, builder, segs, reasons=REASONS)
    assert sorted(ad2.calls) == ["C__seg3"] and st2["done_before"] == 2


def test_still_invalid_after_repair_is_recorded_not_filled(tmp_path):
    _write_segment(tmp_path / "frames", "A__seg1", 1)
    exp = _exp(tmp_path)
    cfg = {"type": "mock", "default": "garbage"}
    spec = vlm.RunSpec(exp=exp, model_key="mock", model_cfg=cfg, n_frames=1)
    vlm.run_one(spec, vlm.MockAdapter("mock", cfg), vlm.PromptBuilder.from_experiment(exp), _segments("A__seg1"),
                reasons=REASONS)
    p = vlm._read_jsonl(spec.run_dir() / "predictions.jsonl")[-1]
    assert p["status"] == "invalid" and p["attempt"] == "repair" and p["result"] is None


def test_run_config_change_requires_new_name_or_force(tmp_path):
    _write_segment(tmp_path / "frames", "A__seg1", 1)
    exp = _exp(tmp_path)
    cfg = {"type": "mock"}
    spec = vlm.RunSpec(exp=exp, model_key="mock", model_cfg=cfg, n_frames=1)
    vlm.run_one(spec, vlm.MockAdapter("mock", cfg), vlm.PromptBuilder.from_experiment(exp), _segments("A__seg1"), reasons=REASONS)
    exp2 = dict(exp, prompt={**exp["prompt"], "version": "t2"})
    spec2 = vlm.RunSpec(exp=exp2, model_key="mock", model_cfg=cfg, n_frames=1)
    with pytest.raises(vlm.RunConfigMismatch):
        vlm.run_one(spec2, vlm.MockAdapter("mock", cfg), vlm.PromptBuilder.from_experiment(exp2), _segments("A__seg1"), reasons=REASONS)
    vlm.run_one(spec2, vlm.MockAdapter("mock", cfg), vlm.PromptBuilder.from_experiment(exp2), _segments("A__seg1"),
                reasons=REASONS, force=True)
    assert list(spec2.run_dir().glob("predictions.*.bak.jsonl"))


def test_run_experiment_dev_only_and_summary(tmp_path, monkeypatch):
    import assembly_monitor.split as split_mod

    for sid in ("A__seg1", "B__seg2"):
        for n in (1, 4):
            _write_segment(tmp_path / "frames", sid, n)
    exp = _exp(tmp_path)
    seen = {}

    def fake_load_split(split_dir, include=("dev",), allow_test=False):
        seen["include"], seen["allow_test"] = list(include), allow_test
        return _segments("B__seg2", "A__seg1")

    monkeypatch.setattr(split_mod, "load_split", fake_load_split)
    models = {"mock": {"type": "mock"}, "majority": {"type": "majority", "result": "success"}}
    res = vlm.run_experiment(exp=exp, model_cfgs=models, model_keys=["majority", "mock"], frames=[1, 4], reasons=REASONS)
    assert seen == {"include": ["dev"], "allow_test": False}
    md = Path(res["summary"]["md"]).read_text(encoding="utf-8")
    assert "| majority | 不看影格 |" in md and "| mock | 1 |" in md and "| mock | 4 |" in md
    rows = {(r["model"], r["n_frames"]): r for r in vlm.collect_summary_rows(exp, _segments("A__seg1", "B__seg2"))}
    assert rows[("majority", 0)]["quick_accuracy"] == 0.5 and rows[("mock", 4)]["n_ok"] == 2
    with pytest.raises(vlm.VLMConfigError):
        vlm.run_experiment(exp=exp, model_cfgs=models, model_keys=["nope"], frames=[1], reasons=REASONS)


def test_locked_test_set_is_refused(tmp_path):
    from assembly_monitor.split import LockedTestSetError

    exp = _exp(tmp_path, include=["test"])
    with pytest.raises(LockedTestSetError):
        vlm.run_experiment(exp=exp, model_cfgs={"mock": {"type": "mock"}}, model_keys=["mock"], frames=[1], reasons=REASONS)


def test_load_error_of_one_model_does_not_stop_others(tmp_path, monkeypatch):
    import assembly_monitor.split as split_mod

    _write_segment(tmp_path / "frames", "A__seg1", 1)
    exp = _exp(tmp_path, frames=[1])
    monkeypatch.setattr(split_mod, "load_split", lambda *a, **k: _segments("A__seg1"))

    class Broken(vlm.MockAdapter):
        def load(self):
            raise RuntimeError("CUDA out of memory while loading")

    real = vlm.make_adapter
    monkeypatch.setattr(vlm, "make_adapter", lambda k, c: Broken(k, c) if k == "big" else real(k, c))
    res = vlm.run_experiment(exp=exp, model_cfgs={"big": {"type": "mock"}, "mock": {"type": "mock"}},
                             model_keys=["big", "mock"], frames=[1], reasons=REASONS)
    assert res["results"][0]["model"] == "big" and "load_error" in res["results"][0]
    md = Path(res["summary"]["md"]).read_text(encoding="utf-8")
    assert "載入失敗：oom" in md and "| mock | 1 |" in md


def test_config_change_skips_only_that_combination(tmp_path, monkeypatch):
    import assembly_monitor.split as split_mod

    for n in (1, 4):
        _write_segment(tmp_path / "frames", "A__seg1", n)
    monkeypatch.setattr(split_mod, "load_split", lambda *a, **k: _segments("A__seg1"))
    exp = _exp(tmp_path)
    models = {"mock": {"type": "mock"}}
    vlm.run_experiment(exp=exp, model_cfgs=models, model_keys=["mock"], frames=[1], reasons=REASONS)
    exp2 = dict(exp, prompt={**exp["prompt"], "version": "t2"})
    res = vlm.run_experiment(exp=exp2, model_cfgs=models, model_keys=["mock"], frames=[1, 4], reasons=REASONS)
    assert [("config_mismatch" in r) for r in res["results"]] == [True, False]
    assert "設定已改" in Path(res["summary"]["md"]).read_text(encoding="utf-8")


def test_parse_strips_thinking_blocks():
    raw = '<think>\nThe connector {maybe} looks seated...\n</think>\n\n{"result": "success", "confidence": 0.8}'
    p = vlm.parse_prediction(raw, segment_id="s", reasons=REASONS, cameras=CAMS, t_range=None)
    assert p["status"] == "ok" and p["result"] == "success"
    cut = vlm.parse_prediction("<think>still thinking {", segment_id="s", reasons=REASONS, cameras=CAMS, t_range=None)
    assert cut["status"] == "invalid" and cut["errors"] == ["unfinished_thinking"]


def test_qwen35_configs():
    models = vlm.load_model_configs(ROOT / "configs/model.yaml")
    for k in ("qwen35_9b", "qwen35_4b"):
        assert models[k]["model_class"] == "AutoModelForMultimodalLM"
        assert models[k]["chat_template_kwargs"] == {"enable_thinking": False}
    # 既有模型的設定不能變，否則已跑完的開發集結果會被判定「設定已改」
    assert "model_class" not in models["qwen3vl_8b"] and "chat_template_kwargs" not in models["qwen3vl_8b"]


def test_finished_models_are_not_loaded_again(tmp_path, monkeypatch):
    import assembly_monitor.split as split_mod

    for n in (1, 4):
        _write_segment(tmp_path / "frames", "A__seg1", n)
    monkeypatch.setattr(split_mod, "load_split", lambda *a, **k: _segments("A__seg1"))
    exp = _exp(tmp_path)
    models = {"mock": {"type": "mock"}, "new": {"type": "mock"}}
    vlm.run_experiment(exp=exp, model_cfgs=models, model_keys=["mock"], frames=[1, 4], reasons=REASONS)
    loaded = []

    class Counting(vlm.MockAdapter):
        def load(self):
            loaded.append(self.key)
            super().load()

    monkeypatch.setattr(vlm, "make_adapter", lambda k, c: Counting(k, c))
    res = vlm.run_experiment(exp=exp, model_cfgs=models, model_keys=["mock", "new"], frames=[1, 4], reasons=REASONS)
    assert loaded == ["new"] and res["results"][0] == {"model": "mock", "skipped": "complete"}
    md = Path(res["summary"]["md"]).read_text(encoding="utf-8")
    assert "| mock | 1 |" in md and "| new | 4 |" in md


def test_download_plan_reports_what_is_missing(monkeypatch):
    import sys
    import types

    huggingface_hub = types.ModuleType("huggingface_hub")  # 雲端測試環境沒有裝，用假的模組
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    sib = lambda name, size: types.SimpleNamespace(rfilename=name, size=size)  # noqa: E731
    infos = {"org/old": [sib("a.safetensors", 4e9), sib("config.json", 1e3)],
             "org/new": [sib("a.safetensors", 9e9), sib("config.json", 1e3)]}

    class FakeApi:
        def model_info(self, repo, revision=None, files_metadata=False):
            return types.SimpleNamespace(siblings=infos[repo], sha="abc123", card_data=types.SimpleNamespace(license="apache-2.0"))

    huggingface_hub.HfApi = FakeApi
    huggingface_hub.try_to_load_from_cache = lambda repo, fn, revision=None: f"/cache/{repo}/{fn}" if repo == "org/old" else None
    cfgs = {"old": {"type": "transformers", "hf_id": "org/old"}, "new": {"type": "encoder", "hf_id": "org/new"},
            "majority": {"type": "majority"}}
    plan = {p["key"]: p for p in vlm.model_download_plan(cfgs, ["old", "new", "majority"])}
    assert set(plan) == {"old", "new"}
    assert plan["old"]["missing_gb"] == 0 and plan["new"]["missing_gb"] == 9.0 and plan["new"]["size_gb"] == 9.0


def test_prompt_v2_config_is_consistent_and_differs_from_v1():
    from assembly_monitor.mapping import load_mapping
    from assembly_monitor.schema import load_annotation_config

    acfg = load_annotation_config(ROOT / "configs/annotation.yaml", load_mapping(ROOT / "configs/field_mapping.yaml"))
    models = vlm.load_model_configs(ROOT / "configs/model.yaml")
    e1 = vlm.load_experiment(ROOT / "configs/experiment.yaml")
    e2 = vlm.load_experiment(ROOT / "configs/experiment_p2.yaml")
    vlm.check_task_against_annotation(e2["task"], list(acfg.reasons), e2["cameras"])
    assert e2["include"] == ["dev"] and e2["name"] != e1["name"] and set(e2["models"]) <= set(models)
    # 只改提示詞：切分、取樣、相機、幀數和第 1 版相同，結果才能直接比較
    for k in ("split_dir", "frames_root", "sampling_template", "frames", "cameras", "timeout_sec", "repair_retry"):
        assert e2[k] == e1[k], k
    b1, b2 = vlm.PromptBuilder.from_experiment(e1, ROOT), vlm.PromptBuilder.from_experiment(e2, ROOT)
    assert b2.version == "bnc-p2" and b2.sha256 != b1.sha256
    text = b2.instruction(-0.5, 12.0, 24)
    assert "{" not in text.replace('{"result"', "") and "Image 24" in text and "NOT visible" in text


def test_model_keys_default_from_experiment():
    from assembly_monitor.cli import DEFAULT_VLM_MODELS, _model_keys

    assert _model_keys("a, b", {"models": ["x"]}) == ["a", "b"]
    assert _model_keys(None, {"models": ["qwen35_9b", "qwen3vl_8b"]}) == ["qwen35_9b", "qwen3vl_8b"]
    assert _model_keys(None, {}) == DEFAULT_VLM_MODELS.split(",")


def test_summary_counts_false_alarms_and_balanced_accuracy(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    segs = [{"segment_id": f"s{i}", "dataset_result": r} for i, r in enumerate(["failure", "failure", "success", "success", "success"])]
    preds = [("s0", "failure"), ("s1", "uncertain"), ("s2", "failure"), ("s3", "success"), ("s4", "success")]
    with (rd / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for sid, res in preds:
            fh.write(json.dumps({"segment_id": sid, "status": "ok", "result": res}) + "\n")
    s = vlm.summarize_run(rd, segs)
    assert (s["n_caught"], s["n_failures"], s["n_false_alarms"], s["n_success_correct"], s["n_correct"]) == (1, 2, 1, 2, 3)
    assert s["quick_balanced_accuracy"] == round((1 / 2 + 2 / 3) / 2, 3)


def test_compare_prompt_versions(tmp_path, monkeypatch):
    import yaml

    import assembly_monitor.split as split_mod

    _write_segment(tmp_path / "frames", "A__seg1", 1)
    _write_segment(tmp_path / "frames", "B__seg2", 1)
    monkeypatch.setattr(split_mod, "load_split", lambda *a, **k: _segments("A__seg1", "B__seg2"))
    models = {"mock": {"type": "mock"}, "majority": {"type": "majority", "result": "success"}}
    paths = []
    for name, ver, keys in (("v1", "t1", ["majority", "mock"]), ("v2", "t2", ["mock"])):
        exp = _exp(tmp_path, name=name, frames=[1], prompt={"version": ver, "template": str(tmp_path / "tpl.txt")})
        vlm.run_experiment(exp=exp, model_cfgs=models, model_keys=keys, frames=[1], reasons=REASONS)
        p = tmp_path / f"experiment_{name}.yaml"
        p.write_text(yaml.safe_dump({"experiment": exp}, allow_unicode=True), encoding="utf-8")
        paths.append(p)
    locked = tmp_path / "experiment_final.yaml"
    locked.write_text(yaml.safe_dump({"experiment": _exp(tmp_path, name="final", include=["test"])}), encoding="utf-8")
    rep = vlm.compare_experiments(paths + [locked])
    md = Path(rep["md"]).read_text(encoding="utf-8")
    assert "| mock | 1 | t1 |" in md and "| mock | 1 | t2 |" in md
    assert md.count("| majority |") == 1 and "含測試集" in md


def test_gemma4_configs():
    models = vlm.load_model_configs(ROOT / "configs/model.yaml")
    for k in ("gemma4_12b", "gemma4_e4b"):
        c = models[k]
        assert c["model_class"] == "AutoModelForMultimodalLM" and c["quantization"] == "4bit" and "lm_head" in c["quant_skip_modules"]
        assert c["chat_template_kwargs"] == {"enable_thinking": False} and c["license"] == "Apache-2.0"
    from assembly_monitor.cli import DEFAULT_VLM_MODELS

    assert {"gemma4_12b", "gemma4_e4b"} <= set(DEFAULT_VLM_MODELS.split(","))


def test_reasoning_outputs_are_parsed_from_the_last_answer():
    rr = dict(segment_id="s", reasons=REASONS, cameras=CAMS, t_range=None)
    txt = ("1. Final state: Image 24 shows {the connector} on the socket.\n2. What happened: ...\n4. Decision: success\n"
           '{"result": "success", "failure_reason": null, "confidence": 0.7}')
    assert vlm.parse_prediction(txt, **rr)["result"] == "success"
    # 思考裡的例子 JSON 不算答案，取最後一個有 result 的物件
    txt2 = 'thought\nMaybe {"result": "failure"}? No, it stays.\n{"result": "success", "confidence": 0.6}'
    assert vlm.parse_prediction(txt2, **rr)["result"] == "success"
    gem = '<|channel>thought\nThe connector is tilted {x}\n<channel|>{"result": "failure", "failure_reason": "misalignment"}'
    p = vlm.parse_prediction(gem, **rr)
    assert p["status"] == "ok" and p["failure_reason"] == "misalignment"
    assert vlm.parse_prediction("thought\nThe gripper opens and", **rr)["errors"] == ["unfinished_thinking"]
    assert vlm.parse_prediction("<|channel>thought\nstill going", **rr)["errors"] == ["unfinished_thinking"]
    assert vlm.parse_prediction("no json here", **rr)["errors"] == ["json_not_found"]


def test_model_overrides_change_only_that_experiment(tmp_path):
    base = {"type": "mock", "max_new_tokens": 384}
    exp = _exp(tmp_path, model_overrides={"m": {"max_new_tokens": 4096, "chat_template_kwargs": {"enable_thinking": True}}})
    eff = vlm.effective_model_cfg(exp, "m", base)
    assert eff["max_new_tokens"] == 4096 and base["max_new_tokens"] == 384
    assert vlm.effective_model_cfg(exp, "other", base) is base
    b = vlm.PromptBuilder.from_experiment(exp)
    h1 = vlm.run_config(vlm.RunSpec(exp=exp, model_key="m", model_cfg=base, n_frames=8), b)["config_sha256"]
    h2 = vlm.run_config(vlm.RunSpec(exp=exp, model_key="m", model_cfg=eff, n_frames=8), b)["config_sha256"]
    assert h1 != h2


def test_preregistered_experiments_differ_from_v1_only_where_stated():
    from assembly_monitor.mapping import load_mapping
    from assembly_monitor.schema import load_annotation_config

    acfg = load_annotation_config(ROOT / "configs/annotation.yaml", load_mapping(ROOT / "configs/field_mapping.yaml"))
    models = vlm.load_model_configs(ROOT / "configs/model.yaml")
    e1 = vlm.load_experiment(ROOT / "configs/experiment.yaml")
    b1 = vlm.PromptBuilder.from_experiment(e1, ROOT)
    names = set()
    for f, eb in (("experiment_r1.yaml", "2"), ("experiment_r2.yaml", "2"), ("experiment_r3.yaml", "0.5"), ("experiment_e1.yaml", "1"),
                  ("experiment_e05.yaml", "0.5")):
        e = vlm.load_experiment(ROOT / "configs" / f)
        vlm.check_task_against_annotation(e["task"], list(acfg.reasons), e["cameras"])
        assert e["task"] == e1["task"] and e["include"] == ["dev"] and set(e["models"]) <= set(models), f
        assert e["split_dir"] == e1["split_dir"] and e["cameras"] == e1["cameras"]
        assert f"_eb{eb}_" in e["sampling_template"]
        assert set(e.get("model_overrides") or {}) <= set(e["models"])
        b = vlm.PromptBuilder.from_experiment(e, ROOT)
        assert "{" not in b.instruction(-0.5, 12.0, 24).replace('{"result"', "")
        names.add(e["name"])
        if f.startswith("experiment_e"):
            assert b.sha256 == b1.sha256 and "model_overrides" not in e   # 只改最後一張的時間
        else:
            assert b.sha256 != b1.sha256
    assert len(names) == 5 and e1["name"] not in names
    # R3＝E0.5（第 1 版提示詞、+0.5 秒）再加一段「對準時接頭下墜」：其他設定完全相同，差別只在提示詞
    e05, r3 = vlm.load_experiment(ROOT / "configs/experiment_e05.yaml"), vlm.load_experiment(ROOT / "configs/experiment_r3.yaml")
    skip = ("name", "prompt", "frames")
    assert {k: v for k, v in r3.items() if k not in skip} == {k: v for k, v in e05.items() if k not in skip}
    t1 = (ROOT / "configs/prompts/bnc_p1.txt").read_text(encoding="utf-8")
    t3 = (ROOT / "configs/prompts/bnc_r3.txt").read_text(encoding="utf-8")
    assert t3.startswith(t1.split("If the images do not let you decide")[0]) and "drops down onto it" in t3 and len(t3) > len(t1)
    r2 = vlm.load_experiment(ROOT / "configs/experiment_r2.yaml")
    assert r2["prompt"]["template"] == e1["prompt"]["template"]   # R2 只打開思考模式
    assert all(o["chat_template_kwargs"] == {"enable_thinking": True} for o in r2["model_overrides"].values())
