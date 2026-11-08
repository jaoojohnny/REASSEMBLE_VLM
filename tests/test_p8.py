"""第 8 包：商業模型 API（假的用戶端）、放大鏡 Skill、執行時間、USB 測試集設定。不需要 GPU、網路或 API key。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from assembly_monitor import api_models, final, skill, timing, vlm  # noqa: E402
from test_vlm import CAMS, REASONS, _exp, _segments, _write_segment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 設定


def test_repo_configs_p8():
    mc = vlm.load_model_configs([ROOT / "configs/model.yaml", ROOT / "configs/model_p8.yaml"])
    assert mc["claude_sonnet55"]["type"] == "anthropic" and mc["gpt61_sol"]["type"] == "openai"
    z, base = mc["qwen3vl_8b_zoom"], mc["qwen3vl_8b"]
    assert z["skill"] == "zoom"
    assert {k: v for k, v in z.items() if not k.startswith(("skill", "zoom", "locate"))} == base  # 只多了 Skill 的設定
    for k in ("claude_sonnet55", "gpt61_sol"):
        assert mc[k]["image_max_side"] == base["image_max_side"]  # 和 Qwen 同樣的影格

    bnc = vlm.load_experiment(ROOT / "configs/experiment_bnc_p8.yaml")
    usb = vlm.load_experiment(ROOT / "configs/experiment_usb_ft.yaml")
    ft = vlm.load_experiment(ROOT / "configs/experiment_ft.yaml")
    for e in (bnc, usb):
        assert e["prompt"] == ft["prompt"] and e["sampling_template"] == ft["sampling_template"] and e["frames"] == [4]
    assert bnc["task"] == ft["task"] and bnc["include"] == ["dev"] and bnc["split_dir"].endswith("bnc_alldev")
    assert usb["task"]["part"] == "USB connector" and {k: v for k, v in usb["task"].items() if k != "part"} == \
        {k: v for k, v in ft["task"].items() if k != "part"}
    assert usb["frames_root"].endswith("insert_USB") and usb["split_dir"].endswith("usb_test100")


def test_final_test_usb_config():
    cfg = final.load_final_config(ROOT / "configs/final_test_usb.yaml")
    assert cfg["frozen"] and cfg["part"] == "USB" and cfg["out_name"] == "final_test_usb"
    assert cfg["primary"] == "qwen3vl8b_ft_eb05_f4"
    ids = [m["id"] for m in cfg["methods"]]
    assert ids == ["majority", "qwen3vl8b_ft_eb05_f4", "probe_usb_eb1_f1", "claude_sonnet55_ft_eb05_f4", "gpt61_sol_ft_eb05_f4"]
    assert not any("zoom" in m.get("model", "") for m in cfg["methods"])  # Skill 不進這次測試集
    files = final.locked_files(cfg)
    for f in ("configs/model_p8.yaml", "configs/experiment_usb_ft.yaml", "configs/probe_usb.yaml",
              "configs/prompts/ft_generic.txt"):
        assert f in files
    assert final.needed_extractions(cfg) == [{"end_buffer": 0.5, "frames": [4]}, {"end_buffer": 1.0, "frames": [1]}]
    from assembly_monitor import probe

    pc = probe.load_probe_config(ROOT / "configs/probe_usb.yaml")
    eb1 = probe.load_probe_config(ROOT / "configs/probe_eb1.yaml")
    assert pc["target_part"] == "USB" and pc["encoder"] == eb1["encoder"] and pc["classifier"] == eb1["classifier"]
    assert pc["sampling_template"] == eb1["sampling_template"] and pc["final_experiment_name"] == "final_test_usb"


def test_final_majority_uses_final_split(tmp_path):
    """第 8 包修正：majority 用預設 experiment.yaml（BNC 切分），測試段卻要是凍結設定的切分（USB）。"""
    cfg = final.load_final_config(ROOT / "configs/final_test_usb.yaml")
    maj = next(m for m in cfg["methods"] if m["kind"] == "majority")
    assert vlm.load_experiment(ROOT / maj["experiment"])["split_dir"] != cfg["split_dir"]  # 預設設定是 BNC
    e = final.experiment_for_test(cfg, maj)
    assert e["split_dir"] == cfg["split_dir"] and e["include"] == ["test"] and e["name"] == "final_test_usb/majority"
    q = next(m for m in cfg["methods"] if m["id"] == "qwen3vl8b_ft_eb05_f4")
    assert final.experiment_for_test(cfg, q)["split_dir"] == cfg["split_dir"]
    bad = dict(cfg, split_dir="data/splits/other")
    with pytest.raises(final.FinalConfigError):
        final.experiment_for_test(bad, q)  # VLM 方法的切分和凍結設定不同 → 拒絕
    assert final.experiment_for_test(bad, maj)["split_dir"] == "data/splits/other"


def test_timing_lines_untimed_rows():
    rows = [{"id": "p", "timing": {"model_type": "probe", "n_timed": 0, "where": "GPU"}},
            {"id": "c", "timing": {"model_type": "anthropic", "n_timed": 0, "where": "anthropic API（含網路）"}},
            {"id": "m", "timing": {"model_type": "majority", "n_timed": 0, "where": "不需計算"}}]
    lines = timing.timing_lines(rows)
    assert "特徵在快取裡" in lines[2] and "API 呼叫失敗" in lines[3] and "不需計算" in lines[4]


def test_bad_api_and_skill_configs(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("models:\n  a: {type: anthropic}\n", encoding="utf-8")
    with pytest.raises(vlm.VLMConfigError):
        vlm.load_model_configs(p)
    p.write_text("models:\n  a: {type: openai, model_id: x, skill: zoom}\n", encoding="utf-8")
    with pytest.raises(vlm.VLMConfigError):
        vlm.load_model_configs(p)


# ---------------------------------------------------------------- API（假的用戶端）


def test_load_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("P8_TEST_KEY", raising=False)
    monkeypatch.setenv("P8_KEEP", "old")
    f = tmp_path / ".env"
    f.write_text("# 註解\nP8_TEST_KEY = 'abc123'\nP8_KEEP=new\nEMPTY=\nnot a line\n", encoding="utf-8")
    names = api_models.load_env_file(f)
    import os

    assert names == ["P8_TEST_KEY", "P8_KEEP"] and os.environ["P8_TEST_KEY"] == "abc123"
    assert os.environ["P8_KEEP"] == "old"  # 已有的不覆蓋
    assert api_models.load_env_file(tmp_path / "none.env") == []


def test_missing_key_is_a_load_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ad = api_models.AnthropicAdapter("c", {"type": "anthropic", "model_id": "m", "env_file": str(tmp_path / "x.env")})
    with pytest.raises(vlm.VLMConfigError):
        ad.load()


class _FakeAnthropic:
    def __init__(self, text='{"result": "failure"}', exc=None):
        self.calls = []
        self.text, self.exc = text, exc
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)], model=kw["model"],
                               stop_reason="end_turn", usage=SimpleNamespace(input_tokens=3000, output_tokens=20))


class _FakeOpenAI:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content='{"result": "success"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model=kw["model"],
                               usage=SimpleNamespace(prompt_tokens=2000, completion_tokens=300,
                                                     completion_tokens_details=SimpleNamespace(reasoning_tokens=250)))


def _prompt(tmp_path):
    _write_segment(tmp_path / "frames", "A__seg1", 4)
    exp = _exp(tmp_path)
    seg = vlm.load_segment_frames(exp["frames_root"], exp["sampling_template"].format(n=4), "A__seg1", CAMS)
    paths = vlm.prepare_images(seg.frames, 512, tmp_path / "in")
    return vlm.PromptBuilder.from_experiment(exp).build(seg, paths)


def test_anthropic_adapter_sends_same_images_and_estimates_cost(tmp_path):
    prompt = _prompt(tmp_path)
    cfg = {"type": "anthropic", "model_id": "claude-x", "max_tokens": 100, "price_in_per_mtok": 2.0, "price_out_per_mtok": 10.0}
    ad = api_models.AnthropicAdapter("c", cfg)
    ad.client = _FakeAnthropic()
    out = ad.predict(prompt)
    call = ad.client.calls[0]
    blocks = call["messages"][0]["content"]
    imgs = [b for b in blocks if b["type"] == "image"]
    assert len(imgs) == 12 and imgs[0]["source"]["media_type"] == "image/jpeg" and call["model"] == "claude-x"
    assert [b["text"] for b in blocks if b["type"] == "text"][0] == prompt.content[0]["text"]
    assert out.text == '{"result": "failure"}' and out.input_tokens == 3000
    assert out.extra["cost_usd"] == pytest.approx((3000 * 2 + 20 * 10) / 1e6)


def test_openai_adapter_uses_data_urls_and_counts_reasoning(tmp_path):
    prompt = _prompt(tmp_path)
    ad = api_models.OpenAIAdapter("g", {"type": "openai", "model_id": "gpt-x", "image_detail": "high"})
    ad.client = _FakeOpenAI()
    out = ad.predict(prompt)
    parts = ad.client.calls[0]["messages"][0]["content"]
    urls = [p for p in parts if p["type"] == "image_url"]
    assert len(urls) == 12 and urls[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert urls[0]["image_url"]["detail"] == "high" and out.extra["reasoning_tokens"] == 250
    assert out.extra["cost_usd"] is None  # 沒有單價就不估


def test_auth_error_stops_only_that_model(tmp_path, monkeypatch):
    import assembly_monitor.split as split_mod

    class AuthenticationError(Exception):
        pass

    _write_segment(tmp_path / "frames", "A__seg1", 4)
    _write_segment(tmp_path / "frames", "B__seg2", 4)
    exp = _exp(tmp_path, frames=[4])
    monkeypatch.setattr(split_mod, "load_split", lambda *a, **k: _segments("A__seg1", "B__seg2"))
    fake = _FakeAnthropic(exc=AuthenticationError("invalid x-api-key"))

    def mk(key, cfg):
        if cfg["type"] == "anthropic":
            ad = api_models.AnthropicAdapter(key, cfg)
            ad.load = lambda: setattr(ad, "client", fake)
            return ad
        return vlm.MockAdapter(key, cfg)

    monkeypatch.setattr(vlm, "make_adapter", mk)
    res = vlm.run_experiment(exp=exp, model_cfgs={"claude": {"type": "anthropic", "model_id": "m"}, "mock": {"type": "mock"}},
                             model_keys=["claude", "mock"], frames=[4], reasons=REASONS)
    assert len(fake.calls) == 1  # 第一段就停，不會每段都試
    assert any(r.get("model") == "claude" and "load_error" in r for r in res["results"])
    md = Path(res["summary"]["md"]).read_text(encoding="utf-8")
    assert "停止" in md and "| mock | 4 |" in md


# ---------------------------------------------------------------- 放大鏡 Skill


def test_parse_boxes_formats():
    cams = ["hama1", "hama2", "hand"]
    raw = 'ok ```json\n{"boxes": {"hama1": [100, 200, 300, 400], "hama2": null, "hand": [500, 100, 400, 300]}}\n```'
    b = skill.parse_boxes(raw, cams, {})
    assert b["hama1"] == [0.1, 0.2, 0.3, 0.4] and b["hama2"] is None and b["hand"] == [0.4, 0.1, 0.5, 0.3]
    px = skill.parse_boxes('{"boxes": {"hama1": [0, 0, 1024, 768]}}', cams, {"hama1": (2048, 1536)})
    assert px["hama1"] == [0.0, 0.0, 0.5, 0.5]  # 像素座標換算
    assert skill.parse_boxes("no json", cams, {}) == {c: None for c in cams}
    assert skill.parse_boxes('{"boxes": {"hama1": [1, 2, 3]}}', cams, {})["hama1"] is None


def test_crop_box_expands_and_stays_inside():
    assert skill.crop_box(640, 480, [0.0, 0.0, 0.05, 0.05]) == (0, 0, 120, 120)  # 至少短邊的 25%，貼著左上角
    l, t, r, b = skill.crop_box(640, 480, [0.4, 0.4, 0.6, 0.6], margin=0.5)
    assert r - l == pytest.approx(256, abs=1) and b - t == pytest.approx(192, abs=1)
    l, t, r, b = skill.crop_box(640, 480, [0.9, 0.9, 1.0, 1.0])
    assert r == 640 and b == 480 and l >= 0 and t >= 0


def test_zoom_skill_adds_zoomed_views(tmp_path, monkeypatch):
    prompt = _prompt(tmp_path)
    calls = []

    def fake_generate(self, messages, max_time, max_new_tokens=None):
        content = messages[0]["content"]
        calls.append(content)
        if len(calls) == 1:
            n_img = sum(1 for c in content if c["type"] == "image")
            assert n_img == 3 and "BNC connector" in content[-1]["text"]
            return vlm.RawOutput(text='{"boxes": {"hama1": [400, 400, 600, 600], "hama2": null, "hand": [0, 0, 500, 500]}}',
                                 latency_s=1.0, input_tokens=900, output_tokens=40)
        return vlm.RawOutput(text='{"result": "failure"}', latency_s=2.0, input_tokens=4000, output_tokens=8)

    monkeypatch.setattr(skill.ZoomSkillAdapter, "_generate", fake_generate)
    ad = skill.ZoomSkillAdapter("z", {"type": "transformers", "hf_id": "x", "skill": "zoom"})
    out = ad.predict(prompt)
    second = calls[1]
    assert sum(1 for c in second if c["type"] == "image") == 12 + 2
    assert second[-1]["text"].strip() == "Now answer with the JSON object only."
    assert out.text == '{"result": "failure"}' and out.input_tokens == 4900
    sk = out.extra["skill"]
    assert sk["n_zoom"] == 2 and set(sk["zooms"]) == {"hama1", "hand"} and sk["locate_latency_s"] == 1.0
    zp = Path(prompt.images[0]["input"]).parent / "zoom" / "zoom_hama1.jpg"
    img = cv2.imread(str(zp))
    assert img is not None and max(img.shape[:2]) == 512  # 原圖 640x480 裁切後放大到長邊 512


def test_zoom_skill_without_boxes_is_plain_answer(tmp_path, monkeypatch):
    prompt = _prompt(tmp_path)
    calls = []

    def fake_generate(self, messages, max_time, max_new_tokens=None):
        calls.append(messages[0]["content"])
        return vlm.RawOutput(text="I cannot see it" if len(calls) == 1 else '{"result": "success"}', latency_s=0.5)

    monkeypatch.setattr(skill.ZoomSkillAdapter, "_generate", fake_generate)
    out = skill.ZoomSkillAdapter("z", {"type": "transformers", "hf_id": "x", "skill": "zoom"}).predict(prompt)
    assert sum(1 for c in calls[1] if c["type"] == "image") == 12 and out.extra["skill"]["n_zoom"] == 0


def test_explain_configs_and_prompt():
    """蒐集推理：設定檔讀得進來、提示詞能組出來、答案格式（evidence＋decisive_cue＋result）解析得到 result。"""
    mc = vlm.load_model_configs([ROOT / "configs/model.yaml", ROOT / "configs/model_p8.yaml", ROOT / "configs/model_explain.yaml"])
    c, g = mc["claude_sonnet55_explain"], mc["gpt61_sol_explain"]
    assert c["model_id"] == mc["claude_sonnet55"]["model_id"] and g["model_id"] == mc["gpt61_sol"]["model_id"]
    assert c["extra_params"]["extra_body"]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert c["max_tokens"] >= 4096 and c["image_max_side"] == g["image_max_side"] == 512
    exp = vlm.load_experiment(ROOT / "configs/experiment_bnc_explain.yaml")
    base = vlm.load_experiment(ROOT / "configs/experiment_bnc_p8.yaml")
    for k in ("split_dir", "frames_root", "sampling_template", "frames", "cameras", "task", "include"):
        assert exp[k] == base[k], k  # 只換提示詞與模型
    assert exp["name"] != base["name"] and exp["prompt"]["version"] == "explain-v1"
    text = vlm.PromptBuilder.from_experiment(exp).instruction(-0.5, 20.0, 12)
    assert '"decisive_cue"' in text and '{"evidence": [{"image": 12' in text and "BNC connector" in text
    ans = ('{"evidence": [{"image": 12, "camera": "hand", "seen": "connector still between the fingers"}], '
           '"decisive_cue": "the gripper still holds the connector", "result": "failure"}')
    p = vlm.parse_prediction(ans, segment_id="s", reasons=exp["task"]["failure_reasons"], cameras=exp["cameras"], t_range=(-0.5, 20.0))
    assert p["status"] == "ok" and p["result"] == "failure"


def test_checklist_skill_v3_configs():
    """Skill 第三版：檢查清單提示詞（兩種回答格式）× 有沒有手腕固定裁切；其他設定和 ft-generic 相同。"""
    mc = vlm.load_model_configs([ROOT / "configs/model.yaml", ROOT / "configs/model_p8.yaml", ROOT / "configs/model_skill2.yaml"])
    base = vlm.load_experiment(ROOT / "configs/experiment_bnc_p8.yaml")
    names = set()
    for f, ver in (("configs/experiment_bnc_checklist.yaml", "checklist-v1"), ("configs/experiment_bnc_checklist_fields.yaml", "checklist-fields-v1")):
        exp = vlm.load_experiment(ROOT / f)
        assert exp["prompt"]["version"] == ver and exp["models"] == ["qwen3vl_8b", "qwen3vl_8b_zoomfix"]
        assert all(m in mc for m in exp["models"])
        for k in ("split_dir", "frames_root", "sampling_template", "frames", "cameras", "task", "include"):
            assert exp[k] == base[k], k
        names.add(exp["name"])
        text = vlm.PromptBuilder.from_experiment(exp).instruction(-0.5, 20.0, 12)
        assert "still between the gripper fingers" in text and "did the task board itself move" in text and "BNC connector" in text
    assert len(names) == 2 and base["name"] not in names
    fields = vlm.load_experiment(ROOT / "configs/experiment_bnc_checklist_fields.yaml")
    ans = '{"held_by_gripper": "yes", "place_empty_or_part_misplaced": "no", "board_moved": "no", "result": "failure"}'
    p = vlm.parse_prediction(ans, segment_id="s", reasons=fields["task"]["failure_reasons"], cameras=fields["cameras"], t_range=(-0.5, 20.0))
    assert p["status"] == "ok" and p["result"] == "failure"


def test_anthropic_thinking_summary_is_recorded(tmp_path):
    prompt = _prompt(tmp_path)
    cfg = {"type": "anthropic", "model_id": "claude-x", "max_tokens": 8000, "price_in_per_mtok": 2.0, "price_out_per_mtok": 10.0,
           "extra_params": {"extra_body": {"thinking": {"type": "adaptive", "display": "summarized"}}}}
    ad = api_models.AnthropicAdapter("c", cfg)
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="The wrist image shows the connector held.", signature="x"),
                                        SimpleNamespace(type="text", text='{"result": "failure"}')],
                               model="claude-x", stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=3000, output_tokens=400,
                                                     output_tokens_details=SimpleNamespace(thinking_tokens=380)))

    ad.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    out = ad.predict(prompt)
    assert calls[0]["extra_body"] == {"thinking": {"type": "adaptive", "display": "summarized"}} and calls[0]["max_tokens"] == 8000
    assert out.text == '{"result": "failure"}'  # 推理摘要不混進答案
    assert out.extra["thinking_text"].startswith("The wrist image") and out.extra["thinking_tokens"] == 380
    # 預設（omitted）：推理欄位是空字串 → 不記
    ad.client = _FakeAnthropic()
    assert "thinking_text" not in ad.predict(prompt).extra


def test_repo_configs_skill2():
    mc = vlm.load_model_configs([ROOT / "configs/model.yaml", ROOT / "configs/model_p8.yaml", ROOT / "configs/model_skill2.yaml"])
    base = mc["qwen3vl_8b"]
    for k in ("qwen3vl_8b_zoomfix", "qwen3vl_8b_zoomhand"):
        m = mc[k]
        assert m["skill"] == "zoom" and m["zoom_cameras"] == ["hand"]
        assert {a: b for a, b in m.items() if not a.startswith(("skill", "zoom", "locate"))} == base  # 只多了 Skill 的設定
    assert mc["qwen3vl_8b_zoomfix"]["zoom_mode"] == "fixed" and mc["qwen3vl_8b_zoomfix"]["zoom_fixed_box"] == [0.2, 0.2, 0.8, 0.8]
    assert "zoom_mode" not in mc["qwen3vl_8b_zoom"]  # 第一版設定沒被補值（雜湊不變）


def test_bad_zoom_configs(tmp_path):
    p = tmp_path / "m.yaml"
    base = "models:\n  a: {type: transformers, hf_id: x, skill: zoom, %s}\n"
    for bad in ("zoom_mode: magic", "zoom_mode: fixed", "zoom_mode: fixed, zoom_fixed_box: [0.5, 0.2, 0.4, 0.8]",
                "zoom_cameras: hand", "zoom_cameras: []"):
        p.write_text(base % bad, encoding="utf-8")
        with pytest.raises(vlm.VLMConfigError):
            vlm.load_model_configs(p)
    p.write_text(base % "zoom_mode: fixed, zoom_fixed_box: [0.2, 0.2, 0.8, 0.8], zoom_cameras: [hand]", encoding="utf-8")
    assert vlm.load_model_configs(p)["a"]["zoom_mode"] == "fixed"


def test_zoom_fixed_wrist_only_one_call(tmp_path, monkeypatch):
    """第二版 A：不叫模型找位置，只裁手腕相機的固定區域，模型只呼叫一次。"""
    prompt = _prompt(tmp_path)
    calls = []

    def fake_generate(self, messages, max_time, max_new_tokens=None):
        calls.append(messages[0]["content"])
        return vlm.RawOutput(text='{"result": "success"}', latency_s=2.0, input_tokens=4200, output_tokens=8)

    monkeypatch.setattr(skill.ZoomSkillAdapter, "_generate", fake_generate)
    cfg = {"type": "transformers", "hf_id": "x", "skill": "zoom", "zoom_mode": "fixed", "zoom_cameras": ["hand"],
           "zoom_fixed_box": [0.2, 0.2, 0.8, 0.8]}
    out = skill.ZoomSkillAdapter("z", cfg).predict(prompt)
    assert len(calls) == 1 and sum(1 for c in calls[0] if c["type"] == "image") == 12 + 1
    assert any("fixed region" in c.get("text", "") for c in calls[0] if c["type"] == "text")
    sk = out.extra["skill"]
    assert sk["mode"] == "fixed" and sk["cameras"] == ["hand"] and sk["locate_latency_s"] is None and sk["locate_text"] is None
    assert sk["zooms"]["hand"]["crop_px"] == [128, 96, 512, 384]  # 640x480 的中間 60%，不外擴
    img = cv2.imread(str(Path(prompt.images[0]["input"]).parent / "zoom" / "zoom_hand.jpg"))
    assert img.shape[:2] == (384, 512)
    assert out.input_tokens == 4200


def test_zoom_locate_wrist_only(tmp_path, monkeypatch):
    """第二版 B：找位置只給手腕相機那一張，也只放大手腕相機。"""
    prompt = _prompt(tmp_path)
    calls = []

    def fake_generate(self, messages, max_time, max_new_tokens=None):
        calls.append(messages[0]["content"])
        if len(calls) == 1:
            return vlm.RawOutput(text='{"boxes": {"hand": [300, 300, 600, 700]}}', latency_s=1.0)
        return vlm.RawOutput(text='{"result": "failure"}', latency_s=2.0)

    monkeypatch.setattr(skill.ZoomSkillAdapter, "_generate", fake_generate)
    cfg = {"type": "transformers", "hf_id": "x", "skill": "zoom", "zoom_cameras": ["hand"]}
    out = skill.ZoomSkillAdapter("z", cfg).predict(prompt)
    assert sum(1 for c in calls[0] if c["type"] == "image") == 1 and '"hand": [x1, y1, x2, y2]' in calls[0][-1]["text"]
    assert "hama1" not in calls[0][-1]["text"]
    assert sum(1 for c in calls[1] if c["type"] == "image") == 12 + 1
    assert out.extra["skill"]["n_zoom"] == 1 and set(out.extra["skill"]["zooms"]) == {"hand"}


# ---------------------------------------------------------------- 執行時間


def test_run_timing_and_lines(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "run_config.json").write_text(json.dumps({"model": {"type": "anthropic"}, "metadata": {"load_s": 0.2}}), encoding="utf-8")
    recs = [{"segment_id": f"s{i}", "latency_s": float(i), "cost_usd": 0.01} for i in range(1, 5)]
    recs.append({"segment_id": "s4", "latency_s": 4.0, "repair_latency_s": 1.0, "cost_usd": 0.01, "repair_cost_usd": 0.002})
    (rd / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    t = timing.run_timing(rd)
    assert t["n_timed"] == 4 and t["total_s"] == 1 + 2 + 3 + 5 and t["mean_s"] == 2.75
    assert t["cost_usd"] == pytest.approx(0.042) and t["where"].startswith("anthropic")
    assert timing.run_timing(rd, ["s1"])["n_timed"] == 1
    lines = timing.timing_lines([{"id": "claude", "timing": t}])
    assert lines[2].startswith("| claude | anthropic API") and "US$0.042" in lines[2]
    assert timing.fmt_s(0.05) == "50 ms" and timing.fmt_s(300) == "5.0 分"


def test_summary_has_cost_and_median(tmp_path):
    _write_segment(tmp_path / "frames", "A__seg1", 4)
    exp = _exp(tmp_path, frames=[4])
    cfg = {"type": "mock"}
    spec = vlm.RunSpec(exp=exp, model_key="mock", model_cfg=cfg, n_frames=4)
    st = vlm.run_one(spec, vlm.MockAdapter("mock", cfg), vlm.PromptBuilder.from_experiment(exp), _segments("A__seg1"),
                     reasons=REASONS)
    assert st["latency_median_s"] is not None and st["cost_total_usd"] is None


# ---------------------------------------------------------------- 單一集合的切分


def test_single_set_split(fake_index, tmp_path):
    from assembly_monitor.split import load_split, make_split
    from assembly_monitor.split import LockedTestSetError

    recs = [(f"2025-01-0{i}-10-00-00", [(0, 5, True), (10, 15, i % 2 == 0)]) for i in range(1, 6)]
    idx = fake_index(recs, part="USB")
    res = make_split(index_dir=idx, out_root=tmp_path / "splits", action="insert", part="USB", ratios="test=1",
                     name="usb_test100", hash_frames=False)
    assert res["summary"]["splits"]["test"]["n_segments"] == 10 and res["leakage"]["status"] != "fail"
    with pytest.raises(LockedTestSetError):
        load_split(res["dir"], include=("test",))
    assert len(load_split(res["dir"], include=("test",), allow_test=True)) == 10
    assert load_split(res["dir"], include=("dev",)) == []
