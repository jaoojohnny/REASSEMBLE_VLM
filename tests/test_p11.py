"""第 11 包：讓 Qwen 判斷更準的推理方式（機率門檻、前後對照、答失敗再檢查一次、看範例）與開發集設定。
不需要 GPU、網路或 API key（強制作答的前向計算用假模型測，沒有 torch 就略過那一個測試）。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from assembly_monitor import final, reason, reason_eval, skill, vlm

cv2 = pytest.importorskip("cv2")
from test_p8 import _prompt  # noqa: E402  共用第 8 包測試的假影格

ROOT = Path(__file__).resolve().parent.parent
MODELS = [ROOT / f"configs/{f}" for f in ("model.yaml", "model_p8.yaml", "model_skill2.yaml", "model_reason.yaml")]
PARTS = [("sqpeg2", "square peg", "square_peg_2"), ("rpeg3", "round peg", "round_peg_3"),
         ("dsub", "D-SUB connector", "D_SUB"), ("mgear", "gear", "medium_gear")]


def test_model_reason_configs():
    mc = vlm.load_model_configs(MODELS)
    base = mc["qwen3vl_8b_zoomfix"]
    for k in ("qwen3vl_8b_rverify", "qwen3vl_8b_rcontrast", "qwen3vl_8b_rfewshot"):
        v = mc[k]
        # 模型本身和第三版相同：同一個權重、4-bit、長邊 512、同一個手腕固定裁切
        for f in ("hf_id", "revision", "quantization", "quant_skip_modules", "dtype", "image_max_side", "zoom_fixed_box", "zoom_side"):
            assert v[f] == base[f], (k, f)
        assert v["decision"] == {"rule": "any", "threshold": 0.5} and "skill" not in v
    assert mc["qwen3vl_8b_rverify"]["reason"] == "verify" and mc["qwen3vl_8b_rverify"]["verify_template"] == "configs/prompts/verify_v1.txt"
    assert mc["qwen3vl_8b_rcontrast"]["contrast"] is True and mc["qwen3vl_8b_rfewshot"]["fewshot_required"] is True
    assert isinstance(vlm.make_adapter("x", mc["qwen3vl_8b_rverify"]), reason.ReasonAdapter)


def test_bad_reason_configs(tmp_path):
    p = tmp_path / "m.yaml"
    base = "models:\n  a: {type: transformers, hf_id: x, zoom_fixed_box: [0.2, 0.2, 0.8, 0.8], %s}\n"
    for bad in ("reason: think, decision: {rule: any, threshold: 0.5}",
                "reason: score, decision: {rule: max, threshold: 0.5}",
                "reason: score, decision: {rule: any, threshold: 1.5}",
                "reason: score, decision: {rule: any, threshold: 0.5}, skill: zoom",
                "reason: verify, decision: {rule: any, threshold: 0.5}",
                "reason: verify, decision: {rule: combined, threshold: 0.9}, verify_template: v.txt",
                "reason: score, decision: {rule: any, threshold: 0.5}, contrast: 1",
                "reason: score, decision: {rule: any, threshold: 0.5}, fewshot: {split_dir: s, set: dev, frames_root: f, sampling_key: k, seed: 0}"):
        p.write_text(base % bad, encoding="utf-8")
        with pytest.raises(vlm.VLMConfigError):
            vlm.load_model_configs(p)
    p.write_text("models:\n  a: {type: transformers, hf_id: x, reason: score, decision: {rule: any, threshold: 0.5}}\n", encoding="utf-8")
    with pytest.raises(vlm.VLMConfigError):  # 沒有裁切範圍
        vlm.load_model_configs(p)


@pytest.mark.parametrize("key,phrase,froot", PARTS)
def test_rdev_experiments(key, phrase, froot):
    """開發集實驗：任務文字、取樣、相機和 bolt 4 相同（只換零件名稱、切分、影格資料夾）；第三版提示詞沒改。"""
    b = vlm.load_experiment(ROOT / "configs/experiment_bolt4_checklistf.yaml")
    v3 = vlm.load_experiment(ROOT / f"configs/experiment_rdev_{key}_v3.yaml")
    co = vlm.load_experiment(ROOT / f"configs/experiment_rdev_{key}_contrast.yaml")
    split = f"data/splits/{key}_ref10_dev90"
    for e in (v3, co):
        assert e["split_dir"] == split and e["include"] == ["dev"] and e["frames"] == [4]
        assert e["frames_root"] == f"data/derived/frames/insert_{froot}"
        assert e["sampling_template"] == b["sampling_template"] and e["cameras"] == b["cameras"]
        assert e["task"]["part"] == phrase
        assert {k: v for k, v in e["task"].items() if k != "part"} == {k: v for k, v in b["task"].items() if k != "part"}
    assert v3["prompt"] == b["prompt"] and v3["models"] == ["qwen3vl_8b_zoomfix", "qwen3vl_8b_rverify", "qwen3vl_8b_rfewshot"]
    assert co["prompt"] == {"version": "contrast-v1", "template": "configs/prompts/contrast_v1.txt"}
    assert co["models"] == ["qwen3vl_8b_rcontrast"]
    mc = vlm.load_model_configs(MODELS)
    fs = vlm.effective_model_cfg(v3, "qwen3vl_8b_rfewshot", mc["qwen3vl_8b_rfewshot"])["fewshot"]
    assert fs == {"split_dir": split, "set": "ref", "frames_root": v3["frames_root"],
                  "sampling_key": "uniform_n4_sb0.5_eb0.5_jpg", "seed": 0}
    reason.check_reason_cfg("f", vlm.effective_model_cfg(v3, "qwen3vl_8b_rfewshot", mc["qwen3vl_8b_rfewshot"]))
    vlm.PromptBuilder.from_experiment(co, ROOT).instruction(-0.5, 10.0, 12)  # 提示詞的欄位都填得進去
    cfg = reason_eval.load_cfg(ROOT / "configs/reason_dev.yaml")
    assert [p["key"] for p in cfg["parts"]] == [k for k, _, _ in PARTS]


def test_probability_helpers():
    assert reason.two_way(math.log(0.3), math.log(0.1)) == pytest.approx(0.75)
    p = {"held_by_gripper": 0.6, "place_empty_or_part_misplaced": 0.5, "board_moved": 0.0}
    assert reason.fail_score(p) == pytest.approx(1 - 0.4 * 0.5)
    assert reason.decide(p, "any", 0.5) == "failure" and reason.decide(p, "any", 0.7) == "success"
    assert reason.decide(p, "combined", 0.75) == "failure" and reason.decide(p, "combined", 0.85) == "success"
    t = json.loads(reason.answer_text(p, "success"))
    assert t == {"held_by_gripper": "yes", "place_empty_or_part_misplaced": "no", "board_moved": "no", "result": "success"}
    pred = vlm.parse_prediction(reason.answer_text(p, "failure"), segment_id="s", reasons=["other"], cameras=["hand"], t_range=None)
    assert pred["status"] == "ok" and pred["result"] == "failure"


def _chars(s: str) -> list[int]:
    return [ord(c) for c in s]


def test_probe_positions_point_before_each_answer():
    """位置 p 的 logits 預測 p+1 的 token：每個欄位要讀的是答案第一個字的前一格。"""
    pieces = reason.checklist_pieces()
    text = "".join(t for t, _, _ in pieces)
    assert json.loads(text) == {"held_by_gripper": "no", "place_empty_or_part_misplaced": "no", "board_moved": "no", "result": "success"}
    ids, probes = reason.probe_positions(100, [_chars(t) for t, _, _ in pieces], pieces)
    assert ids == _chars(text)
    full = [0] * 100 + ids
    for pos, fld, cands in probes:
        assert chr(full[pos + 1]) == ("s" if fld == "result" else "n") and chr(full[pos]) == '"'
    assert [f for _, f, _ in probes] == [*reason.CHECKS, "result"]
    v = reason.verify_pieces()
    assert json.loads("".join(t for t, _, _ in v)) == {"first_check_correct": "no"}


def test_pick_refs_fixed():
    rows = [{"segment_id": f"r{i}", "dataset_result": "failure" if i in (2, 5) else "success"} for i in range(8)]
    a, b = reason.pick_refs(rows, 0), reason.pick_refs(list(reversed(rows)), 0)
    assert a == b and a["failure"] in ("r2", "r5") and a["success"] not in ("r2", "r5")
    with pytest.raises(vlm.VLMConfigError):
        reason.pick_refs([r for r in rows if r["dataset_result"] == "success"], 0)


def _cfg(**kw):
    c = {"type": "transformers", "hf_id": "x", "image_max_side": 512, "reason": "score",
         "decision": {"rule": "any", "threshold": 0.5}, "zoom_fixed_box": [0.2, 0.2, 0.8, 0.8], "zoom_side": 512}
    c.update(kw)
    return c


def test_build_content_same_as_skill_v3(tmp_path, monkeypatch):
    """沒有前後對照、沒有範例時，送給模型的內容和第三版（ZoomSkillAdapter 固定裁切）完全相同。"""
    prompt = _prompt(tmp_path)
    got = []
    monkeypatch.setattr(skill.ZoomSkillAdapter, "_generate",
                        lambda self, m, t, max_new_tokens=None: got.append(m[0]["content"]) or vlm.RawOutput(text="{}", latency_s=0.0))
    skill.ZoomSkillAdapter("z", {"type": "transformers", "hf_id": "x", "skill": "zoom", "zoom_mode": "fixed",
                                 "zoom_cameras": ["hand"], "zoom_fixed_box": [0.2, 0.2, 0.8, 0.8]}).predict(prompt)
    ad = reason.ReasonAdapter("r", _cfg())
    ad.refs = []
    content, info = ad.build_content(prompt)

    def norm(c):
        return [x["text"] if x["type"] == "text" else "IMG" for x in c]

    assert norm(content) == norm(got[0])
    a = cv2.imread(info["zoom_path"])
    b = cv2.imread([x["path"] for x in got[0] if x["type"] == "image"][-1])
    assert a.shape == b.shape and float(abs(a.astype(int) - b.astype(int)).mean()) == 0.0


def test_build_content_contrast_and_fewshot(tmp_path):
    prompt = _prompt(tmp_path)
    ad = reason.ReasonAdapter("r", _cfg(contrast=True, contrast_side=1024))
    ref_img = prompt.images[0]["input"]
    ad.refs = [{"result": "success", "segment_id": "S", "time_sec": 9.0, "image": ref_img, "crop": ref_img},
               {"result": "failure", "segment_id": "F", "time_sec": 9.0, "image": ref_img, "crop": ref_img}]
    content, info = ad.build_content(prompt)
    imgs = [c for c in content if c["type"] == "image"]
    assert len(imgs) == 12 + 1 + 1 + 4  # 原本 12 張＋並排圖＋裁切＋範例 2×2
    texts = "".join(c.get("text", "") for c in content)
    assert texts.index("Side-by-side view") < texts.index("Zoom of camera hand") < texts.index("Reference examples")
    assert "Example 1 (SUCCESS)" in texts and "Example 2 (FAILURE)" in texts and content[-1]["text"].strip().startswith("Now answer")
    sb = cv2.imread(info.get("zoom_path").replace("zoom_hand.jpg", "hand_first_last.jpg"))
    assert sb.shape[1] == 1024  # 兩張 640×480 並排＋8 px，縮到長邊 1024
    assert info["fewshot_refs"][1] == {"result": "failure", "segment_id": "F", "time_sec": 9.0}


def _adapter_with_scores(tmp_path, monkeypatch, cfg, scores):
    """forced() 換成固定機率：第一次呼叫回傳清單機率，第二次回傳再檢查一次的機率。"""
    prompt = _prompt(tmp_path)
    calls = []

    def fake_forced(self, content, pieces):
        calls.append(content)
        fld = [f for _, f, _ in pieces if f]
        out = {f: {"p": scores[f], "mass": 0.99} for f in fld}
        return out, vlm.RawOutput(text="", latency_s=1.5, input_tokens=4000, peak_vram_mb=9000.0)

    monkeypatch.setattr(reason.ReasonAdapter, "forced", fake_forced)
    ad = reason.ReasonAdapter("r", cfg)
    ad.refs = []
    tpl = (ROOT / "configs/prompts/verify_v1.txt").read_text(encoding="utf-8")
    ad.verify_text = tpl
    return ad, prompt, calls


def test_predict_score_rules(tmp_path, monkeypatch):
    sc = {"held_by_gripper": 0.7, "place_empty_or_part_misplaced": 0.2, "board_moved": 0.01, "result": 0.1}
    ad, prompt, calls = _adapter_with_scores(tmp_path, monkeypatch, _cfg(), sc)
    out = ad.predict(prompt)
    r = out.extra["reason"]
    assert len(calls) == 1 and r["result"] == "failure" and r["fail_score"] == pytest.approx(1 - 0.3 * 0.8 * 0.99)
    assert json.loads(out.text)["held_by_gripper"] == "yes" and out.input_tokens == 4000 and out.output_tokens == 0
    ad2, prompt2, _ = _adapter_with_scores(tmp_path / "b", monkeypatch, _cfg(decision={"rule": "combined", "threshold": 0.9}), sc)
    assert ad2.predict(prompt2).extra["reason"]["result"] == "success"  # 0.766 < 0.9


def test_predict_verify(tmp_path, monkeypatch):
    """第一階段判失敗才問第二階段；第二階段 P(對) ≤ 0.5 就改成成功。第二階段只給手腕並排圖＋裁切圖。"""
    sc = {"held_by_gripper": 0.2, "place_empty_or_part_misplaced": 0.8, "board_moved": 0.0, "result": 0.1, "first_check_correct": 0.3}
    cfg = _cfg(reason="verify", verify_template="configs/prompts/verify_v1.txt", contrast_side=1024)
    ad, prompt, calls = _adapter_with_scores(tmp_path, monkeypatch, cfg, sc)
    r = ad.predict(prompt).extra["reason"]
    assert len(calls) == 2 and r["stage1_result"] == "failure" and r["result"] == "success"
    assert r["verify"]["reason_field"] == "place_empty_or_part_misplaced" and len(r["forward_latency_s"]) == 2
    v = calls[1]
    assert sum(1 for c in v if c["type"] == "image") == 2
    vt = "".join(c.get("text", "") for c in v)
    assert "the place on the board seemed empty" in vt and "{" not in vt.replace('{"first_check_correct": "no"}', "")
    assert '{"first_check_correct": "no"}' in vt and "t = +" in vt
    sc2 = dict(sc, place_empty_or_part_misplaced=0.1)
    ad2, prompt2, calls2 = _adapter_with_scores(tmp_path / "b", monkeypatch, cfg, sc2)
    r2 = ad2.predict(prompt2).extra["reason"]
    assert len(calls2) == 1 and r2["result"] == "success" and "verify" not in r2


def test_forced_reads_logits_at_right_positions(monkeypatch):
    """假模型：每一格最可能的是下一個字；答案是 no 的前一格另外給 yes 一個分數。讀錯位置就會變成 0.5。"""
    torch = pytest.importorskip("torch")

    class Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in s]}

    class Inputs(dict):
        def to(self, device, dtype=None):
            return self

    class Proc:
        tokenizer = Tok()

        def apply_chat_template(self, messages, **kw):
            return Inputs(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), attention_mask=torch.ones((1, 5), dtype=torch.long))

    class Model:
        device = "cpu"

        def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
            ids = input_ids[0].tolist()
            L, V = len(ids), 256
            lg = torch.zeros((1, L, V))
            for q in range(L - 1):
                nxt = ids[q + 1]
                lg[0, q, nxt] = 5.0
                if nxt == ord("n"):
                    lg[0, q, ord("y")] = 3.0
                if nxt == ord("s"):
                    lg[0, q, ord("f")] = 1.0
            return SimpleNamespace(logits=lg[:, -logits_to_keep:])

    ad = reason.ReasonAdapter("r", _cfg())
    ad.torch, ad.processor, ad.model, ad.dtype, ad.max_positions = torch, Proc(), Model(), torch.float32, None
    ad.cand = {w: ord(w[0]) for w in ("yes", "no", "success", "failure")}
    sc, raw = ad.forced([{"type": "text", "text": "x"}], reason.checklist_pieces())
    for c in reason.CHECKS:
        assert sc[c]["p"] == pytest.approx(math.exp(3) / (math.exp(3) + math.exp(5)), rel=1e-5)
    assert sc["result"]["p"] == pytest.approx(math.exp(1) / (math.exp(1) + math.exp(5)), rel=1e-5)
    assert raw.input_tokens == 5 + len("".join(t for t, _, _ in reason.checklist_pieces()))


def test_locked_files_include_verify_template(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    p = tmp_path / "f.yaml"
    p.write_text(f"""final:
  split_dir: data/splits/sqpeg2_ref10_dev90
  out_name: t
  models: [configs/model.yaml, configs/model_p8.yaml, configs/model_skill2.yaml, configs/model_reason.yaml]
  methods:
    - {{id: a, kind: vlm, experiment: configs/experiment_rdev_sqpeg2_v3.yaml, model: qwen3vl_8b_rverify, frames: 4}}
""", encoding="utf-8")
    files = final.locked_files(final.load_final_config(p))
    assert "configs/prompts/verify_v1.txt" in files and "configs/model_reason.yaml" in files
    old = final.locked_files(final.load_final_config(ROOT / "configs/final_test_sqpeg1.yaml"))
    assert not any("verify" in f for f in old)  # 舊的凍結設定不受影響


def _recs():
    """4 個零件的假紀錄：第三版把一半成功說成失敗；機率門檻提高後能救回來。"""
    out = []
    for k in ("a", "b", "c", "d"):
        for i in range(10):
            fail = i < 3
            pheld = 0.97 if fail else (0.8 if i % 2 else 0.1)
            p = {"held_by_gripper": pheld, "place_empty_or_part_misplaced": 0.05, "board_moved": 0.0}
            out.append({"segment_id": f"{k}{i}", "part": k, "truth": "failure" if fail else "success",
                        "v3_text": "failure" if pheld > 0.5 else "success", "v3_text_s": 4.7,
                        "score_p": p, "score_s": 2.5, "score_s1_s": 2.0, "verify": "failure" if fail else "success",
                        "verify_asked": pheld > 0.5, "contrast_p": p, "contrast_s": 3.0, "fewshot_p": None, "fewshot_s": None})
    return out


def test_reason_eval_lopo_and_selection(monkeypatch, tmp_path):
    recs = _recs()
    monkeypatch.setattr(reason_eval, "part_records", lambda p: [r for r in recs if r["part"] == p["key"]])
    cfg = {"parts": [{"key": k, "name": k.upper()} for k in "abcd"]}
    res = reason_eval.evaluate(cfg)
    rows = {r["id"]: r for r in res["rows"]}
    # 成功段 i＝3…9 中奇數（4 段／零件）P(held)＝0.8 → 第三版說失敗：答對＝失敗 12＋成功 12
    assert rows["v3_text"]["pooled"]["correct"] == 12 + 12 and rows["v3_text"]["pooled"]["bal"] == pytest.approx((1 + 12 / 28) / 2)
    assert rows["score_any"]["pooled"]["correct"] == rows["v3_text"]["pooled"]["correct"] and res["agree_v3_score"] == 40
    t = rows["score_tau"]
    assert t["pooled"]["correct"] == 40 and 0.8 < t["tau_all"] < 0.97 and all(0.8 < v < 0.97 for v in t["tau_lopo"].values())
    assert rows["fewshot_any"]["pooled"]["n_missing"] == 40  # 沒有跑的方法算答錯
    assert res["best"] in ("verify", "score_tau", "contrast_tau") and res["passes"]
    assert rows["score_tau"]["vs_v3"] == {"a_only": 16, "b_only": 0, "p_value": pytest.approx(2 / 2 ** 16)}
    md = reason_eval.write_report(res, tmp_path)
    text = Path(md).read_text(encoding="utf-8")
    assert "留一個零件" in text and "通過" in text and (tmp_path / "per_segment.jsonl").exists()


# ---------------------------------------------------------------- 第二輪（2026-10-31，結合 Codex 的 evidence_review）


def test_round2_configs():
    mc = vlm.load_model_configs(MODELS)
    v = mc["qwen3vl_8b_rv4"]
    for f in ("hf_id", "revision", "quantization", "quant_skip_modules", "dtype", "image_max_side", "zoom_fixed_box", "zoom_side"):
        assert v[f] == mc["qwen3vl_8b_zoomfix"][f]
    assert v["zoom_last_n"] == 2 and v["second_view"] == "configs/prompts/wrist_v1.txt" and v["decision"] == {"rule": "any", "threshold": 0.5}
    for key, phrase, _ in PARTS:
        e4 = vlm.load_experiment(ROOT / f"configs/experiment_rdev_{key}_v4.yaml")
        e3 = vlm.load_experiment(ROOT / f"configs/experiment_rdev_{key}_v3.yaml")
        assert e4["prompt"] == {"version": "checklist-fields-v2", "template": "configs/prompts/checklist_fields_v2.txt"}
        assert e4["models"] == ["qwen3vl_8b_rv4"] and e4["name"] == f"rdev_{key}_v4"
        for f in ("split_dir", "include", "frames_root", "sampling_template", "frames", "cameras", "task"):
            assert e4[f] == e3[f], (key, f)
        text = vlm.PromptBuilder.from_experiment(e4, ROOT).instruction(-0.5, 10.0, 12)
        assert "NOT signs of failure" in text and f"insert the {phrase}" in text and '"board_moved": "no"' in text
    cfg = reason_eval.load_cfg(ROOT / "configs/reason_dev.yaml")
    assert all(p["v4"] == f"configs/experiment_rdev_{p['key']}_v4.yaml" for p in cfg["parts"])


def test_round2_bad_configs(tmp_path):
    p = tmp_path / "m.yaml"
    base = "models:\n  a: {type: transformers, hf_id: x, reason: score, zoom_fixed_box: [0.2, 0.2, 0.8, 0.8], %s}\n"
    for bad in ("decision: {rule: any, threshold: 0.5}, zoom_last_n: 4",
                "decision: {rule: both_and, threshold: 0.5}",
                "decision: {rule: both_mean, threshold: 0.9}, second_view: ''"):
        p.write_text(base % bad, encoding="utf-8")
        with pytest.raises(vlm.VLMConfigError):
            vlm.load_model_configs(p)
    p.write_text("models:\n  a: {type: transformers, hf_id: x, reason: verify, zoom_fixed_box: [0.2, 0.2, 0.8, 0.8], "
                 "decision: {rule: any, threshold: 0.5}, verify_template: v.txt, second_view: w.txt}\n", encoding="utf-8")
    with pytest.raises(vlm.VLMConfigError):
        vlm.load_model_configs(p)


def test_both_rules():
    a = {"held_by_gripper": 0.9, "place_empty_or_part_misplaced": 0.1, "board_moved": 0.0}
    b = {"held_by_gripper": 0.2, "place_empty_or_part_misplaced": 0.3}
    assert reason.decide(a, "both_and", 0.5, b) == "success"           # 只有一次說失敗 → 成功
    assert reason.decide(a, "both_and", 0.5, dict(b, place_empty_or_part_misplaced=0.7)) == "failure"
    s1, s2 = reason.fail_score(a), reason.fail_score(b)
    m = reason.mean_score(s1, s2)
    assert m == pytest.approx(1 / (1 + math.exp(-(reason.logit(s1) + reason.logit(s2)) / 2)))
    assert reason.decide(a, "both_mean", m - 1e-6, b) == "failure" and reason.decide(a, "both_mean", m + 1e-6, b) == "success"
    with pytest.raises(ValueError):
        reason.decide(a, "both_mean", 0.5)


def test_build_content_two_crops(tmp_path):
    prompt = _prompt(tmp_path)
    ad = reason.ReasonAdapter("r", _cfg(zoom_last_n=2))
    ad.refs = []
    content, info = ad.build_content(prompt)
    assert sum(1 for c in content if c["type"] == "image") == 12 + 2
    texts = [c["text"] for c in content if c["type"] == "text"]
    hands = sorted((im for im in prompt.images if im["camera"] == "hand"), key=lambda im: im["time_sec"])[-2:]
    assert any("last two wrist-camera images" in t for t in texts)
    for im in hands:
        assert any(f"(detail of Image {im['i']})" in t and f"{im['time_sec']:+.2f}" in t for t in texts)
    assert [z["t"] for z in info["zooms"]] == [im["time_sec"] for im in hands] and info["zoom_path"] == info["zooms"][-1]["path"]
    assert content[-1]["text"].strip().startswith("Now answer")


def test_predict_second_view(tmp_path, monkeypatch):
    """第二次觀察：只有手腕並排圖＋兩張裁切，提示詞裡沒有第一次的答案；兩次的機率都存下來。"""
    sc = {"held_by_gripper": 0.8, "place_empty_or_part_misplaced": 0.1, "board_moved": 0.0, "result": 0.2}
    cfg = _cfg(zoom_last_n=2, contrast_side=1024, second_view="configs/prompts/wrist_v1.txt")
    ad, prompt, calls = _adapter_with_scores(tmp_path, monkeypatch, cfg, sc)
    ad.second_text = (ROOT / "configs/prompts/wrist_v1.txt").read_text(encoding="utf-8")
    out = ad.predict(prompt)
    r = out.extra["reason"]
    assert len(calls) == 2 and r["second"]["p_yes"] == {"held_by_gripper": 0.8, "place_empty_or_part_misplaced": 0.1}
    w = calls[1]
    assert sum(1 for c in w if c["type"] == "image") == 3
    wt = "".join(c.get("text", "") for c in w)
    assert "first check" not in wt and "FAILED" not in wt and "{" not in wt.replace('{"held_by_gripper": "no", "place_empty_or_part_misplaced": "no", "result": "success"}', "")
    assert "attempt to insert the" in wt and "LEFT is the first image (t = -" in wt
    assert r["result"] == "failure" and len(r["forward_latency_s"]) == 2 and "zooms" not in r
    # 兩次觀察的失敗分數都是 1 − 0.2 × 0.9 ＝ 0.82 → 平均 0.82
    for rule, thr, want in (("both_and", 0.5, "failure"), ("both_mean", 0.8, "failure"), ("both_mean", 0.85, "success")):
        ad2, p2, _ = _adapter_with_scores(tmp_path / f"{rule}{thr}", monkeypatch, dict(cfg, decision={"rule": rule, "threshold": thr}), sc)
        ad2.second_text = ad.second_text
        assert ad2.predict(p2).extra["reason"]["result"] == want, (rule, thr)


def test_locked_files_include_second_view(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    p = tmp_path / "f.yaml"
    p.write_text("""final:
  split_dir: data/splits/sqpeg2_ref10_dev90
  out_name: t
  models: [configs/model.yaml, configs/model_p8.yaml, configs/model_skill2.yaml, configs/model_reason.yaml]
  methods:
    - {id: a, kind: vlm, experiment: configs/experiment_rdev_sqpeg2_v4.yaml, model: qwen3vl_8b_rv4, frames: 4}
""", encoding="utf-8")
    files = final.locked_files(final.load_final_config(p))
    assert "configs/prompts/wrist_v1.txt" in files and "configs/prompts/checklist_fields_v2.txt" in files


def test_reason_eval_round2(monkeypatch):
    recs = _recs()
    for r in recs:
        fail = r["truth"] == "failure"
        r["v4_p"] = {"held_by_gripper": 0.9 if fail else 0.6, "place_empty_or_part_misplaced": 0.0, "board_moved": 0.0}
        r["v4w_p"] = {"held_by_gripper": 0.9 if fail else 0.2, "place_empty_or_part_misplaced": 0.0}
        r["v4_s"], r["v4_s1_s"] = 2.0, 1.2
    monkeypatch.setattr(reason_eval, "part_records", lambda p: [r for r in recs if r["part"] == p["key"]])
    res = reason_eval.evaluate({"parts": [{"key": k, "name": k, "v4": "x"} for k in "abcd"]})
    rows = {r["id"]: r for r in res["rows"]}
    assert rows["v4_any"]["pooled"]["succ_ok"] == 0 and rows["v4w_any"]["pooled"]["correct"] == 40
    assert rows["v4and"]["pooled"]["correct"] == 40 and rows["v4both_tau"]["pooled"]["correct"] == 40
    assert rows["v4and"]["mean_s"] == pytest.approx(2.0) and rows["v4_any"]["mean_s"] == pytest.approx(1.2)


# ---------------------------------------------------------------- 測試集（2026-10-31 凍結）

TEST_PARTS = [("sqpeg4", "square peg 4", "square peg", "square_peg_4"), ("sqpeg3", "square peg 3", "square peg", "square_peg_3"),
              ("rpeg2", "round peg 2", "round peg", "round_peg_2"), ("wproof", "waterproof", "waterproof connector", "waterproof")]


def test_second_any_rule():
    a = {"held_by_gripper": 0.99, "place_empty_or_part_misplaced": 0.0, "board_moved": 0.0}
    assert reason.decide(a, "second_any", 0.5, {"held_by_gripper": 0.3, "place_empty_or_part_misplaced": 0.2}) == "success"
    assert reason.decide({}, "second_any", 0.5, {"held_by_gripper": 0.6}) == "failure"
    with pytest.raises(ValueError):
        reason.decide(a, "second_any", 0.5)


@pytest.mark.parametrize("key,part,phrase,froot", TEST_PARTS)
def test_final_test_p11_configs(key, part, phrase, froot, monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = final.load_final_config(ROOT / f"configs/final_test_{key}.yaml")
    assert cfg["frozen"] and cfg["part"] == part and cfg["split_dir"] == f"data/splits/{key}_test100"
    assert cfg["out_name"] == f"final_test_{key}" and cfg["primary"] == "qwen3vl8b_prob_t0999"
    m = {x["id"]: x for x in cfg["methods"]}
    assert list(m) == ["majority", "qwen3vl8b_prob_t0999", "qwen3vl8b_wrist_05", "qwen3vl8b_v3_eb05_f4"]
    assert (m["qwen3vl8b_prob_t0999"]["model"], m["qwen3vl8b_wrist_05"]["model"], m["qwen3vl8b_v3_eb05_f4"]["model"]) == \
        ("qwen3vl_8b_rscore_t0999", "qwen3vl_8b_rv4_wrist", "qwen3vl_8b_zoomfix")
    mc = vlm.load_model_configs(MODELS)
    t = mc["qwen3vl_8b_rscore_t0999"]
    assert t["decision"] == {"rule": "combined", "threshold": 0.999089} and "zoom_last_n" not in t and "second_view" not in t
    for f in ("hf_id", "quantization", "image_max_side", "zoom_fixed_box", "zoom_side"):
        assert t[f] == mc["qwen3vl_8b_rverify"][f]          # 和開發集第一階段相同（不再檢查一次）
    assert t["reason"] == "score"
    w = mc["qwen3vl_8b_rv4_wrist"]
    assert {k: v for k, v in w.items() if k != "decision"} == {k: v for k, v in mc["qwen3vl_8b_rv4"].items() if k != "decision"}
    assert w["decision"] == {"rule": "second_any", "threshold": 0.5}
    # τ＝開發集 4 個零件合起來選的值（logit 7）
    assert reason_eval.TAU_GRID[28 + 24] == pytest.approx(0.999089, abs=1e-6)
    for kind, dev in (("v3", "v3"), ("v4", "v4")):
        e = vlm.load_experiment(ROOT / f"configs/experiment_test_{key}_{kind}.yaml")
        d = vlm.load_experiment(ROOT / f"configs/experiment_rdev_sqpeg2_{dev}.yaml")
        assert e["split_dir"] == cfg["split_dir"] and e["include"] == ["test"] and e["frames_root"] == f"data/derived/frames/insert_{froot}"
        assert e["task"]["part"] == phrase and "model_overrides" not in e
        for f in ("prompt", "sampling_template", "frames", "cameras"):
            assert e[f] == d[f]
        assert {k: v for k, v in e["task"].items() if k != "part"} == {k: v for k, v in d["task"].items() if k != "part"}
    for x in cfg["methods"]:
        if x["kind"] == "vlm":
            final.experiment_for_test(cfg, x)
    files = final.locked_files(cfg)
    for f in ("configs/model_reason.yaml", f"configs/experiment_test_{key}_v3.yaml", f"configs/experiment_test_{key}_v4.yaml",
              "configs/prompts/checklist_fields_v1.txt", "configs/prompts/checklist_fields_v2.txt", "configs/prompts/wrist_v1.txt"):
        assert f in files
    assert final.needed_extractions(cfg) == [{"end_buffer": 0.5, "frames": [4]}]


def test_pool_final(monkeypatch, tmp_path):
    """4 份測試結果合併：McNemar、兩類平均、以錄影為單位的區間、主要問題是否成立。"""
    def fake_rows(cp):
        k = Path(cp).stem
        segs = [{"segment_id": f"{k}R{i // 2}__seg{i}", "recording_id": f"{k}R{i // 2}",
                 "dataset_result": "failure" if i < 4 else "success"} for i in range(20)]
        preds = {"majority": {s["segment_id"]: {"status": "ok", "result": "success"} for s in segs},
                 "qwen3vl8b_prob_t0999": {s["segment_id"]: {"status": "ok", "result": "failure" if i < 3 else "success"} for i, s in enumerate(segs)},
                 "qwen3vl8b_wrist_05": {s["segment_id"]: {"status": "ok", "result": "failure" if i < 8 else "success"} for i, s in enumerate(segs)},
                 "qwen3vl8b_v3_eb05_f4": {s["segment_id"]: {"status": "ok", "result": "failure" if i < 14 else "success"} for i, s in enumerate(segs)}}
        cfg = {"out_name": k, "part": k, "primary": "qwen3vl8b_prob_t0999",
               "methods": [{"id": x, "kind": "majority" if x == "majority" else "vlm"} for x in preds]}
        return cfg, segs, preds

    monkeypatch.setattr(reason_eval, "_final_rows", fake_rows)
    monkeypatch.setattr(final, "method_run_dir", lambda cfg, m: tmp_path / "none")
    res = reason_eval.pool_final([f"p{i}.yaml" for i in range(4)], n_boot=200)
    rows = {r["id"]: r for r in res["rows"]}
    p, b = rows["qwen3vl8b_prob_t0999"]["pooled"], rows["qwen3vl8b_v3_eb05_f4"]["pooled"]
    assert (p["fail_ok"], p["succ_ok"], b["fail_ok"], b["succ_ok"]) == (12, 64, 16, 24)
    assert rows["qwen3vl8b_prob_t0999"]["vs_baseline"]["a_only"] == 40 and rows["qwen3vl8b_prob_t0999"]["vs_baseline"]["b_only"] == 4
    assert res["holds"] and res["d_bal"] == pytest.approx((12 / 16 + 64 / 64) / 2 - (16 / 16 + 24 / 64) / 2)
    lo, hi = rows["qwen3vl8b_prob_t0999"]["ci_bal"]
    assert lo <= p["bal"] <= hi
    md = reason_eval.write_pooled_report(res, tmp_path / "out")
    assert "成立" in Path(md).read_text(encoding="utf-8")
