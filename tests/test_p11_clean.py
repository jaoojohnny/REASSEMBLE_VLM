"""第 11 包：乾淨開發集（盲評判斷和資料集標記一致的段）。"""

import json
from pathlib import Path

import pytest

from assembly_monitor import clean_dev


def _recs():
    out = []
    for k in ("a", "b", "c", "d"):
        for i in range(10):
            fail = i < 3
            pheld = 0.97 if fail else (0.8 if i % 2 else 0.1)
            p = {"held_by_gripper": pheld, "place_empty_or_part_misplaced": 0.05, "board_moved": 0.0}
            out.append({"segment_id": f"{k}{i}", "part": k, "truth": "failure" if fail else "success",
                        "v3_text": "failure" if pheld > 0.5 else "success", "score_p": p, "verify": "success",
                        "contrast_p": p, "fewshot_p": None, "v4_p": p, "v4w_p": p})
    return out


def _files(tmp_path, recs, judge):
    """a0–a4 放在 30 段那邊（12 張影格），其餘放 211 段那邊（3 張）。judge(rec) → 判斷。"""
    clean, blind = tmp_path / "clean", tmp_path / "blind"
    clean.mkdir()
    blind.mkdir()
    blind_recs = [r for r in recs if r["segment_id"] in {"a0", "a1", "a2", "a3", "a4"}]
    rest = [r for r in recs if r not in blind_recs]
    order = [{"case_id": f"R{i:03d}", "part": r["part"], "segment_id": r["segment_id"], "sheet": "S01"} for i, r in enumerate(rest, 1)]
    (clean / "order.json").write_text(json.dumps(order), encoding="utf-8")
    zh = {"success": "success", "failure": "failure", "uncertain": "uncertain"}
    (clean / "claude_judgments.jsonl").write_text(
        "".join(json.dumps({"case_id": o["case_id"], "judgment": zh[judge(r)], "note": ""}) + "\n" for o, r in zip(order, rest)),
        encoding="utf-8")
    key = [{"case_id": f"B{i:02d}", "segment_id": r["segment_id"], "part": r["part"], "truth": r["truth"]} for i, r in enumerate(blind_recs, 1)]
    (blind / "PRIVATE_answer_key.json").write_text(json.dumps(key), encoding="utf-8")
    cn = {"success": "成功", "failure": "失敗", "uncertain": "看不出"}
    rv = [{"case_id": k["case_id"], "judgment": cn[judge(r)], "evidence": "x"} for k, r in zip(key, blind_recs)]
    (blind / "blind_review_dev30_answers_claude.json").write_text(json.dumps({"reviews": rv}, ensure_ascii=False), encoding="utf-8")
    return clean, blind


def test_load_judgments_and_checks(tmp_path):
    recs = _recs()
    clean, blind = _files(tmp_path, recs, lambda r: "uncertain" if r["segment_id"] == "a1" else r["truth"])
    J = clean_dev.load_judgments(clean, blind)
    assert len(J) == 40 and J["a1"]["judgment"] == "uncertain" and J["a0"]["protocol"] == "12" and J["b0"]["protocol"] == "3"
    assert J["a0"]["judgment"] == "failure" and J["a5"]["judgment"] == "success"
    lines = (clean / "claude_judgments.jsonl").read_text(encoding="utf-8").splitlines()
    (clean / "claude_judgments.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="還沒有判斷"):
        clean_dev.load_judgments(clean, blind)
    (clean / "claude_judgments.jsonl").write_text("\n".join(lines + lines[:1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重複"):
        clean_dev.load_judgments(clean, blind)


def test_build_clean_subset(tmp_path, monkeypatch):
    recs = _recs()

    def judge(r):  # 每個零件：成功段 i＝3 說失敗、i＝4 看不出；失敗段 i＝0 說成功
        i = int(r["segment_id"][1:])
        return {0: "success", 3: "failure", 4: "uncertain"}.get(i, r["truth"])

    clean, blind = _files(tmp_path, recs, judge)
    monkeypatch.setattr(clean_dev, "part_records", lambda p: [dict(r) for r in recs if r["part"] == p["key"]])
    cfg = {"parts": [{"key": k, "name": k.upper()} for k in "abcd"]}
    res = clean_dev.build(cfg, clean, blind, frozen_tau=None)
    A = res["agree_all"]
    assert A["n"] == 40 and A["usable"] == 28 and A["said_s_label_f"] == 4 and A["said_f_label_s"] == 4 and A["unc_label_s"] == 4
    assert res["clean_counts"]["a"] == {**res["clean_counts"]["a"], "n": 7, "n_fail": 2, "n_succ": 5}
    rows = {r["id"]: r for r in res["clean_rows"]}
    # 乾淨段：每零件失敗 i＝1,2；成功 i＝5…9（奇數 5,7,9 P＝0.8 → 第三版說失敗）
    assert rows["v3_text"]["pooled"]["correct"] == 8 + 8 and rows["score_tau"]["pooled"]["correct"] == 28
    assert {r["id"] for r in res["full_rows"]} == set(rows)
    md = clean_dev.write_outputs(res, tmp_path / "out")
    text = Path(md).read_text(encoding="utf-8")
    assert "標記不改" in text and "乾淨開發集：**28 段**" in text
    u = json.loads((tmp_path / "out" / "usable_dev_segments.json").read_text(encoding="utf-8"))
    assert u["n_usable"] == 28 and len(u["usable_segment_ids"]) == 28 and "a0" not in u["usable_segment_ids"]


def test_build_rejects_missing_segments(tmp_path, monkeypatch):
    recs = _recs()
    clean, blind = _files(tmp_path, recs, lambda r: r["truth"])
    extra = recs + [dict(recs[-1], segment_id="d99")]
    monkeypatch.setattr(clean_dev, "part_records", lambda p: [dict(r) for r in extra if r["part"] == p["key"]])
    with pytest.raises(ValueError, match="對不上"):
        clean_dev.build({"parts": [{"key": k, "name": k} for k in "abcd"]}, clean, blind, frozen_tau=None)
