"""第 4 包標註格式（P3-1、P3-2）：非法類別與時間被拒、草稿可不完整、CSV／JSONL 往返不遺失、合併不覆蓋原始標籤。"""

from __future__ import annotations

import json
import math

import pytest

from assembly_monitor.schema import (SegmentInfo, json_schema, merge_annotations, normalize_record, read_annotations,
                                     validate_record, validate_records, write_csv, write_jsonl)

SEG = SegmentInfo(segment_id="2025-01-09-14-28-47__seg11", recording_id="2025-01-09-14-28-47", segment_index=11,
                  action="insert", part="BNC", text="Insert BNC.", start_unix=1000.0, end_unix=1020.0,
                  dataset_result="failure", validation_status="ok")


def good(**kw):
    r = {"segment_id": SEG.segment_id, "recording_id": SEG.recording_id, "segment_index": 11, "part": "BNC",
         "dataset_result": "failure", "result": "failure", "failure_reason": "misalignment",
         "evidence_start_sec": 12.0, "evidence_end_sec": 15.5, "best_camera": "cam_wrist",
         "camera_visibility": {"cam_left": "partial", "cam_right": "occluded", "cam_wrist": "clear"},
         "visible_evidence": "接頭停在插座外側", "notes": "", "annotator": "A", "annotator_type": "human",
         "label_version": "v1", "reason_set": "bnc-pilot-v1", "review_status": "submitted"}
    r.update(kw)
    return r


def test_valid_and_illegal_categories(ann_cfg):
    assert validate_record(good(), ann_cfg, SEG) == []
    bad = {
        "result": good(result="partial"),
        "reason": good(failure_reason="bent_pin"),
        "reason_on_success": good(result="success", dataset_result="failure"),
        "no_reason": good(failure_reason=None),
        "other_without_notes": good(failure_reason="other"),
        "camera": good(best_camera="top"),
        "visibility": good(camera_visibility={"cam_left": "maybe"}),
        "unknown_camera_vis": good(camera_visibility={"cam_top": "clear"}),
        "status": good(review_status="done"),
        "annotator": good(annotator=" "),
        "segment_id": good(segment_id="seg11"),
        "no_evidence_text": good(visible_evidence=""),
    }
    for name, r in bad.items():
        assert validate_record(r, ann_cfg, SEG), name
    assert validate_record(good(failure_reason="other", notes="接頭卡在旁邊的螺絲"), ann_cfg, SEG) == []
    assert validate_record(good(failure_reason="unknown"), ann_cfg, SEG) == []
    # 原始標籤不可改
    assert any("原始標籤" in e for e in validate_record(good(dataset_result="success"), ann_cfg, SEG))


def test_time_rules(ann_cfg):
    w1 = SEG.duration + ann_cfg.end_buffer
    assert validate_record(good(evidence_start_sec=-0.5, evidence_end_sec=w1), ann_cfg, SEG) == []
    assert validate_record(good(evidence_start_sec=-0.53, evidence_end_sec=w1 + 0.04), ann_cfg, SEG) == []  # 容許誤差內
    for a, b in ((16.0, 15.0), (-1.0, 3.0), (5.0, w1 + 1.0), (math.nan, 3.0), ("12", 15.0), (True, 15.0), (12.0, None)):
        assert validate_record(good(evidence_start_sec=a, evidence_end_sec=b), ann_cfg, SEG), (a, b)


def test_draft_may_be_incomplete(ann_cfg):
    d = good(review_status="draft", failure_reason=None, evidence_start_sec=None, evidence_end_sec=None,
             best_camera=None, visible_evidence="")
    assert validate_record(d, ann_cfg, SEG) == []
    assert validate_record(dict(d, review_status="submitted"), ann_cfg, SEG)
    assert validate_record(dict(d, evidence_start_sec=30.0, evidence_end_sec=31.0), ann_cfg, SEG)  # 草稿也查範圍
    ref = good(annotator="dataset_lowlevel", annotator_type="reference", best_camera=None, visible_evidence="")
    assert validate_record(ref, ann_cfg, SEG) == []  # 參考標籤不需要視角與文字依據


def test_validate_records_duplicates_and_unknown(ann_cfg):
    recs = [good(), good(), good(segment_id="2025-01-09-14-28-47__seg99")]
    rep = validate_records(recs, ann_cfg, {SEG.segment_id: SEG})
    msgs = [e for p in rep["problems"] for e in p["errors"]]
    assert any("重複" in m for m in msgs) and any("找不到這個 segment_id" in m for m in msgs)
    assert not rep["ok"]


def test_json_schema_rejects_illegal_values(ann_cfg):
    jsonschema = pytest.importorskip("jsonschema")
    sch = json_schema(ann_cfg)
    jsonschema.Draft202012Validator.check_schema(sch)
    v = jsonschema.Draft202012Validator(sch)
    assert not list(v.iter_errors(good()))
    for r in (good(result="partial"), good(failure_reason="bent_pin"), good(result="success", failure_reason="misalignment"),
              good(failure_reason=None), good(best_camera="top"), good(evidence_start_sec="12"), good(evidence_start_sec=-9.0),
              good(review_status="done"), good(failure_reason="other", notes=""), good(visible_evidence="")):
        assert list(v.iter_errors(r)), r
    assert not list(v.iter_errors(good(review_status="draft", failure_reason=None, best_camera=None, visible_evidence="",
                                        evidence_start_sec=None, evidence_end_sec=None)))


def test_csv_jsonl_roundtrip_keeps_notes_and_labels(ann_cfg, tmp_path):
    recs = [normalize_record(good(notes='第一行，有逗號\n第二行 "引號"', extra_col="保留我"), ann_cfg),
            normalize_record(good(segment_id="2025-01-09-14-28-47__seg12", segment_index=12, result="uncertain",
                                  failure_reason=None, dataset_result="success", evidence_start_sec=None,
                                  evidence_end_sec=None, best_camera=None, visible_evidence="", notes="看不清楚"), ann_cfg)]
    pc = write_csv(recs, tmp_path / "a.csv", ann_cfg)
    pj = write_jsonl(recs, tmp_path / "a.jsonl", ann_cfg)
    assert pc.read_bytes()[:3] == b"\xef\xbb\xbf"  # Excel 用的 BOM
    back_c = [normalize_record(r, ann_cfg) for r in read_annotations(pc, ann_cfg)]
    back_j = [normalize_record(r, ann_cfg) for r in read_annotations(pj, ann_cfg)]
    assert back_j == recs
    assert back_c == recs
    # 再轉一次也一樣
    pc2 = write_csv(back_j, tmp_path / "b.csv", ann_cfg)
    assert pc2.read_bytes() == pc.read_bytes()


def test_merge_absent_column_keeps_value_and_label_conflict(ann_cfg, tmp_path):
    base = [normalize_record(good(notes="原本的備註"), ann_cfg)]
    # 欄位缺席 → 保留
    p = tmp_path / "edit.csv"
    p.write_text("segment_id,annotator,result,failure_reason\n"
                 f"{SEG.segment_id},A,failure,incomplete_insert\n", encoding="utf-8")
    merged, rep = merge_annotations(base, read_annotations(p, ann_cfg), ann_cfg)
    assert merged[0]["notes"] == "原本的備註" and merged[0]["failure_reason"] == "incomplete_insert" and rep["updated"] == 1
    assert merged[0]["dataset_result"] == "failure" and merged[0]["visible_evidence"] == "接頭停在插座外側"
    # 欄位存在但空白 → 清除；原始標籤被改 → 不覆蓋並回報
    p.write_text(f"segment_id,annotator,notes,dataset_result\n{SEG.segment_id},A,,success\n", encoding="utf-8")
    merged2, rep2 = merge_annotations(merged, read_annotations(p, ann_cfg), ann_cfg)
    assert merged2[0]["notes"] == "" and merged2[0]["dataset_result"] == "failure"
    assert rep2["conflicts"] and rep2["conflicts"][0]["field"] == "dataset_result"


def test_big5_csv_from_excel(ann_cfg, tmp_path):
    p = tmp_path / "excel.csv"
    p.write_bytes(f"segment_id,annotator,result,visible_evidence\r\n{SEG.segment_id},A,failure,接頭歪掉\r\n".encode("cp950"))
    r = read_annotations(p, ann_cfg)[0]
    assert r["visible_evidence"] == "接頭歪掉" and r["result"] == "failure"


def test_jsonl_errors_are_reported(ann_cfg, tmp_path):
    from assembly_monitor.schema import AnnotationError

    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(good()) + "\n{壞掉\n", encoding="utf-8")
    with pytest.raises(AnnotationError):
        read_annotations(p, ann_cfg)


def test_v2_visual_difficulty_and_min_length(ann_cfg):
    """v2：畫面難度（目前版本的正式紀錄必填）、證據區間最短長度；舊版紀錄不受影響。"""
    from assembly_monitor.schema import SegmentInfo, json_schema, validate_record

    seg = SegmentInfo(segment_id="2025-01-09-10-00-00__seg1", recording_id="2025-01-09-10-00-00", segment_index=1, action="insert",
                      part="BNC", text="Insert BNC.", start_unix=100.0, end_unix=120.0, dataset_result="failure", validation_status="ok")
    assert ann_cfg.label_version == "v2" and set(ann_cfg.difficulty) == {"clear", "subtle", "not_visible"}
    assert "board_lifted" in ann_cfg.reasons and ann_cfg.public()["difficulty_default"] == "clear"
    base = {"segment_id": seg.segment_id, "dataset_result": "failure", "result": "failure", "failure_reason": "board_lifted",
            "evidence_start_sec": 15.3, "evidence_end_sec": 18.0, "best_camera": "cam_left", "visible_evidence": "板子被抬起",
            "annotator": "A", "annotator_type": "human", "label_version": "v2", "review_status": "submitted"}
    errs = validate_record(base, ann_cfg, seg)
    assert any("visual_difficulty" in e for e in errs)
    assert validate_record({**base, "visual_difficulty": "subtle"}, ann_cfg, seg) == []
    assert any("visual_difficulty" in e for e in validate_record({**base, "visual_difficulty": "hard"}, ann_cfg, seg))
    assert validate_record({**base, "review_status": "draft"}, ann_cfg, seg) == []                 # 草稿可以先不填
    v1 = {**base, "label_version": "v1", "failure_reason": "wrong_contact", "evidence_end_sec": 15.3}
    assert validate_record(v1, ann_cfg, seg) == []                                                 # 舊版：不要求難度、不查長度
    short = {**base, "visual_difficulty": "clear", "evidence_end_sec": 15.35}
    assert any("太短" in e for e in validate_record(short, ann_cfg, seg))
    ref = {**base, "annotator_type": "reference", "best_camera": None, "visible_evidence": ""}
    assert validate_record(ref, ann_cfg, seg) == []                                                # 參考標籤不要求
    sch = json_schema(ann_cfg)
    assert sch["properties"]["visual_difficulty"]["enum"][-1] is None
    try:
        import jsonschema
    except ImportError:
        return
    v = jsonschema.Draft202012Validator(sch) if hasattr(jsonschema, "Draft202012Validator") else jsonschema.Draft7Validator(sch)
    assert list(v.iter_errors(base))                                  # v2 正式紀錄缺難度 → schema 也擋
    assert not list(v.iter_errors({**base, "visual_difficulty": "not_visible"}))
    assert not list(v.iter_errors(v1))
