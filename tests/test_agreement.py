"""第 4 包 P3-4：κ 與 IoU、30 段合成 pilot 的一致率與爭議清單、資料集參考標籤、裁決合併。"""

from __future__ import annotations

import csv
import json

import pytest

from assembly_monitor.agreement import (cohen_kappa, compare_pair, compare_sources, interval_iou, load_skills, merge_gold,
                                        reference_from_lowlevel)
from assembly_monitor.schema import SegmentInfo, load_segment_infos, normalize_record


def test_kappa_and_iou():
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("aaaa"), list("aaaa")) == 1.0
    assert cohen_kappa([], []) is None
    assert interval_iou(0, 2, 1, 3) == pytest.approx(1 / 3)
    assert interval_iou(0, 1, 2, 3) == 0.0
    assert interval_iou(5, 5, 5, 5) == 1.0 and interval_iou(5, 5, 6, 6) == 0.0


def pilot30():
    segs = {}
    for i in range(30):
        sid = f"2025-01-{9 + i % 5:02d}-10-{i:02d}-00__seg{i}"
        segs[sid] = SegmentInfo(segment_id=sid, recording_id=sid.split("__")[0], segment_index=i, action="insert", part="BNC",
                                text="Insert BNC.", start_unix=1000.0 * i, end_unix=1000.0 * i + 20.0,
                                dataset_result="failure" if i < 20 else "success", validation_status="ok")
    return segs


def rec(seg, annot, atype="human", **kw):
    reasons = ["misalignment", "incomplete_insert", "slip_or_drop", "wrong_contact"]
    fail = seg.dataset_result == "failure"
    r = {"segment_id": seg.segment_id, "recording_id": seg.recording_id, "segment_index": seg.segment_index, "part": "BNC",
         "dataset_result": seg.dataset_result, "result": seg.dataset_result,
         "failure_reason": reasons[seg.segment_index % 4] if fail else None,
         "evidence_start_sec": 12.0, "evidence_end_sec": 18.0, "best_camera": "cam_wrist",
         "camera_visibility": {"cam_left": "partial", "cam_right": "occluded", "cam_wrist": "clear"},
         "visible_evidence": "x", "annotator": annot, "annotator_type": atype, "label_version": "v1", "review_status": "submitted"}
    r.update(kw)
    return r


def test_pilot30_agreement_and_disputes(ann_cfg, tmp_path):
    segs = pilot30()
    ids = list(segs)
    A = [normalize_record(rec(segs[s], "A"), ann_cfg) for s in ids]
    B = []
    for i, s in enumerate(ids):
        kw = {}
        if i in (0, 1, 2):
            kw["failure_reason"] = "unknown"               # 3 段原因不同
        if i in (3, 21):
            kw["best_camera"] = "cam_left"                 # 2 段視角不同
        if i in (4, 22):
            kw.update(evidence_start_sec=2.0, evidence_end_sec=5.0)   # 2 段證據區間不重疊
        if i == 5:
            kw.update(evidence_start_sec=12.5, evidence_end_sec=18.8)  # 小偏移：仍算命中
        if i == 6:
            kw.update(result="uncertain", failure_reason=None)       # 1 段結果不同
        if i == 7:
            kw["camera_visibility"] = {"cam_left": "clear"}           # 只有可見度不同 → 不算爭議
        B.append(normalize_record(rec(segs[s], "claude", "ai", **kw), ann_cfg))
    res = compare_pair({r["segment_id"]: r for r in A}, {r["segment_id"]: r for r in B}, ann_cfg, ("A", "B"))
    m = res["metrics"]
    assert m["n_common"] == 30
    assert m["result"]["agree"] == 29 and m["result"]["n"] == 30
    assert m["failure_reason"]["n"] == 19 and m["failure_reason"]["agree"] == 16   # 第 6 段 B 判 uncertain，不算原因
    assert m["best_camera"]["agree"] == 28
    assert m["evidence"]["n"] == 30 and m["evidence"]["hit"] == 28
    # 預設設定：證據區間差異只列入統計，不需裁決（compare.blocking_fields）
    assert ann_cfg.blocking_fields == ["result", "failure_reason", "best_camera"]
    assert m["n_disputes"] == 6 and sorted(int(r["segment_id"].split("seg")[1]) for r in res["disputes"]) == [0, 1, 2, 3, 6, 21]
    assert m["n_with_differences"] == 8 and m["difference_fields"]["evidence"] == 2
    seg4 = next(r for r in res["rows"] if r["segment_id"].endswith("seg4"))
    assert seg4["differences"] == ["evidence"] and seg4["disputes"] == []
    assert m["visibility"]["cam_left"]["agree"] == 29
    assert m["failure_reason"]["confusion"]["misalignment"]["unknown"] == 1
    assert m["result"]["kappa_ci95"] is not None
    # 三方比較並輸出檔案
    skills = {}
    ref = reference_from_lowlevel(segs, skills, ann_cfg)
    out = compare_sources({"A": A, "B": B, "C": ref}, ann_cfg, tmp_path / "rep", "pilot30")
    assert out["second"] == "B" and out["reference"] == "C" and len(out["disputes"]) == 6
    with open(out["disputes_csv"], encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 and all(r["differences"] for r in rows)
    js = (tmp_path / "rep" / "compare_data.js").read_text(encoding="utf-8")
    data = json.loads(js[js.index("=") + 1 : js.rstrip().rindex(";")])
    assert set(data["sources"]) == {"A", "B", "C"} and len(data["disputes"]) == 6
    md = (tmp_path / "rep" / "agreement.md").read_text(encoding="utf-8")
    assert "爭議清單" in md and "A vs B" in md and "A vs C" in md and "blocking_fields" in md


def test_evidence_blocking_option_and_zero_length(ann_cfg):
    import dataclasses

    segs = pilot30()
    ids = list(segs)[:4]
    A = {s: normalize_record(rec(segs[s], "A"), ann_cfg) for s in ids}
    B = {s: normalize_record(rec(segs[s], "claude", "ai"), ann_cfg) for s in ids}
    B[ids[0]].update(evidence_start_sec=2.0, evidence_end_sec=5.0)          # 只有證據區間不同
    A[ids[1]].update(evidence_start_sec=13.4, evidence_end_sec=13.4)        # A 的區間長度為 0（起點＝終點）
    res = compare_pair(A, B, ann_cfg, ("A", "B"))
    by = {r["segment_id"]: r for r in res["rows"]}
    assert by[ids[0]]["disputes"] == [] and "evidence" in by[ids[0]]["differences"]
    assert by[ids[1]]["disputes"] == ["evidence_length"] and by[ids[1]]["differences"] == ["evidence", "evidence_length"]
    assert by[ids[1]]["evidence_too_short"] == ["A"]
    strict = dataclasses.replace(ann_cfg, blocking_fields=["result", "failure_reason", "best_camera", "evidence"])
    res2 = compare_pair(A, B, strict, ("A", "B"))
    assert {r["segment_id"] for r in res2["disputes"]} == {ids[0], ids[1]}


def test_reference_from_lowlevel(fake_index, ann_cfg):
    idx = fake_index([
        ("2025-01-09-10-00-00", [(10.0, 30.0, False, [("Approach", 0, 5, True), ("Align", 5, 12, False), ("Twist", 12, 18, False)]),
                                 (30.017, 50.0, True, [("Approach", 0, 5, True), ("Twist", 6, 15, True), ("Release", 15, 19.9, True)])]),
        ("2025-01-09-11-00-00", [(10.0, 30.0, False, [("Approach", 0, 5, True), ("Align", 5, 12, True), ("Twist", 12, 25, False)]),
                                 (40.0, 50.0, False, [("Approach", 0, 5, True)])]),
    ])
    segs = load_segment_infos(idx, "insert", "BNC")
    refs = {r["segment_id"]: r for r in reference_from_lowlevel(segs, load_skills(idx), ann_cfg)}
    a = refs["2025-01-09-10-00-00__seg0"]
    assert a["failure_reason"] == "misalignment" and (a["evidence_start_sec"], a["evidence_end_sec"]) == (5.0, 18.0)
    assert refs["2025-01-09-10-00-00__seg1"]["failure_reason"] is None
    assert (refs["2025-01-09-10-00-00__seg1"]["evidence_start_sec"], refs["2025-01-09-10-00-00__seg1"]["evidence_end_sec"]) == (15.0, 19.9)
    b = refs["2025-01-09-11-00-00__seg0"]
    assert b["failure_reason"] == "incomplete_insert" and b["evidence_end_sec"] == 22.0  # 超出視窗的部分被截到終點 + 2 秒
    assert refs["2025-01-09-11-00-00__seg1"]["failure_reason"] == "unknown"
    assert refs["2025-01-09-11-00-00__seg1"]["evidence_start_sec"] is None
    assert all(r["annotator_type"] == "reference" and "低階技能" in r["notes"] for r in refs.values())


def test_merge_gold(ann_cfg):
    from assembly_monitor.schema import validate_record

    segs = pilot30()
    ids = list(segs)[:4]
    A = [normalize_record(rec(segs[s], "A"), ann_cfg) for s in ids]               # pilot 用 v1 標的
    adj = [normalize_record(rec(segs[ids[0]], "A", failure_reason="board_lifted", review_status="adjudicated",
                                label_version=ann_cfg.label_version, visual_difficulty="subtle"), ann_cfg)]
    gold, rep = merge_gold(A, adj, disputes=[ids[0], ids[1]], cfg=ann_cfg)
    by = {g["segment_id"]: g for g in gold}
    assert by[ids[0]]["review_status"] == "adjudicated" and by[ids[0]]["failure_reason"] == "board_lifted"
    assert by[ids[0]]["visual_difficulty"] == "subtle" and "visual_difficulty_source" not in by[ids[0]]
    assert by[ids[1]]["review_status"] == "disputed" and rep["disputed_unresolved"] == [ids[1]]
    assert by[ids[1]]["visual_difficulty"] is None and by[ids[1]]["label_version"] == "v1"   # 未裁決的不算正式答案，不改版本
    assert by[ids[2]]["review_status"] == "approved" and rep["approved"] == 2 and rep["adjudicated"] == 1
    # 正式答案蓋上目前版本，原版本另存；無爭議段補預設難度
    assert by[ids[2]]["label_version"] == ann_cfg.label_version and by[ids[2]]["reason_set"] == ann_cfg.reason_set
    assert by[ids[2]]["annotated_label_version"] == "v1"
    assert by[ids[2]]["visual_difficulty"] == ann_cfg.difficulty_default == "clear"
    assert by[ids[2]]["visual_difficulty_source"] == "default_for_approved" and rep["difficulty_defaulted"] == [ids[2], ids[3]]
    assert rep["visual_difficulty"] == {"subtle": 1, "clear": 2} and rep["failure_reasons"]["board_lifted"] == 1
    for sid in (ids[0], ids[2], ids[3]):
        assert validate_record(by[sid], ann_cfg, segs[sid]) == []


def test_merge_gold_fills_difficulty_for_reexported_undisputed(ann_cfg):
    """裁決頁也匯出了沒有爭議的段（沒填難度）：沿用主標註者的難度，沒有就補預設；爭議段不補。"""
    segs = pilot30()
    ids = list(segs)[:4]
    A = [normalize_record(rec(segs[ids[0]], "A"), ann_cfg),
         normalize_record(rec(segs[ids[1]], "A", label_version=ann_cfg.label_version, visual_difficulty="subtle"), ann_cfg),
         normalize_record(rec(segs[ids[2]], "A"), ann_cfg)]
    adj = [normalize_record(rec(segs[s], "A", review_status="adjudicated", label_version=ann_cfg.label_version), ann_cfg)
           for s in ids[:3]]
    gold, rep = merge_gold(A, adj, disputes=[ids[2]], cfg=ann_cfg)
    by = {g["segment_id"]: g for g in gold}
    assert (by[ids[0]]["visual_difficulty"], by[ids[0]]["visual_difficulty_source"]) == ("clear", "default_for_approved")
    assert (by[ids[1]]["visual_difficulty"], by[ids[1]]["visual_difficulty_source"]) == ("subtle", "primary_annotation")
    assert by[ids[2]]["visual_difficulty"] is None   # 爭議段一定要裁決者自己選
    assert rep["difficulty_defaulted"] == [ids[0]]
