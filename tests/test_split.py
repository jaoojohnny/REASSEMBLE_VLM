"""第 4 包切分與洩漏檢查（P4-1～P4-5）：規格要求的三個測試＋人造重複資料、時間重疊、鎖定與防看 test。"""

from __future__ import annotations

import csv
import shutil

import numpy as np
import pytest

from assembly_monitor.split import (SplitError, LockedTestSetError, assign_groups, load_split, load_units, make_split,
                                    parse_ratios, verify_split)


def recordings(n_rec=24, seed=1):
    """n_rec 個錄製檔；約三分之一含失敗；每檔 1–3 段（同檔第二段緊接第一段，視窗會重疊）。"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_rec):
        k = int(rng.integers(1, 4))
        segs, t = [], 10.0
        for j in range(k):
            fail = (i % 3 == 0) and j == 0
            segs.append((t, t + 20.0, not fail, [("Approach", 0, 5, True), ("Align", 5, 12, not fail), ("Twist", 12, 18, True)]))
            t += 20.017
        out.append((f"2025-01-{9 + i % 5:02d}-10-{i:02d}-00", segs))
    return out


def split_rows(res):
    with open(res["split_csv"], encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def test_parse_ratios():
    assert parse_ratios("dev=1,test=3") == {"dev": 0.25, "test": 0.75}
    assert list(parse_ratios("train=0.7,val=0.15,test=0.15")) == ["train", "val", "test"]
    assert parse_ratios("test=1") == {"test": 1.0}  # 第 8 包：單一集合（USB 全部當測試集）
    for bad in ("", "dev=0,test=1", "dev=1,dev=2", "dev:1,test:2"):
        with pytest.raises(SplitError):
            parse_ratios(bad)


def test_same_recording_two_segments_same_split(fake_index, tmp_path):
    idx = fake_index(recordings())
    res = make_split(index_dir=idx, out_root=tmp_path / "splits", action="insert", part="BNC", seed=0, hash_frames=False)
    rows = split_rows(res)
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r["recording_id"], set()).add(r["split"])
    assert any(len([x for x in rows if x["recording_id"] == rid]) >= 2 for rid in by_rec)  # 確實有同檔多段
    assert all(len(s) == 1 for s in by_rec.values())
    checks = {c["id"]: c["status"] for c in res["leakage"]["checks"]}
    assert checks["group_crossing"] == "pass" and checks["time_overlap"] == "pass"
    s = res["summary"]["splits"]
    assert s["dev"]["n_segments"] + s["test"]["n_segments"] == len(rows)
    assert s["dev"]["n_failure"] >= 1 and s["test"]["n_failure"] > s["dev"]["n_failure"]


def test_seed_determinism(fake_index):
    units = load_units(fake_index(recordings(40)), "insert", "BNC")
    r = parse_ratios("dev=0.25,test=0.75")
    a1, a2 = assign_groups(units, r, seed=0), assign_groups(units, r, seed=0)
    assert a1 == a2
    assert any(assign_groups(units, r, seed=s) != a1 for s in (1, 2, 3))


def _fake_frames(root, rows, rng):
    """模擬第 3 包輸出：三個相機 × 1/4/8 幀，每段一個資料夾。"""
    import cv2

    for n in (1, 4, 8):
        for r in rows:
            d = root / f"uniform_n{n}_sb0.5_eb2_jpg" / "segments" / r["segment_id"]
            d.mkdir(parents=True, exist_ok=True)
            for cam in ("cam_left", "cam_right", "cam_wrist"):
                for k in range(n):
                    img = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
                    cv2.imwrite(str(d / f"{cam}__{k:02d}__t+000.000s__f{k:06d}.jpg"), img)
            (d / "meta.json").write_text("{}", encoding="utf-8")


def test_derived_samples_never_cross_and_duplicates_are_caught(fake_index, tmp_path):
    pytest.importorskip("cv2")
    idx = fake_index(recordings(12))
    rng = np.random.default_rng(0)
    first = make_split(index_dir=idx, out_root=tmp_path / "s", action="insert", part="BNC", seed=0, hash_frames=False)
    rows = split_rows(first)
    frames = tmp_path / "derived" / "insert_BNC"
    _fake_frames(frames, rows, rng)
    res = make_split(index_dir=idx, out_root=tmp_path / "s", action="insert", part="BNC", seed=0, derived_roots=[frames])
    checks = {c["id"]: c for c in res["leakage"]["checks"]}
    assert checks["derived_samples"]["status"] == "pass"
    assert checks["derived_samples"]["n_files"] == len(rows) * (3 * (1 + 4 + 8) + 3)
    assert checks["image_hash_exact"]["status"] == "pass"
    # 每個衍生檔都只屬於一個集合（同段的各相機、各幀數版本同一集合）
    with open(res["dir"] + "/derived_manifest.csv", encoding="utf-8-sig") as fh:
        man = list(csv.DictReader(fh))
    per_seg = {}
    for m in man:
        per_seg.setdefault(m["segment_id"], set()).add(m["split"])
    assert all(len(v) == 1 for v in per_seg.values())
    # 人造重複：把一張 test 段的影格複製進 dev 段的資料夾
    dev = next(r for r in rows if r["split"] == "dev")
    test = next(r for r in rows if r["split"] == "test")
    src = frames / "uniform_n8_sb0.5_eb2_jpg" / "segments" / test["segment_id"] / "cam_wrist__07__t+000.000s__f000007.jpg"
    shutil.copy(src, frames / "uniform_n8_sb0.5_eb2_jpg" / "segments" / dev["segment_id"] / "copied.jpg")
    res2 = make_split(index_dir=idx, out_root=tmp_path / "s", action="insert", part="BNC", seed=0, derived_roots=[frames])
    checks2 = {c["id"]: c for c in res2["leakage"]["checks"]}
    assert checks2["image_hash_exact"]["status"] == "fail" and res2["leakage"]["status"] == "fail"
    import cv2

    (frames / "uniform_n8_sb0.5_eb2_jpg" / "segments" / dev["segment_id"] / "copied.jpg").unlink()
    # 只有一張相似影格（像是相同場景）：記錄近似對數，但不是重複序列 → 通過
    one = frames / "uniform_n8_sb0.5_eb2_jpg" / "segments" / dev["segment_id"] / "cam_wrist__90__t+000.000s__f000090.jpg"
    cv2.imwrite(str(one), cv2.imread(str(src)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    res3 = make_split(index_dir=idx, out_root=tmp_path / "s", action="insert", part="BNC", seed=0, derived_roots=[frames])
    near3 = {c["id"]: c for c in res3["leakage"]["checks"]}["image_hash_near"]
    assert near3["status"] == "pass" and near3["n_pairs"] >= 1 and not near3["examples"]
    one.unlink()
    # 整段重新編碼後掛在 dev 段底下（位元組不同、每台相機都相似）→ 疑似重複序列 → 警告
    src_dir = frames / "uniform_n8_sb0.5_eb2_jpg" / "segments" / test["segment_id"]
    dup_dir = frames / "recoded_copy" / "segments" / dev["segment_id"]
    dup_dir.mkdir(parents=True)
    for n in (1, 4, 8):
        for p in (frames / f"uniform_n{n}_sb0.5_eb2_jpg" / "segments" / test["segment_id"]).glob("*.jpg"):
            cv2.imwrite(str(dup_dir / p.name.replace("__t", f"__n{n}__t", 1)), cv2.imread(str(p)),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert len(list(dup_dir.glob("*.jpg"))) == len(list(src_dir.glob("*.jpg"))) + 3 * (1 + 4)
    res4 = make_split(index_dir=idx, out_root=tmp_path / "s", action="insert", part="BNC", seed=0, derived_roots=[frames])
    checks4 = {c["id"]: c for c in res4["leakage"]["checks"]}
    assert checks4["image_hash_exact"]["status"] == "pass"
    near4 = checks4["image_hash_near"]
    assert near4["status"] == "warn" and res4["leakage"]["status"] == "warn"
    sus = near4["examples"][0]
    assert {sus["a"], sus["b"]} == {dev["segment_id"], test["segment_id"]} and sus["min_coverage"] >= 0.9
    assert set(sus["coverage"]) == {"cam_left", "cam_right", "cam_wrist"}
    md = (tmp_path / "s" / res4["name"] / "leakage_report.md").read_text(encoding="utf-8")
    assert "疑似重複序列" in md and "| 是 |" in md


def test_frame_camera_and_composites():
    from assembly_monitor.split import frame_camera

    assert frame_camera("uniform_n4_sb0.5_eb2_jpg/segments/2025-01-09-14-28-47__seg11/hama1__00__t+047.722s__f007442.jpg") == "hama1"
    assert frame_camera("pilot30/frames/2025-01-09-15-37-27__seg25/hand__0012.jpg") == "hand"
    assert frame_camera("uniform_n1_sb0.5_eb2_jpg/segments/2025-01-09-14-28-47__seg11/sheet.jpg") is None
    assert frame_camera("pilot30/review/2025-01-09-15-37-27__seg25__hand.jpg") is None
    assert frame_camera("pilot30/review/2025-01-09-15-37-27__seg25__overview.jpg") is None
    assert frame_camera("x/2025-01-09-15-37-27__seg25__overview.jpg") is None


def test_static_scene_is_not_a_duplicate_sequence():
    """外部相機畫面幾乎不變（背景相同）而腕部相機不同：不算疑似重複序列。"""
    from assembly_monitor.split import near_duplicate_sequences

    rng = np.random.default_rng(3)
    static = int(rng.integers(0, 2**62))
    frames = []
    for seg, split in (("r1__seg1", "dev"), ("r2__seg1", "test")):
        for k in range(10):
            frames.append({"segment_id": seg, "cam": "hama1", "dhash": static ^ (1 << int(rng.integers(0, 64))), "path": f"{seg}/hama1__{k}"})
            frames.append({"segment_id": seg, "cam": "hand", "dhash": int(rng.integers(0, 2**62)), "path": f"{seg}/hand__{k}"})
    nd = near_duplicate_sequences(frames, {"r1__seg1": "dev", "r2__seg1": "test"}, threshold=4)
    assert nd["n_pairs"] >= 10 and nd["per_camera"].get("hama1", 0) >= 10
    assert nd["suspects"] == [] and nd["top"][0]["coverage"]["hama1"] == 1.0 and nd["top"][0]["min_coverage"] < 0.5
    # 腕部相機也一樣 → 疑似重複
    for f in frames:
        if f["cam"] == "hand":
            f["dhash"] = static
    nd2 = near_duplicate_sequences(frames, {"r1__seg1": "dev", "r2__seg1": "test"}, threshold=4)
    assert len(nd2["suspects"]) == 1


def test_grouping_by_segment_is_flagged(fake_index, tmp_path):
    idx = fake_index(recordings(30))
    res = make_split(index_dir=idx, out_root=tmp_path / "s", action="insert", part="BNC", seed=0, group_key="segment_id",
                     stratify="none", name="bad", hash_frames=False)
    checks = {c["id"]: c["status"] for c in res["leakage"]["checks"]}
    assert checks["time_overlap"] == "fail" and res["leakage"]["status"] == "fail"


def test_lock_verify_and_test_guard(fake_index, tmp_path):
    idx = fake_index(recordings())
    res = make_split(index_dir=idx, out_root=tmp_path / "s", action="insert", part="BNC", seed=0, hash_frames=False)
    d = res["dir"]
    assert verify_split(d) == []
    dev = load_split(d)
    assert dev and all(r["split"] == "dev" for r in dev)
    with pytest.raises(LockedTestSetError):
        load_split(d, include=("dev", "test"))
    assert all(r["split"] == "test" for r in load_split(d, include=("test",), allow_test=True))
    # 同設定重跑 → 結果相同，可以覆蓋
    make_split(index_dir=idx, out_root=tmp_path / "s", action="insert", part="BNC", seed=0, hash_frames=False)
    # 同名但內容不同 → 拒絕，除非 --force
    with pytest.raises(SplitError):
        make_split(index_dir=idx, out_root=tmp_path / "s", action="insert", part="BNC", seed=5, name=res["name"], hash_frames=False)
    # 手動改動 split.csv → verify 抓得到，load_split 拒絕
    p = tmp_path / "s" / res["name"] / "split.csv"
    text = p.read_text(encoding="utf-8-sig")
    p.write_text(text.replace(",dev,", ",test,", 1), encoding="utf-8-sig")
    assert verify_split(d)
    with pytest.raises(SplitError):
        load_split(d)
