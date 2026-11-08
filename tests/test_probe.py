"""做法 A：訓練池排除測試錄影與目標零件、依錄影分組交叉驗證、輸出格式與 VLM 相同。不需要 GPU。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from assembly_monitor import probe, vlm  # noqa: E402

CAMS = ["hama1", "hama2", "hand"]
SAMPLING = "uniform_n{n}_sb0.5_eb2_jpg"


def _frames(root: Path, part_slug: str, sid: str, n: int, fail: bool, seed: int) -> None:
    d = root / f"insert_{part_slug}" / SAMPLING.format(n=n) / "segments" / sid
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    times = [12.0] if n == 1 else [-0.5 + 12.5 * i / (n - 1) for i in range(n)]
    cams = []
    for c in CAMS:
        paths = []
        for k, _t in enumerate(times):
            base = 200 if fail else 60  # 失敗段整體偏亮：讓假特徵可以分得開
            img = np.clip(rng.normal(base, 20, (48, 64, 3)), 0, 255).astype(np.uint8)
            name = f"{c}__{k:02d}.jpg"
            cv2.imwrite(str(d / name), img)
            paths.append(name)
        cams.append({"camera": c, "frame_paths": paths, "source_times_sec": times})
    meta = {"start_unix": 0.0, "end_unix": 10.0, "duration_s": 10.0, "window_unix": [-0.5, 12.0], "cameras": cams}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    """3 種零件 × 多支錄影；BNC 的 dev/test 由 split.csv 決定。"""
    idx, split, frames = tmp_path / "index", tmp_path / "split", tmp_path / "frames"
    idx.mkdir()
    split.mkdir()
    seg_rows, split_rows = [], []
    k = 0
    for r in range(12):
        rid = f"2025-01-0{1 + r // 10}-1{r % 10}-00-00"
        role = "test" if r < 4 else ("dev" if r < 6 else "other")
        for j, part in enumerate(("BNC", "USB", "bolt 4")):
            fail = (r + j) % 3 == 0
            sid = f"{rid}__seg{j + 1}"
            seg_rows.append({"recording_id": rid, "segment_id": str(j + 1), "action": "insert", "part": part,
                             "success": str(not fail), "validation_status": "ok"})
            if part == "BNC" and role in ("test", "dev"):
                split_rows.append({"segment_id": sid, "recording_id": rid, "split": role,
                                   "dataset_result": "failure" if fail else "success"})
            for n in (1, 4):
                _frames(frames, part.replace(" ", "_"), sid, n, fail, k)
                k += 1
    with (idx / "segments.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(seg_rows[0]))
        w.writeheader()
        w.writerows(seg_rows)
    with (split / "split.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(split_rows[0]))
        w.writeheader()
        w.writerows(split_rows)
    cfg = {"name": "probe_fake", "experiment_name": "dev_t", "final_experiment_name": "final_t", "split_dir": str(split),
           "index_dir": str(idx), "action": "insert", "target_part": "BNC", "frames_root": str(frames),
           "sampling_template": SAMPLING, "frames": [1, 4], "cameras": CAMS, "out_dir": str(tmp_path / "exp"),
           "encoder": {"type": "fake"}, "classifier": {"C_grid": [0.1, 1.0], "pca_grid": [4], "cv_folds": 3, "seed": 0}}
    return cfg, split_rows


@pytest.fixture
def fake_load_split(monkeypatch):
    import assembly_monitor.split as split_mod

    calls = []

    def _ls(split_dir, include=("dev",), allow_test=False):
        inc = list(include)
        calls.append((inc, allow_test))
        if "test" in inc and not allow_test:
            raise split_mod.LockedTestSetError("locked")
        rows = list(csv.DictReader(open(Path(split_dir) / "split.csv", encoding="utf-8")))
        return [r for r in rows if r["split"] in inc]

    monkeypatch.setattr(split_mod, "load_split", _ls)
    return calls


def test_pool_excludes_target_part_and_test_recordings(world, fake_load_split):
    cfg, split_rows = world
    pools = probe.build_pools(cfg)
    test_recs = {r["recording_id"] for r in split_rows if r["split"] == "test"}
    assert pools["train"] and all(t["part"] != "BNC" for t in pools["train"])
    assert not ({t["recording_id"] for t in pools["train"]} & test_recs)
    assert len(pools["train"]) == 8 * 2 and len(pools["dev"]) == 2 and pools["test"] == []
    assert fake_load_split == [(["dev"], False)]
    s = probe.pool_summary(pools["train"])
    assert set(s["per_part"]) == {"USB", "bolt 4"} and s["n_recordings"] == 8


def test_run_probe_writes_vlm_compatible_outputs(world, fake_load_split):
    cfg, _ = world
    rep = probe.run_probe(cfg)
    assert [r["n_frames"] for r in rep["results"]] == [1, 4]
    for r in rep["results"]:
        assert r["n_segments"] == 2 and r["n_ok"] == 2
        assert r["cv"]["cv_balanced_accuracy"] >= 0.9  # 假特徵：亮度可以分開成功／失敗
    rd = Path(cfg["out_dir"]) / "dev_t" / "probe_fake" / "f4"
    preds = vlm._read_jsonl(rd / "predictions.jsonl")
    assert {p["result"] for p in preds} <= {"success", "failure"} and all(0.5 <= p["confidence"] <= 1 for p in preds)
    rc = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
    assert rc["eval_split"] == "dev" and rc["train_pool"]["n"] == 16 and rc["excluded_test_recordings"] == 4
    md = Path(rep["summary"]["md"]).read_text(encoding="utf-8")
    assert "| probe_fake | 1 |" in md and "| probe_fake | 4 |" in md
    # 特徵有快取：第二次不再讀影格
    cache = list((Path(cfg["out_dir"]) / "_features" / "fake").rglob("*.npy"))
    assert len(cache) == (16 + 2) * 2


def test_probe_test_set_needs_unlock(world, fake_load_split):
    cfg, _ = world
    rep = probe.run_probe(cfg, frames=[1], allow_test=True)
    assert rep["eval_split"] == "test" and rep["results"][0]["n_segments"] == 4
    assert (Path(cfg["out_dir"]) / "final_t" / "probe_fake" / "f1" / "predictions.jsonl").exists()
    assert (["test"], True) in fake_load_split


def test_probe_rerun_keeps_timing(world, fake_load_split):
    """第 8 包修正：重跑時特徵都在快取，每段時間沿用第一次當場編碼的量測，不會被清掉。"""
    cfg, _ = world
    probe.run_probe(cfg, frames=[1], allow_test=True)
    rd = Path(cfg["out_dir"]) / "final_t" / "probe_fake" / "f1"
    first = {r["segment_id"]: r for r in vlm._read_jsonl(rd / "raw.jsonl")}
    assert first and all(r["encode_s"] is not None and "timing_from" not in r for r in first.values())
    probe.run_probe(cfg, frames=[1], allow_test=True)
    second = {r["segment_id"]: r for r in vlm._read_jsonl(rd / "raw.jsonl")}
    assert set(second) == set(first)
    for sid, r in second.items():
        assert r["encode_s"] == first[sid]["encode_s"] and r["timing_from"] == first[sid]["created_at"]
    probe.run_probe(cfg, frames=[1], allow_test=True)  # 第三次：仍指向第一次量測的時間
    third = {r["segment_id"]: r for r in vlm._read_jsonl(rd / "raw.jsonl")}
    assert all(third[s]["timing_from"] == first[s]["created_at"] for s in third)


def test_to_vector_fills_missing_with_zero():
    f = np.full((2, 3, 4), np.nan, np.float32)
    f[0, 0] = 1
    v = probe.to_vector(f)
    assert v.shape == (24,) and v[:4].tolist() == [1, 1, 1, 1] and not np.isnan(v).any()


def test_repo_probe_config_loads():
    root = Path(__file__).resolve().parent.parent
    cfg = probe.load_probe_config(root / "configs/probe.yaml")
    assert cfg["target_part"] == "BNC" and cfg["encoder"]["hf_id"] == "facebook/dinov2-base"
    models = vlm.load_model_configs(root / "configs/model.yaml")
    assert models["dinov2_base"]["type"] == "encoder"
    with pytest.raises(vlm.VLMConfigError):
        vlm.make_adapter("dinov2_base", models["dinov2_base"])


def test_probe_end_buffer_configs(tmp_path):
    from assembly_monitor import probe as pr

    for tag, eb in (("eb1", 1.0), ("eb05", 0.5)):
        cfg = pr.load_probe_config(ROOT / "configs" / f"probe_{tag}.yaml")
        assert cfg["end_buffer"] == eb and cfg["name"] == f"probe_dinov2_{tag}" and cfg["experiment_name"] != "dev_smoke"
    bad = tmp_path / "bad.yaml"
    bad.write_text((ROOT / "configs/probe_eb1.yaml").read_text(encoding="utf-8").replace("end_buffer: 1 ", "end_buffer: 3 "),
                   encoding="utf-8")
    with pytest.raises(pr.ProbeConfigError):
        pr.load_probe_config(bad)
