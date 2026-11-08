"""再訓練：VLM 特徵＋小分類器（不需要 GPU：用假特徵測訓練、挑選與輸出；確認不讀測試集）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("cv2")

from assembly_monitor import vlm, vlm_probe  # noqa: E402
from test_probe import SAMPLING, fake_load_split, world  # noqa: E402,F401


def test_repo_config():
    cfg = vlm_probe.load_vlm_probe_config(ROOT / "configs/vlm_probe.yaml")
    assert cfg["base_model"] == "qwen3vl_8b" and cfg["n_frames"] == 4 and cfg["features"] == ["zs", "L50", "L75", "L100"]
    assert cfg["part_names"]["D-SUB"] == "D-SUB connector"  # 和 LoRA 訓練時相同
    exp = vlm.load_experiment(ROOT / cfg["prompt_experiment"])
    assert exp["sampling_template"] == "uniform_n{n}_sb0.5_eb0.5_jpg" and exp["include"] == ["dev"]
    assert "adapter_path" not in vlm.load_model_configs(ROOT / cfg["models"])[cfg["base_model"]]


def test_layer_index_and_names():
    assert [vlm_probe.layer_index(f, 37) for f in (0.5, 0.75, 1.0)] == [18, 27, 36]
    assert vlm_probe.layer_index(0.01, 37) == 1  # 不會取到嵌入層
    assert [vlm_probe.layer_name(f) for f in (0.5, 0.75, 1.0)] == ["L50", "L75", "L100"]


def test_config_validation(tmp_path):
    base = yaml.safe_load((ROOT / "configs/vlm_probe.yaml").read_text(encoding="utf-8"))
    base["vlm_probe"].pop("part_names_from")
    for bad in ({"layers": [1.5]}, {"features": ["zs", "L33"]}):
        d = {"vlm_probe": dict(base["vlm_probe"], **bad)}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(d), encoding="utf-8")
        with pytest.raises(vlm_probe.VLMProbeConfigError):
            vlm_probe.load_vlm_probe_config(p)


def test_select_best_rule():
    rows = [{"order": 1, "cv_balanced_accuracy": 0.8, "cv_failure_recall": 0.6, "dim": 4096},
            {"order": 2, "cv_balanced_accuracy": 0.8, "cv_failure_recall": 0.6, "dim": 1},
            {"order": 3, "cv_balanced_accuracy": 0.79, "cv_failure_recall": 0.9, "dim": 1}]
    assert vlm_probe.select_best(rows)["order"] == 2
    assert vlm_probe.select_best([]) is None


def _cfg(world_cfg, tmp_path):
    exp = {"experiment": {"name": "dev_vp_t", "split_dir": world_cfg["split_dir"], "include": ["dev"],
                          "frames_root": str(Path(world_cfg["frames_root"]) / "insert_BNC"), "sampling_template": SAMPLING,
                          "frames": [4], "cameras": world_cfg["cameras"], "out_dir": str(tmp_path / "exp"),
                          "prompt": {"version": "ft-generic", "template": str(ROOT / "configs/prompts/ft_generic.txt")},
                          "task": vlm.load_experiment(ROOT / "configs/experiment_ft.yaml")["task"]}}
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(exp, allow_unicode=True), encoding="utf-8")
    d = yaml.safe_load((ROOT / "configs/vlm_probe.yaml").read_text(encoding="utf-8"))["vlm_probe"]
    d.pop("part_names_from")
    d.update(prompt_experiment=str(tmp_path / "exp.yaml"), split_dir=world_cfg["split_dir"], index_dir=world_cfg["index_dir"],
             frames_root=world_cfg["frames_root"], feature_dir=str(tmp_path / "feat"), cache_dir=str(tmp_path / "cache"),
             out_dir=str(tmp_path / "out"), selection_out=str(tmp_path / "sel"),
             classifier={"C_grid": [0.1, 1.0], "pca_grid": [2], "cv_folds": 3, "seed": 0})
    p = tmp_path / "vp.yaml"
    p.write_text(yaml.safe_dump({"vlm_probe": d}, allow_unicode=True), encoding="utf-8")
    return vlm_probe.load_vlm_probe_config(p)


def test_fit_with_fake_features(world, fake_load_split, tmp_path, monkeypatch):
    wcfg, _ = world
    monkeypatch.chdir(tmp_path)  # 開發集報告找 gold 的相對路徑：這裡沒有 gold
    cfg = _cfg(wcfg, tmp_path)
    exp, data = vlm_probe.samples(cfg)
    assert len(data["train"]) == 16 and len(data["dev"]) == 2
    rng = np.random.default_rng(0)
    for s in data["train"] + data["dev"]:
        y = 1.0 if s["result"] == "failure" else -1.0
        fp = vlm_probe.feature_file(cfg, s["segment_id"])
        fp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fp, meta=np.array(json.dumps({"segment_id": s["segment_id"]})),
                 zs=np.array([3 * y + rng.normal()], np.float32),           # 分得開
                 L50=rng.normal(size=8).astype(np.float32),                  # 純雜訊
                 L75=(rng.normal(size=8) + y).astype(np.float32),
                 L100=rng.normal(size=8).astype(np.float32))
    X, keep, miss = vlm_probe.load_matrix(cfg, data["train"], "L75")
    assert X.shape == (16, 8) and not miss
    pl = vlm_probe.describe_plan(cfg)
    assert pl["features_done"] == {"train": 16, "dev": 2}
    rep = vlm_probe.fit(cfg, n_boot=50)
    assert [r["feature"] for r in rep["rows"]] == ["zs", "L50", "L75", "L100"]
    zs = next(r for r in rep["rows"] if r["feature"] == "zs")
    assert zs["cv_balanced_accuracy"] >= 0.9 and zs["dim"] == 1
    assert rep["best"]["cv_balanced_accuracy"] == max(r["cv_balanced_accuracy"] for r in rep["rows"])
    rd = Path(cfg["out_dir"]) / cfg["experiment_name"] / f"{cfg['name']}_zs" / "f4"
    preds = vlm._read_jsonl(rd / "predictions.jsonl")
    assert len(preds) == 2 and {p["result"] for p in preds} <= {"success", "failure"}
    rc = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
    assert rc["feature"] == "zs" and rc["eval_split"] == "dev" and rc["train_n"] == 16
    assert "★" in Path(rep["md"]).read_text(encoding="utf-8") and Path(rep["report"]).exists()
    assert fake_load_split and all(c == (["dev"], False) for c in fake_load_split)  # 從來沒有讀測試集


def test_missing_features_reported(world, fake_load_split, tmp_path, monkeypatch):
    wcfg, _ = world
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(wcfg, tmp_path)
    with pytest.raises(vlm_probe.VLMProbeConfigError):
        vlm_probe.fit(cfg, n_boot=10)
