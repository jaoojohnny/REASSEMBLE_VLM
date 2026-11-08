"""命令列：批次檢查、壞檔不中止、總表；fetch 本機 zip。"""

from __future__ import annotations

import json
import zipfile

from assembly_monitor.cli import expand_paths, fetch_main, inspect_main


def test_inspect_cli_batch_and_bad_file(make_synth, tmp_path):
    make_synth("normal")
    make_synth("dup_id")
    (tmp_path / "broken.h5").write_bytes(b"not an hdf5 file at all")
    out = tmp_path / "inspect"
    rc = inspect_main([str(tmp_path), "--out", str(out), "--config", str(tmp_path / "nope.yaml")])
    assert rc == 1  # 有一個壞檔 → 1，但其他檔仍完成
    assert (out / "normal" / "tree.txt").exists()
    assert (out / "dup_id" / "questions.md").exists()
    summary = json.loads((out / "_summary.json").read_text(encoding="utf-8"))
    assert len(summary["files"]) == 2
    assert summary["failures"][0]["file"].endswith("broken.h5")
    assert "SEG_DUPLICATE_ID" in {a["code"] for a in summary["anomaly_counts"]}
    assert (out / "_summary.md").read_text(encoding="utf-8").startswith("# HDF5 檢查總表")


def test_inspect_cli_no_files(tmp_path):
    assert inspect_main([str(tmp_path / "missing_dir" / "*.h5"), "--out", str(tmp_path / "o")]) == 2


def test_expand_paths_dedup(make_synth, tmp_path):
    info = make_synth("normal")
    got = expand_paths([info["path"], str(tmp_path), str(tmp_path / "*.h5")])
    assert len(got) == 1


def test_fetch_cli_local_zip(tmp_path):
    zp = tmp_path / "data.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for d in range(1, 5):
            zf.writestr(f"data/2025-01-0{d}-00-00-00.h5", bytes([d]) * (1000 * d))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"paths:\n  raw_dir: {(tmp_path / 'raw').as_posix()}\n  index_dir: {(tmp_path / 'idx').as_posix()}\n"
        "samples:\n  n: 2\n  strategy: spread\n",
        encoding="utf-8",
    )
    rc = fetch_main(["--config", str(cfg), "--zip-path", str(zp), "--yes"])
    assert rc == 0
    got = sorted(p.name for p in (tmp_path / "raw").glob("*.h5"))
    assert got == ["2025-01-01-00-00-00.h5", "2025-01-04-00-00-00.h5"]
    assert fetch_main(["--config", str(cfg), "--zip-path", str(zp), "--list"]) == 0
