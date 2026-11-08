"""上傳 GitHub／Docker 打包：檢查規則與檔案格式（不連網、不呼叫 git）。"""

from pathlib import Path

from assembly_monitor import repo_check as rc

ROOT = Path(__file__).resolve().parent.parent


def test_check_rules(tmp_path):
    files = {
        "src/a.py": "print('hi')\n",
        ".env": "OPENAI_API_KEY=x\n",
        ".env.example": "OPENAI_API_KEY=\nANTHROPIC_API_KEY=\n",
        "data/raw/x.h5": "x",
        "data/raw/.gitkeep": "",
        "experiments/r/report.md": "x",
        "docs/report.pdf": "x",
        "docs/package13_reboot.md": "說明\n",
        "data/splits/s/split.csv": "segment_id\n",
        "configs/leak.yaml": "key: " + "sk-" + "a" * 30 + "\n",       # 測試用的假 key（執行時才組出來）
        "notes/env_copy.txt": "ANTHROPIC_API_KEY=" + "abc" * 5 + "\n",
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    probs = rc.check(list(files), tmp_path)
    bad = {x.split("：")[0] for x in probs}
    assert bad == {".env", "data/raw/x.h5", "experiments/r/report.md", "docs/report.pdf", "configs/leak.yaml",
                   "notes/env_copy.txt"}, bad


def test_big_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "MAX_BYTES", 10)
    (tmp_path / "big.csv").write_text("x" * 50, encoding="utf-8")
    assert rc.check(["big.csv"], tmp_path) and "超過" in rc.check(["big.csv"], tmp_path)[0]


def test_commit_msg_has_attribution(tmp_path):
    rc.write_commit_msg(tmp_path / "m.txt")
    t = (tmp_path / "m.txt").read_text(encoding="utf-8")
    assert "Co-Authored-By:" in t and "Claude-Session:" in t


def test_ignore_files():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for need in (".env", "models/", "experiments/*", "logs/*", "data/raw/*", "data/derived/*", "docs/*.pptx", "docs/*.pdf"):
        assert need in gi, need
    di = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for need in (".env", ".git", "data/raw", "models", "experiments", "logs", "docs/*.pptx", "docs/*.pdf"):
        assert need in di, need


def test_docker_files():
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for need in ("FROM python:3.11", "requirements.txt", "requirements-vlm.txt", "torch_constraint", "pip install --no-deps -e ."):
        assert need in df, need
    for line in df.splitlines():
        if line.startswith("COPY ") and "requirements" in line:
            for f in line.split()[1:-1]:
                assert (ROOT / f).exists(), f
    dc = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "./data:/workspace/data" in dc and "required: false" in dc


def test_bats_ascii_crlf():
    for name in ("push_github.bat", "docker_build_test.bat"):
        b = (ROOT / name).read_bytes()
        assert all(x < 128 for x in b), name
        assert b.count(b"\n") == b.count(b"\r\n") > 10, name
    pb = (ROOT / "push_github.bat").read_bytes()
    assert b"scripts.repo_check" in pb and b"--private" in pb and b"REASSEMBLE_VLM" in pb
