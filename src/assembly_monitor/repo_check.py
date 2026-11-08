"""上傳 GitHub 前的檢查：準備 commit 的檔案裡不能有 API key、原始資料、模型權重、實驗結果、簡報／PDF，也不能有超大檔。

push_github.bat 在 `git add -A` 之後呼叫：有問題就停，不 commit、不上傳。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

FORBIDDEN_PREFIX = ("data/raw/", "data/derived/", "data/slim/", "models/", "experiments/", "logs/")
ALLOWED = {"data/raw/.gitkeep", "data/derived/.gitkeep", "experiments/.gitkeep", "logs/.gitkeep"}
FORBIDDEN_SUFFIX = (".h5", ".hdf5", ".pptx", ".pdf", ".zip", ".part", ".safetensors", ".bin", ".pt", ".lnk")
MAX_BYTES = 50 * 2**20          # GitHub 50 MB 以上會警告、100 MB 以上拒收
SCAN_MAX_BYTES = 2 * 2**20      # 只掃 2 MB 以下的文字檔
SECRET_RE = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]{10,}|sk-(?:proj-)?[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}")
ENV_ASSIGN_RE = re.compile(r"^[ \t]*(ANTHROPIC_API_KEY|OPENAI_API_KEY|HF_TOKEN|GITHUB_TOKEN)[ \t]*=[ \t]*[^\s#]+", re.M)


def staged_paths(root: str | Path = ".") -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--cached"], cwd=root, capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def check(paths: list[str], root: str | Path = ".") -> list[str]:
    """回傳問題清單（空＝可以上傳）。"""
    root = Path(root)
    problems = []
    for rel in paths:
        name = rel.rsplit("/", 1)[-1]
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            problems.append(f"{rel}：API key 檔不能上傳")
            continue
        if rel not in ALLOWED and rel.startswith(FORBIDDEN_PREFIX):
            problems.append(f"{rel}：原始資料／模型權重／實驗結果／紀錄不上傳")
            continue
        if rel.lower().endswith(FORBIDDEN_SUFFIX):
            problems.append(f"{rel}：這種檔案（資料、權重、簡報、PDF）不上傳")
            continue
        p = root / rel
        if not p.is_file():
            continue
        size = p.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"{rel}：{size / 2**20:.0f} MB，超過 50 MB")
            continue
        if size <= SCAN_MAX_BYTES:
            data = p.read_bytes()
            if b"\0" in data[:4096]:
                continue
            text = data.decode("utf-8", errors="ignore")
            if SECRET_RE.search(text) or ENV_ASSIGN_RE.search(text):
                problems.append(f"{rel}：看起來有 API key 或 token")
    return problems


COMMIT_MSG = """REASSEMBLE × VLM 組裝視覺監督：原始碼、設定、測試與 Docker 環境

- src/assembly_monitor：HDF5 檢查、索引、影格擷取、標註、切分、VLM／API 評估、第 13 包 REBOOT 外部資料流程
- configs、tests、scripts、Windows 一鍵檔（*.bat）、docs（說明文件）
- Dockerfile、docker-compose.yml（用法見 docs/docker.md）
- 不含：原始資料、衍生影格、模型權重、實驗結果、.env（API key）、進度報告簡報與 PDF

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AobrD5XmtnXCKLBPRYg7zW
"""


def write_commit_msg(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(COMMIT_MSG, encoding="utf-8")
