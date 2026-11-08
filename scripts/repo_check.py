"""上傳 GitHub 前的檢查（push_github.bat 用）。用法：
  python -m scripts.repo_check                         # 檢查 git 已加入（staged）的檔案
  python -m scripts.repo_check --write-msg logs/commit_msg.txt
有問題回傳 1（不要 commit）；沒問題回傳 0。
"""

from __future__ import annotations

import argparse
import sys

from assembly_monitor import repo_check as rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="repo_check")
    ap.add_argument("--write-msg", default=None)
    args = ap.parse_args(argv)
    paths = rc.staged_paths()
    problems = rc.check(paths)
    print(f"準備上傳 {len(paths)} 個檔案。")
    if problems:
        print("[STOP] 這些檔案不能上傳：")
        for p in problems[:50]:
            print("  - " + p)
        return 1
    print("檢查通過：沒有 API key、原始資料、模型權重、實驗結果、簡報／PDF 或超過 50 MB 的檔案。")
    if args.write_msg:
        rc.write_commit_msg(args.write_msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
