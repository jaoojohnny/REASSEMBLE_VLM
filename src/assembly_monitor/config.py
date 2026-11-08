"""讀取 configs/*.yaml；缺少的欄位用預設值補齊。所有路徑相對於執行時的工作目錄（專案根目錄）。"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "dataset": {
        "name": "REASSEMBLE",
        "record_url": "https://researchdata.tuwien.ac.at/records/0ewrv-8cb44",
        "zip_url": "https://researchdata.tuwien.ac.at/records/0ewrv-8cb44/files/data.zip?download=1",
        "readme_url": "https://researchdata.tuwien.ac.at/records/0ewrv-8cb44/files/README.txt?download=1",
        "license": "CC-BY-4.0",
    },
    "paths": {
        "raw_dir": "data/raw",
        "index_dir": "data/index",
        "inspect_dir": "data/index/inspect",
        "logs_dir": "logs",
    },
    "samples": {
        "n": 3,
        "strategy": "spread",
        "seed": 0,
        "member_suffix": ".h5",
        "exclude_known_issues": True,
        "max_total_gb": 3.0,
        "with_readme": True,
    },
    "known_issue_files": [],
    "inspect": {
        "expected_cameras": [],
        "preview_n": 5,
        "max_read_mb": 64,
        "media_probe": True,
        "count_frames": True,
        "collapse_threshold": 8,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULTS)
    p = Path(path)
    if not p.exists():
        log.debug("找不到設定檔 %s，使用預設值", p)
        return copy.deepcopy(DEFAULTS)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"設定檔格式錯誤（最外層必須是 mapping）：{p}")
    return _deep_merge(DEFAULTS, data)
