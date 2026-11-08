"""精簡版 HDF5：只留下視覺研究需要的內容（各相機 mp4、相機時間戳、segment 標註）。

路徑與原檔相同，所以同一份 field_mapping.yaml、inspect-h5、check_segments 都能直接用在精簡版上。
一個錄製檔約 30–60 MB（原檔解壓後 0.1–3.4 GB），之後的包不必再重新下載整包資料。
"""

from __future__ import annotations

import os
import posixpath
from pathlib import Path
from typing import Any

from . import __version__
from .mapping import Mapping
from .util import now_iso


def slim_paths(mapping: Mapping) -> list[str]:
    paths = []
    for spec in mapping.cameras.values():
        paths += [spec["video"], spec["timestamps"]]
    paths.append(mapping.seg_container)
    seen: set[str] = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def make_slim(src_path: str | Path, dst_path: str | Path, mapping: Mapping, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    import h5py

    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    copied, missing = [], []
    with h5py.File(os.fspath(src_path), "r") as s, h5py.File(os.fspath(tmp), "w") as d:
        for p in slim_paths(mapping):
            key = "/" + p.strip("/")
            if key not in s:
                missing.append(key)
                continue
            parent = posixpath.dirname(key)
            target = d.require_group(parent) if parent not in ("", "/") else d
            s.copy(s[key], target, name=posixpath.basename(key))
            copied.append(key)
        d.attrs["slim"] = "videos + camera timestamps + segments (assembly_monitor)"
        d.attrs["created_at"] = now_iso()
        d.attrs["tool"] = f"assembly_monitor.slim {__version__}"
        for k, v in (meta or {}).items():
            if v is not None:
                d.attrs[k] = str(v)
    os.replace(tmp, dst)
    return {"path": str(dst), "size_bytes": dst.stat().st_size, "copied": copied, "missing": missing}
