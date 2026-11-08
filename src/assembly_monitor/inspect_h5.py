"""HDF5 結構探索（第 1 包 P1-1 / P1-2）。

逐一列出 group / dataset 的 shape、dtype、壓縮、儲存大小與少量樣本值；
大型陣列只讀頭尾或角落，內嵌影片用檔案位移探測，不整段載入。
不預設任何固定 key：所有欄位意義都由 candidates.py 依結構與名稱「推測」，
並在報告中標成待確認。

註：模組名稱刻意不用 inspect.py，以免與 Python 標準庫 inspect 撞名。
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .media import AUDIO_TYPES, VIDEO_TYPES, make_blob_ref, probe_blob, sniff_magic
from .timeaxis import is_time_like_path, looks_like_epoch_array, read_time_blocks, summarize_time_blocks
from .util import decode_text, human_bytes, jsonable, natural_key, now_iso

log = logging.getLogger(__name__)

MAX_DEPTH_HARD = 32


@dataclass
class InspectOptions:
    preview_n: int = 5
    max_read_mb: float = 64.0
    media_probe: bool = True
    count_frames: bool = True
    collapse_threshold: int = 8
    expected_cameras: list[str] = field(default_factory=list)
    max_depth: int | None = None

    @property
    def max_read_bytes(self) -> int:
        return int(self.max_read_mb * 1024 * 1024)


@dataclass
class Node:
    path: str
    name: str
    kind: str  # group | dataset | softlink | extlink | datatype | unresolved
    depth: int
    attrs: dict[str, Any] = field(default_factory=dict)
    n_members: int | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    dtype_kind: str | None = None
    size: int | None = None
    nbytes_logical: int | None = None
    storage_bytes: int | None = None
    layout: str | None = None
    chunks: list[int] | None = None
    compression: str | None = None
    compression_opts: Any = None
    preview: Any = None
    preview_note: str | None = None
    media: dict[str, Any] | None = None
    time_stats: dict[str, Any] | None = None
    link_target: str | None = None
    error: str | None = None


# ---------------------------------------------------------------- 小工具


def _join(parent: str, key: str) -> str:
    return "/" + key if parent in ("", "/") else f"{parent}/{key}"


def _read_attrs(obj: Any, preview_n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        keys = list(obj.attrs.keys())
    except Exception as e:  # noqa: BLE001
        return {"<error>": str(e)}
    for k in sorted(keys, key=natural_key):
        try:
            v = obj.attrs[k]
            if isinstance(v, np.ndarray) and v.size > preview_n * 4:
                out[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "head": jsonable(v.ravel()[:preview_n])}
            else:
                out[k] = jsonable(v)
        except Exception as e:  # noqa: BLE001
            out[k] = f"<無法讀取: {e}>"
    return out


def describe_dtype(dt: np.dtype) -> str:
    import h5py

    try:
        sinfo = h5py.check_string_dtype(dt)
    except Exception:  # noqa: BLE001
        sinfo = None
    if sinfo is not None:
        if sinfo.length is None:
            return f"str(vlen,{sinfo.encoding})"
        return f"bytes[{sinfo.length}]({sinfo.encoding})"
    try:
        vlen = h5py.check_vlen_dtype(dt)
    except Exception:  # noqa: BLE001
        vlen = None
    if vlen is not None:
        name = getattr(vlen, "__name__", None) or str(np.dtype(vlen))
        return f"vlen<{name}>"
    if dt.names:
        fields = ", ".join(f"{n}:{dt.fields[n][0].str}" for n in dt.names[:8])
        more = "…" if len(dt.names) > 8 else ""
        return f"compound{{{fields}{more}}}"
    if dt.kind == "V":
        return f"opaque[{human_bytes(dt.itemsize)}]"
    try:
        enum = h5py.check_enum_dtype(dt)
    except Exception:  # noqa: BLE001
        enum = None
    if enum:
        return f"enum{dict(list(enum.items())[:6])}"
    return dt.name


def _layout_name(ds: Any) -> str | None:
    try:
        import h5py

        code = ds.id.get_create_plist().get_layout()
        return {
            h5py.h5d.COMPACT: "compact",
            h5py.h5d.CONTIGUOUS: "contiguous",
            h5py.h5d.CHUNKED: "chunked",
            getattr(h5py.h5d, "VIRTUAL", -99): "virtual",
        }.get(code, str(code))
    except Exception:  # noqa: BLE001
        return None


def _storage_size(ds: Any) -> int | None:
    try:
        return int(ds.id.get_storage_size())
    except Exception:  # noqa: BLE001
        return None


def _convert_value(v: Any) -> Any:
    """單一元素轉成好讀的預覽值（bytes 嘗試解碼、陣列只留摘要）。"""
    if isinstance(v, np.ndarray) and v.dtype.kind != "V" and v.size > 8:
        return {"array_len": int(v.size), "dtype": str(v.dtype), "head": jsonable(v.ravel()[:4])}
    return jsonable(v)


def preview_dataset(ds: Any, opts: InspectOptions, is_blob: bool) -> tuple[Any, str | None]:
    """只讀取少量資料做預覽，回傳 (preview, note)。"""
    shape = ds.shape
    dt = ds.dtype
    k = max(1, int(opts.preview_n))
    if shape is None:
        return None, "空資料集（null dataspace）"
    size = int(np.prod(shape)) if len(shape) else 1
    if size == 0:
        return [], "空資料集（0 筆）"
    if is_blob:
        return None, "內嵌位元組（見 media）"
    if len(shape) == 0:
        if dt.kind == "V" and not dt.names and dt.itemsize > opts.max_read_bytes:
            return None, "不透明純量過大，未讀取"
        val = ds[()]
        return _convert_value(val), None
    n0 = int(shape[0])
    if len(shape) == 1:
        head = ds[0 : min(k, n0)]
        prev: dict[str, Any] = {"head": [_convert_value(v) for v in np.asarray(head, dtype=object if dt.kind == "O" else None)]}
        if n0 > k:
            tail_start = max(k, n0 - k)
            tail = ds[tail_start:n0]
            prev["tail"] = [_convert_value(v) for v in np.asarray(tail, dtype=object if dt.kind == "O" else None)]
        return prev, None
    sl = (slice(0, min(k, n0)),) + tuple(slice(0, min(3, int(s))) for s in shape[1:])
    corner = np.asarray(ds[sl])
    note = None
    if dt.kind == "u" and dt.itemsize == 1 and len(shape) >= 3:
        note = "像影像陣列（uint8，維度 ≥ 3）；只預覽左上角"
    return {"corner_index": [f"0:{s.stop}" for s in sl], "corner": jsonable(corner)}, note


# ---------------------------------------------------------------- 資料集描述


def describe_dataset(ds: Any, path: str, depth: int, opts: InspectOptions, file_path: str) -> Node:
    name = path.rsplit("/", 1)[-1]
    node = Node(path=path, name=name, kind="dataset", depth=depth)
    node.attrs = _read_attrs(ds, opts.preview_n)
    shape = ds.shape
    dt = ds.dtype
    node.shape = [int(s) for s in shape] if shape is not None else None
    node.dtype = describe_dtype(dt)
    node.dtype_kind = dt.kind
    node.size = int(np.prod(shape)) if shape is not None and len(shape) else (1 if shape == () else 0)
    if dt.kind != "O" and node.size is not None:
        node.nbytes_logical = int(node.size * dt.itemsize)
    node.storage_bytes = _storage_size(ds)
    node.layout = _layout_name(ds)
    try:
        node.chunks = [int(c) for c in ds.chunks] if ds.chunks else None
        node.compression = ds.compression
        node.compression_opts = jsonable(ds.compression_opts)
    except Exception as e:  # noqa: BLE001
        node.error = f"讀取儲存設定失敗: {e}"

    blob = None
    try:
        blob = make_blob_ref(ds, file_path, opts.max_read_bytes)
    except Exception as e:  # noqa: BLE001
        log.debug("blob 判斷失敗 %s", path, exc_info=True)
        node.error = f"blob 判斷失敗: {e}"

    magic = None
    if blob is not None:
        try:
            head = blob.head(64)
            magic = sniff_magic(head)
            if magic is None and blob.access != "unavailable":
                node.media = {"type": None, "bytes": blob.length, "access": blob.access, "head_hex": head[:16].hex()}
            elif blob.access == "unavailable":
                node.media = {"type": None, "bytes": blob.length, "access": blob.access, "note": blob.note}
        except Exception as e:  # noqa: BLE001
            node.media = {"type": None, "bytes": blob.length, "access": blob.access, "error": str(e)}
    if magic is not None and blob is not None:
        node.media = {"type": magic, "bytes": blob.length, "access": blob.access}
        if blob.note:
            node.media["note"] = blob.note
        if opts.media_probe and (magic in VIDEO_TYPES or magic in AUDIO_TYPES):
            log.info("    探測媒體 %s（%s, %s）…", path, magic, human_bytes(blob.length))
            node.media["probe"] = probe_blob(blob, magic, count_frames=opts.count_frames)

    is_blob = magic is not None or (blob is not None and dt.kind == "V")
    try:
        node.preview, node.preview_note = preview_dataset(ds, opts, is_blob=is_blob)
    except Exception as e:  # noqa: BLE001
        node.preview_note = f"預覽失敗: {type(e).__name__}: {e}"

    # 純量 bytes / vlen 字串若其實是二進位媒體，也標出來
    if node.media is None and shape == () and dt.kind in "OS":
        try:
            raw = ds[()]
            if isinstance(raw, (bytes, np.bytes_)) and decode_text(bytes(raw)) is None:
                m = sniff_magic(bytes(raw[:64]))
                node.media = {"type": m, "bytes": len(raw), "access": "in-memory"}
        except Exception:  # noqa: BLE001
            pass

    by_name = is_time_like_path(path)
    by_value = False
    if magic is None and not by_name:
        by_value = looks_like_epoch_array(ds)
    if magic is None and (by_name or by_value):
        try:
            res = read_time_blocks(ds, opts.max_read_bytes)
            if res is not None:
                blocks, sampled, note = res
                stats = summarize_time_blocks(blocks, int(shape[0]) if shape else 0, sampled)
                stats["detected_by"] = "name" if by_name else "value"
                if note:
                    stats["note"] = note
                node.time_stats = stats
        except Exception as e:  # noqa: BLE001
            node.time_stats = {"error": f"{type(e).__name__}: {e}"}
    return node


# ---------------------------------------------------------------- 走訪


def walk_file(f: Any, opts: InspectOptions, file_path: str) -> list[Node]:
    import h5py

    nodes: list[Node] = [
        Node(path="/", name="/", kind="group", depth=0, attrs=_read_attrs(f, opts.preview_n), n_members=len(f))
    ]
    limit = MAX_DEPTH_HARD if opts.max_depth is None else min(opts.max_depth, MAX_DEPTH_HARD)

    def rec(grp: Any, gpath: str, depth: int) -> None:
        try:
            keys = sorted(grp.keys(), key=natural_key)
        except Exception as e:  # noqa: BLE001
            nodes.append(Node(path=gpath, name=gpath.rsplit("/", 1)[-1], kind="unresolved", depth=depth, error=str(e)))
            return
        for key in keys:
            p = _join(gpath, key)
            try:
                link = grp.get(key, getlink=True)
            except Exception as e:  # noqa: BLE001
                nodes.append(Node(path=p, name=key, kind="unresolved", depth=depth + 1, error=str(e)))
                continue
            if isinstance(link, h5py.ExternalLink):
                nodes.append(
                    Node(path=p, name=key, kind="extlink", depth=depth + 1, link_target=f"{link.filename}:{link.path}")
                )
                continue
            soft_target = link.path if isinstance(link, h5py.SoftLink) else None
            try:
                obj = grp[key]
            except Exception as e:  # noqa: BLE001
                nodes.append(
                    Node(path=p, name=key, kind="unresolved", depth=depth + 1, link_target=soft_target, error=str(e))
                )
                continue
            if isinstance(obj, h5py.Group):
                node = Node(
                    path=p,
                    name=key,
                    kind="softlink" if soft_target else "group",
                    depth=depth + 1,
                    attrs=_read_attrs(obj, opts.preview_n),
                    n_members=len(obj),
                    link_target=soft_target,
                )
                nodes.append(node)
                if soft_target is None and depth + 1 < limit:
                    rec(obj, p, depth + 1)
            elif isinstance(obj, h5py.Dataset):
                node = describe_dataset(obj, p, depth + 1, opts, file_path)
                if soft_target:
                    node.link_target = soft_target
                nodes.append(node)
            else:
                nodes.append(Node(path=p, name=key, kind="datatype", depth=depth + 1))

    rec(f, "/", 0)
    return nodes


def inspect_file(path: str | Path, opts: InspectOptions | None = None) -> dict[str, Any]:
    """檢查單一 HDF5，回傳 {file, nodes, candidates}；不寫檔（寫檔見 report.py）。"""
    import h5py

    from .candidates import build_candidates

    opts = opts or InspectOptions()
    p = Path(path)
    st = p.stat()
    meta: dict[str, Any] = {
        "path": str(p),
        "name": p.name,
        "stem": p.stem,
        "size_bytes": int(st.st_size),
        "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
        "inspected_at": now_iso(),
        "tool": f"assembly_monitor.inspect_h5 {__version__}",
        "h5py": getattr(h5py, "__version__", "?"),
        "hdf5": getattr(getattr(h5py, "version", None), "hdf5_version", "?"),
        "options": {k: v for k, v in asdict(opts).items()},
    }
    with h5py.File(os.fspath(p), "r") as f:
        try:
            meta["libver"] = jsonable(getattr(f, "libver", None))
            meta["userblock_size"] = jsonable(getattr(f, "userblock_size", None))
        except Exception:  # noqa: BLE001
            pass
        nodes = walk_file(f, opts, os.fspath(p))
    node_dicts = [asdict(n) for n in nodes]
    candidates = build_candidates(node_dicts, opts)
    return {"file": meta, "nodes": node_dicts, "candidates": candidates}
