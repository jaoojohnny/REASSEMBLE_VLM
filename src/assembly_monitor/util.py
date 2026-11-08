"""共用小工具：自然排序、ID 正規化、位元組格式化、JSON 轉換與日誌設定。"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_INT_RE = re.compile(r"^[+-]?\d+$")


def natural_key(s: str) -> list:
    """讓 '2' 排在 '10' 前面的排序鍵。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def is_int_name(s: str) -> bool:
    return bool(_INT_RE.match(str(s).strip()))


def normalize_id(s: str) -> str:
    """'03' 與 '3' 視為同一個 ID；非數字則轉小寫比較。"""
    t = str(s).strip()
    return str(int(t)) if is_int_name(t) else t.lower()


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "?"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} TB"


def decode_text(b: bytes, max_len: int | None = None) -> str | None:
    """嘗試把位元組解成 UTF-8 文字；看起來像二進位資料就回傳 None。"""
    try:
        s = bytes(b).decode("utf-8")
    except UnicodeDecodeError:
        return None
    if s:
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in s)
        if printable / len(s) < 0.9:
            return None
    if max_len is not None and len(s) > max_len:
        return s[:max_len] + "…"
    return s


def jsonable(x: Any, max_str: int = 300) -> Any:
    """把 numpy / h5py 讀出的值轉成可寫進 JSON 的型別（不會展開大型資料）。"""
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, str):
        return x if len(x) <= max_str else x[:max_str] + "…"
    if isinstance(x, np.void):
        if x.dtype.names:
            return {name: jsonable(x[name], max_str) for name in x.dtype.names}
        return {"opaque_bytes": int(x.dtype.itemsize)}
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        raw = bytes(x)
        t = decode_text(raw, max_str)
        if t is not None:
            return t
        return {"binary_bytes": len(raw), "head_hex": raw[:16].hex()}
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return jsonable(float(x), max_str)
    if isinstance(x, np.str_):
        return jsonable(str(x), max_str)
    if isinstance(x, np.ndarray):
        if x.dtype.kind == "V" and not x.dtype.names:
            return {"opaque_bytes": int(x.nbytes)}
        if x.ndim == 0:
            return jsonable(x[()], max_str)
        if x.dtype.names:
            return [jsonable(r, max_str) for r in x]
        return [jsonable(v, max_str) for v in x]
    if isinstance(x, (list, tuple)):
        return [jsonable(v, max_str) for v in x]
    if isinstance(x, dict):
        return {str(k): jsonable(v, max_str) for k, v in x.items()}
    if isinstance(x, Path):
        return str(x)
    return repr(x)[:max_str]


def as_number(v: Any) -> float | None:
    """預覽值若是單一數字（非布林）就轉 float，否則 None。"""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, list) and len(v) == 1:
        return as_number(v[0])
    return None


def fmt_num(v: Any) -> str:
    """數值的精簡顯示：大數字（如 Unix 秒）保留到微秒。"""
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        if abs(v) >= 1e5:
            return f"{v:.6f}".rstrip("0").rstrip(".")
        return f"{v:.6g}"
    return str(v)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def setup_logging(log_path: str | Path | None = None, verbose: bool = False) -> logging.Logger:
    """主控台顯示精簡訊息；若給 log_path，另存含時間的完整紀錄（UTF-8）。"""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)
    if log_path:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
    return root


class KeepAwake:
    """Windows：執行期間阻止系統睡眠（離開時恢復）。其他系統不做事。"""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __enter__(self) -> "KeepAwake":
        if os.name == "nt":
            try:
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)  # type: ignore[attr-defined]
                logging.getLogger(__name__).info("已暫停系統睡眠（結束或關閉視窗後恢復）")
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).warning("無法暫停系統睡眠：%s（請手動把電源設定改成不睡眠）", e)
        return self

    def __exit__(self, *exc: Any) -> None:
        if os.name == "nt":
            try:
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
