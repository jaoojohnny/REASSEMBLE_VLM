"""時間軸統計：單位推測、單調性、取樣率與缺口。

規格提醒：不要直接用 frame_idx / 30 推定秒數。這裡只「描述」時間陣列，
單位與時間基準一律標為推測，交給使用者核對。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import numpy as np

# (單位, 換算成秒的倍率, Unix epoch 合理下限, 上限)  ≈ 西元 1998–2103 年
_EPOCH_RANGES = [
    ("s", 1.0, 9.0e8, 4.2e9),
    ("ms", 1e-3, 9.0e11, 4.2e12),
    ("us", 1e-6, 9.0e14, 4.2e15),
    ("ns", 1e-9, 9.0e17, 4.2e18),
]

_TIME_NAME_RE = re.compile(r"(time|stamp|clock|^ts$|^t$|_ts$|^ts_)", re.IGNORECASE)

SAMPLE_BLOCK = 200_000
N_MIDDLE_BLOCKS = 8


def guess_time_unit(vmin: float, vmax: float, median_dt: float | None = None) -> dict[str, Any]:
    """依數值大小推測時間單位與基準；回傳 confidence 供報告標示。"""
    for unit, scale, lo, hi in _EPOCH_RANGES:
        if lo <= vmin <= hi and lo <= vmax <= hi:
            return {"unit": unit, "to_seconds": scale, "base": "unix_epoch", "confidence": "high"}
    if median_dt is not None and median_dt > 0 and abs(vmax) < 1e7 and 1e-5 <= median_dt <= 60:
        return {"unit": "s?", "to_seconds": 1.0, "base": "relative?", "confidence": "low"}
    return {"unit": "unknown", "to_seconds": None, "base": "unknown", "confidence": "none"}


def _iso_utc(seconds: float) -> str | None:
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="milliseconds")
    except (OverflowError, OSError, ValueError):
        return None


def summarize_time_blocks(blocks: list[np.ndarray], total_n: int, sampled: bool) -> dict[str, Any]:
    """blocks 為依序取出的一段或多段時間值；差分只在段內計算。"""
    vals = [np.asarray(b, dtype=np.float64).ravel() for b in blocks if np.size(b)]
    if not vals:
        return {"n": int(total_n), "empty": True, "sampled": sampled}
    allv = np.concatenate(vals)
    finite_mask = np.isfinite(allv)
    n_nonfinite = int((~finite_mask).sum())
    if not finite_mask.any():
        return {"n": int(total_n), "sampled": sampled, "n_nonfinite": n_nonfinite, "all_nonfinite": True}
    diffs_list = []
    for v in vals:
        fv = v[np.isfinite(v)]
        if fv.size >= 2:
            diffs_list.append(np.diff(fv))
    diffs = np.concatenate(diffs_list) if diffs_list else np.array([], dtype=np.float64)
    vmin = float(np.min(allv[finite_mask]))
    vmax = float(np.max(allv[finite_mask]))
    first_vals = vals[0][np.isfinite(vals[0])]
    last_vals = vals[-1][np.isfinite(vals[-1])]
    first = float(first_vals[0]) if first_vals.size else vmin
    last = float(last_vals[-1]) if last_vals.size else vmax
    n_back = int((diffs < 0).sum())
    n_rep = int((diffs == 0).sum())
    pos = diffs[diffs > 0]
    med = float(np.median(pos)) if pos.size else None
    unit = guess_time_unit(vmin, vmax, med)
    out: dict[str, Any] = {
        "n": int(total_n),
        "sampled": bool(sampled),
        "first": first,
        "last": last,
        "min": vmin,
        "max": vmax,
        "monotonic_non_decreasing": n_back == 0,
        "n_backward_steps": n_back,
        "n_repeated": n_rep,
        "n_nonfinite": n_nonfinite,
        "median_dt": med,
        "unit_guess": unit,
    }
    scale = unit.get("to_seconds")
    if scale and med:
        out["median_dt_s"] = med * scale
        out["rate_hz"] = 1.0 / (med * scale)
        out["span_s"] = (last - first) * scale
        out["n_gaps_gt_3x_median"] = int((diffs > 3 * med).sum())
        out["max_dt_s"] = float(diffs.max() * scale) if diffs.size else None
    if unit.get("base") == "unix_epoch" and scale:
        out["first_utc"] = _iso_utc(first * scale)
        out["last_utc"] = _iso_utc(last * scale)
    return out


def is_time_like_path(path: str) -> bool:
    """路徑中任一段名稱像時間（timestamps、time、stamp、ts…）。"""
    return any(_TIME_NAME_RE.search(part) for part in path.strip("/").split("/") if part)


def _looks_sec_nsec(head: np.ndarray) -> bool:
    if head.ndim != 2 or head.shape[1] != 2 or head.shape[0] == 0:
        return False
    col0, col1 = head[:, 0].astype(np.float64), head[:, 1].astype(np.float64)
    return bool(np.all((col0 >= 9.0e8) & (col0 <= 4.2e9)) and np.all((col1 >= 0) & (col1 < 1e9)))


def read_time_blocks(ds: Any, max_read_bytes: int) -> tuple[list[np.ndarray], bool, str | None] | None:
    """從 h5py Dataset 讀出時間值；超過上限就只取頭、尾與中間數段。

    回傳 (blocks, sampled, note)；不像時間陣列時回傳 None。
    """
    shape = ds.shape
    if shape is None or len(shape) not in (1, 2) or ds.dtype.kind not in "iuf":
        return None
    n = int(shape[0])
    if n == 0:
        return [], False, None
    note = None
    combine = None
    if len(shape) == 2:
        head = np.asarray(ds[0 : min(n, 16)])
        if _looks_sec_nsec(head):
            combine = "sec_nsec"
            note = "二維 (秒, 奈秒) 組合，已合併成秒"
        elif shape[1] == 1:
            combine = "squeeze"
        else:
            return None
    row_bytes = ds.dtype.itemsize * (int(shape[1]) if len(shape) == 2 else 1)

    def _convert(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if combine == "sec_nsec":
            return a[:, 0].astype(np.float64) + a[:, 1].astype(np.float64) * 1e-9
        if combine == "squeeze":
            return a[:, 0]
        return a

    blk = min(SAMPLE_BLOCK, max(1000, n // 20))
    if n * row_bytes <= max_read_bytes or n <= 2 * blk:
        return [_convert(ds[0:n])], False, note
    starts = {0, n - blk}
    for i in range(1, N_MIDDLE_BLOCKS + 1):
        starts.add(int((n - blk) * i / (N_MIDDLE_BLOCKS + 1)))
    blocks = [_convert(ds[s : min(n, s + blk)]) for s in sorted(starts)]
    msg = f"陣列過大（{n} 筆），只取 {len(blocks)} 段各 {blk} 筆做統計"
    return blocks, True, f"{note}；{msg}" if note else msg


def looks_like_epoch_array(ds: Any) -> bool:
    """名稱看不出來時，用頭尾兩個值判斷一維數值陣列是否像 Unix 時間（只讀 2 個元素）。"""
    shape = ds.shape
    if shape is None or len(shape) != 1 or int(shape[0]) < 2 or ds.dtype.kind not in "iuf":
        return False
    try:
        first = float(np.asarray(ds[0:1])[0])
        last = float(np.asarray(ds[int(shape[0]) - 1 : int(shape[0])])[0])
    except Exception:  # noqa: BLE001
        return False
    if not (np.isfinite(first) and np.isfinite(last)) or last < first:
        return False
    return guess_time_unit(first, last)["base"] == "unix_epoch"
