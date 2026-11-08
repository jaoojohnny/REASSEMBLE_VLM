"""內嵌媒體（mp4 / mp3 等位元組）的辨識與中繼資料探測。

不把整段影片讀進記憶體：連續儲存的不透明資料集直接用檔案位移開一個「視窗」，
一維 uint8 資料集用 h5py 分段切片讀取。探測順序 PyAV → OpenCV，失敗會記錄原因。
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

VIDEO_TYPES = {"mp4", "mkv/webm", "avi"}
AUDIO_TYPES = {"mp3", "wav", "flac", "ogg"}
IMAGE_TYPES = {"png", "jpeg"}


def sniff_magic(head: bytes) -> str | None:
    """用檔頭判斷常見媒體格式。"""
    if not head or len(head) < 4:
        return None
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "mkv/webm"
    if head[:4] == b"RIFF" and len(head) >= 12:
        kind = head[8:12]
        if kind == b"AVI ":
            return "avi"
        if kind == b"WAVE":
            return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:3] == b"ID3":
        return "mp3"
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0 and (head[1] & 0x06) != 0:
        return "mp3"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    return None


class FileWindow(io.RawIOBase):
    """把 HDF5 檔中 [offset, offset+length) 當成獨立、可 seek 的檔案。"""

    def __init__(self, path: str, offset: int, length: int):
        super().__init__()
        self._f = open(path, "rb")
        self._off = int(offset)
        self._len = int(length)
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = pos
        elif whence == io.SEEK_CUR:
            new = self._pos + pos
        elif whence == io.SEEK_END:
            new = self._len + pos
        else:
            raise ValueError(f"不支援的 whence: {whence}")
        if new < 0:
            raise ValueError("seek 位置不可為負")
        self._pos = new
        return new

    def readinto(self, b) -> int:  # type: ignore[override]
        remaining = self._len - self._pos
        if remaining <= 0:
            return 0
        n = min(len(b), remaining)
        self._f.seek(self._off + self._pos)
        data = self._f.read(n)
        k = len(data)
        memoryview(b)[:k] = data
        self._pos += k
        return k

    def close(self) -> None:
        try:
            self._f.close()
        finally:
            super().close()


class DatasetBytesReader(io.RawIOBase):
    """把一維 uint8/int8 資料集當成檔案讀（支援 chunk 與壓縮）。"""

    def __init__(self, ds: Any):
        super().__init__()
        self._ds = ds
        self._len = int(ds.shape[0])
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = pos
        elif whence == io.SEEK_CUR:
            new = self._pos + pos
        elif whence == io.SEEK_END:
            new = self._len + pos
        else:
            raise ValueError(f"不支援的 whence: {whence}")
        if new < 0:
            raise ValueError("seek 位置不可為負")
        self._pos = new
        return new

    def readinto(self, b) -> int:  # type: ignore[override]
        remaining = self._len - self._pos
        if remaining <= 0:
            return 0
        n = min(len(b), remaining)
        arr = np.asarray(self._ds[self._pos : self._pos + n])
        data = arr.astype(np.uint8, copy=False).tobytes()
        k = len(data)
        memoryview(b)[:k] = data
        self._pos += k
        return k


@dataclass
class BlobRef:
    """一段內嵌位元組的取用方式。access: file-offset / dataset-slices / in-memory / unavailable。"""

    length: int
    access: str
    file_path: str | None = None
    offset: int | None = None
    dataset: Any = None
    data: bytes | None = None
    note: str | None = None

    def open(self) -> io.BufferedIOBase:
        if self.access == "file-offset" and self.file_path is not None and self.offset is not None:
            return io.BufferedReader(FileWindow(self.file_path, self.offset, self.length), buffer_size=1 << 20)
        if self.access == "dataset-slices" and self.dataset is not None:
            return io.BufferedReader(DatasetBytesReader(self.dataset), buffer_size=1 << 20)
        if self.access == "in-memory" and self.data is not None:
            return io.BytesIO(self.data)
        raise OSError(self.note or "無法讀取這段位元組")

    def head(self, n: int = 64) -> bytes:
        if self.access == "unavailable":
            return b""
        with self.open() as fh:
            return fh.read(n)


def _safe_offset(ds: Any) -> int | None:
    try:
        off = ds.id.get_offset()
    except Exception:  # noqa: BLE001 - h5py 會因版面配置拋各種例外
        return None
    return int(off) if off is not None else None


def _can_open(path: str) -> bool:
    """確認能另開一個唯讀檔案代碼（Windows 檔案鎖定時改走其他讀法）。"""
    try:
        with open(path, "rb") as fh:
            fh.read(1)
        return True
    except OSError:
        return False


def make_blob_ref(ds: Any, file_path: str, max_read_bytes: int) -> BlobRef | None:
    """判斷資料集是否可能是一段內嵌位元組（影片、音訊、圖片），並決定怎麼讀。"""
    shape = ds.shape
    dt = ds.dtype
    if shape is None:
        return None
    is_opaque = dt.kind == "V" and not dt.names
    is_byte_array = dt.kind in "ui" and dt.itemsize == 1
    if is_opaque and (shape == () or shape == (1,)):
        length = int(dt.itemsize)
        off = _safe_offset(ds)
        if off is not None and not getattr(ds, "compression", None) and _can_open(file_path):
            return BlobRef(length=length, access="file-offset", file_path=file_path, offset=off)
        if length <= max_read_bytes:
            val = ds[()] if shape == () else ds[0]
            return BlobRef(length=length, access="in-memory", data=np.asarray(val).tobytes())
        return BlobRef(
            length=length,
            access="unavailable",
            note=f"不透明資料 {length} bytes 不是連續儲存，超過 --max-read-mb 上限，未讀取",
        )
    # 真實 REASSEMBLE 的音訊是「每個位元組存成一個 int64」的一維陣列（開頭 FF F3 = MP3 frame sync）
    if dt.kind in "iu" and dt.itemsize > 1 and len(shape) == 1 and shape[0] >= 64:
        head = np.asarray(ds[0:64])
        if head.size and int(head.min()) >= 0 and int(head.max()) <= 255 and sniff_magic(head.astype(np.uint8).tobytes()):
            return BlobRef(length=int(shape[0]), access="dataset-slices", dataset=ds,
                           note=f"{dt.name} 陣列逐位元組存放（每個元素一個位元組）")
        return None
    if is_byte_array and len(shape) == 1 and shape[0] >= 64:
        length = int(shape[0])
        off = _safe_offset(ds)
        if (off is not None and not getattr(ds, "compression", None) and not getattr(ds, "chunks", None)
                and _can_open(file_path)):
            return BlobRef(length=length, access="file-offset", file_path=file_path, offset=off)
        return BlobRef(length=length, access="dataset-slices", dataset=ds)
    return None


# ---------------------------------------------------------------- 探測後端


def _f(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _probe_with_av(blob: BlobRef, count_frames: bool) -> dict[str, Any]:
    import av  # type: ignore[import-not-found]

    info: dict[str, Any] = {"backend": f"pyav {getattr(av, '__version__', '?')}"}
    with blob.open() as fh:
        container = av.open(fh, mode="r")
        try:
            fmt = getattr(container, "format", None)
            info["format"] = getattr(fmt, "name", None)
            dur = getattr(container, "duration", None)
            info["container_duration_s"] = (dur / 1_000_000.0) if dur else None
            streams = []
            vstream = None
            for s in container.streams:
                cc = getattr(s, "codec_context", None)
                tb = getattr(s, "time_base", None)
                d: dict[str, Any] = {
                    "index": s.index,
                    "type": s.type,
                    "codec": getattr(cc, "name", None),
                    "frames_header": int(getattr(s, "frames", 0) or 0),
                }
                if getattr(s, "duration", None) is not None and tb is not None:
                    d["duration_s"] = _f(s.duration * tb)
                if getattr(s, "start_time", None) is not None and tb is not None:
                    d["start_time_s"] = _f(s.start_time * tb)
                if s.type == "video":
                    d["width"] = getattr(cc, "width", None)
                    d["height"] = getattr(cc, "height", None)
                    d["average_fps"] = _f(getattr(s, "average_rate", None))
                    d["guessed_fps"] = _f(getattr(s, "guessed_rate", None))
                    pf = getattr(cc, "pix_fmt", None)
                    d["pix_fmt"] = str(pf) if pf is not None else None
                    if vstream is None:
                        vstream = s
                elif s.type == "audio":
                    d["sample_rate"] = getattr(cc, "sample_rate", None)
                    layout = getattr(cc, "layout", None)
                    ch = getattr(layout, "nb_channels", None) if layout is not None else None
                    if ch is None and layout is not None and isinstance(getattr(layout, "channels", None), tuple):
                        ch = len(layout.channels)
                    d["channels"] = ch if ch is not None else getattr(cc, "channels", None)
                streams.append(d)
            info["streams"] = streams
            if vstream is not None and count_frames and not int(getattr(vstream, "frames", 0) or 0):
                n = 0
                pmin = pmax = None
                for pkt in container.demux(vstream):
                    if pkt.pts is None:
                        continue
                    n += 1
                    pmin = pkt.pts if pmin is None else min(pmin, pkt.pts)
                    pmax = pkt.pts if pmax is None else max(pmax, pkt.pts)
                info["video_packets_counted"] = n
                if pmin is not None and vstream.time_base is not None:
                    info["video_pts_span_s"] = _f((pmax - pmin) * vstream.time_base)
        finally:
            container.close()
    return info


def _fourcc(code: float) -> str | None:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if c <= 0:
        return None
    s = "".join(chr((c >> (8 * i)) & 0xFF) for i in range(4))
    return s if s.isprintable() else None


def _probe_with_cv2(blob: BlobRef, count_frames: bool) -> dict[str, Any]:
    import cv2  # type: ignore[import-not-found]

    tmp = tempfile.NamedTemporaryFile(prefix="am_probe_", suffix=".mp4", delete=False)
    tmp_path = tmp.name
    try:
        with blob.open() as src:
            shutil.copyfileobj(src, tmp, 4 << 20)
        tmp.close()
        cap = cv2.VideoCapture(tmp_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("OpenCV 無法開啟這段影片")
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            d: dict[str, Any] = {
                "index": 0,
                "type": "video",
                "codec": _fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None,
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None,
                "average_fps": _f(cap.get(cv2.CAP_PROP_FPS)) or None,
                "frames_header": frames,
            }
            info: dict[str, Any] = {"backend": f"opencv {cv2.__version__}", "streams": [d]}
            if count_frames and frames <= 0:
                n = 0
                while cap.grab():
                    n += 1
                info["video_packets_counted"] = n
            return info
        finally:
            cap.release()
    finally:
        try:
            tmp.close()
        except OSError:
            pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def summarize_probe(info: dict[str, Any]) -> dict[str, Any]:
    """從探測結果取出報告常用欄位：frames / fps / 解析度 / 長度 / 取樣率。"""
    out: dict[str, Any] = {"backend": info.get("backend")}
    streams = info.get("streams") or []
    v = next((s for s in streams if s.get("type") == "video"), None)
    a = next((s for s in streams if s.get("type") == "audio"), None)
    if v:
        frames = v.get("frames_header") or info.get("video_packets_counted")
        out.update(
            codec=v.get("codec"),
            width=v.get("width"),
            height=v.get("height"),
            fps_nominal=v.get("average_fps") or v.get("guessed_fps"),
            frames=int(frames) if frames else None,
            frames_source="header" if v.get("frames_header") else ("counted" if frames else None),
            duration_s=v.get("duration_s") or info.get("container_duration_s"),
        )
    if a:
        out.update(
            audio_codec=a.get("codec"),
            sample_rate=a.get("sample_rate"),
            channels=a.get("channels"),
            audio_duration_s=a.get("duration_s") or info.get("container_duration_s"),
        )
    return out


def probe_blob(blob: BlobRef, media_type: str, count_frames: bool = True) -> dict[str, Any]:
    """依序嘗試 PyAV、OpenCV（後者只看影片）。所有錯誤寫進 errors，不中斷檢查。"""
    errors: dict[str, str] = {}
    backends = ["av", "cv2"] if media_type in VIDEO_TYPES else ["av"]
    for name in backends:
        try:
            raw = _probe_with_av(blob, count_frames) if name == "av" else _probe_with_cv2(blob, count_frames)
            result = {"ok": True, "summary": summarize_probe(raw), "raw": raw}
            if errors:
                result["errors"] = errors
            return result
        except ImportError as e:
            errors[name] = f"未安裝：{e}"
        except Exception as e:  # noqa: BLE001 - 探測失敗只記錄，不中斷
            errors[name] = f"{type(e).__name__}: {e}"
            log.debug("probe %s 失敗", name, exc_info=True)
    return {"ok": False, "errors": errors}
