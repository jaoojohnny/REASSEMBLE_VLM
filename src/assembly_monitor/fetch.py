"""最小樣本取得：只從 REASSEMBLE 的 data.zip 取出 1–3 個 .h5，不整包下載。

官方資料是 TU Wien Research Data 上的單一 data.zip（約 54.8 GiB）。這個工具用 HTTP Range
（分段下載）只讀 zip 檔尾的目錄，再只下載選中的成員；也能對本機已下載的 zip 使用。
伺服器若不支援 Range，會立刻停止並提示改用瀏覽器下載，不會誤下載整包。
每個檔案都會驗證 zip 內建的 CRC-32，並在 manifest.json 記錄來源、大小與 SHA-256。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import random
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import __version__
from .util import human_bytes, natural_key, now_iso

log = logging.getLogger(__name__)

USER_AGENT = f"assembly-monitor-fetch/{__version__} (python-requests; research sample download)"


class RangeNotSupported(RuntimeError):
    """伺服器不支援分段下載（或回應格式不符），不能只取部分檔案。"""


class FetchCancelled(RuntimeError):
    pass


class HttpRangeReader(io.RawIOBase):
    """把遠端檔案包成可 seek 的唯讀檔；每次讀取只發出需要的 Range 請求。

    連續讀取時預讀量會逐步加倍（64 KB → 16 MB），隨機跳讀（讀 zip 目錄）時回到最小值。
    """

    MIN_READAHEAD = 64 * 1024
    MAX_READAHEAD = 16 * 1024 * 1024

    def __init__(self, url: str, session: Any = None, timeout: tuple[float, float] = (20, 120), retries: int = 4):
        super().__init__()
        import requests

        self.url = url
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.session.headers["Accept-Encoding"] = "identity"
        self.timeout = timeout
        self.retries = retries
        self.requests_made = 0
        self.bytes_fetched = 0
        self._pos = 0
        self._buf = b""
        self._buf_start = 0
        self._readahead = self.MIN_READAHEAD
        self.size = self._probe()

    def _probe(self) -> int:
        r = self.session.get(self.url, headers={"Range": "bytes=0-0"}, stream=True, timeout=self.timeout)
        try:
            self.requests_made += 1
            if r.status_code == 206:
                cr = r.headers.get("Content-Range", "")
                total = cr.rsplit("/", 1)[-1].strip()
                if total.isdigit():
                    return int(total)
                raise RangeNotSupported(f"伺服器的 Content-Range 沒有總長度：{cr!r}")
            if r.status_code == 200:
                raise RangeNotSupported("伺服器回傳 200（不支援 Range 分段下載）")
            if r.status_code in (401, 403):
                raise PermissionError(f"HTTP {r.status_code}：伺服器拒絕存取（可能需要先在瀏覽器同意條款或登入）")
            raise OSError(f"HTTP {r.status_code}：{self.url}")
        finally:
            r.close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self.size + offset
        else:
            raise ValueError(f"不支援的 whence: {whence}")
        if new < 0:
            raise ValueError("seek 位置不可為負")
        self._pos = new
        return new

    def readinto(self, b) -> int:  # type: ignore[override]
        if self._pos >= self.size:
            return 0
        buf_end = self._buf_start + len(self._buf)
        if not (self._buf_start <= self._pos < buf_end):
            self._fill(len(b))
        off = self._pos - self._buf_start
        n = min(len(b), len(self._buf) - off)
        memoryview(b)[:n] = self._buf[off : off + n]
        self._pos += n
        return n

    def _fill(self, want: int) -> None:
        sequential = len(self._buf) > 0 and self._pos == self._buf_start + len(self._buf)
        self._readahead = min(self._readahead * 2, self.MAX_READAHEAD) if sequential else self.MIN_READAHEAD
        start = self._pos
        end = min(self.size, start + max(want, self._readahead)) - 1
        self._buf = self._get(start, end)
        self._buf_start = start

    def _get(self, start: int, end: int) -> bytes:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=self.timeout
                )
                try:
                    self.requests_made += 1
                    if r.status_code == 200:
                        raise RangeNotSupported("伺服器中途改回 200（不支援 Range）")
                    if r.status_code in (401, 403, 404):
                        raise PermissionError(f"HTTP {r.status_code}")
                    if r.status_code != 206:
                        raise OSError(f"HTTP {r.status_code}")
                    cr = r.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {start}-"):
                        raise OSError(f"Content-Range 不符：要 {start}-{end}，收到 {cr!r}")
                    data = r.content
                finally:
                    r.close()
                if len(data) != end - start + 1:
                    raise OSError(f"長度不符：要 {end - start + 1} bytes，收到 {len(data)}")
                self.bytes_fetched += len(data)
                return data
            except (RangeNotSupported, PermissionError):
                raise
            except Exception as e:  # noqa: BLE001 - 網路錯誤重試
                last = e
                wait = min(2**attempt, 15)
                log.debug("Range %s-%s 失敗（第 %d 次）：%s；%ss 後重試", start, end, attempt + 1, e, wait)
                time.sleep(wait)
        raise OSError(f"下載區段 bytes={start}-{end} 失敗：{last}")


# ---------------------------------------------------------------- zip 操作


def open_zip(zip_url: str | None = None, zip_path: str | Path | None = None, session: Any = None):
    """回傳 (ZipFile, 來源描述, 關閉函式)。"""
    if zip_path:
        p = Path(zip_path)
        zf = zipfile.ZipFile(p)
        return zf, {"type": "local", "path": str(p), "size_bytes": p.stat().st_size}, zf.close
    if not zip_url:
        raise ValueError("需要 --zip-url 或 --zip-path")
    reader = HttpRangeReader(zip_url, session=session)
    buffered = io.BufferedReader(reader, buffer_size=1 << 16)
    zf = zipfile.ZipFile(buffered)

    def _close() -> None:
        zf.close()
        buffered.close()

    src = {"type": "remote", "url": zip_url, "size_bytes": reader.size, "reader": reader}
    return zf, src, _close


def member_rows(infos: list[zipfile.ZipInfo]) -> list[dict[str, Any]]:
    methods = {zipfile.ZIP_STORED: "stored", zipfile.ZIP_DEFLATED: "deflated"}
    return [
        {
            "name": i.filename,
            "size_bytes": i.file_size,
            "compressed_bytes": i.compress_size,
            "compress_type": methods.get(i.compress_type, str(i.compress_type)),
            "crc32": f"{i.CRC:08x}",
            "date_time": "%04d-%02d-%02d %02d:%02d:%02d" % i.date_time,
        }
        for i in infos
    ]


def write_listing_csv(infos: list[zipfile.ZipInfo], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = member_rows(infos)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)


def select_members(
    infos: list[zipfile.ZipInfo],
    n: int,
    strategy: str = "spread",
    seed: int = 0,
    exclude_names: list[str] | tuple[str, ...] = (),
) -> list[zipfile.ZipInfo]:
    """spread：依檔名（錄製時間）排序取頭、中、尾；smallest：最小的 n 個；random：固定 seed 隨機。"""
    excl = {PurePosixPath(x).name.lower() for x in exclude_names}
    pool = sorted(
        (i for i in infos if PurePosixPath(i.filename).name.lower() not in excl),
        key=lambda i: natural_key(PurePosixPath(i.filename).name),
    )
    if n <= 0 or not pool:
        return []
    n = min(n, len(pool))
    if strategy == "smallest":
        chosen = sorted(pool, key=lambda i: (i.file_size, i.filename))[:n]
    elif strategy == "random":
        chosen = random.Random(seed).sample(pool, n)
    elif strategy == "spread":
        if n == 1:
            idx = [len(pool) // 2]
        else:
            idx = sorted({round(k * (len(pool) - 1) / (n - 1)) for k in range(n)})
        chosen = [pool[i] for i in idx]
    else:
        raise ValueError(f"未知的 strategy：{strategy}（可用 spread / smallest / random）")
    return sorted(chosen, key=lambda i: natural_key(PurePosixPath(i.filename).name))


def _progress(total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]

        return tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc, ascii=True)
    except ImportError:
        return None


def extract_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, out_dir: str | Path, progress: bool = True) -> dict[str, Any]:
    """串流解出單一成員到 out_dir（檔名攤平），邊寫邊算 SHA-256；CRC 錯誤會拋例外。

    progress=False 時不顯示進度條（多個下載同時進行時用）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = PurePosixPath(info.filename).name
    dest = out / name
    tmp = out / (name + ".part")
    sha = hashlib.sha256()
    bar = _progress(info.file_size, name) if progress else None
    done = 0
    next_report = 0.1
    t0 = time.time()
    try:
        with zf.open(info) as src, open(tmp, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                if bar is not None:
                    bar.update(len(chunk))
                elif progress and info.file_size and done / info.file_size >= next_report:
                    log.info("    %s：%.0f%%", name, 100 * done / info.file_size)
                    next_report += 0.1
    except BaseException:
        if bar is not None:
            bar.close()
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    if bar is not None:
        bar.close()
    if done != info.file_size:
        tmp.unlink(missing_ok=True)
        raise OSError(f"{name} 大小不符：預期 {info.file_size}，實得 {done}")
    os.replace(tmp, dest)
    return {
        "member": info.filename,
        "path": str(dest),
        "size_bytes": done,
        "crc32": f"{info.CRC:08x}",
        "sha256": sha.hexdigest(),
        "seconds": round(time.time() - t0, 1),
        "downloaded_at": now_iso(),
    }


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def disk_free_bytes(path: str | Path) -> int | None:
    """回傳 path（或最近的已存在上層資料夾）所在磁碟的剩餘空間。"""
    p = Path(path).resolve()
    for cand in [p, *p.parents]:
        if cand.exists():
            try:
                return int(shutil.disk_usage(cand).free)
            except OSError:
                return None
    return None


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("manifest.json 無法解析，將重建")
    return {"files": {}}


def download_small(url: str, dest: Path, session: Any = None, max_bytes: int = 5 << 20) -> bool:
    """下載小檔（例如官方 README.txt），超過 max_bytes 就放棄。"""
    import requests

    s = session or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    try:
        with s.get(url, stream=True, timeout=(20, 60)) as r:
            if r.status_code != 200:
                log.warning("下載 %s 失敗：HTTP %s", url, r.status_code)
                return False
            data = b""
            for chunk in r.iter_content(65536):
                data += chunk
                if len(data) > max_bytes:
                    log.warning("%s 超過 %s，放棄", url, human_bytes(max_bytes))
                    return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("下載 %s 失敗：%s", url, e)
        return False


def fetch_samples(
    *,
    zip_url: str | None,
    zip_path: str | None,
    out_dir: str | Path,
    index_dir: str | Path,
    n: int = 3,
    strategy: str = "spread",
    seed: int = 0,
    suffix: str = ".h5",
    exclude_names: list[str] | tuple[str, ...] = (),
    members: list[str] | None = None,
    max_total_bytes: int | None = None,
    assume_yes: bool = False,
    list_only: bool = False,
    readme_url: str | None = None,
    record_url: str | None = None,
    confirm: Callable[[str], bool] | None = None,
    session: Any = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    idx = Path(index_dir)
    log.info("== REASSEMBLE 最小樣本取得 ==")
    log.info("來源：%s", zip_path or zip_url)
    if not zip_path:
        log.info("（只讀 zip 檔尾目錄與選中的檔案，不會下載整包）")
    zf, src, close = open_zip(zip_url=zip_url, zip_path=zip_path, session=session)
    reader = src.pop("reader", None)
    try:
        log.info("zip 大小：%s", human_bytes(src["size_bytes"]))
        infos = [i for i in zf.infolist() if not i.is_dir()]
        listing_path = idx / "zip_listing.csv"
        write_listing_csv(infos, listing_path)
        cands = [i for i in infos if i.filename.lower().endswith(suffix.lower())]
        log.info("zip 內 %d 個檔案，其中 %d 個 %s；完整清單已存 %s", len(infos), len(cands), suffix, listing_path)
        result: dict[str, Any] = {"source": src, "n_members": len(infos), "n_candidates": len(cands),
                                  "listing": str(listing_path), "downloaded": [], "skipped": []}
        if list_only:
            for i in sorted(cands, key=lambda i: natural_key(i.filename))[:30]:
                log.info("  %-50s %10s", i.filename, human_bytes(i.file_size))
            if len(cands) > 30:
                log.info("  …另有 %d 個（見 %s）", len(cands) - 30, listing_path)
            return result
        if members:
            by_name = {i.filename: i for i in cands}
            by_base = {PurePosixPath(i.filename).name: i for i in cands}
            chosen, missing = [], []
            for m in members:
                hit = by_name.get(m) or by_base.get(PurePosixPath(m).name)
                (chosen if hit else missing).append(hit or m)
            if missing:
                raise ValueError(f"zip 裡找不到：{missing}")
        else:
            n_excl = sum(1 for i in cands if PurePosixPath(i.filename).name.lower() in
                         {PurePosixPath(x).name.lower() for x in exclude_names})
            if n_excl:
                log.info("已排除 %d 個官方註記有問題的錄製檔", n_excl)
            chosen = select_members(cands, n, strategy, seed, exclude_names)
        if not chosen:
            log.warning("沒有可下載的檔案")
            result["status"] = "nothing_selected"
            return result
        dl_total = sum(i.compress_size for i in chosen)
        disk_total = sum(i.file_size for i in chosen)
        log.info("預計取得 %d 個檔（策略 %s）：", len(chosen), "指定" if members else strategy)
        log.info("  %-3s %-34s %10s %10s", "#", "zip 內檔名", "下載量", "解壓後")
        for k, i in enumerate(chosen, 1):
            log.info("  %-3s %-34s %10s %10s", f"{k}.", i.filename, human_bytes(i.compress_size), human_bytes(i.file_size))
        log.info("合計下載 %s（解壓後佔用 %s），存到 %s", human_bytes(dl_total), human_bytes(disk_total), out)
        if max_total_bytes is not None and dl_total > max_total_bytes:
            raise ValueError(
                f"下載量 {human_bytes(dl_total)} 超過上限 {human_bytes(max_total_bytes)}；請減少 --n 或調高 --max-total-gb"
            )
        free = disk_free_bytes(out)
        if free is not None and free < disk_total * 1.05 + (1 << 30):
            raise ValueError(f"磁碟剩餘 {human_bytes(free)}，不夠放解壓後的 {human_bytes(disk_total)}（另需 1 GB 餘裕）")
        manifest_path = out / "manifest.json"
        manifest = load_manifest(manifest_path)
        todo = []
        for i in chosen:
            dest = out / PurePosixPath(i.filename).name
            if dest.exists() and dest.stat().st_size == i.file_size:
                result["skipped"].append(str(dest))
                entry = manifest["files"].get(dest.name)
                if not entry or entry.get("size_bytes") != i.file_size:
                    manifest["files"][dest.name] = {"member": i.filename, "path": str(dest), "size_bytes": i.file_size,
                                                    "crc32": f"{i.CRC:08x}", "sha256": _sha256_file(dest),
                                                    "downloaded_at": "unknown（已存在）"}
                log.info("  已存在，略過：%s", dest.name)
            else:
                todo.append(i)
        if todo and not assume_yes:
            ask = confirm or (lambda msg: input(msg).strip().lower() in ("y", "yes"))
            dl = human_bytes(sum(i.compress_size for i in todo))
            disk = human_bytes(sum(i.file_size for i in todo))
            if not ask(f"確定下載 {len(todo)} 個檔（下載 {dl}，解壓後 {disk}）嗎？輸入 y 再按 Enter [y/N] "):
                raise FetchCancelled("使用者取消下載")
        manifest.update({
            "dataset": "REASSEMBLE",
            "record_url": record_url,
            "source": {k: v for k, v in src.items()},
            "license": "CC-BY-4.0（請在報告中引用 Sliwowski et al., 2025）",
            "updated_at": now_iso(),
        })
        for i in todo:
            log.info("下載 %s …", i.filename)
            info = extract_member(zf, i, out)
            manifest["files"][Path(info["path"]).name] = info
            result["downloaded"].append(info)
            out.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
            log.info("  完成：%s（解壓後 %s，CRC 驗證通過，%ss）", Path(info["path"]).name, human_bytes(info["size_bytes"]), info["seconds"])
        out.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        result["manifest"] = str(manifest_path)
        if readme_url and not zip_path:
            dest = out / "README_TUData.txt"
            if not dest.exists() and download_small(readme_url, dest, session=getattr(reader, "session", None)):
                log.info("官方 README 已存：%s", dest)
        if reader is not None:
            result["http_requests"] = reader.requests_made
            result["http_bytes"] = reader.bytes_fetched
        result["status"] = "ok"
        return result
    finally:
        close()
