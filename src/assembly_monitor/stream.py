"""全量串流索引：逐一下載 zip 成員 → 建索引 → 存精簡版 → 刪原檔，可中斷續跑。

- 用 HTTP Range 讀官方 data.zip（不需要先把 54.8 GB 整包下載到硬碟）
- 每個錄製檔的結果寫成 data/index/recordings/<id>.json；已完成的會略過，失敗的下次重試
- data/raw 已經有的完整檔直接使用，不重新下載
- 多個下載同時進行（--workers），每個 worker 有自己的連線
- Windows 上執行期間防止電腦睡眠
- 進度寫在 data/index/stream_progress.json（Claude 可以讀這個檔確認狀態）
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .fetch import disk_free_bytes, extract_member, open_zip
from .index import index_h5, save_recording, save_recording_error, write_index_outputs
from .mapping import Mapping
from .slim import make_slim
from .util import KeepAwake, human_bytes, natural_key, now_iso  # noqa: F401 - KeepAwake 供既有呼叫端使用

log = logging.getLogger(__name__)


def _rid(info: zipfile.ZipInfo) -> str:
    return PurePosixPath(info.filename).stem


def stream_index(
    *,
    mapping: Mapping,
    index_dir: str | Path,
    zip_url: str | None = None,
    zip_path: str | None = None,
    raw_dir: str | Path = "data/raw",
    slim_dir: str | Path | None = "data/slim",
    tmp_dir: str | Path = "data/raw/_stream",
    workers: int = 2,
    keep_raw: bool = False,
    limit: int | None = None,
    members: list[str] | None = None,
    suffix: str = ".h5",
    known_issue_files: set[str] | None = None,
    retries: int = 2,
    rebuild_every: int = 5,
    session_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    index_dir = Path(index_dir)
    raw_dir, tmp_dir = Path(raw_dir), Path(tmp_dir)
    slim = Path(slim_dir) if slim_dir else None
    rec_dir = index_dir / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    known = known_issue_files or set()

    zf0, src0, close0 = open_zip(zip_url=zip_url, zip_path=zip_path,
                                 session=session_factory() if session_factory and zip_url else None)
    src0.pop("reader", None)
    try:
        infos = sorted((i for i in zf0.infolist() if not i.is_dir() and i.filename.lower().endswith(suffix)),
                       key=lambda i: natural_key(i.filename))
    finally:
        close0()
    if members:
        want = {PurePosixPath(m).name for m in members}
        infos = [i for i in infos if PurePosixPath(i.filename).name in want]
    done = {p.stem for p in rec_dir.glob("*.json") if not p.name.endswith(".error.json")}
    todo = [i for i in infos if _rid(i) not in done]
    if limit is not None:
        todo = todo[:limit]
    total_dl = sum(i.compress_size for i in todo)
    biggest = max((i.file_size for i in todo), default=0)
    log.info("== 全量索引 ==  來源：%s", zip_path or zip_url)
    log.info("zip 內 %d 個錄製檔；已完成 %d 個；這次要處理 %d 個（下載約 %s）", len(infos), len(done), len(todo), human_bytes(total_dl))
    free = disk_free_bytes(tmp_dir)
    need = biggest * max(1, workers) * 1.1 + (2 << 30)
    if free is not None and todo and free < need:
        raise OSError(f"磁碟剩餘 {human_bytes(free)}，同時處理 {workers} 個檔至少需要約 {human_bytes(need)}")

    progress_path = index_dir / "stream_progress.json"
    state = {"started_at": now_iso(), "total": len(todo), "done": 0, "failed": 0, "bytes_done": 0,
             "bytes_total": total_dl, "workers": workers, "current": [], "finished": False, "eta_s": None}
    lock = threading.Lock()
    t_start = time.time()

    def save_progress() -> None:
        with lock:
            elapsed = time.time() - t_start
            rate = state["bytes_done"] / elapsed if elapsed > 0 else 0
            state["rate_bytes_per_s"] = rate
            state["eta_s"] = (state["bytes_total"] - state["bytes_done"]) / rate if rate > 0 else None
            state["updated_at"] = now_iso()
            snapshot = dict(state)
        tmp = progress_path.with_suffix(".json.part")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, progress_path)

    local = threading.local()
    closers: list[Callable[[], None]] = []

    def get_zip() -> zipfile.ZipFile:
        zf = getattr(local, "zf", None)
        if zf is None:
            sess = session_factory() if session_factory and zip_url else None
            zf, _src, close = open_zip(zip_url=zip_url, zip_path=zip_path, session=sess)
            local.zf = zf
            with lock:
                closers.append(close)
        return zf

    def reset_zip() -> None:
        local.zf = None  # 連線出錯時下次重開

    def work(info: zipfile.ZipInfo) -> dict[str, Any]:
        rid = _rid(info)
        name = PurePosixPath(info.filename).name
        with lock:
            state["current"].append(rid)
        local_file = raw_dir / name
        use_local = local_file.exists() and local_file.stat().st_size == info.file_size
        last_err: Exception | None = None
        try:
            for attempt in range(retries + 1):
                h5_path, sha, temp = None, None, False
                try:
                    if use_local:
                        h5_path, sha = local_file, None
                    else:
                        res = extract_member(get_zip(), info, tmp_dir, progress=False)
                        h5_path, sha, temp = Path(res["path"]), res["sha256"], True
                    result = index_h5(h5_path, mapping, rid, source=f"zip:{info.filename}", known_issue_files=known)
                    result["source_size_bytes"] = info.file_size
                    result["source_compressed_bytes"] = info.compress_size
                    result["source_crc32"] = f"{info.CRC:08x}"
                    if sha:
                        result["source_sha256"] = sha
                    if slim is not None:
                        result["slim"] = make_slim(h5_path, slim / f"{rid}.h5", mapping,
                                                   {"source": info.filename, "source_crc32": f"{info.CRC:08x}",
                                                    "source_sha256": sha})
                    save_recording(result, index_dir)
                    if temp and keep_raw:
                        os.replace(h5_path, raw_dir / name)
                        temp = False
                    n_ins = sum(1 for r in result["segments"] if r.get("action") == (mapping.target_action or "insert"))
                    return {"rid": rid, "ok": True, "segments": len(result["segments"]), "insert": n_ins,
                            "local": use_local}
                except Exception as e:  # noqa: BLE001 - 重試後仍失敗才記錄
                    last_err = e
                    log.warning("  %s 第 %d 次失敗：%s: %s", rid, attempt + 1, type(e).__name__, e)
                    reset_zip()
                    time.sleep(min(5 * (attempt + 1), 20))
                finally:
                    if temp and h5_path is not None:
                        try:
                            os.remove(h5_path)
                        except OSError:
                            pass
            save_recording_error(rid, f"zip:{info.filename}", f"{type(last_err).__name__}: {last_err}", index_dir)
            return {"rid": rid, "ok": False, "error": f"{type(last_err).__name__}: {last_err}"}
        finally:
            with lock:
                if rid in state["current"]:
                    state["current"].remove(rid)

    results: list[dict] = []
    save_progress()
    with KeepAwake(), ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: dict[Future, zipfile.ZipInfo] = {}
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < max(1, workers):
                info = queue.pop(0)
                pending[pool.submit(work, info)] = info
            finished, _ = wait(list(pending), timeout=30, return_when=FIRST_COMPLETED)
            if not finished:
                save_progress()
            for fut in finished:
                info = pending.pop(fut)
                r = fut.result()
                results.append(r)
                with lock:
                    state["done" if r["ok"] else "failed"] += 1
                    state["bytes_done"] += info.compress_size
                save_progress()
                k = state["done"] + state["failed"]
                eta = state.get("eta_s")
                if r["ok"]:
                    log.info("[%d/%d] %s 完成：%d 段（Insert %d）%s；%.2f MB/s，預計剩 %s",
                             k, len(todo), r["rid"], r["segments"], r["insert"], "（用本機檔）" if r.get("local") else "",
                             (state.get("rate_bytes_per_s") or 0) / 1e6,
                             f"{eta / 3600:.1f} 小時" if eta else "?")
                else:
                    log.error("[%d/%d] %s 失敗：%s（下次執行會重試）", k, len(todo), r["rid"], r["error"])
                if k % max(1, rebuild_every) == 0:
                    write_index_outputs(index_dir, mapping)
    for close in closers:
        try:
            close()
        except Exception:  # noqa: BLE001
            pass
    outs = write_index_outputs(index_dir, mapping)
    with lock:
        state["finished"] = True
        state["finished_at"] = now_iso()
    save_progress()
    return {"processed": len(results), "ok": sum(1 for r in results if r["ok"]),
            "failed": [r for r in results if not r["ok"]], "outputs": outs,
            "already_done": len(done), "total_in_zip": len(infos)}


def local_index(
    *,
    mapping: Mapping,
    index_dir: str | Path,
    paths: list[Path],
    known_issue_files: set[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """本機檔案（data/raw 或 data/slim）建索引；單檔失敗不中止。"""
    rec_dir = Path(index_dir) / "recordings"
    done = {p.stem for p in rec_dir.glob("*.json") if not p.name.endswith(".error.json")} if rec_dir.exists() else set()
    ok, failed, skipped = 0, [], 0
    for i, p in enumerate(paths, 1):
        rid = p.stem
        if rid in done and not force:
            skipped += 1
            continue
        try:
            res = index_h5(p, mapping, rid, source=str(p), known_issue_files=known_issue_files)
            save_recording(res, index_dir)
            ok += 1
            log.info("[%d/%d] %s：%d 段", i, len(paths), rid, len(res["segments"]))
        except Exception as e:  # noqa: BLE001
            save_recording_error(rid, str(p), f"{type(e).__name__}: {e}", index_dir)
            failed.append({"rid": rid, "error": f"{type(e).__name__}: {e}"})
            log.error("[%d/%d] %s 失敗：%s", i, len(paths), rid, e)
    outs = write_index_outputs(index_dir, mapping)
    return {"ok": ok, "failed": failed, "skipped": skipped, "outputs": outs}

