"""命令列入口：inspect-h5、fetch-samples、envinfo。路徑一律由 CLI 或 YAML 指定。"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .config import load_config

log = logging.getLogger("assembly_monitor")

H5_SUFFIXES = (".h5", ".hdf5", ".he5")


def expand_paths(items: list[str]) -> list[Path]:
    """接受檔案、資料夾（取其中的 .h5/.hdf5，不遞迴）或萬用字元；去重並保持順序。"""
    out: list[Path] = []
    seen: set[str] = set()
    for it in items:
        p = Path(it)
        if p.is_dir():
            found = sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in H5_SUFFIXES)
        elif any(ch in it for ch in "*?["):
            found = sorted(Path(x) for x in glob.glob(it))
        else:
            found = [p]
        for q in found:
            key = str(q.resolve()) if q.exists() else str(q)
            if key not in seen:
                seen.add(key)
                out.append(q)
    return out


# ---------------------------------------------------------------- inspect-h5


def inspect_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(
        prog="inspect-h5",
        description="列出 HDF5 的 group/dataset/shape/dtype、候選欄位與相機時間軸（不假設固定 key）。",
    )
    ap.add_argument("paths", nargs="+", help="HDF5 檔、資料夾或萬用字元，例如 data/raw 或 data/raw/*.h5")
    ap.add_argument("--config", default="configs/dataset.yaml", help="資料集設定檔（預設 configs/dataset.yaml）")
    ap.add_argument("--out", default=None, help="輸出資料夾（預設 config 的 paths.inspect_dir）")
    ap.add_argument("--expect-cameras", default=None, help="逗號分隔的預期相機名稱，缺少時報錯，例如 hama1,hama2,hand")
    ap.add_argument("--preview-n", type=int, default=None, help="每個資料集預覽頭尾幾筆（預設 5）")
    ap.add_argument("--max-read-mb", type=float, default=None, help="單一陣列整段讀取的上限 MB（預設 64）")
    ap.add_argument("--no-media-probe", action="store_true", help="不探測內嵌影片 / 音訊的中繼資料")
    ap.add_argument("--no-count-frames", action="store_true", help="影片標頭沒有幀數時不逐包計數")
    ap.add_argument("--collapse-threshold", type=int, default=None, help="同構子群組超過幾個就在 tree.txt 摺疊（0=不摺疊）")
    ap.add_argument("--max-depth", type=int, default=None, help="最深走訪層數（預設不限）")
    ap.add_argument("--log", default=None, help="另存紀錄檔路徑，例如 logs/inspect.log")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"assembly_monitor {__version__}")
    args = ap.parse_args(argv)
    setup_logging(args.log, args.verbose)

    from .inspect_h5 import InspectOptions, inspect_file
    from .report import write_file_outputs, write_summary

    cfg = load_config(args.config)
    icfg = cfg["inspect"]
    expected = (
        [c.strip() for c in args.expect_cameras.split(",") if c.strip()]
        if args.expect_cameras is not None
        else list(icfg.get("expected_cameras") or [])
    )
    opts = InspectOptions(
        preview_n=args.preview_n if args.preview_n is not None else int(icfg["preview_n"]),
        max_read_mb=args.max_read_mb if args.max_read_mb is not None else float(icfg["max_read_mb"]),
        media_probe=(not args.no_media_probe) and bool(icfg["media_probe"]),
        count_frames=(not args.no_count_frames) and bool(icfg["count_frames"]),
        collapse_threshold=args.collapse_threshold if args.collapse_threshold is not None else int(icfg["collapse_threshold"]),
        expected_cameras=expected,
        max_depth=args.max_depth,
    )
    out_dir = Path(args.out or cfg["paths"]["inspect_dir"])
    files = expand_paths(args.paths)
    if not files:
        log.error("找不到任何 HDF5 檔：%s", args.paths)
        return 2
    log.info("inspect-h5 %s：%d 個檔案 → %s", __version__, len(files), out_dir)
    results, failures = [], []
    for i, fp in enumerate(files, 1):
        t0 = time.time()
        log.info("[%d/%d] %s", i, len(files), fp)
        try:
            if not fp.exists():
                raise FileNotFoundError(str(fp))
            res = inspect_file(fp, opts)
            outs = write_file_outputs(res, out_dir, opts.collapse_threshold)
            c = res["candidates"]
            n_err = sum(1 for a in c["anomalies"] if a["severity"] == "error")
            n_warn = sum(1 for a in c["anomalies"] if a["severity"] == "warn")
            log.info(
                "    節點 %d、影片 %d、時間陣列 %d、高層 segment %d、低層 %d、error %d、warn %d（%.1fs）",
                len(res["nodes"]), len(c["streams"]["cameras"]), len(c["streams"]["time_arrays"]),
                c["segments"]["n_level0"], c["segments"]["n_level1plus"], n_err, n_warn, time.time() - t0,
            )
            log.info("    → %s", Path(outs["tree"]).parent)
            results.append(res)
        except Exception as e:  # noqa: BLE001 - 單檔失敗不中止整批（P1-3）
            log.error("    失敗：%s: %s", type(e).__name__, e)
            log.debug("詳細錯誤", exc_info=True)
            failures.append({"file": str(fp), "error": f"{type(e).__name__}: {e}"})
    if results or failures:
        s = write_summary(results, failures, out_dir)
        log.info("總表：%s", s["summary_md"])
    return 0 if not failures else 1


# ---------------------------------------------------------------- fetch-samples


def fetch_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(
        prog="fetch-samples",
        description="只從 REASSEMBLE 的 data.zip 取出少數 .h5 樣本（HTTP Range 分段下載，或讀本機 zip）。",
    )
    ap.add_argument("--config", default="configs/dataset.yaml")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--zip-url", default=None, help="遠端 data.zip 網址（預設取 config 的 dataset.zip_url）")
    src.add_argument("--zip-path", default=None, help="本機已下載的 data.zip（只解出選中的檔，不整包解壓）")
    ap.add_argument("--out", default=None, help="存放資料夾（預設 config 的 paths.raw_dir）")
    ap.add_argument("--n", type=int, default=None, help="取幾個檔（預設 3）")
    ap.add_argument("--strategy", choices=("spread", "smallest", "random"), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--members", nargs="*", default=None, help="直接指定檔名（例如 2025-01-09-13-59-54.h5）")
    ap.add_argument("--include-known-issues", action="store_true", help="不要排除官方註記有問題的錄製檔")
    ap.add_argument("--max-total-gb", type=float, default=None, help="下載量上限，以 zip 內壓縮大小計（預設 3 GB）")
    ap.add_argument("--list", action="store_true", help="只列出 zip 內容並存 zip_listing.csv，不下載")
    ap.add_argument("--no-readme", action="store_true", help="不下載官方 README.txt")
    ap.add_argument("-y", "--yes", action="store_true", help="不詢問直接下載")
    ap.add_argument("--log", default=None, help="另存紀錄檔路徑，例如 logs/fetch_samples.log")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.log, args.verbose)

    from .fetch import FetchCancelled, RangeNotSupported, fetch_samples

    cfg = load_config(args.config)
    ds, sc = cfg["dataset"], cfg["samples"]
    zip_url = None if args.zip_path else (args.zip_url or ds.get("zip_url"))
    max_gb = args.max_total_gb if args.max_total_gb is not None else float(sc.get("max_total_gb") or 3.0)
    try:
        res = fetch_samples(
            zip_url=zip_url,
            zip_path=args.zip_path,
            out_dir=args.out or cfg["paths"]["raw_dir"],
            index_dir=cfg["paths"]["index_dir"],
            n=args.n if args.n is not None else int(sc["n"]),
            strategy=args.strategy or sc["strategy"],
            seed=args.seed if args.seed is not None else int(sc["seed"]),
            suffix=sc.get("member_suffix", ".h5"),
            exclude_names=[] if args.include_known_issues or not sc.get("exclude_known_issues", True)
            else list(cfg.get("known_issue_files") or []),
            members=args.members,
            max_total_bytes=int(max_gb * 1024**3) if max_gb > 0 else None,
            assume_yes=args.yes,
            list_only=args.list,
            readme_url=None if args.no_readme or not sc.get("with_readme", True) else ds.get("readme_url"),
            record_url=ds.get("record_url"),
        )
    except RangeNotSupported as e:
        log.error("無法只下載部分檔案：%s", e)
        log.error("請改用瀏覽器開啟 %s 下載 data.zip（約 54.8 GiB），再執行：", ds.get("record_url"))
        log.error("  python -m scripts.fetch_samples --zip-path <data.zip 的路徑>")
        return 3
    except FetchCancelled as e:
        log.warning("%s", e)
        return 5
    except PermissionError as e:
        log.error("伺服器拒絕存取：%s", e)
        log.error("請在瀏覽器開啟 %s 確認是否能下載。", ds.get("record_url"))
        return 4
    except Exception as e:  # noqa: BLE001
        log.error("取得樣本失敗：%s: %s", type(e).__name__, e)
        log.debug("詳細錯誤", exc_info=True)
        return 1
    if res.get("status") == "nothing_selected":
        return 2
    if not args.list:
        log.info("完成：新下載 %d 個、已存在 %d 個。下一步：python -m scripts.inspect_h5 %s",
                 len(res["downloaded"]), len(res["skipped"]), args.out or cfg["paths"]["raw_dir"])
    return 0


# ---------------------------------------------------------------- export-check-clips


def check_clips_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(
        prog="export-check-clips",
        description="把要人工核對的 segment 剪成多視角對齊短片（綠框＝標註區間），並產生核對清單。",
    )
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--inspect-dir", default=None, help="inspect-h5 的輸出（預設 config 的 paths.inspect_dir）")
    ap.add_argument("--raw-dir", default=None, help="HDF5 原檔資料夾（預設 config 的 paths.raw_dir）")
    ap.add_argument("--out", default="data/derived/manual_check", help="短片輸出資料夾")
    ap.add_argument("--n", type=int, default=5, help="幾段（規格至少 5 段）")
    ap.add_argument("--pad", type=float, default=1.5, help="前後各加幾秒緩衝")
    ap.add_argument("--cameras", default=None, help="逗號分隔的相機名稱（預設用 candidates 裡全部相機）")
    ap.add_argument("--tile", default="640x480", help="每個視角的畫面大小，例如 640x480")
    ap.add_argument("--fps", type=float, default=30.0, help="輸出影片播放幀率")
    ap.add_argument("--from-index", action="store_true", help="第 2 包：從 segments.csv 抽選定零件的段（讀 data/slim）")
    ap.add_argument("--part", default=None, help="--from-index：零件名稱（預設 field_mapping 的 target_part）")
    ap.add_argument("--action", default=None, help="--from-index：動作（預設 field_mapping 的 target_action）")
    ap.add_argument("--index-dir", default=None, help="--from-index：索引資料夾（預設 config 的 paths.index_dir）")
    ap.add_argument("--slim-dir", default="data/slim", help="--from-index：精簡版資料夾")
    ap.add_argument("--mapping", default="configs/field_mapping.yaml")
    ap.add_argument("--seed", type=int, default=0, help="--from-index：抽樣種子")
    ap.add_argument("--log", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.log, args.verbose)

    from .check_clips import export_check_clips, export_check_clips_from_index

    cfg = load_config(args.config)
    try:
        tw, th = (int(x) for x in args.tile.lower().split("x"))
    except ValueError:
        log.error("--tile 格式要像 640x480")
        return 2
    try:
        if args.from_index:
            from .mapping import load_mapping

            res = export_check_clips_from_index(
                index_dir=args.index_dir or cfg["paths"]["index_dir"],
                slim_dir=args.slim_dir,
                out_dir=args.out,
                mapping=load_mapping(args.mapping),
                part=args.part,
                action=args.action,
                n=args.n,
                seed=args.seed,
                pad=args.pad,
                tile=(tw, th),
                fps=args.fps,
            )
        else:
            res = export_check_clips(
                inspect_dir=args.inspect_dir or cfg["paths"]["inspect_dir"],
                raw_dir=args.raw_dir or cfg["paths"]["raw_dir"],
                out_dir=args.out,
                n=args.n,
                pad=args.pad,
                cameras=[c.strip() for c in args.cameras.split(",") if c.strip()] if args.cameras else None,
                tile=(tw, th),
                fps=args.fps,
            )
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    log.info("完成 %d 段、失敗 %d 段。核對清單：%s", len(res["results"]), len(res["failures"]), res["checklist"])
    return 0 if not res["failures"] else 1


# ---------------------------------------------------------------- index-segments


def index_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(
        prog="index-segments",
        description="依 configs/field_mapping.yaml 建 segment 索引與驗證；可本機掃描或從遠端 zip 全量串流。",
    )
    ap.add_argument("paths", nargs="*", help="本機模式：HDF5 檔或資料夾（預設 data/raw）")
    ap.add_argument("--source", choices=("local", "remote"), default="local", help="local=本機檔；remote=逐一下載 zip 成員")
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--mapping", default="configs/field_mapping.yaml")
    ap.add_argument("--out", default=None, help="索引輸出資料夾（預設 config 的 paths.index_dir）")
    ap.add_argument("--zip-url", default=None, help="remote：zip 網址（預設 config 的 dataset.zip_url）")
    ap.add_argument("--zip-path", default=None, help="remote：改用本機已下載的 data.zip")
    ap.add_argument("--slim-dir", default="data/slim", help="remote：精簡版存放處（影片+時間戳+標註）")
    ap.add_argument("--no-slim", action="store_true", help="remote：不存精簡版")
    ap.add_argument("--keep-raw", action="store_true", help="remote：保留完整原檔到 data/raw（很佔空間）")
    ap.add_argument("--workers", type=int, default=2, help="remote：同時下載幾個檔（預設 2）")
    ap.add_argument("--limit", type=int, default=None, help="remote：這次最多處理幾個檔（測試用）")
    ap.add_argument("--force", action="store_true", help="local：已索引過的也重做")
    ap.add_argument("--report-only", action="store_true", help="只從 recordings/*.json 重建 CSV 與報表")
    ap.add_argument("--log", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.log, args.verbose)

    from .index import write_index_outputs
    from .mapping import MappingError, load_mapping
    from .stream import local_index, stream_index

    cfg = load_config(args.config)
    try:
        mapping = load_mapping(args.mapping)
    except MappingError as e:
        log.error("%s", e)
        return 2
    index_dir = Path(args.out or cfg["paths"]["index_dir"])
    known = {str(x) for x in (cfg.get("known_issue_files") or [])}
    if args.report_only:
        outs = write_index_outputs(index_dir, mapping)
        log.info("已重建：%s", ", ".join(Path(v).name for v in outs.values()))
        return 0
    if args.source == "remote":
        try:
            res = stream_index(
                mapping=mapping,
                index_dir=index_dir,
                zip_url=None if args.zip_path else (args.zip_url or cfg["dataset"].get("zip_url")),
                zip_path=args.zip_path,
                raw_dir=cfg["paths"]["raw_dir"],
                slim_dir=None if args.no_slim else args.slim_dir,
                tmp_dir=str(Path(cfg["paths"]["raw_dir"]) / "_stream"),
                workers=args.workers,
                keep_raw=args.keep_raw,
                limit=args.limit,
                known_issue_files=known,
            )
        except Exception as e:  # noqa: BLE001
            log.error("全量索引中止：%s: %s", type(e).__name__, e)
            log.debug("詳細錯誤", exc_info=True)
            return 1
        log.info("完成：這次成功 %d、失敗 %d；總共已索引 %d / %d。報表：%s",
                 res["ok"], len(res["failed"]), res["already_done"] + res["ok"], res["total_in_zip"],
                 res["outputs"]["action_summary"])
        return 0 if not res["failed"] else 1
    files = expand_paths(args.paths or [cfg["paths"]["raw_dir"]])
    if not files:
        log.error("找不到 HDF5 檔")
        return 2
    res = local_index(mapping=mapping, index_dir=index_dir, paths=files, known_issue_files=known, force=args.force)
    log.info("完成：成功 %d、失敗 %d、略過（已索引）%d。報表：%s",
             res["ok"], len(res["failed"]), res["skipped"], res["outputs"]["action_summary"])
    return 0 if not res["failed"] else 1


# ---------------------------------------------------------------- extract-frames


def extract_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(
        prog="extract-frames",
        description="第 3 包：依真實時間戳擷取選定零件各段影格（1/4/8 幀均勻取樣含首尾）、多視角預覽與 metadata。",
    )
    ap.add_argument("--set", dest="which", choices=("pilot", "all", "dev", "test"), default="pilot",
                    help="pilot=失敗/成功各取若干段（預設 20+20）；all=該零件全部有效段；dev=切分的開發集（第 5 包試跑用）；"
                         "test=測試集（只在最終評估，要加 --unlock-test）")
    ap.add_argument("--unlock-test", action="store_true", help="--set test 時必須加（最終評估才用）")
    ap.add_argument("--split-dir", default="data/splits/bnc_dev25_test75_s0", help="--set dev 時讀的切分資料夾")
    ap.add_argument("--part", default=None, help="零件（預設 field_mapping 的 target_part）")
    ap.add_argument("--action", default=None, help="動作（預設 field_mapping 的 target_action）")
    ap.add_argument("--n-fail", type=int, default=20)
    ap.add_argument("--n-ok", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-frames", type=int, nargs="+", default=[1, 4, 8], help="取樣幀數，可給多個（預設 1 4 8）")
    ap.add_argument("--start-buffer", type=float, default=None, help="標註起點前幾秒（預設 field_mapping 的 extraction）")
    ap.add_argument("--end-buffer", type=float, default=None, help="標註終點後幾秒（預設 field_mapping 的 extraction）")
    ap.add_argument("--format", dest="fmt", choices=("jpg", "png"), default="jpg")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--cameras", default=None, help="逗號分隔（預設 best_camera_candidates）")
    ap.add_argument("--workers", type=int, default=4, help="同時處理幾個錄製檔")
    ap.add_argument("--force", action="store_true", help="忽略快取全部重做")
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--mapping", default="configs/field_mapping.yaml")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--slim-dir", default="data/slim")
    ap.add_argument("--out", default="data/derived/frames")
    ap.add_argument("--log", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.log, args.verbose)

    from .extract import run_extraction
    from .mapping import MappingError, load_mapping

    cfg = load_config(args.config)
    try:
        mapping = load_mapping(args.mapping)
        seg_ids = None
        if args.which in ("dev", "test"):
            from .split import load_split

            seg_ids = [r["segment_id"] for r in load_split(args.split_dir, include=(args.which,),
                                                           allow_test=args.which == "test" and args.unlock_test)]
            log.info("%s %d 段（%s）", "開發集" if args.which == "dev" else "測試集", len(seg_ids), args.split_dir)
        res = run_extraction(
            mapping=mapping,
            index_dir=args.index_dir or cfg["paths"]["index_dir"],
            slim_dir=args.slim_dir,
            out_dir=args.out,
            part=args.part,
            action=args.action,
            which=args.which,
            n_fail=args.n_fail,
            n_ok=args.n_ok,
            seed=args.seed,
            n_frames_list=tuple(args.n_frames),
            start_buffer=args.start_buffer,
            end_buffer=args.end_buffer,
            fmt=args.fmt,
            jpeg_quality=args.jpeg_quality,
            cameras=[c.strip() for c in args.cameras.split(",") if c.strip()] if args.cameras else None,
            workers=args.workers,
            force=args.force,
            segment_ids=seg_ids,
        )
    except (MappingError, FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("%s", e)
        return 2
    log.info("完成：新做 %d、沿用快取 %d、缺幀 %d。預覽：%s", res["n_done"], res["n_skipped"], res["n_missing_frames"],
             res["gallery"])
    if res["errors"]:
        log.warning("有錯誤：%s", res["errors"])
    return 0 if not res["errors"] else 1


# ---------------------------------------------------------------- run-vlm（第 5 包）


DEFAULT_VLM_MODELS = "majority,smolvlm2,qwen3vl_4b,qwen3vl_8b,qwen35_4b,qwen35_9b,gemma4_e4b,gemma4_12b"


def _model_keys(use: str | None, exp: dict) -> list[str]:
    """--use 優先；否則用 experiment.models；都沒有就跑全部候選。"""
    if use:
        return [k.strip() for k in use.split(",") if k.strip()]
    if exp.get("models"):
        return [str(k) for k in exp["models"]]
    return DEFAULT_VLM_MODELS.split(",")


def run_vlm_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(
        prog="run-vlm",
        description="第 5 包：用現成 VLM 逐段判斷成敗（預設只讀開發集），輸出 raw.jsonl、predictions.jsonl、errors.csv 與摘要。",
    )
    ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--models", nargs="+", default=["configs/model.yaml"],
                    help="模型設定檔（可給多個，例如 configs/model.yaml configs/model_train.yaml）")
    ap.add_argument("--use", default=None,
                    help="要跑的模型 key，逗號分隔（依序執行）；預設讀 experiment.models，沒有就跑全部候選")
    ap.add_argument("--frames", type=int, nargs="+", default=None, help="幀數，例如 1 4 8（預設讀 experiment.frames）")
    ap.add_argument("--name", default=None, help="覆寫 experiment.name（輸出資料夾）")
    ap.add_argument("--limit", type=int, default=None, help="每個組合最多跑幾段（小樣本檢查用）")
    ap.add_argument("--force", action="store_true", help="重新開始（舊結果改名保留）")
    ap.add_argument("--unlock-test", action="store_true", help="讀測試集（只在最終評估時使用）")
    ap.add_argument("--annotation", default="configs/annotation.yaml", help="失敗原因的類別以這份為準")
    ap.add_argument("--mapping", default="configs/field_mapping.yaml")
    ap.add_argument("--check", action="store_true", help="只檢查 GPU 與套件環境")
    ap.add_argument("--compare", nargs="+", default=None, metavar="YAML",
                    help="比較多個提示詞版本的開發集結果（可用萬用字元，例如 \"configs/experiment*.yaml\"），寫 experiments/prompt_versions.md")
    ap.add_argument("--log", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.log, args.verbose)

    from . import vlm

    if args.check:
        rep = vlm.env_report()
        for k, v in rep.items():
            log.info("  %-16s %s", k, v)
        if not rep.get("cuda_available"):
            log.error("CUDA 不可用：torch 沒有 CUDA 版或驅動太舊（請看 logs\\nvidia_smi.txt）")
            return 3
        miss = [m for m in ("transformers", "accelerate", "bitsandbytes", "huggingface_hub", "num2words", "torchvision")
                if not rep.get(m) or str(rep.get(m)).startswith("error")]
        if miss:
            log.error("缺少套件：%s", miss)
            return 3
        return 0
    if args.compare:
        import glob

        paths = sorted({p for pat in args.compare for p in (glob.glob(pat) or [pat])})
        try:
            rep = vlm.compare_experiments(paths)
        except Exception as e:  # noqa: BLE001
            log.error("%s: %s", type(e).__name__, e)
            return 2
        log.info("提示詞版本比較：%s（%d 列）", rep["md"], len(rep["rows"]))
        for sk in rep["skipped"]:
            log.info("  略過 %s", sk)
        return 0
    try:
        from .mapping import load_mapping
        from .schema import load_annotation_config

        exp = vlm.load_experiment(args.experiment)
        if args.name:
            exp["name"] = args.name
        model_cfgs = vlm.load_model_configs(args.models)
        acfg = load_annotation_config(args.annotation, load_mapping(args.mapping))
        from .util import KeepAwake

        with KeepAwake():  # 長時間推論時不讓 Windows 睡眠
            res = vlm.run_experiment(exp=exp, model_cfgs=model_cfgs,
                                     model_keys=_model_keys(args.use, exp),
                                     frames=args.frames or list(exp["frames"]), reasons=list(acfg.reasons),
                                     force=args.force, limit=args.limit, unlock_test=args.unlock_test)
    except Exception as e:  # noqa: BLE001
        log.error("%s: %s", type(e).__name__, e)
        return 2
    log.info("完成。摘要：%s", res["summary"]["md"])
    bad = [r for r in res["results"] if r.get("load_error") or r.get("config_mismatch")]
    return 1 if bad else 0


def download_models_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(prog="download-models", description="第 5 包：從 Hugging Face 下載模型權重（先列大小）。")
    ap.add_argument("--models", nargs="+", default=["configs/model.yaml"], help="模型設定檔（可給多個）")
    ap.add_argument("--use", default="smolvlm2,qwen3vl_4b,qwen3vl_8b,qwen35_4b,qwen35_9b,gemma4_e4b,gemma4_12b,dinov2_base")
    ap.add_argument("--plan", action="store_true", help="只列出大小與授權，不下載")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    import os

    from . import vlm

    try:
        cfgs = vlm.load_model_configs(args.models)
        keys = [k.strip() for k in args.use.split(",") if k.strip()]
        plan = vlm.model_download_plan(cfgs, keys)
        total = sum(p["size_gb"] for p in plan)
        todo = sum(p.get("missing_gb", p["size_gb"]) for p in plan)
        for p in plan:
            m = p.get("missing_gb", p["size_gb"])
            state = "已下載" if m == 0 else f"還要下載 {m:.1f} GB"
            log.info("  %-11s %-38s %6.1f GB  %-16s license=%s  sha=%s", p["key"], p["hf_id"], p["size_gb"], state,
                     p["license"], (p["sha"] or "")[:10])
        log.info("  合計約 %.1f GB，其中還要下載約 %.1f GB（存到 HF_HOME=%s）", total, todo,
                 os.environ.get("HF_HOME") or "預設快取")
        if args.plan:
            return 0
        vlm.download_models(cfgs, keys)
    except Exception as e:  # noqa: BLE001
        log.error("%s: %s", type(e).__name__, e)
        return 2
    log.info("下載完成")
    return 0


def probe_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(
        prog="probe",
        description="做法 A：用其他零件的插入片段訓練小分類器（凍結影像編碼器＋邏輯迴歸），在 BNC 開發集評分。",
    )
    ap.add_argument("step", choices=("pool", "extract", "run", "all", "sweep", "time"),
                    help="pool＝列出訓練池；extract＝擷取訓練池與開發集的影格；run＝特徵、訓練、預測；all＝extract＋run；"
                         "sweep＝第 7 包：跑 probe_sweep 設定的所有候選，依交叉驗證挑一個（--config configs/probe_sweep_v2.yaml）；"
                         "time＝第 8 包：量每段的執行時間（重新編碼開發集的段，不讀快取、不碰測試集）")
    ap.add_argument("--config", default="configs/probe.yaml")
    ap.add_argument("--frames", type=int, nargs="+", default=None)
    ap.add_argument("--unlock-test", action="store_true", help="在 BNC 測試集評分（只在最終評估時使用）")
    ap.add_argument("--mapping", default="configs/field_mapping.yaml")
    ap.add_argument("--slim-dir", default="data/slim")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-segments", type=int, default=20, help="time：量幾段")
    ap.add_argument("--log", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.log, args.verbose)
    from . import probe

    try:
        if args.step == "sweep":
            from .util import KeepAwake

            sw = probe.load_sweep_config(args.config)
            with KeepAwake():
                rep = probe.run_sweep(sw)
            b = rep["best"]
            if b:
                log.info("選到：%s，最後一張 +%g 秒，%d 幀，%s（交叉驗證平衡準確率 %.3f）→ %s", b["encoder"], b["end_buffer"],
                         b["n_frames"], b["temporal"], b["cv_balanced_accuracy"], rep["selected_config"])
            log.info("候選比較：%s（%d 列；失敗的候選 %d 個）", rep["md"], len(rep["rows"]), len(rep["errors"]))
            return 0 if b else 2
        cfg = probe.load_probe_config(args.config)
        if args.step == "time":
            probe.time_probe(cfg, n_segments=args.n_segments, frames=(args.frames or [None])[0])
            return 0
        if args.step == "pool":
            log.info("%s", probe.describe_pool(cfg))
            return 0
        if args.step in ("extract", "all"):
            from .mapping import load_mapping

            pools = probe.build_pools(cfg, allow_test=args.unlock_test)
            segs = pools["train"] + (pools["test"] if args.unlock_test else pools["dev"])
            res = probe.extract_frames(cfg, segs, mapping=load_mapping(args.mapping), slim_dir=args.slim_dir, workers=args.workers)
            log.info("影格：新做 %d、沿用 %d、缺幀 %d", res["n_done"], res["n_skipped"], res["n_missing_frames"])
            if res["errors"]:
                log.warning("擷取錯誤：%s", res["errors"][:5])
        if args.step in ("run", "all"):
            from .util import KeepAwake

            with KeepAwake():
                rep = probe.run_probe(cfg, frames=args.frames, allow_test=args.unlock_test)
            log.info("完成。摘要：%s", (rep.get("summary") or {}).get("md"))
    except Exception as e:  # noqa: BLE001
        log.error("%s: %s", type(e).__name__, e)
        return 2
    return 0


# ---------------------------------------------------------------- finetune（第 7 包）


def finetune_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(prog="finetune", description="第 7 包：Qwen3-VL-8B 的 QLoRA 微調（用其他零件的插入片段；不讀 BNC 測試集）。")
    ap.add_argument("step", choices=("plan", "smoke", "train"),
                    help="plan＝列出訓練資料與一個樣本的提示詞（不用 GPU）；smoke＝幾段試跑（顯存、速度、估計時間）；train＝正式訓練")
    ap.add_argument("--config", default="configs/finetune_qwen3vl8b.yaml")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    from . import finetune

    try:
        cfg = finetune.load_finetune_config(args.config)
        if args.step == "plan":
            pl = finetune.describe_plan(cfg)
            t, d = pl["train"], pl["dev"]
            log.info("訓練資料：%d 段（失敗 %d），%d 支錄影；缺影格 %d 段（%s）", t["n"], t["n_failure"], t["n_recordings"],
                     len(pl["train_missing"]), pl["sampling_key"])
            for part, c in t["per_part"].items():
                log.info("  %-14s 成功 %3d  失敗 %3d  → 提示詞裡叫「%s」", part, c["success"], c["failure"], pl["part_labels"][part])
            log.info("BNC 開發集（只記錄）：%d 段（失敗 %d）；已排除 BNC 測試集的 %d 支錄影", d["n"], d["n_failure"],
                     pl["n_test_recordings_excluded"])
            log.info("類別權重 %s；更新次數 %d", pl["class_weights"], pl["optimizer_steps"])
            if pl["example"]:
                ex = pl["example"]
                log.info("範例樣本 %s（%s，%s，%d 張圖）目標答案：%s\n%s", ex["segment_id"], ex["part"], ex["result"],
                         ex["n_images"], ex["target"], ex["prompt"])
            return 0 if t["n"] >= 2 else 2
        from .util import KeepAwake

        with KeepAwake():
            rep = finetune.train(cfg, smoke=args.step == "smoke")
        if args.step == "smoke":
            log.info("試跑完成：%.1f 秒/段，顯存峰值 %s MB；正式訓練估計約 %s 分鐘", rep.get("sec_per_sample") or 0,
                     rep.get("peak_vram_mb"), rep.get("estimated_full_minutes"))
    except Exception as e:  # noqa: BLE001
        log.error("%s: %s", type(e).__name__, e)
        return 2
    return 0


# ---------------------------------------------------------------- vlm-probe（再訓練：VLM 特徵＋小分類器）

DEFAULT_VLM_PROBE_REFS = ["experiments/dev_ft/qwen3vl_8b/f4", "experiments/dev_eb05/qwen3vl_8b/f4",
                          "experiments/dev_eb1/probe_dinov2_eb1/f1"]


def vlm_probe_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(prog="vlm-probe", description="再訓練：Qwen3-VL-8B 的內部特徵＋邏輯迴歸（其他零件訓練，只看 BNC 開發集）。")
    ap.add_argument("step", choices=("plan", "extract", "fit", "all"),
                    help="plan＝列出資料與已抽的特徵（不用 GPU）；extract＝抽特徵（GPU，可續跑）；fit＝交叉驗證、訓練、預測開發集；all＝extract＋fit")
    ap.add_argument("--config", default="configs/vlm_probe.yaml")
    ap.add_argument("--limit", type=int, default=None, help="extract 最多抽幾段（試跑用）")
    ap.add_argument("--references", nargs="*", default=DEFAULT_VLM_PROBE_REFS, help="開發集報告裡一起列的參考結果資料夾")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    from . import vlm_probe

    try:
        cfg = vlm_probe.load_vlm_probe_config(args.config)
        if args.step == "plan":
            pl = vlm_probe.describe_plan(cfg)
            log.info("訓練池 %d 段（失敗 %d，%d 種零件）、開發集 %d 段；缺影格 %d／%d；已抽特徵 訓練 %d、開發 %d；已排除 BNC 測試集 %d 支錄影",
                     pl["train"]["n"], pl["train"]["n_failure"], len(pl["train"]["per_part"]), pl["dev"]["n"],
                     len(pl["train_missing_frames"]), len(pl["dev_missing_frames"]), pl["features_done"]["train"],
                     pl["features_done"]["dev"], pl["n_test_recordings_excluded"])
            return 0 if pl["train"]["n"] >= 10 else 2
        from .util import KeepAwake

        with KeepAwake():
            if args.step in ("extract", "all"):
                r = vlm_probe.extract(cfg, limit=args.limit)
                log.info("抽特徵完成：新增 %d 段%s", r["n_new"], f"，錯誤 {len(r.get('errors') or [])} 段" if r.get("errors") else "")
            if args.step in ("fit", "all"):
                rep = vlm_probe.fit(cfg, references=args.references)
                b = rep["best"]
                if b:
                    log.info("選到：%s（交叉驗證平衡準確率 %.3f）；開發集 %s/%s、抓到失敗 %s/%s、成功被誤判 %s/%s", b["feature"],
                             b["cv_balanced_accuracy"], b["dev_correct"], b["dev_n"], b["dev_caught"], b["dev_failures"],
                             b["dev_false_alarms"], b["dev_successes"])
                log.info("候選比較：%s；開發集報告：%s", rep["md"], rep["report"])
    except Exception as e:  # noqa: BLE001
        log.error("%s: %s", type(e).__name__, e)
        return 2
    return 0


# ---------------------------------------------------------------- make-split（第 4 包）


def split_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="make-split",
        description="第 4 包：以錄製檔分組切分（固定 seed 與比例），輸出 split.csv、分布摘要、洩漏檢查報告並鎖定 test 清單。",
    )
    ap.add_argument("--ratios", default="dev=0.25,test=0.75", help="集合與比例（預設 dev=0.25,test=0.75）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-key", default="recording_id", help="分組依據（預設 recording_id＝完整錄製檔）")
    ap.add_argument("--stratify", default="has_failure", choices=("has_failure", "none"), help="分層：依錄製檔是否含失敗段")
    ap.add_argument("--name", default=None, help="切分名稱（預設依零件、比例、seed 自動命名）")
    ap.add_argument("--part", default=None)
    ap.add_argument("--action", default=None)
    ap.add_argument("--derived", nargs="*", default=None,
                    help="要稽核的衍生資料夾（預設 data/derived/frames/<動作>_<零件> 與 data/derived/annotation）")
    ap.add_argument("--gold", nargs="*", default=None, help="gold 標註檔（算失敗原因分布；預設 data/annotations/*/gold.jsonl）")
    ap.add_argument("--no-hash", action="store_true", help="不算影格 hash（較快）")
    ap.add_argument("--near-threshold", type=int, default=4, help="dHash 近似門檻（漢明距離）")
    ap.add_argument("--seq-coverage", type=float, default=0.5,
                    help="同一對段每台相機都有這個比例以上的影格近似，才算疑似重複序列（預設 0.5）")
    ap.add_argument("--verify", default=None, metavar="NAME", help="只檢查既有切分是否被改動")
    ap.add_argument("--force", action="store_true", help="已鎖定的切分內容不同時仍覆蓋")
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--mapping", default="configs/field_mapping.yaml")
    ap.add_argument("--annotation-config", default="configs/annotation.yaml")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--log", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    from .util import setup_logging

    setup_logging(args.log, args.verbose)
    from .extract import _slug
    from .mapping import MappingError, load_mapping
    from .schema import load_annotation_config, read_annotations
    from .split import SplitError, make_split, verify_split

    if args.verify:
        problems = verify_split(Path(args.out) / args.verify)
        if problems:
            for p in problems:
                log.error("%s", p)
            return 1
        log.info("切分 %s 未被改動（雜湊相符）", args.verify)
        return 0
    cfg = load_config(args.config)
    try:
        mapping = load_mapping(args.mapping)
        action = args.action or mapping.target_action or "insert"
        part = args.part or mapping.target_part
        derived = args.derived if args.derived is not None else [f"data/derived/frames/{_slug(action)}_{_slug(part)}", "data/derived/annotation"]
        gold_files = args.gold if args.gold is not None else sorted(str(p) for p in Path("data/annotations").glob("*/gold.jsonl"))
        gold = None
        if gold_files:
            acfg = load_annotation_config(args.annotation_config, mapping)
            gold = {}
            for gf in gold_files:
                for r in read_annotations(gf, acfg):
                    if r.get("review_status") in ("approved", "adjudicated"):
                        gold[str(r["segment_id"])] = r
        res = make_split(index_dir=args.index_dir or cfg["paths"]["index_dir"], out_root=args.out, action=action, part=part,
                         ratios=args.ratios, seed=args.seed, group_key=args.group_key, stratify=args.stratify, name=args.name,
                         derived_roots=derived, gold=gold, start_buffer=mapping.start_buffer_sec, end_buffer=mapping.end_buffer_sec,
                         hash_frames=not args.no_hash, near_threshold=args.near_threshold,
                         seq_coverage=args.seq_coverage, force=args.force)
    except (MappingError, SplitError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    s = res["summary"]["splits"]
    for k, v in s.items():
        log.info("  %-5s group %3d、段 %3d（失敗 %2d／成功 %2d）", k, v["n_groups"], v["n_segments"], v["n_failure"], v["n_success"])
    mark = {"pass": "通過", "warn": "通過（有警告，見報告）", "fail": "未通過"}[res["leakage"]["status"]]
    log.info("洩漏檢查：%s　報告：%s", mark, res["reports"]["md"])
    return 0 if res["leakage"]["status"] != "fail" else 1


# ---------------------------------------------------------------- annotate（第 4 包）


def annotate_main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", default=None, help="任務名稱（預設 configs/annotation.yaml 的 task.name）")
    common.add_argument("--config", default="configs/dataset.yaml")
    common.add_argument("--mapping", default="configs/field_mapping.yaml")
    common.add_argument("--annotation-config", default="configs/annotation.yaml")
    common.add_argument("--index-dir", default=None)
    common.add_argument("--pages-dir", default="data/derived/annotation", help="標註頁與影格的根資料夾")
    common.add_argument("--ann-dir", default="data/annotations", help="標註檔的根資料夾")
    common.add_argument("--log", default=None)
    common.add_argument("-v", "--verbose", action="store_true")
    ap = argparse.ArgumentParser(prog="annotate", description="第 4 包標註工具：產生標註頁、收檔、驗證、參考標籤、比較、合併 gold。")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", parents=[common], help="選段並產生密集影格與標註頁")
    b.add_argument("--n-fail", type=int, default=None)
    b.add_argument("--n-ok", type=int, default=None)
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--fps", type=float, default=None)
    b.add_argument("--selection", default=None, help="候選清單（預設第 3 包的 selection_pilot.csv）")
    b.add_argument("--rest-of", nargs="+", default=None, metavar="TASK",
                   help="改成標其餘的段：全部有效段扣掉這些任務已有的段（例如 --task rest56 --rest-of pilot30）")
    b.add_argument("--slim-dir", default="data/slim")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--force", action="store_true")
    sub.add_parser("reference", parents=[common], help="由資料集低階技能產生參考標籤")
    c = sub.add_parser("collect", parents=[common], help="從下載資料夾收標註頁匯出的檔")
    c.add_argument("--from", dest="src", nargs="*", default=None, help="來源資料夾（預設 使用者的 Downloads）")
    v = sub.add_parser("validate", parents=[common], help="驗證標註檔")
    v.add_argument("files", nargs="+")
    v.add_argument("--complete", action="store_true", help="草稿也要求必填")
    cv = sub.add_parser("convert", parents=[common], help="JSONL ↔ CSV")
    cv.add_argument("file")
    cv.add_argument("--to", choices=("csv", "jsonl"), required=True)
    cv.add_argument("--out", default=None)
    cp = sub.add_parser("compare", parents=[common], help="比較各來源、算一致率與爭議清單")
    cp.add_argument("--primary", default=None, help="主標註者代號（預設第一個 human__*.jsonl）")
    mg = sub.add_parser("merge", parents=[common], help="依裁決結果合併 gold")
    mg.add_argument("--primary", default=None)
    sc = sub.add_parser("schema", parents=[common], help="輸出 JSON Schema")
    sc.add_argument("--out", default="configs/annotation.schema.json")
    sub.add_parser("refresh", parents=[common], help="只更新標註頁與設定（改了 annotation.yaml 後用；不重新解碼影片）")
    args = ap.parse_args(argv)
    from .util import setup_logging

    setup_logging(args.log, args.verbose)
    import json as _json

    from .mapping import MappingError, load_mapping
    from .schema import (AnnotationError, json_schema, load_annotation_config, load_segment_infos, read_annotations,
                         validate_records, write_csv, write_jsonl)

    cfg_ds = load_config(args.config)
    try:
        mapping = load_mapping(args.mapping)
        acfg = load_annotation_config(args.annotation_config, mapping)
        action = mapping.target_action or "insert"
        part = mapping.target_part
        index_dir = args.index_dir or cfg_ds["paths"]["index_dir"]
        task = args.task or acfg.task.get("name", "pilot30")
        task_dir = Path(args.pages_dir) / task
        ann_dir = Path(args.ann_dir) / task
        if args.cmd == "schema":
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(_json.dumps(json_schema(acfg), ensure_ascii=False, indent=1), encoding="utf-8")
            log.info("JSON Schema：%s", args.out)
            return 0
        if args.cmd == "refresh":
            from .annotate import refresh_page

            pages = refresh_page(task_dir, acfg)
            log.info("標註頁已更新為 %s／%s：%s", acfg.reason_set, acfg.label_version, pages["index"])
            return 0
        if args.cmd == "convert":
            recs = read_annotations(args.file, acfg)
            out = args.out or str(Path(args.file).with_suffix("." + args.to))
            (write_csv if args.to == "csv" else write_jsonl)(recs, out, acfg)
            log.info("已轉成 %s（%d 筆）", out, len(recs))
            return 0
        segs = load_segment_infos(index_dir, action, part)
        if args.cmd == "build":
            from .annotate import build_task, rest_segments, select_task_segments
            from .extract import _slug

            if args.rest_of:
                if task in args.rest_of:
                    log.error("--task %s 不能同時出現在 --rest-of", task)
                    return 2
                chosen = rest_segments(segs, [Path(args.pages_dir) / t for t in args.rest_of])
                log.info("其餘的段：全部 %d 段扣掉 %s → %d 段", len(segs), "、".join(args.rest_of), len(chosen))
            else:
                sel = args.selection or f"data/derived/frames/{_slug(action)}_{_slug(part)}/selection_pilot.csv"
                if not Path(sel).exists():
                    log.warning("找不到 %s，改從全部有效段挑", sel)
                chosen = select_task_segments(segs, sel, args.n_fail if args.n_fail is not None else int(acfg.task.get("n_fail", 20)),
                                              args.n_ok if args.n_ok is not None else int(acfg.task.get("n_ok", 10)),
                                              args.seed if args.seed is not None else int(acfg.task.get("seed", 0)))
            if not chosen:
                log.error("沒有可標註的段")
                return 2
            res = build_task(mapping=mapping, cfg=acfg, segs=segs, chosen=chosen, index_dir=index_dir, slim_dir=args.slim_dir,
                             out_root=args.pages_dir, task=task, title=f"{action.capitalize()} {part}：{task}",
                             workers=args.workers, force=args.force, fps=args.fps)
            ann_dir.mkdir(parents=True, exist_ok=True)
            log.info("完成：新做 %d、沿用 %d、密集影格 %d 張。標註頁：%s", res["n_done"], res["n_skipped"], res["n_frames"], res["index"])
            if res["errors"]:
                log.warning("有錯誤：%s", res["errors"])
            return 0 if not res["errors"] else 1
        from .annotate import collect_exports, run_compare, run_merge, task_segments, write_reference

        tsegs = task_segments(task_dir, segs)
        if args.cmd == "reference":
            p = write_reference(tsegs, index_dir, ann_dir, acfg)
            log.info("資料集參考標籤：%s（%d 段）", p, len(tsegs))
            return 0
        if args.cmd == "collect":
            src = args.src if args.src is not None else [str(Path.home() / "Downloads")]
            rep = collect_exports(src, task, ann_dir, acfg, tsegs)
            if not rep["collected"]:
                log.warning("在 %s 沒有找到 %s__*.jsonl / .csv（標註頁按「匯出 JSONL」後會存到下載資料夾）", src, task)
            for x in rep["collected"]:
                log.info("  收進 %s → %s（新增 %d、更新 %d，共 %d 筆；問題 %d 筆）", x["file"], x["target"], x["added"], x["updated"],
                         x["n"], len(x["problems"]))
                for p in x["problems"][:5]:
                    log.warning("    第 %s 筆 %s：%s", p["row"], p["segment_id"], "；".join(p["errors"]))
            for p in rep["problems"]:
                log.error("  %s：%s", p["file"], p["error"])
            return 0 if not rep["problems"] else 1
        if args.cmd == "validate":
            bad = 0
            for f in args.files:
                rep = validate_records(read_annotations(f, acfg), acfg, tsegs, True if args.complete else None)
                log.info("%s：%d 筆，問題 %d 筆", f, rep["n_records"], rep["n_problem_rows"])
                for p in rep["problems"][:30]:
                    log.warning("  第 %s 筆 %s：%s", p["row"], p["segment_id"], "；".join(p["errors"]))
                bad += 0 if rep["ok"] else 1
            return 0 if not bad else 1
        if args.cmd == "compare":
            res = run_compare(ann_dir, task_dir, acfg, task, args.primary)
            for k, p in res["pairs"].items():
                m = p["metrics"]
                log.info("  %s：共同 %d 段；結果一致 %s、原因一致 %s（n=%d）、證據命中 %s（n=%d）；爭議 %d 段", k, m["n_common"],
                         m["result"]["rate"], m["failure_reason"]["rate"], m["failure_reason"]["n"], m["evidence"]["hit_rate"],
                         m["evidence"]["n"], m["n_disputes"])
            log.info("報告：%s　爭議清單：%s", res["agreement_html"], res["disputes_csv"])
            return 0
        if args.cmd == "merge":
            rep = run_merge(ann_dir, acfg, args.primary)
            log.info("gold：%s（共 %d 段：同意 %d、裁決 %d、尚未裁決 %d）", rep["gold"], rep["n"], rep["approved"], rep["adjudicated"],
                     len(rep["disputed_unresolved"]))
            if rep.get("difficulty_defaulted"):
                log.info("  畫面難度預設為 %s 的無爭議段：%d 段", acfg.difficulty_default, len(rep["difficulty_defaulted"]))
            return 0
    except (MappingError, AnnotationError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    return 2


# ---------------------------------------------------------------- evaluate / final_test（第 6 包）


def evaluate_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(prog="evaluate", description="第 6 包：開發集評分（所有已跑過的組合，含信賴區間）。測試集請用 final_test。")
    ap.add_argument("--experiments", nargs="+", default=["configs/experiment*.yaml"], help="實驗設定檔（可用萬用字元）")
    ap.add_argument("--out", default="experiments/dev_evaluation")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    import glob

    from . import evaluate

    paths = sorted({p for pat in args.experiments for p in (glob.glob(pat) or [pat])})
    try:
        rep = evaluate.evaluate_dev(paths, args.out, n_boot=args.n_boot)
    except Exception as e:  # noqa: BLE001
        log.error("%s: %s", type(e).__name__, e)
        return 2
    best = rep["results"][0]
    log.info("開發集評分：%s（%d 個組合；兩類平均最高：%s %.0f%%）", rep["md"], len(rep["results"]), best["id"],
             100 * (best["metrics"]["balanced_accuracy"] or 0))
    return 0


def final_test_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(prog="final-test", description="第 6 包：用凍結的設定在測試集跑一次並評分（configs/final_test.yaml）。")
    ap.add_argument("step", choices=("plan", "extract", "run", "evaluate", "all"),
                    help="plan＝列出方法與狀態（不解鎖）；extract＝擷取測試段影格；run＝跑各方法；evaluate＝評分；all＝extract＋run＋evaluate")
    ap.add_argument("--config", default="configs/final_test.yaml")
    ap.add_argument("--annotation", default="configs/annotation.yaml")
    ap.add_argument("--mapping", default="configs/field_mapping.yaml")
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--slim-dir", default="data/slim")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    from . import final

    try:
        cfg = final.load_final_config(args.config)
        if args.step == "plan":
            log.info("凍結：%s（%s）；主方法：%s；輸出：%s/%s", cfg["frozen"], cfg.get("frozen_at"), cfg.get("primary"),
                     cfg["out_dir"], cfg["out_name"])
            for m in cfg["methods"]:
                done = (final.method_run_dir(cfg, m) / "predictions.jsonl").exists()
                log.info("  %-24s %-8s %s%s", m["id"], m["kind"],
                         {k: m[k] for k in m if k not in ("id", "kind")}, "　（已有結果）" if done else "")
            for x in final.needed_extractions(cfg):
                log.info("  需要的測試段影格：最後一張 +%g 秒，幀數 %s", x["end_buffer"], x["frames"])
            return 0 if cfg["frozen"] else 3
        if not cfg["frozen"]:
            log.error("configs/final_test.yaml 還沒凍結（frozen: false）：先在開發集選定方法，再改成 frozen: true")
            return 3
        from .util import KeepAwake

        with KeepAwake():
            if args.step in ("extract", "all"):
                from .extract import run_extraction
                from .mapping import load_mapping
                from .split import load_split

                final.check_or_write_lock(cfg)
                seg_ids = [r["segment_id"] for r in load_split(cfg["split_dir"], include=("test",), allow_test=True)]
                mapping = load_mapping(args.mapping)
                for x in final.needed_extractions(cfg):
                    log.info("擷取測試段 %d 段：最後一張 +%g 秒，幀數 %s", len(seg_ids), x["end_buffer"], x["frames"])
                    run_extraction(mapping=mapping, index_dir=load_config(args.dataset_config)["paths"]["index_dir"],
                                   slim_dir=args.slim_dir, out_dir="data/derived/frames", which="test",
                                   part=cfg.get("part"), n_frames_list=tuple(x["frames"]), end_buffer=x["end_buffer"],
                                   workers=args.workers, segment_ids=seg_ids)
            if args.step in ("run", "all"):
                from .mapping import load_mapping
                from .schema import load_annotation_config

                acfg = load_annotation_config(args.annotation, load_mapping(args.mapping))
                final.run_final(cfg, reasons=list(acfg.reasons))
            if args.step in ("evaluate", "all"):
                rep = final.evaluate_final(cfg)
                log.info("測試集評分：%s", rep["md"])
                for r in rep["results"]:
                    m = r["metrics"]
                    log.info("  %-24s 任務判斷成功率 %d/%d　抓到失敗 %d/%d　成功被誤判 %d/%d", r["id"], m["n_correct"], m["n"],
                             m["n_caught"], m["n_failures"], m["n_false_alarms"], m["n_successes"])
    except Exception as e:  # noqa: BLE001
        log.error("%s: %s", type(e).__name__, e)
        return 2
    return 0


def check_api_main(argv: list[str] | None = None) -> int:
    """第 8 包：確認 API key 與模型名稱可用（每個模型送一句很短的文字，費用不到 0.01 美元）；不印出 key。"""
    from .util import setup_logging

    ap = argparse.ArgumentParser(prog="check-api", description="第 8 包：檢查商業模型 API 是否能用（key、模型名稱、網路）。")
    ap.add_argument("--models", nargs="+", default=["configs/model_p8.yaml"])
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    from . import vlm
    from .api_models import ENV_KEYS, load_env_file

    names = load_env_file(".env")
    log.info(".env 裡有：%s", "、".join(n for n in names if n in ENV_KEYS.values()) or "（沒有 API key）")
    cfgs = vlm.load_model_configs(args.models)
    ok_all, n = True, 0
    for key, cfg in cfgs.items():
        if cfg["type"] not in vlm.API_TYPES:
            continue
        n += 1
        ad = vlm.make_adapter(key, cfg)
        try:
            ad.load()
            out = ad._run(ad._convert([{"type": "text", "text": 'Reply with the JSON {"result": "success"} only.'}]))
            log.info("  %-16s 可以用：%s 回答 %r（%.1f 秒）", key, out.extra.get("api_model") or cfg["model_id"],
                     (out.text or "")[:40], out.latency_s)
        except Exception as e:  # noqa: BLE001
            ok_all = False
            log.error("  %-16s 不能用：%s: %s", key, type(e).__name__, str(e)[:200])
        finally:
            ad.close()
    if not n:
        log.error("設定檔裡沒有 API 模型")
        return 2
    return 0 if ok_all else 1


def timing_main(argv: list[str] | None = None) -> int:
    from .util import setup_logging

    ap = argparse.ArgumentParser(prog="timing", description="第 8 包：已跑過的凍結測試各方法的執行時間（只讀輸出，不重跑）。")
    ap.add_argument("--final", nargs="+", default=["configs/final_test.yaml", "configs/final_test_train.yaml"])
    ap.add_argument("--out", default="experiments/timing")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    setup_logging(args.log, False)
    from .timing import timing_report

    try:
        rep = timing_report(args.final, args.out)
    except Exception as e:  # noqa: BLE001
        log.error("%s: %s", type(e).__name__, e)
        return 2
    log.info("執行時間：%s", rep["md"])
    return 0


# ---------------------------------------------------------------- envinfo


def envinfo_main(argv: list[str] | None = None) -> int:
    from .envinfo import format_text, write

    ap = argparse.ArgumentParser(prog="envinfo", description="記錄 Python、套件、RAM、磁碟與 GPU 資訊")
    ap.add_argument("--out", default="logs", help="輸出資料夾（寫 envinfo.json / envinfo.txt）")
    args = ap.parse_args(argv)
    info = write(args.out)
    print(format_text(info))
    return 0


if __name__ == "__main__":
    sys.exit(inspect_main())
