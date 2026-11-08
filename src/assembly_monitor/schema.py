"""第 4 包標註格式（規格 P3-1、P3-2）：欄位、驗證、JSON Schema、CSV / JSONL 匯入匯出。

一筆標註＝一個標註者對一個 segment 的判斷：
  segment_id（"<錄製檔>__seg<編號>"）、dataset_result（資料集原始 success，唯讀）、result、failure_reason、
  evidence_start_sec / evidence_end_sec（相對標註起點的秒數，可為負，範圍是該段取影格的視窗）、best_camera、
  camera_visibility（各視角接觸區可見度）、visible_evidence、notes、annotator、annotator_type、label_version、
  reason_set、review_status、created_at、updated_at、source

規則：
- result ∈ success / failure / uncertain；failure 必須有 failure_reason，其他結果 failure_reason 必須是空的
- failure_reason ∈ configs/annotation.yaml 的類別（至少含 unknown）；選 other 必須寫 notes
- success / failure（人或 AI 標的）必須有證據區間、best_camera、visible_evidence；草稿（draft）可以先不完整
- 證據區間：起 ≤ 迄，且落在 [−start_buffer, 段長 + end_buffer]（允許 tolerance_sec 誤差）
- 匯入時欄位缺席＝保留原值，欄位存在但空白＝清除；dataset_result 與索引不符一律報錯，不會被覆蓋
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .util import now_iso

SCHEMA_ID = "reassemble-vlm/annotation-v1"
BASE_FIELDS = ["segment_id", "recording_id", "segment_index", "part", "dataset_result",
               "result", "failure_reason", "evidence_start_sec", "evidence_end_sec", "best_camera",
               "visual_difficulty", "visible_evidence", "notes"]
META_FIELDS = ["annotator", "annotator_type", "label_version", "reason_set", "review_status",
               "created_at", "updated_at", "source"]
TIME_FIELDS = ("evidence_start_sec", "evidence_end_sec")
TEXT_FIELDS = ("visible_evidence", "notes", "source")
RESULTS = ("success", "failure", "uncertain")
DISPUTE_FIELDS = ("result", "failure_reason", "best_camera", "evidence")
SEG_ID_RE = re.compile(r"^(?P<rid>.+)__seg(?P<idx>\d+)$")


class AnnotationError(ValueError):
    pass


# ---------------------------------------------------------------- 設定


@dataclass
class AnnotationConfig:
    path: str
    raw: dict[str, Any]
    label_version: str
    reason_set: str
    results: dict[str, dict]
    reasons: dict[str, dict]
    visibility: dict[str, dict]
    review_status: list[str]
    annotator_types: list[str]
    cameras: list[str]
    evidence_required_for: list[str]
    tolerance_sec: float
    start_buffer: float
    end_buffer: float
    task: dict[str, Any]
    dense: dict[str, Any]
    review: dict[str, Any]
    compare: dict[str, Any]
    reference: dict[str, Any]
    difficulty: dict[str, dict] = field(default_factory=dict)       # v2：畫面難度（clear / subtle / not_visible）
    difficulty_required: bool = False
    difficulty_default: str | None = None                           # gold 合併時無爭議段的預設值
    blocking_fields: list[str] = field(default_factory=lambda: list(DISPUTE_FIELDS))
    evidence_min_len: float = 0.0
    evidence_hint: str = ""

    def needs_difficulty(self, rec: dict[str, Any]) -> bool:
        """目前版本（label_version）的人工／AI 正式紀錄才要求畫面難度；舊版紀錄與參考標籤不要求。"""
        return (bool(self.difficulty) and self.difficulty_required and rec.get("label_version") == self.label_version
                and (rec.get("annotator_type") or "human") != "reference")

    def vis_columns(self) -> list[str]:
        return [f"vis_{c}" for c in self.cameras]

    def csv_fields(self) -> list[str]:
        return BASE_FIELDS + self.vis_columns() + META_FIELDS

    def public(self) -> dict[str, Any]:
        """給標註頁的設定（只含顯示與驗證需要的部分）。"""
        return {
            "label_version": self.label_version, "reason_set": self.reason_set,
            "results": self.results, "reasons": self.reasons, "visibility": self.visibility,
            "review_status": self.review_status, "cameras": self.cameras,
            "evidence_required_for": self.evidence_required_for, "tolerance_sec": self.tolerance_sec,
            "start_buffer": self.start_buffer, "end_buffer": self.end_buffer,
            "csv_fields": self.csv_fields(),
            "difficulty": self.difficulty, "difficulty_required": self.difficulty_required,
            "difficulty_default": self.difficulty_default,
            "evidence_min_len": self.evidence_min_len, "evidence_hint": self.evidence_hint,
        }


def load_annotation_config(path: str | Path, mapping: Any) -> AnnotationConfig:
    p = Path(path)
    if not p.exists():
        raise AnnotationError(f"找不到標註設定 {p}")
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    results = {k: dict(v or {}) for k, v in (d.get("results") or {}).items()}
    if set(results) != set(RESULTS):
        raise AnnotationError(f"results 必須正好是 {RESULTS}（目前 {sorted(results)}）")
    reasons = {k: dict(v or {}) for k, v in (d.get("failure_reasons") or {}).items()}
    if "unknown" not in reasons:
        raise AnnotationError("failure_reasons 至少要有 unknown（規格 P3-1）")
    for k in reasons:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", k):
            raise AnnotationError(f"failure_reasons 的代號只能用小寫英數與底線：{k!r}")
    vis = {k: dict(v or {}) for k, v in (d.get("visibility") or {"clear": {}, "partial": {}, "occluded": {}}).items()}
    ev = d.get("evidence") or {}
    req = list(ev.get("required_for") or ["success", "failure"])
    if not set(req) <= set(RESULTS):
        raise AnnotationError(f"evidence.required_for 只能是 {RESULTS}")
    vd = d.get("visual_difficulty") or {}
    levels = {k: dict(v or {}) for k, v in (vd.get("levels") or {}).items()}
    d_default = vd.get("default_for_approved")
    if d_default is not None and d_default not in levels:
        raise AnnotationError(f"visual_difficulty.default_for_approved={d_default!r} 不在 levels（{sorted(levels)}）")
    cmp = dict(d.get("compare") or {})
    blocking = list(cmp.get("blocking_fields") or DISPUTE_FIELDS)
    bad = [f for f in blocking if f not in DISPUTE_FIELDS]
    if bad:
        raise AnnotationError(f"compare.blocking_fields 只能是 {DISPUTE_FIELDS}（目前多了 {bad}）")
    return AnnotationConfig(
        path=str(p), raw=d,
        label_version=str(d.get("label_version") or "v1"),
        reason_set=str(d.get("reason_set") or "pilot"),
        results=results, reasons=reasons, visibility=vis,
        review_status=list(d.get("review_status") or ["draft", "submitted", "approved", "disputed", "adjudicated", "rejected"]),
        annotator_types=list(d.get("annotator_types") or ["human", "ai", "reference"]),
        cameras=list(mapping.best_candidates),
        evidence_required_for=req,
        tolerance_sec=float(ev.get("tolerance_sec", 0.05)),
        start_buffer=float(mapping.start_buffer_sec),
        end_buffer=float(mapping.end_buffer_sec),
        task=dict(d.get("task") or {}),
        dense=dict(d.get("dense_frames") or {}),
        review=dict(d.get("review_sheets") or {}),
        compare=cmp,
        reference=dict(d.get("reference") or {}),
        difficulty=levels,
        difficulty_required=bool(vd.get("required", False)) and bool(levels),
        difficulty_default=d_default,
        blocking_fields=blocking,
        evidence_min_len=float(ev.get("min_length_sec", 0.0) or 0.0),
        evidence_hint=str(ev.get("hint") or ""),
    )


# ---------------------------------------------------------------- segment


@dataclass(frozen=True)
class SegmentInfo:
    segment_id: str
    recording_id: str
    segment_index: int
    action: str
    part: str
    text: str
    start_unix: float
    end_unix: float
    dataset_result: str | None
    validation_status: str

    @property
    def duration(self) -> float:
        return self.end_unix - self.start_unix

    def window(self, start_buffer: float, end_buffer: float) -> tuple[float, float]:
        """相對標註起點的秒數視窗。"""
        return -start_buffer, self.duration + end_buffer


def segment_key(recording_id: str, index: int | str) -> str:
    return f"{recording_id}__seg{int(index)}"


def parse_segment_key(key: str) -> tuple[str, int] | None:
    m = SEG_ID_RE.match(str(key))
    return (m.group("rid"), int(m.group("idx"))) if m else None


def dataset_result_of(success: Any) -> str | None:
    if success in (True, "True", "true", "1", 1):
        return "success"
    if success in (False, "False", "false", "0", 0):
        return "failure"
    return None


def load_segment_infos(index_dir: str | Path, action: str, part: str,
                       statuses: tuple[str, ...] = ("ok", "warn")) -> dict[str, SegmentInfo]:
    p = Path(index_dir) / "segments.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}，請先執行 index-segments")
    out: dict[str, SegmentInfo] = {}
    with p.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("action") != action or r.get("part") != part or r.get("validation_status") not in statuses:
                continue
            key = segment_key(r["recording_id"], r["segment_id"])
            out[key] = SegmentInfo(
                segment_id=key, recording_id=r["recording_id"], segment_index=int(r["segment_id"]),
                action=action, part=part, text=r.get("text") or "",
                start_unix=float(r["start_unix"]), end_unix=float(r["end_unix"]),
                dataset_result=dataset_result_of(r.get("success")),
                validation_status=r.get("validation_status") or "",
            )
    return out


# ---------------------------------------------------------------- 紀錄與驗證


def new_record(seg: SegmentInfo, cfg: AnnotationConfig, annotator: str, annotator_type: str = "human",
               source: str = "") -> dict[str, Any]:
    ts = now_iso()
    return normalize_record({
        "segment_id": seg.segment_id, "recording_id": seg.recording_id, "segment_index": seg.segment_index,
        "part": seg.part, "dataset_result": seg.dataset_result, "result": seg.dataset_result,
        "annotator": annotator, "annotator_type": annotator_type, "label_version": cfg.label_version,
        "reason_set": cfg.reason_set, "review_status": "draft", "created_at": ts, "updated_at": ts,
        "source": source,
    }, cfg)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_record(rec: dict[str, Any], cfg: AnnotationConfig, seg: SegmentInfo | None = None,
                    complete: bool | None = None) -> list[str]:
    """回傳錯誤訊息清單（空＝通過）。complete=None 時，draft 只查型別與範圍，其他狀態還要查必填。"""
    errs: list[str] = []
    sid = rec.get("segment_id")
    if not isinstance(sid, str) or not SEG_ID_RE.match(sid):
        errs.append(f"segment_id 格式錯誤（要像 2025-01-09-14-28-47__seg11）：{sid!r}")
    annot = rec.get("annotator")
    if not isinstance(annot, str) or not annot.strip():
        errs.append("annotator 不可空白")
    atype = rec.get("annotator_type") or "human"
    if atype not in cfg.annotator_types:
        errs.append(f"annotator_type 必須是 {'/'.join(cfg.annotator_types)}（目前 {atype!r}）")
    status = rec.get("review_status")
    if status not in cfg.review_status:
        errs.append(f"review_status 必須是 {'/'.join(cfg.review_status)}（目前 {status!r}）")
    if not rec.get("label_version"):
        errs.append("label_version 不可空白")
    result = rec.get("result")
    if result not in cfg.results:
        errs.append(f"result 必須是 {'/'.join(cfg.results)}（目前 {result!r}）")
    reason = rec.get("failure_reason")
    if reason is not None and reason not in cfg.reasons:
        errs.append(f"failure_reason {reason!r} 不在類別清單（{'/'.join(cfg.reasons)}）")
    if result != "failure" and reason is not None and result in cfg.results:
        errs.append("failure_reason 只能在 result=failure 時填")
    cam = rec.get("best_camera")
    if cam is not None and cam not in cfg.cameras:
        errs.append(f"best_camera 必須是 {'/'.join(cfg.cameras)}（目前 {cam!r}）")
    vis = rec.get("camera_visibility") or {}
    if not isinstance(vis, dict):
        errs.append("camera_visibility 必須是物件")
    else:
        for c, v in vis.items():
            if c not in cfg.cameras:
                errs.append(f"camera_visibility 有未定義的相機 {c!r}")
            elif v is not None and v not in cfg.visibility:
                errs.append(f"camera_visibility.{c} 必須是 {'/'.join(cfg.visibility)}（目前 {v!r}）")
    diff = rec.get("visual_difficulty")
    if diff is not None and diff not in cfg.difficulty:
        allowed = "/".join(cfg.difficulty) or "（此版本沒有畫面難度欄位）"
        errs.append(f"visual_difficulty 必須是 {allowed}（目前 {diff!r}）")
    a, b = rec.get("evidence_start_sec"), rec.get("evidence_end_sec")
    for name, v in (("evidence_start_sec", a), ("evidence_end_sec", b)):
        if v is not None and not _is_number(v):
            errs.append(f"{name} 必須是數字（目前 {v!r}）")
    times_ok = _is_number(a) and _is_number(b)
    if (a is None) != (b is None):
        errs.append("證據區間的起點與終點要一起填")
    if times_ok and a > b:
        errs.append(f"證據區間起點 {a} 大於終點 {b}")
    if seg is not None:
        if sid != seg.segment_id:
            errs.append(f"segment_id 與索引不符：{sid!r} ≠ {seg.segment_id!r}")
        ds = rec.get("dataset_result")
        if ds is not None and ds != seg.dataset_result:
            errs.append(f"dataset_result 與索引不符（{ds!r} ≠ {seg.dataset_result!r}；原始標籤不可修改）")
        w0, w1 = seg.window(cfg.start_buffer, cfg.end_buffer)
        tol = cfg.tolerance_sec
        for name, v in (("evidence_start_sec", a), ("evidence_end_sec", b)):
            if _is_number(v) and not (w0 - tol <= v <= w1 + tol):
                errs.append(f"{name}={v} 超出該段視窗 [{w0:.2f}, {w1:.2f}] 秒")
    elif rec.get("dataset_result") not in (None, "success", "failure"):
        errs.append(f"dataset_result 必須是 success/failure（目前 {rec.get('dataset_result')!r}）")
    if complete is None:
        complete = status not in ("draft", None)
    if complete and result in cfg.results:
        if result == "failure" and reason is None:
            errs.append("result=failure 必須選 failure_reason")
        if reason is not None and cfg.reasons.get(reason, {}).get("requires_notes") and not (rec.get("notes") or "").strip():
            errs.append(f"failure_reason={reason} 必須在 notes 寫出原因")
        if result in cfg.evidence_required_for and atype != "reference":
            if a is None or b is None:
                errs.append(f"result={result} 必須填證據區間")
            if cam is None:
                errs.append(f"result={result} 必須選 best_camera")
            if not (rec.get("visible_evidence") or "").strip():
                errs.append(f"result={result} 必須寫 visible_evidence（畫面上看得到的依據）")
            if times_ok and cfg.evidence_min_len and b - a < cfg.evidence_min_len and rec.get("label_version") == cfg.label_version:
                errs.append(f"證據區間太短（{b - a:.2f} 秒 < {cfg.evidence_min_len} 秒），起點和終點要分別設定")
        if cfg.needs_difficulty(rec) and diff is None:
            errs.append(f"label_version={cfg.label_version} 必須選 visual_difficulty（{'/'.join(cfg.difficulty)}）")
    return errs


def validate_records(recs: Iterable[dict[str, Any]], cfg: AnnotationConfig,
                     segs: dict[str, SegmentInfo] | None = None, complete: bool | None = None) -> dict[str, Any]:
    """整批驗證：每筆錯誤＋重複（同一 segment_id 與 annotator 出現兩次）＋不在索引的段。"""
    problems: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], int] = {}
    n = 0
    for i, r in enumerate(recs):
        n += 1
        key = (str(r.get("segment_id")), str(r.get("annotator")))
        if key in seen:
            problems.append({"row": i + 1, "segment_id": key[0], "errors": [f"重複：與第 {seen[key]} 筆同一段、同一標註者"]})
        seen.setdefault(key, i + 1)
        seg = None
        if segs is not None:
            seg = segs.get(str(r.get("segment_id")))
            if seg is None:
                problems.append({"row": i + 1, "segment_id": key[0], "errors": ["找不到這個 segment_id（不在索引的目標段中）"]})
                continue
        errs = validate_record(r, cfg, seg, complete)
        if errs:
            problems.append({"row": i + 1, "segment_id": key[0], "errors": errs})
    return {"n_records": n, "n_problem_rows": len({p["row"] for p in problems}), "problems": problems, "ok": not problems}


def json_schema(cfg: AnnotationConfig) -> dict[str, Any]:
    """JSON Schema（draft 2020-12）。跨欄位的「起 ≤ 迄」與「在該段視窗內」需要段長，由 validate_record 負責。"""
    reasons = list(cfg.reasons)
    cams = list(cfg.cameras)
    lo = -(cfg.start_buffer + cfg.tolerance_sec)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "REASSEMBLE × VLM 標註紀錄",
        "description": ("evidence_*_sec 為相對標註起點的秒數。evidence_end_sec ≥ evidence_start_sec 與"
                        "「落在 [−start_buffer, 段長 + end_buffer]」需要段長，由 assembly_monitor.schema.validate_record 檢查。"),
        "type": "object",
        "required": ["segment_id", "annotator", "result", "label_version", "review_status"],
        "additionalProperties": True,
        "properties": {
            "segment_id": {"type": "string", "pattern": SEG_ID_RE.pattern},
            "recording_id": {"type": ["string", "null"]},
            "segment_index": {"type": ["integer", "null"], "minimum": 0},
            "part": {"type": ["string", "null"]},
            "dataset_result": {"enum": ["success", "failure", None]},
            "result": {"enum": list(cfg.results)},
            "failure_reason": {"enum": reasons + [None]},
            "evidence_start_sec": {"type": ["number", "null"], "minimum": lo},
            "evidence_end_sec": {"type": ["number", "null"], "minimum": lo},
            "best_camera": {"enum": cams + [None]},
            "visual_difficulty": {"enum": list(cfg.difficulty) + [None]},
            "camera_visibility": {"type": "object", "additionalProperties": False,
                                  "properties": {c: {"enum": list(cfg.visibility) + [None]} for c in cams}},
            "visible_evidence": {"type": ["string", "null"]},
            "notes": {"type": ["string", "null"]},
            "annotator": {"type": "string", "minLength": 1},
            "annotator_type": {"enum": list(cfg.annotator_types)},
            "label_version": {"type": "string", "minLength": 1},
            "reason_set": {"type": ["string", "null"]},
            "review_status": {"enum": list(cfg.review_status)},
            "created_at": {"type": ["string", "null"]},
            "updated_at": {"type": ["string", "null"]},
            "source": {"type": ["string", "null"]},
        },
        "allOf": [
            {"if": {"properties": {"result": {"const": "failure"}, "review_status": {"not": {"const": "draft"}}},
                    "required": ["result", "review_status"]},
             "then": {"required": ["failure_reason"], "properties": {"failure_reason": {"enum": reasons}}}},
            {"if": {"properties": {"result": {"enum": [k for k in cfg.results if k != "failure"]}}, "required": ["result"]},
             "then": {"properties": {"failure_reason": {"const": None}}}},
            {"if": {"properties": {"result": {"enum": list(cfg.evidence_required_for)},
                                   "annotator_type": {"enum": ["human", "ai"]},
                                   "review_status": {"not": {"const": "draft"}}},
                    "required": ["result", "review_status"]},
             "then": {"required": ["evidence_start_sec", "evidence_end_sec", "best_camera", "visible_evidence"],
                      "properties": {"evidence_start_sec": {"type": "number"}, "evidence_end_sec": {"type": "number"},
                                     "best_camera": {"enum": cams}, "visible_evidence": {"type": "string", "minLength": 1}}}},
            {"if": {"properties": {"failure_reason": {"enum": [k for k, v in cfg.reasons.items() if v.get("requires_notes")] or ["__none__"]},
                                   "review_status": {"not": {"const": "draft"}}},
                    "required": ["failure_reason", "review_status"]},
             "then": {"required": ["notes"], "properties": {"notes": {"type": "string", "minLength": 1}}}},
        ] + ([
            {"if": {"properties": {"label_version": {"const": cfg.label_version}, "annotator_type": {"enum": ["human", "ai"]},
                                   "review_status": {"not": {"const": "draft"}}},
                    "required": ["label_version", "review_status"]},
             "then": {"required": ["visual_difficulty"], "properties": {"visual_difficulty": {"enum": list(cfg.difficulty)}}}},
        ] if cfg.difficulty_required else []),
    }


# ---------------------------------------------------------------- 正規化與匯入匯出


def _num(v: Any) -> Any:
    """CSV 字串轉數字；空白 → None；轉不了就原樣保留（交給驗證報錯）。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s.replace(",", ".")) if s.count(",") == 1 and "." not in s else float(s)
    except ValueError:
        return s


def _round_time(v: Any) -> Any:
    return round(float(v), 3) if _is_number(v) else v


def normalize_record(rec: dict[str, Any], cfg: AnnotationConfig) -> dict[str, Any]:
    """補齊欄位、統一型別與順序；不認得的欄位原樣保留在最後。"""
    out: dict[str, Any] = {}
    for k in BASE_FIELDS:
        v = rec.get(k)
        if k in TIME_FIELDS:
            v = _round_time(_num(v))
        elif k == "segment_index":
            n = _num(v)
            v = int(n) if _is_number(n) and float(n).is_integer() else n
        elif k in TEXT_FIELDS:
            v = "" if v is None else str(v)
        elif isinstance(v, str) and v.strip() == "":
            v = None
        out[k] = v
    vis_in = rec.get("camera_visibility") or {}
    vis = {}
    for c in cfg.cameras:
        v = vis_in.get(c) if isinstance(vis_in, dict) else None
        vis[c] = None if (v is None or (isinstance(v, str) and not v.strip())) else v
    if isinstance(vis_in, dict):
        for c, v in vis_in.items():
            if c not in vis:
                vis[c] = v  # 保留未定義相機，交給驗證報錯
    out["camera_visibility"] = vis
    for k in META_FIELDS:
        v = rec.get(k)
        if k in TEXT_FIELDS:
            v = "" if v is None else str(v)
        elif isinstance(v, str) and v.strip() == "":
            v = None
        out[k] = v
    if out.get("annotator_type") is None:
        out["annotator_type"] = "human"
    known = set(BASE_FIELDS) | set(META_FIELDS) | {"camera_visibility"} | set(cfg.vis_columns())
    for k, v in rec.items():
        if k not in known:
            out[k] = v
    return out


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp950"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise AnnotationError(f"{path} 不是 UTF-8 或 Big5 編碼，請用 Excel「CSV UTF-8」另存")


def read_annotations(path: str | Path, cfg: AnnotationConfig) -> list[dict[str, Any]]:
    """讀 .jsonl / .json / .csv。只回傳檔案裡「有出現」的欄位（合併時缺席欄位＝保留原值）。"""
    p = Path(path)
    text = _read_text(p)
    recs: list[dict[str, Any]] = []
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        for ln, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise AnnotationError(f"{p.name} 第 {ln} 行不是合法 JSON：{e}") from e
            if not isinstance(d, dict):
                raise AnnotationError(f"{p.name} 第 {ln} 行必須是物件")
            recs.append(d)
        return recs
    if p.suffix.lower() == ".json":
        d = json.loads(text)
        items = d.get("records", d) if isinstance(d, dict) else d
        if not isinstance(items, list):
            raise AnnotationError(f"{p.name} 需要是陣列或含 records 的物件")
        return [dict(x) for x in items]
    if p.suffix.lower() != ".csv":
        raise AnnotationError(f"不支援的格式：{p.suffix}（請用 .jsonl 或 .csv）")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    cols = reader.fieldnames or []
    vis_cols = {c: c[4:] for c in cols if c.startswith("vis_")}
    for row in reader:
        d: dict[str, Any] = {}
        vis: dict[str, Any] = {}
        for c in cols:
            v = row.get(c)
            if c in vis_cols:
                vis[vis_cols[c]] = v if (v or "").strip() else None
            else:
                d[c] = v
        if vis_cols:
            d["camera_visibility"] = vis
        for k in TIME_FIELDS:
            if k in d:
                d[k] = _num(d[k])
        if "segment_index" in d:
            n = _num(d["segment_index"])
            d["segment_index"] = int(n) if _is_number(n) and float(n).is_integer() else n
        known = set(BASE_FIELDS) | set(META_FIELDS)
        for k in list(d):
            if k not in known and k != "camera_visibility" and (d[k] is None or str(d[k]).strip() == ""):
                del d[k]  # 自訂欄位空白＝沒有這個欄位（CSV 每列都有同樣的欄）
            elif k not in TEXT_FIELDS and isinstance(d[k], str) and d[k].strip() == "":
                d[k] = None
        recs.append(d)
    return recs


def write_jsonl(recs: Iterable[dict[str, Any]], path: str | Path, cfg: AnnotationConfig) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(normalize_record(r, cfg), ensure_ascii=False) + "\n")
    tmp.replace(p)
    return p


def write_csv(recs: Iterable[dict[str, Any]], path: str | Path, cfg: AnnotationConfig) -> Path:
    """UTF-8（含 BOM，Excel 直接開不會亂碼）。camera_visibility 展開成 vis_<相機> 欄。"""
    rows = [normalize_record(r, cfg) for r in recs]
    fields = cfg.csv_fields()
    extra: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields and k != "camera_visibility" and k not in extra:
                extra.append(k)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + extra)
        w.writeheader()
        for r in rows:
            row = {k: ("" if r.get(k) is None else r.get(k)) for k in BASE_FIELDS + META_FIELDS + extra}
            for c in cfg.cameras:
                v = (r.get("camera_visibility") or {}).get(c)
                row[f"vis_{c}"] = "" if v is None else v
            for k in extra:
                v = r.get(k)
                row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else ("" if v is None else v)
            w.writerow(row)
    tmp.replace(p)
    return p


def merge_annotations(base: list[dict[str, Any]], incoming: list[dict[str, Any]], cfg: AnnotationConfig,
                      segs: dict[str, SegmentInfo] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """以 (segment_id, annotator) 合併。incoming 缺席的欄位保留 base 的值；dataset_result 不可被改。"""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for r in base:
        k = (str(r.get("segment_id")), str(r.get("annotator")))
        if k not in merged:
            order.append(k)
        merged[k] = normalize_record(r, cfg)
    report: dict[str, Any] = {"added": 0, "updated": 0, "conflicts": []}
    for r in incoming:
        k = (str(r.get("segment_id")), str(r.get("annotator")))
        cur = merged.get(k)
        if cur is None:
            new = normalize_record(r, cfg)
            seg = segs.get(k[0]) if segs else None
            if seg is not None and new.get("dataset_result") is None:
                new["dataset_result"] = seg.dataset_result
            merged[k] = new
            order.append(k)
            report["added"] += 1
            continue
        upd = dict(cur)
        for f, v in r.items():
            if f == "dataset_result" and v not in (None, "") and cur.get("dataset_result") not in (None, v):
                report["conflicts"].append({"segment_id": k[0], "annotator": k[1], "field": f,
                                            "base": cur.get("dataset_result"), "incoming": v})
                continue
            if f == "camera_visibility" and isinstance(v, dict):
                vis = dict(cur.get("camera_visibility") or {})
                vis.update(v)
                upd["camera_visibility"] = vis
            else:
                upd[f] = v
        merged[k] = normalize_record(upd, cfg)
        report["updated"] += 1
    return [merged[k] for k in order], report
