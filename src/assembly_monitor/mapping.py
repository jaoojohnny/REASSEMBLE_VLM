"""讀取並檢查 configs/field_mapping.yaml（第 2 包起所有程式都從這裡拿 HDF5 key）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class MappingError(ValueError):
    pass


@dataclass
class Mapping:
    path: str
    raw: dict[str, Any]
    seg_container: str
    seg_fields: dict[str, str | None]
    exclude_text: set[str]
    low_container: str | None
    low_fields: dict[str, str | None]
    action_regex: re.Pattern
    action_lowercase: bool
    target_action: str | None
    target_part: str | None
    cameras: dict[str, dict[str, Any]]
    best_candidates: list[str]
    start_buffer_sec: float = 0.5
    end_buffer_sec: float = 2.0
    confirmed_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def parse_text(self, text: str | None) -> tuple[str | None, str | None]:
        """依確認過的規則把描述拆成 (action, part)；對不上回傳 (None, None)。"""
        if not text:
            return None, None
        m = self.action_regex.match(text)
        if not m:
            return None, None
        action = m.groupdict().get("action")
        part = m.groupdict().get("part")
        if action and self.action_lowercase:
            action = action.lower()
        return action, (part.strip() if part else part)

    def is_excluded(self, text: str | None) -> bool:
        return (text or "").strip() in self.exclude_text


def load_mapping(path: str | Path, require_confirmed: bool = True) -> Mapping:
    p = Path(path)
    if not p.exists():
        raise MappingError(f"找不到欄位映射 {p}（第 1 包核對後要另存成 configs/field_mapping.yaml）")
    with p.open("r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    if require_confirmed and d.get("status") != "confirmed":
        raise MappingError(f"{p} 的 status 是 {d.get('status')!r}，必須是 confirmed 才能建索引")
    seg = d.get("segments") or {}
    fields = seg.get("fields") or {}
    for k in ("start", "end"):
        if not fields.get(k):
            raise MappingError(f"segments.fields.{k} 未設定")
    if not seg.get("container"):
        raise MappingError("segments.container 未設定")
    low = seg.get("low_level") or {}
    ap = d.get("action_parse") or {}
    try:
        rx = re.compile(ap.get("regex") or r"^\s*(?P<action>[A-Za-z]+)\s+(?P<part>.+?)\.?\s*$")
    except re.error as e:
        raise MappingError(f"action_parse.regex 無法編譯：{e}") from e
    cams = {k: v for k, v in (d.get("cameras") or {}).items() if isinstance(v, dict)}
    for name, spec in cams.items():
        if not spec.get("video") or not spec.get("timestamps"):
            raise MappingError(f"cameras.{name} 需要 video 與 timestamps")
    best = list(d.get("best_camera_candidates") or cams.keys())
    unknown = [b for b in best if b not in cams]
    if unknown:
        raise MappingError(f"best_camera_candidates 有未定義的相機：{unknown}")
    ext = d.get("extraction") or {}
    return Mapping(
        path=str(p),
        raw=d,
        seg_container=seg["container"],
        seg_fields={k: fields.get(k) for k in ("start", "end", "success", "text", "order")},
        exclude_text={str(t).strip() for t in (seg.get("exclude_text") or [])},
        low_container=low.get("container"),
        low_fields={k: (low.get("fields") or {}).get(k) for k in ("start", "end", "success", "text")},
        action_regex=rx,
        action_lowercase=bool(ap.get("action_lowercase", True)),
        target_action=ap.get("target_action"),
        target_part=ap.get("target_part"),
        cameras=cams,
        best_candidates=best,
        start_buffer_sec=float(ext.get("start_buffer_sec", 0.5)),
        end_buffer_sec=float(ext.get("end_buffer_sec", 2.0)),
        confirmed_at=str(d.get("confirmed_at")) if d.get("confirmed_at") else None,
    )
