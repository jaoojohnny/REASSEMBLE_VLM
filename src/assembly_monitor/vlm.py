"""第 5 包：現成 VLM 推論器與結構化輸出。

元件（對應規格第 8 頁）
- ModelAdapter：load(config)、predict(prompt)、metadata()；提供者可替換（transformers、majority、mock）。
- PromptBuilder：指定零件與完成條件；每張影格標視角與秒數；明示看不清可以答 uncertain。
- Parser：保留 raw_text；解析 JSON；檢查 enum、欄位與 evidence 時間範圍。不替模型補答案。
- Runner：依切分（預設只讀 dev）逐段推論；可續跑；每段各自捕捉 OOM、timeout 與格式錯。
- Output：raw.jsonl、predictions.jsonl、errors.csv（含延遲與 VRAM），以及跨模型的 summary.csv／summary.md。

重型套件（torch、transformers）只在 TransformersAdapter.load() 才匯入；其餘部分在沒有 GPU 的環境也能測試。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from . import __version__
from .util import now_iso

log = logging.getLogger(__name__)

RESULTS = ("success", "failure", "uncertain")
JSON_ERRORS = {"json_not_found", "json_decode_error", "not_an_object"}  # 這些才做格式修正
TOOL = "vlm-runner-v1"
API_TYPES = ("anthropic", "openai")             # 第 8 包：商業模型（API）
MODEL_TYPES = ("transformers", "majority", "mock", "encoder", *API_TYPES)
SKILLS = ("zoom", "evidence_review")


class VLMConfigError(ValueError):
    pass


class RunConfigMismatch(RuntimeError):
    pass


# ---------------------------------------------------------------- 設定


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise VLMConfigError(f"找不到設定檔 {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_model_configs(path: str | Path | list[str | Path] | tuple[str | Path, ...]) -> dict[str, dict[str, Any]]:
    """讀模型設定。可以給多個檔案（例如 model.yaml＋model_train.yaml）合併；同一個 key 出現兩次就報錯，
    這樣新增的模型可以放在新檔案，不必改已凍結的 model.yaml。"""
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    d: dict[str, dict[str, Any]] = {}
    for p in paths:
        one = load_yaml(p).get("models") or {}
        if not one:
            raise VLMConfigError(f"{p} 沒有 models")
        dup = sorted(set(one) & set(d))
        if dup:
            raise VLMConfigError(f"{p} 的模型 key 和前面的設定檔重複：{dup}")
        d.update(one)
    for k, v in d.items():
        if v.get("type") not in MODEL_TYPES:
            raise VLMConfigError(f"模型 {k} 的 type 必須是 {' / '.join(MODEL_TYPES)}")
        if v["type"] in ("transformers", "encoder") and not v.get("hf_id"):
            raise VLMConfigError(f"模型 {k} 缺 hf_id")
        if v["type"] in API_TYPES and not v.get("model_id"):
            raise VLMConfigError(f"模型 {k}（{v['type']} API）缺 model_id")
        if v.get("skill") and (v["type"] != "transformers" or v["skill"] not in SKILLS):
            raise VLMConfigError(f"模型 {k}：skill 只能是 {SKILLS}，而且只用在 transformers 模型")
        if v.get("skill") == "zoom":
            _check_zoom_cfg(k, v)
        if v.get("reason") is not None:  # 第 11 包：推理方式（機率門檻、前後對照、再檢查一次、看範例）
            from .reason import check_reason_cfg

            check_reason_cfg(k, v)
        if v.get("adapter_path") and v["type"] != "transformers":
            raise VLMConfigError(f"模型 {k}：只有 transformers 模型可以加 adapter_path")
    return d


def _check_zoom_cfg(k: str, v: dict[str, Any]) -> None:
    """放大鏡 Skill 的設定（第 8 包第二版加 zoom_mode、zoom_cameras、zoom_fixed_box）。只檢查、不補預設值，
    避免改到已跑過的設定內容（run_config 的雜湊包含模型設定）。"""
    mode = v.get("zoom_mode", "locate")
    if mode not in ("locate", "fixed"):
        raise VLMConfigError(f"模型 {k}：zoom_mode 只能是 locate（模型先框位置）或 fixed（固定位置裁切）")
    cams = v.get("zoom_cameras")
    if cams is not None and (not isinstance(cams, list) or not cams or not all(isinstance(c, str) for c in cams)):
        raise VLMConfigError(f"模型 {k}：zoom_cameras 要是相機名稱的清單，例如 [hand]")
    if mode == "fixed":
        b = v.get("zoom_fixed_box")
        ok = isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b)
        if not ok or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            raise VLMConfigError(f"模型 {k}：zoom_mode: fixed 需要 zoom_fixed_box: [x1, y1, x2, y2]（0–1 的相對座標）")


def adapter_sha256(adapter_path: str | Path) -> str:
    """LoRA adapter 的內容 hash（adapter_config.json＋權重檔）。重新訓練後 hash 會變，舊的推論結果不會被沿用。"""
    d = Path(adapter_path)
    files = [d / "adapter_config.json"] + sorted(d.glob("adapter_model.*"))
    missing = [str(f) for f in files[:1] if not f.exists()]
    if missing or len(files) < 2:
        raise VLMConfigError(f"找不到 LoRA adapter：{d}（先執行 run_finetune.bat 訓練）")
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def resolve_adapter(cfg: dict[str, Any]) -> dict[str, Any]:
    """有 adapter_path 的模型：把 adapter 內容的 hash 放進設定（會寫進 run_config，影響「是否已跑完」的判斷）。"""
    if not cfg.get("adapter_path"):
        return cfg
    return {**cfg, "adapter_sha256": adapter_sha256(cfg["adapter_path"])}


def load_experiment(path: str | Path) -> dict[str, Any]:
    d = load_yaml(path).get("experiment") or {}
    need = ["name", "split_dir", "frames_root", "sampling_template", "cameras", "out_dir", "prompt", "task"]
    miss = [k for k in need if k not in d]
    if miss:
        raise VLMConfigError(f"{path} 缺少：{miss}")
    d.setdefault("include", ["dev"])
    d.setdefault("frames", [1, 4, 8])
    d.setdefault("timeout_sec", 180)
    d.setdefault("repair_retry", True)
    d.setdefault("evidence_tolerance_sec", 0.05)
    return d


def check_task_against_annotation(task: dict[str, Any], reason_keys: Iterable[str], cameras: Iterable[str]) -> None:
    """prompt 的失敗原因與相機必須和標註設定一致，避免模型輸出 gold 裡沒有的類別。"""
    rk, ck = list(task.get("failure_reasons") or {}), list(task.get("cameras") or {})
    if set(rk) != set(reason_keys):
        raise VLMConfigError(f"experiment.task.failure_reasons {sorted(rk)} 和標註設定 {sorted(reason_keys)} 不一致")
    if set(ck) != set(cameras):
        raise VLMConfigError(f"experiment.task.cameras {sorted(ck)} 和實驗相機 {sorted(cameras)} 不一致")


# ---------------------------------------------------------------- 影格


@dataclass(frozen=True)
class Frame:
    camera: str
    time_sec: float  # 相對標註起點
    path: str        # 原始影格
    k: int           # 取樣序號


@dataclass
class SegmentInput:
    segment_id: str
    duration_s: float
    window: tuple[float, float]  # 相對秒數
    frames: list[Frame]
    sampling_key: str
    n_missing: int = 0


def load_segment_frames(frames_root: str | Path, sampling_key: str, segment_id: str,
                        cameras: list[str]) -> SegmentInput | None:
    """讀第 3 包的 meta.json；沒有這段的影格就回傳 None。影格依（時間序號、相機順序）排列。"""
    d = Path(frames_root) / sampling_key / "segments" / segment_id
    meta_p = d / "meta.json"
    if not meta_p.exists():
        return None
    m = json.loads(meta_p.read_text(encoding="utf-8"))
    frames: list[Frame] = []
    missing = 0
    by_cam = {c["camera"]: c for c in m.get("cameras", [])}
    for ci, cam in enumerate(cameras):
        c = by_cam.get(cam)
        if c is None:
            missing += 1
            continue
        for k, (p, t) in enumerate(zip(c.get("frame_paths") or [], c.get("source_times_sec") or [])):
            if not p or t is None:
                missing += 1
                continue
            frames.append(Frame(camera=cam, time_sec=float(t), path=str(d / p), k=k))
    order = {c: i for i, c in enumerate(cameras)}
    frames.sort(key=lambda f: (f.k, order[f.camera]))
    w = m.get("window_unix") or [m["start_unix"], m["end_unix"]]
    return SegmentInput(segment_id=segment_id, duration_s=float(m.get("duration_s", m["end_unix"] - m["start_unix"])),
                        window=(w[0] - m["start_unix"], w[1] - m["start_unix"]), frames=frames,
                        sampling_key=sampling_key, n_missing=missing)


def prepare_images(frames: list[Frame], max_side: int | None, out_dir: Path, quality: int = 95) -> list[str]:
    """把影格縮到長邊 max_side（不放大）另存，控制每張圖的 token 數；回傳新路徑。"""
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        img = cv2.imdecode(np.fromfile(f.path, dtype=np.uint8), cv2.IMREAD_COLOR) if os.path.exists(f.path) else None
        if img is None:
            raise FileNotFoundError(f"讀不到影格 {f.path}")
        h, w = img.shape[:2]
        if max_side and max(h, w) > max_side:
            s = max_side / float(max(h, w))
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        p = out_dir / f"{i:02d}_{f.camera}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise OSError(f"影格編碼失敗 {f.path}")
        buf.tofile(str(p))  # 用 tofile 才能寫入含非 ASCII 字元的 Windows 路徑
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------- prompt


@dataclass
class Prompt:
    text: str                    # 開頭的完整指示（含輸出格式）
    content: list[dict[str, Any]]  # chat content：text 與 image 交錯；image 用 {"type":"image","path":...}
    version: str
    sha256: str
    t_range: tuple[float, float]
    images: list[dict[str, Any]] = field(default_factory=list)  # 每張圖的相機、秒數、原始影格
    task: dict[str, Any] = field(default_factory=dict)          # 任務文字（零件名稱等；Skill 會用到）


class PromptBuilder:
    def __init__(self, template: str, version: str, task: dict[str, Any]):
        self.template = template
        self.version = version
        self.task = task
        payload = json.dumps({"template": template, "task": task, "version": version}, ensure_ascii=False, sort_keys=True)
        self.sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_experiment(cls, exp: dict[str, Any], base_dir: str | Path = ".") -> "PromptBuilder":
        pc = exp["prompt"]
        tpl_path = Path(base_dir) / pc["template"]
        if not tpl_path.exists():
            raise VLMConfigError(f"找不到 prompt 範本 {tpl_path}")
        return cls(tpl_path.read_text(encoding="utf-8"), str(pc["version"]), exp["task"])

    def instruction(self, t_min: float, t_max: float, n_images: int) -> str:
        t = self.task
        reasons = t["failure_reasons"]
        cams = t["cameras"]
        return self.template.format(
            part=t["part"], part_note=t.get("part_note", ""),
            success_definition=t["success_definition"], failure_definition=t["failure_definition"],
            camera_lines="\n".join(f"- {k}: {v}" for k, v in cams.items()),
            reason_lines="\n".join(f"- {k}: {v}" for k, v in reasons.items()),
            reason_keys=", ".join(f'"{k}"' for k in reasons), camera_keys=", ".join(f'"{k}"' for k in cams),
            t_min=t_min, t_max=t_max, n_images=n_images,
        ).strip()

    def build(self, seg: SegmentInput, image_paths: list[str] | None = None) -> Prompt:
        if not seg.frames:
            raise ValueError(f"{seg.segment_id} 沒有可用影格")
        paths = image_paths or [f.path for f in seg.frames]
        t_min = min(f.time_sec for f in seg.frames)
        t_max = max(f.time_sec for f in seg.frames)
        text = self.instruction(t_min, t_max, len(seg.frames))
        content: list[dict[str, Any]] = [{"type": "text", "text": text + "\n"}]
        images = []
        for i, (f, p) in enumerate(zip(seg.frames, paths), 1):
            label = f"Image {i}: camera {f.camera}, t = {f.time_sec:+.2f} s"
            content.append({"type": "text", "text": f"\n{label}\n"})
            content.append({"type": "image", "path": p})
            images.append({"i": i, "camera": f.camera, "time_sec": round(f.time_sec, 3), "source": f.path, "input": p})
        content.append({"type": "text", "text": "\nNow answer with the JSON object only."})
        return Prompt(text=text, content=content, version=self.version, sha256=self.sha256, t_range=(t_min, t_max),
                      images=images, task=dict(self.task))


REPAIR_INSTRUCTION = (
    "The text below was supposed to be a single JSON object with the keys result, failure_reason, "
    "evidence_start_sec, evidence_end_sec, best_camera, visible_evidence and confidence. "
    "Rewrite it as one valid JSON object only. Do not add, guess or change any information; "
    "use null for anything the text does not state.\n\nTEXT:\n"
)


# ---------------------------------------------------------------- parser


def extract_json_text(raw: str) -> str | None:
    """從回答中取出第一個完整的 JSON 物件（容許 ```json 圍欄與前後文字）。"""
    if not raw:
        return None
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S | re.I)
    if m:
        s = m.group(1)
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        start = s.find("{", start + 1)
    return None


def _balanced_objects(s: str) -> list[str]:
    """所有最外層、括號平衡的 {...} 片段（依出現順序）。"""
    out, depth, in_str, esc, start = [], 0, False, False, -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                out.append(s[start : i + 1])
    return out


def extract_answer_json(raw: str) -> str | None:
    """推理模式會先寫一段推理再給 JSON：取最後一個能解析、而且有 result 欄位的物件；都沒有就退回第一個完整物件。"""
    if not raw:
        return None
    for frag in reversed(_balanced_objects(raw)):
        try:
            obj = json.loads(frag)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "result" in obj:
            return frag
    return extract_json_text(raw)


THOUGHT_PATTERNS = (re.compile(r"<think>.*?</think>", re.S),                 # Qwen3.5
                    re.compile(r"<\|channel>thought.*?<channel\|>", re.S))   # Gemma 4


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _enum(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def parse_prediction(raw: str, *, segment_id: str, reasons: Iterable[str], cameras: Iterable[str],
                     t_range: tuple[float, float] | None, tolerance: float = 0.05) -> dict[str, Any]:
    """解析模型回答。只做格式正規化（大小寫、空白）；不合法的值設為 null 並記在 warnings，不會補答案。
    result 缺失或不合法 → status=invalid。"""
    reasons, cameras = set(reasons), set(cameras)
    out: dict[str, Any] = {"segment_id": segment_id, "status": "invalid", "errors": [], "warnings": [],
                           "result": None, "failure_reason": None, "evidence_start_sec": None, "evidence_end_sec": None,
                           "best_camera": None, "visible_evidence": None, "confidence": None}
    text = raw or ""
    for pat in THOUGHT_PATTERNS:  # 有思考模式的模型：去掉思考內容再找 JSON
        text = pat.sub("", text)
    if ("<think>" in text and "</think>" not in text) or ("<|channel>thought" in text and "<channel|>" not in text):
        out["errors"].append("unfinished_thinking")  # 思考還沒結束就被截斷，沒有答案
        return out
    js = extract_answer_json(text)
    if js is None:
        # 特殊 token 被拿掉後，Gemma 4 的思考只剩開頭的 "thought"；沒有任何 JSON 就是思考被截斷
        out["errors"].append("unfinished_thinking" if text.lstrip().startswith("thought") else "json_not_found")
        return out
    try:
        obj = json.loads(js)
    except json.JSONDecodeError:
        out["errors"].append("json_decode_error")
        return out
    if not isinstance(obj, dict):
        out["errors"].append("not_an_object")
        return out
    res = _enum(obj.get("result"))
    if res not in RESULTS:
        out["errors"].append("bad_result")
        return out
    out["result"] = res
    fr = _enum(obj.get("failure_reason"))
    if fr in ("", "null", "none"):
        fr = None
    if res == "failure":
        if fr is None:
            out["warnings"].append("missing_failure_reason")
        elif fr not in reasons:
            out["warnings"].append("bad_failure_reason")
        else:
            out["failure_reason"] = fr
    elif fr is not None:
        out["warnings"].append("reason_given_for_non_failure")
    s0, s1 = obj.get("evidence_start_sec"), obj.get("evidence_end_sec")
    e0, e1 = _num(s0), _num(s1)
    if (s0 is not None and e0 is None) or (s1 is not None and e1 is None):
        out["warnings"].append("evidence_not_number")
    elif e0 is not None and e1 is not None:
        lo, hi = (t_range or (float("-inf"), float("inf")))
        if e0 > e1:
            out["warnings"].append("evidence_start_after_end")
        elif e0 < lo - tolerance or e1 > hi + tolerance:
            out["warnings"].append("evidence_out_of_range")
        else:
            out["evidence_start_sec"], out["evidence_end_sec"] = e0, e1
    elif (e0 is None) != (e1 is None):
        out["warnings"].append("evidence_half_missing")
    cam = _enum(obj.get("best_camera"))
    if cam in ("", "null", "none"):
        cam = None
    if cam is not None:
        if cam in cameras:
            out["best_camera"] = cam
        else:
            out["warnings"].append("bad_best_camera")
    ve = obj.get("visible_evidence")
    if ve is not None:
        out["visible_evidence"] = ve if isinstance(ve, str) else json.dumps(ve, ensure_ascii=False)
    c = obj.get("confidence")
    cv = _num(c)
    if c is not None:
        if cv is None or not (0.0 <= cv <= 1.0):
            out["warnings"].append("bad_confidence")
        else:
            out["confidence"] = cv
    out["status"] = "ok"
    return out


# ---------------------------------------------------------------- adapters


@dataclass
class RawOutput:
    text: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    peak_vram_mb: float | None = None
    stopped_by_time: bool = False
    extra: dict[str, Any] | None = None  # 第 8 包：API 費用、Skill 的中間結果等，原樣寫進 raw.jsonl


class FatalAdapterError(RuntimeError):
    """整個模型都不能用的錯（例如 API key 錯、沒有權限）：不再逐段重試，直接跳到下一個模型。"""


class ModelAdapter:
    """介面：load(config)、predict(prompt)、repair(raw_text)、metadata()、recover()、close()。"""

    uses_frames = True

    def __init__(self, key: str, cfg: dict[str, Any]):
        self.key, self.cfg = key, dict(cfg)

    def load(self) -> None:  # pragma: no cover - 介面
        pass

    def predict(self, prompt: Prompt, max_time: float | None = None) -> RawOutput:  # pragma: no cover
        raise NotImplementedError

    def repair(self, raw_text: str, max_time: float | None = None) -> RawOutput:  # pragma: no cover
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"key": self.key, "type": self.cfg.get("type"), "config": self.cfg}

    def recover(self) -> None:
        pass

    def close(self) -> None:
        pass


class MajorityAdapter(ModelAdapter):
    """多數類基準：不看影格，一律回答設定的類別（預設 success）。"""

    uses_frames = False

    def predict(self, prompt: Prompt | None = None, max_time: float | None = None) -> RawOutput:
        ans = {"result": self.cfg.get("result", "success"), "failure_reason": None, "evidence_start_sec": None,
               "evidence_end_sec": None, "best_camera": None, "visible_evidence": "", "confidence": None}
        return RawOutput(text=json.dumps(ans), latency_s=0.0)

    def repair(self, raw_text: str, max_time: float | None = None) -> RawOutput:
        return RawOutput(text=raw_text, latency_s=0.0)


class MockAdapter(ModelAdapter):
    """測試與部署檢查用：不載入模型。cfg.script 可指定某段回答的文字或錯誤（oom / timeout / crash）。"""

    def __init__(self, key: str, cfg: dict[str, Any]):
        super().__init__(key, cfg)
        self.calls: list[str] = []

    def predict(self, prompt: Prompt, max_time: float | None = None) -> RawOutput:
        seg = self._seg_of(prompt)
        self.calls.append(seg)
        item = (self.cfg.get("script") or {}).get(seg, self.cfg.get("default", "valid"))
        if item == "oom":
            raise MemoryError("CUDA out of memory (mock)")
        if item == "crash":
            raise RuntimeError("mock crash")
        if item == "valid":
            t0, t1 = prompt.t_range
            text = json.dumps({"result": "failure", "failure_reason": "misalignment", "evidence_start_sec": t0,
                               "evidence_end_sec": t1, "best_camera": prompt.images[-1]["camera"],
                               "visible_evidence": "mock", "confidence": 0.5})
            return RawOutput(text=text, latency_s=0.01, input_tokens=10, output_tokens=10)
        return RawOutput(text=str(item), latency_s=0.01, stopped_by_time=(item == "timeout"))

    def repair(self, raw_text: str, max_time: float | None = None) -> RawOutput:
        fixed = (self.cfg.get("repair") or {}).get(raw_text)
        return RawOutput(text=fixed if fixed is not None else raw_text, latency_s=0.01)

    @staticmethod
    def _seg_of(prompt: Prompt) -> str:
        src = prompt.images[0]["source"] if prompt.images else ""
        parts = Path(src).parts
        return parts[-2] if len(parts) >= 2 else ""


class TransformersAdapter(ModelAdapter):
    """Hugging Face transformers 的影像文字模型（Qwen3-VL、SmolVLM2…），可選 bitsandbytes 4-bit。"""

    def load(self) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor

        c = self.cfg
        if c.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("找不到可用的 CUDA GPU（請先執行 setup_vlm.bat，確認 torch 有 CUDA 版）")
        self.torch = torch
        self.dtype = getattr(torch, c.get("dtype", "bfloat16"))
        rev = c.get("revision") or None
        # 先載 processor：缺套件（例如 num2words）時在下載／載入大權重之前就失敗
        self.processor = AutoProcessor.from_pretrained(c["hf_id"], revision=rev)
        ip = getattr(self.processor, "image_processor", None)
        if c.get("image_splitting") is False:
            if ip is None or not hasattr(ip, "do_image_splitting"):
                raise VLMConfigError(f"{c['hf_id']} 的 image processor 沒有 do_image_splitting，無法關閉切塊")
            ip.do_image_splitting = False
        kw: dict[str, Any] = {"device_map": c.get("device_map") or {"": 0},  # 整個模型放在 GPU 0，不悄悄卸到 CPU
                              "attn_implementation": c.get("attn_implementation", "sdpa")}
        q = c.get("quantization")
        if q in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            skip = c.get("quant_skip_modules") or None  # 例如 [visual]：視覺編碼器保持 bf16
            kw["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.dtype,
                                   bnb_4bit_use_double_quant=True, llm_int8_skip_modules=skip)
                if q == "4bit" else BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip))
        t0 = time.perf_counter()
        cls_name = c.get("model_class") or "AutoModelForImageTextToText"  # Qwen3.5 用 AutoModelForMultimodalLM
        model_cls = getattr(transformers, cls_name, None)
        if model_cls is None:
            raise VLMConfigError(f"這版 transformers（{transformers.__version__}）沒有 {cls_name}，請更新 transformers")
        self.template_kwargs = dict(c.get("chat_template_kwargs") or {})  # 例如 Qwen3.5 關掉思考模式
        self.model = model_cls.from_pretrained(c["hf_id"], revision=rev, dtype=self.dtype, **kw)
        if c.get("adapter_path"):  # 第 7 包：LoRA 微調後的模型（基礎模型照常載入，再套上 adapter）
            from peft import PeftModel

            if c.get("adapter_sha256") and adapter_sha256(c["adapter_path"]) != c["adapter_sha256"]:
                raise VLMConfigError(f"{c['adapter_path']} 在設定讀取之後被改過")
            self.model = PeftModel.from_pretrained(self.model, c["adapter_path"], is_trainable=False)
        self.model.eval()
        self.quantized_visual_layers = sum(
            1 for name, m in self.model.named_modules()
            if ("visual" in name or "vision" in name) and type(m).__name__ in ("Linear4bit", "Linear8bitLt"))
        if q in ("4bit", "8bit") and c.get("quant_skip_modules") and self.quantized_visual_layers:
            log.warning("  視覺編碼器有 %d 層仍被量化（quant_skip_modules 沒有對上模組名稱）", self.quantized_visual_layers)
        tc = getattr(self.model.config, "text_config", self.model.config)
        self.max_positions = getattr(tc, "max_position_embeddings", None)
        self.load_s = time.perf_counter() - t0
        self.vram_after_load_mb = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None
        log.info("  載入 %s：%.0f 秒，顯存 %.0f MB", c["hf_id"], self.load_s, self.vram_after_load_mb or 0)

    def _generate(self, messages: list[dict[str, Any]], max_time: float | None,
                  max_new_tokens: int | None = None) -> RawOutput:
        torch = self.torch
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt",
                                                    **getattr(self, "template_kwargs", {}))
        inputs = inputs.to(self.model.device, dtype=self.dtype)
        n_in = int(inputs["input_ids"].shape[1])
        if self.max_positions and n_in >= self.max_positions:
            raise ValueError(f"輸入 {n_in} tokens 超過模型上限 {self.max_positions}（請減少幀數或縮小影像）")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gen = {"max_new_tokens": int(max_new_tokens or self.cfg.get("max_new_tokens", 384)), "do_sample": False,
               "temperature": None, "top_p": None, "top_k": None}  # 貪婪解碼；清掉模型預設的取樣參數
        if max_time:
            gen["max_time"] = float(max_time)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        dt = time.perf_counter() - t0
        new = out[:, n_in:]
        text = self.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        n_out = int(new.shape[1])
        return RawOutput(text=text, latency_s=dt, input_tokens=n_in, output_tokens=n_out, peak_vram_mb=peak,
                         stopped_by_time=bool(max_time and dt >= max_time * 0.98 and n_out < gen["max_new_tokens"]))

    def predict(self, prompt: Prompt, max_time: float | None = None) -> RawOutput:
        return self._generate([{"role": "user", "content": prompt.content}], max_time)

    def repair(self, raw_text: str, max_time: float | None = None) -> RawOutput:
        msg = [{"role": "user", "content": [{"type": "text", "text": REPAIR_INSTRUCTION + raw_text}]}]
        return self._generate(msg, max_time)

    def metadata(self) -> dict[str, Any]:
        md = super().metadata()
        try:
            import torch
            import transformers

            md.update({"torch": torch.__version__, "transformers": transformers.__version__,
                       "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       "load_s": round(getattr(self, "load_s", 0.0), 1),
                       "vram_after_load_mb": round(self.vram_after_load_mb, 0) if getattr(self, "vram_after_load_mb", None) else None,
                       "quantized_visual_layers": getattr(self, "quantized_visual_layers", None),
                       "max_positions": getattr(self, "max_positions", None)})
            try:
                import bitsandbytes

                md["bitsandbytes"] = bitsandbytes.__version__
            except ImportError:
                pass
            if self.cfg.get("adapter_path"):
                try:
                    import peft

                    md["peft"] = peft.__version__
                except ImportError:
                    pass
        except ImportError:
            pass
        return md

    def recover(self) -> None:
        try:
            import gc

            gc.collect()
            self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        for a in ("model", "processor"):
            if hasattr(self, a):
                delattr(self, a)
        self.recover()


def make_adapter(key: str, cfg: dict[str, Any]) -> ModelAdapter:
    t = cfg.get("type")
    if t == "transformers" and cfg.get("reason") is not None:  # 第 11 包
        from .reason import ReasonAdapter

        return ReasonAdapter(key, cfg)
    if t == "transformers" and cfg.get("skill") == "evidence_review":
        from .evidence_review import EvidenceReviewAdapter

        return EvidenceReviewAdapter(key, cfg)
    if t == "transformers" and cfg.get("skill") == "zoom":
        from .skill import ZoomSkillAdapter

        return ZoomSkillAdapter(key, cfg)
    if t == "transformers":
        return TransformersAdapter(key, cfg)
    if t in API_TYPES:
        from .api_models import make_api_adapter

        return make_api_adapter(key, cfg)
    if t == "majority":
        return MajorityAdapter(key, cfg)
    if t == "mock":
        return MockAdapter(key, cfg)
    if t == "encoder":
        raise VLMConfigError(f"{key} 是影像編碼器（做法 A 的分類器用），不能直接回答；請用 python -m scripts.probe")
    raise VLMConfigError(f"未知的模型類型 {t}")


def classify_error(e: BaseException) -> str:
    s = f"{type(e).__name__}: {e}".lower()
    if "out of memory" in s or "outofmemory" in s:
        return "oom"
    if "timeout" in s or "timed out" in s:
        return "timeout"
    return "exception"


# ---------------------------------------------------------------- runner


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _append_jsonl(p: Path, rec: dict[str, Any]) -> None:
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def latest_by_segment(recs: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in recs:
        out[r["segment_id"]] = r
    return out


ERROR_COLS = ["segment_id", "model", "n_frames", "kind", "detail", "created_at"]


@dataclass
class RunSpec:
    exp: dict[str, Any]
    model_key: str
    model_cfg: dict[str, Any]
    n_frames: int  # 0 = 不看影格（majority）

    @property
    def sampling_key(self) -> str | None:
        return self.exp["sampling_template"].format(n=self.n_frames) if self.n_frames else None

    def run_dir(self) -> Path:
        return Path(self.exp["out_dir"]) / self.exp["name"] / self.model_key / f"f{self.n_frames}"


def run_config(spec: RunSpec, builder: PromptBuilder) -> dict[str, Any]:
    cfg = {"tool": TOOL, "model_key": spec.model_key, "model": spec.model_cfg, "n_frames": spec.n_frames,
           "sampling_key": spec.sampling_key, "cameras": spec.exp["cameras"], "prompt_version": builder.version,
           "prompt_sha256": builder.sha256, "split_dir": spec.exp["split_dir"], "include": spec.exp["include"],
           "repair_retry": spec.exp["repair_retry"], "timeout_sec": spec.exp["timeout_sec"]}
    cfg["config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return cfg


def combo_complete(spec: RunSpec, builder: PromptBuilder, segments: list[dict[str, Any]]) -> bool:
    """這個組合是否已用相同設定跑完所有段（ok 或 invalid）。用來略過不必要的模型載入。"""
    rd = spec.run_dir()
    rc_path = rd / "run_config.json"
    if not rc_path.exists():
        return False
    try:
        old = json.loads(rc_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if old.get("config_sha256") != run_config(spec, builder)["config_sha256"]:
        return False
    done = latest_by_segment(_read_jsonl(rd / "predictions.jsonl"))
    return all(done.get(s["segment_id"], {}).get("status") in ("ok", "invalid") for s in segments)


def run_one(spec: RunSpec, adapter: ModelAdapter, builder: PromptBuilder, segments: list[dict[str, Any]], *,
            reasons: list[str], force: bool = False, limit: int | None = None) -> dict[str, Any]:
    """一個（模型 × 幀數）組合跑一輪。已有答案的段略過（error 會重試）；每段的例外各自記錄。"""
    rd = spec.run_dir()
    rd.mkdir(parents=True, exist_ok=True)
    rc = run_config(spec, builder)
    rc_path = rd / "run_config.json"
    raw_p, pred_p, err_p = rd / "raw.jsonl", rd / "predictions.jsonl", rd / "errors.csv"
    if rc_path.exists() and not force:
        old = json.loads(rc_path.read_text(encoding="utf-8"))
        if old.get("config_sha256") != rc["config_sha256"]:
            raise RunConfigMismatch(
                f"{rd} 已有不同設定的結果（prompt／模型／取樣有改）。請換 experiment.name，或加 --force 重新開始")
    if force:  # 重新開始：舊檔改名保留，不刪除
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for p in (raw_p, pred_p, err_p):
            if p.exists():
                p.rename(p.with_name(f"{p.stem}.{stamp}.bak{p.suffix}"))
    rc["metadata"] = adapter.metadata()
    rc["updated_at"] = now_iso()
    rc_path.write_text(json.dumps(rc, ensure_ascii=False, indent=1), encoding="utf-8")
    if not err_p.exists():
        with err_p.open("w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerow(ERROR_COLS)
    done = latest_by_segment(_read_jsonl(pred_p))
    todo = [s for s in segments if done.get(s["segment_id"], {}).get("status") not in ("ok", "invalid")]
    stats = {"run_dir": str(rd), "done_before": len(segments) - len(todo), "new": 0}
    if limit is not None:
        todo = todo[:limit]
    tmp_root = rd / "_tmp_inputs"  # 放在實驗資料夾內（避開含非 ASCII 字元的 %TEMP%），跑完刪除
    tmp_root.mkdir(parents=True, exist_ok=True)
    max_time = float(spec.exp["timeout_sec"]) if spec.exp.get("timeout_sec") else None
    try:
        for i, s in enumerate(todo, 1):
            sid = s["segment_id"]
            base = {"segment_id": sid, "model": spec.model_key, "n_frames": spec.n_frames,
                    "prompt_version": builder.version, "prompt_sha256": builder.sha256,
                    "sampling_key": spec.sampling_key, "created_at": now_iso()}
            raw_rec: dict[str, Any] = dict(base)
            pred: dict[str, Any]
            try:
                if adapter.uses_frames:
                    seg = load_segment_frames(spec.exp["frames_root"], spec.sampling_key, sid, spec.exp["cameras"])
                    if seg is None or not seg.frames:
                        raise FileNotFoundError(f"沒有影格：{spec.exp['frames_root']}/{spec.sampling_key}/segments/{sid}")
                    img_dir = tmp_root / sid
                    paths = prepare_images(seg.frames, spec.model_cfg.get("image_max_side"), img_dir)
                    prompt = builder.build(seg, paths)
                    raw_rec.update({"n_images": len(paths), "n_missing_frames": seg.n_missing,
                                    "images": [{k: v for k, v in im.items() if k != "input"} for im in prompt.images]})
                    t_range = prompt.t_range
                else:
                    prompt, t_range = None, None
                out = adapter.predict(prompt, max_time=max_time)
                raw_rec.update({"raw_text": out.text, "latency_s": round(out.latency_s, 3),
                                "input_tokens": out.input_tokens, "output_tokens": out.output_tokens,
                                "peak_vram_mb": round(out.peak_vram_mb, 1) if out.peak_vram_mb else None,
                                "stopped_by_time": out.stopped_by_time})
                if out.extra:
                    raw_rec.update(out.extra)
                pred = parse_prediction(out.text, segment_id=sid, reasons=reasons, cameras=spec.exp["cameras"],
                                        t_range=t_range, tolerance=float(spec.exp["evidence_tolerance_sec"]))
                pred["attempt"] = "first"
                if pred["status"] == "invalid" and set(pred["errors"]) & JSON_ERRORS and spec.exp["repair_retry"]:
                    rep = adapter.repair(out.text, max_time=max_time)
                    raw_rec.update({"repair_text": rep.text, "repair_latency_s": round(rep.latency_s, 3)})
                    if rep.extra and rep.extra.get("cost_usd") is not None:
                        raw_rec["repair_cost_usd"] = rep.extra["cost_usd"]
                    p2 = parse_prediction(rep.text, segment_id=sid, reasons=reasons, cameras=spec.exp["cameras"],
                                          t_range=t_range, tolerance=float(spec.exp["evidence_tolerance_sec"]))
                    p2["attempt"] = "repair"
                    p2["first_errors"] = pred["errors"]
                    pred = p2
                if out.stopped_by_time:
                    pred["warnings"].append("stopped_by_time")
            except FatalAdapterError:
                raise
            except Exception as e:  # noqa: BLE001 - 每段各自記錄，不中斷整輪
                kind = "frames_missing" if isinstance(e, FileNotFoundError) else classify_error(e)
                raw_rec.update({"error": kind, "error_detail": f"{type(e).__name__}: {e}"[:500]})
                pred = {"segment_id": sid, "status": "error", "errors": [kind], "warnings": [], "result": None,
                        "failure_reason": None, "evidence_start_sec": None, "evidence_end_sec": None,
                        "best_camera": None, "visible_evidence": None, "confidence": None, "attempt": None}
            if pred["status"] == "error":
                adapter.recover()  # 在 except 區塊外清顯存，避免 traceback 還抓著張量
            pred.update({k: base[k] for k in ("model", "n_frames", "prompt_version", "sampling_key", "created_at")})
            pred["dataset_result"] = s.get("dataset_result")
            _append_jsonl(raw_p, raw_rec)
            _append_jsonl(pred_p, pred)
            if pred["status"] != "ok":
                with err_p.open("a", newline="", encoding="utf-8-sig") as fh:
                    csv.writer(fh).writerow([sid, spec.model_key, spec.n_frames, pred["status"],
                                             ";".join(pred["errors"]) + (f" | {raw_rec.get('error_detail')}" if raw_rec.get("error_detail") else ""),
                                             base["created_at"]])
            stats["new"] += 1
            log.info("    [%d/%d] %s → %s%s（%.1f 秒）", i, len(todo), sid, pred["status"],
                     f" {pred['result']}" if pred.get("result") else f" {pred['errors']}",
                     raw_rec.get("latency_s") or 0.0)
            shutil.rmtree(tmp_root / sid, ignore_errors=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    stats.update(summarize_run(spec.run_dir(), segments))
    return stats


def summarize_run(run_dir: Path, segments: list[dict[str, Any]]) -> dict[str, Any]:
    """快速檢查（開發集）：用資料集的成功／失敗標記算粗略準確率。正式評估在第 6 包、用 gold。"""
    preds = latest_by_segment(_read_jsonl(run_dir / "predictions.jsonl"))
    raws = latest_by_segment(_read_jsonl(run_dir / "raw.jsonl"))
    ids = [s["segment_id"] for s in segments]
    truth = {s["segment_id"]: s.get("dataset_result") for s in segments}
    got = [preds[i] for i in ids if i in preds]
    n = len(got)
    ok = [p for p in got if p["status"] == "ok"]
    correct = sum(1 for p in ok if p["result"] == truth.get(p["segment_id"]))
    fails = [i for i in ids if truth.get(i) == "failure" and i in preds]
    caught = sum(1 for i in fails if preds[i].get("status") == "ok" and preds[i].get("result") == "failure")
    succs = [i for i in ids if truth.get(i) == "success" and i in preds]
    succ_ok = sum(1 for i in succs if preds[i].get("status") == "ok" and preds[i].get("result") == "success")
    false_alarm = sum(1 for i in succs if preds[i].get("status") == "ok" and preds[i].get("result") == "failure")
    # 兩類平均：失敗召回與成功答對率的平均（無法判斷、無效、錯誤都算沒答對），不受成功段比較多影響
    balanced = round((caught / len(fails) + succ_ok / len(succs)) / 2, 3) if fails and succs else None
    lat = [raws[i]["latency_s"] + (raws[i].get("repair_latency_s") or 0.0)
           for i in ids if i in raws and raws[i].get("latency_s") is not None]
    vram = [raws[i]["peak_vram_mb"] for i in ids if i in raws and raws[i].get("peak_vram_mb")]
    costs = [(raws[i].get("cost_usd") or 0.0) + (raws[i].get("repair_cost_usd") or 0.0)
             for i in ids if i in raws and raws[i].get("cost_usd") is not None]
    return {
        "n_segments": len(ids), "n_answered": n,
        "n_ok": len(ok), "n_invalid": sum(1 for p in got if p["status"] == "invalid"),
        "n_error": sum(1 for p in got if p["status"] == "error"),
        "n_uncertain": sum(1 for p in ok if p["result"] == "uncertain"),
        "n_repaired": sum(1 for p in ok if p.get("attempt") == "repair"),
        "n_with_warnings": sum(1 for p in ok if p.get("warnings")),
        "quick_accuracy": round(correct / n, 3) if n else None,
        "quick_failure_recall": round(caught / len(fails), 3) if fails else None,
        "quick_balanced_accuracy": balanced,
        "n_correct": correct, "n_failures": len(fails), "n_caught": caught,
        "n_successes": len(succs), "n_success_correct": succ_ok, "n_false_alarms": false_alarm,
        "latency_mean_s": round(sum(lat) / len(lat), 2) if lat else None,
        "latency_median_s": round(sorted(lat)[len(lat) // 2], 2) if lat else None,
        "latency_max_s": round(max(lat), 2) if lat else None,
        "latency_total_s": round(sum(lat), 1) if lat else None,
        "cost_total_usd": round(sum(costs), 4) if costs else None,
        "peak_vram_mb": round(max(vram), 0) if vram else None,
    }


SUMMARY_COLS = ["model", "n_frames", "n_segments", "n_answered", "n_ok", "n_invalid", "n_error", "n_uncertain",
                "n_repaired", "n_with_warnings", "quick_accuracy", "quick_failure_recall", "quick_balanced_accuracy",
                "n_correct", "n_failures", "n_caught", "n_successes", "n_success_correct", "n_false_alarms",
                "latency_mean_s", "latency_median_s", "latency_max_s", "latency_total_s", "peak_vram_mb",
                "cost_total_usd", "note"]


def _frac(a: Any, b: Any) -> str:
    return f"{a}/{b}" if a is not None and b else "–"


def write_summary(exp: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, str]:
    root = Path(exp["out_dir"]) / exp["name"]
    root.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (str(r.get("model")), r.get("n_frames") or 0))
    csv_p = root / "summary.csv"
    with csv_p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def f(v: Any, pct: bool = False) -> str:
        if v is None or v == "":
            return "–"
        return f"{v * 100:.0f}%" if pct else str(v)

    lines = [f"# 第 5 包試跑摘要：{exp['name']}", "",
             f"產生時間：{now_iso()}　切分：{exp['split_dir']}（{', '.join(exp['include'])}）", "",
             "**快速檢查**：準確率用資料集的成功／失敗標記算（無效、錯誤也算答錯）；失敗召回＝真的失敗中，模型答「失敗」的比例；",
             "成功被誤判＝真的成功卻答「失敗」的段數；兩類平均＝（失敗召回＋成功答對率）÷ 2，不受成功段比較多的影響（一律答成功＝50%）。",
             "這只是確認模型能跑、會照格式回答；正式評估在第 6 包，用人工標準答案。", "",
             "| 模型 | 幀數 | 段數 | 格式正確 | 無效 | 錯誤 | 無法判斷 | 準確率 | 失敗召回 | 成功被誤判 | 兩類平均 | 平均秒數 | 峰值顯存 MB | API 費用 US$ | 備註 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['n_frames'] or '不看影格'} | {f(r.get('n_segments'))} | {f(r.get('n_ok'))} | "
                     f"{f(r.get('n_invalid'))} | {f(r.get('n_error'))} | {f(r.get('n_uncertain'))} | "
                     f"{f(r.get('quick_accuracy'), True)} | {f(r.get('quick_failure_recall'), True)} | "
                     f"{_frac(r.get('n_false_alarms'), r.get('n_successes'))} | {f(r.get('quick_balanced_accuracy'), True)} | "
                     f"{f(r.get('latency_mean_s'))} | {f(r.get('peak_vram_mb'))} | {f(r.get('cost_total_usd'))} | {r.get('note') or ''} |")
    lines += ["", "每個組合的原始回答在 `<模型>/f<幀數>/raw.jsonl`，解析結果在 `predictions.jsonl`，錯誤在 `errors.csv`。", ""]
    md_p = root / "summary.md"
    md_p.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_p), "md": str(md_p)}


def collect_summary_rows(exp: dict[str, Any], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """掃描實驗資料夾下所有已跑過的組合（含之前跑的），重算摘要。"""
    root = Path(exp["out_dir"]) / exp["name"]
    rows = []
    for rc in sorted(root.glob("*/f*/run_config.json")):
        cfg = json.loads(rc.read_text(encoding="utf-8"))
        r = {"model": cfg["model_key"], "n_frames": cfg["n_frames"], "prompt_version": cfg.get("prompt_version"),
             "sampling_key": cfg.get("sampling_key")}
        r.update(summarize_run(rc.parent, segments))
        rows.append(r)
    return rows


def effective_model_cfg(exp: dict[str, Any], key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """實驗可以覆寫模型設定（例如推理實驗打開思考模式、放寬輸出長度），不必改 model.yaml 的既有 key。
    覆寫後的設定會寫進 run_config，所以和原本的結果分開。"""
    ov = (exp.get("model_overrides") or {}).get(key) or {}
    return {**cfg, **ov} if ov else cfg


def run_experiment(*, exp: dict[str, Any], model_cfgs: dict[str, dict[str, Any]], model_keys: list[str],
                   frames: list[int], reasons: list[str], base_dir: str | Path = ".", force: bool = False,
                   limit: int | None = None, unlock_test: bool = False) -> dict[str, Any]:
    from .split import load_split

    unknown = [k for k in model_keys if k not in model_cfgs]
    if unknown:
        raise VLMConfigError(f"模型設定檔（--models）沒有這些模型：{unknown}")
    check_task_against_annotation(exp["task"], reasons, exp["cameras"])
    builder = PromptBuilder.from_experiment(exp, base_dir)
    segments = load_split(exp["split_dir"], include=exp["include"], allow_test=unlock_test)
    segments = sorted(segments, key=lambda r: r["segment_id"])
    log.info("第 5 包試跑：%s，%d 段（%s），模型 %s，幀數 %s，prompt %s（%s）", exp["name"], len(segments),
             "/".join(exp["include"]), model_keys, frames, builder.version, builder.sha256[:12])
    notes: dict[tuple[str, int], str] = {}
    results = []
    for mk in model_keys:
        try:
            cfg = resolve_adapter(effective_model_cfg(exp, mk, model_cfgs[mk]))
        except VLMConfigError as e:  # adapter 還沒訓練：只跳過這個模型
            log.error("[%s] %s", mk, e)
            for n in frames:
                notes[(mk, n)] = "沒有 adapter"
            results.append({"model": mk, "load_error": str(e)[:300]})
            continue
        adapter = make_adapter(mk, cfg)
        combos = frames if adapter.uses_frames else [0]
        if not force and all(combo_complete(RunSpec(exp=exp, model_key=mk, model_cfg=cfg, n_frames=n), builder, segments)
                             for n in combos):
            log.info("[%s] %s 都已跑完，略過（不載入模型）", mk, "／".join(f"{n} 幀" if n else "不看影格" for n in combos))
            results.append({"model": mk, "skipped": "complete"})
            continue
        try:
            log.info("[%s] 載入模型 …", mk)
            adapter.load()
        except Exception as e:  # noqa: BLE001 - 一個模型載入失敗不影響其他模型
            kind = classify_error(e)
            log.error("[%s] 載入失敗（%s）：%s", mk, kind, e)
            adapter.close()
            for n in (frames if adapter.uses_frames else [0]):
                notes[(mk, n)] = f"載入失敗：{kind}"
            results.append({"model": mk, "load_error": f"{type(e).__name__}: {e}"[:300]})
            continue
        try:
            for n in (frames if adapter.uses_frames else [0]):
                spec = RunSpec(exp=exp, model_key=mk, model_cfg=cfg, n_frames=n)
                log.info("[%s] %s：", mk, f"{n} 幀" if n else "不看影格")
                try:
                    st = run_one(spec, adapter, builder, segments, reasons=reasons, force=force, limit=limit)
                except RunConfigMismatch as e:  # 只跳過這個組合
                    log.error("[%s] %s", mk, e)
                    notes[(mk, n)] = "設定已改：請換實驗名稱或加 --force"
                    results.append({"model": mk, "n_frames": n, "config_mismatch": str(e)})
                    continue
                except FatalAdapterError as e:  # 例如 API key 錯：這個模型整個跳過，換下一個
                    log.error("[%s] 停止：%s", mk, e)
                    notes[(mk, n)] = f"停止：{str(e)[:60]}"
                    results.append({"model": mk, "n_frames": n, "load_error": str(e)[:300]})
                    break
                results.append({"model": mk, "n_frames": n, **st})
        finally:
            adapter.close()
    rows = collect_summary_rows(exp, segments)
    for (mk, n), note in notes.items():
        hit = [r for r in rows if r["model"] == mk and r["n_frames"] == n]
        if hit:
            hit[0]["note"] = note
        else:
            rows.append({"model": mk, "n_frames": n, "n_segments": len(segments), "note": note})
    paths = write_summary(exp, rows)
    return {"results": results, "summary": paths, "n_segments": len(segments), "prompt_sha256": builder.sha256,
            "tool": f"assembly_monitor {__version__} / {TOOL}"}


def compare_experiments(exp_paths: list[str | Path], out_path: str | Path | None = None) -> dict[str, Any]:
    """把多個提示詞版本（各自一份 experiment*.yaml）的開發集結果排成一張表：同一模型、同一幀數放在一起比較。
    含測試集的實驗不讀（測試集鎖定）。"""
    from .split import load_split

    rows, skipped, out_dir = [], [], None
    for p in exp_paths:
        exp = load_experiment(p)
        if "test" in exp["include"]:
            skipped.append(f"{p}（含測試集，不比較）")
            continue
        out_dir = out_dir or Path(exp["out_dir"])
        if not (Path(exp["out_dir"]) / exp["name"]).exists():
            skipped.append(f"{p}（{exp['name']} 還沒跑）")
            continue
        segments = load_split(exp["split_dir"], include=exp["include"])
        for r in collect_summary_rows(exp, segments):
            no_prompt = not r.get("n_frames") or not r.get("prompt_version")
            r["experiment"], r["version"] = exp["name"], "—" if no_prompt else r["prompt_version"]
            m = re.search(r"_eb([0-9.]+)_", str(r.get("sampling_key") or ""))
            r["end_buffer"] = f"+{m.group(1)} s" if m else None
            rows.append(r)
    # 不用提示詞的方法（多數類、做法 A）只列一次
    seen, uniq = set(), []
    for r in rows:
        key = (r["model"], r["n_frames"], r["version"], r.get("end_buffer"))
        if r["version"] == "—" and key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    uniq.sort(key=lambda r: (r["version"] != "—", str(r["model"]), r.get("n_frames") or 0, str(r["version"]),
                             str(r.get("end_buffer"))))
    out = Path(out_path) if out_path else (out_dir or Path("experiments")) / "prompt_versions.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    def pct(v: Any) -> str:
        return "–" if v is None else f"{v * 100:.0f}%"

    lines = ["# 提示詞版本比較（開發集）", "", f"產生時間：{now_iso()}", "",
             "用資料集的成功／失敗標記快速比較；「—」是不用提示詞的方法（多數類基準、做法 A），當作參考線。",
             "兩類平均＝（失敗召回＋成功答對率）÷ 2；一律答成功＝50%。開發集只有 20 段，1 段約差 5 個百分點。", "",
             "最後一張＝段結束後幾秒（2 秒時可能已拍到下一個動作）。", "",
             "| 模型 | 幀數 | 提示詞 | 最後一張 | 答對 | 抓到失敗 | 成功被誤判 | 無法判斷 | 無效＋錯誤 | 兩類平均 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in uniq:
        bad = (r.get("n_invalid") or 0) + (r.get("n_error") or 0)
        lines.append(f"| {r['model']} | {r.get('n_frames') or '不看影格'} | {r['version']} | {r.get('end_buffer') or '–'} | "
                     f"{_frac(r.get('n_correct'), r.get('n_segments'))} | {_frac(r.get('n_caught'), r.get('n_failures'))} | "
                     f"{_frac(r.get('n_false_alarms'), r.get('n_successes'))} | {r.get('n_uncertain', '–')} | {bad} | "
                     f"{pct(r.get('quick_balanced_accuracy'))} |")
    if skipped:
        lines += ["", "略過：" + "；".join(skipped)]
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return {"md": str(out), "rows": uniq, "skipped": skipped}


# ---------------------------------------------------------------- 環境檢查與下載


def env_report() -> dict[str, Any]:
    rep: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        rep.update({"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            rep.update({"gpu": p.name, "vram_gb": round(p.total_memory / 2**30, 1),
                        "bf16": bool(torch.cuda.is_bf16_supported())})
    except ImportError:
        rep["torch"] = None
    for mod in ("transformers", "accelerate", "bitsandbytes", "huggingface_hub", "PIL", "num2words", "torchvision"):
        try:
            m = __import__(mod)
            rep[mod] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            rep[mod] = None if isinstance(e, ImportError) else f"error: {e}"[:120]
    rep["hf_home"] = os.environ.get("HF_HOME")
    return rep


def model_download_plan(model_cfgs: dict[str, dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, try_to_load_from_cache

    api = HfApi()
    plan = []
    for k in keys:
        c = model_cfgs[k]
        if c.get("type") not in ("transformers", "encoder"):
            continue
        info = api.model_info(c["hf_id"], revision=c.get("revision") or None, files_metadata=True)
        sibs = info.siblings or []
        pats = c.get("allow_patterns")  # 只下載需要的檔案（例如只要 safetensors，不要重複的 .bin）
        if pats:
            import fnmatch

            sibs = [s for s in sibs if any(fnmatch.fnmatch(s.rfilename, pt) for pt in pats)]
        size = sum((s.size or 0) for s in sibs)
        missing = 0  # 本機快取（HF_HOME）還沒有的檔案大小：已下載的模型不會重抓
        for s in sibs:
            try:
                hit = try_to_load_from_cache(c["hf_id"], s.rfilename, revision=info.sha)
            except Exception:  # noqa: BLE001 - 判斷不了就當作還沒下載
                hit = None
            if not isinstance(hit, str):
                missing += s.size or 0
        plan.append({"key": k, "hf_id": c["hf_id"], "revision": c.get("revision") or "main", "sha": info.sha,
                     "size_gb": round(size / 1e9, 1), "missing_gb": round(missing / 1e9, 1),
                     "license": getattr(info.card_data, "license", None)})
    return plan


def download_models(model_cfgs: dict[str, dict[str, Any]], keys: list[str]) -> list[str]:
    from huggingface_hub import snapshot_download

    out = []
    for k in keys:
        c = model_cfgs[k]
        if c.get("type") not in ("transformers", "encoder"):
            continue
        log.info("下載 %s（%s）…", k, c["hf_id"])
        out.append(snapshot_download(c["hf_id"], revision=c.get("revision") or None,
                                     allow_patterns=c.get("allow_patterns") or None))
    return out
