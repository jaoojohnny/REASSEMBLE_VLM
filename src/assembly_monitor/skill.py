"""第 8 包：幫 Qwen 加的 Skill——放大鏡工具（zoom）。

問題：影格縮到長邊 512 後，BNC／USB 接頭在畫面裡只佔很小一塊，「差一點沒插到底」這種細節看不到。
做法（兩次呼叫同一個模型，中間由程式執行工具）：
  1. 找位置：只給模型「最後一個時間點」三台相機的影像，請它在每張圖框出零件與插座（0–1000 的相對座標）。
  2. 工具：程式依框從**原始解析度**（640×480）的影格裁切、外擴一圈、放大到長邊 zoom_side。
  3. 判斷：原本的所有影像＋放大圖一起給模型，照原本的提示詞回答。
框不出來（格式錯、看不到）就只用原本的影像回答，等於沒有 Skill。兩次呼叫的時間都算進這一段的執行時間。

第二版（2026-10-27，第一版在外部相機框不準）：
  zoom_cameras：只對這些相機放大（例如 [hand]，手腕相機的框大多正確）。
  zoom_mode: fixed：不叫模型找位置，直接裁切固定區域 zoom_fixed_box（手腕相機裝在夾爪上，
    接頭與插座幾乎都在畫面中下方、兩根手指之間），只呼叫模型一次。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .vlm import Prompt, RawOutput, TransformersAdapter, extract_json_text

log = logging.getLogger(__name__)

LOCATE_TEMPLATE = (
    "These are the final images from {n} cameras, taken after a robot arm tried to insert the {part} into its place "
    "on the task board.\n"
    "For each image, find the {part} and the place (socket or hole) it should be inserted into. "
    "If the gripper is still next to it, include the gripper fingertips. Give ONE tight box per image that covers them.\n"
    "Use integer coordinates from 0 to 1000 relative to the image: (0, 0) is the top-left corner and (1000, 1000) "
    "is the bottom-right corner. If the {part} is not visible in an image, use null for that image.\n"
    "Reply with ONE JSON object only, in exactly this form:\n"
    '{{"boxes": {{{example}}}}}'
)
ZOOM_INTRO = ("\nZoomed views: a tool cropped the original full-resolution final images around the {part} "
              "(using the boxes you gave) and enlarged them. Use them together with the images above.\n")
FIXED_INTRO = ("\nZoomed view: a tool cropped a fixed region of the original full-resolution final image of the wrist "
               "camera (the area between the gripper fingers, where the {part} and its socket usually are) and enlarged it. "
               "Use it together with the images above.\n")
FINAL = "\nNow answer with the JSON object only."


def last_images(prompt: Prompt) -> list[dict[str, Any]]:
    """每台相機時間最晚的那張（＝最後狀態），依提示詞裡出現的相機順序。"""
    best: dict[str, dict[str, Any]] = {}
    for im in prompt.images:
        cur = best.get(im["camera"])
        if cur is None or im["time_sec"] > cur["time_sec"]:
            best[im["camera"]] = im
    order = []
    for im in prompt.images:
        if im["camera"] not in order:
            order.append(im["camera"])
    return [best[c] for c in order]


def locate_prompt(prompt: Prompt, ims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    part = (prompt.task or {}).get("part") or "part"
    example = ", ".join(f'"{im["camera"]}": [x1, y1, x2, y2]' for im in ims)
    content: list[dict[str, Any]] = []
    for im in ims:
        content.append({"type": "text", "text": f"Image of camera {im['camera']} (t = {im['time_sec']:+.2f} s):\n"})
        content.append({"type": "image", "path": im["input"]})
    content.append({"type": "text", "text": "\n" + LOCATE_TEMPLATE.format(n=len(ims), part=part, example=example)})
    return content


def parse_boxes(raw: str, cameras: list[str], input_sizes: dict[str, tuple[int, int]]) -> dict[str, list[float] | None]:
    """回傳每台相機的框（0–1 的相對座標 [x1, y1, x2, y2]）；不合法就 None。
    模型若給的是像素座標（數值超過 1000），用送進模型的影像大小換算。"""
    out: dict[str, list[float] | None] = {c: None for c in cameras}
    js = extract_json_text(raw or "")
    if not js:
        return out
    try:
        d = json.loads(js)
    except ValueError:
        return out
    boxes = d.get("boxes", d) if isinstance(d, dict) else None
    if not isinstance(boxes, dict):
        return out
    for c in cameras:
        b = boxes.get(c)
        if isinstance(b, dict):
            b = b.get("bbox_2d") or b.get("box")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        if max(x1, y1, x2, y2) > 1000.5:  # 像素座標
            w, h = input_sizes.get(c, (1000, 1000))
            x1, x2, y1, y2 = x1 / w, x2 / w, y1 / h, y2 / h
        else:
            x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
        x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
        y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            continue
        out[c] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
    return out


def crop_box(w: int, h: int, box: list[float], margin: float = 0.3, min_frac: float = 0.25) -> tuple[int, int, int, int]:
    """相對座標的框 → 原圖上的裁切範圍（像素）：每邊外擴 margin × 框的寬高，且至少 min_frac × 短邊，不超出畫面。"""
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    bw, bh = x2 - x1, y2 - y1
    side_min = min_frac * min(w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max(bw * (1 + 2 * margin), side_min) / 2
    half_h = max(bh * (1 + 2 * margin), side_min) / 2
    l, r = cx - half_w, cx + half_w
    t, b = cy - half_h, cy + half_h
    if l < 0:
        r, l = r - l, 0
    if r > w:
        l, r = max(0, l - (r - w)), w
    if t < 0:
        b, t = b - t, 0
    if b > h:
        t, b = max(0, t - (b - h)), h
    return int(round(l)), int(round(t)), int(round(r)), int(round(b))


def make_zoom(src: str, box: list[float], out_path: Path, side: int = 512, margin: float = 0.3,
              min_frac: float = 0.25) -> dict[str, Any]:
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"讀不到原始影格 {src}")
    h, w = img.shape[:2]
    l, t, r, b = crop_box(w, h, box, margin, min_frac)
    crop = img[t:b, l:r]
    ch, cw = crop.shape[:2]
    s = side / float(max(ch, cw))
    if s > 1:  # 只放大；比 side 大就不動
        crop = cv2.resize(crop, (max(1, round(cw * s)), max(1, round(ch * s))), interpolation=cv2.INTER_CUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise OSError(f"放大圖編碼失敗 {src}")
    buf.tofile(str(out_path))
    return {"crop_px": [l, t, r, b], "zoom": round(max(1.0, s), 2), "path": str(out_path)}


def image_size(path: str) -> tuple[int, int]:
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return (img.shape[1], img.shape[0]) if img is not None else (1000, 1000)


class ZoomSkillAdapter(TransformersAdapter):
    """Qwen＋放大鏡工具。設定：skill: zoom、zoom_side（預設 512）、zoom_margin（0.3）、zoom_min_frac（0.25）、
    locate_max_new_tokens（160）；第二版：zoom_cameras、zoom_mode（locate／fixed）、zoom_fixed_box。"""

    def predict(self, prompt: Prompt, max_time: float | None = None) -> RawOutput:
        ims = last_images(prompt)
        allowed = self.cfg.get("zoom_cameras")
        if allowed:
            ims = [im for im in ims if im["camera"] in allowed]
        cams = [im["camera"] for im in ims]
        mode = self.cfg.get("zoom_mode", "locate")
        t0 = time.perf_counter()
        loc = None
        if mode == "fixed":  # 不叫模型找位置：每台相機用同一個固定區域、不外擴
            boxes = {c: [float(x) for x in self.cfg["zoom_fixed_box"]] for c in cams}
            margin, min_frac = 0.0, 0.0
        else:
            boxes = {c: None for c in cams}
            if ims:
                loc = self._generate([{"role": "user", "content": locate_prompt(prompt, ims)}], max_time,
                                     max_new_tokens=int(self.cfg.get("locate_max_new_tokens", 160)))
                boxes = parse_boxes(loc.text, cams, {im["camera"]: image_size(im["input"]) for im in ims})
            margin, min_frac = float(self.cfg.get("zoom_margin", 0.3)), float(self.cfg.get("zoom_min_frac", 0.25))
        zooms, zinfo = [], {}
        zdir = Path(ims[0]["input"]).parent / "zoom" if ims else Path(".")
        for im in ims:
            b = boxes.get(im["camera"])
            if not b:
                continue
            try:
                z = make_zoom(im["source"], b, zdir / f"zoom_{im['camera']}.jpg", side=int(self.cfg.get("zoom_side", 512)),
                              margin=margin, min_frac=min_frac)
            except (OSError, FileNotFoundError) as e:
                log.warning("  放大失敗 %s：%s", im["camera"], e)
                continue
            zooms.append((im, z))
            zinfo[im["camera"]] = {"box": b, "crop_px": z["crop_px"], "zoom": z["zoom"]}
        content = list(prompt.content)
        if content and content[-1].get("type") == "text" and content[-1]["text"].strip().startswith("Now answer"):
            content = content[:-1]
        if zooms:
            part = (prompt.task or {}).get("part") or "part"
            content.append({"type": "text", "text": (FIXED_INTRO if mode == "fixed" else ZOOM_INTRO).format(part=part)})
            for im, z in zooms:
                content.append({"type": "text", "text": f"\nZoom of camera {im['camera']} at t = {im['time_sec']:+.2f} s\n"})
                content.append({"type": "image", "path": z["path"]})
        content.append({"type": "text", "text": FINAL})
        ans = self._generate([{"role": "user", "content": content}], max_time)
        dt = time.perf_counter() - t0
        extra = {"skill": {"name": "zoom", "mode": mode, "cameras": cams,
                           "locate_text": (loc.text or "")[:1500] if loc else None, "boxes": boxes, "zooms": zinfo,
                           "n_zoom": len(zooms), "locate_latency_s": round(loc.latency_s, 3) if loc else None,
                           "answer_latency_s": round(ans.latency_s, 3)}}
        peaks = [x for x in ((loc.peak_vram_mb if loc else None), ans.peak_vram_mb) if x is not None]
        return RawOutput(text=ans.text, latency_s=dt,
                         input_tokens=((loc.input_tokens or 0) if loc else 0) + (ans.input_tokens or 0),
                         output_tokens=((loc.output_tokens or 0) if loc else 0) + (ans.output_tokens or 0),
                         peak_vram_mb=max(peaks) if peaks else None, stopped_by_time=ans.stopped_by_time, extra=extra)
