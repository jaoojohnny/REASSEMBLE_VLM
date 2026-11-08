"""Two independent visual observations with deterministic evidence validation.

No dataset labels or first-pass answers enter the second model call.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .skill import make_zoom
from .vlm import Prompt, RawOutput, TransformersAdapter, extract_json_text

RULES = '''Inspect the ordered camera images for assembly of {part}.
Task success definition: {success}
Task failure definition: {failure}
Judge observable final state, using earlier frames only to establish motion.
Gripper proximity, closed fingers, occlusion, missing rotation, and an unseen
insertion depth are NOT positive evidence of failure. Wrist camera movement is
not board movement; use fixed external cameras to assess board displacement.
Do not assume success just because failure is invisible. Small parts or hidden
contact surfaces may be impossible to judge. Do not invent hidden depth or locks.
Choose success only with positive evidence of a correctly seated final part;
choose failure only with visible positive evidence of the final failure state.
Use uncertain if neither is established or cameras contradict each other.
Return JSON only, with result (success/failure/uncertain), evidence_images
(array of original integer image IDs supporting the FINAL state), and observation
(one brief concrete visual observation, not a reasoning transcript).
'''


def validate_observation(raw: str, images: list[dict]) -> dict:
    """Require real image references and at least one final frame reference."""
    try:
        d = json.loads(extract_json_text(raw) or "null")
    except (ValueError, TypeError):
        d = None
    invalid = {"result": "uncertain", "evidence_images": [], "observation": "Invalid or unsupported observation"}
    if not isinstance(d, dict) or d.get("result") not in {"success", "failure", "uncertain"}:
        return invalid
    ids = d.get("evidence_images")
    available = {im["i"]: im for im in images}
    if not isinstance(ids, list) or any(type(i) is not int or i not in available for i in ids):
        return invalid
    observation = d.get("observation")
    if not isinstance(observation, str) or not observation.strip():
        return invalid
    latest = {}
    for im in images:
        latest[im["camera"]] = max(latest.get(im["camera"], float("-inf")), im["time_sec"])
    if d["result"] != "uncertain" and not any(
        available[i]["time_sec"] == latest[available[i]["camera"]] for i in ids
    ):
        return invalid
    return {"result": d["result"], "evidence_images": list(dict.fromkeys(ids)), "observation": observation[:1200]}


def combine(first: dict, second: dict, images: list[dict]) -> dict:
    result = first["result"] if first["result"] == second["result"] else "uncertain"
    ids = sorted(set(first["evidence_images"] + second["evidence_images"]))
    evidence = [im for im in images if im["i"] in ids]
    times = [im["time_sec"] for im in evidence]
    return {"result": result, "failure_reason": "unknown" if result == "failure" else None,
            "evidence_start_sec": min(times) if times else None,
            "evidence_end_sec": max(times) if times else None,
            "best_camera": evidence[-1]["camera"] if evidence else None,
            "visible_evidence": first["observation"] + " | Review: " + second["observation"],
            "confidence": None}


class EvidenceReviewAdapter(TransformersAdapter):
    def metadata(self):
        md = super().metadata()
        md["evidence_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return md

    def predict(self, prompt: Prompt, max_time: float | None = None) -> RawOutput:
        start = time.perf_counter()
        calls, observations, crops, crop_errors = [], [], [], []
        instruction = RULES.format(part=prompt.task.get("part", "part"),
                                   success=prompt.task.get("success_definition", "correctly seated"),
                                   failure=prompt.task.get("failure_definition", "not correctly seated"))
        base = [{"type": "text", "text": instruction}]
        for im in prompt.images:
            base.extend([{"type": "text", "text": f"Image {im['i']}: camera {im['camera']}, t={im['time_sec']} s"},
                         {"type": "image", "path": im["input"]}])
        for stage in range(2):
            content = list(base)
            if stage == 1:
                # Two latest wrist frames preserve local before/after context.
                wrist = sorted([im for im in prompt.images if im["camera"] == "hand"], key=lambda im: im["time_sec"])[-2:]
                for im in wrist:
                    if max_time is not None and time.perf_counter() - start >= max_time:
                        break
                    try:
                        z = make_zoom(im["source"], [0.2, 0.2, 0.8, 0.8],
                                      Path(im["input"]).parent / "evidence_review" / f"crop_{im['i']}.jpg",
                                      side=512, margin=0, min_frac=0)
                        crops.append({"image_id": im["i"], **z})
                        content.extend([{"type": "text", "text": f"Detail of original Image {im['i']}; same evidence, not another camera. Enlargement adds no new physical detail."},
                                        {"type": "image", "path": z["path"]}])
                    except OSError as exc:
                        crop_errors.append(str(exc))
                content.append({"type": "text", "text": "Independently check both possible outcomes against the final images. If physical seating cannot be resolved, use uncertain."})
            remaining = None if max_time is None else max_time - (time.perf_counter() - start)
            if remaining is not None and remaining <= 0:
                break
            answer = self._generate([{"role": "user", "content": content}], remaining)
            calls.append(answer)
            observations.append(validate_observation(answer.text if not answer.stopped_by_time else "", prompt.images))
            if answer.stopped_by_time:
                break
        while len(observations) < 2:
            observations.append(validate_observation("", prompt.images))
        decision = combine(*observations, prompt.images)
        peaks = [c.peak_vram_mb for c in calls if c.peak_vram_mb is not None]
        return RawOutput(json.dumps(decision, ensure_ascii=False), time.perf_counter() - start,
                         input_tokens=sum(c.input_tokens or 0 for c in calls),
                         output_tokens=sum(c.output_tokens or 0 for c in calls),
                         peak_vram_mb=max(peaks) if peaks else None,
                         stopped_by_time=len(calls) < 2 or any(c.stopped_by_time for c in calls),
                         extra={"skill": {"name": "evidence_review_v1", "observations": observations,
                                          "raw_stages": [c.text for c in calls], "crops": crops,
                                          "crop_errors": crop_errors, "model_calls": len(calls)}})
