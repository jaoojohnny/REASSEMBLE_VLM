"""第 11 包：讓 Qwen 判斷更準的四種推理方式（2026-10-30 設計，只在 4 個新零件的開發集挑選）。

背景：Qwen＋Skill 第三版（清單逐項 yes/no＋手腕裁切）的錯誤幾乎都是**把成功說成失敗**
（bolt 4 21 個錯裡 19 個、Ethernet 25 個裡 24 個、square peg 1 48 個裡 44 個）。
第 5 包試過「先寫推理再回答」與「內建思考模式」，在 BNC 都沒有幫助，所以這次不讓模型多寫字，而是針對這個偏差：

1. 機率門檻（reason: score）：不讀 Qwen 寫出來的 yes/no，改讀它說 yes 的機率。
   做法是「強制作答」：把題目＋固定答案 {"held_by_gripper": "no", ..., "result": "success"} 一起送進模型，
   一次前向計算就拿到每一格寫 yes／no 的機率（每一項都是「前面幾項都答 no」時的條件機率）。
   失敗分數 s = 1 − Π(1 − P(yes))；s > 門檻 τ 就判失敗。τ 在開發集選（用其他零件選、在留下的零件上算）。
   rule: any、threshold 0.5 ＝ 任一項 P(yes) > 0.5 就失敗（約等於原本寫字的答案）。
2. 前後對照（contrast: true，配 contrast-v1 提示詞）：多給一張手腕相機「第一張｜最後一張」並排圖，
   問「最後一張裡零件是不是還在指尖之間、大小位置和第一張差不多」（成功時夾爪放開往上移，零件變小、留在板上）。
3. 答失敗再檢查一次（reason: verify）：第一階段同 1（rule any 0.5）。判失敗時，只看手腕相機的並排圖與裁切圖，
   告訴模型第一階段的理由，問「第一階段對不對」；P(對) > 0.5 才維持失敗。
4. 看範例（fewshot）：每種零件先從切分的 ref 集（不評分）挑 1 段成功、1 段失敗，
   把它們最後一張手腕影格與裁切圖當範例一起給模型。前提是新零件要先有標好的例子。

所有方法都用同一個 Qwen3-VL-8B（4-bit），影格和第三版相同（4 幀 × 3 相機、+0.5 秒、長邊 512）＋手腕固定裁切。
重型套件只在 load() 才匯入；機率與判斷規則是純 Python，可以在沒有 GPU 的環境測試。

第二輪（2026-10-31，使用者要求結合 Codex 的 evidence_review 修改 Skill 與提示詞）：
5. 新清單 checklist-fields-v2：加入 Codex 的規則（夾爪剛放開還靠近零件、手指擋住、零件貼平、手腕相機移動都「不是失敗的證據」；
   只有清楚看到才答 yes），Skill 改成裁切**最後兩張**手腕影格（zoom_last_n: 2，Codex 的局部前後比較）。
6. 手腕第二次觀察（second_view）：Codex 的「第二次獨立觀察、不告訴第一次的答案」，只給手腕「第一張｜最後一張」並排圖＋兩張裁切，
   用中立的提示詞（wrist-v1）再讀一次機率。第一輪「再檢查一次」告訴模型第一階段的答案又說「常常錯」，結果一律否定；這次不給。
7. 兩次觀察合併：Codex 是「兩次一樣才接受、不一樣答不確定」；這裡不確定算答錯，所以改成
   both_and（兩次都說失敗才失敗）或 both_mean（兩次失敗分數的 logit 平均 > 門檻）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

from .vlm import Prompt, RawOutput, TransformersAdapter, VLMConfigError, load_segment_frames

log = logging.getLogger(__name__)

REASONS = ("score", "verify")
CHECKS = ("held_by_gripper", "place_empty_or_part_misplaced", "board_moved")
RULES = ("any", "combined", "both_mean", "both_and", "second_any")  # second_any：只用手腕第二次觀察（第 11 包測試集）
WRIST_CHECKS = ("held_by_gripper", "place_empty_or_part_misplaced")
REASON_TEXT = {
    "held_by_gripper": "at the end, the {part} still seemed to be between the gripper fingers",
    "place_empty_or_part_misplaced": "at the end, the place on the board seemed empty, or the {part} seemed tilted or beside its place",
    "board_moved": "the task board itself seemed to move",
    "result": "the {part} did not seem to be correctly inserted",
}
CONTRAST_INTRO = ("\nSide-by-side view of the wrist camera (hand): LEFT = its first image (t = {t0:+.2f} s, "
                  "the gripper is holding the {part}), RIGHT = its last image (t = {t1:+.2f} s).\n")
ZOOM2_INTRO = ("\nZoomed views: a tool cropped a fixed region (the area between the gripper fingers) of the last two "
               "wrist-camera images at full resolution and enlarged them. They are details of those images, not another camera.\n")
FEWSHOT_INTRO = ("\nReference examples: two OTHER attempts with the same {part} (not the attempt you are checking). "
                 "They show what the end of a successful attempt and of a failed attempt look like in the wrist camera.\n")


# ---------------------------------------------------------------- 設定檢查


def check_reason_cfg(k: str, v: dict[str, Any]) -> None:
    """只檢查、不補預設值（run_config 的雜湊包含模型設定）。"""
    if v.get("type") != "transformers" or v.get("skill"):
        raise VLMConfigError(f"模型 {k}：reason 只用在 transformers 模型，而且不能和 skill 一起用")
    if v.get("reason") not in REASONS:
        raise VLMConfigError(f"模型 {k}：reason 只能是 {REASONS}")
    b = v.get("zoom_fixed_box")
    ok = isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b)
    if not ok or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
        raise VLMConfigError(f"模型 {k}：需要 zoom_fixed_box: [x1, y1, x2, y2]（0–1 的相對座標，和第三版相同）")
    d = v.get("decision")
    if not isinstance(d, dict) or d.get("rule") not in RULES:
        raise VLMConfigError(f"模型 {k}：decision 要是 {{rule: any|combined, threshold: 0–1}}")
    t = d.get("threshold")
    if not isinstance(t, (int, float)) or not 0 < float(t) < 1:
        raise VLMConfigError(f"模型 {k}：decision.threshold 要在 0 和 1 之間")
    if "contrast" in v and not isinstance(v["contrast"], bool):
        raise VLMConfigError(f"模型 {k}：contrast 要是 true／false")
    if "zoom_last_n" in v and v["zoom_last_n"] not in (1, 2, 3):
        raise VLMConfigError(f"模型 {k}：zoom_last_n 只能是 1、2、3（裁切最後幾張手腕影格）")
    if "second_view" in v and not (isinstance(v["second_view"], str) and v["second_view"]):
        raise VLMConfigError(f"模型 {k}：second_view 要是第二次觀察的提示詞檔路徑")
    if d["rule"] in ("both_mean", "both_and", "second_any") and not v.get("second_view"):
        raise VLMConfigError(f"模型 {k}：rule {d['rule']} 需要 second_view（兩次觀察）")
    if v["reason"] == "verify" and v.get("second_view"):
        raise VLMConfigError(f"模型 {k}：verify 和 second_view 不能一起用")
    if v["reason"] == "verify":
        if not v.get("verify_template"):
            raise VLMConfigError(f"模型 {k}：reason: verify 需要 verify_template（第二階段的提示詞檔）")
        if d["rule"] != "any" or float(t) != 0.5:
            raise VLMConfigError(f"模型 {k}：verify 的第一階段固定用 rule any、threshold 0.5（約等於第三版寫字的答案）")
    fs = v.get("fewshot")
    if fs is not None:
        need = ("split_dir", "set", "frames_root", "sampling_key", "seed")
        if not isinstance(fs, dict) or any(x not in fs for x in need):
            raise VLMConfigError(f"模型 {k}：fewshot 需要 {need}（通常由實驗的 model_overrides 指定）")
        if fs["set"] in ("dev", "test"):
            raise VLMConfigError(f"模型 {k}：範例不能取自評分的集合（{fs['set']}）")


# ---------------------------------------------------------------- 機率與判斷（純 Python）


def two_way(lp_a: float, lp_b: float) -> float:
    """兩個候選 token 的 log 機率 → 第一個的機率（只在兩者之間重新正規化）。"""
    m = max(lp_a, lp_b)
    ea, eb = math.exp(lp_a - m), math.exp(lp_b - m)
    return ea / (ea + eb)


def fail_score(p_yes: dict[str, float]) -> float:
    """至少一項答 yes 的機率：1 − Π(1 − P(yes))。"""
    q = 1.0
    for c in CHECKS:
        q *= 1.0 - float(p_yes.get(c, 0.0))
    return 1.0 - q


def logit(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def mean_score(s1: float, s2: float) -> float:
    """兩次觀察的失敗分數在 logit 尺度平均（兩次都很確定才會很高）。"""
    return 1.0 / (1.0 + math.exp(-(logit(s1) + logit(s2)) / 2))


def decide(p_yes: dict[str, float], rule: str, threshold: float, p2: dict[str, float] | None = None) -> str:
    if rule == "any":
        return "failure" if any(float(p_yes.get(c, 0.0)) > threshold for c in CHECKS) else "success"
    if rule == "combined":
        return "failure" if fail_score(p_yes) > threshold else "success"
    if rule == "second_any":
        if p2 is None:
            raise ValueError("規則 second_any 需要第二次觀察的機率")
        return decide(p2, "any", threshold)
    if rule in ("both_mean", "both_and"):
        if p2 is None:
            raise ValueError(f"規則 {rule} 需要第二次觀察的機率")
        if rule == "both_and":
            a, b = decide(p_yes, "any", threshold), decide(p2, "any", threshold)
            return "failure" if a == b == "failure" else "success"
        return "failure" if mean_score(fail_score(p_yes), fail_score(p2)) > threshold else "success"
    raise ValueError(f"未知的判斷規則 {rule}")


def answer_text(p_yes: dict[str, float], result: str) -> str:
    """寫成和第三版相同格式的 JSON（每項 P(yes) > 0.5 寫 yes），讓原本的解析與評分照常運作。"""
    d = {c: ("yes" if float(p_yes.get(c, 0.0)) > 0.5 else "no") for c in CHECKS}
    d["result"] = result
    return json.dumps(d)


def checklist_pieces() -> list[tuple[str, str | None, tuple[str, str] | None]]:
    """強制作答的片段：（文字、要讀機率的欄位、兩個候選答案）。欄位的文字是固定答案（no／success）。"""
    out: list[tuple[str, str | None, tuple[str, str] | None]] = []
    for i, c in enumerate(CHECKS):
        out.append((('{"' if i == 0 else '", "') + c + '": "', None, None))
        out.append(("no", c, ("yes", "no")))
    out.append(('", "result": "', None, None))
    out.append(("success", "result", ("failure", "success")))
    out.append(('"}', None, None))
    return out


def wrist_pieces() -> list[tuple[str, str | None, tuple[str, str] | None]]:
    out: list[tuple[str, str | None, tuple[str, str] | None]] = []
    for i, c in enumerate(WRIST_CHECKS):
        out.append((('{"' if i == 0 else '", "') + c + '": "', None, None))
        out.append(("no", c, ("yes", "no")))
    out.append(('", "result": "', None, None))
    out.append(("success", "result", ("failure", "success")))
    out.append(('"}', None, None))
    return out


def verify_pieces() -> list[tuple[str, str | None, tuple[str, str] | None]]:
    return [('{"first_check_correct": "', None, None), ("no", "first_check_correct", ("yes", "no")), ('"}', None, None)]


def probe_positions(prompt_len: int, piece_ids: list[list[int]],
                    pieces: list[tuple[str, str | None, tuple[str, str] | None]]) -> tuple[list[int], list[tuple[int, str, tuple[str, str]]]]:
    """把各片段的 token 接在題目後面；回傳（答案的 token、每個欄位要讀 logits 的位置）。
    位置 p 的 logits 預測位置 p+1 的 token，所以欄位答案第一個 token 前一格就是要讀的位置。"""
    ids: list[int] = []
    probes = []
    for (text, fld, cands), t in zip(pieces, piece_ids):
        if fld is not None:
            probes.append((prompt_len + len(ids) - 1, fld, cands))
        ids += t
    return ids, probes


def pick_refs(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """從 ref 集挑 1 段成功、1 段失敗（依 segment_id 排序後用 seed 抽，結果固定）。"""
    rng = random.Random(seed)
    out = {}
    for res in ("success", "failure"):
        c = sorted(r["segment_id"] for r in rows if r.get("dataset_result") == res)
        if not c:
            raise VLMConfigError(f"ref 集沒有{'成功' if res == 'success' else '失敗'}段，不能挑範例")
        out[res] = rng.choice(c)
    return out


# ---------------------------------------------------------------- 影像


def _read(path: str):
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"讀不到影格 {path}")
    return img


def _write(img, out_path: Path) -> str:
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise OSError(f"影像編碼失敗 {out_path}")
    buf.tofile(str(out_path))
    return str(out_path)


def side_by_side(first: str, last: str, out_path: Path, side: int = 1024) -> str:
    """兩張原始影格左右並排（中間 8 px 白線），縮到長邊 side。"""
    import cv2
    import numpy as np

    a, b = _read(first), _read(last)
    h = min(a.shape[0], b.shape[0])
    a = cv2.resize(a, (round(a.shape[1] * h / a.shape[0]), h)) if a.shape[0] != h else a
    b = cv2.resize(b, (round(b.shape[1] * h / b.shape[0]), h)) if b.shape[0] != h else b
    img = np.concatenate([a, np.full((h, 8, 3), 255, dtype=a.dtype), b], axis=1)
    s = side / float(max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return _write(img, out_path)


def resized(src: str, out_path: Path, side: int) -> str:
    import cv2

    img = _read(src)
    s = side / float(max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return _write(img, out_path)


def wrist_first_last(prompt: Prompt, camera: str = "hand") -> tuple[dict[str, Any], dict[str, Any]] | None:
    ims = sorted((im for im in prompt.images if im["camera"] == camera), key=lambda im: im["time_sec"])
    return (ims[0], ims[-1]) if len(ims) >= 2 else None


# ---------------------------------------------------------------- adapter


class ReasonAdapter(TransformersAdapter):
    """設定：reason（score／verify）、decision {rule, threshold}、zoom_fixed_box、zoom_side、contrast、contrast_side、
    verify_template、fewshot {split_dir, set, frames_root, sampling_key, seed}。"""

    def load(self) -> None:
        super().load()
        tok = self.processor.tokenizer
        self.cand = {w: tok(w, add_special_tokens=False)["input_ids"][0] for w in ("yes", "no", "success", "failure")}
        if len({self.cand["yes"], self.cand["no"]}) != 2 or len({self.cand["success"], self.cand["failure"]}) != 2:
            raise VLMConfigError(f"候選答案的第一個 token 相同，不能讀機率：{self.cand}")
        self._prepare_static()

    def _prepare_static(self) -> None:
        """第二階段提示詞與範例影像（不需要模型，測試可以直接呼叫）。"""
        check_reason_cfg(self.key, self.cfg)  # 實驗的 model_overrides（例如 fewshot）合併後再檢查一次
        if self.cfg.get("fewshot_required") and not self.cfg.get("fewshot"):
            raise VLMConfigError(f"{self.key} 要看範例，但實驗沒有用 model_overrides 指定 fewshot")
        self.verify_text = None
        if self.cfg.get("reason") == "verify":
            p = Path(self.cfg["verify_template"])
            self.verify_text = p.read_text(encoding="utf-8")
            self.verify_sha256 = hashlib.sha256(self.verify_text.encode("utf-8")).hexdigest()
        self.second_text = None
        if self.cfg.get("second_view"):
            self.second_text = Path(self.cfg["second_view"]).read_text(encoding="utf-8")
            self.second_sha256 = hashlib.sha256(self.second_text.encode("utf-8")).hexdigest()
        self.refs: list[dict[str, Any]] = []
        fs = self.cfg.get("fewshot")
        if fs:
            from .split import load_split

            rows = load_split(fs["split_dir"], include=(fs["set"],))
            chosen = pick_refs(rows, int(fs["seed"]))
            cache = Path(self.cfg.get("fewshot_cache", "experiments/_fewshot_refs")) / Path(fs["split_dir"]).name
            from .skill import make_zoom

            for res in ("success", "failure"):
                sid = chosen[res]
                seg = load_segment_frames(fs["frames_root"], fs["sampling_key"], sid, ["hand"])
                if seg is None or not seg.frames:
                    raise FileNotFoundError(f"範例 {sid} 沒有手腕影格（{fs['frames_root']}/{fs['sampling_key']}）")
                last = max(seg.frames, key=lambda f: f.time_sec)
                img = resized(last.path, cache / f"{sid}_{res}_last.jpg", int(self.cfg.get("image_max_side") or 512))
                z = make_zoom(last.path, list(self.cfg["zoom_fixed_box"]), cache / f"{sid}_{res}_crop.jpg",
                              side=int(self.cfg.get("zoom_side", 512)), margin=0.0, min_frac=0.0)
                self.refs.append({"result": res, "segment_id": sid, "time_sec": round(last.time_sec, 3),
                                  "image": img, "crop": z["path"]})
            log.info("  範例（%s）：成功 %s、失敗 %s", fs["split_dir"], chosen["success"], chosen["failure"])

    # ------------------------------------------------------------ 題目內容

    def build_content(self, prompt: Prompt) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """第三版的內容（原本的影像＋手腕固定裁切），依設定加上前後並排圖、範例。"""
        from .skill import FINAL, FIXED_INTRO, last_images, make_zoom

        part = (prompt.task or {}).get("part") or "part"
        content = list(prompt.content)
        if content and content[-1].get("type") == "text" and content[-1]["text"].strip().startswith("Now answer"):
            content = content[:-1]
        info: dict[str, Any] = {}
        hand = [im for im in last_images(prompt) if im["camera"] == "hand"]
        work = Path(hand[0]["input"]).parent / "reason" if hand else None
        fl = wrist_first_last(prompt)
        if self.cfg.get("contrast") and fl and work:
            p = side_by_side(fl[0]["source"], fl[1]["source"], work / "hand_first_last.jpg",
                             side=int(self.cfg.get("contrast_side", 1024)))
            content.append({"type": "text", "text": CONTRAST_INTRO.format(t0=fl[0]["time_sec"], t1=fl[1]["time_sec"], part=part)})
            content.append({"type": "image", "path": p})
            info["contrast"] = {"first_t": fl[0]["time_sec"], "last_t": fl[1]["time_sec"]}
        n_zoom = int(self.cfg.get("zoom_last_n", 1))
        if hand and work and n_zoom == 1:  # 第三版：只裁最後一張（內容和 ZoomSkillAdapter 完全相同）
            im = hand[0]
            z = make_zoom(im["source"], list(self.cfg["zoom_fixed_box"]), work / "zoom_hand.jpg",
                          side=int(self.cfg.get("zoom_side", 512)), margin=0.0, min_frac=0.0)
            content.append({"type": "text", "text": FIXED_INTRO.format(part=part)})
            content.append({"type": "text", "text": f"\nZoom of camera {im['camera']} at t = {im['time_sec']:+.2f} s\n"})
            content.append({"type": "image", "path": z["path"]})
            info["zoom_path"] = z["path"]
            info["zooms"] = [{"t": im["time_sec"], "path": z["path"]}]
        elif hand and work:  # 第二輪：裁最後 n 張（Codex 的局部前後比較）
            ims = sorted((x for x in prompt.images if x["camera"] == "hand"), key=lambda x: x["time_sec"])[-n_zoom:]
            content.append({"type": "text", "text": ZOOM2_INTRO})
            info["zooms"] = []
            for j, im in enumerate(ims):
                z = make_zoom(im["source"], list(self.cfg["zoom_fixed_box"]), work / f"zoom_hand_{j}.jpg",
                              side=int(self.cfg.get("zoom_side", 512)), margin=0.0, min_frac=0.0)
                content.append({"type": "text", "text": f"\nZoom of camera hand at t = {im['time_sec']:+.2f} s (detail of Image {im['i']})\n"})
                content.append({"type": "image", "path": z["path"]})
                info["zooms"].append({"t": im["time_sec"], "path": z["path"]})
            info["zoom_path"] = info["zooms"][-1]["path"]
        refs = getattr(self, "refs", [])
        if refs:
            content.append({"type": "text", "text": FEWSHOT_INTRO.format(part=part)})
            for i, r in enumerate(refs, 1):
                lab = r["result"].upper()
                content.append({"type": "text", "text": f"\nExample {i} ({lab}): last wrist-camera image\n"})
                content.append({"type": "image", "path": r["image"]})
                content.append({"type": "text", "text": f"\nExample {i} ({lab}): enlarged crop of that image\n"})
                content.append({"type": "image", "path": r["crop"]})
            info["fewshot_refs"] = [{k: r[k] for k in ("result", "segment_id", "time_sec")} for r in refs]
        content.append({"type": "text", "text": FINAL})
        return content, info

    def verify_content(self, prompt: Prompt, reason_field: str, info: dict[str, Any]) -> list[dict[str, Any]] | None:
        part = (prompt.task or {}).get("part") or "part"
        fl = wrist_first_last(prompt)
        if not fl or not info.get("zoom_path"):
            return None
        work = Path(info["zoom_path"]).parent
        p = side_by_side(fl[0]["source"], fl[1]["source"], work / "hand_first_last.jpg",
                         side=int(self.cfg.get("contrast_side", 1024)))
        text = self.verify_text.format(part=part, reason=REASON_TEXT[reason_field].format(part=part),
                                       t0=fl[0]["time_sec"], t1=fl[1]["time_sec"], IMAGES="{IMAGES}").strip()
        parts = text.split("{IMAGES}")
        if len(parts) != 2:
            raise VLMConfigError("verify_template 要有一個 {IMAGES} 標記（影像放的位置）")
        return [{"type": "text", "text": parts[0].rstrip() + "\n"},
                {"type": "text", "text": "\nSide-by-side view (LEFT = first, RIGHT = last):\n"},
                {"type": "image", "path": p},
                {"type": "text", "text": "\nEnlarged crop of the last wrist-camera image:\n"},
                {"type": "image", "path": info["zoom_path"]},
                {"type": "text", "text": "\n" + parts[1].strip()}]

    def second_content(self, prompt: Prompt, info: dict[str, Any]) -> list[dict[str, Any]] | None:
        """第二次觀察：只有手腕相機（第一張｜最後一張並排＋裁切），不告訴模型第一次的答案。"""
        part = (prompt.task or {}).get("part") or "part"
        fl = wrist_first_last(prompt)
        if not fl or not info.get("zooms"):
            return None
        work = Path(info["zoom_path"]).parent
        p = side_by_side(fl[0]["source"], fl[1]["source"], work / "hand_first_last.jpg",
                         side=int(self.cfg.get("contrast_side", 1024)))
        text = self.second_text.format(part=part, t0=fl[0]["time_sec"], t1=fl[1]["time_sec"], IMAGES="{IMAGES}").strip()
        parts = text.split("{IMAGES}")
        if len(parts) != 2:
            raise VLMConfigError("second_view 提示詞要有一個 {IMAGES} 標記（影像放的位置）")
        out = [{"type": "text", "text": parts[0].rstrip() + "\n"},
               {"type": "text", "text": "\nSide-by-side view (LEFT = first, RIGHT = last):\n"},
               {"type": "image", "path": p}]
        for z in info["zooms"]:
            out.append({"type": "text", "text": f"\nEnlarged crop of the wrist-camera image at t = {z['t']:+.2f} s:\n"})
            out.append({"type": "image", "path": z["path"]})
        out.append({"type": "text", "text": "\n" + parts[1].strip()})
        return out

    # ------------------------------------------------------------ 強制作答

    def forced(self, content: list[dict[str, Any]], pieces) -> tuple[dict[str, dict[str, float]], RawOutput]:
        """一次前向計算：題目＋固定答案，讀每個欄位兩個候選答案的 log 機率。"""
        torch = self.torch
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                    add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                                    **getattr(self, "template_kwargs", {}))
        tok = self.processor.tokenizer
        n_prompt = int(inputs["input_ids"].shape[1])
        piece_ids = [tok(t, add_special_tokens=False)["input_ids"] for t, _, _ in pieces]
        ans_ids, probes = probe_positions(n_prompt, piece_ids, pieces)
        L = n_prompt + len(ans_ids)
        if self.max_positions and L >= self.max_positions:
            raise ValueError(f"輸入 {L} tokens 超過模型上限 {self.max_positions}")
        extra_ids = torch.tensor([ans_ids], dtype=inputs["input_ids"].dtype)
        for key in list(inputs.keys()):
            v = inputs[key]
            if key == "input_ids":
                inputs[key] = torch.cat([v, extra_ids], dim=1)
            elif key == "attention_mask":
                inputs[key] = torch.cat([v, torch.ones_like(extra_ids, dtype=v.dtype)], dim=1)
            elif hasattr(v, "shape") and v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == n_prompt:
                inputs[key] = torch.cat([v, torch.zeros((1, len(ans_ids)), dtype=v.dtype)], dim=1)
        inputs = inputs.to(self.model.device, dtype=self.dtype)
        keep = L - min(p for p, _, _ in probes)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            try:
                logits = self.model(**inputs, use_cache=False, logits_to_keep=keep).logits[0, -keep:]
            except TypeError:  # 舊版 transformers 沒有 logits_to_keep：自己算最後幾格
                hs = self.model.model(**inputs, use_cache=False).last_hidden_state
                logits = self.model.lm_head(hs[:, -keep:])[0]
            lp = torch.log_softmax(logits.float(), dim=-1)
        dt = time.perf_counter() - t0
        out: dict[str, dict[str, float]] = {}
        for pos, fld, cands in probes:
            row = lp[pos - (L - keep)]
            a, b = (float(row[self.cand[c]]) for c in cands)
            out[fld] = {"p": two_way(a, b), "mass": math.exp(a) + math.exp(b), cands[0]: a, cands[1]: b}
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        return out, RawOutput(text="", latency_s=dt, input_tokens=L, output_tokens=0, peak_vram_mb=peak)

    # ------------------------------------------------------------ 預測

    def predict(self, prompt: Prompt, max_time: float | None = None) -> RawOutput:
        t0 = time.perf_counter()
        content, info = self.build_content(prompt)
        sc, r1 = self.forced(content, checklist_pieces())
        p_yes = {c: sc[c]["p"] for c in CHECKS}
        d = self.cfg["decision"]
        n_in, peaks, lat = r1.input_tokens or 0, [r1.peak_vram_mb], [round(r1.latency_s, 3)]
        p2, second = None, None
        if self.cfg.get("second_view"):
            wc = self.second_content(prompt, info)
            if wc is not None:
                s2, r2 = self.forced(wc, wrist_pieces())
                p2 = {c: s2[c]["p"] for c in WRIST_CHECKS}
                second = {"p_yes": {c: round(p2[c], 6) for c in WRIST_CHECKS}, "p_result_failure": round(s2["result"]["p"], 6),
                          "mass": {k: round(v["mass"], 4) for k, v in s2.items()}, "fail_score": round(fail_score(p2), 6)}
                lat.append(round(r2.latency_s, 3))
                n_in += r2.input_tokens or 0
                peaks.append(r2.peak_vram_mb)
        rule = d["rule"]
        if rule in ("both_mean", "both_and", "second_any") and p2 is None:
            rule = "combined" if rule == "both_mean" else "any"  # 沒有手腕影格（很少見）：只用第一次觀察
        result = decide(p_yes, rule, float(d["threshold"]), p2)
        rec: dict[str, Any] = {"mode": self.cfg["reason"], "rule": d["rule"], "rule_used": rule, "threshold": float(d["threshold"]),
                               "p_yes": {c: round(p_yes[c], 6) for c in CHECKS},
                               "p_result_failure": round(sc["result"]["p"], 6),
                               "mass": {k: round(v["mass"], 4) for k, v in sc.items()},
                               "fail_score": round(fail_score(p_yes), 6), "stage1_result": result,
                               "forward_latency_s": lat, **{k: v for k, v in info.items() if k != "zooms"}}
        if second is not None:
            rec["second"] = second
        if self.cfg["reason"] == "verify" and result == "failure":
            fld = max(CHECKS, key=lambda c: p_yes[c])
            vc = self.verify_content(prompt, fld, info)
            if vc is not None:
                sv, r2 = self.forced(vc, verify_pieces())
                pc = sv["first_check_correct"]["p"]
                result = "failure" if pc > 0.5 else "success"
                rec["verify"] = {"reason_field": fld, "p_correct": round(pc, 6),
                                 "mass": round(sv["first_check_correct"]["mass"], 4), "result": result}
                rec["forward_latency_s"].append(round(r2.latency_s, 3))
                n_in += r2.input_tokens or 0
                peaks.append(r2.peak_vram_mb)
        rec["result"] = result
        peaks = [p for p in peaks if p is not None]
        return RawOutput(text=answer_text(p_yes, result), latency_s=time.perf_counter() - t0, input_tokens=n_in,
                         output_tokens=0, peak_vram_mb=max(peaks) if peaks else None, extra={"reason": rec})

    def repair(self, raw_text: str, max_time: float | None = None) -> RawOutput:
        return RawOutput(text=raw_text, latency_s=0.0)  # 答案是程式寫的 JSON，不會需要修正

    def metadata(self) -> dict[str, Any]:
        md = super().metadata()
        md["reason_candidates"] = getattr(self, "cand", None)
        if getattr(self, "verify_text", None):
            md["verify_sha256"] = self.verify_sha256
        if getattr(self, "second_text", None):
            md["second_view_sha256"] = self.second_sha256
        if getattr(self, "refs", None):
            md["fewshot_refs"] = [{k: r[k] for k in ("result", "segment_id", "time_sec")} for r in self.refs]
        return md
