"""第 17 包：REBOOT M12 影格先用 OpenCV 調亮度／對比／顏色，再套 bolt 4 模型（模型、門檻 0.5 都不改）。

- L0 不調：第 16 包的原始影格（重算一次，應和第 16 包一樣）。
- L1 整批調整（主要）：M12 全部 65 格合起來的 LAB 統計 → 對齊 bolt 4 訓練 83 格；65 格同一條公式。
- L2 每張各自調整（次要）：每一格自己的 LAB 統計 → 對齊 bolt 4。
統計只算橘框內（中間 60%，模型看的範圍）；目標只用 bolt 4 訓練影格的像素，不用答案。
說明：docs/package17_m12_light.md；輸出：experiments/p17_m12_light/
"""

from __future__ import annotations

import glob
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

P16 = Path("experiments/p16_m12")
OUT = Path("experiments/p17_m12_light")
FRAMES = Path("data/derived/frames/reboot_m12_light")
MODEL = "models/perpart_final/bolt_4"
B4_FRAMES = Path("data/derived/frames/insert_bolt_4/uniform_n1_sb0.5_eb1_jpg/segments")
BOX = (0.2, 0.2, 0.8, 0.8)
VARIANTS = {"L0": "不調（第 16 包原始影格）", "L1": "整批調整（主要）", "L2": "每張各自調整（次要）"}
SAME_TOL = 1e-4


class Stop(RuntimeError):
    """缺檔或檢查沒過，不計分。"""


# ---------------------------------------------------------------- 影像調整（純函式，可單獨測試）


def path_of(p: str | Path) -> Path:
    """apply.json 在 Windows 寫的是反斜線；統一成 Path。"""
    return Path(str(p).replace("\\", "/"))


def crop(img: np.ndarray, box: tuple[float, float, float, float] = BOX) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]


def to_lab(img_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)


def lab_stats(labs: list[np.ndarray]) -> dict[str, float]:
    """多張（已裁切的）LAB 影像合起來的像素統計：L 平均、L 標準差、a 平均、b 平均（OpenCV 8 位元刻度）。"""
    if not labs:
        raise ValueError("沒有影像")
    px = np.concatenate([np.asarray(x, np.float64).reshape(-1, 3) for x in labs])
    return {"L_mean": float(px[:, 0].mean()), "L_std": float(px[:, 0].std()),
            "a_mean": float(px[:, 1].mean()), "b_mean": float(px[:, 2].mean())}


def adjust(img_bgr: np.ndarray, src: dict[str, float], tgt: dict[str, float]) -> np.ndarray:
    """L' = (L − 來源平均) / 來源標準差 × 目標標準差 + 目標平均；a、b 平移；截到 0–255。整張影格同一條公式。"""
    import cv2

    lab = to_lab(img_bgr)
    lab[..., 0] = (lab[..., 0] - src["L_mean"]) / max(src["L_std"], 1e-6) * tgt["L_std"] + tgt["L_mean"]
    lab[..., 1] += tgt["a_mean"] - src["a_mean"]
    lab[..., 2] += tgt["b_mean"] - src["b_mean"]
    lab = np.clip(np.rint(lab), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_stats(img_bgr: np.ndarray) -> dict[str, float]:
    return lab_stats([to_lab(crop(img_bgr))])


def adjust_all(imgs: list[np.ndarray], tgt: dict[str, float]) -> dict[str, Any]:
    """回傳 L1（整批）和 L2（每張）調整後的影像，以及 L1 用的來源統計。"""
    labs = [to_lab(crop(im)) for im in imgs]
    src_all = lab_stats(labs)
    l1 = [adjust(im, src_all, tgt) for im in imgs]
    l2 = [adjust(im, lab_stats([lab]), tgt) for im, lab in zip(imgs, labs)]
    return {"src_all": src_all, "L1": l1, "L2": l2}


# ---------------------------------------------------------------- 資料


def bolt4_frames(model_stem: str = MODEL, root: Path = B4_FRAMES) -> list[Path]:
    meta = json.loads(Path(f"{model_stem}.json").read_text(encoding="utf-8"))
    out = []
    for sid in meta["segment_ids"]:
        g = sorted(glob.glob(str(root / sid / "hand__00__*.jpg")))
        if not g:
            raise Stop(f"bolt 4 訓練影格不見了：{sid}")
        out.append(Path(g[0]))
    return out


def load_rows(p16: Path = P16) -> list[dict[str, Any]]:
    f = Path(p16) / "apply.json"
    if not f.exists():
        raise Stop("找不到第 16 包的 apply.json（要先跑 run_m12_apply.bat）")
    rows = json.loads(f.read_text(encoding="utf-8"))["rows"]
    if not rows:
        raise Stop("第 16 包沒有可計分的段")
    for r in rows:
        if not path_of(r["image"]).exists():
            raise Stop(f"第 16 包的影格不見了：{r['image']}")
    return rows


def _imread(p: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(p))
    if img is None:
        raise Stop(f"影格讀不到：{p}")
    return img


def _imwrite(p: Path, img: np.ndarray) -> None:
    import cv2

    p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise Stop(f"寫不出影格：{p}")


def summarize(rows: list[dict[str, Any]], imgs: list[np.ndarray]) -> dict[str, dict[str, float]]:
    """橘框內亮度、對比的平均（分成功、失敗），寫進報告。"""
    out = {}
    for name, lab in (("成功段", 0), ("失敗段", 1)):
        st = [crop_stats(im) for r, im in zip(rows, imgs) if int(r["label"]) == lab]
        if st:
            out[name] = {"亮度": float(np.mean([s["L_mean"] for s in st])), "對比": float(np.mean([s["L_std"] for s in st]))}
    return out


# ---------------------------------------------------------------- 主流程


def run(p16: Path = P16, out: Path = OUT, frames_root: Path = FRAMES, model_stem: str = MODEL,
        encoder: Any | None = None) -> dict[str, Any]:
    from . import perpart as pp
    from . import perpart_final as pf
    from . import reboot_m12 as rm

    t0 = time.time()
    out, frames_root = Path(out), Path(frames_root)
    rows = load_rows(p16)
    model = pf.load_model(model_stem)
    b4 = [_imread(p) for p in bolt4_frames(model_stem)]
    tgt = lab_stats([to_lab(crop(im)) for im in b4])
    imgs = [_imread(path_of(r["image"])) for r in rows]
    adj = adjust_all(imgs, tgt)
    paths: dict[str, list[Path]] = {"L0": [path_of(r["image"]) for r in rows]}
    images: dict[str, list[np.ndarray]] = {"L0": imgs, "L1": adj["L1"], "L2": adj["L2"]}
    for v in ("L1", "L2"):
        paths[v] = []
        for r, im in zip(rows, images[v]):
            p = frames_root / v / r["kind"] / path_of(r["image"]).name
            _imwrite(p, im)
            paths[v].append(p)
    own = encoder is None
    enc = pp.make_encoders()["crop"] if own else encoder
    res_v: dict[str, Any] = {}
    try:
        for v in VARIANTS:
            vr = []
            for r, p in zip(rows, paths[v]):
                pfail = float(pf.predict_proba(model, pf.image_vector(enc, str(p))[None])[0])
                vr.append({"kind": r["kind"], "episode": r["episode"], "label": int(r["label"]), "image": str(p),
                           "p_failure": pfail, "pred": "失敗" if pfail > 0.5 else "成功"})
            res_v[v] = {"name": VARIANTS[v], "score": rm.score(vr), "rows": vr, "stats": summarize(rows, images[v])}
            s = res_v[v]["score"]
            log.info("%s %s：失敗段 %d/%d、成功段 %d/%d、兩類平均 %.0f%%、AUC %s", v, VARIANTS[v], s["ok_failure"],
                     s["n_failure"], s["ok_success"], s["n_success"], s["bal"] * 100,
                     "–" if s["auc"] is None else f"{s['auc']:.2f}")
    finally:
        if own:
            enc.close()
    diff = max(abs(a["p_failure"] - float(r["p_failure"])) for a, r in zip(res_v["L0"]["rows"], rows))
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": model_stem, "target_bolt4": tgt,
           "source_m12_all": adj["src_all"], "bolt4_n": len(b4), "l0_max_diff_vs_p16": diff,
           "l0_same_as_p16": bool(diff <= SAME_TOL), "variants": res_v, "seconds": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_sheet(rows, images, res_v, b4, out / "before_after.jpg")
    write_report(res, out / "report.md")
    return res


def _tile(img: np.ndarray, text: str, w: int = 256, h: int = 192) -> np.ndarray:
    import cv2

    t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    x0, y0, x1, y1 = int(BOX[0] * w), int(BOX[1] * h), int(BOX[2] * w), int(BOX[3] * h)
    cv2.rectangle(t, (x0, y0), (x1, y1), (46, 122, 224), 2)
    bar = np.full((24, w, 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([t, bar])


def _write_sheet(rows: list[dict[str, Any]], images: dict[str, list[np.ndarray]], res_v: dict[str, Any],
                 b4: list[np.ndarray], path: Path, n_each: int = 4) -> None:
    """每列一段：L0／L1／L2 並排，下方寫失敗機率；最上面一列放 2 張 bolt 4 訓練影格當目標參考。"""
    import cv2

    order = sorted(range(len(rows)), key=lambda i: (-int(rows[i]["label"]), rows[i]["episode"]))
    pick = [i for i in order if int(rows[i]["label"]) == 1][:n_each] + [i for i in order if int(rows[i]["label"]) == 0][:n_each]
    blank = np.full((216, 256, 3), 255, np.uint8)
    ref = np.hstack([_tile(b4[0], "bolt 4 train (target look)"), _tile(b4[len(b4) // 2], "bolt 4 train (target look)"), blank])
    lines = [ref]
    for i in pick:
        tag = f"{'F' if int(rows[i]['label']) else 'S'} ep{rows[i]['episode']}"
        lines.append(np.hstack([_tile(images[v][i], f"{tag} {v} p={res_v[v]['rows'][i]['p_failure']:.2f}")
                                for v in VARIANTS]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 90])


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def write_report(res: dict[str, Any], path: Path) -> None:
    t, s0 = res["target_bolt4"], res["source_m12_all"]
    L = ["# 第 17 包：M12 影格先用 OpenCV 調光線，再套 bolt 4 模型", "",
         f"產生時間：{res['generated']}；模型 `{res['model']}`（沒有重新訓練、門檻 0.5 不改）。事先登記：`docs/package17_m12_light.md`。", "",
         "## 結果", "", "| 做法 | 失敗段答對 | 成功段答對 | 兩類平均 | AUC | 失敗機率中位數（失敗段／成功段） |",
         "|---|---|---|---|---|---|"]
    for v, d in res["variants"].items():
        s = d["score"]
        m = s["median_p_failure"]
        L.append(f"| {v} {d['name']} | {s['ok_failure']}/{s['n_failure']} | {s['ok_success']}/{s['n_success']} | "
                 f"{_pct(s['bal'])} | {'–' if s['auc'] is None else format(s['auc'], '.2f')} | "
                 f"{m['失敗段']:.3f}／{m['成功段']:.3f} |")
    L += ["| 一律答成功（基準） | 0 | 全部 | 50% | 0.50 | |", "",
          f"- L0 和第 16 包的失敗機率最大差 {res['l0_max_diff_vs_p16']:.6f} → {'一樣（流程沒問題）' if res['l0_same_as_p16'] else '不一樣，要查'}",
          "- 判定（事先寫好）：L1 兩類平均 ≥ 65% 而且 AUC ≥ 0.65，才說調光線有明顯幫助。",
          "- 失敗段只有 5 段：1 段答對或答錯，兩類平均就差 10 點，只能當參考。", "",
          "## 調整前後的亮度、對比（橘框內，OpenCV LAB 的 L，0–255）", "",
          f"- 目標（bolt 4 訓練 {res['bolt4_n']} 格合起來）：亮度 {t['L_mean']:.0f}、對比 {t['L_std']:.0f}、a {t['a_mean'] - 128:+.1f}、b {t['b_mean'] - 128:+.1f}",
          f"- M12 原本（65 格合起來）：亮度 {s0['L_mean']:.0f}、對比 {s0['L_std']:.0f}、a {s0['a_mean'] - 128:+.1f}、b {s0['b_mean'] - 128:+.1f}", "",
          "| 做法 | 成功段亮度 | 成功段對比 | 失敗段亮度 | 失敗段對比 |", "|---|---|---|---|---|"]
    for v, d in res["variants"].items():
        st = d["stats"]
        L.append(f"| {v} | {st['成功段']['亮度']:.0f} | {st['成功段']['對比']:.0f} | {st['失敗段']['亮度']:.0f} | {st['失敗段']['對比']:.0f} |")
    L += ["", "## 每段失敗機率", "", "| 段 | 答案 | L0 | L1 | L2 |", "|---|---|---|---|---|"]
    r0, r1, r2 = (res["variants"][v]["rows"] for v in ("L0", "L1", "L2"))
    for a, b, c in sorted(zip(r0, r1, r2), key=lambda x: (-x[0]["label"], x[0]["episode"])):
        L.append(f"| {'失敗' if a['label'] else '成功'} ep{a['episode']} | {'失敗' if a['label'] else '成功'} | "
                 f"{a['p_failure']:.3f} | {b['p_failure']:.3f} | {c['p_failure']:.3f} |")
    L += ["", "## 圖", "", "- `before_after.jpg`：最上面是 bolt 4 訓練影格（目標的樣子）；下面每列一段，左到右 L0／L1／L2，"
          "F＝失敗段、S＝成功段，p＝失敗機率，橘框＝模型看的範圍。", ""]
    path.write_text("\n".join(L), encoding="utf-8")
