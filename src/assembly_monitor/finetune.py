"""第 7 包主線：Qwen3-VL-8B 的 QLoRA 微調（4-bit 基礎模型＋LoRA，RTX 3060 12 GB）。

- 訓練資料：其他零件的插入片段（probe.build_pools 的訓練池：零件 ≠ BNC、錄影不在 BNC 測試集裡），
  答案只用資料集的成功／失敗標記。BNC 的段一律不進訓練。
- 輸入和推論完全相同：同一個提示詞範本（experiment_ft.yaml）、同樣的影格、同樣縮到長邊 512 的 JPEG；
  每個樣本只把 task.part 換成該零件的名稱。
- 目標：chat template 在 assistant 回合產生的文字（{"result": ...} 加結束符號）。只對這幾個 token 算 loss；
  前面的提示詞與影像 token 不算。失敗段比較少，loss 依類別加權。
- 為了省顯存：只算答案位置的 logits（logits_to_keep）、gradient checkpointing（non-reentrant）、
  基礎模型維持 4-bit，不把視覺編碼器與輸出層轉成 float32。
- 訓練輪數事先固定；訓練前與每輪結束在 BNC 開發集用「success／failure 第一個 token 的機率」打分，只記錄不挑選。
- 輸出：models/ft/<name>/epoch<k>/ 與 final/（adapter），train_report.json、train_log.jsonl。
  推論用 configs/model_train.yaml 的 qwen3vl_8b_ft（adapter_path 指向 final）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from . import vlm
from .util import now_iso

log = logging.getLogger(__name__)
TOOL = "finetune/1"
RESULTS = ("success", "failure")


class FinetuneConfigError(ValueError):
    pass


# ---------------------------------------------------------------- 設定


def load_finetune_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FinetuneConfigError(f"找不到 {p}")
    d = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("finetune") or {}
    need = ["name", "base_model", "models", "prompt_experiment", "split_dir", "index_dir", "target_part",
            "frames_root", "n_frames", "out_dir", "lora", "train"]
    miss = [k for k in need if k not in d]
    if miss:
        raise FinetuneConfigError(f"{p} 缺少 finetune.{miss}")
    d.setdefault("action", "insert")
    d.setdefault("cache_dir", "experiments/_ft_inputs")
    d.setdefault("part_names", {})
    d.setdefault("smoke", {"n_train": 6, "n_dev": 4, "grad_accum": 2})
    lc = d["lora"]
    for k in ("r", "alpha", "target_modules"):
        if k not in lc:
            raise FinetuneConfigError(f"{p}：lora 缺少 {k}")
    tc = d["train"]
    tc.setdefault("epochs", 2)
    tc.setdefault("lr", 1e-4)
    tc.setdefault("grad_accum", 8)
    tc.setdefault("warmup_ratio", 0.1)
    tc.setdefault("weight_decay", 0.0)
    tc.setdefault("max_grad_norm", 1.0)
    tc.setdefault("class_weight", "balanced")
    tc.setdefault("max_seq_len", 6000)
    tc.setdefault("seed", 0)
    if int(tc["epochs"]) < 1 or int(tc["grad_accum"]) < 1:
        raise FinetuneConfigError(f"{p}：epochs 與 grad_accum 至少 1")
    if tc["class_weight"] not in ("balanced", "none"):
        raise FinetuneConfigError(f"{p}：class_weight 只能是 balanced 或 none")
    if not isinstance(d["n_frames"], int) or d["n_frames"] < 1:
        raise FinetuneConfigError(f"{p}：n_frames 必須是正整數")
    d["_path"] = str(p)
    return d


def part_label(part: str, names: dict[str, str]) -> str:
    """給模型看的零件名稱：設定檔有列就用；否則去掉結尾編號、底線換空白（round peg 2、round_peg_2 → round peg）。
    索引裡的零件名稱用空白（例如 "round peg 2"、"D-SUB"），影格資料夾名稱才用底線。"""
    if part in names:
        return str(names[part])
    return re.sub(r"[ _]\d+$", "", part.strip()).replace("_", " ").strip()


def part_task(task: dict[str, Any], part: str, names: dict[str, str]) -> dict[str, Any]:
    return dict(task, part=part_label(part, names))


def answer_text(result: str) -> str:
    if result not in RESULTS:
        raise ValueError(f"result 必須是 {RESULTS}")
    return json.dumps({"result": result})


def class_weights(samples: list[dict[str, Any]], mode: str = "balanced") -> dict[str, float]:
    """balanced：每類權重 = 總數 /（2 × 該類數量），兩類的總權重相同。"""
    c = Counter(s["result"] for s in samples)
    if mode == "none" or any(c[r] == 0 for r in RESULTS):
        return {r: 1.0 for r in RESULTS}
    n = sum(c.values())
    return {r: round(n / (2.0 * c[r]), 6) for r in RESULTS}


def lr_factor(step: int, total: int, warmup: int) -> float:
    """線性暖身後線性降到 0（step 從 0 起算）。"""
    if total <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def n_optimizer_steps(n_samples: int, epochs: int, grad_accum: int) -> int:
    return epochs * math.ceil(n_samples / grad_accum)


def pick_balanced(samples: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """試跑用：兩類各挑一半（不夠就補另一類），固定 seed。"""
    rng = random.Random(seed)
    by = {r: [s for s in samples if s["result"] == r] for r in RESULTS}
    for v in by.values():
        rng.shuffle(v)
    half = n // 2
    out = by["failure"][:half] + by["success"][: n - half]
    rest = by["failure"][half:] + by["success"][n - half:]
    out += rest[: max(0, n - len(out))]
    return sorted(out, key=lambda s: s["segment_id"])


def first_divergence(a: list[int], b: list[int]) -> int:
    """兩個 token 序列第一個不同的位置（用來在 success／failure 的第一個不同 token 上比機率）。"""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    raise ValueError("兩個答案的 token 完全相同或互為前綴，無法比較")


def seq_fill(key: str) -> int | None:
    """把答案 token 接在提示詞後面時，其他「每個 token 一個值」的欄位要補什麼：attention_mask 補 1，
    token type（文字＝0）補 0；input_ids 另外處理。"""
    if key == "attention_mask":
        return 1
    return 0


def answer_suffix(processor: Any, answer: str, template_kwargs: dict[str, Any] | None = None) -> str:
    """用模型自己的 chat template 算出 assistant 回合的文字（含結束符號），確保訓練目標和推論時的格式一致。"""
    kw = dict(template_kwargs or {})
    user = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    full_msgs = user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]

    def render(msgs: list[dict[str, Any]], gen: bool) -> str:
        t = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, **kw)
        return t[0] if isinstance(t, list) else t

    prompt, full = render(user, True), render(full_msgs, False)
    if not full.startswith(prompt):
        raise FinetuneConfigError("chat template 的 assistant 回合和推論時的開頭不一致，無法切出答案文字")
    rest = full[len(prompt):].rstrip()
    if answer not in rest:
        raise FinetuneConfigError(f"chat template 沒有把答案原樣放進 assistant 回合：{rest!r}")
    return rest


# ---------------------------------------------------------------- 資料


def pool_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """probe.build_pools 需要的欄位。"""
    return {k: cfg[k] for k in ("split_dir", "index_dir", "action", "target_part", "frames_root")}


def build_samples(cfg: dict[str, Any], exp: dict[str, Any]) -> dict[str, Any]:
    """train＝其他零件（有影格的段）；dev＝BNC 開發集。不讀測試集。"""
    from . import probe

    pc = pool_cfg(cfg)
    pools = probe.build_pools(pc, allow_test=False)
    sampling = exp["sampling_template"].format(n=cfg["n_frames"])
    cams = list(exp["cameras"])
    out: dict[str, Any] = {"sampling_key": sampling, "n_test_recordings_excluded": pools["n_test_recordings_excluded"]}
    for name in ("train", "dev"):
        keep, missing = [], []
        for s in pools[name]:
            if name == "train" and s["part"] == cfg["target_part"]:
                raise FinetuneConfigError("訓練池出現目標零件（不應該發生）")
            seg = vlm.load_segment_frames(probe.frames_dir(pc, s["part"]), sampling, s["segment_id"], cams)
            if seg is None or not seg.frames:
                missing.append(s["segment_id"])
                continue
            keep.append({"segment_id": s["segment_id"], "recording_id": s.get("recording_id"), "part": s["part"],
                         "result": s["dataset_result"], "seg": seg})
        out[name], out[f"{name}_missing"] = keep, missing
    return out


def sample_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter((s["part"], s["result"]) for s in samples)
    parts = sorted({s["part"] for s in samples})
    return {"n": len(samples), "n_failure": sum(1 for s in samples if s["result"] == "failure"),
            "n_recordings": len({s.get("recording_id") for s in samples}),
            "per_part": {p: {"success": c[(p, "success")], "failure": c[(p, "failure")]} for p in parts}}


def train_sha(cfg: dict[str, Any], exp: dict[str, Any], model_cfg: dict[str, Any], builder_sha: str,
              train_ids: list[str]) -> str:
    """同一份設定、同一批訓練段 → 同一個 hash（final 已存在而且 hash 相同就不重訓）。"""
    keys = ("name", "base_model", "n_frames", "target_part", "part_names", "lora", "train")
    payload = {"cfg": {k: cfg.get(k) for k in keys}, "model": model_cfg, "prompt": builder_sha,
               "sampling": exp["sampling_template"], "cameras": exp["cameras"], "train_ids": sorted(train_ids)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cached_images(cfg: dict[str, Any], sample: dict[str, Any], max_side: int | None) -> list[str]:
    """縮好的影格放在 cache_dir/<取樣>/<段>/（和推論的 prepare_images 相同），已存在就沿用。"""
    seg = sample["seg"]
    d = Path(cfg["cache_dir"]) / seg.sampling_key / f"side{max_side or 0}" / sample["segment_id"]
    paths = [str(d / f"{i:02d}_{f.camera}.jpg") for i, f in enumerate(seg.frames)]
    if all(Path(p).exists() for p in paths):
        return paths
    return vlm.prepare_images(seg.frames, max_side, d)


def describe_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    """不需要 GPU：列出訓練資料、類別權重、更新次數，以及一個樣本的提示詞（給人看）。"""
    exp = vlm.load_experiment(cfg["prompt_experiment"])
    data = build_samples(cfg, exp)
    tc = cfg["train"]
    builder = vlm.PromptBuilder.from_experiment(exp)
    ex = None
    if data["train"]:
        s = data["train"][0]
        b = vlm.PromptBuilder(builder.template, builder.version, part_task(exp["task"], s["part"], cfg["part_names"]))
        p = b.build(s["seg"])
        ex = {"segment_id": s["segment_id"], "part": s["part"], "result": s["result"], "prompt": p.text,
              "n_images": len(s["seg"].frames), "target": answer_text(s["result"])}
    return {"sampling_key": data["sampling_key"], "train": sample_summary(data["train"]),
            "dev": sample_summary(data["dev"]), "train_missing": data["train_missing"], "dev_missing": data["dev_missing"],
            "class_weights": class_weights(data["train"], tc["class_weight"]),
            "optimizer_steps": n_optimizer_steps(len(data["train"]), int(tc["epochs"]), int(tc["grad_accum"])),
            "part_labels": {p: part_label(p, cfg["part_names"]) for p in sorted({s["part"] for s in data["train"]})},
            "example": ex, "n_test_recordings_excluded": data["n_test_recordings_excluded"]}


# ---------------------------------------------------------------- GPU（只在使用者的電腦上執行）


def load_for_training(model_cfg: dict[str, Any], lora_cfg: dict[str, Any]):  # pragma: no cover - 需要 GPU
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("找不到可用的 CUDA GPU（請先執行 setup_vlm.bat）")
    if model_cfg.get("quantization") != "4bit":
        raise FinetuneConfigError("QLoRA 需要 4-bit 基礎模型（model.yaml 的 quantization: 4bit）")
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    rev = model_cfg.get("revision") or None
    processor = AutoProcessor.from_pretrained(model_cfg["hf_id"], revision=rev)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                           bnb_4bit_use_double_quant=True,
                           llm_int8_skip_modules=model_cfg.get("quant_skip_modules") or None)
    cls = getattr(transformers, model_cfg.get("model_class") or "AutoModelForImageTextToText")
    model = cls.from_pretrained(model_cfg["hf_id"], revision=rev, dtype=dtype, device_map={"": 0},
                                attn_implementation=model_cfg.get("attn_implementation", "sdpa"), quantization_config=q)
    for c in (model.config, getattr(model.config, "text_config", None)):
        if c is not None:
            c.use_cache = False
    for prm in model.parameters():
        prm.requires_grad_(False)
    # non-reentrant：輸入不必 requires_grad，視覺編碼器不會建反向傳播的計算圖
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lc = LoraConfig(r=int(lora_cfg["r"]), lora_alpha=int(lora_cfg["alpha"]), lora_dropout=float(lora_cfg.get("dropout", 0.0)),
                    bias="none", target_modules=lora_cfg["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lc)
    names = [n for n, prm in model.named_parameters() if prm.requires_grad]
    if not names:
        raise FinetuneConfigError(f"LoRA 沒有對上任何層（target_modules={lora_cfg['target_modules']}）")
    bad = [n for n in names if "visual" in n or "vision" in n]
    if bad:
        raise FinetuneConfigError(f"LoRA 加到了視覺編碼器：{bad[:3]}")
    n_train = sum(prm.numel() for prm in model.parameters() if prm.requires_grad)
    return model, processor, {"trainable_params": int(n_train), "n_lora_tensors": len(names), "dtype": str(dtype)}


def build_batch(prompt_inputs: Any, extra_ids: list[int], device: Any, dtype: Any) -> dict[str, Any]:  # pragma: no cover
    """把答案（或答案的前綴）token 接在提示詞後面；其他「每個 token 一個值」的欄位一起延長。batch 大小固定 1。"""
    import torch

    ids = prompt_inputs["input_ids"]
    L = int(ids.shape[1])
    T = len(extra_ids)
    out: dict[str, Any] = {}
    for k, v in prompt_inputs.items():
        if not torch.is_tensor(v):
            continue
        if T and v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == L:
            add = (torch.tensor([extra_ids], dtype=v.dtype) if k == "input_ids"
                   else torch.full((1, T), seq_fill(k), dtype=v.dtype))
            v = torch.cat([v, add], dim=1)
        out[k] = v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
    return out


def train(cfg: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:  # pragma: no cover - 需要 GPU 與 peft
    import torch
    import torch.nn.functional as F

    t_start = time.perf_counter()
    exp = vlm.load_experiment(cfg["prompt_experiment"])
    model_cfgs = vlm.load_model_configs(cfg["models"])
    if cfg["base_model"] not in model_cfgs:
        raise FinetuneConfigError(f"{cfg['models']} 沒有 {cfg['base_model']}")
    mcfg = model_cfgs[cfg["base_model"]]
    if mcfg.get("adapter_path"):
        raise FinetuneConfigError("base_model 不能是已經有 adapter 的模型")
    base_builder = vlm.PromptBuilder.from_experiment(exp)
    data = build_samples(cfg, exp)
    train_s, dev_s = data["train"], data["dev"]
    tc = dict(cfg["train"])
    name = cfg["name"]
    if smoke:
        sc = cfg["smoke"]
        train_s = pick_balanced(train_s, int(sc.get("n_train", 6)), int(tc["seed"]))
        dev_s = pick_balanced(dev_s, int(sc.get("n_dev", 4)), int(tc["seed"]))
        tc.update(epochs=1, grad_accum=int(sc.get("grad_accum", 2)))
        name += "_smoke"
    if len(train_s) < 2:
        raise FinetuneConfigError(f"訓練段只有 {len(train_s)} 段有影格（先執行 python -m scripts.probe extract --config configs/probe_eb05.yaml）")
    out = Path(cfg["out_dir"]) / name
    sha = train_sha(cfg, exp, mcfg, base_builder.sha256, [s["segment_id"] for s in train_s])
    rep_p = out / "final" / "train_report.json"
    if not smoke and rep_p.exists():
        old = json.loads(rep_p.read_text(encoding="utf-8"))
        if old.get("train_sha256") == sha:
            log.info("已經用相同設定訓練完成：%s（略過訓練）", rep_p.parent)
            return old
        raise FinetuneConfigError(f"{rep_p.parent} 已有不同設定的 adapter；請換 finetune.name，或把舊資料夾改名保留")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(int(tc["seed"]))
    torch.manual_seed(int(tc["seed"]))
    builders = {p: vlm.PromptBuilder(base_builder.template, base_builder.version, part_task(exp["task"], p, cfg["part_names"]))
                for p in {s["part"] for s in train_s + dev_s}}
    weights = class_weights(train_s, tc["class_weight"])
    log.info("訓練 %d 段（失敗 %d；%d 種零件），開發集 %d 段；類別權重 %s；%d 輪、每 %d 段更新一次",
             len(train_s), sum(1 for s in train_s if s["result"] == "failure"), len({s["part"] for s in train_s}),
             len(dev_s), weights, tc["epochs"], tc["grad_accum"])
    t0 = time.perf_counter()
    model, processor, minfo = load_for_training(mcfg, cfg["lora"])
    load_s = time.perf_counter() - t0
    dtype = getattr(torch, mcfg.get("dtype", "bfloat16"))
    device = torch.device("cuda:0")
    tkw = dict(mcfg.get("chat_template_kwargs") or {})
    tok = processor.tokenizer
    targets = {r: tok(answer_suffix(processor, answer_text(r), tkw), add_special_tokens=False)["input_ids"] for r in RESULTS}
    d = first_divergence(targets["success"], targets["failure"])
    choice = {"prefix": targets["success"][:d], "success": targets["success"][d], "failure": targets["failure"][d]}
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    if eos and targets["success"][-1] not in eos:
        log.warning("答案最後一個 token（%s）不在 generation_config 的結束符號 %s 裡，推論時可能不會停", targets["success"][-1], eos)
    log.info("載入 %.0f 秒，LoRA 參數 %.1f M；答案 token：success %s／failure %s", load_s, minfo["trainable_params"] / 1e6,
             targets["success"], targets["failure"])
    max_side = mcfg.get("image_max_side")

    def encode(s: dict[str, Any]) -> Any:
        paths = cached_images(cfg, s, max_side)
        prompt = builders[s["part"]].build(s["seg"], paths)
        return processor.apply_chat_template([{"role": "user", "content": prompt.content}], tokenize=True,
                                             add_generation_prompt=True, return_dict=True, return_tensors="pt", **tkw)

    keep_arg = {"ok": True}

    def forward(batch: dict[str, Any], n_keep: int) -> Any:
        with torch.autocast(device_type="cuda", dtype=dtype):
            if keep_arg["ok"]:
                try:
                    return model(**batch, logits_to_keep=n_keep, use_cache=False)
                except TypeError:
                    keep_arg["ok"] = False
                    log.warning("這版模型不支援 logits_to_keep，改算全部位置的 logits（較吃顯存）")
            return model(**batch, use_cache=False)

    def score_dev(epoch: int) -> dict[str, Any]:
        model.eval()
        rows = []
        with torch.inference_mode():
            for s in dev_s:
                try:
                    batch = build_batch(encode(s), choice["prefix"], device, dtype)
                    o = forward(batch, 1)
                    lp = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
                    pf = float(torch.sigmoid(lp[choice["failure"]] - lp[choice["success"]]))
                    rows.append({"segment_id": s["segment_id"], "result": s["result"], "prob_failure": round(pf, 4)})
                except torch.cuda.OutOfMemoryError:
                    rows.append({"segment_id": s["segment_id"], "result": s["result"], "prob_failure": None, "error": "oom"})
                    torch.cuda.empty_cache()
        model.train()
        ok = [r for r in rows if r["prob_failure"] is not None]
        fails = [r for r in ok if r["result"] == "failure"]
        succs = [r for r in ok if r["result"] == "success"]
        caught = sum(1 for r in fails if r["prob_failure"] >= 0.5)
        succ_ok = sum(1 for r in succs if r["prob_failure"] < 0.5)
        res = {"epoch": epoch, "n": len(rows), "n_scored": len(ok), "correct": caught + succ_ok,
               "failures_caught": f"{caught}/{len(fails)}", "successes_ok": f"{succ_ok}/{len(succs)}",
               "balanced_accuracy": round((caught / len(fails) + succ_ok / len(succs)) / 2, 3) if fails and succs else None,
               "rows": rows}
        log.info("  [第 %d 輪] BNC 開發集（只記錄）：答對 %d/%d，抓到失敗 %s，成功答對 %s，兩類平均 %s", epoch,
                 res["correct"], len(rows), res["failures_caught"], res["successes_ok"], res["balanced_accuracy"])
        return res

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    ga, epochs = int(tc["grad_accum"]), int(tc["epochs"])
    total = n_optimizer_steps(len(train_s), epochs, ga)
    warm = int(round(float(tc["warmup_ratio"]) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: lr_factor(st, total, warm))
    log_p = out / "train_log.jsonl"
    log_p.unlink(missing_ok=True)
    step, skipped, dev_hist, sample_times = 0, [], [], []
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt.zero_grad(set_to_none=True)
    rng = random.Random(int(tc["seed"]))
    dev_hist.append(score_dev(0))  # 訓練前（同一個提示詞的原始模型），對照用
    (out / "dev_epoch0.json").write_text(json.dumps(dev_hist[-1], ensure_ascii=False, indent=1), encoding="utf-8")
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_s)))
        rng.shuffle(order)
        acc_loss, acc_n, micro = 0.0, 0, 0
        for j, idx in enumerate(order, 1):
            s = train_s[idx]
            ts = time.perf_counter()
            oom = False
            try:
                pin = encode(s)
                L = int(pin["input_ids"].shape[1])
                tgt = targets[s["result"]]
                if L + len(tgt) > int(tc["max_seq_len"]):
                    skipped.append({"segment_id": s["segment_id"], "epoch": epoch, "reason": f"too_long:{L}"})
                else:
                    batch = build_batch(pin, tgt, device, dtype)
                    o = forward(batch, len(tgt) + 1)
                    logits = o.logits[0, -(len(tgt) + 1):-1].float()
                    loss = F.cross_entropy(logits, torch.tensor(tgt, device=device)) * weights[s["result"]]
                    (loss / ga).backward()
                    acc_loss += float(loss.detach())
                    acc_n += 1
                    micro += 1
                    sample_times.append(time.perf_counter() - ts)
                    del batch, o, logits, loss
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:  # 在 except 區塊外清顯存；這一輪累積的梯度一起丟掉
                skipped.append({"segment_id": s["segment_id"], "epoch": epoch, "reason": "oom"})
                log.warning("  %s 顯存不足，略過", s["segment_id"])
                opt.zero_grad(set_to_none=True)
                acc_loss, acc_n, micro = 0.0, 0, 0
                torch.cuda.empty_cache()
            if micro and (micro >= ga or j == len(order)):
                gn = torch.nn.utils.clip_grad_norm_(params, float(tc["max_grad_norm"]))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"epoch": epoch, "step": step, "of": total, "loss": round(acc_loss / max(1, acc_n), 5),
                       "grad_norm": round(float(gn), 4), "lr": sched.get_last_lr()[0], "samples": j,
                       "sec_per_sample": round(sum(sample_times[-acc_n:]) / max(1, acc_n), 2),
                       "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20), "time": now_iso()}
                with log_p.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                log.info("  第 %d 輪 更新 %d/%d：loss %.4f，%.1f 秒/段，顯存峰值 %d MB", epoch, step, total, rec["loss"],
                         rec["sec_per_sample"], rec["peak_vram_mb"])
                acc_loss, acc_n, micro = 0.0, 0, 0
        model.save_pretrained(str(out / f"epoch{epoch}"))
        dev_hist.append(score_dev(epoch))
        (out / f"dev_epoch{epoch}.json").write_text(json.dumps(dev_hist[-1], ensure_ascii=False, indent=1), encoding="utf-8")
    final = out / "final"
    model.save_pretrained(str(final))
    import peft
    import transformers

    spm = sum(sample_times) / len(sample_times) if sample_times else None
    rep = {"tool": TOOL, "name": name, "smoke": smoke, "train_sha256": sha, "config": {k: v for k, v in cfg.items() if k != "_path"},
           "config_path": cfg["_path"], "base_model": mcfg, "prompt_version": base_builder.version,
           "prompt_sha256": base_builder.sha256, "sampling_key": data["sampling_key"], "train": sample_summary(train_s),
           "dev": sample_summary(dev_s), "train_missing_frames": data["train_missing"], "skipped": skipped,
           "class_weights": weights, "answer_tokens": targets, "optimizer_steps": step, "model": minfo,
           "load_s": round(load_s, 1), "sec_per_sample": round(spm, 2) if spm else None,
           "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
           "train_minutes": round((time.perf_counter() - t_start) / 60, 1), "dev_each_epoch": dev_hist,
           "final_epoch": epochs, "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                               "peft": peft.__version__},
           "train_segment_ids": [s["segment_id"] for s in train_s], "finished_at": now_iso()}
    if smoke and spm:
        full = len(data["train"]) * int(cfg["train"]["epochs"])
        n_dev_passes = int(cfg["train"]["epochs"]) + 1  # 訓練前一次＋每輪一次
        rep["estimated_full_minutes"] = round((full * spm + n_dev_passes * len(data["dev"]) * spm / 2) / 60 + load_s / 60, 0)
    for p in (final / "train_report.json", out / "train_report.json"):
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("完成：%s（%.1f 分鐘）", final, rep["train_minutes"])
    return rep


__all__ = ["load_finetune_config", "part_label", "part_task", "answer_text", "class_weights", "lr_factor",
           "n_optimizer_steps", "pick_balanced", "first_divergence", "answer_suffix", "build_samples", "describe_plan",
           "train", "train_sha"]
