"""第 6 包：測試集最終評估（只跑一次）。

流程：在開發集比完（scripts.evaluate）→ 把選定的方法寫進 configs/final_test.yaml、frozen 改成 true →
run_final_test.bat：擷取測試段影格 → 每個方法在測試集跑一次 → 評分（evaluate.py 的指標與報告格式）。

凍結鎖：第一次執行時，把 final_test.yaml、模型設定、每個方法引用的實驗設定、提示詞範本、做法 A 設定的 SHA-256
寫進 <out_dir>/<out_name>/lock.json。之後任何一個檔案內容改變就拒絕執行（FinalLockError），
避免看了測試結果再回頭調整。同一份設定可以續跑（已完成的段會略過）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .util import now_iso

log = logging.getLogger(__name__)
TOOL = "final_test/1"
DEFAULT_EXPERIMENT = "configs/experiment.yaml"


class FinalConfigError(ValueError):
    pass


class FinalLockError(PermissionError):
    pass


# ---------------------------------------------------------------- 設定


def load_final_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FinalConfigError(f"找不到 {p}")
    d = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("final") or {}
    miss = [k for k in ("split_dir", "out_name", "methods") if k not in d]
    if miss:
        raise FinalConfigError(f"{p} 缺少 final.{miss}")
    if not d["methods"]:
        raise FinalConfigError(f"{p}：methods 是空的")
    ids = [m.get("id") for m in d["methods"]]
    if not all(ids) or len(set(ids)) != len(ids):
        raise FinalConfigError(f"{p}：每個方法都要有不重複的 id（目前 {ids}）")
    need = {"majority": (), "vlm": ("experiment", "model", "frames"), "probe": ("config", "frames"),
            "cascade": ("first", "second")}
    for m in d["methods"]:
        if m.get("kind") not in need:
            raise FinalConfigError(f"{p}：方法 {m.get('id')} 的 kind 必須是 majority、vlm、probe 或 cascade")
        lack = [k for k in need[m["kind"]] if k not in m]
        if lack:
            raise FinalConfigError(f"{p}：方法 {m['id']} 缺少 {lack}")
        if m["kind"] == "majority":
            m.setdefault("experiment", DEFAULT_EXPERIMENT)
            m.setdefault("model", "majority")
            m["frames"] = 0
        elif m["kind"] == "cascade":  # 第 10 包：兩階段（第一階段沒答成功才用第二階段的答案），由其他方法的輸出組成
            kinds = {x.get("id"): x.get("kind") for x in d["methods"]}
            for k in ("first", "second"):
                if kinds.get(m[k]) not in ("vlm", "probe"):
                    raise FinalConfigError(f"{p}：方法 {m['id']} 的 {k}={m[k]} 必須是清單裡的 vlm 或 probe 方法")
            if m["first"] == m["second"]:
                raise FinalConfigError(f"{p}：方法 {m['id']} 的 first 和 second 不能相同")
            m.setdefault("model", "cascade")
            m["frames"] = 0
        elif not isinstance(m["frames"], int) or m["frames"] < (0 if m["kind"] == "vlm" else 1):
            raise FinalConfigError(f"{p}：方法 {m['id']} 的 frames 不合法：{m['frames']}")
        for k in ("temporal", "final_experiment_name"):
            if k in m and m["kind"] != "probe":
                raise FinalConfigError(f"{p}：方法 {m['id']}：{k} 只用在 probe")
    if d.get("primary") is not None and d["primary"] not in ids:
        raise FinalConfigError(f"{p}：primary={d['primary']} 不在方法清單 {ids} 裡")
    d.setdefault("frozen", False)
    d.setdefault("out_dir", "experiments")
    d.setdefault("models", "configs/model.yaml")
    d.setdefault("title", "測試集最終評估")
    d.setdefault("part", None)  # 第 8 包：測試集是哪個零件（擷取影格用）；沒寫就用 field_mapping 的 target_part（BNC）
    d["_path"] = str(p)
    return d


def _probe_cfg(m: dict[str, Any]) -> dict[str, Any]:
    """做法 A 的設定。方法可以用 final_experiment_name 改測試集輸出的資料夾（第 7 包重跑第 6 包的做法 A 時，
    不覆蓋第 6 包的結果）；沒寫就用設定檔裡的。"""
    from . import probe

    pc = probe.load_probe_config(m["config"])
    if m.get("final_experiment_name"):
        pc["final_experiment_name"] = str(m["final_experiment_name"])
    return pc


def _experiment(m: dict[str, Any]) -> dict[str, Any]:
    from .vlm import load_experiment

    return load_experiment(m["experiment"])


def experiment_for_test(cfg: dict[str, Any], m: dict[str, Any]) -> dict[str, Any]:
    """方法在測試集用的實驗設定：提示詞、取樣、任務文字與開發集相同，只換輸出位置與集合。

    測試集一律是凍結設定的 split_dir（第 8 包修正：majority 沒寫 experiment 時用預設的 experiment.yaml，
    它的切分是 BNC；USB 測試因此答的是 BNC 的段。VLM 方法的實驗設定和凍結設定用不同切分則直接拒絕）。"""
    exp = _experiment(m)
    if m["kind"] != "majority" and Path(exp["split_dir"]) != Path(cfg["split_dir"]):
        raise FinalConfigError(f"方法 {m['id']} 的實驗設定用切分 {exp['split_dir']}，凍結設定的測試集是 {cfg['split_dir']}")
    return dict(exp, name=f"{cfg['out_name']}/{m['id']}", include=["test"], out_dir=cfg["out_dir"],
                split_dir=cfg["split_dir"])


def model_files(cfg: dict[str, Any]) -> list[str]:
    """models 可以是一個檔案或多個檔案（第 7 包：model.yaml＋model_train.yaml）。"""
    v = cfg["models"]
    return [str(v)] if isinstance(v, (str, Path)) else [str(x) for x in v]


def method_run_dir(cfg: dict[str, Any], m: dict[str, Any]) -> Path:
    if m["kind"] == "cascade":
        return Path(cfg["out_dir"]) / cfg["out_name"] / m["id"] / "cascade" / "f0"
    if m["kind"] == "probe":
        from .probe import model_key

        pc = _probe_cfg(m)
        return (Path(pc["out_dir"]) / pc.get("final_experiment_name", "final_test") / model_key(pc, m.get("temporal", "concat"))
                / f"f{m['frames']}")
    return Path(cfg["out_dir"]) / cfg["out_name"] / m["id"] / m["model"] / f"f{m['frames']}"


def needed_extractions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """測試段需要的影格：依「最後一張在段結束後幾秒」分組。"""
    groups: dict[float, set[int]] = {}
    for m in cfg["methods"]:
        if m["frames"] == 0:
            continue
        tpl = _probe_cfg(m)["sampling_template"] if m["kind"] == "probe" else _experiment(m)["sampling_template"]
        mm = re.search(r"_eb([0-9.]+)_", tpl)
        if not mm:
            raise FinalConfigError(f"方法 {m['id']} 的取樣名稱看不出最後一張的時間：{tpl}")
        groups.setdefault(float(mm.group(1)), set()).add(int(m["frames"]))
    return [{"end_buffer": eb, "frames": sorted(fr)} for eb, fr in sorted(groups.items())]


# ---------------------------------------------------------------- 凍結鎖


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def locked_files(cfg: dict[str, Any]) -> list[str]:
    """凍結鎖要檢查的檔案：設定本身、模型設定、每個方法的實驗設定與提示詞、做法 A 設定，
    以及 LoRA adapter（第 7 包：微調後的權重也不能在凍結後換掉）。"""
    files = [cfg["_path"], *model_files(cfg)]
    adapters: dict[str, str] = {}
    templates: dict[str, list[str]] = {}
    if any(m["kind"] == "vlm" for m in cfg["methods"]):
        from .vlm import load_model_configs

        mc = load_model_configs(model_files(cfg))
        adapters = {k: v["adapter_path"] for k, v in mc.items() if v.get("adapter_path")}
        # 第 11 包：再檢查一次／第二次觀察的提示詞（只有用到的模型才有，舊的凍結設定不受影響）
        templates = {k: [v[x] for x in ("verify_template", "second_view") if v.get(x)] for k, v in mc.items()}
    for m in cfg["methods"]:
        if m["kind"] == "probe":
            files.append(m["config"])
        elif m["kind"] == "cascade":
            continue  # 只組合其他方法的輸出，沒有自己的設定檔
        else:
            files.append(m["experiment"])
            files.append(_experiment(m)["prompt"]["template"])
            files += templates.get(m["model"], [])
            if m["model"] in adapters:
                d = Path(adapters[m["model"]])
                files.append(str(d / "adapter_config.json"))
                files += [str(f) for f in sorted(d.glob("adapter_model.*"))] or [str(d / "adapter_model.safetensors")]
    out: list[str] = []
    for f in files:
        f = str(f)
        if f not in out:
            out.append(f)
    return out


def lock_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["out_dir"]) / cfg["out_name"] / "lock.json"


def check_or_write_lock(cfg: dict[str, Any]) -> dict[str, Any]:
    """沒凍結 → 拒絕；第一次 → 寫 lock.json；之後設定檔內容不同 → 拒絕。"""
    if not cfg.get("frozen"):
        raise FinalLockError("configs/final_test.yaml 還沒凍結（frozen: false）：先在開發集選定方法再改成 frozen: true")
    files = {}
    for f in locked_files(cfg):
        p = Path(f)
        if not p.exists():
            raise FinalConfigError(f"凍結的設定檔不存在：{f}")
        files[f] = _sha256(p)
    lp = lock_path(cfg)
    if lp.exists():
        old = json.loads(lp.read_text(encoding="utf-8"))
        changed = sorted(k for k in set(old["files"]) | set(files) if old["files"].get(k) != files.get(k))
        if changed:
            raise FinalLockError("凍結之後這些設定檔被改過：" + "、".join(changed) + "。測試集只能用事先決定的設定跑一次；"
                                 f"真的必須重跑時，請把 {lp.parent} 整個改名保留、在報告中說明原因後再執行")
        return old
    rec = {"tool": TOOL, "locked_at": now_iso(), "frozen_at": cfg.get("frozen_at"), "decided": cfg.get("decided"),
           "primary": cfg.get("primary"), "methods": cfg["methods"], "files": files}
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("已寫凍結鎖：%s（%d 個設定檔）", lp, len(files))
    return rec


# ---------------------------------------------------------------- 執行


def run_final(cfg: dict[str, Any], *, reasons: list[str]) -> list[dict[str, Any]]:
    """每個方法在測試集跑一次（已完成的段略過）。一個方法失敗不影響其他方法，最後一起回報。"""
    from . import probe
    from .vlm import load_model_configs, run_experiment

    check_or_write_lock(cfg)
    model_cfgs = load_model_configs(model_files(cfg))
    results = []
    for m in cfg["methods"]:
        if m["kind"] == "cascade":
            continue  # 其他方法都跑完後再組合
        log.info("[%s] 測試集 …", m["id"])
        try:
            if m["kind"] == "probe":
                probe.run_probe(_probe_cfg(m), frames=[m["frames"]], allow_test=True, write_summary=False,
                                temporal=[m.get("temporal", "concat")])
            else:
                run_experiment(exp=experiment_for_test(cfg, m), model_cfgs=model_cfgs, model_keys=[m["model"]],
                               frames=[m["frames"]], reasons=reasons, unlock_test=True)
            results.append({"id": m["id"], "ok": True})
        except Exception as e:  # noqa: BLE001 - 記下來，繼續下一個方法
            log.error("[%s] 失敗：%s: %s", m["id"], type(e).__name__, e)
            results.append({"id": m["id"], "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
    by_id = {m["id"]: m for m in cfg["methods"]}
    for m in cfg["methods"]:
        if m["kind"] != "cascade":
            continue
        try:
            st = build_cascade(cfg, m, by_id)
            log.info("[%s] 兩階段組合：第二階段 %d/%d 段", m["id"], st["n_escalated"], st["n_segments"])
            results.append({"id": m["id"], "ok": st["n_missing"] == 0, **st})
        except Exception as e:  # noqa: BLE001
            log.error("[%s] 失敗：%s: %s", m["id"], type(e).__name__, e)
            results.append({"id": m["id"], "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
    return results


def _latest_jsonl(p: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("segment_id"):
                out[r["segment_id"]] = r
    return out


def build_cascade(cfg: dict[str, Any], m: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """兩階段：第一階段答「成功」就用它；其他（失敗、無法判斷、無效、錯誤、缺）都改用第二階段的答案。
    每段時間＝第一階段＋（有用到時）第二階段；費用只算有用到的第二階段（API）。輸出格式和其他方法相同。"""
    from .split import load_split

    segs = load_split(cfg["split_dir"], include=("test",), allow_test=True)
    d1, d2 = method_run_dir(cfg, by_id[m["first"]]), method_run_dir(cfg, by_id[m["second"]])
    p1, p2 = _latest_jsonl(d1 / "predictions.jsonl"), _latest_jsonl(d2 / "predictions.jsonl")
    r1, r2 = _latest_jsonl(d1 / "raw.jsonl"), _latest_jsonl(d2 / "raw.jsonl")
    rc1 = json.loads((d1 / "run_config.json").read_text(encoding="utf-8")) if (d1 / "run_config.json").exists() else {}
    rc2 = json.loads((d2 / "run_config.json").read_text(encoding="utf-8")) if (d2 / "run_config.json").exists() else {}
    rd = method_run_dir(cfg, m)
    rd.mkdir(parents=True, exist_ok=True)
    preds, raws, n_esc, n_missing = [], [], 0, 0
    for s in segs:
        sid = s["segment_id"]
        a = p1.get(sid)
        esc = not (a and a.get("status") == "ok" and a.get("result") == "success")
        b = p2.get(sid) if esc else a
        if b is None:
            n_missing += 1
            b = {"segment_id": sid, "status": "error", "errors": ["stage_missing"], "warnings": [], "result": None}
        n_esc += int(esc)
        preds.append({**b, "segment_id": sid, "stage": 2 if esc else 1, "escalated": esc,
                      "first_result": (a or {}).get("result")})
        x1, x2 = r1.get(sid, {}), r2.get(sid, {})
        lat = None
        if x1.get("latency_s") is not None:
            lat = float(x1["latency_s"]) + (float(x2.get("latency_s") or 0.0) if esc else 0.0)
        cost = ((x1.get("cost_usd") or 0.0) + ((x2.get("cost_usd") or 0.0) if esc else 0.0))
        raws.append({"segment_id": sid, "model": "cascade", "escalated": esc, "latency_s": None if lat is None else round(lat, 4),
                     "cost_usd": round(cost, 6), "first_latency_s": x1.get("latency_s"),
                     "second_latency_s": x2.get("latency_s") if esc else None})
    (rd / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in preds), encoding="utf-8")
    (rd / "raw.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raws), encoding="utf-8")
    md1 = rc1.get("metadata") or {}
    rc = {"tool": TOOL, "model_key": "cascade", "model": {"type": "cascade", "first": m["first"], "second": m["second"],
                                                          "escalate_unless": "success"},
          "n_frames": rc1.get("n_frames"), "sampling_key": rc1.get("sampling_key"),
          "prompt_version": f"{rc1.get('prompt_version') or '—'} → {rc2.get('prompt_version') or '—'}",
          "split_dir": cfg["split_dir"], "first_run_dir": str(d1), "second_run_dir": str(d2),
          "metadata": {"load_s": md1.get("load_s") if md1.get("load_s") is not None else rc1.get("encoder_load_s")},
          "n_segments": len(segs), "n_escalated": n_esc, "n_missing": n_missing, "updated_at": now_iso()}
    (rd / "run_config.json").write_text(json.dumps(rc, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"n_segments": len(segs), "n_escalated": n_esc, "n_missing": n_missing}


# ---------------------------------------------------------------- 評分


def evaluate_final(cfg: dict[str, Any], gold_paths: list[str] | None = None, n_boot: int = 2000) -> dict[str, Any]:
    from . import evaluate as ev
    from .split import load_split

    segs = load_split(cfg["split_dir"], include=("test",), allow_test=True)
    gold = ev.load_gold(gold_paths if gold_paths is not None else ev.default_gold_paths())
    results, missing = [], []
    for m in cfg["methods"]:
        rd = method_run_dir(cfg, m)
        if not (rd / "predictions.jsonl").exists():
            missing.append(m["id"])
        desc = ev.describe_run(rd)
        if m["kind"] == "probe":
            desc["model"] = f"做法 A（{desc['model']}）"
        results.append({"id": m["id"], "desc": desc, **ev.evaluate_run(rd, segs, gold, n_boot=n_boot)})
    lock = json.loads(lock_path(cfg).read_text(encoding="utf-8")) if lock_path(cfg).exists() else {}
    notes = [f"設定凍結：{cfg.get('frozen_at') or '（未填）'}；凍結鎖寫入：{lock.get('locked_at') or '（無）'}。",
             f"選擇理由：{cfg.get('decided') or '（未填）'}",
             "真實答案：有正式人工標準答案的段用標準答案，其餘用資料集的成功／失敗標記。",
             "測試集只跑這一次；方法與提示詞都在看到測試結果之前決定。"]
    notes += [str(x) for x in (cfg.get("notes") or [])]
    if missing:
        notes.append("這些方法沒有測試集結果（全部算答錯）：" + "、".join(missing))
    paths = ev.write_report(results, Path(cfg["out_dir"]) / cfg["out_name"], title=cfg["title"], split_name="test",
                            primary=cfg.get("primary"), notes=notes, n_boot=n_boot)
    return {"results": results, "missing": missing, **paths}
