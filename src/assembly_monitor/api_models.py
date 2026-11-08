"""第 8 包：商業模型（Claude、GPT）透過 API 判斷成敗。

和本地 VLM 用同一個提示詞、同一批影格（同樣縮到長邊 512 的 JPEG）、同一個解析器，只換「誰回答」。
- API key 只從環境變數或專案根目錄的 .env 讀（ANTHROPIC_API_KEY、OPENAI_API_KEY）；.env 不進 Git、程式不印出 key。
- 每段記下延遲（從送出到收到回答，含網路）、輸入／輸出 token 與估計費用（設定檔的單價 × token）。
- key 錯或沒有權限時丟 FatalAdapterError：整個模型跳過，不會對每一段重試一次。
- 官方 SDK（anthropic、openai）只在 load() 才匯入；沒裝 SDK 的環境也能跑其他模型與測試。
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

from .vlm import REPAIR_INSTRUCTION, FatalAdapterError, ModelAdapter, Prompt, RawOutput, VLMConfigError

log = logging.getLogger(__name__)

ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def load_env_file(path: str | Path = ".env") -> list[str]:
    """讀 KEY=VALUE 格式的 .env（# 開頭是註解）。已經在環境變數裡的不覆蓋。回傳讀到的 key 名稱（不含值）。"""
    p = Path(path)
    if not p.exists():
        return []
    names = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.startswith("export "):
            k = k[7:].strip()
        if not k or not v:
            continue
        names.append(k)
        os.environ.setdefault(k, v)
    return names


def image_b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def media_type(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    return {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/jpeg")


def estimate_cost(cfg: dict[str, Any], n_in: int | None, n_out: int | None) -> float | None:
    """設定檔的單價（美元／百萬 token）× token 數。沒有單價就不估。"""
    pi, po = cfg.get("price_in_per_mtok"), cfg.get("price_out_per_mtok")
    if pi is None or po is None or n_in is None or n_out is None:
        return None
    return round((n_in * float(pi) + n_out * float(po)) / 1e6, 6)


class ApiAdapter(ModelAdapter):
    provider = ""

    def __init__(self, key: str, cfg: dict[str, Any]):
        super().__init__(key, cfg)
        self.client: Any = None
        self.sdk_version: str | None = None

    # ---- 子類別實作
    def _make_client(self) -> Any:  # pragma: no cover - 需要 SDK
        raise NotImplementedError

    def _convert(self, prompt_content: list[dict[str, Any]]) -> list[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def _call(self, content: list[dict[str, Any]]) -> tuple[str, int | None, int | None, dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def _is_fatal(self, e: BaseException) -> bool:
        name = type(e).__name__
        return name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError") or (
            name == "BadRequestError" and "model" in str(e).lower() and "not" in str(e).lower())

    # ---- 共用
    def load(self) -> None:
        load_env_file(self.cfg.get("env_file", ".env"))
        env = ENV_KEYS[self.provider]
        if not os.environ.get(env):
            raise VLMConfigError(f"沒有 {env}：請先執行 setup_api.bat，在專案資料夾的 .env 填入 API key")
        self.client = self._make_client()
        log.info("  %s API：%s（key 已讀取，不顯示）", self.provider, self.cfg["model_id"])

    def _run(self, content: list[dict[str, Any]]) -> RawOutput:
        t0 = time.perf_counter()
        try:
            text, n_in, n_out, info = self._call(content)
        except Exception as e:  # noqa: BLE001
            if self._is_fatal(e):
                raise FatalAdapterError(f"{type(e).__name__}: {str(e)[:200]}") from e
            raise
        dt = time.perf_counter() - t0
        extra = {"cost_usd": estimate_cost(self.cfg, n_in, n_out), "api_model": info.get("model"),
                 "stop_reason": info.get("stop_reason")}
        for k in ("reasoning_tokens", "thinking_tokens", "thinking_text"):  # 推理 token 數；Claude 開 display: summarized 時的推理摘要
            if info.get(k) is not None:
                extra[k] = info[k]
        return RawOutput(text=text or "", latency_s=dt, input_tokens=n_in, output_tokens=n_out, extra=extra)

    def predict(self, prompt: Prompt, max_time: float | None = None) -> RawOutput:
        return self._run(self._convert(prompt.content))

    def repair(self, raw_text: str, max_time: float | None = None) -> RawOutput:
        return self._run(self._convert([{"type": "text", "text": REPAIR_INSTRUCTION + raw_text}]))

    def metadata(self) -> dict[str, Any]:
        md = super().metadata()
        md.update({"provider": self.provider, "model_id": self.cfg.get("model_id"), "sdk_version": self.sdk_version})
        return md

    def close(self) -> None:
        c = self.client
        self.client = None
        try:
            if c is not None and hasattr(c, "close"):
                c.close()
        except Exception:  # noqa: BLE001
            pass


class AnthropicAdapter(ApiAdapter):
    provider = "anthropic"

    def _make_client(self) -> Any:
        import anthropic

        self.sdk_version = getattr(anthropic, "__version__", None)
        return anthropic.Anthropic(timeout=float(self.cfg.get("request_timeout", 180)),
                                   max_retries=int(self.cfg.get("max_retries", 4)))

    def _convert(self, prompt_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for c in prompt_content:
            if c["type"] == "text":
                out.append({"type": "text", "text": c["text"]})
            elif c["type"] == "image":
                out.append({"type": "image", "source": {"type": "base64", "media_type": media_type(c["path"]),
                                                        "data": image_b64(c["path"])}})
        return out

    def _call(self, content: list[dict[str, Any]]) -> tuple[str, int | None, int | None, dict[str, Any]]:
        kw = dict(self.cfg.get("extra_params") or {})
        resp = self.client.messages.create(model=self.cfg["model_id"], max_tokens=int(self.cfg.get("max_tokens", 1024)),
                                           messages=[{"role": "user", "content": content}], **kw)
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        # 推理摘要（thinking: {type: adaptive, display: summarized} 時才有內容；預設 omitted 是空字串）
        thinking = "\n".join(getattr(b, "thinking", "") or "" for b in resp.content if getattr(b, "type", "") == "thinking").strip()
        u = getattr(resp, "usage", None)
        det = getattr(u, "output_tokens_details", None)
        return text, getattr(u, "input_tokens", None), getattr(u, "output_tokens", None), {
            "model": getattr(resp, "model", None), "stop_reason": getattr(resp, "stop_reason", None),
            "thinking_text": thinking[:8000] or None,
            "thinking_tokens": getattr(det, "thinking_tokens", None) if det is not None else None}


class OpenAIAdapter(ApiAdapter):
    provider = "openai"

    def _make_client(self) -> Any:
        import openai

        self.sdk_version = getattr(openai, "__version__", None)
        return openai.OpenAI(timeout=float(self.cfg.get("request_timeout", 180)),
                             max_retries=int(self.cfg.get("max_retries", 4)))

    def _convert(self, prompt_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for c in prompt_content:
            if c["type"] == "text":
                out.append({"type": "text", "text": c["text"]})
            elif c["type"] == "image":
                url = f"data:{media_type(c['path'])};base64,{image_b64(c['path'])}"
                out.append({"type": "image_url", "image_url": {"url": url, "detail": self.cfg.get("image_detail", "high")}})
        return out

    def _call(self, content: list[dict[str, Any]]) -> tuple[str, int | None, int | None, dict[str, Any]]:
        kw = dict(self.cfg.get("extra_params") or {})
        resp = self.client.chat.completions.create(model=self.cfg["model_id"], messages=[{"role": "user", "content": content}],
                                                   max_completion_tokens=int(self.cfg.get("max_tokens", 4096)), **kw)
        ch = resp.choices[0] if resp.choices else None
        text = (getattr(getattr(ch, "message", None), "content", None) or "") if ch else ""
        u = getattr(resp, "usage", None)
        det = getattr(u, "completion_tokens_details", None)
        return text, getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None), {
            "model": getattr(resp, "model", None), "stop_reason": getattr(ch, "finish_reason", None) if ch else None,
            "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det is not None else None}


def make_api_adapter(key: str, cfg: dict[str, Any]) -> ApiAdapter:
    t = cfg.get("type")
    if t == "anthropic":
        return AnthropicAdapter(key, cfg)
    if t == "openai":
        return OpenAIAdapter(key, cfg)
    raise VLMConfigError(f"未知的 API 類型 {t}")
