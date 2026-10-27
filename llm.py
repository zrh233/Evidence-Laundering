"""OpenAI-compatible chat client configured through environment variables."""

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class LLMNotConfigured(ValueError):
    """Raised when the API key or model name is absent."""


def _load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


class LLM:
    def __init__(self, model_override=None):
        self.cfg = _load_config()
        self.lc = self.cfg["llm"]
        api_key = os.environ.get(self.lc["api_key_env"])
        model = model_override or os.environ.get(self.lc["model_env"]) or self.lc["model"]
        base_url = os.environ.get(self.lc["base_url_env"])
        if not api_key or not model:
            raise LLMNotConfigured(
                "Set OPENAI_API_KEY and EVIDENCE_LAUNDERING_MODEL, "
                "or pass --model. Use --stub for the offline plumbing test."
            )

        from openai import APIError, OpenAI

        self.request_errors = (APIError,)

        kwargs = {
            "api_key": api_key,
            "timeout": self.lc["request_timeout_s"],
            "max_retries": self.lc["max_retries"],
        }
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        endpoint = base_url or "openai-default"
        self.endpoint_fingerprint = hashlib.sha256(
            endpoint.encode("utf-8")
        ).hexdigest()[:16]

    def chat(self, messages, temperature=None, max_tokens=None):
        temperature = self.lc["temperature"] if temperature is None else temperature
        max_tokens = self.lc["max_tokens"] if max_tokens is None else max_tokens
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def get_llm(model_override=None):
    return LLM(model_override=model_override)
