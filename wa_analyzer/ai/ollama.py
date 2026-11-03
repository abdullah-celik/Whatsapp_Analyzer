"""Minimal client for a local Ollama server (https://ollama.com). No extra dependencies.

Ollama decides itself whether to use the NVIDIA GPU. With device='cpu' we ask it to keep every layer on the
CPU (num_gpu=0); with 'cuda' or 'auto' it uses the GPU when one is available.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

DEFAULT_MODEL_CPU = "qwen2.5:3b"
DEFAULT_MODEL_GPU = "qwen2.5:7b"


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, model: str | None = None, host: str = "http://localhost:11434", device: str = "auto",
                 num_ctx: int = 8192, timeout: float = 900, temperature: float = 0.0):
        self.host = host.rstrip("/")
        self.device = device
        self.model = model or (DEFAULT_MODEL_CPU if device == "cpu" else DEFAULT_MODEL_GPU)
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.temperature = temperature

    # -------------------------------------------------------------- http
    def _req(self, path: str, payload: dict | None = None, timeout: float | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise OllamaError(f"Ollama HTTP {e.code}: {body}") from None
        except urllib.error.URLError as e:
            raise OllamaError(f"Cannot reach Ollama at {self.host} ({e.reason}). Is it running? Start it with: ollama serve") from None

    def available(self) -> bool:
        try:
            self._req("/api/tags", timeout=5)
            return True
        except Exception:
            return False

    def models(self) -> list[str]:
        return [m["name"] for m in self._req("/api/tags", timeout=10).get("models", [])]

    def check(self) -> None:
        names = self.models()
        base = self.model if ":" in self.model else self.model + ":latest"
        if self.model not in names and base not in names:
            raise OllamaError(f"Model '{self.model}' is not installed in Ollama. Install it with:  ollama pull {self.model}\n"
                              f"Installed: {', '.join(names) or '(none)'}")

    # -------------------------------------------------------------- chat
    def chat_json(self, system: str, user: str, schema: dict | None = None, retries: int = 2) -> dict:
        options = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.device == "cpu":
            options["num_gpu"] = 0
        payload = {"model": self.model, "stream": False, "options": options,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "format": schema or "json"}
        last = None
        for attempt in range(retries + 1):
            res = self._req("/api/chat", payload)
            content = res.get("message", {}).get("content", "")
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                start, end = content.find("{"), content.rfind("}")
                if start >= 0 and end > start:
                    try:
                        return json.loads(content[start:end + 1])
                    except json.JSONDecodeError:
                        pass
                last = content[:200]
                payload["format"] = "json"   # fall back to plain JSON mode
                time.sleep(0.5)
        raise OllamaError(f"The model did not return valid JSON: {last!r}")

    def name_topic(self, keywords: list[str], examples: list[str]) -> str:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        user = ("Keywords: " + ", ".join(keywords) + "\n\nExample chat snippets:\n" +
                "\n---\n".join(e[:400] for e in examples[:3]) +
                "\n\nGive this conversation topic a short English name (2-4 words).")
        out = self.chat_json("You name topics of private chat conversations. Answer in JSON.", user, schema)
        return str(out.get("name", "")).strip()[:40]
