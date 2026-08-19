from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .models import AgentCallRecord, LLMResult, WorldState


DEFAULT_MODEL = "gpt-5.6-luna-openai-compact"
DEFAULT_BASE_URL = "https://kuaipao.ai/v1"
DEFAULT_LOCAL_MODEL = "qwen3.5:9b-q4_K_M"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
RoutingMode = Literal["auto", "local", "remote"]


def _load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class LLMSettings(BaseModel):
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    enabled: bool = False
    timeout_seconds: float = 30.0
    cooldown_seconds: float = 60.0
    max_retries: int = 1
    retry_delay_seconds: float = 0.35
    max_output_tokens: int = 400
    context_window: int | None = None
    temperature: float | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None = None

    @classmethod
    def from_env(cls) -> "LLMSettings":
        _load_dotenv()
        raw_enabled = os.getenv("LIVING_TABLETOP_LLM_ENABLED", "false").lower()
        api_key = os.getenv("LIVING_TABLETOP_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LIVING_TABLETOP_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"
        raw_temperature = os.getenv("LIVING_TABLETOP_LLM_TEMPERATURE")
        raw_reasoning_effort = os.getenv("LIVING_TABLETOP_LLM_REASONING_EFFORT")
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=os.getenv("LIVING_TABLETOP_MODEL", DEFAULT_MODEL),
            enabled=raw_enabled in {"1", "true", "yes", "on"} and bool(api_key),
            timeout_seconds=float(os.getenv("LIVING_TABLETOP_LLM_TIMEOUT", "30")),
            cooldown_seconds=float(os.getenv("LIVING_TABLETOP_LLM_COOLDOWN", "60")),
            max_retries=max(0, int(os.getenv("LIVING_TABLETOP_LLM_MAX_RETRIES", "1"))),
            retry_delay_seconds=max(0.0, float(os.getenv("LIVING_TABLETOP_LLM_RETRY_DELAY", "0.35"))),
            max_output_tokens=max(64, int(os.getenv("LIVING_TABLETOP_LLM_MAX_OUTPUT_TOKENS", "400"))),
            context_window=(
                max(1024, int(value))
                if (value := os.getenv("LIVING_TABLETOP_LLM_CONTEXT_WINDOW"))
                else None
            ),
            temperature=float(raw_temperature) if raw_temperature is not None else None,
            reasoning_effort=raw_reasoning_effort or None,
        )

    @classmethod
    def local_from_env(cls) -> "LLMSettings":
        _load_dotenv()
        raw_enabled = os.getenv("LIVING_TABLETOP_LOCAL_LLM_ENABLED", "true").lower()
        base_url = os.getenv("LIVING_TABLETOP_LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL)
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"
        raw_temperature = os.getenv("LIVING_TABLETOP_LOCAL_LLM_TEMPERATURE", "0")
        raw_reasoning_effort = os.getenv("LIVING_TABLETOP_LOCAL_LLM_REASONING_EFFORT", "none")
        return cls(
            api_key=os.getenv("LIVING_TABLETOP_LOCAL_LLM_API_KEY", "ollama"),
            base_url=base_url,
            model=os.getenv("LIVING_TABLETOP_LOCAL_LLM_MODEL", DEFAULT_LOCAL_MODEL),
            enabled=raw_enabled in {"1", "true", "yes", "on"},
            timeout_seconds=float(os.getenv("LIVING_TABLETOP_LOCAL_LLM_TIMEOUT", "60")),
            cooldown_seconds=float(os.getenv("LIVING_TABLETOP_LOCAL_LLM_COOLDOWN", "15")),
            max_retries=max(0, int(os.getenv("LIVING_TABLETOP_LOCAL_LLM_MAX_RETRIES", "1"))),
            retry_delay_seconds=max(
                0.0,
                float(os.getenv("LIVING_TABLETOP_LOCAL_LLM_RETRY_DELAY", "0.1")),
            ),
            max_output_tokens=max(
                64,
                int(os.getenv("LIVING_TABLETOP_LOCAL_LLM_MAX_OUTPUT_TOKENS", "900")),
            ),
            context_window=max(
                1024,
                int(os.getenv("LIVING_TABLETOP_LOCAL_LLM_CONTEXT_WINDOW", "8192")),
            ),
            temperature=float(raw_temperature) if raw_temperature else None,
            reasoning_effort=raw_reasoning_effort or None,
        )


class LLMUnavailable(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        public_message: str | None = None,
        failures: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.public_message = public_message
        self.failures = failures or {}


class OpenAICompatibleLLM:
    def __init__(self, settings: LLMSettings | None = None, *, provider: str = "remote"):
        self.settings = settings or LLMSettings.from_env()
        self.provider = provider
        self._client = None
        self._native_client = None
        self._circuit_open_until = 0.0
        self._models_cache: list[str] = []
        self._models_cached_at = 0.0
        self.last_error: str | None = None
        self.last_success_at: float | None = None

    @property
    def configured(self) -> bool:
        return self.settings.enabled and bool(self.settings.api_key)

    @property
    def enabled(self) -> bool:
        return self.configured and time.monotonic() >= self._circuit_open_until

    def _get_client(self):
        if not self.enabled:
            raise LLMUnavailable("LLM integration is disabled or no API key is configured")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
                # Retry explicitly below so compatible gateways and the
                # response_format fallback share the same small retry budget.
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(error, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    @classmethod
    def _safe_error(cls, error: Exception) -> str:
        status_code = cls._status_code(error)
        message = str(error).lower()
        if status_code == 404 and ("model" in message or "not_found" in message):
            return "所选模型当前没有可用推理渠道"
        if status_code in {401, 403}:
            return "鉴权失败或当前账户无权使用该模型"
        if status_code == 429:
            return "服务繁忙或额度受限"
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return f"上游服务暂时不可用（{status_code}）"
        if type(error).__name__ in {
            "APIConnectionError",
            "ConnectError",
            "ConnectTimeout",
            "ReadError",
        }:
            return "无法连接模型服务"
        if type(error).__name__ in {"APITimeoutError", "ReadTimeout", "TimeoutError"}:
            return "模型响应超时"
        if isinstance(error, (json.JSONDecodeError, ValueError)):
            return "模型没有返回有效的结构化结果"
        return "模型调用失败"

    def _record_failure(self, error: Exception) -> None:
        self.last_error = self._safe_error(error)

    def set_model(self, model: str) -> None:
        selected = model.strip()
        if not selected or len(selected) > 160:
            raise ValueError("Invalid model name")
        self.settings.model = selected
        self._circuit_open_until = 0.0
        self.last_error = None

    def list_models(self, *, refresh: bool = False) -> tuple[list[str], str | None]:
        if not self.configured:
            return [], "提供方尚未启用"
        if not refresh and self._models_cache and time.monotonic() - self._models_cached_at < 30:
            return list(self._models_cache), self.last_error
        try:
            response = self._get_client().models.list()
            models = sorted({item.id for item in response.data if getattr(item, "id", None)})
            self._models_cache = models
            self._models_cached_at = time.monotonic()
            return list(models), self.last_error
        except Exception as error:
            self._record_failure(error)
            return list(self._models_cache), self.last_error

    def status(self, *, refresh: bool = False) -> dict[str, Any]:
        models, discovery_error = self.list_models(refresh=refresh)
        cooling_down = self.configured and not self.enabled
        if cooling_down:
            state = "cooldown"
        elif models:
            state = "online"
        elif self.configured:
            state = "unknown"
        else:
            state = "disabled"
        return {
            "provider": self.provider,
            "state": state,
            "configured": self.configured,
            "enabled": self.enabled,
            "model": self.settings.model,
            "context_window": self.settings.context_window,
            "models": models,
            "last_error": discovery_error,
            "last_success_at": self.last_success_at,
        }

    @classmethod
    def _is_retryable(cls, error: Exception) -> bool:
        status_code = cls._status_code(error)
        if status_code in {408, 409, 429}:
            return True
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return True
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        return type(error).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectTimeout",
            "ReadError",
            "ReadTimeout",
        }

    def _create_with_retries(self, create, **kwargs):
        for retry_index in range(self.settings.max_retries + 1):
            try:
                return create(**kwargs)
            except Exception as error:
                if retry_index >= self.settings.max_retries or not self._is_retryable(error):
                    raise
                delay = self.settings.retry_delay_seconds * (2**retry_index)
                if delay:
                    time.sleep(delay)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        value = text.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            value = "\n".join(lines)
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Model output must be a JSON object")
        return parsed

    def complete_json(
        self,
        *,
        system: str,
        user_payload: dict[str, Any],
        max_output_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "structured_output",
        temperature: float | None = None,
        reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None = None,
    ) -> LLMResult:
        if (
            self.provider == "local"
            and ":11434" in self.settings.base_url
            and self.settings.base_url.rstrip("/").endswith("/v1")
        ):
            return self._complete_ollama_json(
                system=system,
                user_payload=user_payload,
                max_output_tokens=max_output_tokens,
                response_schema=response_schema,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        client = self._get_client()
        started = time.perf_counter()
        request = {
            "model": self.settings.model,
            "max_tokens": max_output_tokens or self.settings.max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        selected_temperature = self.settings.temperature if temperature is None else temperature
        selected_reasoning_effort = (
            self.settings.reasoning_effort if reasoning_effort is None else reasoning_effort
        )
        if selected_temperature is not None:
            request["temperature"] = selected_temperature
        if selected_reasoning_effort is not None:
            request["reasoning_effort"] = selected_reasoning_effort
        response_format: dict[str, Any]
        if response_schema is None:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            }
        try:
            response = self._create_with_retries(
                client.chat.completions.create,
                **request,
                response_format=response_format,
            )
        except Exception as exc:
            # Some OpenAI-compatible gateways implement Chat Completions but not
            # response_format. Retry only a 400-class compatibility failure; the
            # prompt still requires a JSON object and parsing remains fail-closed.
            status_code = getattr(exc, "status_code", None)
            if status_code not in {400, 404, 422}:
                self._circuit_open_until = time.monotonic() + self.settings.cooldown_seconds
                self._record_failure(exc)
                raise
            try:
                response = self._create_with_retries(client.chat.completions.create, **request)
            except Exception as fallback_error:
                self._circuit_open_until = time.monotonic() + self.settings.cooldown_seconds
                self._record_failure(fallback_error)
                raise
        latency_ms = round((time.perf_counter() - started) * 1000)
        text = response.choices[0].message.content or "{}"
        usage = getattr(response, "usage", None)
        try:
            result = LLMResult(
                data=self._parse_json(text),
                latency_ms=latency_ms,
                input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            )
        except Exception as error:
            self._record_failure(error)
            raise
        self.last_error = None
        self.last_success_at = time.time()
        return result

    def _complete_ollama_json(
        self,
        *,
        system: str,
        user_payload: dict[str, Any],
        max_output_tokens: int | None,
        response_schema: dict[str, Any] | None,
        temperature: float | None,
        reasoning_effort: Literal["none", "low", "medium", "high", "max"] | None,
    ) -> LLMResult:
        """Use Ollama's native endpoint so per-request num_ctx is actually honored."""

        if not self.enabled:
            raise LLMUnavailable("LLM integration is disabled or no API key is configured")
        if self._native_client is None:
            import httpx

            self._native_client = httpx.Client(timeout=self.settings.timeout_seconds)

        selected_temperature = self.settings.temperature if temperature is None else temperature
        selected_reasoning_effort = (
            self.settings.reasoning_effort if reasoning_effort is None else reasoning_effort
        )
        options: dict[str, Any] = {
            "num_predict": max_output_tokens or self.settings.max_output_tokens,
        }
        if self.settings.context_window is not None:
            options["num_ctx"] = self.settings.context_window
        if selected_temperature is not None:
            options["temperature"] = selected_temperature
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "format": response_schema or "json",
            "options": options,
        }
        if selected_reasoning_effort == "none":
            payload["think"] = False

        native_url = f"{self.settings.base_url.rstrip('/')[:-3]}/api/chat"
        started = time.perf_counter()
        response = None
        for retry_index in range(self.settings.max_retries + 1):
            try:
                response = self._native_client.post(native_url, json=payload)
                response.raise_for_status()
                break
            except Exception as error:
                retryable = self._is_retryable(error)
                if retryable and retry_index < self.settings.max_retries:
                    delay = self.settings.retry_delay_seconds * (2**retry_index)
                    if delay:
                        time.sleep(delay)
                    continue
                if retryable:
                    self._circuit_open_until = time.monotonic() + self.settings.cooldown_seconds
                self._record_failure(error)
                raise

        assert response is not None
        try:
            raw = response.json()
            content = raw.get("message", {}).get("content", "{}")
            result = LLMResult(
                data=self._parse_json(content),
                latency_ms=round((time.perf_counter() - started) * 1000),
                input_tokens=raw.get("prompt_eval_count"),
                output_tokens=raw.get("eval_count"),
            )
        except Exception as error:
            # The model was reachable. Invalid JSON is repaired by StructuredHarness
            # and must not masquerade as a connection outage or open the circuit.
            self._record_failure(error)
            raise
        self.last_error = None
        self.last_success_at = time.time()
        return result


class RoutedLLM:
    """Runtime-selectable local/remote router with local-first failover."""

    def __init__(
        self,
        *,
        local: OpenAICompatibleLLM | None,
        remote: OpenAICompatibleLLM | None,
        mode: RoutingMode = "auto",
    ):
        self.providers = {key: value for key, value in {"local": local, "remote": remote}.items() if value}
        self._lock = threading.RLock()
        self.mode: RoutingMode = "auto"
        self.last_used: dict[str, Any] | None = None
        self.last_failures: dict[str, str] = {}
        self.configure(mode=mode)

    @classmethod
    def from_env(cls) -> "RoutedLLM":
        _load_dotenv()
        raw_mode = os.getenv("LIVING_TABLETOP_LLM_MODE", "auto").lower()
        mode: RoutingMode = raw_mode if raw_mode in {"auto", "local", "remote"} else "auto"
        return cls(
            local=OpenAICompatibleLLM(LLMSettings.local_from_env(), provider="local"),
            remote=OpenAICompatibleLLM(LLMSettings.from_env(), provider="remote"),
            mode=mode,
        )

    @property
    def settings(self) -> LLMSettings:
        ordered = self._ordered_providers(include_unavailable=True)
        if ordered:
            return ordered[0][1].settings
        return LLMSettings(enabled=False)

    @property
    def configured(self) -> bool:
        return any(provider.configured for provider in self.providers.values())

    @property
    def enabled(self) -> bool:
        return any(provider.enabled for _, provider in self._ordered_providers())

    def _ordered_providers(
        self,
        *,
        include_unavailable: bool = False,
    ) -> list[tuple[str, OpenAICompatibleLLM]]:
        with self._lock:
            names = ["local", "remote"] if self.mode == "auto" else [self.mode]
            selected = [(name, self.providers[name]) for name in names if name in self.providers]
        if include_unavailable:
            return selected
        return [(name, provider) for name, provider in selected if provider.enabled]

    @staticmethod
    def _public_failure_message(mode: RoutingMode) -> str:
        if mode == "local":
            return "本地模型暂时不可用，请确认 Ollama 正在运行且所选模型已安装。行动未提交，游戏状态没有改变。"
        if mode == "remote":
            return "远程模型暂时不可用，请检查模型权限或稍后重试。行动未提交，游戏状态没有改变。"
        return "本地与远程模型目前都不可用。行动未提交，游戏状态没有改变；可在右上角的模型设置中检查或切换。"

    def complete_json(self, **kwargs) -> LLMResult:
        with self._lock:
            mode = self.mode
        candidates = self._ordered_providers()
        failures: dict[str, str] = {}
        for name, provider in candidates:
            try:
                result = provider.complete_json(**kwargs)
            except Exception as error:
                failures[name] = provider._safe_error(error)
                if mode != "auto":
                    break
                continue
            with self._lock:
                self.last_used = {
                    "provider": name,
                    "model": provider.settings.model,
                    "latency_ms": result.latency_ms,
                    "at": time.time(),
                }
                self.last_failures = failures
            return result

        if not candidates:
            for name, provider in self._ordered_providers(include_unavailable=True):
                failures[name] = provider.last_error or (
                    "模型处于短暂冷却中" if provider.configured else "提供方尚未启用"
                )
        with self._lock:
            self.last_failures = failures
        raise LLMUnavailable(
            "No configured LLM provider completed the request",
            public_message=self._public_failure_message(mode),
            failures=failures,
        )

    def configure(
        self,
        *,
        mode: RoutingMode,
        local_model: str | None = None,
        remote_model: str | None = None,
    ) -> None:
        if mode not in {"auto", "local", "remote"}:
            raise ValueError("Invalid LLM routing mode")
        with self._lock:
            if local_model is not None:
                if "local" not in self.providers:
                    raise ValueError("Local provider is unavailable")
                self.providers["local"].set_model(local_model)
            if remote_model is not None:
                if "remote" not in self.providers:
                    raise ValueError("Remote provider is unavailable")
                self.providers["remote"].set_model(remote_model)
            self.mode = mode

    def configuration(self, *, refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            mode = self.mode
            last_used = dict(self.last_used) if self.last_used else None
            last_failures = dict(self.last_failures)
        return {
            "mode": mode,
            "order": [name for name, _ in self._ordered_providers(include_unavailable=True)],
            "providers": {
                name: provider.status(refresh=refresh)
                for name, provider in self.providers.items()
            },
            "last_used": last_used,
            "last_failures": last_failures,
        }


def record_agent_call(
    state: WorldState,
    *,
    role: str,
    result: LLMResult | None,
    validation: str,
    error: bool = False,
) -> None:
    structured_output = result.data if result else None
    digest_source = json.dumps(structured_output or {}, ensure_ascii=False, sort_keys=True)
    state.agent_calls.append(
        AgentCallRecord(
            id=f"agent_call_{len(state.agent_calls) + 1:05d}",
            role=role,
            input_state_version=state.version,
            output_digest=hashlib.sha256(digest_source.encode("utf-8")).hexdigest(),
            structured_output=structured_output,
            validation=validation,
            latency_ms=result.latency_ms if result else 0,
            input_tokens=result.input_tokens if result else None,
            output_tokens=result.output_tokens if result else None,
            result="error" if error else "success",
            created_at=state.world_time,
        )
    )
