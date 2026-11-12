"""Тонка обгортка над LLM API (Gemini через OpenAI-сумісний endpoint).

Обмеження воркшопу:
  * модель викликаємо лише через офіційний `openai` SDK (без streaming);
  * модель — з LLM_MODEL, ключ — з GEMINI_API_KEY (читаємо в момент виклику).

Ретраї тут вимкнені (max_retries=0): ними керує triage() з загальним дедлайном,
бо дефолтні 2 ретраї SDK не переживають 3×429, а дефолтний таймаут SDK — 10 хвилин.
"""
import os
import threading

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_client = None
_client_key = None
_lock = threading.Lock()

# Необов'язкові параметри, які не кожна модель/endpoint підтримує.
# Якщо API відповідає 400 через них — вимикаємо назавжди для цього процесу.
_OPTIONAL = {
    "reasoning_effort": "none",  # без "thinking": менше токенів і затримки
    "response_format": {"type": "json_object"},
}
_disabled = set()


def model_name() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODEL


def get_client() -> OpenAI:
    global _client, _client_key
    key = os.environ["GEMINI_API_KEY"]
    with _lock:
        if _client is None or _client_key != key:
            _client = OpenAI(
                api_key=key,
                base_url=os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL,
                max_retries=0,
                timeout=8.0,
            )
            _client_key = key
        return _client


def chat(messages, timeout: float = 8.0, **kwargs) -> str:
    """Один виклик моделі. Кидає виняток при помилці API — ретраї робить викликач."""
    client = get_client().with_options(timeout=timeout, max_retries=0)
    params = {k: v for k, v in _OPTIONAL.items() if k not in _disabled}
    params.update(kwargs)
    try:
        resp = client.chat.completions.create(model=model_name(), messages=messages, **params)
    except BadRequestError as e:
        bad = [k for k in _OPTIONAL if k in params and k not in kwargs]
        if not bad:
            raise
        text = str(e).lower()
        named = [k for k in bad if k.split("_")[0] in text or k in text]
        _disabled.update(named or bad)
        params = {k: v for k, v in params.items() if k not in _disabled}
        resp = client.chat.completions.create(model=model_name(), messages=messages, **params)
    if not resp.choices:
        return ""
    return resp.choices[0].message.content or ""
