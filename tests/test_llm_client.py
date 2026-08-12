"""
Рівні стійкості vision-клієнта.

Тести не ходять у мережу: `_call_model` підмінюється, і перевіряється саме
логіка навколо нього — коли повторюємо, коли здаємось, коли просимо модель
виправити власну відповідь.
"""

from __future__ import annotations

import json

import pytest

from src.llm import (
    ExtractionError,
    GeminiVisionClient,
    _is_daily_quota,
    _is_retryable,
    _parse_json,
    _server_retry_delay,
)
from tests.conftest import make_extraction


class FakeClientError(Exception):
    """Імітація genai ClientError: важливий і код, і текст із quotaId."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def make_client(monkeypatch) -> GeminiVisionClient:
    # Клієнт створюється без мережі: genai.Client лише зберігає ключ,
    # виклик відбувається в _call_model, який ми й підміняємо в тестах.
    return GeminiVisionClient(api_key="fake-key", rpm=0)


def test_daily_quota_recognised_by_quota_id():
    """
    Пастка Gemini: retryDelay приходить і в денному 429. Розрізняти можна
    тільки по quotaId, інакше прогін годинами крутить безнадійні retry.
    """
    daily = FakeClientError(429, "RESOURCE_EXHAUSTED 'quotaId': 'GenerateRequestsPerDayPerProject', "
                                 "'retryDelay': '30s'")
    minute = FakeClientError(429, "'quotaId': 'GenerateRequestsPerMinutePerProject', "
                                  "'retryDelay': '25s'")
    assert _is_daily_quota(daily) and not _is_daily_quota(minute)


def test_server_retry_delay_is_capped():
    assert _server_retry_delay(Exception("'retryDelay': '46s'")) == 46.0
    assert _server_retry_delay(Exception("'retryDelay': '600s'")) == 65.0
    assert _server_retry_delay(Exception("без підказки")) is None


def test_non_429_client_errors_are_not_retried():
    """400 і 403 самі не розсмокчуться — повторювати їх означає палити час."""
    assert not _is_retryable(FakeClientError(400, "INVALID_ARGUMENT"))
    assert _is_retryable(ConnectionError("обрив"))


@pytest.mark.parametrize(
    "raw",
    ['{"a": 1}', '```json\n{"a": 1}\n```', '```\n{"a": 1}\n```'],
)
def test_parse_json_survives_markdown_fences(raw):
    assert _parse_json(raw) == {"a": 1}


def test_parse_json_rejects_non_object():
    with pytest.raises(ValueError):
        _parse_json("[1, 2, 3]")


async def test_self_repair_retries_with_error_text(monkeypatch):
    """
    Перша відповідь невалідна — моделі показують її ж помилку. Перевіряємо
    саме це: у другому промпті має бути текст помилки, інакше «self-repair»
    вироджується у звичайний повтор того самого запиту.
    """
    client = make_client(monkeypatch)
    prompts: list[str] = []
    payloads = ["не JSON зовсім", make_extraction().model_dump_json()]

    async def fake_call(data, mime_type, prompt):
        prompts.append(prompt)
        return payloads.pop(0), 100, 50, 1.5

    monkeypatch.setattr(client, "_call_model", fake_call)
    result = await client.extract(b"pdf", "application/pdf")

    assert result.retries_used == 1
    assert "невалідною" in prompts[1]
    assert result.input_tokens == 200, "токени обох спроб мають підсумовуватись"
    assert result.api_latency_s == 3.0, "час обох викликів моделі теж підсумовується"


async def test_extraction_error_after_exhausting_repairs(monkeypatch):
    client = make_client(monkeypatch)

    async def always_broken(data, mime_type, prompt):
        return "{", 10, 10, 0.1

    monkeypatch.setattr(client, "_call_model", always_broken)
    with pytest.raises(ExtractionError):
        await client.extract(b"pdf", "application/pdf")


async def test_schema_violation_triggers_repair_not_crash(monkeypatch):
    """Відповідь — валідний JSON, але не за схемою: це теж привід на self-repair."""
    client = make_client(monkeypatch)
    payloads = [json.dumps({"doc_type": "рахунок", "total": "багато"}),
                make_extraction().model_dump_json()]

    async def fake_call(data, mime_type, prompt):
        return payloads.pop(0), 10, 10, 0.2

    monkeypatch.setattr(client, "_call_model", fake_call)
    result = await client.extract(b"pdf", "application/pdf")
    assert result.extraction.total == 3846.00
