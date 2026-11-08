"""Фолбек між Gemini-ключами для vision-екстракції: порядок, перемикання, без-ключа.

Без мережі й без ключа: ланцюжок — на стабах, білдер — на env-словниках.
"""

from __future__ import annotations

import asyncio

import pytest

from src.llm import (
    DailyQuotaExceeded,
    ExtractionError,
    ExtractionResult,
    FallbackVisionClient,
    GeminiVisionClient,
    build_vision_client,
)
from src.schema import InvoiceExtraction


def _result() -> ExtractionResult:
    return ExtractionResult(extraction=InvoiceExtraction.model_construct())


class _Stub:
    def __init__(self, name: str, *, exc: Exception | None = None) -> None:
        self.name, self.exc, self.calls = name, exc, 0

    async def extract(self, data: bytes, mime_type: str, hint: str = "") -> ExtractionResult:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return _result()


def _run(members):
    return asyncio.run(FallbackVisionClient(members).extract(b"x", "application/pdf"))


def test_first_working_key_short_circuits() -> None:
    a, b = _Stub("a"), _Stub("b")
    _run([("a", a), ("b", b)])
    assert (a.calls, b.calls) == (1, 0)


def test_quota_and_extraction_error_fall_through() -> None:
    a = _Stub("gemini:GEMINI_API_KEY", exc=DailyQuotaExceeded("per-day"))
    b = _Stub("gemini:GEMINI_API_KEY_2", exc=ExtractionError("transport"))
    c = _Stub("gemini:GEMINI_API_KEY_3")
    _run([("gemini:GEMINI_API_KEY", a), ("gemini:GEMINI_API_KEY_2", b), ("gemini:GEMINI_API_KEY_3", c)])
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)


def test_all_keys_failing_raises_extraction_error_naming_each() -> None:
    a = _Stub("gemini:GEMINI_API_KEY", exc=ExtractionError("x"))
    b = _Stub("gemini:GEMINI_API_KEY_2", exc=DailyQuotaExceeded("y"))
    with pytest.raises(ExtractionError) as e:
        _run([("gemini:GEMINI_API_KEY", a), ("gemini:GEMINI_API_KEY_2", b)])
    assert "GEMINI_API_KEY" in str(e.value) and "GEMINI_API_KEY_2" in str(e.value)


def test_build_vision_client_orders_dedups_single_and_none() -> None:
    members = build_vision_client(env={"GEMINI_API_KEY": "k1", "GEMINI_API_KEY_2": "k2",
                                       "GEMINI_API_KEY_3": "k3"}).members
    assert [n for n, _ in members] == ["gemini:GEMINI_API_KEY", "gemini:GEMINI_API_KEY_2",
                                       "gemini:GEMINI_API_KEY_3"]
    assert isinstance(build_vision_client(env={"GEMINI_API_KEY": "only"}), GeminiVisionClient)
    assert build_vision_client(env={}) is None
    assert isinstance(build_vision_client(env={"GEMINI_API_KEY": "s", "GEMINI_API_KEY_2": "s"}),
                      GeminiVisionClient)
