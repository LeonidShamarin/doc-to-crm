"""
Спільні фікстури.

Головна — `FakeVisionClient`: він реалізує той самий протокол `VisionClient`,
що й Gemini, тому весь набір тестів проходить без ключа й без мережі. Клієнт
уміє не тільки віддавати готову екстракцію, а й падати заданою помилкою —
інакше гілки fallback і денної квоти лишилися б непокритими.
"""

from __future__ import annotations

import pytest

from src.config import PipelineConfig
from src.llm import ExtractionResult
from src.pipeline import DocumentPipeline
from src.schema import FieldConfidence, InvoiceExtraction, LineItem
from src.sinks import JsonlSink
from src.store import DocumentStore


def make_extraction(**overrides) -> InvoiceExtraction:
    """Коректний рахунок, у якому все сходиться. База для більшості тестів."""
    data = {
        "doc_type": "рахунок",
        "supplier_name": "ТОВ «Агросвіт-Плюс»",
        "supplier_edrpou": "30123453",
        "buyer_name": "ТОВ «Промінвест Груп»",
        "buyer_edrpou": "31112222",
        "invoice_number": "СФ-100",
        "invoice_date": "2026-03-11",
        "currency": "UAH",
        "line_items": [
            LineItem(description="Папір A4", quantity=10, unit="пач.",
                     unit_price=120.50, amount=1205.00),
            LineItem(description="Картридж", quantity=2, unit="шт.",
                     unit_price=1000.00, amount=2000.00),
        ],
        "subtotal": 3205.00,
        "vat_rate": 20.0,
        "vat_amount": 641.00,
        "total": 3846.00,
        "confidence": FieldConfidence(**{f: 0.95 for f in FieldConfidence.model_fields}),
    }
    data.update(overrides)
    return InvoiceExtraction.model_validate(data)


class FakeVisionClient:
    """
    Фейковий клієнт. `responses` — черга: що віддавати на послідовні виклики.
    Елемент-виняток кидається замість відповіді.
    """

    def __init__(self, responses=None, default=None):
        self.responses = list(responses or [])
        self.default = default if default is not None else make_extraction()
        self.calls: list[tuple[int, str]] = []

    async def extract(self, data: bytes, mime_type: str, hint: str = "") -> ExtractionResult:
        self.calls.append((len(data), mime_type))
        item = self.responses.pop(0) if self.responses else self.default
        if isinstance(item, Exception):
            raise item
        return ExtractionResult(
            extraction=item, retries_used=0, input_tokens=1200, output_tokens=300
        )


@pytest.fixture
def store(tmp_path) -> DocumentStore:
    st = DocumentStore(tmp_path / "state.sqlite3")
    yield st
    st.close()


@pytest.fixture
def sink(tmp_path) -> JsonlSink:
    return JsonlSink(tmp_path / "ledger.jsonl")


@pytest.fixture
def cfg() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def pipeline(store, sink, cfg) -> DocumentPipeline:
    return DocumentPipeline(client=FakeVisionClient(), store=store, sink=sink, cfg=cfg)
