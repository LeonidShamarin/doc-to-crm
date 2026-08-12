"""
Контракт даних документа — єдине джерело правди.

Та сама модель `InvoiceExtraction` іде в Gemini як `response_schema` і нею ж
валідується сира відповідь. Дублювати опис полів у промпті не потрібно:
`description` полів SDK передає в схему сам.

Два принципові рішення:

1. **Усі числа й дати приходять від моделі як є, без коерсії.** Дата — рядок,
   а не `date`: якщо модель віддасть "31.02.2026", Pydantic із типом `date`
   зронив би весь документ у помилку валідації, і ми б не побачили, що саме
   вона прочитала. Розбирає дату Python у `validate.py` — і формує зрозумілу
   розбіжність замість винятку.

2. **Впевненість — по кожному полю, а не на документ.** Одне число на весь
   рахунок нічого не дає рев'юверу: він однаково перечитує все. Впевненість по
   полю дозволяє підсвітити рівно ті два поля, які модель прочитала невпевнено.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Статус обробки документа. `duplicate` — не помилка, а нормальний результат
# повторної подачі того самого файлу (див. store.py).
DocStatus = Literal["auto_ok", "needs_review", "fallback_error", "duplicate"]

# Серйозність розбіжності. `error` блокує автопроведення, `warning` — ні,
# але показується рев'юверу.
Severity = Literal["error", "warning"]

# Поля, по яких модель повертає впевненість. Список винесений окремо, бо по
# ньому ходять і review-UI, і eval — і розсинхрон між ними ловиться тестом.
CONFIDENCE_FIELDS: tuple[str, ...] = (
    "supplier_name",
    "supplier_edrpou",
    "buyer_name",
    "buyer_edrpou",
    "invoice_number",
    "invoice_date",
    "currency",
    "line_items",
    "subtotal",
    "vat_rate",
    "vat_amount",
    "total",
)


class LineItem(BaseModel):
    """Один рядок табличної частини рахунку."""

    description: str = Field(description="Назва товару або послуги, як у документі")
    quantity: Optional[float] = Field(default=None, description="Кількість")
    unit: Optional[str] = Field(default=None, description="Одиниця виміру, напр. шт, год, кг")
    unit_price: Optional[float] = Field(default=None, description="Ціна за одиницю без ПДВ")
    amount: Optional[float] = Field(default=None, description="Сума по рядку без ПДВ")


class FieldConfidence(BaseModel):
    """
    Впевненість моделі по кожному полю, 0.0–1.0.

    Поля перелічені явно, а не як `dict[str, float]`: словник із довільними
    ключами в JSON-схемі Gemini перетворюється на `additionalProperties`, і
    модель починає вигадувати ключі, яких у документі немає. Явний перелік
    робить схему замкненою, а розсинхрон із `CONFIDENCE_FIELDS` ловить тест.
    """

    supplier_name: float = Field(default=0.0, ge=0.0, le=1.0)
    supplier_edrpou: float = Field(default=0.0, ge=0.0, le=1.0)
    buyer_name: float = Field(default=0.0, ge=0.0, le=1.0)
    buyer_edrpou: float = Field(default=0.0, ge=0.0, le=1.0)
    invoice_number: float = Field(default=0.0, ge=0.0, le=1.0)
    invoice_date: float = Field(default=0.0, ge=0.0, le=1.0)
    currency: float = Field(default=0.0, ge=0.0, le=1.0)
    line_items: float = Field(default=0.0, ge=0.0, le=1.0)
    subtotal: float = Field(default=0.0, ge=0.0, le=1.0)
    vat_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    vat_amount: float = Field(default=0.0, ge=0.0, le=1.0)
    total: float = Field(default=0.0, ge=0.0, le=1.0)

    def lowest(self) -> tuple[str, float]:
        """Найслабше поле — саме воно вирішує долю документа при маршрутизації."""
        values = self.model_dump()
        name = min(values, key=lambda k: values[k])
        return name, values[name]

    def below(self, threshold: float) -> dict[str, float]:
        """Поля, впевненість яких нижча за поріг — їх підсвічує review-UI."""
        return {k: v for k, v in self.model_dump().items() if v < threshold}


class InvoiceExtraction(BaseModel):
    """Те, що ми очікуємо від моделі для одного документа."""

    doc_type: Literal["рахунок", "акт", "видаткова накладна", "інше"] = Field(
        description="Тип документа"
    )

    supplier_name: Optional[str] = Field(default=None, description="Повна назва постачальника")
    supplier_edrpou: Optional[str] = Field(
        default=None,
        description="ЄДРПОУ постачальника: 8 цифр для юрособи, 10 — ІПН для ФОП. "
        "Тільки цифри, без пробілів. null, якщо в документі немає.",
    )
    buyer_name: Optional[str] = Field(default=None, description="Повна назва покупця")
    buyer_edrpou: Optional[str] = Field(default=None, description="ЄДРПОУ або ІПН покупця")

    invoice_number: Optional[str] = Field(default=None, description="Номер документа, як у бланку")
    invoice_date: Optional[str] = Field(
        default=None,
        description="Дата документа у форматі YYYY-MM-DD. Якщо в документі "
        "формат інший — переведи, не вигадуючи відсутніх частин.",
    )
    currency: Optional[str] = Field(
        default=None, description="Валюта, код ISO: UAH, USD, EUR"
    )

    line_items: list[LineItem] = Field(
        default_factory=list, description="Рядки табличної частини, зверху вниз"
    )

    subtotal: Optional[float] = Field(default=None, description="Сума без ПДВ")
    vat_rate: Optional[float] = Field(
        default=None, description="Ставка ПДВ у відсотках, напр. 20.0. 0.0 — якщо без ПДВ"
    )
    vat_amount: Optional[float] = Field(default=None, description="Сума ПДВ")
    total: Optional[float] = Field(default=None, description="Загальна сума до сплати з ПДВ")

    confidence: FieldConfidence = Field(
        default_factory=FieldConfidence,
        description="Твоя впевненість по кожному полю окремо, 0.0–1.0. "
        "Низька — якщо місце в документі нерозбірливе, обрізане або поля немає.",
    )

    @field_validator("supplier_edrpou", "buyer_edrpou")
    @classmethod
    def _digits_only(cls, v: Optional[str]) -> Optional[str]:
        """
        Модель регулярно повертає «ЄДРПОУ 12345678» або «123 456 78».
        Це не помилка змісту — просто формат, і чистити його дешевше тут,
        ніж у кожному місці, де код порівнює коди між собою.
        """
        if v is None:
            return None
        digits = "".join(ch for ch in v if ch.isdigit())
        return digits or None

    @field_validator("currency")
    @classmethod
    def _normalize_currency(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        return {"ГРН": "UAH", "UAH.": "UAH", "₴": "UAH", "$": "USD", "€": "EUR"}.get(v, v) or None


class ValidationIssue(BaseModel):
    """
    Одна розбіжність, знайдена Python-перевіркою (не моделлю).

    `expected`/`actual` заповнюються там, де є що порівняти: рев'юверу треба
    бачити не «сума не зійшлась», а «мало бути 12 000.00, у документі 12 000.50».
    """

    code: str
    severity: Severity
    field: Optional[str] = None
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


class ProcessedDocument(BaseModel):
    """Повний запис одного документа: вхід, екстракція, перевірки, маршрут."""

    doc_id: str
    source_path: str
    content_hash: str
    mime_type: str
    received_at: str

    extraction: Optional[InvoiceExtraction] = None
    issues: list[ValidationIssue] = Field(default_factory=list)

    status: DocStatus = "needs_review"
    review_reasons: list[str] = Field(default_factory=list)

    # Метадані пайплайну (не від моделі) — потрібні, щоб чесно показати ціну
    # й місце, де обробка «здалась».
    processing_error: Optional[str] = None
    retries_used: int = 0
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    # Заповнюється тільки для status="duplicate": який документ уже є в базі.
    duplicate_of: Optional[str] = None

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def to_row(self) -> dict:
        """
        Плаский рядок для sink-а: одна лінія JSONL = один документ.

        Позиції зводяться в один рядок навмисно: sink імітує таблицю обліку,
        де рахунок — це запис, а не набір записів. Повна вкладена структура
        лишається в SQLite і доступна через review-UI.
        """
        e = self.extraction
        row = {
            "doc_id": self.doc_id,
            "status": self.status,
            "source_file": self.source_path,
            "content_hash": self.content_hash,
            "received_at": self.received_at,
            "doc_type": e.doc_type if e else None,
            "supplier_name": e.supplier_name if e else None,
            "supplier_edrpou": e.supplier_edrpou if e else None,
            "buyer_name": e.buyer_name if e else None,
            "buyer_edrpou": e.buyer_edrpou if e else None,
            "invoice_number": e.invoice_number if e else None,
            "invoice_date": e.invoice_date if e else None,
            "currency": e.currency if e else None,
            "line_items_count": len(e.line_items) if e else 0,
            "subtotal": e.subtotal if e else None,
            "vat_rate": e.vat_rate if e else None,
            "vat_amount": e.vat_amount if e else None,
            "total": e.total if e else None,
            "min_confidence": (e.confidence.lowest()[1] if e else 0.0),
            "min_confidence_field": (e.confidence.lowest()[0] if e else None),
            "issues": [i.code for i in self.issues],
            "review_reasons": self.review_reasons,
            "processing_error": self.processing_error,
            "latency_s": round(self.latency_s, 2),
            "duplicate_of": self.duplicate_of,
        }
        return row
