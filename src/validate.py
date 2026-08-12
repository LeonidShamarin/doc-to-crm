"""
Шар перевірок **поза LLM**.

Головна теза проекту: модель галюцинує суми — тому суми перевіряє Python.
Мовна модель бачить «12 400,00» і «12 480,00» як два схожі рядки; різницю в
80 гривень вона не «розуміє» як арифметичну помилку. Тому все, що можна
перерахувати або перевірити контрольною цифрою, перераховується тут.

Кожна перевірка повертає `ValidationIssue` з `expected`/`actual`, а не булеве
«погано»: рев'ювер має бачити, що саме не зійшлося, без відкривання документа.

Функції не ходять у мережу і не залежать від конфігу глобально — тому весь
модуль покривається тестами без ключа.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from src.config import PipelineConfig
from src.schema import InvoiceExtraction, ValidationIssue

# Валюти, які пайплайн вважає осмисленими. Усе інше — не помилка моделі, а
# сигнал рев'юверу подивитись очима.
KNOWN_CURRENCIES = {"UAH", "USD", "EUR", "GBP", "PLN"}

# Ставки ПДВ, чинні в Україні. 7% — ліки й медвироби, 14% — окремі
# сільгосптовари, 0% — експорт.
KNOWN_VAT_RATES = {0.0, 7.0, 14.0, 20.0}

# Документ, датований раніше цієї межі, — майже напевно неправильно прочитаний
# рік (типова помилка на сканах: 2016 замість 2026).
MIN_PLAUSIBLE_YEAR = 2000

# Верхня межа суми для санітарної перевірки. Не бізнес-правило, а захист від
# зсуву коми: «1 200 000 000,00» замість «12 000,00».
MAX_PLAUSIBLE_TOTAL = 1_000_000_000.0


# --------------------------------------------------------------------------
# Контрольні цифри українських кодів
# --------------------------------------------------------------------------

def edrpou_control_digit(code: str) -> Optional[int]:
    """
    Контрольна цифра ЄДРПОУ (8 цифр) за офіційним алгоритмом.

    Ваги залежать від діапазону коду: для 30000000–60000000 вони зсунуті на
    одну позицію. Якщо залишок від ділення на 11 дорівнює 10 — рахунок
    повторюється з вагами, збільшеними на 2. Якщо і тоді 10 — контрольну
    цифру для такого коду визначити неможливо, повертаємо None.
    """
    if len(code) != 8 or not code.isdigit():
        return None
    digits = [int(c) for c in code[:7]]
    number = int(code)
    base = [7, 1, 2, 3, 4, 5, 6] if 30_000_000 <= number <= 60_000_000 else [1, 2, 3, 4, 5, 6, 7]

    remainder = sum(d * w for d, w in zip(digits, base)) % 11
    if remainder >= 10:
        shifted = [w + 2 for w in base]
        remainder = sum(d * w for d, w in zip(digits, shifted)) % 11
    if remainder >= 10:
        return None
    return remainder


def is_valid_edrpou(code: str) -> bool:
    control = edrpou_control_digit(code)
    return control is not None and control == int(code[7])


def ipn_control_digit(code: str) -> Optional[int]:
    """
    Контрольна цифра РНОКПП (ІПН, 10 цифр) — коду фізособи або ФОП.
    Ваги фіксовані, перша від'ємна; результат береться по модулю 11, потім 10.
    """
    if len(code) != 10 or not code.isdigit():
        return None
    weights = [-1, 5, 7, 9, 4, 6, 10, 5, 7]
    total = sum(int(d) * w for d, w in zip(code[:9], weights))
    return (total % 11) % 10


def is_valid_ipn(code: str) -> bool:
    control = ipn_control_digit(code)
    return control is not None and control == int(code[9])


def make_valid_ipn(prefix9: str) -> str:
    """Добудовує валідний 10-значний РНОКПП — для рахунків від ФОП у датасеті."""
    control = ipn_control_digit(prefix9 + "0")
    if control is None:
        raise ValueError(f"некоректний префікс РНОКПП: {prefix9}")
    return prefix9 + str(control)


def is_valid_tax_code(code: str) -> bool:
    """ЄДРПОУ (8) або РНОКПП (10) — обидва варіанти легальні в полі контрагента."""
    if len(code) == 8:
        return is_valid_edrpou(code)
    if len(code) == 10:
        return is_valid_ipn(code)
    return False


def make_valid_edrpou(prefix7: str) -> str:
    """
    Добудовує валідний 8-значний ЄДРПОУ до семи заданих цифр.

    Потрібно генератору датасету: синтетичні рахунки мають нести коди, які
    проходять ту саму перевірку, що й справжні, — інакше весь шар валідації
    на власному датасеті світився б помилками і нічого не вимірював.
    Якщо для префікса контрольну цифру визначити неможливо, останню цифру
    зсуваємо і пробуємо далі — вихід гарантований за 10 кроків.
    """
    digits = list(prefix7)
    for _ in range(10):
        control = edrpou_control_digit("".join(digits) + "0")
        if control is not None:
            return "".join(digits) + str(control)
        digits[-1] = str((int(digits[-1]) + 1) % 10)
    raise ValueError(f"не вдалося добудувати ЄДРПОУ для префікса {prefix7}")


# --------------------------------------------------------------------------
# Перевірки документа
# --------------------------------------------------------------------------

def parse_doc_date(raw: str) -> Optional[date]:
    """
    Дата з документа. Модель просять віддати ISO, але вона регулярно віддає
    те, що бачить у бланку, — тому приймаємо і поширені українські формати.
    """
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y.%m.%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


def _fmt(value: Optional[float]) -> Optional[str]:
    return None if value is None else f"{value:.2f}"


def validate_extraction(
    extraction: InvoiceExtraction,
    cfg: PipelineConfig,
    today: Optional[date] = None,
) -> list[ValidationIssue]:
    """Усі перевірки одного документа. Порядок — від реквізитів до сум."""
    today = today or date.today()
    issues: list[ValidationIssue] = []
    tol = cfg.amount_tolerance

    def add(code: str, severity: str, message: str, **kw) -> None:
        issues.append(
            ValidationIssue(code=code, severity=severity, message=message, **kw)  # type: ignore[arg-type]
        )

    # --- обов'язкові реквізити ---------------------------------------------
    for field, label in (
        ("supplier_name", "постачальник"),
        ("invoice_number", "номер документа"),
        ("invoice_date", "дата документа"),
        ("total", "загальна сума"),
    ):
        if getattr(extraction, field) in (None, ""):
            add(
                "missing_required",
                "error",
                f"У документі не знайдено обов'язкове поле: {label}",
                field=field,
            )

    # --- податкові коди ----------------------------------------------------
    for field, label in (("supplier_edrpou", "постачальника"), ("buyer_edrpou", "покупця")):
        code = getattr(extraction, field)
        if code is None:
            continue
        if len(code) not in (8, 10):
            add(
                "tax_code_length",
                "error",
                f"Код {label} має {len(code)} цифр — очікується 8 (ЄДРПОУ) або 10 (РНОКПП)",
                field=field,
                expected="8 або 10 цифр",
                actual=code,
            )
        elif not is_valid_tax_code(code):
            add(
                "tax_code_checksum",
                "error",
                f"Код {label} не проходить перевірку контрольної цифри — "
                "найімовірніше, одна цифра прочитана невірно",
                field=field,
                actual=code,
            )

    # --- дата --------------------------------------------------------------
    if extraction.invoice_date:
        parsed = parse_doc_date(extraction.invoice_date)
        if parsed is None:
            add(
                "date_unparseable",
                "error",
                "Дату документа не вдалося розібрати",
                field="invoice_date",
                expected="YYYY-MM-DD",
                actual=extraction.invoice_date,
            )
        elif parsed > today:
            add(
                "date_in_future",
                "error",
                "Дата документа — у майбутньому",
                field="invoice_date",
                expected=f"не пізніше {today.isoformat()}",
                actual=parsed.isoformat(),
            )
        elif parsed.year < MIN_PLAUSIBLE_YEAR:
            add(
                "date_implausible",
                "warning",
                "Рік документа виглядає неправдоподібно — можлива помилка розпізнавання",
                field="invoice_date",
                actual=parsed.isoformat(),
            )

    # --- валюта ------------------------------------------------------------
    if extraction.currency and extraction.currency not in KNOWN_CURRENCIES:
        add(
            "currency_unknown",
            "warning",
            "Невідомий код валюти",
            field="currency",
            expected="/".join(sorted(KNOWN_CURRENCIES)),
            actual=extraction.currency,
        )

    # --- таблична частина --------------------------------------------------
    if not extraction.line_items:
        add(
            "no_line_items",
            "warning",
            "У документі не розпізнано жодної позиції",
            field="line_items",
        )

    for idx, item in enumerate(extraction.line_items, start=1):
        if item.amount is not None and item.amount < 0:
            add(
                "negative_amount",
                "error",
                f"Рядок {idx}: від'ємна сума",
                field=f"line_items[{idx}].amount",
                actual=_fmt(item.amount),
            )
        if item.quantity is None or item.unit_price is None or item.amount is None:
            continue
        expected = round(item.quantity * item.unit_price, 2)
        if not _close(expected, item.amount, tol):
            add(
                "line_amount_mismatch",
                "error",
                f"Рядок {idx}: кількість × ціна не дорівнює сумі рядка",
                field=f"line_items[{idx}].amount",
                expected=_fmt(expected),
                actual=_fmt(item.amount),
            )

    # --- суми --------------------------------------------------------------
    amounts = [i.amount for i in extraction.line_items if i.amount is not None]
    if amounts and extraction.subtotal is not None:
        expected = round(sum(amounts), 2)
        # Толерантність росте з кількістю рядків: кожен рядок округлюють
        # окремо, тож на 20 позиціях законна розбіжність теж до 20 копійок.
        if not _close(expected, extraction.subtotal, tol * max(1, len(amounts))):
            add(
                "subtotal_mismatch",
                "error",
                "Сума позицій не дорівнює підсумку без ПДВ",
                field="subtotal",
                expected=_fmt(expected),
                actual=_fmt(extraction.subtotal),
            )

    if extraction.vat_rate is not None and extraction.vat_rate not in KNOWN_VAT_RATES:
        add(
            "vat_rate_unknown",
            "warning",
            "Нетипова ставка ПДВ",
            field="vat_rate",
            expected="0 / 7 / 14 / 20",
            actual=f"{extraction.vat_rate:g}",
        )

    if (
        extraction.subtotal is not None
        and extraction.vat_rate is not None
        and extraction.vat_amount is not None
    ):
        expected = round(extraction.subtotal * extraction.vat_rate / 100.0, 2)
        if not _close(expected, extraction.vat_amount, tol):
            add(
                "vat_mismatch",
                "error",
                "Сума ПДВ не відповідає ставці та базі оподаткування",
                field="vat_amount",
                expected=_fmt(expected),
                actual=_fmt(extraction.vat_amount),
            )

    if extraction.subtotal is not None and extraction.total is not None:
        vat = extraction.vat_amount or 0.0
        expected = round(extraction.subtotal + vat, 2)
        if not _close(expected, extraction.total, tol):
            add(
                "total_mismatch",
                "error",
                "Підсумок без ПДВ разом із ПДВ не дорівнює сумі до сплати",
                field="total",
                expected=_fmt(expected),
                actual=_fmt(extraction.total),
            )

    if extraction.total is not None:
        if extraction.total < 0:
            add(
                "negative_total",
                "error",
                "Загальна сума від'ємна",
                field="total",
                actual=_fmt(extraction.total),
            )
        elif extraction.total > MAX_PLAUSIBLE_TOTAL:
            add(
                "total_implausible",
                "warning",
                "Загальна сума неправдоподібно велика — можливий зсув розділювача",
                field="total",
                actual=_fmt(extraction.total),
            )

    return issues


def route(
    extraction: Optional[InvoiceExtraction],
    issues: list[ValidationIssue],
    cfg: PipelineConfig,
) -> tuple[str, list[str]]:
    """
    Рішення про долю документа: у облік автоматично чи в чергу на перевірку.

    Дві незалежні причини потрапити в чергу — і це навмисно різні сигнали:

    * **валідація** ловить те, що модель прочитала впевнено й неправильно
      (класика: акуратно надрукована сума, яка не сходиться з позиціями);
    * **впевненість** ловить те, що модель сама не змогла прочитати
      (зім'ятий кут, обрізаний скан) — там помилки ще немає, але довіри теж.

    Одне без іншого діряве, і eval це показує числом.
    """
    if extraction is None:
        return "fallback_error", ["екстракція не вдалася"]

    reasons: list[str] = []

    errors = [i for i in issues if i.severity == "error"]
    if errors:
        reasons.extend(f"{i.code}: {i.message}" for i in errors)

    low = extraction.confidence.below(cfg.min_field_confidence)
    if low:
        listed = ", ".join(f"{k}={v:.2f}" for k, v in sorted(low.items(), key=lambda kv: kv[1]))
        reasons.append(f"низька впевненість моделі: {listed}")

    return ("needs_review" if reasons else "auto_ok"), reasons
