"""Шар перевірок поза LLM — найважливіша частина проекту, тому й тестів найбільше."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.config import PipelineConfig
from src.schema import LineItem
from src.validate import (
    edrpou_control_digit,
    is_valid_edrpou,
    is_valid_ipn,
    is_valid_tax_code,
    make_valid_edrpou,
    make_valid_ipn,
    parse_doc_date,
    route,
    validate_extraction,
)
from tests.conftest import make_extraction

TODAY = date(2026, 8, 12)


def codes(issues) -> set[str]:
    return {i.code for i in issues}


# --- контрольні цифри -----------------------------------------------------

@pytest.mark.parametrize("code", ["14360570", "21560045", "20077720", "00032129", "37193071"])
def test_real_edrpou_codes_pass(code):
    """Публічні коди реальних юросіб — перевірка самого алгоритму, не даних."""
    assert is_valid_edrpou(code)


def test_edrpou_uses_shifted_weights_in_middle_range():
    """
    Коди 30000000–60000000 рахуються іншими вагами. Якщо переплутати гілки,
    половина реальних кодів «зламається» — тому діапазон перевіряється явно.
    """
    assert edrpou_control_digit("37193071") == 1
    assert is_valid_edrpou("37193071")


def test_single_digit_error_is_detected():
    """Сенс контрольної цифри: одна невірно прочитана цифра має ловитись."""
    code = "14360570"
    broken = [code[:i] + str((int(code[i]) + 1) % 10) + code[i + 1:] for i in range(7)]
    assert not any(is_valid_edrpou(b) for b in broken)


def test_generated_codes_are_valid():
    assert all(is_valid_edrpou(make_valid_edrpou(f"{i:07d}")) for i in range(0, 9_999_999, 1013))


def test_ipn_roundtrip_and_length_rules():
    ipn = make_valid_ipn("301611207")
    assert len(ipn) == 10 and is_valid_ipn(ipn)
    assert is_valid_tax_code(ipn)
    assert not is_valid_tax_code("123")
    assert not is_valid_tax_code("1234567890123")


# --- дати -----------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-03-05", date(2026, 3, 5)),
        ("05.03.2026", date(2026, 3, 5)),
        ("05/03/2026", date(2026, 3, 5)),
        ("2026.03.05", date(2026, 3, 5)),
        ("31.02.2026", None),
        ("хтозна", None),
    ],
)
def test_parse_doc_date(raw, expected):
    assert parse_doc_date(raw) == expected


def test_future_date_is_error():
    extraction = make_extraction(invoice_date=(TODAY + timedelta(days=3)).isoformat())
    assert "date_in_future" in codes(validate_extraction(extraction, PipelineConfig(), today=TODAY))


def test_unparseable_date_is_error_not_exception():
    """
    Дата приходить від моделі рядком навмисно: з типом `date` Pydantic зронив
    би весь документ, і причина лишилась би невидимою.
    """
    issues = validate_extraction(make_extraction(invoice_date="31.02.2026"), PipelineConfig())
    assert "date_unparseable" in codes(issues)


# --- арифметика -----------------------------------------------------------

def test_clean_document_has_no_errors():
    issues = validate_extraction(make_extraction(), PipelineConfig(), today=TODAY)
    assert [i for i in issues if i.severity == "error"] == []


def test_subtotal_mismatch_detected_with_expected_value():
    issues = validate_extraction(make_extraction(subtotal=3285.00, vat_amount=657.00,
                                                 total=3942.00), PipelineConfig(), today=TODAY)
    found = [i for i in issues if i.code == "subtotal_mismatch"]
    assert found and found[0].expected == "3205.00" and found[0].actual == "3285.00"


def test_line_amount_mismatch_detected():
    items = [LineItem(description="Папір", quantity=10, unit="пач.",
                      unit_price=120.50, amount=1250.00)]
    issues = validate_extraction(
        make_extraction(line_items=items, subtotal=1250.00, vat_amount=250.00, total=1500.00),
        PipelineConfig(), today=TODAY,
    )
    assert "line_amount_mismatch" in codes(issues)


def test_vat_and_total_mismatch_detected():
    """ПДВ не від бази (3205 × 20% = 641) і підсумок не дорівнює сумі частин."""
    issues = validate_extraction(make_extraction(vat_amount=600.00, total=3900.00),
                                 PipelineConfig(), today=TODAY)
    assert {"vat_mismatch", "total_mismatch"} <= codes(issues)


def test_rounding_tolerance_scales_with_line_count():
    """
    Кожен рядок округлюють окремо, тому на багатьох позиціях законна
    розбіжність у копійки накопичується. Толерантність має рости — інакше
    шар валідації почне сипати хибними тривогами на нормальних документах.
    """
    items = [
        LineItem(description=f"Позиція {n}", quantity=3, unit="шт.",
                 unit_price=33.33, amount=99.99)
        for n in range(10)
    ]
    extraction = make_extraction(line_items=items, subtotal=999.98, vat_rate=0.0,
                                 vat_amount=0.0, total=999.98)
    assert "subtotal_mismatch" not in codes(
        validate_extraction(extraction, PipelineConfig(), today=TODAY)
    )


def test_zero_vat_document_is_clean():
    extraction = make_extraction(vat_rate=0.0, vat_amount=0.0, total=3205.00)
    assert [i for i in validate_extraction(extraction, PipelineConfig(), today=TODAY)
            if i.severity == "error"] == []


def test_missing_required_fields_reported_per_field():
    issues = validate_extraction(
        make_extraction(supplier_name=None, invoice_number=None), PipelineConfig(), today=TODAY
    )
    missing = {i.field for i in issues if i.code == "missing_required"}
    assert missing == {"supplier_name", "invoice_number"}


def test_broken_tax_code_checksum_is_error():
    issues = validate_extraction(make_extraction(supplier_edrpou="30123454"),
                                 PipelineConfig(), today=TODAY)
    assert "tax_code_checksum" in codes(issues)


def test_implausible_total_is_warning_not_error():
    """Зсув коми не має блокувати запис мовчки, але й не має валити прогін."""
    extraction = make_extraction(subtotal=3_205_000_000.0, vat_rate=0.0, vat_amount=0.0,
                                 total=3_205_000_000.0)
    warnings = {i.code for i in validate_extraction(extraction, PipelineConfig(), today=TODAY)
                if i.severity == "warning"}
    assert "total_implausible" in warnings


# --- маршрутизація --------------------------------------------------------

def test_clean_and_confident_goes_auto():
    extraction = make_extraction()
    status, reasons = route(extraction, [], PipelineConfig())
    assert status == "auto_ok" and reasons == []


def test_low_confidence_alone_sends_to_review():
    extraction = make_extraction()
    extraction.confidence.total = 0.4
    status, reasons = route(extraction, [], PipelineConfig())
    assert status == "needs_review" and "total=0.40" in reasons[0]


def test_validation_error_sends_to_review_even_when_model_is_sure():
    """
    Головний сценарій продукту: модель упевнена на 0.99, а сума не сходиться.
    Саме тут «перевіряє Python, а не модель» перестає бути гаслом.
    """
    extraction = make_extraction(total=9999.00)
    issues = validate_extraction(extraction, PipelineConfig(), today=TODAY)
    status, reasons = route(extraction, issues, PipelineConfig())
    assert status == "needs_review" and any("total_mismatch" in r for r in reasons)


def test_no_extraction_routes_to_fallback():
    status, reasons = route(None, [], PipelineConfig())
    assert status == "fallback_error" and reasons
