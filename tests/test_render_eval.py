"""Генератор датасету і підрахунок метрик."""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.config import PipelineConfig
from src.evaluate import (
    aggregate,
    build_records,
    compare_document,
    compare_line_items,
    routing_metrics,
    values_match,
)
from src.render import amount_in_words, build_document, fmt_amount, generate_dataset
from src.schema import ProcessedDocument
from src.validate import is_valid_tax_code, validate_extraction
from src.store import file_hash, utc_now
from tests.conftest import make_extraction

import random

TODAY = date(2026, 8, 12)


# --- форматування ---------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [(1205.0, "1 205,00"), (0.5, "0,50"), (1234567.89, "1 234 567,89")],
)
def test_fmt_amount_uses_ukrainian_style(value, expected):
    assert fmt_amount(value) == expected


@pytest.mark.parametrize(
    "value,tail",
    [
        (181.06, "одна гривня 06 коп."),
        (4.99, "чотири гривні 99 коп."),
        (51664.66, "чотири гривні 66 коп."),
        (1000.0, "одна тисяча гривень 00 коп."),
    ],
)
def test_amount_in_words_agrees_in_gender_and_number(value, tail):
    """
    «Гривня» жіночого роду, тому останній розряд теж жіночий, а форму слова
    визначає ціла частина: 4,99 — це «чотири гривні», а не «п'ять гривень».
    """
    assert amount_in_words(value).lower().endswith(tail)


# --- генератор ------------------------------------------------------------

def test_generated_document_is_internally_consistent():
    doc = build_document(random.Random(1), 1, TODAY)
    f = doc.fields
    assert round(sum(i["amount"] for i in f["line_items"]), 2) == f["subtotal"]
    assert round(f["subtotal"] + f["vat_amount"], 2) == f["total"]
    assert is_valid_tax_code(f["supplier_edrpou"])
    assert is_valid_tax_code(f["buyer_edrpou"])


def test_line_amounts_equal_quantity_times_price():
    doc = build_document(random.Random(5), 1, TODAY)
    for item in doc.fields["line_items"]:
        assert round(item["quantity"] * item["unit_price"], 2) == item["amount"]


def test_generation_is_deterministic(tmp_path):
    """Той самий seed — той самий датасет, інакше числа в README не відтворити."""
    args = dict(count=3, photo_ratio=0.0, seed=42, today=TODAY)
    first = generate_dataset(out_dir=tmp_path / "a", golden_path=tmp_path / "a.jsonl", **args)
    second = generate_dataset(out_dir=tmp_path / "b", golden_path=tmp_path / "b.jsonl", **args)
    assert [d.fields for d in first] == [d.fields for d in second]

    def golden_without_paths(path):
        # Шлях до файлу відрізняється за побудовою (різні теки) — порівнюємо
        # усе інше, тобто власне зміст еталона.
        return [
            {k: v for k, v in json.loads(line).items() if k != "file"}
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    assert golden_without_paths(tmp_path / "a.jsonl") == golden_without_paths(tmp_path / "b.jsonl")


def test_generated_files_and_golden_match(tmp_path):
    docs = generate_dataset(out_dir=tmp_path / "d", golden_path=tmp_path / "g.jsonl",
                            count=4, photo_ratio=0.5, seed=7, today=TODAY)
    golden = [json.loads(l) for l in (tmp_path / "g.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(golden) == len(docs) == 4
    for entry in golden:
        from pathlib import Path

        assert Path(entry["file"]).exists(), "еталон посилається на файл, якого немає"
        assert Path(entry["file"]).stat().st_size > 0


def test_planted_issues_are_detectable_by_validator(tmp_path):
    """
    Ключова перевірка датасету: закладена невідповідність має ловитись тим
    самим шаром валідації, який працює в проді. Інакше eval міряв би себе.
    """
    docs = generate_dataset(out_dir=tmp_path / "d", golden_path=tmp_path / "g.jsonl",
                            count=12, photo_ratio=0.0, seed=11, today=TODAY)
    planted = [d for d in docs if d.planted_issue]
    assert planted, "у наборі має бути хоча б одна підміна"

    for doc in planted:
        extraction = make_extraction(
            **{
                "supplier_edrpou": doc.fields["supplier_edrpou"],
                "invoice_date": doc.fields["invoice_date"],
                "subtotal": doc.fields["subtotal"],
                "vat_rate": doc.fields["vat_rate"],
                "vat_amount": doc.fields["vat_amount"],
                "total": doc.fields["total"],
                "line_items": [
                    {"description": i["description"], "quantity": i["quantity"],
                     "unit": i["unit"], "unit_price": i["unit_price"], "amount": i["amount"]}
                    for i in doc.fields["line_items"]
                ],
            }
        )
        codes = {i.code for i in validate_extraction(extraction, PipelineConfig(), today=TODAY)}
        assert doc.planted_issue in codes, f"{doc.doc_id}: {doc.planted_issue} не спіймано"


# --- порівняння з еталоном ------------------------------------------------

@pytest.mark.parametrize(
    "field,expected,actual,ok",
    [
        ("supplier_name", "ТОВ «Агро»", "ТОВ Агро", True),
        ("supplier_name", "ТОВ «Агро»", "ТОВ «Агрос»", False),
        ("supplier_edrpou", "30123453", "ЄДРПОУ 30123453", True),
        ("invoice_number", "СФ-100", "№ СФ-100", True),
        ("invoice_date", "2026-03-05", "05.03.2026", True),
        ("invoice_date", "2026-03-05", "2026-05-03", False),
        ("total", 3846.0, 3846.004, True),
        ("total", 3846.0, 3846.5, False),
        ("total", None, None, True),
        ("total", 10.0, None, False),
    ],
)
def test_values_match_normalises_the_right_things(field, expected, actual, ok):
    assert values_match(field, expected, actual) is ok


def test_line_items_compared_by_amounts_not_by_wording():
    expected = [{"amount": 100.0}, {"amount": 250.5}]
    extraction = make_extraction()
    extraction.line_items[0].amount = 250.5
    extraction.line_items[1].amount = 100.0
    result = compare_line_items(expected, extraction.line_items)
    assert result["all_match"] and result["amounts_matched"] == 2


def test_missing_extraction_counts_as_all_fields_wrong():
    result = compare_document({"line_items": []}, None)
    assert result["fully_correct"] is False
    assert all(v is False for v in result["fields"].values())


def test_routing_metrics_count_misses_and_false_alarms():
    records = [
        {"should_review": True, "r": "needs_review"},   # спіймали
        {"should_review": True, "r": "auto_ok"},        # пропустили
        {"should_review": False, "r": "needs_review"},  # хибна тривога
        {"should_review": False, "r": "auto_ok"},       # чисто
    ]
    m = routing_metrics(records, "r")
    assert (m["caught"], m["missed"], m["false_alarms"], m["clean_auto"]) == (1, 1, 1, 1)
    assert m["recall"] == 0.5 and m["precision"] == 0.5


def _doc_from(extraction, doc_id="doc-001", source="data/synthetic/doc-001.pdf"):
    return ProcessedDocument(
        doc_id=doc_id, source_path=source, content_hash=file_hash(b"x"),
        mime_type="application/pdf", received_at=utc_now(), extraction=extraction,
        status="auto_ok", latency_s=1.5, input_tokens=1000, output_tokens=200,
    )


def test_ablation_variants_are_computed_from_one_run():
    """
    Три конфігурації маршрутизації рахуються з ОДНІЄЇ екстракції — саме тому
    таблиця порівняння коштує один прогін, а не три.
    """
    extraction = make_extraction(total=9999.0)      # арифметика не сходиться
    extraction.confidence.total = 0.99              # але модель упевнена
    golden = {
        "doc-001": {
            "doc_id": "doc-001", "kind": "pdf", "template": "classic", "planted_issue": None,
            "fields": {**make_extraction().model_dump(), "line_items": [
                {"description": "Папір A4", "quantity": 10, "unit": "пач.",
                 "unit_price": 120.5, "amount": 1205.0},
                {"description": "Картридж", "quantity": 2, "unit": "шт.",
                 "unit_price": 1000.0, "amount": 2000.0}]},
        }
    }
    records = build_records([_doc_from(extraction)], golden, PipelineConfig())

    assert len(records) == 1
    rec = records[0]
    assert rec["route_full"] == "needs_review"
    assert rec["route_conf_only"] == "auto_ok", "сама впевненість цю помилку не ловить"
    assert rec["route_valid_only"] == "needs_review"


def test_aggregate_reports_cost_and_roi():
    extraction = make_extraction()
    golden = {
        "doc-001": {
            "doc_id": "doc-001", "kind": "pdf", "template": "classic", "planted_issue": None,
            "fields": {**extraction.model_dump(), "line_items": [
                {"description": "Папір A4", "quantity": 10, "unit": "пач.",
                 "unit_price": 120.5, "amount": 1205.0},
                {"description": "Картридж", "quantity": 2, "unit": "шт.",
                 "unit_price": 1000.0, "amount": 2000.0}]},
        }
    }
    cfg = PipelineConfig()
    report = aggregate(build_records([_doc_from(extraction)], golden, cfg), cfg)

    assert report["documents"] == 1 and report["fully_correct"] == 1.0
    assert report["cost_usd"]["per_1000_docs"] > 0
    assert report["roi"]["saved_hours_per_1000"] > 0
