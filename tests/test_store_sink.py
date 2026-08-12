"""Стан: SQLite-стор, рішення рев'ювера, JSONL-леджер."""

from __future__ import annotations

import json

import pytest

from src.schema import ProcessedDocument
from src.sinks import JsonlSink
from src.store import DocumentStore, file_hash, utc_now
from tests.conftest import make_extraction


def make_doc(doc_id="doc-1", content=b"pdf", status="needs_review", **kw) -> ProcessedDocument:
    return ProcessedDocument(
        doc_id=doc_id,
        source_path=f"{doc_id}.pdf",
        content_hash=file_hash(content),
        mime_type="application/pdf",
        received_at=utc_now(),
        extraction=kw.pop("extraction", make_extraction()),
        status=status,
        **kw,
    )


def test_hash_ignores_filename():
    assert file_hash(b"same bytes") == file_hash(b"same bytes")
    assert file_hash(b"a") != file_hash(b"b")


def test_save_and_reload_roundtrip(store):
    doc = make_doc()
    store.save(doc)
    loaded = store.get(doc.doc_id)
    assert loaded is not None
    assert loaded.extraction.total == doc.extraction.total
    assert loaded.content_hash == doc.content_hash


def test_find_by_hash(store):
    doc = make_doc()
    store.save(doc)
    assert store.find_by_hash(doc.content_hash)["doc_id"] == doc.doc_id
    assert store.find_by_hash("невідомий") is None


def test_soft_duplicate_needs_both_parts(store):
    store.save(make_doc("doc-1"))
    assert store.find_soft_duplicate("30123453", "СФ-100", "doc-2")["doc_id"] == "doc-1"
    # без одного зі складників ключа м'який дубль не спрацьовує — інакше всі
    # документи без номера злиплися б в одну «пару»
    assert store.find_soft_duplicate(None, "СФ-100", "doc-2") is None
    assert store.find_soft_duplicate("30123453", None, "doc-2") is None
    assert store.find_soft_duplicate("30123453", "СФ-100", "doc-1") is None


def test_queue_returns_only_unresolved(store):
    store.save(make_doc("a", b"1", status="auto_ok"))
    store.save(make_doc("b", b"2", status="needs_review"))
    store.save(make_doc("c", b"3", status="fallback_error"))
    store.save(make_doc("d", b"4", status="duplicate"))
    assert {d.doc_id for d in store.queue()} == {"b", "c"}


def test_review_applies_corrections_and_approves(store):
    store.save(make_doc("a", b"1", status="needs_review"))
    updated = store.record_review("a", "approve", reviewer="olha",
                                  corrections={"total": 4000.0, "invoice_number": "СФ-101"})
    assert updated.status == "auto_ok"
    assert updated.extraction.total == 4000.0
    assert store.get("a").extraction.invoice_number == "СФ-101"


def test_review_history_is_not_overwritten(store):
    store.save(make_doc("a", b"1"))
    store.record_review("a", "approve", reviewer="olha", corrections={"total": 1.0})
    store.record_review("a", "reject", reviewer="ihor", comment="не той контрагент")
    history = store.reviews_for("a")
    assert [r["decision"] for r in history] == ["approve", "reject"]


def test_reject_marks_document_as_failed(store):
    store.save(make_doc("a", b"1"))
    doc = store.record_review("a", "reject", comment="скан нечитабельний")
    assert doc.status == "fallback_error" and "нечитабельний" in doc.processing_error


def test_review_of_unknown_document_raises(store):
    with pytest.raises(KeyError):
        store.record_review("немає", "approve")


def test_unknown_decision_raises(store):
    store.save(make_doc("a", b"1"))
    with pytest.raises(ValueError):
        store.record_review("a", "maybe")


# --- леджер ---------------------------------------------------------------

def test_sink_writes_row_and_dedupes(tmp_path):
    sink = JsonlSink(tmp_path / "ledger.jsonl")
    doc = make_doc(status="auto_ok")
    assert sink.write(doc) is True
    assert sink.write(doc) is False
    assert len(sink.rows()) == 1


def test_sink_takes_only_confirmed_documents(tmp_path):
    """
    В облік іде лише те, що пройшло перевірки або підтверджене людиною.
    Дубль, провал і документ у черзі туди не належать.
    """
    sink = JsonlSink(tmp_path / "ledger.jsonl")
    assert sink.write(make_doc("a", b"1", status="duplicate")) is False
    assert sink.write(make_doc("b", b"2", status="fallback_error")) is False
    assert sink.write(make_doc("c", b"3", status="needs_review")) is False
    assert sink.rows() == []


def test_corrections_reach_the_ledger_after_approval(tmp_path, store):
    """
    Регресія: поки документ у черзі, рядка в леджері немає. Якби він там був,
    дедуплікація за хешем не пустила б виправлену версію, і в обліку назавжди
    лишилися б непідтверджені числа.
    """
    sink = JsonlSink(tmp_path / "ledger.jsonl")
    doc = make_doc("a", b"1", status="needs_review")
    store.save(doc)
    sink.write(doc)
    assert sink.rows() == []

    approved = store.record_review("a", "approve", corrections={"total": 4242.0})
    assert sink.write(approved) is True
    assert sink.rows()[0]["total"] == 4242.0


def test_sink_reloads_hashes_after_restart(tmp_path):
    path = tmp_path / "ledger.jsonl"
    doc = make_doc(status="auto_ok")
    JsonlSink(path).write(doc)

    restarted = JsonlSink(path)
    assert restarted.contains(doc.content_hash)
    assert restarted.write(doc) is False, "після перезапуску дедуплікація має вижити"


def test_sink_survives_truncated_last_line(tmp_path):
    """Обірваний після kill -9 рядок не має ламати старт сервісу."""
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        json.dumps({"content_hash": "abc", "doc_id": "a"}, ensure_ascii=False) + "\n{\"broke",
        encoding="utf-8",
    )
    sink = JsonlSink(path)
    assert sink.contains("abc")


def test_row_is_flat_and_carries_min_confidence(tmp_path):
    extraction = make_extraction()
    extraction.confidence.vat_amount = 0.42
    row = make_doc(status="auto_ok", extraction=extraction).to_row()
    assert row["min_confidence"] == 0.42 and row["min_confidence_field"] == "vat_amount"
    assert row["line_items_count"] == 2
    assert not any(isinstance(v, dict) for v in row.values())
