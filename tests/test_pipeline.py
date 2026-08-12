"""Пайплайн: маршрутизація, ідемпотентність, поведінка при збоях."""

from __future__ import annotations

import pytest

from src.llm import DailyQuotaExceeded, ExtractionError
from src.pipeline import DocumentPipeline, UnsupportedDocument, detect_mime
from tests.conftest import FakeVisionClient, make_extraction

PDF = b"%PDF-1.4 fake bytes"


async def test_clean_document_goes_auto_and_reaches_sink(pipeline, sink):
    doc = await pipeline.process_bytes(PDF, "invoice.pdf", "application/pdf")
    assert doc.status == "auto_ok"
    assert doc.extraction is not None
    rows = sink.rows()
    assert len(rows) == 1 and rows[0]["total"] == 3846.00


async def test_same_file_twice_does_not_duplicate_in_ledger(pipeline, sink):
    first = await pipeline.process_bytes(PDF, "invoice.pdf", "application/pdf")
    second = await pipeline.process_bytes(PDF, "invoice-copy.pdf", "application/pdf")

    assert first.status == "auto_ok"
    assert second.status == "duplicate" and second.duplicate_of == first.doc_id
    assert len(sink.rows()) == 1, "повторна подача не має створювати другий рядок обліку"


async def test_duplicate_costs_no_model_call(store, sink, cfg):
    client = FakeVisionClient()
    pipe = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)
    await pipe.process_bytes(PDF, "a.pdf", "application/pdf")
    await pipe.process_bytes(PDF, "b.pdf", "application/pdf")
    assert len(client.calls) == 1, "дубль має відсікатись ДО виклику моделі"


async def test_same_invoice_number_from_other_file_is_flagged_not_dropped(store, sink, cfg):
    """
    Перескан того самого рахунку: байти інші, реквізити ті самі. Це не привід
    мовчки відкинути — це привід показати людині.
    """
    client = FakeVisionClient(default=make_extraction())
    pipe = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)
    await pipe.process_bytes(PDF, "scan1.pdf", "application/pdf")
    second = await pipe.process_bytes(PDF + b" rescanned", "scan2.pdf", "application/pdf")

    assert second.status == "needs_review"
    assert "duplicate_invoice_number" in {i.code for i in second.issues}


async def test_failed_extraction_still_produces_a_record(store, sink, cfg):
    """Жоден вхідний файл не губиться: провал стає записом, а не тишею."""
    client = FakeVisionClient(responses=[ExtractionError("модель не відповіла")])
    pipe = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)
    doc = await pipe.process_bytes(PDF, "broken.pdf", "application/pdf")

    assert doc.status == "fallback_error"
    assert "модель не відповіла" in doc.processing_error
    assert store.get(doc.doc_id) is not None
    assert sink.rows() == [], "провальна екстракція не має потрапляти в облік"


async def test_unexpected_sdk_error_is_caught_as_fallback(store, sink, cfg):
    client = FakeVisionClient(responses=[RuntimeError("щось несподіване з SDK")])
    pipe = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)
    doc = await pipe.process_bytes(PDF, "weird.pdf", "application/pdf")
    assert doc.status == "fallback_error" and "RuntimeError" in doc.processing_error


async def test_memory_error_is_not_swallowed(store, sink, cfg):
    """
    MemoryError — стан процесу, а не помилка документа. Якщо загорнути його у
    fallback, цикл поїде далі й доб'є машину.
    """
    client = FakeVisionClient(responses=[MemoryError()])
    pipe = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)
    with pytest.raises(MemoryError):
        await pipe.process_bytes(PDF, "huge.pdf", "application/pdf")


async def test_daily_quota_stops_the_batch(store, sink, cfg, tmp_path):
    """Денна квота зупиняє прогін цілком, а не малює fallback на кожен файл."""
    files = []
    for n in range(4):
        path = tmp_path / f"doc{n}.pdf"
        path.write_bytes(PDF + str(n).encode())
        files.append(path)

    client = FakeVisionClient(
        responses=[make_extraction(), DailyQuotaExceeded("PerDay"), make_extraction()]
    )
    pipe = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)
    docs = await pipe.process_many(files, concurrency=1)

    assert len(docs) < len(files)
    assert all(d.status != "fallback_error" for d in docs)


async def test_low_confidence_document_goes_to_review(store, sink, cfg):
    extraction = make_extraction()
    extraction.confidence.supplier_edrpou = 0.3
    pipe = DocumentPipeline(client=FakeVisionClient(default=extraction), store=store,
                            sink=sink, cfg=cfg)
    doc = await pipe.process_bytes(PDF, "blurry.jpg", "image/jpeg")
    assert doc.status == "needs_review"
    assert sink.rows() == [], "документ, що чекає на людину, в обліку ще не існує"
    assert store.get(doc.doc_id) is not None, "але з черги він не зникає"


async def test_validation_can_be_disabled_for_comparison(store, sink, cfg):
    """Вимикач потрібен рівно для одного рядка таблиці в eval."""
    cfg.validation_enabled = False
    pipe = DocumentPipeline(client=FakeVisionClient(default=make_extraction(total=9999.0)),
                            store=store, sink=sink, cfg=cfg)
    doc = await pipe.process_bytes(PDF, "x.pdf", "application/pdf")
    assert doc.issues == [] and doc.status == "auto_ok"


async def test_unsupported_extension_rejected(tmp_path, pipeline):
    path = tmp_path / "notes.txt"
    path.write_text("не документ", encoding="utf-8")
    with pytest.raises(UnsupportedDocument):
        await pipeline.process_path(path)


def test_detect_mime_maps_known_types():
    from pathlib import Path

    assert detect_mime(Path("a.PDF")) == "application/pdf"
    assert detect_mime(Path("b.jpeg")) == "image/jpeg"


async def test_summary_counts_only_successful_documents(store, sink, cfg):
    client = FakeVisionClient(responses=[make_extraction(), ExtractionError("збій")])
    pipe = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)
    docs = [
        await pipe.process_bytes(PDF, "a.pdf", "application/pdf"),
        await pipe.process_bytes(PDF + b"2", "b.pdf", "application/pdf"),
    ]
    summary = pipe.summary(docs)

    assert summary["documents"] == 2
    assert summary["by_status"]["fallback_error"] == 1
    assert summary["cost_usd"]["per_1000_docs"] > 0
