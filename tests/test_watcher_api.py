"""Watcher теки і HTTP-шар."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.pipeline import DocumentPipeline
from src.watcher import StabilityTracker, scan_folder, watch_folder
from tests.conftest import FakeVisionClient, make_extraction

PDF = b"%PDF-1.4 fake"


# --- watcher --------------------------------------------------------------

def test_scan_folder_filters_and_limits(tmp_path):
    (tmp_path / "a.pdf").write_bytes(PDF)
    (tmp_path / "b.jpg").write_bytes(b"jpg")
    (tmp_path / "c.txt").write_text("ні", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.pdf").write_bytes(PDF)

    names = {p.name for p in scan_folder(tmp_path)}
    assert names == {"a.pdf", "b.jpg"}, "без рекурсії і без чужих розширень"
    assert len(scan_folder(tmp_path, max_files=1)) == 1


def test_scan_missing_folder_is_empty(tmp_path):
    assert scan_folder(tmp_path / "немає") == []


def test_file_is_processed_only_after_it_stops_growing(tmp_path):
    """Наполовину скопійований скан не має потрапити в модель."""
    path = tmp_path / "a.pdf"
    path.write_bytes(b"part")
    tracker = StabilityTracker()

    assert tracker.is_stable(path) is False, "перше бачення — ще не підтвердження"
    path.write_bytes(b"part + more")
    assert tracker.is_stable(path) is False, "розмір змінився — файл ще пишеться"
    assert tracker.is_stable(path) is True


def test_tracker_forgets_removed_files(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(PDF)
    tracker = StabilityTracker()
    tracker.is_stable(path)
    tracker.prune(set())
    assert tracker._seen == {}, "словник не має рости весь час роботи сервісу"


async def test_watch_processes_new_file_once(tmp_path, store, sink, cfg):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.pdf").write_bytes(PDF)

    client = FakeVisionClient()
    pipe = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)
    # Три оберти: перший фіксує файл, другий обробляє, третій має пройти вхолосту.
    processed = await watch_folder(inbox, pipe, interval=0, max_cycles=3)

    assert processed == 1
    assert len(client.calls) == 1, "той самий файл не має оброблятись повторно"


async def test_watch_moves_processed_file_aside(tmp_path, store, sink, cfg):
    inbox, done = tmp_path / "inbox", tmp_path / "done"
    inbox.mkdir()
    (inbox / "a.pdf").write_bytes(PDF)
    pipe = DocumentPipeline(client=FakeVisionClient(), store=store, sink=sink, cfg=cfg)

    await watch_folder(inbox, pipe, interval=0, max_cycles=2, processed_dir=done)
    assert (done / "a.pdf").exists() and not (inbox / "a.pdf").exists()


async def test_watch_stops_after_max_cycles(tmp_path, pipeline):
    """
    Цикл має явний вихід. Без max_cycles тест не завершився б ніколи — і це
    рівно та сама властивість, яка в проді не дає watcher-у зациклитись мовчки.
    """
    assert await watch_folder(tmp_path, pipeline, interval=0, max_cycles=2) == 0


# --- HTTP -----------------------------------------------------------------

@pytest.fixture
def client(store, sink, cfg):
    # Стор і леджер передаються явно: API має бачити те саме сховище, у яке
    # пише пайплайн, інакше завантажений документ не з'явився б у черзі.
    pipe = DocumentPipeline(client=FakeVisionClient(), store=store, sink=sink, cfg=cfg)
    return TestClient(create_app(pipeline=pipe, store=store, sink=sink))


def test_health_reports_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["extraction_enabled"] is True


def test_upload_returns_processed_document(client):
    response = client.post("/documents", files={"file": ("inv.pdf", PDF, "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "auto_ok" and body["extraction"]["total"] == 3846.00


def test_upload_rejects_unsupported_type(client):
    response = client.post("/documents", files={"file": ("a.txt", b"text", "text/plain")})
    assert response.status_code == 415


def test_upload_rejects_empty_file(client):
    response = client.post("/documents", files={"file": ("a.pdf", b"", "application/pdf")})
    assert response.status_code == 400


def test_service_without_key_answers_503(tmp_path, monkeypatch):
    """Без ключа сервіс піднімається, але не вдає, що вміє екстракцію."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = create_app(db_path=tmp_path / "n.sqlite3", sink_path=tmp_path / "n.jsonl")
    with TestClient(app) as no_key:
        assert no_key.get("/health").json()["extraction_enabled"] is False
        assert no_key.post("/documents",
                           files={"file": ("a.pdf", PDF, "application/pdf")}).status_code == 503


def test_queue_and_review_roundtrip(store, sink, cfg):
    extraction = make_extraction()
    extraction.confidence.total = 0.2
    pipe = DocumentPipeline(client=FakeVisionClient(default=extraction), store=store,
                            sink=sink, cfg=cfg)
    api = TestClient(create_app(pipeline=pipe, store=store, sink=sink))

    doc_id = api.post("/documents", files={"file": ("inv.pdf", PDF, "application/pdf")}).json()["doc_id"]
    queue = api.get("/queue").json()
    assert queue["count"] == 1 and queue["items"][0]["doc_id"] == doc_id

    approved = api.post(f"/review/{doc_id}",
                        json={"decision": "approve", "corrections": {"total": 4000.0}}).json()
    assert approved["status"] == "auto_ok" and approved["extraction"]["total"] == 4000.0
    assert api.get("/queue").json()["count"] == 0


def test_review_of_unknown_document_is_404(client):
    assert client.post("/review/немає", json={"decision": "approve"}).status_code == 404


def test_review_rejects_unknown_decision(client):
    assert client.post("/review/x", json={"decision": "хтозна"}).status_code == 400
