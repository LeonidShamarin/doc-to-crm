"""
Стан пайплайна: ідемпотентність, черга ручної перевірки, рішення рев'ювера.

SQLite, а не «все в пам'яті», рівно з однієї причини: після перезапуску
сервісу пам'ять порожня, а тека з документами — ні. Watcher обробив би все
заново і створив дублі в обліку. Саме це відрізняє продакшн від демо.

Два рівні захисту від дублів — навмисно різні:

* **точний** — SHA-256 вмісту файлу в UNIQUE-індексі. Той самий файл під іншою
  назвою (`рахунок.pdf` → `рахунок (1).pdf`) не проходить двічі.
* **м'який** — пара «номер документа + ЄДРПОУ постачальника». Ловить випадок,
  коли той самий рахунок прийшов ще раз як перескан або фото: байти інші,
  документ той самий. Такий випадок не блокується мовчки, а йде рев'юверу —
  бо це може бути й легітимне виправлення.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.schema import InvoiceExtraction, ProcessedDocument, ValidationIssue

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    content_hash    TEXT NOT NULL UNIQUE,
    source_path     TEXT NOT NULL,
    mime_type       TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    status          TEXT NOT NULL,
    supplier_edrpou TEXT,
    invoice_number  TEXT,
    payload         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- М'який ключ дубля. Не UNIQUE: повторна поява того самого номера — привід
-- показати людині, а не привід відкинути запис.
CREATE INDEX IF NOT EXISTS idx_documents_softkey
    ON documents(supplier_edrpou, invoice_number);

-- Ключ — сурогатний id, а НЕ (doc_id, decided_at). Час тут із точністю до
-- секунди, і два рішення по одному документу в межах тієї самої секунди
-- (правка → одразу відхилення) затирали б одне одного через INSERT OR REPLACE.
-- Історія дій людини не має губитись із тієї ж причини, з якої не губиться
-- жоден вхідний документ.
CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT NOT NULL,
    decided_at   TEXT NOT NULL,
    decision     TEXT NOT NULL,
    reviewer     TEXT,
    comment      TEXT,
    corrections  TEXT
);

CREATE INDEX IF NOT EXISTS idx_reviews_doc ON reviews(doc_id);
"""


def file_hash(data: bytes) -> str:
    """SHA-256 вмісту. Ім'я файлу навмисно не бере участі."""
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI віддає запити з пулу потоків, а всі
        # записи тут короткі й проходять через один невеликий модуль.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL — щоб читання черги в UI не блокувалося записом watcher-а.
        if str(path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- пошук дублів ------------------------------------------------------

    def find_by_hash(self, content_hash: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        )
        return cur.fetchone()

    def find_soft_duplicate(
        self, supplier_edrpou: Optional[str], invoice_number: Optional[str], exclude_doc_id: str
    ) -> Optional[sqlite3.Row]:
        if not supplier_edrpou or not invoice_number:
            return None
        cur = self.conn.execute(
            "SELECT * FROM documents WHERE supplier_edrpou = ? AND invoice_number = ? "
            "AND doc_id != ? ORDER BY received_at LIMIT 1",
            (supplier_edrpou, invoice_number, exclude_doc_id),
        )
        return cur.fetchone()

    # -- запис -------------------------------------------------------------

    def save(self, doc: ProcessedDocument) -> None:
        """
        Записує документ. `INSERT OR REPLACE` по doc_id: повторний запис того
        самого документа після ручної правки має оновлювати запис, а не падати.
        """
        e = doc.extraction
        self.conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, content_hash, source_path, mime_type, received_at, status, "
            " supplier_edrpou, invoice_number, payload) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                doc.doc_id,
                doc.content_hash,
                doc.source_path,
                doc.mime_type,
                doc.received_at,
                doc.status,
                e.supplier_edrpou if e else None,
                e.invoice_number if e else None,
                doc.model_dump_json(),
            ),
        )
        self.conn.commit()

    def get(self, doc_id: str) -> Optional[ProcessedDocument]:
        cur = self.conn.execute("SELECT payload FROM documents WHERE doc_id = ?", (doc_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return ProcessedDocument.model_validate_json(row["payload"])

    def queue(self, limit: int = 100) -> list[ProcessedDocument]:
        """Черга на ручну перевірку — найстаріші першими."""
        cur = self.conn.execute(
            "SELECT payload FROM documents WHERE status IN ('needs_review','fallback_error') "
            "ORDER BY received_at LIMIT ?",
            (limit,),
        )
        return [ProcessedDocument.model_validate_json(r["payload"]) for r in cur.fetchall()]

    def all_documents(self, limit: int = 1000) -> list[ProcessedDocument]:
        cur = self.conn.execute(
            "SELECT payload FROM documents ORDER BY received_at LIMIT ?", (limit,)
        )
        return [ProcessedDocument.model_validate_json(r["payload"]) for r in cur.fetchall()]

    def counts_by_status(self) -> dict[str, int]:
        cur = self.conn.execute("SELECT status, COUNT(*) AS n FROM documents GROUP BY status")
        return {r["status"]: r["n"] for r in cur.fetchall()}

    # -- рішення рев'ювера -------------------------------------------------

    def record_review(
        self,
        doc_id: str,
        decision: str,
        reviewer: Optional[str] = None,
        comment: Optional[str] = None,
        corrections: Optional[dict] = None,
    ) -> ProcessedDocument:
        """
        Фіксує рішення людини і застосовує правки до збереженої екстракції.

        Історія рішень не перетирається (ключ — doc_id + час): якщо документ
        правили двічі, видно обидва рази. Це той самий принцип, що й «жоден
        запис не губиться», тільки для дій людини.
        """
        doc = self.get(doc_id)
        if doc is None:
            raise KeyError(f"документ {doc_id} не знайдено")

        if corrections and doc.extraction is not None:
            data = doc.extraction.model_dump()
            for key, value in corrections.items():
                if key in data:
                    data[key] = value
            doc.extraction = InvoiceExtraction.model_validate(data)

        if decision == "approve":
            doc.status = "auto_ok"
            doc.review_reasons = []
            # Правки людини — джерело правди; попередні розбіжності вже
            # враховані нею, тому лишається тільки слід у таблиці reviews.
            doc.issues = [i for i in doc.issues if i.severity == "warning"]
        elif decision == "reject":
            doc.status = "fallback_error"
            doc.processing_error = comment or "відхилено рев'ювером"
        else:
            raise ValueError(f"невідоме рішення: {decision}")

        self.conn.execute(
            "INSERT INTO reviews (doc_id, decided_at, decision, reviewer, comment, corrections) "
            "VALUES (?,?,?,?,?,?)",
            (
                doc_id,
                utc_now(),
                decision,
                reviewer,
                comment,
                json.dumps(corrections or {}, ensure_ascii=False),
            ),
        )
        self.save(doc)
        return doc

    def reviews_for(self, doc_id: str) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM reviews WHERE doc_id = ? ORDER BY id", (doc_id,))
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()


def duplicate_record(
    doc_id: str,
    content_hash: str,
    source_path: str,
    mime_type: str,
    existing_doc_id: str,
) -> ProcessedDocument:
    """
    Запис про повторну подачу. Створюється замість повторної екстракції —
    саме тут економиться виклик моделі, і саме тут не з'являється дубль у
    вивідному леджері.
    """
    return ProcessedDocument(
        doc_id=doc_id,
        source_path=source_path,
        content_hash=content_hash,
        mime_type=mime_type,
        received_at=utc_now(),
        status="duplicate",
        duplicate_of=existing_doc_id,
        issues=[
            ValidationIssue(
                code="duplicate_content",
                severity="warning",
                message=f"Файл із таким самим вмістом уже оброблено як {existing_doc_id}",
            )
        ],
    )
