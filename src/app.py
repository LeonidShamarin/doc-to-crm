"""
HTTP-сервіс: приймання документів і review-UI.

Логіка тут не живе — тільки транспорт. Обробка — у `pipeline`, стан — у
`store`. Причина проста: те саме приймання має працювати і з теки (`watch`),
і з CLI (`process`), і по HTTP; якщо маршрутизація документа опиниться в
обробнику запиту, два з трьох входів поводитимуться інакше.

Сервіс піднімається БЕЗ ключа Gemini: черга, review-UI і рішення людини
працюють, `POST /documents` віддає 503. Показувати порожню форму замість
екстракції означало б робити демо, яке виглядає працюючим і не є ним.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.config import config_from_env
from src.llm import DailyQuotaExceeded, GeminiVisionClient
from src.pipeline import SUPPORTED_TYPES, MAX_FILE_BYTES, DocumentPipeline
from src.sinks import JsonlSink
from src.store import DocumentStore

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ReviewRequest(BaseModel):
    decision: str
    reviewer: Optional[str] = None
    comment: Optional[str] = None
    corrections: dict = {}


def create_app(
    db_path: Path | str = "state/documents.sqlite3",
    sink_path: Path | str = "output/ledger.jsonl",
    pipeline: Optional[DocumentPipeline] = None,
    store: Optional[DocumentStore] = None,
    sink: Optional[JsonlSink] = None,
) -> FastAPI:
    """
    `pipeline`, `store` і `sink` передаються ззовні у тестах — з фейковим
    клієнтом замість Gemini. У проді збираються тут, із ключем з оточення.

    Передавати сховище окремо доводиться тому, що обробники тримають його в
    замиканні: підміна `app.state.store` після створення застосунку виглядала б
    робочою і мовчки не діяла б.
    """
    app = FastAPI(title="doc-to-crm", version="1.0")
    store = store if store is not None else DocumentStore(db_path)
    sink = sink if sink is not None else JsonlSink(sink_path)
    cfg = config_from_env()

    if pipeline is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if api_key:
            client = GeminiVisionClient(
                api_key=api_key, model_name=cfg.model, rpm=cfg.rpm, temperature=cfg.temperature
            )
            pipeline = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)
        else:
            logger.warning("GEMINI_API_KEY не заданий — приймання документів вимкнено")

    app.state.store = store
    app.state.sink = sink
    app.state.pipeline = pipeline
    app.state.cfg = cfg

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "extraction_enabled": pipeline is not None,
            "model": cfg.model,
            "min_field_confidence": cfg.min_field_confidence,
            "documents": store.counts_by_status(),
        }

    @app.post("/documents")
    async def upload(file: UploadFile = File(...)) -> JSONResponse:
        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="екстракція вимкнена: не заданий GEMINI_API_KEY",
            )
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"підтримуються {', '.join(sorted(SUPPORTED_TYPES))}",
            )
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="порожній файл")
        if len(data) > MAX_FILE_BYTES:
            # Ліміт перевіряється ПІСЛЯ читання, бо UploadFile уже на диску у
            # тимчасовому файлі; сенс перевірки — не пустити великий скан у
            # модель і в пам'ять пайплайна.
            raise HTTPException(
                status_code=413,
                detail=f"файл більший за {MAX_FILE_BYTES // 1024 // 1024} МБ",
            )
        try:
            doc = await pipeline.process_bytes(data, file.filename or "upload", SUPPORTED_TYPES[suffix])
        except DailyQuotaExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return JSONResponse(doc.model_dump(mode="json"))

    @app.get("/queue")
    def queue(limit: int = 50) -> dict:
        docs = store.queue(limit=limit)
        return {
            "count": len(docs),
            "threshold": cfg.min_field_confidence,
            "items": [d.model_dump(mode="json") for d in docs],
        }

    @app.get("/documents")
    def list_documents(limit: int = 200) -> dict:
        docs = store.all_documents(limit=limit)
        return {"count": len(docs), "items": [d.to_row() for d in docs]}

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str) -> dict:
        doc = store.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="документ не знайдено")
        return {"document": doc.model_dump(mode="json"), "reviews": store.reviews_for(doc_id)}

    @app.post("/review/{doc_id}")
    def review(doc_id: str, body: ReviewRequest) -> dict:
        if body.decision not in ("approve", "reject"):
            raise HTTPException(status_code=400, detail="decision має бути approve або reject")
        try:
            doc = store.record_review(
                doc_id,
                decision=body.decision,
                reviewer=body.reviewer,
                comment=body.comment,
                corrections=body.corrections,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # Схвалений людиною документ іде в леджер тим самим шляхом, що й
        # автоматичний, — облік не має розрізняти, хто підтвердив запис.
        if doc.status == "auto_ok":
            sink.write(doc)
        return doc.model_dump(mode="json")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app
