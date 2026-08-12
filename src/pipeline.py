"""
Оркестрація обробки документа.

Шлях одного файлу:

    байти → SHA-256 → чи вже бачили? → екстракція моделлю →
    → перевірки Python-ом → маршрут (auto_ok / needs_review) →
    → SQLite + леджер

Два інваріанти, які тут тримаються явно:

1. **Жоден вхідний файл не губиться.** Якщо модель провалилась після всіх
   retry — з'являється запис зі статусом `fallback_error` і технічною
   причиною, а не тиша в логах. Агрегати рахуються тільки по успішних.
2. **Повторна обробка того самого файлу не створює другий запис в обліку.**
   Перевірка йде ДО виклику моделі, тому дубль ще й безкоштовний.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from src.config import PipelineConfig
from src.llm import DailyQuotaExceeded, VisionClient
from src.schema import ProcessedDocument, ValidationIssue
from src.sinks import Sink
from src.store import DocumentStore, duplicate_record, file_hash, utc_now
from src.validate import route, validate_extraction

logger = logging.getLogger(__name__)

# Що вміє прийняти мультимодальний вхід. Розширення → MIME.
SUPPORTED_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# Стеля на розмір файлу. Не про модель, а про пам'ять: 50 сканів по 40 МБ,
# прочитаних одночасно, кладуть процес швидше за будь-який API-ліміт.
MAX_FILE_BYTES = 20 * 1024 * 1024


class UnsupportedDocument(Exception):
    """Тип файлу, який пайплайн не приймає."""


def detect_mime(path: Path) -> str:
    mime = SUPPORTED_TYPES.get(path.suffix.lower())
    if mime is None:
        raise UnsupportedDocument(
            f"{path.name}: підтримуються {', '.join(sorted(SUPPORTED_TYPES))}"
        )
    return mime


class DocumentPipeline:
    def __init__(
        self,
        client: VisionClient,
        store: DocumentStore,
        sink: Sink,
        cfg: Optional[PipelineConfig] = None,
    ):
        self.client = client
        self.store = store
        self.sink = sink
        self.cfg = cfg or PipelineConfig()

    # -- один документ -----------------------------------------------------

    async def process_bytes(
        self, data: bytes, filename: str, mime_type: str
    ) -> ProcessedDocument:
        content_hash = file_hash(data)
        doc_id = f"{Path(filename).stem}-{content_hash[:8]}"

        existing = self.store.find_by_hash(content_hash)
        if existing is not None:
            # Найдешевша гілка: те саме вже лежить у базі, модель не турбуємо.
            logger.info("%s — дубль %s, екстракція пропущена", filename, existing["doc_id"])
            doc = duplicate_record(doc_id, content_hash, filename, mime_type, existing["doc_id"])
            self.store.save(doc)
            return doc

        doc = ProcessedDocument(
            doc_id=doc_id,
            source_path=filename,
            content_hash=content_hash,
            mime_type=mime_type,
            received_at=utc_now(),
        )

        started = time.monotonic()
        try:
            result = await self.client.extract(data, mime_type)
            doc.extraction = result.extraction
            doc.retries_used = result.retries_used
            doc.input_tokens = result.input_tokens
            doc.output_tokens = result.output_tokens
        except DailyQuotaExceeded:
            # Квота — не властивість документа. Прокидаємо далі, щоб прогін
            # зупинився цілком, а не малював fallback на кожен файл поспіль.
            raise
        except (MemoryError, RecursionError):
            # Ці дві — не «помилка документа», а стан процесу. Якщо загорнути їх
            # у fallback-запис, цикл поїде далі й доведе машину до OOM замість
            # того, щоб упасти на першому ж документі. Прокидаємо нагору.
            raise
        except Exception as exc:  # noqa: BLE001 — сюди ж усе несподіване від SDK
            # Запис усе одно створюється — з причиною і статусом, який видно
            # у звіті. Документ не зникає навіть тоді, коли впав увесь виклик.
            doc.latency_s = time.monotonic() - started
            doc.status = "fallback_error"
            doc.processing_error = f"{type(exc).__name__}: {exc}"[:500]
            doc.review_reasons = ["екстракція не вдалася"]
            logger.error("%s — екстракція провалилась: %s", filename, str(exc)[:300])
            self.store.save(doc)
            return doc

        doc.latency_s = time.monotonic() - started

        issues: list[ValidationIssue] = []
        if self.cfg.validation_enabled:
            issues = validate_extraction(doc.extraction, self.cfg)

        soft = self.store.find_soft_duplicate(
            doc.extraction.supplier_edrpou, doc.extraction.invoice_number, doc_id
        )
        if soft is not None:
            # Байти інші, реквізити ті самі — перескан або повторна відправка.
            # Не відкидаємо: це може бути й виправлений документ, вирішує людина.
            issues.append(
                ValidationIssue(
                    code="duplicate_invoice_number",
                    severity="error",
                    field="invoice_number",
                    message=f"Документ із цим номером від цього постачальника вже є: "
                    f"{soft['doc_id']}",
                    actual=doc.extraction.invoice_number,
                )
            )

        doc.issues = issues
        status, reasons = route(doc.extraction, issues, self.cfg)
        doc.status = status  # type: ignore[assignment]
        doc.review_reasons = reasons

        self.store.save(doc)
        self.sink.write(doc)
        return doc

    async def process_path(self, path: Path) -> ProcessedDocument:
        mime = detect_mime(path)
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise UnsupportedDocument(
                f"{path.name}: {size / 1024 / 1024:.1f} МБ перевищує ліміт "
                f"{MAX_FILE_BYTES // 1024 // 1024} МБ"
            )
        return await self.process_bytes(path.read_bytes(), str(path), mime)

    # -- пачка -------------------------------------------------------------

    async def process_many(
        self,
        paths: list[Path],
        concurrency: int = 2,
        limit: Optional[int] = None,
    ) -> list[ProcessedDocument]:
        """
        Обробка списку файлів із обмеженою конкурентністю.

        Семафор, а не `gather` по всьому списку: 40 одночасних запитів у
        безкоштовний тір — це 40 відповідей 429 і жодного результату.
        """
        candidates = [p for p in paths if p.suffix.lower() in SUPPORTED_TYPES]
        skipped = len(paths) - len(candidates)
        if skipped:
            logger.info("пропущено %d файлів непідтримуваного типу", skipped)
        if limit is not None:
            candidates = candidates[:limit]

        semaphore = asyncio.Semaphore(max(1, concurrency))
        results: list[ProcessedDocument] = []
        quota_hit = False

        async def worker(path: Path) -> Optional[ProcessedDocument]:
            nonlocal quota_hit
            if quota_hit:
                return None
            async with semaphore:
                try:
                    return await self.process_path(path)
                except DailyQuotaExceeded as exc:
                    quota_hit = True
                    logger.error("%s — прогін зупинено", exc)
                    return None
                except UnsupportedDocument as exc:
                    logger.warning("%s", exc)
                    return None

        done = await asyncio.gather(*(worker(p) for p in candidates))
        results = [d for d in done if d is not None]

        if quota_hit:
            logger.error(
                "оброблено %d з %d файлів: денна квота вичерпана. "
                "Решта не позначена як помилкова — їх просто ще не обробляли.",
                len(results),
                len(candidates),
            )
        return results

    # -- підсумок ----------------------------------------------------------

    def summary(self, docs: list[ProcessedDocument]) -> dict:
        """
        Зведення прогону. Вартість рахується з реальних лічильників токенів,
        а не з припущень — інакше рядок «скільки коштує 1000 документів» у
        README був би вигадкою.
        """
        by_status: dict[str, int] = {}
        for doc in docs:
            by_status[doc.status] = by_status.get(doc.status, 0) + 1

        processed = [d for d in docs if d.status in ("auto_ok", "needs_review")]
        latencies = sorted(d.latency_s for d in processed)
        tokens_in = sum(d.input_tokens for d in docs)
        tokens_out = sum(d.output_tokens for d in docs)

        issue_counts: dict[str, int] = {}
        for doc in docs:
            for issue in doc.issues:
                issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1

        cost = (
            tokens_in / 1_000_000 * self.cfg.input_price_per_mtok
            + tokens_out / 1_000_000 * self.cfg.output_price_per_mtok
        )
        n = max(1, len(processed))

        return {
            "documents": len(docs),
            "by_status": by_status,
            "auto_rate": round(by_status.get("auto_ok", 0) / max(1, len(processed)), 4),
            "issues": dict(sorted(issue_counts.items(), key=lambda kv: -kv[1])),
            "latency_s": {
                "mean": round(sum(latencies) / n, 2) if latencies else 0.0,
                "p50": round(latencies[len(latencies) // 2], 2) if latencies else 0.0,
                "p95": round(latencies[int(len(latencies) * 0.95)], 2)
                if len(latencies) > 1
                else (round(latencies[0], 2) if latencies else 0.0),
            },
            "tokens": {"input": tokens_in, "output": tokens_out},
            "cost_usd": {
                "run": round(cost, 4),
                "per_1000_docs": round(cost / n * 1000, 2),
            },
            "config": self.cfg.to_dict(),
        }
