"""
Куди лягає результат.

Зараз реалізація одна — JSONL-леджер на диску. Але інтерфейс `Sink` існує не
«про запас»: у нього винесено те, що відрізняє зовнішній облік від файлу —
**перевірка, чи запис уже там**. Google Sheets і Odoo не мають UNIQUE-індексу
за нашим хешем, тому ідемпотентність для них — це окремий пошук перед записом,
а не властивість сховища. Тримати цю відповідальність у sink-у, а не в
пайплайні, — єдиний спосіб не переписувати пайплайн під кожну інтеграцію.

Другий принцип, перенесений з ai-request-classifier: **помилка зовнішнього
запису не валить прогін**. Локальний артефакт формується завжди, зовнішній
запис — по можливості, і його провал видно у звіті.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from src.schema import ProcessedDocument

logger = logging.getLogger(__name__)


class Sink(Protocol):
    def contains(self, content_hash: str) -> bool:
        ...

    def write(self, doc: ProcessedDocument) -> bool:
        ...


class JsonlSink:
    """
    Леджер: один рядок JSON на документ, дописування в кінець.

    Формат обраний свідомо: його можна читати `Get-Content -Tail`, вантажити в
    pandas і дифати між прогонами. Дублі відсікаються за `content_hash` — при
    старті наявний файл прочитується у множину хешів.
    """

    def __init__(self, path: Path | str, dedupe: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dedupe = dedupe
        self._hashes: set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Обірваний рядок після kill -9 не має ламати старт сервісу:
                    # леджер append-only, тому пошкоджений може бути лише останній.
                    logger.warning("леджер %s: рядок %d не розібрано, пропущено",
                                   self.path, line_no)
                    continue
                if isinstance(row, dict) and row.get("content_hash"):
                    self._hashes.add(row["content_hash"])

    def contains(self, content_hash: str) -> bool:
        return content_hash in self._hashes

    def write(self, doc: ProcessedDocument) -> bool:
        """
        Повертає True, якщо рядок дописано; False — якщо він там уже був.

        Дублікати й провальні екстракції в леджер не пишуться: перші — щоб не
        задвоїти суму в обліку, другі — щоб «оброблені документи» означали
        оброблені. Обидві категорії лишаються в SQLite і видні у звіті.
        """
        if doc.status in ("duplicate", "fallback_error"):
            return False
        if self.dedupe and doc.content_hash in self._hashes:
            return False
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc.to_row(), ensure_ascii=False) + "\n")
        self._hashes.add(doc.content_hash)
        return True

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
