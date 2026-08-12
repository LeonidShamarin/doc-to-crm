"""
Watcher теки: нові файли — в пайплайн.

Опитування, а не inotify/ReadDirectoryChangesW: тека може лежати на мережевій
шарі або в змонтованому томі контейнера, де події файлової системи або не
приходять, або приходять не всі. Кілька секунд затримки для рахунку — ніщо,
мовчки загублений документ — ні.

Три запобіжники, без яких цикл рано чи пізно з'їдає машину:

* `max_cycles` — цикл має явний вихід, а не тільки Ctrl+C. Тести ганяють рівно
  задану кількість обертів і завершуються.
* **файл обробляється лише коли дописаний.** Розмір і mtime мають збігтися між
  двома опитуваннями: інакше watcher хапає наполовину скопійований PDF і
  платить за екстракцію сміття.
* `_seen` тримає тільки шляхи поточного циклу, а історію — SQLite. Множина, що
  росте вічно в довгограючому процесі, — це той самий OOM, тільки повільний.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.pipeline import SUPPORTED_TYPES, DocumentPipeline, UnsupportedDocument
from src.store import file_hash

logger = logging.getLogger(__name__)

# Скільки файлів беремо за один оберт. Стеля потрібна, щоб вивалений у теку
# архів на 5000 сканів не перетворився на один нескінченний цикл без логів.
MAX_FILES_PER_CYCLE = 50


def scan_folder(folder: Path, max_files: int = MAX_FILES_PER_CYCLE) -> list[Path]:
    """
    Підтримувані файли теки, за глибиною 1.

    Без рекурсії навмисно: `rglob("*")` по теці, куди хтось поклав символьне
    посилання на батьківський каталог, не завершується ніколи.
    """
    if not folder.exists():
        return []
    files = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_TYPES
    ]
    return files[:max_files]


class StabilityTracker:
    """
    Файл вважається готовим, коли (розмір, mtime) не змінилися між опитуваннями.

    Копіювання 8-мегабайтного скана по мережі триває секунди; без цієї
    перевірки watcher устигає прочитати половину і відправити її в модель.
    """

    def __init__(self) -> None:
        self._seen: dict[Path, tuple[int, float]] = {}

    def is_stable(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except OSError:
            self._seen.pop(path, None)
            return False
        signature = (stat.st_size, stat.st_mtime)
        previous = self._seen.get(path)
        self._seen[path] = signature
        return previous == signature

    def forget(self, path: Path) -> None:
        self._seen.pop(path, None)

    def prune(self, present: set[Path]) -> None:
        """Забуваємо зниклі файли — інакше словник росте весь час роботи сервісу."""
        for path in list(self._seen):
            if path not in present:
                del self._seen[path]


async def watch_folder(
    folder: Path,
    pipeline: DocumentPipeline,
    interval: float = 3.0,
    max_cycles: Optional[int] = None,
    processed_dir: Optional[Path] = None,
) -> int:
    """
    Основний цикл. Повертає кількість оброблених документів.

    Оброблені файли не видаляються: вхідна тека — це чужі дані, а не наш стан.
    Повторна поява того самого файлу нічого не зламає — його відсіче хеш.
    """
    tracker = StabilityTracker()
    processed = 0
    cycle = 0

    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        files = scan_folder(folder)
        tracker.prune(set(files))

        for path in files:
            if not tracker.is_stable(path):
                logger.debug("%s ще дописується, чекаємо наступного циклу", path.name)
                continue
            try:
                data = path.read_bytes()
            except OSError as exc:
                logger.warning("%s: не вдалося прочитати (%s)", path.name, exc)
                continue

            # Дешева перевірка ДО пайплайна: знайомий хеш — не витрачаємо навіть
            # виклик у сховище. Це головний захист від того, що watcher
            # перечитує ту саму теку кожні три секунди.
            if pipeline.store.find_by_hash(file_hash(data)) is not None:
                continue

            try:
                doc = await pipeline.process_bytes(data, str(path), _mime_for(path))
            except UnsupportedDocument as exc:
                logger.warning("%s", exc)
                continue
            processed += 1
            logger.info("%s → %s (%s)", path.name, doc.status, doc.doc_id)

            if processed_dir is not None and doc.status != "fallback_error":
                _move_aside(path, processed_dir, tracker)

        if max_cycles is not None and cycle >= max_cycles:
            break
        await asyncio.sleep(interval)

    return processed


def _mime_for(path: Path) -> str:
    return SUPPORTED_TYPES[path.suffix.lower()]


def _move_aside(path: Path, target_dir: Path, tracker: StabilityTracker) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / path.name
    # Колізію імен розв'язуємо суфіксом, а не перезаписом: два різні документи
    # з іменем "scan.pdf" — звичайна ситуація для теки, куди кладуть із телефона.
    counter = 1
    while destination.exists():
        destination = target_dir / f"{path.stem}-{counter}{path.suffix}"
        counter += 1
    try:
        path.rename(destination)
        tracker.forget(path)
    except OSError as exc:
        logger.warning("%s: не вдалося перемістити (%s)", path.name, exc)
