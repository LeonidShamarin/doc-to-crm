"""
Мультимодальний клієнт Gemini: документ на вхід, `InvoiceExtraction` на вихід.

**Окремого OCR немає навмисно.** PDF або фото йдуть у модель як є, inline. Це
не економія рядків коду: класична зв'язка «OCR → регулярки по тексту»
розсипається рівно там, де починається реальний документ — таблиця, знята під
кутом, втрачає прив'язку числа до колонки, і сума з рядка 3 приїжджає в
рядок 2. Модель, яка бачить сторінку, читає таблицю як таблицю.

Рівні стійкості розділені, бо це різні класи проблем:

1. **Rate limiter** — рознесення викликів у часі під квоту безкоштовного тіру.
2. **Backoff** — транспорт: хвилинний 429, 5xx, обрив мережі. Модель ні до чого.
3. **Circuit breaker** — денна квота. Її не перечекати, тому решта запитів
   падає миттєво, без марних викликів.
4. **Self-repair** — зміст: відповідь не спарсилась або не пройшла Pydantic.
   Показуємо моделі її ж помилку і просимо виправити.

Вичерпали все — кидаємо `ExtractionError`, і пайплайн перетворює її на запис
зі статусом `fallback_error`. Документ не зникає.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from src.schema import InvoiceExtraction

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_RPM = 5

TRANSPORT_MAX_ATTEMPTS = 4
TRANSPORT_BASE_DELAY = 2.0
# Стеля на паузу, яку просить сервер: без неї один невдалий запит може
# підвісити весь прогін на кілька хвилин.
MAX_SERVER_RETRY_DELAY = 65.0

_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")

SYSTEM_PROMPT = """\
Ти — асистент бухгалтерії. На вході — скан, фото або PDF українського \
первинного документа: рахунок-фактура, рахунок на оплату, акт виконаних робіт \
або видаткова накладна.

Витягни реквізити рівно за наданою схемою.

Правила:

- Переписуй те, що НАДРУКОВАНО в документі. Нічого не перераховуй і не \
«виправляй»: якщо підсумок у документі не сходиться з позиціями, віддай саме \
той підсумок, що надрукований. Розбіжності шукає інша частина системи.
- Числа віддавай як числа, без пробілів і символів валюти. Український запис \
«12 345,67» — це 12345.67. Крапка в «12.345,67» — розділювач тисяч, а не дробу.
- Дату віддавай у форматі YYYY-MM-DD. «05.03.2026» — це 2026-03-05 \
(день першим), а не 5 травня.
- ЄДРПОУ — тільки цифри: 8 для юрособи, 10 для ФОП (РНОКПП).
- Поля, якого в документі немає, не вигадуй — став null.
- Якщо в документі написано «без ПДВ», став vat_rate = 0 і vat_amount = 0.
- Позиції таблиці віддавай зверху вниз, усі до одної.

Впевненість (confidence) заповнюй ЧЕСНО і ПО КОЖНОМУ ПОЛЮ окремо:

- 0.9–1.0 — значення видно чітко й однозначно;
- 0.5–0.8 — місце розбірливе частково: дрібний шрифт, засвітка, нахил, \
довелось домислювати цифру або межу колонки;
- 0.0–0.4 — поля не видно, воно обрізане, або ти не впевнений, що це саме те поле.

Занижена впевненість коштує дешево (документ подивиться людина), завищена \
коштує дорого (помилка потрапить в облік мовчки).
"""


class ExtractionError(Exception):
    """Вичерпані всі спроби — і транспортні, і змістовні."""


class DailyQuotaExceeded(Exception):
    """Вичерпано денну квоту — до кінця доби прогін не відновиться."""


@dataclass
class ExtractionResult:
    extraction: InvoiceExtraction
    retries_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class VisionClient(Protocol):
    """
    Контракт, від якого залежить пайплайн.

    Саме завдяки йому весь набір тестів проходить без ключа: у тестах на це
    місце стає фейковий клієнт із наперед заданими відповідями.
    """

    async def extract(self, data: bytes, mime_type: str, hint: str = "") -> ExtractionResult:
        ...


def _server_retry_delay(exc: Exception) -> Optional[float]:
    """
    Gemini у тілі 429 повертає RetryInfo.retryDelay (напр. 46s) — це набагато
    точніше за наш сліпий exponential backoff. Дістаємо, якщо є.
    """
    match = _RETRY_DELAY_RE.search(str(exc))
    if not match:
        return None
    return min(float(match.group(1)), MAX_SERVER_RETRY_DELAY)


def _is_daily_quota(exc: Exception) -> bool:
    """
    Не всі 429 однакові. Хвилинну квоту перечекати можна, денну — ні.

    Пастка: Gemini у ВСІХ 429 повертає RetryInfo (напр. 'retryDelay: 25s'),
    навіть коли квота денна. Якщо слухати цю пораду наосліп, прогін годинами
    крутить безнадійні retry. Тому дивимось саме на quotaId.
    """
    return "PerDay" in str(exc)


def _is_retryable(exc: Exception) -> bool:
    """Транспортна помилка, яку має сенс повторити після паузи."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        # Єдина 4xx, яку варто повторювати — хвилинний rate limit.
        # 400/401/403 не «розсмокчуться» самі, денна квота — тим паче.
        if getattr(exc, "code", None) != 429:
            return False
        return not _is_daily_quota(exc)
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))


class _RateLimiter:
    """
    Проста черга на N викликів за хвилину. Не обмежує паралелізм сам по собі —
    лише рознесення стартів у часі, решту робить семафор у пайплайні.
    """

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = now + self._interval


class GeminiVisionClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        rpm: int = DEFAULT_RPM,
        max_repair_retries: int = 1,
    ):
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._limiter = _RateLimiter(rpm)
        self._max_repair_retries = max_repair_retries
        # Запобіжник: денна квота вичерпується на весь проєкт одразу. Немає сенсу
        # ганяти решту запитів по мережі — вони гарантовано впадуть так само.
        self.daily_quota_hit = False
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            response_mime_type="application/json",
            # Pydantic-модель передається напряму: SDK сам мапить Optional[...]
            # у nullable-поле схеми Gemini. Ручна конвертація JSON Schema тут не
            # потрібна і шкідлива (anyOf/default API не приймає).
            response_schema=InvoiceExtraction,
            # Читання бланка за готовою схемою не потребує глибокого reasoning.
            # Саме thinking_level, а не thinking_budget: моделі Gemini 3.x
            # відповідають на budget=0 помилкою 400 INVALID_ARGUMENT.
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )

    async def extract(self, data: bytes, mime_type: str, hint: str = "") -> ExtractionResult:
        """Один документ → структура. Self-repair на змістовних помилках."""
        prompt = hint or "Витягни реквізити цього документа за схемою."
        last_error: Optional[Exception] = None
        tokens_in = tokens_out = 0

        for attempt in range(self._max_repair_retries + 1):
            try:
                raw, used_in, used_out = await self._call_model(data, mime_type, prompt)
                tokens_in += used_in
                tokens_out += used_out
                parsed = InvoiceExtraction.model_validate(_parse_json(raw))
                return ExtractionResult(
                    extraction=parsed,
                    retries_used=attempt,
                    input_tokens=tokens_in,
                    output_tokens=tokens_out,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning("Extraction attempt %d failed: %s", attempt, str(exc)[:300])
                # self-repair: показуємо моделі її помилку і просимо виправити
                prompt = (
                    f"{hint or 'Витягни реквізити цього документа за схемою.'}\n\n"
                    f"Попередня відповідь була невалідною: {exc}\n"
                    "Виправ і поверни ТІЛЬКИ коректний JSON за схемою."
                )

        raise ExtractionError(
            f"екстракція не вдалася після {self._max_repair_retries + 1} спроб: {last_error}"
        )

    async def _call_model(
        self, data: bytes, mime_type: str, prompt: str
    ) -> tuple[str, int, int]:
        """Один логічний виклик моделі, з exponential backoff на 429/5xx/мережу."""
        last_error: Optional[Exception] = None
        part = types.Part.from_bytes(data=data, mime_type=mime_type)

        for attempt in range(TRANSPORT_MAX_ATTEMPTS):
            if self.daily_quota_hit:
                raise DailyQuotaExceeded(f"денна квота вичерпана для моделі {self._model_name}")
            try:
                await self._limiter.acquire()
                response = await self._client.aio.models.generate_content(
                    model=self._model_name,
                    contents=[part, prompt],
                    config=self._config,
                )
                text = response.text
                usage = getattr(response, "usage_metadata", None)
                tokens_in = getattr(usage, "prompt_token_count", 0) or 0
                tokens_out = getattr(usage, "candidates_token_count", 0) or 0
                if not text:
                    # Порожня відповідь — напр. спрацював safety-фільтр. Це вже не
                    # транспорт, а зміст: віддаємо нагору, хай самополагодиться.
                    raise ValueError("модель повернула порожню відповідь")
                return text, tokens_in, tokens_out
            except Exception as exc:  # noqa: BLE001 — розбираємо тип нижче
                if _is_daily_quota(exc):
                    self.daily_quota_hit = True
                    raise DailyQuotaExceeded(
                        f"денна квота вичерпана для моделі {self._model_name}. "
                        "Спробуй іншу модель (--model) або зачекай до скидання квоти."
                    ) from exc
                if not _is_retryable(exc):
                    raise
                last_error = exc
                if attempt == TRANSPORT_MAX_ATTEMPTS - 1:
                    break
                # Якщо сервер сам сказав, скільки чекати — слухаємо його.
                # Наш exponential backoff тут лише запасний варіант.
                delay = _server_retry_delay(exc)
                if delay is None:
                    delay = TRANSPORT_BASE_DELAY * (2**attempt)
                delay += random.uniform(0, 1)
                logger.warning(
                    "Transport error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    TRANSPORT_MAX_ATTEMPTS,
                    delay,
                    str(exc)[:200],
                )
                await asyncio.sleep(delay)

        raise ExtractionError(f"транспорт не витримав {TRANSPORT_MAX_ATTEMPTS} спроб: {last_error}")


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    # захист від випадків, коли модель усе ж обгортає у ```json ... ```
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"очікувався JSON-об'єкт, отримано {type(data).__name__}")
    return data
