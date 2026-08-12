"""
Конфігурація пайплайна.

Окремий об'єкт, а не розсипані константи: `eval` перебирає саме ці параметри
(поріг впевненості, вмикання шару валідації), і кожен рядок таблиці в README —
це один такий конфіг.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

# Екстракція за фіксованою схемою — не задача на міркування, flash-lite
# вистачає, а денна квота безкоштовного тіру в неї помітно щедріша за flash.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Ліміт безкоштовного тіру — 5 запитів/хв на модель.
DEFAULT_RPM = 5

# Ціна flash-lite за мільйон токенів, USD. Використовується тільки для оцінки
# вартості 1000 документів у звіті; змінюється прапорцем, щоб число в README
# не протухало мовчки.
DEFAULT_INPUT_PRICE = 0.10
DEFAULT_OUTPUT_PRICE = 0.40

# Скільки секунд забирає ручне перенесення одного рахунку в таблицю. Береться
# з опису задачі (3–5 хв) по нижній межі — щоб ROI не був завищений.
MANUAL_SECONDS_PER_DOC = 180.0


@dataclass
class PipelineConfig:
    # --- модель ---
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    rpm: int = DEFAULT_RPM
    max_repair_retries: int = 1

    # --- маршрутизація ---
    # Поле з впевненістю нижче порогу відправляє документ на ручну перевірку.
    min_field_confidence: float = 0.75
    # Вимикач шару валідації — потрібен рівно для одного рядка таблиці в eval:
    # «скільки помилок ловить впевненість моделі сама по собі».
    validation_enabled: bool = True
    # Копійчана толерантність для звірки сум: 0.01 замало, бо ПДВ округлюють
    # покомпонентно і розбіжність у 1–2 копійки — норма поліграфії, не помилка.
    amount_tolerance: float = 0.02

    # --- вартість ---
    input_price_per_mtok: float = DEFAULT_INPUT_PRICE
    output_price_per_mtok: float = DEFAULT_OUTPUT_PRICE

    def to_dict(self) -> dict:
        return asdict(self)

    def short_name(self) -> str:
        """Компактна назва конфігурації для рядка таблиці в eval-звіті."""
        parts = [self.model.split("-")[-1], f"conf{self.min_field_confidence:g}"]
        parts.append("valid" if self.validation_enabled else "novalid")
        return "+".join(parts)


def config_from_env() -> PipelineConfig:
    """Конфіг для сервісу: у Docker/Space прапорців CLI немає, є оточення."""
    return PipelineConfig(
        model=os.environ.get("DOC_MODEL", DEFAULT_MODEL),
        min_field_confidence=float(os.environ.get("MIN_FIELD_CONFIDENCE", "0.75")),
        rpm=int(os.environ.get("DOC_RPM", str(DEFAULT_RPM))),
    )
