"""
Генератор синтетичного датасету українських первинних документів.

Навіщо власний генератор, а не готовий набір: щоб мати **еталон**. Точність
екстракції можна виміряти лише тоді, коли для кожного документа відомо, що в
ньому насправді написано, — а для чужих сканів це означало б розмічати їх
руками. Тут ground truth породжується тим самим кодом, що малює документ,
тому розмітка не може розійтися з картинкою.

Ключове рішення — **макет описується один раз**, списком примітивів
(`TextCmd`/`LineCmd`/`RectCmd`), і рендериться двома бекендами:

* reportlab → PDF із текстовим шаром («цифровий» рахунок з бухгалтерії);
* Pillow → растр, який далі навмисно псується: поворот, перспектива,
  нерівне освітлення, шум, JPEG — «фото зі смартфона під кутом».

Якби макет описувався окремо для кожного бекенда, PDF і фото поволі розійшлися
б, і різниця в метриках між ними означала б різницю макетів, а не якості входу.

Реальні чужі документи в репозиторій не кладемо — усі назви, коди й суми
згенеровані. Коди ЄДРПОУ добудовуються з валідною контрольною цифрою
(`validate.make_valid_edrpou`), щоб шар перевірок працював на датасеті так само,
як працював би на справжніх документах.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, Optional

from src.validate import make_valid_edrpou, make_valid_ipn

# A4 у пунктах. Координати всюди від ЛІВОГО ВЕРХНЬОГО кута — так думати про
# макет простіше; перетворення на систему reportlab робить сам бекенд.
PAGE_W = 595.0
PAGE_H = 842.0

Align = Literal["l", "r", "c"]


# --------------------------------------------------------------------------
# Примітиви макета
# --------------------------------------------------------------------------

@dataclass
class TextCmd:
    x: float
    y: float
    text: str
    size: float = 9.0
    bold: bool = False
    align: Align = "l"


@dataclass
class LineCmd:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 0.6


@dataclass
class RectCmd:
    x: float
    y: float
    w: float
    h: float
    gray: float = 0.88


Command = TextCmd | LineCmd | RectCmd


# --------------------------------------------------------------------------
# Шрифти
# --------------------------------------------------------------------------

# Кирилиця обов'язкова, тому вбудовані шрифти reportlab (Helvetica) не годяться.
# Порядок кандидатів: Windows (машина розробки) → типові шляхи Linux (CI та
# Docker-образ) → macOS. Перший знайдений виграє.
FONT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
)


class FontsNotFound(RuntimeError):
    """Жодного кирилічного TTF не знайдено — генерація датасету неможлива."""


def find_fonts() -> tuple[Path, Path]:
    for regular, bold in FONT_CANDIDATES:
        rp, bp = Path(regular), Path(bold)
        if rp.exists() and bp.exists():
            return rp, bp
    raise FontsNotFound(
        "не знайдено кирилічного TTF. Встанови fonts-dejavu (Linux) "
        "або вкажи шлях у FONT_CANDIDATES"
    )


# --------------------------------------------------------------------------
# Форматування чисел і сум прописом
# --------------------------------------------------------------------------

def fmt_amount(value: float) -> str:
    """
    Український формат: пробіл між тисячами, кома як розділювач копійок.

    Саме такий запис — щоденна пастка екстракції: модель, навчена переважно на
    англомовних документах, схильна читати «12 345,67» як 12.345,67 або 1234567.
    """
    whole, frac = divmod(round(value * 100), 100)
    groups = f"{whole:,}".replace(",", "\u00a0")
    return f"{groups},{frac:02d}"


_ONES = (
    "", "один", "два", "три", "чотири", "п'ять", "шість", "сім", "вісім", "дев'ять",
    "десять", "одинадцять", "дванадцять", "тринадцять", "чотирнадцять", "п'ятнадцять",
    "шістнадцять", "сімнадцять", "вісімнадцять", "дев'ятнадцять",
)
_ONES_F = {1: "одна", 2: "дві"}
_TENS = ("", "", "двадцять", "тридцять", "сорок", "п'ятдесят", "шістдесят",
         "сімдесят", "вісімдесят", "дев'яносто")
_HUNDREDS = ("", "сто", "двісті", "триста", "чотириста", "п'ятсот", "шістсот",
             "сімсот", "вісімсот", "дев'ятсот")


def _triple_words(n: int, feminine: bool = False) -> list[str]:
    words: list[str] = []
    if n >= 100:
        words.append(_HUNDREDS[n // 100])
        n %= 100
    if 10 <= n <= 19:
        words.append(_ONES[n])
        return words
    if n >= 20:
        words.append(_TENS[n // 10])
        n %= 10
    if n:
        words.append(_ONES_F[n] if feminine and n in _ONES_F else _ONES[n])
    return words


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    """Українська трійка форм: 1 гривня / 2 гривні / 5 гривень."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def amount_in_words(value: float) -> str:
    """
    Сума прописом — реквізит, який є майже в кожному українському рахунку.

    Для датасету важливий не сам текст, а те, що в документі з'являється друге,
    незалежне джерело правди про суму. Модель, яка «домалювала» цифру, майже
    завжди суперечить прописові — і це видно в помилках екстракції.
    """
    hryvnias, kopecks = divmod(round(value * 100), 100)
    remainder = hryvnias
    if hryvnias == 0:
        words = ["нуль"]
    else:
        words = []
        for power, forms in (
            (1_000_000, ("мільйон", "мільйони", "мільйонів")),
            (1_000, ("тисяча", "тисячі", "тисяч")),
        ):
            part = remainder // power
            remainder %= power
            if part:
                words += _triple_words(part, feminine=(power == 1_000))
                words.append(_plural(part, forms))
        if remainder:
            # «гривня» — жіночий рід, тому останній розряд теж жіночий:
            # «сто вісімдесят одна гривня», а не «один гривня».
            words += _triple_words(remainder, feminine=True)
    # Форму слова визначає ціла частина, а не заокруглена сума: 4,99 — це
    # «чотири гривні 99 коп.», а не «п'ять гривень».
    hryvnia = _plural(hryvnias, ("гривня", "гривні", "гривень"))
    text = " ".join(w for w in words if w)
    return f"{text[:1].upper()}{text[1:]} {hryvnia} {kopecks:02d} коп."


# --------------------------------------------------------------------------
# Дані для генерації
# --------------------------------------------------------------------------

SUPPLIERS = [
    ("ТОВ «Агросвіт-Плюс»", "3012345", "м. Київ, вул. Кільцева, 14, оф. 3"),
    ("ТОВ «Технолайн Україна»", "4123456", "м. Львів, просп. Свободи, 27"),
    ("ПП «Будмайстер-Захід»", "2234567", "м. Івано-Франківськ, вул. Січових Стрільців, 8"),
    ("ТОВ «Медіа Контент Груп»", "3987654", "м. Харків, вул. Сумська, 102, оф. 41"),
    ("ТОВ «Логістик Партнер»", "3455678", "м. Одеса, вул. Грецька, 5"),
    ("ТОВ «Енерго Сервіс Плюс»", "2678901", "м. Дніпро, просп. Яворницького, 60"),
    ("ФОП Кравченко І. М.", "ФОП", "м. Вінниця, вул. Соборна, 33, кв. 7"),
    ("ФОП Левчук О. С.", "ФОП", "м. Полтава, вул. Європейська, 18"),
]

BUYERS = [
    ("ТОВ «Промінвест Груп»", "3111222", "м. Київ, вул. Антоновича, 172"),
    ("ТОВ «Сігма Трейд»", "4222333", "м. Київ, вул. Драгоманова, 4"),
    ("ТОВ «Карпат Фуд»", "2333444", "м. Ужгород, вул. Собранецька, 148"),
    ("ТОВ «Дніпро Ритейл»", "3444555", "м. Дніпро, вул. Робоча, 23"),
]

GOODS = [
    ("Папір офісний A4, 80 г/м2, 500 арк.", "пач.", (95, 190)),
    ("Картридж лазерний HP CF259A (сумісний)", "шт.", (890, 1450)),
    ("Кабель UTP cat.6, бухта 305 м", "бухт", (2100, 3400)),
    ("Монітор 24\", IPS, 75 Гц", "шт.", (4200, 7800)),
    ("Клавіатура механічна, USB", "шт.", (980, 2400)),
    ("Стіл офісний 1400x700, ЛДСП", "шт.", (2800, 5600)),
    ("Крісло офісне, сітка, підлокітники", "шт.", (3100, 6900)),
    ("Вода питна негазована, 19 л", "бут.", (85, 140)),
    ("Серветки паперові, 100 шт.", "уп.", (22, 65)),
    ("Тонер-туба Kyocera TK-1170", "шт.", (1450, 2600)),
]

SERVICES = [
    ("Послуги з розробки та підтримки вебсайту", "год", (600, 1500)),
    ("Консультаційні послуги з автоматизації обліку", "год", (800, 2000)),
    ("Транспортно-експедиційні послуги за маршрутом Київ — Львів", "рейс", (4500, 12000)),
    ("Послуги зі складського зберігання, палето-місце", "міс.", (350, 900)),
    ("Рекламні послуги: розміщення банерів", "міс.", (5000, 18000)),
    ("Технічне обслуговування обладнання", "од.", (1200, 4000)),
    ("Послуги з прибирання приміщень", "міс.", (7000, 15000)),
]

PAYMENT_TERMS = [
    "Оплата протягом 5 банківських днів з дати виставлення рахунку.",
    "Рахунок дійсний до кінця поточного місяця.",
    "Оплата 100% передоплата. Товар відвантажується після зарахування коштів.",
    "Умови оплати: 50% передоплата, 50% після постачання.",
]

BANKS = [
    ("АТ «Універсал Банк»", "UA123222222000026007233566001"),
    ("АТ «Ощадбанк»", "UA903226690000026005300012345"),
    ("АТ КБ «ПриватБанк»", "UA713052990000026007026500001"),
]


# --------------------------------------------------------------------------
# Опис одного згенерованого документа
# --------------------------------------------------------------------------

@dataclass
class GeneratedDoc:
    doc_id: str
    template: str
    kind: Literal["pdf", "photo"]
    fields: dict
    # Навмисно закладена в документ невідповідність (напр. підсумок, що не
    # сходиться з позиціями). eval окремо рахує, скільки таких спіймано.
    planted_issue: Optional[str] = None
    commands: list[Command] = field(default_factory=list)


# --------------------------------------------------------------------------
# Побудова змісту документа
# --------------------------------------------------------------------------

def _make_line_items(rng: random.Random, kind: str, count: int) -> list[dict]:
    pool = SERVICES if kind == "services" else GOODS
    chosen = rng.sample(pool, k=min(count, len(pool)))
    items = []
    for name, unit, (lo, hi) in chosen:
        if unit in ("год", "міс.", "рейс", "од."):
            quantity = float(rng.choice([1, 2, 3, 4, 8, 10, 12, 1.5, 2.5]))
        else:
            quantity = float(rng.randint(1, 25))
        unit_price = round(rng.uniform(lo, hi), 2)
        items.append(
            {
                "description": name,
                "quantity": quantity,
                "unit": unit,
                "unit_price": unit_price,
                "amount": round(quantity * unit_price, 2),
            }
        )
    return items


def build_document(rng: random.Random, index: int, today: date) -> GeneratedDoc:
    """Формує зміст (числа, реквізити) одного документа — ще без макета."""
    supplier = rng.choice(SUPPLIERS)
    buyer = rng.choice(BUYERS)
    is_fop = supplier[1] == "ФОП"

    if is_fop:
        # ФОП на спрощеній системі — без ПДВ, код РНОКПП на 10 цифр.
        supplier_code = make_valid_ipn(f"{rng.randint(20_000_000, 39_999_999):08d}{rng.randint(0, 9)}")
        vat_rate = 0.0
    else:
        supplier_code = make_valid_edrpou(supplier[1])
        vat_rate = rng.choices([20.0, 20.0, 20.0, 7.0, 0.0], k=1)[0]

    buyer_code = make_valid_edrpou(buyer[1])

    doc_kind = rng.choice(["goods", "services"])
    items = _make_line_items(rng, doc_kind, rng.randint(1, 6))

    subtotal = round(sum(i["amount"] for i in items), 2)
    vat_amount = round(subtotal * vat_rate / 100.0, 2)
    total = round(subtotal + vat_amount, 2)

    currency = rng.choices(["UAH", "UAH", "UAH", "UAH", "USD", "EUR"], k=1)[0]
    doc_date = today - timedelta(days=rng.randint(1, 240))

    doc_type = "рахунок"
    if doc_kind == "services" and rng.random() < 0.25:
        doc_type = "акт"

    fields = {
        "doc_type": doc_type,
        "supplier_name": supplier[0],
        "supplier_edrpou": supplier_code,
        "supplier_address": supplier[2],
        "buyer_name": buyer[0],
        "buyer_edrpou": buyer_code,
        "buyer_address": buyer[2],
        "invoice_number": rng.choice([f"{rng.randint(1, 999)}", f"СФ-{rng.randint(100, 9999)}",
                                      f"{rng.randint(1, 99)}/{doc_date.year}"]),
        "invoice_date": doc_date.isoformat(),
        "currency": currency,
        "line_items": items,
        "subtotal": subtotal,
        "vat_rate": vat_rate,
        "vat_amount": vat_amount,
        "total": total,
        "bank": rng.choice(BANKS),
        "terms": rng.choice(PAYMENT_TERMS),
    }
    return GeneratedDoc(
        doc_id=f"doc-{index:03d}",
        template="",
        kind="pdf",
        fields=fields,
    )


def plant_issue(doc: GeneratedDoc, issue: str) -> None:
    """
    Псує документ так, як псують справжні: підсумок не сходиться з позиціями,
    ПДВ порахований від старої бази, дата в майбутньому.

    Псується САМЕ документ (те, що надруковано), а не еталон: еталон і далі
    описує, що в документі написано. Тому в eval такий документ вважається
    прочитаним правильно, якщо модель прочитала надруковані числа — а спіймати
    невідповідність має шар валідації.
    """
    f = doc.fields
    if issue == "subtotal_mismatch":
        f["subtotal"] = round(f["subtotal"] + 80.0, 2)
        f["vat_amount"] = round(f["subtotal"] * f["vat_rate"] / 100.0, 2)
        f["total"] = round(f["subtotal"] + f["vat_amount"], 2)
    elif issue == "total_mismatch":
        f["total"] = round(f["total"] + 1000.0, 2)
    elif issue == "vat_mismatch":
        f["vat_amount"] = round(f["vat_amount"] * 0.9, 2)
        f["total"] = round(f["subtotal"] + f["vat_amount"], 2)
    elif issue == "date_in_future":
        f["invoice_date"] = (date.today() + timedelta(days=14)).isoformat()
    elif issue == "tax_code_checksum":
        code = f["supplier_edrpou"]
        broken = code[:-1] + str((int(code[-1]) + 5) % 10)
        f["supplier_edrpou"] = broken
    else:
        raise ValueError(f"невідома підміна: {issue}")
    doc.planted_issue = issue


# --------------------------------------------------------------------------
# Макети
# --------------------------------------------------------------------------

def _fmt_date_uk(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{d}.{m}.{y}"


def layout_classic(doc: GeneratedDoc) -> list[Command]:
    """
    Т1 — класичний бланк «рахунок-фактура»: шапка з банківськими реквізитами,
    таблиця з рамками, підсумки праворуч, сума прописом.
    """
    f = doc.fields
    c: list[Command] = []
    bank_name, iban = f["bank"]

    c.append(TextCmd(40, 45, "Постачальник:", 8.5, bold=True))
    c.append(TextCmd(120, 45, f["supplier_name"], 9))
    c.append(TextCmd(120, 58, f"код ЄДРПОУ {f['supplier_edrpou']}", 8))
    c.append(TextCmd(120, 70, f["supplier_address"], 8))
    c.append(TextCmd(120, 82, f"п/р {iban}, {bank_name}", 8))

    c.append(TextCmd(40, 104, "Покупець:", 8.5, bold=True))
    c.append(TextCmd(120, 104, f["buyer_name"], 9))
    c.append(TextCmd(120, 116, f"код ЄДРПОУ {f['buyer_edrpou']}", 8))
    c.append(TextCmd(120, 128, f["buyer_address"], 8))

    c.append(LineCmd(40, 142, PAGE_W - 40, 142, 1.0))

    title = "Рахунок-фактура" if f["doc_type"] == "рахунок" else "Акт виконаних робіт"
    c.append(
        TextCmd(
            PAGE_W / 2,
            170,
            f"{title} № {f['invoice_number']} від {_fmt_date_uk(f['invoice_date'])}",
            13,
            bold=True,
            align="c",
        )
    )

    top = 200
    cols = (40, 70, 300, 340, 400, 470, PAGE_W - 40)
    c.append(RectCmd(cols[0], top, cols[-1] - cols[0], 20))
    headers = ("№", "Найменування", "Од.", "К-сть", "Ціна", "Сума")
    for i, head in enumerate(headers):
        c.append(TextCmd(cols[i] + 4, top + 13, head, 8, bold=True))

    y = top + 20
    for n, item in enumerate(f["line_items"], start=1):
        row_h = 18
        c.append(TextCmd(cols[0] + 4, y + 12, str(n), 8))
        c.append(TextCmd(cols[1] + 4, y + 12, _clip(item["description"], 46), 8))
        c.append(TextCmd(cols[2] + 4, y + 12, item["unit"], 8))
        c.append(TextCmd(cols[4] - 6, y + 12, _fmt_qty(item["quantity"]), 8, align="r"))
        c.append(TextCmd(cols[5] - 6, y + 12, fmt_amount(item["unit_price"]), 8, align="r"))
        c.append(TextCmd(cols[6] - 6, y + 12, fmt_amount(item["amount"]), 8, align="r"))
        c.append(LineCmd(cols[0], y + row_h, cols[-1], y + row_h))
        y += row_h

    for x in cols:
        c.append(LineCmd(x, top, x, y))
    c.append(LineCmd(cols[0], top, cols[-1], top))

    y += 16
    cur = f["currency"]
    rows = [("Разом без ПДВ:", f["subtotal"])]
    if f["vat_rate"]:
        rows.append((f"ПДВ {f['vat_rate']:g}%:", f["vat_amount"]))
    else:
        c.append(TextCmd(cols[-1], y, "Без ПДВ", 8.5, align="r"))
        y += 14
    rows.append(("Всього до сплати:", f["total"]))

    for label, value in rows:
        bold = label.startswith("Всього")
        c.append(TextCmd(cols[5] - 6, y + 10, label, 9, bold=bold, align="r"))
        c.append(TextCmd(cols[-1], y + 10, f"{fmt_amount(value)} {cur}", 9, bold=bold, align="r"))
        y += 15

    y += 12
    if cur == "UAH":
        c.append(TextCmd(40, y, f"Сума прописом: {amount_in_words(f['total'])}", 8.5))
        y += 16
    c.append(TextCmd(40, y, f["terms"], 8))

    y += 60
    c.append(LineCmd(40, y, 200, y))
    c.append(TextCmd(40, y + 12, "Керівник", 8))
    c.append(LineCmd(320, y, 480, y))
    c.append(TextCmd(320, y + 12, "Головний бухгалтер", 8))
    return c


def layout_boxed(doc: GeneratedDoc) -> list[Command]:
    """
    Т2 — сучасний макет: заголовок угорі, реквізити сторін у двох колонках,
    таблиця без вертикальних ліній, підсумки в сірій плашці.
    """
    f = doc.fields
    c: list[Command] = []
    bank_name, iban = f["bank"]

    title = "РАХУНОК" if f["doc_type"] == "рахунок" else "АКТ"
    c.append(TextCmd(40, 60, title, 22, bold=True))
    c.append(TextCmd(40, 82, f"№ {f['invoice_number']}", 11))
    c.append(TextCmd(PAGE_W - 40, 60, _fmt_date_uk(f["invoice_date"]), 11, align="r"))
    c.append(TextCmd(PAGE_W - 40, 78, f"Валюта: {f['currency']}", 9, align="r"))
    c.append(LineCmd(40, 96, PAGE_W - 40, 96, 1.4))

    c.append(TextCmd(40, 122, "ВИКОНАВЕЦЬ", 8, bold=True))
    c.append(TextCmd(40, 138, f["supplier_name"], 9.5, bold=True))
    c.append(TextCmd(40, 152, f"ЄДРПОУ: {f['supplier_edrpou']}", 8.5))
    c.append(TextCmd(40, 165, f["supplier_address"], 8))
    c.append(TextCmd(40, 178, f"IBAN {iban}", 7.5))
    c.append(TextCmd(40, 190, bank_name, 7.5))

    c.append(TextCmd(320, 122, "ЗАМОВНИК", 8, bold=True))
    c.append(TextCmd(320, 138, f["buyer_name"], 9.5, bold=True))
    c.append(TextCmd(320, 152, f"ЄДРПОУ: {f['buyer_edrpou']}", 8.5))
    c.append(TextCmd(320, 165, f["buyer_address"], 8))

    top = 225
    c.append(LineCmd(40, top, PAGE_W - 40, top, 1.0))
    for x, head, align in ((44, "Опис", "l"), (360, "К-сть", "r"), (440, "Ціна", "r"),
                           (PAGE_W - 44, "Сума", "r")):
        c.append(TextCmd(x, top + 14, head, 8, bold=True, align=align))  # type: ignore[arg-type]
    c.append(LineCmd(40, top + 20, PAGE_W - 40, top + 20))

    y = top + 20
    for item in f["line_items"]:
        y += 20
        c.append(TextCmd(44, y, _clip(item["description"], 52), 8.5))
        c.append(TextCmd(360, y, f"{_fmt_qty(item['quantity'])} {item['unit']}", 8.5, align="r"))
        c.append(TextCmd(440, y, fmt_amount(item["unit_price"]), 8.5, align="r"))
        c.append(TextCmd(PAGE_W - 44, y, fmt_amount(item["amount"]), 8.5, align="r"))
        c.append(LineCmd(40, y + 7, PAGE_W - 40, y + 7, 0.3))

    y += 24
    box_h = 66 if f["vat_rate"] else 48
    c.append(RectCmd(320, y, PAGE_W - 40 - 320, box_h, gray=0.93))
    yy = y + 16
    c.append(TextCmd(336, yy, "Сума без ПДВ", 8.5))
    c.append(TextCmd(PAGE_W - 52, yy, fmt_amount(f["subtotal"]), 8.5, align="r"))
    if f["vat_rate"]:
        yy += 16
        c.append(TextCmd(336, yy, f"ПДВ {f['vat_rate']:g}%", 8.5))
        c.append(TextCmd(PAGE_W - 52, yy, fmt_amount(f["vat_amount"]), 8.5, align="r"))
    yy += 20
    c.append(TextCmd(336, yy, "ДО СПЛАТИ", 10, bold=True))
    c.append(
        TextCmd(PAGE_W - 52, yy, f"{fmt_amount(f['total'])} {f['currency']}", 10, bold=True, align="r")
    )

    y += box_h + 24
    if f["currency"] == "UAH":
        c.append(TextCmd(40, y, amount_in_words(f["total"]), 8.5))
        y += 18
    c.append(TextCmd(40, y, f["terms"], 8))
    return c


def layout_compact(doc: GeneratedDoc) -> list[Command]:
    """
    Т3 — «дрібний друк»: щільний макет ФОПа, дрібний кегль, реквізити в підвалі.
    Найважчий для розпізнавання шаблон — і саме на ньому видно різницю
    між PDF і фото.
    """
    f = doc.fields
    c: list[Command] = []
    bank_name, iban = f["bank"]

    c.append(
        TextCmd(
            40, 50,
            f"Рахунок на оплату № {f['invoice_number']} від {_fmt_date_uk(f['invoice_date'])}",
            11, bold=True,
        )
    )
    c.append(LineCmd(40, 58, PAGE_W - 40, 58, 0.8))
    c.append(TextCmd(40, 76, f"Постачальник: {f['supplier_name']}, ЄДРПОУ {f['supplier_edrpou']}", 7.5))
    c.append(TextCmd(40, 88, f"Адреса: {f['supplier_address']}", 7.5))
    c.append(TextCmd(40, 100, f"Покупець: {f['buyer_name']}, ЄДРПОУ {f['buyer_edrpou']}", 7.5))

    top = 120
    c.append(TextCmd(40, top, "№", 7.5, bold=True))
    c.append(TextCmd(60, top, "Товар / послуга", 7.5, bold=True))
    c.append(TextCmd(370, top, "Кіль-ть", 7.5, bold=True, align="r"))
    c.append(TextCmd(420, top, "Од.", 7.5, bold=True))
    c.append(TextCmd(500, top, "Ціна", 7.5, bold=True, align="r"))
    c.append(TextCmd(PAGE_W - 40, top, "Сума", 7.5, bold=True, align="r"))
    c.append(LineCmd(40, top + 5, PAGE_W - 40, top + 5, 0.5))

    y = top + 5
    for n, item in enumerate(f["line_items"], start=1):
        y += 14
        c.append(TextCmd(40, y, str(n), 7.5))
        c.append(TextCmd(60, y, _clip(item["description"], 58), 7.5))
        c.append(TextCmd(370, y, _fmt_qty(item["quantity"]), 7.5, align="r"))
        c.append(TextCmd(420, y, item["unit"], 7.5))
        c.append(TextCmd(500, y, fmt_amount(item["unit_price"]), 7.5, align="r"))
        c.append(TextCmd(PAGE_W - 40, y, fmt_amount(item["amount"]), 7.5, align="r"))

    y += 8
    c.append(LineCmd(300, y, PAGE_W - 40, y, 0.5))
    y += 14
    c.append(TextCmd(430, y, "Разом:", 8, align="r"))
    c.append(TextCmd(PAGE_W - 40, y, fmt_amount(f["subtotal"]), 8, align="r"))
    y += 12
    if f["vat_rate"]:
        c.append(TextCmd(430, y, f"ПДВ {f['vat_rate']:g}%:", 8, align="r"))
        c.append(TextCmd(PAGE_W - 40, y, fmt_amount(f["vat_amount"]), 8, align="r"))
    else:
        c.append(TextCmd(430, y, "ПДВ:", 8, align="r"))
        c.append(TextCmd(PAGE_W - 40, y, "без ПДВ", 8, align="r"))
    y += 14
    c.append(TextCmd(430, y, "Усього до сплати:", 8.5, bold=True, align="r"))
    c.append(
        TextCmd(PAGE_W - 40, y, f"{fmt_amount(f['total'])} {f['currency']}", 8.5, bold=True, align="r")
    )

    y += 26
    if f["currency"] == "UAH":
        c.append(TextCmd(40, y, f"Усього на суму: {amount_in_words(f['total'])}", 7.5))
        y += 12
    c.append(TextCmd(40, y, f"{f['terms']}", 7))
    c.append(TextCmd(40, y + 12, f"IBAN {iban} у {bank_name}", 7))
    c.append(TextCmd(40, y + 40, "Підпис ____________________  М.П.", 7.5))
    return c


LAYOUTS = {
    "classic": layout_classic,
    "boxed": layout_boxed,
    "compact": layout_compact,
}


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_qty(q: float) -> str:
    return f"{q:g}".replace(".", ",")


# --------------------------------------------------------------------------
# Бекенд 1: PDF (текстовий шар)
# --------------------------------------------------------------------------

def render_pdf(commands: list[Command], path: Path) -> None:
    from reportlab.lib.colors import Color
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    regular, bold = find_fonts()
    # Імена реєструються один раз на процес; повторна реєстрація тим самим
    # іменем безпечна, тому окремий кеш не потрібен.
    pdfmetrics.registerFont(TTFont("DocSans", str(regular)))
    pdfmetrics.registerFont(TTFont("DocSans-Bold", str(bold)))

    path.parent.mkdir(parents=True, exist_ok=True)
    cv = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
    cv.setTitle(path.stem)

    for cmd in commands:
        if isinstance(cmd, RectCmd):
            cv.setFillColor(Color(cmd.gray, cmd.gray, cmd.gray))
            cv.rect(cmd.x, PAGE_H - cmd.y - cmd.h, cmd.w, cmd.h, stroke=0, fill=1)
            cv.setFillColor(Color(0, 0, 0))
        elif isinstance(cmd, LineCmd):
            cv.setLineWidth(cmd.width)
            cv.line(cmd.x1, PAGE_H - cmd.y1, cmd.x2, PAGE_H - cmd.y2)
        else:
            cv.setFont("DocSans-Bold" if cmd.bold else "DocSans", cmd.size)
            y = PAGE_H - cmd.y
            if cmd.align == "r":
                cv.drawRightString(cmd.x, y, cmd.text)
            elif cmd.align == "c":
                cv.drawCentredString(cmd.x, y, cmd.text)
            else:
                cv.drawString(cmd.x, y, cmd.text)

    cv.showPage()
    cv.save()


# --------------------------------------------------------------------------
# Бекенд 2: растр + навмисне псування («фото зі смартфона»)
# --------------------------------------------------------------------------

def render_image(commands: list[Command], scale: float = 2.0):
    """Той самий макет у растр. Повертає PIL.Image у відтінках сірого."""
    from PIL import Image, ImageDraw, ImageFont

    regular, bold = find_fonts()
    w, h = int(PAGE_W * scale), int(PAGE_H * scale)
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    cache: dict[tuple[bool, int], object] = {}

    def font_for(size: float, is_bold: bool):
        key = (is_bold, int(size * scale * 10))
        if key not in cache:
            cache[key] = ImageFont.truetype(str(bold if is_bold else regular), size * scale)
        return cache[key]

    for cmd in commands:
        if isinstance(cmd, RectCmd):
            gray = int(cmd.gray * 255)
            draw.rectangle(
                [cmd.x * scale, cmd.y * scale, (cmd.x + cmd.w) * scale, (cmd.y + cmd.h) * scale],
                fill=gray,
            )
        elif isinstance(cmd, LineCmd):
            draw.line(
                [cmd.x1 * scale, cmd.y1 * scale, cmd.x2 * scale, cmd.y2 * scale],
                fill=0,
                width=max(1, int(cmd.width * scale)),
            )
        else:
            font = font_for(cmd.size, cmd.bold)
            anchor = {"l": "ls", "r": "rs", "c": "ms"}[cmd.align]
            draw.text((cmd.x * scale, cmd.y * scale), cmd.text, font=font, fill=0, anchor=anchor)

    return img


def degrade(img, rng: random.Random):
    """
    Перетворює чисту сторінку на правдоподібне фото документа.

    Порядок навмисний і повторює фізику: спершу геометрія (аркуш лежить криво і
    знятий не перпендикулярно), потім освітлення (лампа з одного боку), потім
    сенсор (шум), і лише наприкінці — стиснення. Якщо стиснути раніше, артефакти
    JPEG розмажуться наступними кроками і зникнуть.
    """
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter

    # 0. кадрування: людина знімає аркуш, а не порожній стіл під ним. Без
    # цього кроку половина кадру — біле поле, і текст на фото виходить удвічі
    # дрібнішим, ніж на справжньому знімку.
    ink = ImageChops.invert(img).getbbox()
    if ink:
        pad = 24
        img = img.crop(
            (
                max(0, ink[0] - pad),
                max(0, ink[1] - pad),
                min(img.size[0], ink[2] + pad),
                min(img.size[1], ink[3] + rng.randint(pad, pad * 4)),
            )
        )

    # 1. поворот на невеликий кут — аркуш лежить криво
    angle = rng.uniform(-4.0, 4.0)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=235)

    # 2. перспектива — знято не перпендикулярно до площини аркуша
    w2, h2 = img.size
    shift = rng.uniform(0.01, 0.035)
    dx, dy = w2 * shift, h2 * shift * 0.6
    corners = [
        (rng.uniform(0, dx), rng.uniform(0, dy)),
        (w2 - rng.uniform(0, dx), rng.uniform(0, dy) * 0.5),
        (w2 - rng.uniform(0, dx) * 0.5, h2 - rng.uniform(0, dy)),
        (rng.uniform(0, dx) * 0.7, h2 - rng.uniform(0, dy) * 0.7),
    ]
    img = img.transform((w2, h2), Image.PERSPECTIVE, _perspective_coeffs(corners, w2, h2),
                        resample=Image.BICUBIC, fillcolor=235)

    # 3. нерівне освітлення — градієнт яскравості впоперек аркуша
    img = _apply_light_gradient(img, rng)

    # 4. легка розфокусировка і шум сенсора
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.1)))
    img = _add_noise(img, rng, sigma=rng.uniform(3.0, 9.0))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.05))

    # 5. фото зі смартфона доходить до нас у JPEG — і це остання ланка
    return img


def _perspective_coeffs(corners, w: float, h: float) -> tuple[float, ...]:
    """
    Коефіцієнти для PIL.Image.PERSPECTIVE.

    PIL хоче перетворення ВІД цільових координат ДО вихідних, тому система
    розв'язується у зворотному напрямку — інакше картинка «вивертається».
    Розв'язок 8x8 методом Гаусса, без numpy: одна залежність менше в проекті,
    де numpy більше ніде не потрібен.
    """
    target = [(0, 0), (w, 0), (w, h), (0, h)]
    matrix = []
    rhs = []
    for (tx, ty), (sx, sy) in zip(target, corners):
        matrix.append([tx, ty, 1, 0, 0, 0, -sx * tx, -sx * ty])
        rhs.append(sx)
        matrix.append([0, 0, 0, tx, ty, 1, -sy * tx, -sy * ty])
        rhs.append(sy)
    return tuple(_solve(matrix, rhs))


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Гаус із частковим вибором головного елемента, 8 рівнянь."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("вироджена матриця перспективи")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor:
                aug[r] = [v - factor * pv_v for v, pv_v in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def _apply_light_gradient(img, rng: random.Random):
    """Тінь від лампи збоку: маска-градієнт віднімається від яскравості."""
    from PIL import Image, ImageChops, ImageDraw

    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    horizontal = rng.random() < 0.5
    strength = rng.uniform(20, 60)
    steps = 48
    for i in range(steps):
        value = int(strength * (i / steps) ** 1.5)
        if horizontal:
            draw.rectangle([w * i // steps, 0, w * (i + 1) // steps, h], fill=value)
        else:
            draw.rectangle([0, h * i // steps, w, h * (i + 1) // steps], fill=value)
    if rng.random() < 0.5:
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT if horizontal else Image.FLIP_TOP_BOTTOM)
    return ImageChops.subtract(img, mask)


def _add_noise(img, rng: random.Random, sigma: float):
    """
    Шум сенсора. Генерується на вчетверо меншій сітці й розтягується:
    попіксельний `rng.gauss` на 1190x1684 — це 2 млн викликів і секунди часу
    на кожен документ, а візуально різниці немає.

    Шум центрований на 128, тому `offset=-128` робить його двостороннім —
    інакше картинка просто рівномірно світлішала б.
    """
    from PIL import Image, ImageChops

    small_w, small_h = max(1, img.size[0] // 4), max(1, img.size[1] // 4)
    data = bytes(
        max(0, min(255, int(128 + rng.gauss(0, sigma)))) for _ in range(small_w * small_h)
    )
    noise = Image.frombytes("L", (small_w, small_h), data).resize(img.size, Image.BILINEAR)
    return ImageChops.add(img, noise, scale=1, offset=-128)


# --------------------------------------------------------------------------
# Збірка датасету
# --------------------------------------------------------------------------

def generate_dataset(
    out_dir: Path,
    golden_path: Path,
    count: int = 40,
    photo_ratio: float = 0.3,
    seed: int = 20260812,
    today: Optional[date] = None,
) -> list[GeneratedDoc]:
    """
    Генерує датасет і еталон. Детермінований за seed: той самий seed дає
    побайтово ті самі документи, тому числа в README відтворюються.
    """
    rng = random.Random(seed)
    today = today or date.today()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Заплановані підміни: рівно по одній на документ, розкидані по набору.
    planned = ["subtotal_mismatch", "total_mismatch", "vat_mismatch",
               "date_in_future", "tax_code_checksum"]
    planted_at = {int(count * (i + 1) / (len(planned) + 1)): issue
                  for i, issue in enumerate(planned)}

    templates = list(LAYOUTS)
    docs: list[GeneratedDoc] = []

    for i in range(count):
        doc = build_document(rng, i + 1, today)
        doc.template = templates[i % len(templates)]
        if i in planted_at:
            issue = planted_at[i]
            if issue == "vat_mismatch" and not doc.fields["vat_rate"]:
                doc.fields["vat_rate"] = 20.0
                doc.fields["vat_amount"] = round(doc.fields["subtotal"] * 0.2, 2)
                doc.fields["total"] = round(doc.fields["subtotal"] + doc.fields["vat_amount"], 2)
            plant_issue(doc, issue)

        doc.commands = LAYOUTS[doc.template](doc)
        doc.kind = "photo" if rng.random() < photo_ratio else "pdf"

        if doc.kind == "pdf":
            render_pdf(doc.commands, out_dir / f"{doc.doc_id}.pdf")
        else:
            img = degrade(render_image(doc.commands), rng)
            img.convert("L").save(
                out_dir / f"{doc.doc_id}.jpg", quality=rng.randint(45, 70), optimize=True
            )
        docs.append(doc)

    write_golden(docs, out_dir, golden_path)
    return docs


def write_golden(docs: list[GeneratedDoc], out_dir: Path, golden_path: Path) -> None:
    """
    Еталон. Пишеться тим самим кодом, що малює документи, — тому не може
    розійтися з тим, що на них надруковано.
    """
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            ext = "pdf" if doc.kind == "pdf" else "jpg"
            f = doc.fields
            record = {
                "doc_id": doc.doc_id,
                "file": str((out_dir / f"{doc.doc_id}.{ext}").as_posix()),
                "kind": doc.kind,
                "template": doc.template,
                "planted_issue": doc.planted_issue,
                "fields": {
                    "doc_type": f["doc_type"],
                    "supplier_name": f["supplier_name"],
                    "supplier_edrpou": f["supplier_edrpou"],
                    "buyer_name": f["buyer_name"],
                    "buyer_edrpou": f["buyer_edrpou"],
                    "invoice_number": f["invoice_number"],
                    "invoice_date": f["invoice_date"],
                    "currency": f["currency"],
                    "line_items": f["line_items"],
                    "subtotal": f["subtotal"],
                    "vat_rate": f["vat_rate"],
                    "vat_amount": f["vat_amount"],
                    "total": f["total"],
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
