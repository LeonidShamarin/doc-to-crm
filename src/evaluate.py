"""
Вимірювання якості: точність по полях і якість маршрутизації.

Дві різні речі, які легко переплутати і які тут рахуються окремо:

* **Точність екстракції** — чи збіглося прочитане з тим, що надруковано в
  документі. Еталон породжений генератором, тому розмітка не може бути
  неточною (`render.py`).
* **Якість маршрутизації** — чи потрапив на стіл до людини саме той документ,
  який мав. Це головна метрика продукту: точність 95% нічого не варта, якщо
  решта 5% помилок їде в облік мовчки.

Обидві рахуються з ОДНОГО прогону: результати екстракції зберігаються в
`output/eval-run.json`, і всі варіанти маршрутизації (валідація + впевненість,
тільки впевненість, тільки валідація) перераховуються з них без повторних
викликів моделі. Тому таблиця порівняння коштує рівно один прогін, а не три.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.config import MANUAL_SECONDS_PER_DOC, PipelineConfig
from src.schema import InvoiceExtraction, ProcessedDocument
from src.validate import parse_doc_date, route, validate_extraction

logger = logging.getLogger(__name__)

# Скільки часу людина витрачає на один документ у черзі: подивитись
# підсвічені поля й натиснути кнопку. Оцінка навмисно консервативна —
# ROI від неї тільки програє.
REVIEW_SECONDS_PER_DOC = 45.0

SCALAR_FIELDS = (
    "supplier_name",
    "supplier_edrpou",
    "buyer_name",
    "buyer_edrpou",
    "invoice_number",
    "invoice_date",
    "currency",
    "subtotal",
    "vat_rate",
    "vat_amount",
    "total",
)

NUMERIC_FIELDS = {"subtotal", "vat_rate", "vat_amount", "total"}


# --------------------------------------------------------------------------
# Порівняння значень
# --------------------------------------------------------------------------

def norm_text(value: Any) -> str:
    """
    Нормалізація тексту перед порівнянням.

    Лапки прибираються навмисно: «Агросвіт-Плюс», "Агросвіт-Плюс" і
    Агросвіт-Плюс — це та сама назва, і зараховувати різницю в лапках як
    помилку екстракції означало б занижувати метрику на рівному місці.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    for ch in "«»\"'“”„`":
        text = text.replace(ch, "")
    return " ".join(text.split())


def values_match(field: str, expected: Any, actual: Any) -> bool:
    if field in NUMERIC_FIELDS:
        if expected is None or actual is None:
            return expected is None and actual is None
        return abs(float(expected) - float(actual)) <= 0.01
    if field == "invoice_date":
        if not expected or not actual:
            return not expected and not actual
        left, right = parse_doc_date(str(expected)), parse_doc_date(str(actual))
        return left is not None and left == right
    if field.endswith("edrpou"):
        return "".join(filter(str.isdigit, str(expected or ""))) == "".join(
            filter(str.isdigit, str(actual or ""))
        )
    if field == "invoice_number":
        return norm_text(str(expected or "").lstrip("№ ")) == norm_text(
            str(actual or "").lstrip("№ ")
        )
    return norm_text(expected) == norm_text(actual)


def compare_line_items(expected: list[dict], actual: list) -> dict:
    """
    Позиції звіряються за кількістю рядків і за сумами.

    Опис навмисно НЕ звіряється по символах: «Папір офісний A4, 80 г/м2» проти
    «Папір офісний А4 80 г/м2» — та сама позиція, а різниця в латинській A і
    комі. Для обліку критичні числа, і саме вони перевіряються точно.
    """
    actual_amounts = sorted(
        round(float(i.amount), 2) for i in actual if getattr(i, "amount", None) is not None
    )
    expected_amounts = sorted(round(float(i["amount"]), 2) for i in expected)
    matched = 0
    remaining = list(actual_amounts)
    for amount in expected_amounts:
        hit = next((a for a in remaining if abs(a - amount) <= 0.01), None)
        if hit is not None:
            remaining.remove(hit)
            matched += 1
    return {
        "count_match": len(expected) == len(actual),
        "expected_count": len(expected),
        "actual_count": len(actual),
        "amounts_matched": matched,
        "amounts_total": len(expected_amounts),
        "all_match": len(expected) == len(actual) and matched == len(expected_amounts),
    }


def compare_document(golden_fields: dict, extraction: Optional[InvoiceExtraction]) -> dict:
    """Порівняння одного документа з еталоном: по полях і в цілому."""
    if extraction is None:
        return {
            "fields": {f: False for f in SCALAR_FIELDS},
            "line_items": {"all_match": False, "amounts_matched": 0,
                           "amounts_total": len(golden_fields.get("line_items", []))},
            "fully_correct": False,
            "wrong_fields": list(SCALAR_FIELDS) + ["line_items"],
        }

    fields = {
        f: values_match(f, golden_fields.get(f), getattr(extraction, f))
        for f in SCALAR_FIELDS
    }
    items = compare_line_items(golden_fields.get("line_items", []), extraction.line_items)
    wrong = [f for f, ok in fields.items() if not ok]
    if not items["all_match"]:
        wrong.append("line_items")
    return {
        "fields": fields,
        "line_items": items,
        "fully_correct": not wrong,
        "wrong_fields": wrong,
    }


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------

def routing_metrics(records: list[dict], variant_key: str) -> dict:
    """
    Скільки помилкових документів система відправила людині і якою ціною.

    `recall` тут важливіший за `precision`: пропущена помилка потрапляє в
    облік мовчки, зайвий документ у черзі коштує 45 секунд уваги.
    """
    tp = fp = fn = tn = 0
    for rec in records:
        should = rec["should_review"]
        routed = rec[variant_key] != "auto_ok"
        if should and routed:
            tp += 1
        elif should and not routed:
            fn += 1
        elif not should and routed:
            fp += 1
        else:
            tn += 1
    total = max(1, len(records))
    return {
        "caught": tp,
        "missed": fn,
        "false_alarms": fp,
        "clean_auto": tn,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 1.0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 1.0,
        "auto_rate": round((tn + fn) / total, 4),
        "review_rate": round((tp + fp) / total, 4),
    }


def build_records(
    docs: list[ProcessedDocument], golden: dict[str, dict], cfg: PipelineConfig
) -> list[dict]:
    """
    Зшиває результат прогону з еталоном і одразу рахує три варіанти
    маршрутизації з тих самих екстракцій.
    """
    variants = {
        "route_full": PipelineConfig(
            validation_enabled=True, min_field_confidence=cfg.min_field_confidence
        ),
        "route_conf_only": PipelineConfig(
            validation_enabled=False, min_field_confidence=cfg.min_field_confidence
        ),
        "route_valid_only": PipelineConfig(validation_enabled=True, min_field_confidence=0.0),
    }

    records = []
    for doc in docs:
        entry = golden.get(_golden_key(doc))
        if entry is None:
            logger.warning("для %s немає еталона — документ пропущено", doc.doc_id)
            continue

        comparison = compare_document(entry["fields"], doc.extraction)
        rec: dict[str, Any] = {
            "doc_id": doc.doc_id,
            "kind": entry["kind"],
            "template": entry["template"],
            "planted_issue": entry.get("planted_issue"),
            "status_as_run": doc.status,
            "comparison": comparison,
            "latency_s": doc.latency_s,
            "input_tokens": doc.input_tokens,
            "output_tokens": doc.output_tokens,
            "issue_codes": [i.code for i in doc.issues],
        }

        # Документ треба показати людині, якщо модель щось прочитала невірно
        # АБО якщо сам документ внутрішньо суперечливий (планова підміна).
        rec["should_review"] = (not comparison["fully_correct"]) or bool(entry.get("planted_issue"))

        for key, variant_cfg in variants.items():
            if doc.extraction is None:
                rec[key] = "fallback_error"
                continue
            issues = (
                validate_extraction(doc.extraction, variant_cfg)
                if variant_cfg.validation_enabled
                else []
            )
            status, _ = route(doc.extraction, issues, variant_cfg)
            rec[key] = status
            if key == "route_full":
                rec["issue_codes_recomputed"] = [i.code for i in issues]

        records.append(rec)
    return records


def _golden_key(doc: ProcessedDocument) -> str:
    """doc_id пайплайна — це `<ім'я файлу>-<8 символів хеша>`; еталон знає ім'я."""
    return Path(doc.source_path).stem


def aggregate(records: list[dict], cfg: PipelineConfig) -> dict:
    total = len(records)
    if total == 0:
        return {"documents": 0}

    per_field = {}
    for field in SCALAR_FIELDS:
        correct = sum(1 for r in records if r["comparison"]["fields"][field])
        per_field[field] = round(correct / total, 4)
    items_ok = sum(1 for r in records if r["comparison"]["line_items"]["all_match"])
    per_field["line_items"] = round(items_ok / total, 4)

    amounts_matched = sum(r["comparison"]["line_items"]["amounts_matched"] for r in records)
    amounts_total = sum(r["comparison"]["line_items"]["amounts_total"] for r in records)

    by_kind: dict[str, dict] = {}
    for kind in sorted({r["kind"] for r in records}):
        subset = [r for r in records if r["kind"] == kind]
        by_kind[kind] = {
            "documents": len(subset),
            "fully_correct": round(
                sum(1 for r in subset if r["comparison"]["fully_correct"]) / len(subset), 4
            ),
            "total_field_accuracy": round(
                sum(1 for r in subset if r["comparison"]["fields"]["total"]) / len(subset), 4
            ),
            "routing": routing_metrics(subset, "route_full"),
        }

    planted = [r for r in records if r["planted_issue"]]
    planted_caught = [
        r for r in planted if r["planted_issue"] in r.get("issue_codes_recomputed", [])
    ]

    latencies = sorted(r["latency_s"] for r in records if r["latency_s"] > 0)
    tokens_in = sum(r["input_tokens"] for r in records)
    tokens_out = sum(r["output_tokens"] for r in records)
    cost = (
        tokens_in / 1_000_000 * cfg.input_price_per_mtok
        + tokens_out / 1_000_000 * cfg.output_price_per_mtok
    )

    routing = {
        "валідація + впевненість": routing_metrics(records, "route_full"),
        "тільки впевненість": routing_metrics(records, "route_conf_only"),
        "тільки валідація": routing_metrics(records, "route_valid_only"),
    }

    review_rate = routing["валідація + впевненість"]["review_rate"]
    manual_hours = MANUAL_SECONDS_PER_DOC * 1000 / 3600
    auto_hours = REVIEW_SECONDS_PER_DOC * review_rate * 1000 / 3600

    return {
        "documents": total,
        "fully_correct": round(sum(1 for r in records if r["comparison"]["fully_correct"]) / total, 4),
        "field_accuracy": per_field,
        "line_amounts_matched": round(amounts_matched / max(1, amounts_total), 4),
        "by_kind": by_kind,
        "routing": routing,
        "planted_issues": {
            "total": len(planted),
            "caught": len(planted_caught),
            "detail": {r["doc_id"]: {"planted": r["planted_issue"],
                                     "caught": r["planted_issue"] in r.get("issue_codes_recomputed", [])}
                       for r in planted},
        },
        "latency_s": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": round(latencies[len(latencies) // 2], 2) if latencies else 0.0,
            "p95": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 2)
            if latencies
            else 0.0,
        },
        "tokens": {"input": tokens_in, "output": tokens_out,
                   "per_doc_input": round(tokens_in / total), "per_doc_output": round(tokens_out / total)},
        "cost_usd": {"run": round(cost, 4), "per_1000_docs": round(cost / total * 1000, 2)},
        "roi": {
            "manual_seconds_per_doc": MANUAL_SECONDS_PER_DOC,
            "review_seconds_per_doc": REVIEW_SECONDS_PER_DOC,
            "review_rate": review_rate,
            "human_hours_per_1000_manual": round(manual_hours, 1),
            "human_hours_per_1000_auto": round(auto_hours, 1),
            "saved_hours_per_1000": round(manual_hours - auto_hours, 1),
        },
        "config": cfg.to_dict(),
    }


# --------------------------------------------------------------------------
# Звіт
# --------------------------------------------------------------------------

def to_markdown(report: dict) -> str:
    lines = ["# Результати eval", ""]
    lines.append(f"Документів: **{report['documents']}**, "
                 f"повністю коректних: **{report['fully_correct'] * 100:.1f}%**")
    lines.append("")

    lines.append("## Точність по полях")
    lines.append("")
    lines.append("| Поле | Точність |")
    lines.append("|---|---|")
    for field, value in report["field_accuracy"].items():
        lines.append(f"| `{field}` | {value * 100:.1f}% |")
    lines.append("")

    lines.append("## Маршрутизація: що ловить кожен шар")
    lines.append("")
    lines.append("| Конфігурація | Спіймано | Пропущено | Хибних тривог | Recall | Precision | Автоматично |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, m in report["routing"].items():
        lines.append(
            f"| {name} | {m['caught']} | {m['missed']} | {m['false_alarms']} | "
            f"{m['recall'] * 100:.1f}% | {m['precision'] * 100:.1f}% | {m['auto_rate'] * 100:.1f}% |"
        )
    lines.append("")

    lines.append("## PDF проти фото")
    lines.append("")
    lines.append("| Вхід | Документів | Повністю коректних | Точність `total` | Recall маршрутизації |")
    lines.append("|---|---|---|---|---|")
    for kind, m in report["by_kind"].items():
        lines.append(
            f"| {kind} | {m['documents']} | {m['fully_correct'] * 100:.1f}% | "
            f"{m['total_field_accuracy'] * 100:.1f}% | {m['routing']['recall'] * 100:.1f}% |"
        )
    lines.append("")

    planted = report["planted_issues"]
    lines.append(
        f"## Закладені невідповідності\n\nСпіймано **{planted['caught']} з {planted['total']}**.\n"
    )
    lines.append("| Документ | Закладено | Спіймано |")
    lines.append("|---|---|---|")
    for doc_id, info in planted["detail"].items():
        lines.append(f"| {doc_id} | `{info['planted']}` | {'так' if info['caught'] else 'ні'} |")
    lines.append("")

    cost, roi, lat = report["cost_usd"], report["roi"], report["latency_s"]
    lines.append("## Час і вартість")
    lines.append("")
    lines.append(f"- Латентність: середня {lat['mean']} с, p50 {lat['p50']} с, p95 {lat['p95']} с")
    lines.append(f"- Токенів на документ: {report['tokens']['per_doc_input']} вхідних, "
                 f"{report['tokens']['per_doc_output']} вихідних")
    lines.append(f"- Вартість 1000 документів: **${cost['per_1000_docs']}**")
    lines.append(
        f"- Людино-годин на 1000 документів: вручну {roi['human_hours_per_1000_manual']}, "
        f"з системою {roi['human_hours_per_1000_auto']} "
        f"(**економія {roi['saved_hours_per_1000']} год**)"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Запуск
# --------------------------------------------------------------------------

def load_golden(path: Path) -> dict[str, dict]:
    golden = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            golden[entry["doc_id"]] = entry
    return golden


async def run_eval(args: argparse.Namespace) -> int:
    """
    `--from-ledger` рахує метрики з готового прогону і не робить жодного
    мережевого виклику — саме так відтворюються числа README.
    """
    golden = load_golden(args.golden)
    cfg = PipelineConfig()
    if getattr(args, "min_confidence", None) is not None:
        cfg.min_field_confidence = args.min_confidence
    if getattr(args, "model", None):
        cfg.model = args.model

    run_path = Path("output/eval-run.json")

    if args.from_ledger:
        docs = [
            ProcessedDocument.model_validate(d)
            for d in json.loads(Path(args.from_ledger).read_text(encoding="utf-8"))
        ]
        logger.info("метрики з готового прогону: %s (%d документів)", args.from_ledger, len(docs))
    else:
        import os

        from src.llm import GeminiVisionClient
        from src.pipeline import DocumentPipeline
        from src.sinks import JsonlSink
        from src.store import DocumentStore

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.error("GEMINI_API_KEY не заданий: eval без ключа можливий лише з --from-ledger")
            return 2

        # Свіжа база на кожен прогін: інакше другий запуск eval побачить усі
        # документи як дублі й порахує метрики по порожньому набору.
        if Path(args.db).exists():
            Path(args.db).unlink()
        client = GeminiVisionClient(
            api_key=api_key, model_name=cfg.model, rpm=cfg.rpm, temperature=cfg.temperature
        )
        store = DocumentStore(args.db)
        pipeline = DocumentPipeline(
            client=client, store=store, sink=JsonlSink(args.sink), cfg=cfg
        )
        paths = [Path(entry["file"]) for entry in golden.values()]
        if args.limit:
            paths = paths[: args.limit]
        docs = await pipeline.process_many(paths, concurrency=args.concurrency)
        store.close()

        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(
            json.dumps([d.model_dump(mode="json") for d in docs], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("сирий прогін збережено: %s", run_path)

    records = build_records(docs, golden, cfg)
    report = aggregate(records, cfg)
    report["records"] = records

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(to_markdown(report), encoding="utf-8")

    logger.info(
        "документів %d | повністю коректних %.1f%% | recall маршрутизації %.1f%% | $%.2f / 1000 док.",
        report["documents"],
        report["fully_correct"] * 100,
        report["routing"]["валідація + впевненість"]["recall"] * 100,
        report["cost_usd"]["per_1000_docs"],
    )
    logger.info("звіт: %s і %s", args.report, args.markdown)
    return 0
