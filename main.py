"""
CLI обробника первинних документів.

    python main.py gen-dataset                  # згенерувати синтетичні рахунки
    python main.py process data/synthetic       # прогнати теку через пайплайн
    python main.py watch data/inbox             # стежити за текою
    python main.py serve                        # HTTP + review-UI на :7860
    python main.py eval                         # точність по полях проти еталона

Env: GEMINI_API_KEY (обов'язковий для екстракції).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from src.config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("doc-to-crm")

DEFAULT_DATASET = Path("data/synthetic")
DEFAULT_GOLDEN = Path("data/golden.jsonl")
DEFAULT_DB = Path("state/documents.sqlite3")
DEFAULT_SINK = Path("output/ledger.jsonl")


def _use_system_trust_store() -> None:
    """
    httpx (усередині google-genai) довіряє лише бандлу `certifi`. У мережах з
    TLS-інспекцією корпоративний корінь є у сховищі ОС, але не в certifi — і всі
    виклики падають з CERTIFICATE_VERIFY_FAILED. truststore бере довіру звідти,
    де вона реально налаштована. Перевірку сертифікатів це не послаблює.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        logger.debug("truststore недоступний, лишаємось на certifi")


def _require_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        logger.error(
            "GEMINI_API_KEY не заданий. Створи .env за зразком .env.example "
            "або експортуй змінну оточення."
        )
        sys.exit(2)
    return key


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Модель Gemini")
    parser.add_argument("--rpm", type=int, default=None, help="Ліміт запитів на хвилину")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Поріг впевненості по полю; нижче — документ іде на ручну перевірку",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Вимкнути шар арифметичних перевірок (для порівняння в eval)",
    )


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "rpm", None):
        cfg.rpm = args.rpm
    if getattr(args, "min_confidence", None) is not None:
        cfg.min_field_confidence = args.min_confidence
    if getattr(args, "no_validation", False):
        cfg.validation_enabled = False
    return cfg


# --------------------------------------------------------------------------
# Команди
# --------------------------------------------------------------------------

def cmd_gen_dataset(args: argparse.Namespace) -> int:
    from src.render import generate_dataset

    docs = generate_dataset(
        out_dir=args.out,
        golden_path=args.golden,
        count=args.count,
        photo_ratio=args.photo_ratio,
        seed=args.seed,
        today=date.today(),
    )
    photos = sum(1 for d in docs if d.kind == "photo")
    planted = [d for d in docs if d.planted_issue]
    logger.info(
        "згенеровано %d документів (%d PDF, %d фото), шаблонів: %d, з підміною: %d",
        len(docs),
        len(docs) - photos,
        photos,
        len({d.template for d in docs}),
        len(planted),
    )
    for d in planted:
        logger.info("  підміна %s → %s", d.doc_id, d.planted_issue)
    logger.info("еталон: %s", args.golden)
    return 0


async def _run_process(args: argparse.Namespace) -> int:
    from src.pipeline import DocumentPipeline
    from src.llm import GeminiVisionClient
    from src.sinks import JsonlSink
    from src.store import DocumentStore

    _use_system_trust_store()
    cfg = _config_from_args(args)
    client = GeminiVisionClient(api_key=_require_key(), model_name=cfg.model, rpm=cfg.rpm,
                                temperature=cfg.temperature)
    store = DocumentStore(args.db)
    sink = JsonlSink(args.sink)
    pipeline = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)

    paths = sorted(p for p in args.path.rglob("*") if p.is_file()) if args.path.is_dir() else [args.path]
    docs = await pipeline.process_many(paths, concurrency=args.concurrency, limit=args.limit)

    store.close()
    summary = pipeline.summary(docs)
    logger.info(json.dumps(summary, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("звіт: %s", args.report)
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    return asyncio.run(_run_process(args))


async def _run_watch(args: argparse.Namespace) -> int:
    from src.llm import GeminiVisionClient
    from src.pipeline import DocumentPipeline
    from src.sinks import JsonlSink
    from src.store import DocumentStore
    from src.watcher import watch_folder

    _use_system_trust_store()
    cfg = _config_from_args(args)
    client = GeminiVisionClient(api_key=_require_key(), model_name=cfg.model, rpm=cfg.rpm,
                                temperature=cfg.temperature)
    store = DocumentStore(args.db)
    sink = JsonlSink(args.sink)
    pipeline = DocumentPipeline(client=client, store=store, sink=sink, cfg=cfg)

    args.path.mkdir(parents=True, exist_ok=True)
    logger.info("стежу за текою %s (Ctrl+C — вихід)", args.path)
    try:
        await watch_folder(args.path, pipeline, interval=args.interval, max_cycles=args.max_cycles)
    except KeyboardInterrupt:
        logger.info("зупинено користувачем")
    finally:
        store.close()
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    return asyncio.run(_run_watch(args))


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from src.app import create_app

    _use_system_trust_store()
    app = create_app(db_path=args.db, sink_path=args.sink)
    port = args.port or int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from src.evaluate import run_eval

    return asyncio.run(run_eval(args))


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-to-crm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen-dataset", help="Згенерувати синтетичні документи + еталон")
    p_gen.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    p_gen.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p_gen.add_argument("--count", type=int, default=40)
    p_gen.add_argument("--photo-ratio", type=float, default=0.3)
    p_gen.add_argument("--seed", type=int, default=20260812)
    p_gen.set_defaults(func=cmd_gen_dataset)

    p_proc = sub.add_parser("process", help="Обробити файл або теку")
    p_proc.add_argument("path", type=Path)
    p_proc.add_argument("--db", type=Path, default=DEFAULT_DB)
    p_proc.add_argument("--sink", type=Path, default=DEFAULT_SINK)
    p_proc.add_argument("--report", type=Path, default=None)
    p_proc.add_argument("--concurrency", type=int, default=2)
    p_proc.add_argument("--limit", type=int, default=None, help="Обробити не більше N файлів")
    _add_config_args(p_proc)
    p_proc.set_defaults(func=cmd_process)

    p_watch = sub.add_parser("watch", help="Стежити за текою і обробляти нові файли")
    p_watch.add_argument("path", type=Path, nargs="?", default=Path("data/inbox"))
    p_watch.add_argument("--db", type=Path, default=DEFAULT_DB)
    p_watch.add_argument("--sink", type=Path, default=DEFAULT_SINK)
    p_watch.add_argument("--interval", type=float, default=3.0)
    p_watch.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Зупинитись після N циклів опитування (за замовчуванням — без обмеження)",
    )
    _add_config_args(p_watch)
    p_watch.set_defaults(func=cmd_watch)

    p_serve = sub.add_parser("serve", help="HTTP API + review-UI")
    p_serve.add_argument("--db", type=Path, default=DEFAULT_DB)
    p_serve.add_argument("--sink", type=Path, default=DEFAULT_SINK)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_eval = sub.add_parser("eval", help="Точність по полях проти еталона")
    p_eval.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p_eval.add_argument("--db", type=Path, default=Path("state/eval.sqlite3"))
    p_eval.add_argument("--sink", type=Path, default=Path("output/eval-ledger.jsonl"))
    p_eval.add_argument("--report", type=Path, default=Path("output/eval-report.json"))
    p_eval.add_argument("--markdown", type=Path, default=Path("output/eval-report.md"))
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--concurrency", type=int, default=2)
    p_eval.add_argument(
        "--from-ledger",
        type=Path,
        default=None,
        help="Порахувати метрики з готового прогону, без викликів моделі",
    )
    _add_config_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
