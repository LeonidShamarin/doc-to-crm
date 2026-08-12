FROM python:3.11-slim

# Hugging Face Spaces запускає контейнер від НЕ-root користувача з uid 1000.
# Якщо лишити теку власністю root, сервіс упаде при першій спробі створити
# state/documents.sqlite3 — і Space покаже «Runtime error» без причини.
RUN useradd -m -u 1000 user

# Кирилічний шрифт потрібен генератору синтетичних документів (`gen-dataset`):
# у reportlab вбудовані лише латинські. ~2 МБ, зате команда працює і в
# контейнері, а не тільки на машині розробки.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src/ src/
COPY --chown=user web/ web/
COPY --chown=user data/ data/
COPY --chown=user main.py .

# Стан сервісу (черга review, ledger ідемпотентності) і вивід. У образ не
# копіюються: це похідні дані конкретного запуску, а не частина збірки.
RUN mkdir -p state output

# 7860 — порт, який Hugging Face Spaces очікує від Docker-контейнера.
ENV PORT=7860
EXPOSE 7860

# Порт береться з $PORT, а не зашитий: інакше healthcheck мовчки перевіряв би
# не той сервіс, щойно порт змінили через оточення.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT','7860')}/health\").status==200 else 1)"

# GEMINI_API_KEY передавати через `docker run -e ...`, --env-file .env або
# Settings → Variables and secrets у Space. У репозиторії ключа немає ніколи.
# Без ключа сервіс усе одно піднімається: працює черга, review-UI і рішення
# людини, а POST /documents віддає 503.
ENTRYPOINT ["python", "main.py", "serve"]
