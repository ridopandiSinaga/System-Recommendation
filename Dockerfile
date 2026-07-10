FROM python:3.12.11-slim-bookworm

ARG UV_VERSION=0.8.2
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PATH="/opt/app/.venv/bin:$PATH" \
    BOOK_RECOMMENDER_ARTIFACT="/opt/app/models/hybrid.joblib"

WORKDIR /opt/app

RUN pip install --no-cache-dir "uv==${UV_VERSION}" \
    && groupadd --system app \
    && useradd --system --gid app --home /opt/app app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY app ./app

RUN uv sync --frozen --no-dev --extra api \
    && mkdir -p /opt/app/models \
    && chown -R app:app /opt/app

USER app
EXPOSE 8000
VOLUME ["/opt/app/models"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)"]

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
