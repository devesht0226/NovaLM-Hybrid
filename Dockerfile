FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src

RUN pip install --upgrade pip && pip install --no-cache-dir -e .

EXPOSE 8080

# Mount artifacts/checkpoints, artifacts/tokenizers, data/index via volumes or bind mounts.
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
