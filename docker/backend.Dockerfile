FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY configs ./configs
COPY data ./data
COPY docker ./docker

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

CMD ["python", "-m", "app.server.main"]
