FROM python:3.11-slim AS build

ARG INSTALL_HF=false
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m venv /opt/venv && \
    case "$INSTALL_HF" in \
      false) /opt/venv/bin/python -m pip install --no-cache-dir . ;; \
      true) /opt/venv/bin/python -m pip install --no-cache-dir '.[huggingface]' ;; \
      *) echo 'INSTALL_HF must be false or true' >&2; exit 2 ;; \
    esac

FROM python:3.11-slim

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/diagagent/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 diagagent && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin diagagent && \
    install -d -o 10001 -g 10001 /app/data/index /app/data/cache /app/reports/generated "$HF_HOME" && \
    install -d /app/data/raw

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY scripts/ ./scripts/
COPY configs/production.yaml ./configs/production.yaml

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health/live', timeout=3).close()"]
CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000", "--config", "/app/configs/production.yaml"]
