FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY config ./config
RUN pip install ".[otlp]" && useradd --system --uid 10001 guard

USER 10001
EXPOSE 8800
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8800/healthz', timeout=2)" || exit 1

ENTRYPOINT ["mcp-guard"]
CMD ["serve", "--config", "/etc/mcp-guard/gateway.yaml"]
