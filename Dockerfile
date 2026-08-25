FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY app ./app

# The optional host CA bundle lets the image build behind a TLS-inspecting proxy.
RUN --mount=type=secret,id=host_ca_certificates,target=/tmp/host-ca-certificates.crt \
    sh -c 'if [ -s /tmp/host-ca-certificates.crt ]; then export SSL_CERT_FILE=/tmp/host-ca-certificates.crt; fi; \
           uv sync --frozen --no-dev --no-cache'

RUN install -d -o nobody -g nogroup -m 0700 /var/lib/seshat/sources

USER nobody

EXPOSE 8080

CMD ["uvicorn", "app.main:create_app_from_config", "--factory", "--host", "0.0.0.0", "--port", "8080"]
