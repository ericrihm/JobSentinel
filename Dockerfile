# ── Stage 1: build wheel ────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools only in this stage
RUN pip install --no-cache-dir build

COPY pyproject.toml .
COPY sentinel/ sentinel/

# Build the wheel
RUN python -m build --wheel --outdir /dist

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim

# Metadata
LABEL org.opencontainers.image.title="JobSentinel"
LABEL org.opencontainers.image.description="AI-powered job scam detection platform"

# Non-root user for security
RUN groupadd --gid 1001 sentinel && \
    useradd  --uid 1001 --gid sentinel --no-create-home --shell /sbin/nologin sentinel

# Create data directory with correct ownership before switching users
RUN mkdir -p /data && chown sentinel:sentinel /data

WORKDIR /app

# Copy wheel from builder and install with all optional extras (ai, api, web)
COPY --from=builder /dist/*.whl /tmp/

RUN pip install --no-cache-dir /tmp/*.whl[full] && \
    rm /tmp/*.whl

# Switch to non-root user
USER sentinel

# Persistent data lives here; mount a volume to this path
VOLUME ["/data"]

# Point sentinel's storage paths at /data via env vars that config.py respects
ENV SENTINEL_DB_PATH=/data/sentinel.db \
    HOME=/data

EXPOSE 8080

ENTRYPOINT ["sentinel"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
