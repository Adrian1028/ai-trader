# ══════════════════════════════════════════════════════════════════
# AI Trading System — Production Docker Image
# ══════════════════════════════════════════════════════════════════
# Multi-stage build optimized for free-tier cloud VMs:
#   - Oracle Cloud ARM (aarch64) + AMD (x86_64)
#   - Minimal memory footprint (~200MB RSS)
#   - Non-root execution for security
#
# Build:
#   docker build -t ai-trader .
#
# Run:
#   docker run -d --name ai-trader --env-file .env \
#     -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs \
#     --restart unless-stopped ai-trader

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for numpy/pandas C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install only production deps (skip streamlit/plotly/pytest to save ~300MB)
RUN pip install --no-cache-dir --prefix=/install \
        aiohttp>=3.9 \
        numpy>=1.26 \
        pandas>=2.1 \
        pydantic>=2.0 \
        apscheduler>=3.10 \
        python-dotenv>=1.0 \
        google-genai>=1.0

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy pre-built Python packages from builder
COPY --from=builder /install /usr/local

# Copy source code and config
COPY src/  /app/src/
COPY config/ /app/config/

# Create mount points for persistent storage
RUN mkdir -p /app/data /app/logs/memory /app/logs/opro /app/logs/audit

# Non-root user for security
RUN groupadd -r botuser && useradd -r -g botuser botuser \
    && chown -R botuser:botuser /app
USER botuser

# Health check: verify Python and key imports work
HEALTHCHECK --interval=120s --timeout=10s --retries=3 \
    CMD python -c "from src.core.orchestrator import TradingSystem; print('OK')"

# Default entry point
CMD ["python", "src/main.py"]
