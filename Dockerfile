# FraudLens AI - Hugging Face Spaces / lightweight deployment
# For GPU deployment, use Dockerfile.gpu instead.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces runs as uid 1000)
RUN useradd -m -u 1000 user

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Writable dirs for Milvus Lite DB and temp uploads
RUN mkdir -p /tmp/milvus && chown -R user:user /app /tmp/milvus

ENV PYTHONPATH=/app
ENV HOME=/home/user
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

USER user

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f -s http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "ui/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.enableXsrfProtection", "false", \
     "--server.enableCORS", "false"]
