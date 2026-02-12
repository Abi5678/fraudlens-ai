# FraudLens AI - NVIDIA GPU Container
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install system dependencies (curl required for HEALTHCHECK)
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose port
EXPOSE 8501

# Health check (start-period allows slow NVIDIA image boot)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f -s http://localhost:8501/_stcore/health || exit 1

# Run Streamlit (explicit port for HF Spaces)
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
