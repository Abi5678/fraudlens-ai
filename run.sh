#!/usr/bin/env bash
# FraudLens AI — run Streamlit app locally

set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "No venv found. Run: ./setup.sh"
  exit 1
fi

if [ ! -f ".env" ]; then
  echo "No .env found. Copy .env.example to .env and set NVIDIA_API_KEY."
  cp -n .env.example .env 2>/dev/null || true
  if ! grep -q "your_nvidia_api_key_here" .env 2>/dev/null; then
    :
  else
    echo "  Edit .env and add your NVIDIA API key from https://build.nvidia.com"
  fi
fi

source venv/bin/activate
exec streamlit run ui/app.py --server.port 8501
