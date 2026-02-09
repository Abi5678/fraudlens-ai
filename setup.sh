#!/bin/bash
# FraudLens AI v2 - Setup Script

echo "🛡️ FraudLens AI v2 - NVIDIA Edition"
echo "=================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

echo "Checking Python version..."
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.11+ required (found $python_version)"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if not exists
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your NVIDIA_API_KEY"
fi

echo ""
echo "=================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your NVIDIA API key"
echo "  2. Activate the environment: source venv/bin/activate"
echo "  3. Run the UI: streamlit run ui/app.py"
echo "  4. Or use CLI: python fraudlens.py analyze <document.pdf>"
echo ""
echo "Get your NVIDIA API key at: https://build.nvidia.com"
echo ""
