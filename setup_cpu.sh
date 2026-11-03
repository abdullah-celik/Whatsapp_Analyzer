#!/usr/bin/env bash
# Install everything for a computer WITHOUT an NVIDIA GPU (CPU-only PyTorch, much smaller download).
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dashboard.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ai.txt
echo
python -m wa_analyzer.ai doctor --device cpu
echo
echo "Done. Activate the environment in new terminals with:  source .venv/bin/activate"
