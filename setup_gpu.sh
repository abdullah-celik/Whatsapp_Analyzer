#!/usr/bin/env bash
# Install everything for a computer WITH an NVIDIA GPU (CUDA build of PyTorch).
# Needs a recent NVIDIA driver (check with: nvidia-smi). Ollama uses the GPU automatically.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dashboard.txt
pip install torch            # the default Linux build of PyTorch includes CUDA
pip install -r requirements-ai.txt
echo
python -m wa_analyzer.ai doctor --device cuda
echo
echo "Done. Activate the environment in new terminals with:  source .venv/bin/activate"
