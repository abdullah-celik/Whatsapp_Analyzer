"""CPU / NVIDIA GPU selection for transformer models and Ollama."""
from __future__ import annotations

import os


def resolve(pref: str = "auto") -> str:
    """'auto' -> 'cuda' if an NVIDIA GPU is usable, else 'cpu'."""
    pref = (pref or "auto").lower()
    if pref == "cpu":
        _cpu_threads()
        return "cpu"
    try:
        import torch
        ok = torch.cuda.is_available()
    except Exception:
        ok = False
    if pref == "cuda" and not ok:
        print("! --device cuda requested but no usable NVIDIA GPU/CUDA build of PyTorch was found -> using CPU.\n"
              "  (Install the GPU build with: bash setup_gpu.sh and check the driver with nvidia-smi)")
    if ok and pref in ("auto", "cuda"):
        return "cuda"
    _cpu_threads()
    return "cpu"


def _cpu_threads():
    try:
        import torch
        n = os.cpu_count() or 4
        torch.set_num_threads(max(1, n))
    except Exception:
        pass


def describe(device: str) -> str:
    if device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"NVIDIA GPU: {name} ({mem:.1f} GB)"
        except Exception:
            return "NVIDIA GPU"
    return f"CPU ({os.cpu_count()} threads)"


def batch_size(device: str, kind: str = "classifier") -> int:
    if device == "cuda":
        return 128 if kind == "classifier" else 256
    return 16 if kind == "classifier" else 64


def torch_dtype(device: str):
    import torch
    return torch.float16 if device == "cuda" else torch.float32
