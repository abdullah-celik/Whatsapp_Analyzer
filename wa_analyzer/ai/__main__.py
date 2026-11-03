"""Local AI analyses.  Usage:  python -m wa_analyzer.ai COMMAND CHAT [options]

Commands
  doctor     check PyTorch, GPU, Ollama and models
  sentiment  tone + emotion per turn with transformer models
  topics     topic model (multilingual embeddings + clustering)
  read       local LLM (Ollama) reads conversations: warmth, engagement, interest, tone, summaries
  label      local LLM labels a sample of messages (training data)
  review     check the LLM labels yourself in the terminal -> accuracy of LLM and word lists
  train      train a fast classifier on the labels
  all        sentiment + topics + read

Results go to '<chat name>_ai/' next to the chat; the normal report picks them up automatically:
  python -m wa_analyzer CHAT --open
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..analysis import Settings, build_conversations, build_turns, prepare
from ..loading import default_ai_dir, load_filtered


def _common(p: argparse.ArgumentParser, chat=True):
    if chat:
        p.add_argument("chat", help="exported chat (.txt or .zip)")
    p.add_argument("--out", help="results folder (default: '<chat name>_ai' next to the chat)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="where models run: auto (GPU if available), cpu, or cuda (NVIDIA GPU)")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--date-order", choices=["DMY", "MDY", "YMD"])
    p.add_argument("--seed", type=int, default=0)


def _ollama_args(p):
    p.add_argument("--model", help="Ollama model (default: qwen2.5:3b on CPU, qwen2.5:7b on GPU)")
    p.add_argument("--host", default="http://localhost:11434", help="Ollama server address")
    p.add_argument("--num-ctx", type=int, default=8192, help="context window for the LLM")


def _load(a):
    df = load_filtered(a.chat, a.date_order, a.since, a.until, a.exclude)
    s = Settings()
    m = prepare(df, s)
    turns = build_turns(m)
    convs = build_conversations(m, turns, s)
    out = Path(a.out) if a.out else default_ai_dir(a.chat)
    print(f"{len(m):,} messages from {m['author'].nunique()} people · results -> {out}")
    return m, turns, convs, out


def _llm(a):
    from . import device as D
    from .ollama import Ollama
    dev = "cpu" if a.device == "cpu" else ("cuda" if a.device == "cuda" else D.resolve("auto"))
    return Ollama(model=a.model, host=a.host, device=dev, num_ctx=a.num_ctx)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m wa_analyzer.ai", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="check PyTorch, GPU, Ollama and models")
    _common(p, chat=False)
    _ollama_args(p)

    p = sub.add_parser("sentiment", help="tone + emotion per turn")
    _common(p)
    p.add_argument("--sentiment-model", default=None)
    p.add_argument("--turkish-model", default=None, help="Turkish sentiment model ('none' to skip)")
    p.add_argument("--emotion-model", default=None, help="emotion model ('none' to skip)")
    p.add_argument("--max-turns", type=int, help="analyse only a sample of turns (spread over months) - faster on CPU")

    p = sub.add_parser("topics", help="topic model")
    _common(p)
    p.add_argument("--topics", type=int, help="number of topics (default: chosen automatically)")
    p.add_argument("--name-with-llm", action="store_true", help="let the Ollama model name the topics")
    _ollama_args(p)

    p = sub.add_parser("read", help="local LLM reads conversations")
    _common(p)
    _ollama_args(p)
    p.add_argument("--per-month", type=int, help="conversations per month (default: 4 on CPU, 12 on GPU)")
    p.add_argument("--all", action="store_true", help="read every conversation (slow on CPU)")
    p.add_argument("--min-msgs", type=int, default=6, help="only conversations with at least this many messages")
    p.add_argument("--max-chars", type=int, help="max transcript length per conversation (default: 4000 CPU, 8000 GPU)")
    p.add_argument("--no-monthly", action="store_true", help="skip the monthly and overall summaries")

    p = sub.add_parser("label", help="LLM labels a sample of messages")
    _common(p)
    _ollama_args(p)
    p.add_argument("--n", type=int, help="messages to label (default: 600 on CPU, 2000 on GPU)")
    p.add_argument("--batch", type=int, default=15, help="messages per LLM call")

    p = sub.add_parser("review", help="check the LLM labels yourself")
    _common(p)
    p.add_argument("--n", type=int, default=100, help="messages to review in this session")

    p = sub.add_parser("accuracy", help="show accuracy of LLM labels and word lists against your reviews")
    _common(p)

    p = sub.add_parser("train", help="train the fast message classifier")
    _common(p)

    p = sub.add_parser("all", help="sentiment + topics + read")
    _common(p)
    _ollama_args(p)
    p.add_argument("--per-month", type=int)
    p.add_argument("--max-turns", type=int)

    a = ap.parse_args(argv)
    try:
        return run(a)
    except KeyboardInterrupt:
        print("\nStopped. Finished work is cached; run the same command again to continue.")
        return 130
    except ImportError as e:
        print(f"\nMissing package: {e}\nInstall the AI requirements:\n"
              "  CPU:         bash setup_cpu.sh\n"
              "  NVIDIA GPU:  bash setup_gpu.sh")
        return 1
    except Exception as e:
        from .ollama import OllamaError
        if isinstance(e, OllamaError):
            print(f"\nOllama problem: {e}")
            return 1
        raise


def run(a) -> int:
    from . import device as D
    if a.cmd == "doctor":
        return doctor(a)
    if a.cmd in ("review", "accuracy"):
        from . import labeling as L
        out = Path(a.out) if a.out else default_ai_dir(a.chat)
        if a.cmd == "review":
            L.review(out, n=a.n, seed=None)
        else:
            L.accuracy(out)
        return 0
    if a.cmd == "train":
        from .classifier import train
        out = Path(a.out) if a.out else default_ai_dir(a.chat)
        train(out, device=a.device, seed=a.seed)
        return 0

    m, turns, convs, out = _load(a)
    dev = D.resolve(a.device)
    gpu = dev == "cuda"

    if a.cmd in ("sentiment", "all"):
        from . import sentiment as S
        none = lambda x, d: None if (x or "").lower() == "none" else (x or d)
        S.run(m, out, device=a.device,
              sentiment_model=getattr(a, "sentiment_model", None) or S.SENTIMENT_MODEL,
              turkish_model=none(getattr(a, "turkish_model", None), S.TURKISH_MODEL),
              emotion_model=none(getattr(a, "emotion_model", None), S.EMOTION_MODEL),
              max_turns=a.max_turns, seed=a.seed)
    if a.cmd in ("topics", "all"):
        from . import topics as T
        llm = _llm(a) if getattr(a, "name_with_llm", False) or a.cmd == "all" else None
        if llm is not None and not llm.available():
            print("(Ollama not reachable: topics will be named by their keywords)")
            llm = None
        T.run(m, out, device=a.device, n_topics=getattr(a, "topics", None), ollama=llm, seed=a.seed)
    if a.cmd in ("read", "all"):
        from . import reader as R
        llm = _llm(a)
        per = None if getattr(a, "all", False) else (a.per_month or (12 if gpu else 4))
        R.run(m, convs, out, llm, per_month=per, min_msgs=getattr(a, "min_msgs", 6),
              max_chars=getattr(a, "max_chars", None) or (8000 if gpu else 4000),
              monthly=not getattr(a, "no_monthly", False), seed=a.seed)
    if a.cmd == "label":
        from . import labeling as L
        L.label(m, out, _llm(a), n=a.n or (2000 if gpu else 600), batch=a.batch, seed=a.seed)
        print("Next: check a sample yourself with  python -m wa_analyzer.ai review CHAT  "
              "and then  python -m wa_analyzer.ai train CHAT")
    print(f"\nDone. Create the report with:  python -m wa_analyzer \"{a.chat}\" --open")
    return 0


def doctor(a) -> int:
    print("Python packages:")
    for mod in ("torch", "transformers", "sentence_transformers", "sklearn", "joblib"):
        try:
            mm = __import__(mod)
            print(f"  ok   {mod} {getattr(mm, '__version__', '')}")
        except Exception:
            print(f"  --   {mod} (not installed)")
    try:
        import torch
        print(f"\nPyTorch CUDA build: {torch.version.cuda or 'no (CPU-only build)'}")
        print(f"NVIDIA GPU usable:  {'yes - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'}")
    except Exception:
        pass
    from . import device as D
    dev = D.resolve(a.device)
    print(f"Models would run on: {D.describe(dev)}")
    llm = _llm(a)
    print(f"\nOllama at {llm.host}: ", end="")
    if not llm.available():
        print("not reachable. Install from https://ollama.com and start it (ollama serve).")
        return 0
    names = llm.models()
    print(f"running, {len(names)} model(s): {', '.join(names) or '-'}")
    try:
        llm.check()
        print(f"Model for 'read'/'label': {llm.model} (installed)")
    except Exception as e:
        print(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
