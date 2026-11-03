"""Transformer sentiment and emotion per turn (runs locally on CPU or NVIDIA GPU).

Default models (downloaded once from Hugging Face, then cached on your computer):
  sentiment (multilingual): cardiffnlp/twitter-xlm-roberta-base-sentiment   negative / neutral / positive
  sentiment (Turkish):      savasy/bert-base-turkish-sentiment-cased        negatif / pozitif
  emotion (multilingual):   MilaNLProc/xlm-emo-t                            anger / fear / joy / sadness

Unlike word lists, these models read the whole message, so negation and context are understood
("seni sevmiyorum" is negative). Results are cached, so a second run only processes new turns.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import device as D
from .cache import JsonlCache, text_key
from .text import is_turkish, turn_texts

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TURKISH_MODEL = "savasy/bert-base-turkish-sentiment-cased"
EMOTION_MODEL = "MilaNLProc/xlm-emo-t"


def _canon(label: str, n_labels: int, idx: int) -> str:
    l = label.lower()
    if "neg" in l:
        return "negative"
    if "neu" in l:
        return "neutral"
    if "pos" in l or "poz" in l:
        return "positive"
    if l.startswith("label_"):  # LABEL_0.. order is negative, (neutral), positive for these models
        return ["negative", "positive"][idx] if n_labels == 2 else ["negative", "neutral", "positive"][idx]
    return l


class Classifier:
    """Wraps a Hugging Face sequence-classification model; returns label probabilities."""

    def __init__(self, name: str, device: str, max_length: int = 128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        if device == "cuda":
            self.model = self.model.half()
        self.model.to(device).eval()
        self.max_length = max_length
        n = self.model.config.num_labels
        id2 = self.model.config.id2label
        self.labels = [_canon(str(id2.get(i, f"LABEL_{i}")), n, i) for i in range(n)]

    def predict(self, texts: list[str], batch: int) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                out.append(self.torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))


def run(m: pd.DataFrame, out_dir: str | Path, device: str = "auto", sentiment_model: str = SENTIMENT_MODEL,
        turkish_model: str | None = TURKISH_MODEL, emotion_model: str | None = EMOTION_MODEL,
        max_turns: int | None = None, seed: int = 0, log=print) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = D.resolve(device)
    log(f"Device: {D.describe(dev)}")
    turns = turn_texts(m)
    if max_turns and len(turns) > max_turns:
        from .text import even_sample
        turns = even_sample(turns, max_turns, seed=seed).reset_index(drop=True)
        log(f"Using a sample of {len(turns):,} turns spread over all months (--max-turns).")
    turns["is_turkish"] = turns["text"].map(is_turkish)
    bs = D.batch_size(dev)
    cache = JsonlCache(out_dir / "cache_sentiment.jsonl")

    def process(model_name: str, rows: pd.DataFrame, tag: str) -> pd.DataFrame:
        keys = [text_key(model_name, t) for t in rows["text"]]
        todo = [i for i, k in enumerate(keys) if k not in cache]
        if todo:
            log(f"Loading {model_name} ...")
            clf = Classifier(model_name, dev)
            # longest first keeps batches of similar length = faster
            order = sorted(todo, key=lambda i: -len(rows["text"].iloc[i]))
            t0, done = time.time(), 0
            step = bs * 20
            for s in range(0, len(order), step):
                part = order[s:s + step]
                probs = clf.predict([rows["text"].iloc[i] for i in part], bs)
                cache.put_many([(keys[i], {"labels": clf.labels, "p": [round(float(x), 4) for x in pr]})
                                for i, pr in zip(part, probs)])
                done += len(part)
                rate = done / max(time.time() - t0, 1e-6)
                log(f"  {tag}: {done:,}/{len(order):,} turns ({rate:.0f}/s, ~{(len(order) - done) / max(rate, 1e-6) / 60:.1f} min left)")
            del clf
        recs = [cache.get(k) for k in keys]
        labels = recs[0]["labels"] if recs else []
        P = np.array([r["p"] for r in recs]) if recs else np.zeros((0, 0))
        return pd.DataFrame(P, columns=[f"{tag}:{l}" for l in labels], index=rows.index)

    parts = [process(sentiment_model, turns, "sent")]
    if turkish_model:
        tr = turns[turns["is_turkish"]]
        if len(tr):
            parts.append(process(turkish_model, tr, "tr"))
    if emotion_model:
        parts.append(process(emotion_model, turns, "emo"))
    res = pd.concat([turns.drop(columns=["text"])] + parts, axis=1)

    # combined polarity in [-1, 1]: multilingual (pos - neg), averaged with the Turkish model where available
    pol = res.get("sent:positive", 0) - res.get("sent:negative", 0)
    if "tr:positive" in res:
        trpol = res["tr:positive"] - res["tr:negative"]
        pol = np.where(res["tr:positive"].notna(), (pol + trpol) / 2, pol)
    res["polarity"] = pol
    sent_cols = [c for c in res.columns if c.startswith("sent:")]
    res["sentiment"] = res[sent_cols].idxmax(axis=1).str.replace("sent:", "", regex=False)
    emo_cols = [c for c in res.columns if c.startswith("emo:")]
    if emo_cols:
        res["emotion"] = res[emo_cols].idxmax(axis=1).str.replace("emo:", "", regex=False)
    res.to_csv(out_dir / "sentiment.csv", index=False)
    (out_dir / "sentiment_meta.json").write_text(json.dumps({
        "sentiment_model": sentiment_model, "turkish_model": turkish_model, "emotion_model": emotion_model,
        "device": dev, "turns": len(res), "created": pd.Timestamp.now().isoformat()}, indent=1))
    log(f"Saved {len(res):,} turns to {out_dir / 'sentiment.csv'}")
    return res
