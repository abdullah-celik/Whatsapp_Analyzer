"""Train a fast message classifier on the LLM labels and apply it to every message.

Sentence embeddings (multilingual MiniLM) + one logistic regression per category. Your reviewed labels
(from `review`) override the LLM labels. Quality is measured with 5-fold cross-validation, and compared with
the word lists on the same messages.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..lexicon_engine import WordClassifier
from . import device as D
from .cache import JsonlCache, text_key
from .labeling import CATEGORY_HELP, labels_frame
from .topics import EMBED_MODEL


def _embed(texts: list[str], model_name: str, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()
    return model.encode(texts, batch_size=D.batch_size(device, "embed"), convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def train(out_dir: str | Path, device: str = "auto", embed_model: str = EMBED_MODEL, min_pos: int = 15,
          seed: int = 0, log=print, embed_fn=None) -> pd.DataFrame:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    out_dir = Path(out_dir)
    labs = labels_frame(out_dir)
    if len(labs) < 100:
        raise ValueError(f"Only {len(labs)} labelled messages. Label at least a few hundred first "
                         "(python -m wa_analyzer.ai label CHAT --n 1500).")
    rev = JsonlCache(out_dir / "reviews.jsonl")
    n_rev = 0
    for i, k in enumerate(labs["key"]):
        r = rev.get(k)
        if r is not None:
            n_rev += 1
            for c in CATEGORY_HELP:
                labs.loc[i, c] = int(c in r["labels"])
    dev = D.resolve(device)
    log(f"Embedding {len(labs):,} labelled messages ({D.describe(dev)}) ...")
    X = (embed_fn or (lambda t: _embed(t, embed_model, dev)))(labs["text"].tolist())
    lex = WordClassifier()
    lex_pred = pd.DataFrame([lex.classify(t) for t in labs["text"]])

    models, thresholds, rows = {}, {}, []
    for c in CATEGORY_HELP:
        y = labs[c].to_numpy()
        pos = int(y.sum())
        if pos < min_pos or pos > len(y) - min_pos:
            rows.append({"category": c, "positives": pos, "note": f"skipped (needs {min_pos}+ examples)"})
            continue
        lr = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        prob = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
        grid = np.linspace(0.2, 0.9, 29)
        f1s = [f1_score(y, prob >= t, zero_division=0) for t in grid]
        th = float(grid[int(np.argmax(f1s))])
        pred = prob >= th
        lp = lex_pred[c].to_numpy().astype(bool) if c in lex_pred else np.zeros_like(pred)
        rows.append({"category": c, "positives": pos,
                     "model precision": precision_score(y, pred, zero_division=0),
                     "model recall": recall_score(y, pred, zero_division=0),
                     "model F1": f1_score(y, pred, zero_division=0),
                     "word lists F1": f1_score(y, lp, zero_division=0), "threshold": th, "note": ""})
        models[c] = lr.fit(X, y)
        thresholds[c] = th
    metrics = pd.DataFrame(rows).set_index("category")
    path = out_dir / "message_classifier.joblib"
    joblib.dump({"embed_model": embed_model, "models": models, "thresholds": thresholds,
                 "labels": {c: c.replace("_", " ") for c in models}, "n_train": len(labs), "n_reviewed": n_rev}, path)
    metrics.to_csv(out_dir / "classifier_metrics.csv")
    pd.set_option("display.width", 200)
    log(metrics.round(2).to_string())
    log(f"\nSaved classifier to {path}  (agreement measured against the LLM labels"
        f"{f', with your {n_rev} reviews taking priority' if n_rev else ''}).")
    return metrics


def apply_trained(path: str | Path, m: pd.DataFrame, device: str = "auto", embed_fn=None) -> tuple[pd.DataFrame, dict]:
    """0/1 per category for every message (non-text messages are 0). Predictions are cached per text."""
    import joblib
    bundle = joblib.load(path)
    cats = list(bundle["models"])
    is_text = m["kind"] == "text"
    texts = m.loc[is_text, "text"].str.slice(0, 400)
    uniq = pd.unique(texts)
    cache = JsonlCache(Path(path).with_suffix(".predictions.jsonl"))
    tag = Path(path).stat().st_mtime
    keys = {t: text_key(str(tag), t) for t in uniq}
    todo = [t for t in uniq if keys[t] not in cache]
    if todo:
        dev = D.resolve(device)
        E = (embed_fn or (lambda t: _embed(t, bundle["embed_model"], dev)))(list(todo))
        probs = {c: bundle["models"][c].predict_proba(E)[:, 1] for c in cats}
        cache.put_many([(keys[t], {"p": {c: round(float(probs[c][i]), 4) for c in cats}}) for i, t in enumerate(todo)])
    P = pd.DataFrame([cache.get(keys[t])["p"] for t in texts], index=texts.index)
    out = pd.DataFrame(0, index=m.index, columns=cats)
    for c in cats:
        out.loc[is_text, c] = (P[c] >= bundle["thresholds"][c]).astype(int)
    return out, bundle["labels"]
