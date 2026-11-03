"""LLM-labelled training data, and a terminal tool to check the labels yourself.

  label   the local LLM labels a sample of messages with the classifier categories (+ sentiment)
  review  you go through a sample and correct the labels -> accuracy of the LLM AND of the word lists
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ..lexicon_engine import WordClassifier
from .cache import JsonlCache, text_key
from .ollama import Ollama
from .text import even_sample

PROMPT_VERSION = "labels-v1"

CATEGORY_HELP = {
    "affection": "love, missing, endearments (aşkım, canım), kisses/hugs, heart emojis",
    "care_questions": "asks how the other is / what they do / if they arrived, ate, slept; shows care",
    "plans": "proposes or arranges meeting, calling, doing something together, times and places",
    "compliments": "praises the other person (looks, character, achievements)",
    "enthusiasm": "excitement or surprise (wow, harika, süper, maşallah, oha)",
    "gratitude": "thanks",
    "apology": "apologises or asks forgiveness",
    "laughter": "laughs or finds something funny (haha, ksksks, 😂, koptum)",
    "good_morning_night": "good morning / good night / sweet dreams wishes",
    "negative": "annoyance, distance, rejection, sadness, complaints, ending/avoiding the conversation",
    "dry_reply": "the whole message is a minimal, low-effort reply (ok, tmm, hı, peki, 👍)",
}
SENTIMENTS = ["negative", "neutral", "positive"]


def categories() -> list[str]:
    return list(CATEGORY_HELP)


def _system() -> str:
    lines = "\n".join(f"- {k}: {v}" for k, v in CATEGORY_HELP.items())
    return ("You label WhatsApp messages. They can be Turkish, Italian, English or mixed, with slang, abbreviations "
            "and typos. For each message give ALL categories that apply (zero, one or several) and the sentiment. "
            "Use the previous message only as context; label the message itself.\nCategories:\n" + lines +
            "\nAnswer only with JSON.")


def _schema() -> dict:
    return {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "integer"},
        "labels": {"type": "array", "items": {"type": "string", "enum": categories()}},
        "sentiment": {"type": "string", "enum": SENTIMENTS}},
        "required": ["id", "labels", "sentiment"]}}}, "required": ["items"]}


def sample_messages(m: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    tm = m[m["kind"] == "text"].copy()
    prev_text = m["text"].where(m["kind"] == "text", "[media]").shift(1)
    prev_author = m["author"].shift(1)
    tm["prev"] = np.where(prev_author.loc[tm.index] != tm["author"], prev_text.loc[tm.index], "")
    tm["prev"] = tm["prev"].fillna("").astype(str).str.slice(0, 200)
    tm["text"] = tm["text"].str.slice(0, 400)
    tm["stratum"] = tm["month"].astype(str) + "|" + tm["author"]
    return even_sample(tm, n, by="stratum", seed=seed)


def label(m: pd.DataFrame, out_dir: str | Path, llm: Ollama, n: int = 1500, batch: int = 15, seed: int = 0,
          log=print) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm.check()
    smp = sample_messages(m, n, seed)
    cache = JsonlCache(out_dir / "labels.jsonl")
    keys = [text_key(PROMPT_VERSION, t, p) for t, p in zip(smp["text"], smp["prev"])]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    log(f"{len(smp):,} messages sampled, {len(smp) - len(todo):,} already labelled, labelling {len(todo):,} with {llm.model} ...")
    import time
    t0 = time.time()
    for b in range(0, len(todo), batch):
        part = todo[b:b + batch]
        lines = []
        for j, i in enumerate(part, 1):
            prev = smp["prev"].iloc[i]
            ctx = f'(previous message: "{prev}") ' if prev else ""
            lines.append(f'{j}. {ctx}MESSAGE: "{smp["text"].iloc[i]}"')
        try:
            out = llm.chat_json(_system(), "Label these messages:\n" + "\n".join(lines), _schema())
        except Exception as e:
            log(f"  ! batch skipped: {e}")
            continue
        got = {int(it.get("id", -1)): it for it in out.get("items", []) if isinstance(it, dict)}
        items = []
        for j, i in enumerate(part, 1):
            it = got.get(j)
            if it is None:
                continue
            labs = [l for l in it.get("labels", []) if l in CATEGORY_HELP]
            sent = it.get("sentiment") if it.get("sentiment") in SENTIMENTS else "neutral"
            items.append((keys[i], {"index": int(smp.index[i]), "author": smp["author"].iloc[i],
                                    "timestamp": str(smp["timestamp"].iloc[i]), "text": smp["text"].iloc[i],
                                    "prev": smp["prev"].iloc[i], "labels": labs, "sentiment": sent}))
        cache.put_many(items)
        done = b + len(part)
        rate = done / max(time.time() - t0, 1e-6)
        log(f"  {done:,}/{len(todo):,} ({rate * 60:.0f} messages/min, ~{(len(todo) - done) / max(rate, 1e-6) / 60:.0f} min left)")
    df = labels_frame(out_dir)
    df.to_csv(out_dir / "labels.csv", index=False)
    log(f"Saved {len(df):,} labelled messages to {out_dir / 'labels.csv'}")
    return df


def labels_frame(out_dir: str | Path) -> pd.DataFrame:
    cache = JsonlCache(Path(out_dir) / "labels.jsonl")
    rows = []
    for k, r in cache.data.items():
        row = {"key": k, "text": r["text"], "prev": r.get("prev", ""), "author": r.get("author"),
               "timestamp": r.get("timestamp"), "sentiment": r.get("sentiment")}
        for c in CATEGORY_HELP:
            row[c] = int(c in r.get("labels", []))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ review in the terminal

def review(out_dir: str | Path, n: int = 100, seed: int | None = None, inp=input, out=print) -> pd.DataFrame | None:
    out_dir = Path(out_dir)
    labs = labels_frame(out_dir)
    if labs.empty:
        out("No labels yet. Run:  python -m wa_analyzer.ai label CHAT")
        return None
    rev = JsonlCache(out_dir / "reviews.jsonl")
    todo = labs[~labs["key"].isin(rev.data)].sample(frac=1, random_state=seed)
    cats = list(CATEGORY_HELP)
    out("\nCategories:")
    for i, c in enumerate(cats, 1):
        out(f"  {i:>2} {c:<20} {CATEGORY_HELP[c]}")
    out("\nFor each message: press Enter if the labels are right, or type the numbers to switch on/off "
        "(e.g. '1 9'), then Enter.  s = skip,  q = quit and show results.\n")
    done = 0
    for _, r in todo.iterrows():
        if done >= n:
            break
        cur = {c for c in cats if r[c]}
        if r["prev"]:
            out(f"  (previous: {r['prev'][:120]})")
        out(f"  {r['author']}: {r['text']}")
        out(f"  LLM labels: {', '.join(sorted(cur)) or '(none)'}   sentiment: {r['sentiment']}")
        ans = inp("  > ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        for tok in ans.replace(",", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= len(cats):
                cur ^= {cats[int(tok) - 1]}
        rev.put(r["key"], {"labels": sorted(cur)})
        done += 1
        out(f"  saved ({len(rev.data)} reviewed)\n")
    return accuracy(out_dir, out=out)


def accuracy(out_dir: str | Path, out=print) -> pd.DataFrame | None:
    """Precision / recall / F1 of the LLM labels and of the word lists, against your reviewed labels."""
    out_dir = Path(out_dir)
    labs = labels_frame(out_dir).set_index("key")
    rev = JsonlCache(out_dir / "reviews.jsonl")
    keys = [k for k in rev.data if k in labs.index]
    if len(keys) < 10:
        out(f"Only {len(keys)} reviewed messages - review at least 10 (better 100+) to measure accuracy.")
        return None
    clf = WordClassifier()
    rows = []
    for c in CATEGORY_HELP:
        human = np.array([c in rev.get(k)["labels"] for k in keys])
        llm = labs.loc[keys, c].to_numpy().astype(bool)
        lex = np.array([bool(clf.classify(labs.loc[k, "text"]).get(c, 0)) for k in keys])
        rows.append({"category": c, "positives (you)": int(human.sum()),
                     **{f"LLM {k}": v for k, v in _prf(human, llm).items()},
                     **{f"word lists {k}": v for k, v in _prf(human, lex).items()}})
    df = pd.DataFrame(rows).set_index("category")
    df.to_csv(out_dir / "accuracy.csv")
    pd.set_option("display.width", 200)
    out(f"\nAccuracy against your {len(keys)} reviewed messages (F1: 1 = perfect):")
    out(df.round(2).to_string())
    return df


def _prf(y, p) -> dict:
    tp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and np.isfinite(prec) and np.isfinite(rec) else (0.0 if tp == 0 and (fp or fn) else np.nan)
    return {"precision": prec, "recall": rec, "F1": f1}
