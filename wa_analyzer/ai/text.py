"""Text units for the AI models: turns, conversation transcripts and topic documents."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TR_CHARS = re.compile(r"[ışğçöüİŞĞÇÖÜ]")
TR_WORDS = set("ve bir bu ne çok cok ben sen biz mi mı mu mü de da ama yok var değil degil gibi için icin daha "
               "şey sey ya yani bi tamam tmm naber nasılsın nasilsin evet hayır hayir canım canim aşkım askim "
               "napıyon napiyon geliyorum seni beni bana sana olur oldu bugün bugun yarın yarin".split())
MEDIA_TOKENS = {"image": "[photo]", "video": "[video]", "audio": "[voice message]", "sticker": "[sticker]",
                "gif": "[gif]", "document": "[document]", "contact": "[contact]", "media": "[media]",
                "location": "[location]", "poll": "[poll]", "view_once": "[view-once media]", "deleted": "[deleted]"}


def is_turkish(text: str) -> bool:
    if TR_CHARS.search(text):
        return True
    toks = re.findall(r"\w+", text.lower())
    return bool(toks) and sum(t in TR_WORDS for t in toks) / len(toks) >= 0.2


def turn_texts(m: pd.DataFrame, max_chars: int = 500) -> pd.DataFrame:
    """One row per turn with the joined text of its text messages."""
    t = m[m["kind"] == "text"].groupby("turn_id").agg(
        author=("author", "first"), start=("timestamp", "first"), conv_id=("conv_id", "first"),
        text=("text", lambda s: " ".join(x.strip() for x in s)[:max_chars]), n=("text", "size"))
    t = t[t["text"].str.strip().str.len() > 0].reset_index()
    t["month"] = t["start"].dt.to_period("M").dt.to_timestamp()
    return t


def transcript(msgs: pd.DataFrame, max_chars: int = 6000, names: dict | None = None) -> str:
    """'HH:MM Name: text' lines, cut to max_chars (keeps the beginning and the end)."""
    lines = []
    for ts, a, txt, kind in zip(msgs["timestamp"], msgs["author"], msgs["text"], msgs["kind"]):
        body = txt.strip() if kind == "text" else MEDIA_TOKENS.get(kind, "[media]")
        lines.append(f"{ts:%H:%M} {names.get(a, a) if names else a}: {body}")
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full
    head = full[: int(max_chars * 0.7)]
    tail = full[-int(max_chars * 0.3):]
    return head.rsplit("\n", 1)[0] + "\n[... part of the conversation left out ...]\n" + tail.split("\n", 1)[-1]


def topic_documents(m: pd.DataFrame, chunk_msgs: int = 30, min_words: int = 8, max_chars: int = 1500) -> pd.DataFrame:
    """Conversations (long ones split into chunks) as documents for topic modelling."""
    tm = m[m["kind"] == "text"]
    rows = []
    for cid, g in tm.groupby("conv_id"):
        for k in range(0, len(g), chunk_msgs):
            part = g.iloc[k:k + chunk_msgs]
            text = " ".join(part["text"].str.strip())
            if part["words"].sum() < min_words:
                continue
            share = part.groupby("author")["words"].sum()
            rows.append({"doc_id": len(rows), "conv_id": cid, "start": part["timestamp"].iloc[0],
                         "text": text[:max_chars], "n_msgs": len(part),
                         "main_author": share.idxmax(), "authors": dict(share / share.sum())})
    d = pd.DataFrame(rows)
    if not d.empty:
        d["month"] = d["start"].dt.to_period("M").dt.to_timestamp()
    return d


def even_sample(df: pd.DataFrame, n: int | None, by: str = "month", seed: int = 0) -> pd.DataFrame:
    """Up to n rows, spread evenly over months."""
    if n is None or len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(by).groups.values())
    per = max(1, n // len(groups))
    idx = []
    for g in groups:
        g = list(g)
        idx += list(rng.choice(g, size=min(per, len(g)), replace=False))
    if len(idx) < n:
        rest = list(set(df.index) - set(idx))
        idx += list(rng.choice(rest, size=min(n - len(idx), len(rest)), replace=False))
    return df.loc[sorted(idx[:n])]
