"""A local LLM (Ollama) reads conversations and rates them.

For each selected conversation the model returns, per person: warmth, engagement and interest in the other
(1-5), plus who drives the conversation, the tone, whether there is conflict, topics and a one-sentence summary.
Then it writes a short summary per month and an overall summary.

On CPU a 3B model needs roughly 15-60 s per conversation, so by default only a sample of conversations per
month is read (--per-month). Everything is cached: you can stop with Ctrl+C and continue later.
"""
from __future__ import annotations

import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import JsonlCache, text_key
from .ollama import Ollama
from .text import transcript

PROMPT_VERSION = "reader-v1"
TONES = ["affectionate", "friendly", "playful", "neutral", "practical", "tense", "conflict", "cold"]

SYSTEM = """You analyse one WhatsApp conversation between people who know each other.
Messages can be in Turkish, Italian, English or mixed, with slang, abbreviations and typos (e.g. "slm", "nbr",
"cnm", "seviom", "tmm"). Short messages are normal in texting: do not judge someone as cold only because their
messages are short.

Rate EACH participant from 1 to 5, judging only this conversation:
- warmth: affection, kindness and friendliness towards the other (1 cold/hostile, 3 neutral, 5 very warm)
- engagement: effort in the conversation: substantive replies, keeping it going, asking things (1 minimal, 5 very engaged)
- interest_in_other: curiosity and interest in the other person and their life (1 none, 5 strong)

Also give:
- drives_conversation: the name of the person who mainly drives it, or "equal"
- tone: one of affectionate, friendly, playful, neutral, practical, tense, conflict, cold
- conflict: true if there is an argument, tension or someone is upset
- topics: up to 3 short English topic labels
- summary: ONE neutral English sentence about what happens and how they interact (no direct quotes)
Answer only with JSON."""

SCHEMA = {
    "type": "object",
    "properties": {
        "people": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "warmth": {"type": "integer", "minimum": 1, "maximum": 5},
            "engagement": {"type": "integer", "minimum": 1, "maximum": 5},
            "interest_in_other": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["name", "warmth", "engagement", "interest_in_other"]}},
        "drives_conversation": {"type": "string"},
        "tone": {"type": "string", "enum": TONES},
        "conflict": {"type": "boolean"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["people", "drives_conversation", "tone", "conflict", "topics", "summary"],
}

MONTH_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}


def _match(name: str, people: list[str]) -> str | None:
    n = (name or "").strip().lower()
    if not n:
        return None
    best, score = None, 0.0
    for p in people:
        pl = p.lower()
        s = 1.0 if n == pl else 0.9 if (pl.startswith(n) or n.startswith(pl) or n in pl.split()) else \
            SequenceMatcher(None, n, pl).ratio()
        if s > score:
            best, score = p, s
    return best if score >= 0.6 else None


def select_conversations(m: pd.DataFrame, convs: pd.DataFrame, per_month: int | None, min_msgs: int = 6,
                         seed: int = 0) -> pd.DataFrame:
    c = convs[(convs["n_msgs"] >= min_msgs) & (convs["n_people"] >= 2)].copy()
    c["month"] = c["start"].dt.to_period("M").dt.to_timestamp()
    if per_month is None:
        return c
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in c.groupby("month"):
        if len(g) <= per_month:
            picks.append(g)
        else:
            picks.append(g.loc[rng.choice(g.index, size=per_month, replace=False)])
    return pd.concat(picks).sort_values("start") if picks else c.iloc[:0]


def run(m: pd.DataFrame, convs: pd.DataFrame, out_dir: str | Path, llm: Ollama, per_month: int | None = 4,
        min_msgs: int = 6, max_chars: int = 6000, monthly: bool = True, seed: int = 0, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm.check()
    people = list(m["author"].value_counts().index)
    sel = select_conversations(m, convs, per_month, min_msgs, seed)
    log(f"Reading {len(sel):,} conversations with {llm.model} "
        f"({'all' if per_month is None else f'up to {per_month} per month'}) ...")
    cache = JsonlCache(out_dir / "cache_reader.jsonl")
    rows, score_rows = [], []
    t0, done_new = time.time(), 0
    for i, (_, c) in enumerate(sel.iterrows(), 1):
        msgs = m[m["conv_id"] == c["conv_id"]]
        tx = transcript(msgs, max_chars=max_chars)
        parts = sorted(msgs["author"].unique(), key=lambda p: people.index(p))
        k = text_key(PROMPT_VERSION, llm.model, tx)
        rec = cache.get(k)
        if rec is None:
            user = f"Participants: {', '.join(parts)}\nDate: {c['start']:%A %d %B %Y}\n\nConversation:\n{tx}"
            try:
                t1 = time.time()
                out = llm.chat_json(SYSTEM, user, SCHEMA)
                rec = {"result": out, "seconds": time.time() - t1}
                cache.put(k, rec)
                done_new += 1
            except Exception as e:
                log(f"  ! conversation {c['start']:%d %b %Y} skipped: {e}")
                continue
            elapsed = time.time() - t0
            left = (len(sel) - i) * elapsed / max(done_new, 1)
            log(f"  {i}/{len(sel)}  {c['start']:%d %b %Y}  ({rec['seconds']:.0f}s, ~{left / 60:.0f} min left)")
        out = rec["result"]
        drives = out.get("drives_conversation", "")
        drives_m = "equal" if str(drives).strip().lower() in ("equal", "both", "eşit", "") else (_match(drives, parts) or "unclear")
        rows.append({"conv_id": c["conv_id"], "start": c["start"], "month": c["month"], "n_msgs": c["n_msgs"],
                     "tone": out.get("tone"), "conflict": bool(out.get("conflict")), "drives": drives_m,
                     "topics": "; ".join(map(str, out.get("topics", [])[:3])), "summary": out.get("summary", "")})
        used = set()
        for pr in out.get("people", []):
            who = _match(str(pr.get("name", "")), parts)
            if who is None or who in used:
                continue
            used.add(who)
            score_rows.append({"conv_id": c["conv_id"], "start": c["start"], "month": c["month"], "person": who,
                               "warmth": pr.get("warmth"), "engagement": pr.get("engagement"),
                               "interest": pr.get("interest_in_other")})
    conv_df = pd.DataFrame(rows)
    scores = pd.DataFrame(score_rows)
    if conv_df.empty:
        raise RuntimeError("No conversation could be read.")
    conv_df.to_csv(out_dir / "reader_conversations.csv", index=False)
    scores.to_csv(out_dir / "reader_scores.csv", index=False)
    log(f"Saved {len(conv_df)} conversation readings.")

    result = {"conversations": conv_df, "scores": scores}
    if monthly:
        result.update(summarise(conv_df, scores, out_dir, llm, people, log))
    (out_dir / "reader_meta.json").write_text(json.dumps({"model": llm.model, "device": llm.device,
                                                          "conversations": len(conv_df), "per_month": per_month,
                                                          "created": pd.Timestamp.now().isoformat()}, indent=1))
    return result


def summarise(conv_df, scores, out_dir: Path, llm: Ollama, people: list[str], log=print) -> dict:
    cache = JsonlCache(out_dir / "cache_reader.jsonl")
    month_rows = []
    for month, g in conv_df.groupby("month"):
        if len(g) < 2:
            continue
        sc = scores[scores["month"] == month].groupby("person")[["warmth", "engagement", "interest"]].mean().round(1)
        lines = [f"- {r.start:%d %b}: {r.summary} (tone: {r.tone}{', conflict' if r.conflict else ''})" for r in g.itertuples()]
        avg = "; ".join(f"{p}: warmth {v.warmth}, engagement {v.engagement}, interest {v.interest}" for p, v in sc.iterrows())
        user = (f"Month: {pd.Timestamp(month):%B %Y}\nAverage ratings (1-5): {avg}\nConversations:\n" + "\n".join(lines) +
                "\n\nIn 2-3 neutral English sentences, describe how the communication between them looked this month "
                "(closeness, who invests more, notable events or tensions). Do not invent details.")
        k = text_key(PROMPT_VERSION, "month", llm.model, user)
        rec = cache.get(k)
        if rec is None:
            try:
                rec = {"result": llm.chat_json("You summarise chat analyses. Answer in JSON.", user, MONTH_SCHEMA)}
                cache.put(k, rec)
                log(f"  month summary {pd.Timestamp(month):%b %Y} done")
            except Exception as e:
                log(f"  ! month {pd.Timestamp(month):%b %Y}: {e}")
                continue
        month_rows.append({"month": month, "summary": rec["result"].get("summary", "")})
    months = pd.DataFrame(month_rows)
    if not months.empty:
        months.to_csv(out_dir / "reader_months.csv", index=False)
    overall = ""
    if len(months) >= 2:
        user = ("Monthly summaries of a WhatsApp relationship:\n" +
                "\n".join(f"- {pd.Timestamp(r.month):%b %Y}: {r.summary}" for r in months.itertuples()) +
                "\n\nWrite 3-5 neutral English sentences on how the relationship and communication developed over "
                "the whole period, and how balanced the interest seems. Do not invent details.")
        k = text_key(PROMPT_VERSION, "overall", llm.model, user)
        rec = cache.get(k)
        if rec is None:
            try:
                rec = {"result": llm.chat_json("You summarise chat analyses. Answer in JSON.", user, MONTH_SCHEMA)}
                cache.put(k, rec)
            except Exception as e:
                log(f"  ! overall summary: {e}")
                rec = None
        if rec:
            overall = rec["result"].get("summary", "")
            (out_dir / "reader_overall.txt").write_text(overall, encoding="utf-8")
    return {"months": months, "overall": overall}
