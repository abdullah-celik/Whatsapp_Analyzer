"""Core behavioural analysis: activity, conversations, turns and response times.

Definitions (all configurable through Settings):
  * Conversation: messages separated by less than `gap_hours`. A longer silence starts a new one.
  * Turn: consecutive messages by the same person inside a conversation.
  * Reply: the first message of a turn that follows someone else's turn in the same conversation.
    Response time = time between the end of the previous turn and the start of the reply.
  * Initiation: the person who writes first in a conversation, unless that message is a
    *late reply* (the previous conversation ended with someone else's unanswered message or
    question, and the silence was shorter than `late_reply_hours`).
  * Follow-up without reply ("double text"): a person writes again after `followup_minutes`
    without anyone answering in between.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lexicon_engine import looks_like_question


@dataclass
class Settings:
    gap_hours: float = 6.0
    late_reply_hours: float = 36.0
    followup_minutes: float = 30.0
    quick_reply_minutes: float = 5.0


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
QUESTION_RE = re.compile(r"[?¿؟？]")
WORD_RE = re.compile(r"\w+", re.U)


def prepare(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Keep only people's messages and add helper columns."""
    m = df[df["kind"] != "system"].copy().reset_index(drop=True)
    is_text = m["kind"] == "text"
    m["words"] = np.where(is_text, m["text"].map(lambda t: len(WORD_RE.findall(t))), 0)
    m["chars"] = np.where(is_text, m["text"].str.len(), 0)
    m["is_question"] = is_text & m["text"].map(looks_like_question)
    m["is_media"] = ~m["kind"].isin(["text", "deleted"])
    m["hour"] = m["timestamp"].dt.hour
    m["weekday"] = pd.Categorical(m["timestamp"].dt.day_name(), categories=WEEKDAYS, ordered=True)
    m["month"] = m["timestamp"].dt.to_period("M").dt.to_timestamp()
    m["date"] = m["timestamp"].dt.normalize()

    gap = m["timestamp"].diff()
    new_conv = gap.isna() | (gap > pd.Timedelta(hours=s.gap_hours))
    m["conv_id"] = new_conv.cumsum() - 1
    new_turn = new_conv | (m["author"] != m["author"].shift())
    m["turn_id"] = new_turn.cumsum() - 1
    m["gap_before_min"] = gap.dt.total_seconds() / 60
    return m


def build_turns(m: pd.DataFrame) -> pd.DataFrame:
    t = m.groupby("turn_id").agg(
        author=("author", "first"),
        conv_id=("conv_id", "first"),
        start=("timestamp", "first"),
        end=("timestamp", "last"),
        n_msgs=("text", "size"),
        words=("words", "sum"),
        has_question=("is_question", "any"),
        has_media=("is_media", "any"),
    ).reset_index()
    prev_same_conv = t["conv_id"] == t["conv_id"].shift()
    t["prev_author"] = t["author"].shift().where(prev_same_conv)
    t["response_min"] = ((t["start"] - t["end"].shift()).dt.total_seconds() / 60).where(prev_same_conv)
    return t


def build_conversations(m: pd.DataFrame, turns: pd.DataFrame, s: Settings) -> pd.DataFrame:
    c = m.groupby("conv_id").agg(
        start=("timestamp", "first"),
        end=("timestamp", "last"),
        starter=("author", "first"),
        ender=("author", "last"),
        n_msgs=("text", "size"),
        n_people=("author", "nunique"),
    ).reset_index()
    last_turn = turns.groupby("conv_id").last()
    c["last_turn_question"] = c["conv_id"].map(last_turn["has_question"]).fillna(False).astype(bool)
    c["silence_before_h"] = (c["start"] - c["end"].shift()).dt.total_seconds() / 3600

    prev_ender = c["ender"].shift()
    prev_unanswered = (c["n_people"].shift() == 1) | c["last_turn_question"].shift(fill_value=False)
    c["late_reply"] = (
        (c["starter"] != prev_ender)
        & prev_ender.notna()
        & prev_unanswered.astype(bool)
        & (c["silence_before_h"] <= s.late_reply_hours)
    )
    c["late_reply_to"] = prev_ender.where(c["late_reply"])
    c["initiation"] = ~c["late_reply"]
    c["answered"] = c["n_people"] > 1
    return c


def people_summary(m: pd.DataFrame, turns: pd.DataFrame, convs: pd.DataFrame, s: Settings) -> pd.DataFrame:
    authors = m["author"].value_counts().index
    out = pd.DataFrame(index=authors)
    out.index.name = "person"
    g = m.groupby("author")
    out["messages"] = g.size()
    out["words"] = g["words"].sum()
    out["avg_words_per_text"] = m[m["kind"] == "text"].groupby("author")["words"].mean().round(1)
    out["media"] = g["is_media"].sum()
    out["deleted"] = m[m["kind"] == "deleted"].groupby("author").size()
    out["questions"] = g["is_question"].sum()
    out["active_days"] = g["date"].nunique()

    replies = turns[turns["prev_author"].notna()]
    rg = replies.groupby("author")["response_min"]
    out["replies"] = rg.size()
    out["median_response_min"] = rg.median().round(1)
    out["mean_response_min"] = rg.mean().round(1)
    out[f"replies_under_{int(s.quick_reply_minutes)}min_pct"] = (
        rg.apply(lambda x: (x <= s.quick_reply_minutes).mean() * 100).round(1)
    )

    out["conversations_started"] = convs[convs["initiation"]].groupby("starter").size()
    out["late_replies"] = convs[convs["late_reply"]].groupby("starter").size()
    out["conversations_ended"] = convs.groupby("ender").size()
    unanswered = convs[~convs["answered"]]
    out["left_without_answer"] = unanswered.groupby("starter").size()

    # follow-ups without reply: inside a turn, message after a long pause
    fu = m[(m["turn_id"] == m["turn_id"].shift()) & (m["gap_before_min"] >= s.followup_minutes)]
    fu_count = fu.groupby("turn_id")["author"].first().value_counts()
    # plus a new conversation started by the same person whose previous one got no answer
    chase = convs[(~convs["answered"].shift(fill_value=True)) & (convs["starter"] == convs["starter"].shift())]
    out["followups_without_reply"] = fu_count.add(chase.groupby("starter").size(), fill_value=0)
    return out.fillna(0).sort_values("messages", ascending=False)


def reply_matrix(turns: pd.DataFrame) -> pd.DataFrame:
    """rows = who replies, columns = to whom; counts of replies."""
    r = turns[turns["prev_author"].notna()]
    return pd.crosstab(r["author"], r["prev_author"])


def reply_time_matrix(turns: pd.DataFrame) -> pd.DataFrame:
    r = turns[turns["prev_author"].notna()]
    return r.pivot_table(index="author", columns="prev_author", values="response_min", aggfunc="median").round(1)


def activity_tables(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "by_hour": pd.crosstab(m["hour"], m["author"]).reindex(range(24), fill_value=0),
        "by_weekday": pd.crosstab(m["weekday"], m["author"]),
        "by_month": pd.crosstab(m["month"], m["author"]),
        "heatmap": pd.crosstab(m["weekday"], m["hour"]).reindex(columns=range(24), fill_value=0),
    }
