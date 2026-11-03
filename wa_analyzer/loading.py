"""Load a chat export and apply the common filters (dates, excluded people)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .parser import load_chat


def load_filtered(chat: str | Path, date_order: str | None = None, since: str | None = None,
                  until: str | None = None, exclude: list[str] | None = None) -> pd.DataFrame:
    df = load_chat(chat, date_order=date_order)
    order = df.attrs.get("date_order")
    if since:
        df = df[df["timestamp"] >= pd.Timestamp(since)]
    if until:
        df = df[df["timestamp"] < pd.Timestamp(until) + pd.Timedelta(days=1)]
    if exclude:
        df = df[~df["author"].isin(exclude)]
    df = df.reset_index(drop=True)
    df.attrs["date_order"] = order
    return df


def default_ai_dir(chat: str | Path) -> Path:
    """AI results are stored next to the chat: '<chat name>_ai/'."""
    chat = Path(chat)
    return chat.with_name(chat.stem + "_ai")
