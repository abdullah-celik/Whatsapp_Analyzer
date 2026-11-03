"""'Prompts': every turn that could get a reply, with when (and whether) it got one.

Used by the survival model and the reply-prediction model.

For each turn (one person's consecutive messages) we look for the next message written by anyone else.
  wait_h   hours from the end of the turn until that message
  replied  1 if it came within `cap_hours`, else 0 (the wait is then "censored" at cap_hours)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_prompts(m: pd.DataFrame, turns: pd.DataFrame, cats: pd.DataFrame | None = None,
                  cap_hours: float = 48.0) -> pd.DataFrame:
    t = turns.reset_index(drop=True)
    n = len(t)
    authors = t["author"].to_numpy()
    starts = t["start"].to_numpy()
    ends = t["end"].to_numpy()

    # next turn by a different author (scan backwards)
    # invariant while scanning: nearest[0] = the turn right after i; nearest[1] = the closest following
    # turn whose author differs from nearest[0]'s author
    nxt = np.full(n, -1)
    nearest = [-1, -1]
    nearest_auth = [None, None]
    for i in range(n - 1, -1, -1):
        a = authors[i]
        if nearest_auth[0] is not None and nearest_auth[0] != a:
            nxt[i] = nearest[0]
        elif nearest_auth[1] is not None and nearest_auth[1] != a:
            nxt[i] = nearest[1]
        # insert turn i as the nearest following turn
        if nearest_auth[0] == a:
            nearest[0] = i
        else:
            nearest[1], nearest_auth[1] = nearest[0], nearest_auth[0]
            nearest[0], nearest_auth[0] = i, a

    has_next = nxt >= 0
    wait = np.full(n, np.inf)
    wait[has_next] = (starts[nxt[has_next]] - ends[has_next]) / np.timedelta64(1, "h")
    replied = wait <= cap_hours
    # the very last turns of the export cannot be judged fully: censor at the end of the data
    data_end = m["timestamp"].max()
    until_end = (data_end - t["end"]).dt.total_seconds().to_numpy() / 3600
    observed_limit = np.minimum(cap_hours, until_end)
    replied = replied & (wait <= observed_limit)
    wait_obs = np.where(replied, wait, observed_limit)

    p = pd.DataFrame({
        "turn_id": t["turn_id"],
        "conv_id": t["conv_id"],
        "author": authors,
        "end": t["end"],
        "replier": np.where(replied, authors[np.where(has_next, nxt, 0)], None) if n else [],
        "wait_h": np.clip(wait_obs, 1 / 60, None),
        "replied": replied.astype(int),
        "n_msgs": t["n_msgs"],
        "words": t["words"],
        "has_question": t["has_question"].astype(int),
        "has_media": t.get("has_media", pd.Series(False, index=t.index)).astype(int),
    })
    p["hour"] = p["end"].dt.hour
    p["weekday"] = p["end"].dt.dayofweek
    p["weekend"] = (p["weekday"] >= 5).astype(int)
    p["night"] = ((p["hour"] >= 23) | (p["hour"] < 7)).astype(int)
    p["month"] = p["end"].dt.to_period("M").dt.to_timestamp()
    p["first_in_conv"] = (t["conv_id"] != t["conv_id"].shift()).astype(int).to_numpy()
    p["turn_index"] = t.groupby("conv_id").cumcount().to_numpy()
    p["gap_before_h"] = ((t["start"] - t["end"].shift()).dt.total_seconds() / 3600).fillna(0).to_numpy()
    p = p[np.isfinite(p["wait_h"]) & (p["wait_h"] > 0)]

    if cats is not None:
        c = cats.groupby(m["turn_id"]).max()
        for col in c.columns:
            p["cat_" + col] = p["turn_id"].map(c[col]).fillna(0).astype(int)
    return p.reset_index(drop=True)
