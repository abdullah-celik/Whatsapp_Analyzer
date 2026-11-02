"""Interest signals: multilingual word classifier, effort balance, trends and distinctive words."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from .analysis import Settings

from .lexicon_engine import WordClassifier, key, tokens  # noqa: E402  (re-exported)


def normalize(text: str) -> str:
    return " ".join(key(t) for t in tokens(text))


def category_rates(m: pd.DataFrame, cats: pd.DataFrame) -> pd.DataFrame:
    """% of each person's text messages that contain the category."""
    is_text = m["kind"] == "text"
    hits = cats[is_text].groupby(m.loc[is_text, "author"]).sum()
    n = is_text.groupby(m["author"]).sum()
    return (hits.div(n, axis=0) * 100).round(1)


def category_monthly(m: pd.DataFrame, cats: pd.DataFrame, min_msgs: int = 10) -> pd.DataFrame:
    """Long table: month, person, category, rate per 100 text messages."""
    is_text = m["kind"] == "text"
    d = cats[is_text].copy()
    d["author"] = m.loc[is_text, "author"]
    d["month"] = m.loc[is_text, "month"]
    g = d.groupby(["month", "author"])
    n = g.size()
    rates = g.sum().div(n, axis=0) * 100
    rates = rates[n >= min_msgs]
    return rates.reset_index().melt(id_vars=["month", "author"], var_name="category", value_name="rate")


# ---------------------------------------------------------------- effort / interest balance

def _metrics_for(m: pd.DataFrame, turns: pd.DataFrame, convs: pd.DataFrame, cats: pd.DataFrame,
                 people: list[str], s: Settings) -> pd.DataFrame:
    """Raw interest-related numbers per person for a slice of the chat."""
    out = pd.DataFrame(index=people, dtype=float)
    is_text = m["kind"] == "text"
    n_text = is_text.groupby(m["author"]).sum().reindex(people).fillna(0)
    c = cats[is_text].groupby(m.loc[is_text, "author"]).sum().reindex(people).fillna(0)
    replies = turns[turns["prev_author"].notna()]
    med = replies.groupby("author")["response_min"].median().reindex(people)
    quick = replies.groupby("author")["response_min"].apply(lambda x: (x <= s.quick_reply_minutes).mean()).reindex(people)

    out["Starts conversations"] = convs[convs["initiation"]].groupby("starter").size().reindex(people).fillna(0)
    chase_in = m[(m["turn_id"] == m["turn_id"].shift()) & (m["gap_before_min"] >= s.followup_minutes)]
    chase = chase_in.groupby("turn_id")["author"].first().value_counts().reindex(people).fillna(0)
    out["Writes again without reply"] = chase
    out["Questions asked"] = m["is_question"].groupby(m["author"]).sum().reindex(people).fillna(0)
    out["Words written"] = m["words"].groupby(m["author"]).sum().reindex(people).fillna(0)
    out["Reply speed"] = 1 / (med.fillna(med.max() if med.notna().any() else 1) + 1)
    out["Quick replies"] = quick.fillna(0)
    for cat in _LABEL:
        if cat in c:
            out[_LABEL[cat]] = c[cat]
    if "dry_reply" in c:
        out["Few dry replies"] = 1 / (c["dry_reply"] / n_text.replace(0, np.nan) * 100 + 1).fillna(1)
    if "negative" in c:
        out["Few negative words"] = 1 / (c["negative"] / n_text.replace(0, np.nan) * 100 + 1).fillna(1)
    return out


_LABEL = {
    "affection": "Affection words/emojis",
    "care_questions": "Asks about the other",
    "plans": "Makes plans",
    "compliments": "Compliments",
    "laughter": "Laughter",
    "good_morning_night": "Good morning / night",
    "enthusiasm": "Enthusiasm",
}


def effort_balance(m, turns, convs, cats, s: Settings, people: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Share of each signal per person (each column sums to 100 %), and an overall score.

    With 2 people, 50 % means balanced. With n people, 100/n % is an equal share.
    """
    people = people or list(m["author"].value_counts().index)
    raw = _metrics_for(m, turns, convs, cats, people, s)
    totals = raw.sum()
    shares = raw.div(totals.replace(0, np.nan), axis=1) * 100
    shares = shares.dropna(axis=1, how="all")
    score = shares.mean(axis=1).round(1)
    return shares.T.round(1), score


def effort_balance_monthly(m, turns, convs, cats, s: Settings, people: list[str], min_msgs: int = 30) -> pd.DataFrame:
    rows = []
    for month, mm in m.groupby("month"):
        if len(mm) < min_msgs or mm["author"].nunique() < 2:
            continue
        t = turns[turns["turn_id"].isin(mm["turn_id"].unique())]
        c = convs[convs["conv_id"].isin(mm["conv_id"].unique())]
        _, score = effort_balance(mm, t, c, cats.loc[mm.index], s, people)
        rows.append(score.rename(month))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- distinctive words

def distinctive_words(m: pd.DataFrame, top: int = 15, min_count: int = 3) -> dict[str, list[tuple[str, int, float]]]:
    """Words each person uses unusually often compared with everyone else
    (log-odds ratio with an informative Dirichlet prior, Monroe et al. 2008). Works in any language."""
    texts = m[m["kind"] == "text"]
    counts = {a: Counter() for a in texts["author"].unique()}
    shown: dict[str, Counter] = {}
    for a, t in zip(texts["author"], texts["text"]):
        for raw in tokens(t):
            k = key(raw)
            if len(k) > 1 and not k.isdigit():
                counts[a][k] += 1
                shown.setdefault(k, Counter())[raw] += 1
    total = Counter()
    for c in counts.values():
        total.update(c)
    a0 = sum(total.values()) or 1
    alpha_scale = 500 / a0  # prior strength
    result = {}
    for person, c in counts.items():
        rest = total - c
        n_i, n_j = sum(c.values()), sum(rest.values())
        if n_i == 0 or n_j == 0:
            result[person] = [(shown[w].most_common(1)[0][0], n, 0.0) for w, n in c.most_common(top)]
            continue
        a_tot = a0 * alpha_scale
        scored = []
        for w, y_i in c.items():
            if y_i < min_count:
                continue
            aw = total[w] * alpha_scale
            y_j = rest[w]
            d = math.log((y_i + aw) / (n_i + a_tot - y_i - aw)) - math.log((y_j + aw) / (n_j + a_tot - y_j - aw))
            var = 1 / (y_i + aw) + 1 / (y_j + aw)
            scored.append((w, y_i, d / math.sqrt(var)))
        scored.sort(key=lambda x: -x[2])
        result[person] = [(shown[w].most_common(1)[0][0], n, z) for w, n, z in scored[:top] if z > 1.0]
    return result
