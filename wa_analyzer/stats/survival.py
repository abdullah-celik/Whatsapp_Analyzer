"""Reply times as a survival problem.

"Survival" here = a message still waiting for an answer. Messages that never got a reply (within the
cap) are not thrown away; they are 'censored', i.e. we only know the wait was at least that long.

  * Kaplan-Meier curves: share of messages still unanswered after t minutes, per person.
  * Cox proportional-hazards regression: which factors make replies come faster or slower.
    Hazard ratio (HR) 1.5 = replies come 1.5x faster; HR 0.7 = 30% slower.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

LIFELINES_OK = True
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import median_survival_times
except Exception:  # pragma: no cover
    LIFELINES_OK = False


def _label_for(author: str, people: list[str]) -> str:
    if len(people) == 2:
        other = people[1] if author == people[0] else people[0]
        return f"{other} replying to {author}"
    return f"replies to {author}"


def kaplan_meier(prompts: pd.DataFrame, people: list[str], cap_hours: float = 48.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (curves, summary). curves: long table for plotting; summary: per person medians etc."""
    if not LIFELINES_OK:
        raise ImportError("pip install lifelines")
    grid = np.unique(np.concatenate([np.geomspace(1 / 60, cap_hours, 120), [5 / 60, 1, 6, 24]]))
    curves, rows = [], []
    for p in people:
        d = prompts[prompts["author"] == p]
        if len(d) < 10:
            continue
        kmf = KaplanMeierFitter(label=p).fit(d["wait_h"], d["replied"])
        sf = kmf.survival_function_at_times(grid).to_numpy()
        ci = kmf.confidence_interval_survival_function_
        lo = np.interp(grid, ci.index.to_numpy(), ci.iloc[:, 0].to_numpy())
        hi = np.interp(grid, ci.index.to_numpy(), ci.iloc[:, 1].to_numpy())
        label = _label_for(p, people)
        curves.append(pd.DataFrame({"hours": grid, "minutes": grid * 60, "waiting_pct": sf * 100,
                                    "low": lo * 100, "high": hi * 100, "who": label, "prompt_author": p}))
        med = kmf.median_survival_time_
        try:
            mci = median_survival_times(kmf.confidence_interval_)
            mlo, mhi = float(mci.iloc[0, 0]), float(mci.iloc[0, 1])
        except Exception:
            mlo = mhi = np.nan

        def answered_by(h):
            return (1 - float(kmf.survival_function_at_times([h]).iloc[0])) * 100

        rows.append({
            "who": label, "messages waiting": len(d),
            "typical wait (median)": med * 60, "median low": mlo * 60, "median high": mhi * 60,
            "answered < 5 min (%)": answered_by(5 / 60), "answered < 1 h (%)": answered_by(1),
            "answered < 6 h (%)": answered_by(6), "answered < 24 h (%)": answered_by(24),
            f"never answered within {cap_hours:g} h (%)": (1 - d["replied"].mean()) * 100,
        })
    return (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()), pd.DataFrame(rows)


COVARIATE_LABELS = {
    "has_question": "Message contains a question",
    "has_media": "Message is / contains media",
    "log_words": "Longer message (per doubling of words)",
    "multi_msg": "Sent as several messages in a row",
    "night": "Sent at night (23:00–07:00)",
    "weekend": "Sent at the weekend",
    "first_in_conv": "Opens a new conversation",
    "trend_month": "Time trend (per month later)",
    "cat_affection": "Contains affection",
    "cat_care_questions": "Asks about the other",
    "cat_plans": "Makes plans",
    "cat_laughter": "Contains laughter",
    "cat_dry_reply": "Is a dry one-word reply",
    "cat_negative": "Negative / distant",
    "cat_compliments": "Contains a compliment",
}


def cox_model(prompts: pd.DataFrame, people: list[str], min_positive: int = 30, penalizer: float = 0.01):
    """Fit a Cox model. Returns (table, concordance, n) or (None, None, 0) if not enough data."""
    if not LIFELINES_OK:
        raise ImportError("pip install lifelines")
    d = prompts.copy()
    if len(d) < 100 or d["replied"].sum() < 50:
        return None, None, 0
    X = pd.DataFrame(index=d.index)
    # who wrote the message (baseline: the most active person)
    for p in people[1:8]:
        X[f"by::{p}"] = (d["author"] == p).astype(int)
    X["has_question"] = d["has_question"]
    X["has_media"] = d["has_media"]
    X["log_words"] = np.log2(d["words"].clip(lower=0) + 1)
    X["multi_msg"] = (d["n_msgs"] > 1).astype(int)
    X["night"] = d["night"]
    X["weekend"] = d["weekend"]
    X["first_in_conv"] = d["first_in_conv"]
    t0 = d["end"].min()
    X["trend_month"] = (d["end"] - t0).dt.total_seconds() / (30.44 * 86400)
    for c in [c for c in d.columns if c.startswith("cat_")]:
        if c in COVARIATE_LABELS:
            X[c] = d[c]
    # drop rare or constant columns
    keep = [c for c in X.columns if X[c].nunique() > 1 and
            (X[c].nunique() > 2 or min(X[c].sum(), len(X) - X[c].sum()) >= min_positive)]
    X = X[keep]
    X["duration"] = d["wait_h"]
    X["event"] = d["replied"]
    cph = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(X, duration_col="duration", event_col="event")
    s = cph.summary
    rows = []
    for cov, r in s.iterrows():
        if cov.startswith("by::"):
            label = f"Message written by {cov[4:]} (vs by {people[0]})"
        else:
            label = COVARIATE_LABELS.get(cov, cov)
        hr, lo, hi = r["exp(coef)"], r["exp(coef) lower 95%"], r["exp(coef) upper 95%"]
        rows.append({"factor": label, "hazard ratio": hr, "low": lo, "high": hi, "p": r["p"],
                     "meaning": _meaning(hr, lo, hi)})
    table = pd.DataFrame(rows).sort_values("hazard ratio", ascending=False).reset_index(drop=True)
    return table, float(cph.concordance_index_), len(X)


def _meaning(hr, lo, hi) -> str:
    if lo > 1:
        return f"replies come {hr:.2f}× faster"
    if hi < 1:
        return f"replies come {1 / hr:.2f}× slower"
    return "no clear effect"
