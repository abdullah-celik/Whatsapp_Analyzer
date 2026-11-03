"""Per-conversation totals, effort score and bootstrap confidence intervals.

Every measure is rebuilt from per-conversation totals with a weight per conversation:
  * all weights = 1          -> the normal (point) estimate
  * weights = 1 for a month  -> that month's estimate
  * multinomial weights      -> one bootstrap replicate (conversations resampled with replacement)

Resampling whole conversations (a "cluster bootstrap") is important: messages inside a conversation
depend on each other, so resampling single messages would make the intervals far too narrow.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analysis import Settings

# effort-score signals: label -> how it is computed (see _components)
CATEGORY_SIGNALS = {
    "affection": "Affection words/emojis",
    "care_questions": "Asks about the other",
    "plans": "Makes plans",
    "compliments": "Compliments",
    "laughter": "Laughter",
    "good_morning_night": "Good morning / night",
    "enthusiasm": "Enthusiasm",
}


@dataclass
class ConvAggregates:
    people: list[str]
    conv_ids: np.ndarray                    # (C,)
    conv_month: np.ndarray                  # (C,) month timestamp of conversation start
    conv_start: np.ndarray                  # (C,) datetime64
    counts: dict[str, np.ndarray]           # feature -> (C, P) additive counts
    rt_person: np.ndarray                   # (R,) person index of each reply
    rt_conv: np.ndarray                     # (R,) conversation index of each reply
    rt_min: np.ndarray                      # (R,) reply time in minutes
    quick_minutes: float = 5.0
    categories: list[str] = field(default_factory=list)

    @property
    def n_conv(self) -> int:
        return len(self.conv_ids)


def build_aggregates(m: pd.DataFrame, turns: pd.DataFrame, convs: pd.DataFrame, cats: pd.DataFrame | None,
                     s: Settings, people: list[str]) -> ConvAggregates:
    pidx = {p: i for i, p in enumerate(people)}
    conv_ids = convs["conv_id"].to_numpy()
    cpos = {c: i for i, c in enumerate(conv_ids)}
    C, P = len(conv_ids), len(people)

    def mat(conv_series, person_series, values=None):
        out = np.zeros((C, P))
        ci = conv_series.map(cpos).to_numpy()
        pi = person_series.map(pidx)
        ok = pi.notna().to_numpy()
        vals = np.ones(len(ci)) if values is None else np.asarray(values, dtype=float)
        np.add.at(out, (ci[ok], pi[ok].astype(int).to_numpy()), vals[ok])
        return out

    is_text = (m["kind"] == "text").to_numpy()
    counts = {
        "msgs": mat(m["conv_id"], m["author"]),
        "text_msgs": mat(m["conv_id"], m["author"], is_text),
        "words": mat(m["conv_id"], m["author"], m["words"]),
        "questions": mat(m["conv_id"], m["author"], m["is_question"]),
        "media": mat(m["conv_id"], m["author"], m["is_media"]),
    }
    init = convs[convs["initiation"]]
    counts["starts"] = mat(init["conv_id"], init["starter"])
    late = convs[convs["late_reply"]]
    counts["late_replies"] = mat(late["conv_id"], late["starter"])
    counts["ends"] = mat(convs["conv_id"], convs["ender"])
    unans = convs[~convs["answered"]]
    counts["unanswered"] = mat(unans["conv_id"], unans["starter"])

    # follow-ups without reply: long pause inside a turn, or a new conversation started by the same
    # person whose previous conversation got no answer
    fu = m[(m["turn_id"] == m["turn_id"].shift()) & (m["gap_before_min"] >= s.followup_minutes)]
    fu = fu.groupby("turn_id").first()
    chase = convs[(~convs["answered"].shift(fill_value=True)) & (convs["starter"] == convs["starter"].shift())]
    counts["followups"] = mat(fu["conv_id"], fu["author"]) + mat(chase["conv_id"], chase["starter"])

    replies = turns[turns["prev_author"].notna()]
    counts["replies"] = mat(replies["conv_id"], replies["author"])
    counts["quick"] = mat(replies["conv_id"], replies["author"], replies["response_min"] <= s.quick_reply_minutes)

    categories = []
    if cats is not None:
        categories = list(cats.columns)
        for c in categories:
            counts["cat_" + c] = mat(m["conv_id"], m["author"], cats[c].to_numpy() * is_text)

    rp = replies["author"].map(pidx)
    ok = rp.notna()
    return ConvAggregates(
        people=people, conv_ids=conv_ids,
        conv_month=convs["start"].dt.to_period("M").dt.to_timestamp().to_numpy(),
        conv_start=convs["start"].to_numpy(),
        counts=counts,
        rt_person=rp[ok].astype(int).to_numpy(),
        rt_conv=replies.loc[ok, "conv_id"].map(cpos).to_numpy(),
        rt_min=replies.loc[ok, "response_min"].to_numpy(dtype=float),
        quick_minutes=s.quick_reply_minutes, categories=categories,
    )


# ------------------------------------------------------------------ weighted statistics

def _weighted_median(values_sorted: np.ndarray, conv_sorted: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted median for each row of W (B, C). values_sorted ascending."""
    B = W.shape[0]
    if len(values_sorted) == 0:
        return np.full(B, np.nan)
    out = np.empty(B)
    step = max(1, int(2e7 // max(1, len(values_sorted))))
    for s0 in range(0, B, step):
        w = W[s0:s0 + step][:, conv_sorted]
        cum = np.cumsum(w, axis=1)
        tot = cum[:, -1]
        idx = np.argmax(cum >= (tot / 2)[:, None], axis=1)
        med = values_sorted[idx]
        med[tot <= 0] = np.nan
        out[s0:s0 + step] = med
    return out


def compute_metrics(agg: ConvAggregates, W: np.ndarray) -> dict[str, np.ndarray]:
    """All per-person metrics for each weight vector. W: (B, C). Returns name -> (B, P)."""
    W = np.atleast_2d(np.asarray(W, dtype=float))
    T = {k: W @ v for k, v in agg.counts.items()}          # (B, P)
    B, P = W.shape[0], len(agg.people)
    med = np.full((B, P), np.nan)
    for p in range(P):
        sel = agg.rt_person == p
        order = np.argsort(agg.rt_min[sel])
        med[:, p] = _weighted_median(agg.rt_min[sel][order], agg.rt_conv[sel][order], W)

    with np.errstate(invalid="ignore", divide="ignore"):
        tot_msgs = T["msgs"].sum(axis=1, keepdims=True)
        tot_starts = T["starts"].sum(axis=1, keepdims=True)
        text = T["text_msgs"]
        out = {
            "messages": T["msgs"],
            "message_share": T["msgs"] / tot_msgs * 100,
            "words_per_message": T["words"] / T["msgs"],
            "median_reply_min": med,
            "quick_reply_pct": T["quick"] / T["replies"] * 100,
            "start_share": T["starts"] / tot_starts * 100,
            "questions_per_100": T["questions"] / T["msgs"] * 100,
            "followups": T["followups"],
            "unanswered": T["unanswered"],
        }
        for c in agg.categories:
            out[f"{c}_pct"] = T["cat_" + c] / text * 100
        comps = _components(agg, T, med)
        shares = {k: v / v.sum(axis=1, keepdims=True) * 100 for k, v in comps.items()}
        stack = np.stack(list(shares.values()), axis=0)       # (K, B, P)
        valid = np.isfinite(stack) & (np.stack([v.sum(axis=1, keepdims=True) > 0 for v in comps.values()]))
        stack = np.where(valid, stack, np.nan)
        out["effort_score"] = np.nanmean(stack, axis=0) if len(stack) else np.full((B, P), np.nan)
        for k, v in shares.items():
            out["share:" + k] = np.where(np.isfinite(v), v, np.nan)
    return out


def _components(agg, T, med) -> dict[str, np.ndarray]:
    comps = {
        "Starts conversations": T["starts"],
        "Writes again without reply": T["followups"],
        "Questions asked": T["questions"],
        "Words written": T["words"],
    }
    m = med.copy()
    rowmax = np.nanmax(np.where(np.isfinite(m), m, -np.inf), axis=1, keepdims=True)
    rowmax = np.where(np.isfinite(rowmax), rowmax, 1.0)
    m = np.where(np.isfinite(m), m, rowmax)
    comps["Reply speed"] = 1 / (m + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        comps["Quick replies"] = np.nan_to_num(T["quick"] / T["replies"])
        for c, label in CATEGORY_SIGNALS.items():
            if "cat_" + c in T:
                comps[label] = T["cat_" + c]
        text = np.where(T["text_msgs"] > 0, T["text_msgs"], np.nan)
        if "cat_dry_reply" in T:
            comps["Few dry replies"] = np.nan_to_num(1 / (T["cat_dry_reply"] / text * 100 + 1), nan=1.0)
        if "cat_negative" in T:
            comps["Few negative words"] = np.nan_to_num(1 / (T["cat_negative"] / text * 100 + 1), nan=1.0)
    return comps


# ------------------------------------------------------------------ public helpers

def point_estimates(agg: ConvAggregates, weights: np.ndarray | None = None) -> dict[str, pd.Series]:
    w = np.ones(agg.n_conv) if weights is None else weights
    res = compute_metrics(agg, w[None, :])
    return {k: pd.Series(v[0], index=agg.people) for k, v in res.items()}


def bootstrap(agg: ConvAggregates, B: int = 1000, seed: int = 0, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Cluster bootstrap over conversations. `mask` restricts to a subset (e.g. one month)."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask) if mask is not None else np.arange(agg.n_conv)
    W = np.zeros((B, agg.n_conv))
    if len(idx):
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=B)
    return compute_metrics(agg, W)


METRIC_LABELS = {
    "effort_score": ("Effort score (% share)", "{:.1f}"),
    "message_share": ("Share of messages (%)", "{:.1f}"),
    "start_share": ("Share of conversations started (%)", "{:.1f}"),
    "median_reply_min": ("Typical reply time (min)", "{:.1f}"),
    "quick_reply_pct": ("Replies within quick limit (%)", "{:.1f}"),
    "words_per_message": ("Words per message", "{:.2f}"),
    "questions_per_100": ("Questions per 100 messages", "{:.1f}"),
    "affection_pct": ("Affection (% of messages)", "{:.1f}"),
    "care_questions_pct": ("Asks about the other (%)", "{:.1f}"),
    "plans_pct": ("Plans (% of messages)", "{:.1f}"),
    "dry_reply_pct": ("Dry replies (% of messages)", "{:.1f}"),
    "negative_pct": ("Negative / distant (%)", "{:.1f}"),
}


def ci_table(agg: ConvAggregates, B: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Per-person estimates with confidence intervals, and (for 2 people) the difference A - B.

    Returns (per_person, difference). difference has: estimate, CI, P(A > B) and a verdict.
    """
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    pt = point_estimates(agg)
    bs = bootstrap(agg, B=B, seed=seed)
    rows, drows = [], []
    for key, (label, fmt) in METRIC_LABELS.items():
        if key not in pt:
            continue
        for i, p in enumerate(agg.people):
            v = bs[key][:, i]
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            rows.append({"measure": label, "person": p, "estimate": pt[key].iloc[i],
                         "low": np.percentile(v, lo), "high": np.percentile(v, hi)})
        if len(agg.people) == 2:
            d = bs[key][:, 0] - bs[key][:, 1]
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            est = pt[key].iloc[0] - pt[key].iloc[1]
            l, h = np.percentile(d, lo), np.percentile(d, hi)
            p_gt = float((d > 0).mean())
            if l > 0 or h < 0:
                verdict = "clear difference"
            elif 0.2 < p_gt < 0.8:
                verdict = "no real difference"
            else:
                verdict = "uncertain"
            drows.append({"measure": label, "difference": est, "low": l, "high": h,
                          f"P({agg.people[0]} higher)": p_gt, "verdict": verdict})
    per = pd.DataFrame(rows)
    diff = pd.DataFrame(drows).set_index("measure") if drows else None
    return per, diff


def monthly_effort(agg: ConvAggregates, B: int = 300, seed: int = 0, min_msgs: int = 30) -> pd.DataFrame:
    """Long table: month, person, score, low, high."""
    rows = []
    months = np.unique(agg.conv_month)
    for mo in months:
        mask = agg.conv_month == mo
        if agg.counts["msgs"][mask].sum() < min_msgs or (agg.counts["msgs"][mask].sum(axis=0) > 0).sum() < 2:
            continue
        pt = point_estimates(agg, mask.astype(float))["effort_score"]
        bs = bootstrap(agg, B=B, seed=seed, mask=mask)["effort_score"]
        for i, p in enumerate(agg.people):
            v = bs[:, i][np.isfinite(bs[:, i])]
            rows.append({"month": pd.Timestamp(mo), "person": p, "score": pt.iloc[i],
                         "low": np.percentile(v, 2.5) if len(v) else np.nan,
                         "high": np.percentile(v, 97.5) if len(v) else np.nan})
    return pd.DataFrame(rows)


def effort_shares(agg: ConvAggregates) -> tuple[pd.DataFrame, pd.Series]:
    """Signal-by-signal shares (rows = signals, columns = people) and the overall score."""
    pt = point_estimates(agg)
    shares = {k[len("share:"):]: v for k, v in pt.items() if k.startswith("share:")}
    df = pd.DataFrame(shares).T
    df = df.dropna(how="all")
    return df.round(1), pt["effort_score"].round(1)
