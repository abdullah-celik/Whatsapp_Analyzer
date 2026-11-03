"""Change-point detection: find the weeks where the dynamic of the chat clearly shifted.

Weekly series (effort score, reply times, message share, affection, dry replies ...) are standardised
and searched jointly for shifts in their average level: exact optimal partitioning with a penalty per change
(the problem PELT solves; implemented here in numpy, no extra dependency). A change point means the level
before and after that week differs more than random week-to-week noise would explain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregates import ConvAggregates, point_estimates



def segment_l2(X: np.ndarray, pen: float, min_size: int = 4) -> list[int]:
    """Optimal partitioning of X (n, d) into segments with different means (L2 cost + penalty per change).

    Returns the start indices of the new segments (excluding 0). Exact dynamic programming, O(n^2),
    fine for weekly data (a few hundred points)."""
    n = len(X)
    S1 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    S2 = np.concatenate([[0.0], np.cumsum((X ** 2).sum(axis=1))])
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, dtype=int)
    for t in range(min_size, n + 1):
        s = np.arange(0, t - min_size + 1)
        s = s[(s == 0) | (s >= min_size)]
        s = s[np.isfinite(F[s])]
        if not len(s):
            continue
        L = (t - s)[:, None]
        seg = S1[t] - S1[s]
        cost = (S2[t] - S2[s]) - (seg ** 2 / L).sum(axis=1)
        tot = F[s] + cost + pen
        k = int(np.argmin(tot))
        F[t], last[t] = tot[k], s[k]
    cps, t = [], n
    while t > 0:
        t = last[t]
        if t > 0:
            cps.append(int(t))
    return sorted(cps)


def weekly_series(m: pd.DataFrame, turns: pd.DataFrame, agg: ConvAggregates, people: list[str],
                  min_msgs: int = 15) -> pd.DataFrame:
    """One row per week, one column per measure (NaN where the week has too little data)."""
    week = m["timestamp"].dt.to_period("W-SUN").dt.start_time
    weeks = pd.date_range(week.min(), week.max(), freq="W-MON")
    out = pd.DataFrame(index=weeks)
    out.index.name = "week"
    out["Messages per week"] = week.value_counts().reindex(weeks, fill_value=0)
    focus = people[:2] if len(people) == 2 else people[:5]

    conv_week = pd.Series(pd.to_datetime(agg.conv_start)).dt.to_period("W-SUN").dt.start_time.to_numpy()
    eff = {p: [] for p in focus}
    share = {p: [] for p in focus}
    aff = {p: [] for p in focus}
    dry = {p: [] for p in focus}
    for w in weeks:
        mask = (conv_week == np.datetime64(w)).astype(float)
        n = agg.counts["msgs"][mask > 0].sum()
        ok = n >= min_msgs and (agg.counts["msgs"][mask > 0].sum(axis=0) > 0).sum() >= 2
        pt = point_estimates(agg, mask) if ok else None
        for p in focus:
            i = people.index(p)
            eff[p].append(pt["effort_score"].iloc[i] if ok else np.nan)
            share[p].append(pt["message_share"].iloc[i] if ok else np.nan)
            aff[p].append(pt.get("affection_pct", pd.Series(np.nan, index=people)).iloc[i] if ok else np.nan)
            dry[p].append(pt.get("dry_reply_pct", pd.Series(np.nan, index=people)).iloc[i] if ok else np.nan)
    replies = turns[turns["prev_author"].notna()].copy()
    replies["week"] = replies["start"].dt.to_period("W-SUN").dt.start_time
    med = replies.groupby(["week", "author"])["response_min"].agg(["median", "size"])
    for p in focus:
        out[f"{p}: effort score"] = eff[p]
        if len(people) > 2 or p == focus[0]:
            out[f"{p}: share of messages (%)"] = share[p]
        out[f"{p}: affection (%)"] = aff[p]
        out[f"{p}: dry replies (%)"] = dry[p]
        if p in med.index.get_level_values(1):
            mp = med.xs(p, level=1)
            out[f"{p}: typical reply time (min)"] = mp["median"].where(mp["size"] >= 5).reindex(weeks)
    if len(people) == 2:  # effort of the second person is 100 - first: keep one
        out = out.drop(columns=[f"{people[1]}: effort score"])
    return out


def _prep(x: pd.Series, log: bool) -> np.ndarray | None:
    s = x.astype(float)
    if s.notna().sum() < 12:
        return None
    if log:
        s = np.log1p(s)
    s = s.interpolate(limit_direction="both")
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    return ((s - s.mean()) / sd).to_numpy()


def detect(weekly: pd.DataFrame, penalty: float | None = None, min_weeks: int = 4,
           max_changes: int = 6) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    """Joint change points over all series, plus a table of what changed at each one."""
    cols, mats = [], []
    for c in weekly.columns:
        v = _prep(weekly[c], log="reply time" in c or c == "Messages per week")
        if v is not None:
            cols.append(c)
            mats.append(v)
    n = len(weekly)
    if not mats or n < 3 * min_weeks:
        return [], pd.DataFrame()
    X = np.column_stack(mats)
    # BIC-style penalty: k series * log(n), scaled for weekly noise
    pen = penalty if penalty is not None else 2.5 * np.log(n) * X.shape[1] ** 0.5
    bkps = segment_l2(X, pen, min_weeks)
    while len(bkps) > max_changes:  # too many: raise the penalty until only the strongest remain
        pen *= 1.5
        bkps = segment_l2(X, pen, min_weeks)
    dates = [weekly.index[b] for b in bkps]

    rows = []
    edges = [0] + bkps + [n]
    for k, b in enumerate(bkps):
        before = weekly.iloc[edges[k]:b]
        after = weekly.iloc[b:edges[k + 2]]
        for j, c in enumerate(cols):
            jump = np.nanmean(X[b:edges[k + 2], j]) - np.nanmean(X[edges[k]:b, j])
            rows.append({"change week": weekly.index[b], "measure": c,
                         "before": before[c].mean(), "after": after[c].mean(), "size (sd)": jump})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["abs"] = table["size (sd)"].abs()
        table = (table.sort_values(["change week", "abs"], ascending=[True, False])
                 .groupby("change week").head(4).drop(columns="abs").reset_index(drop=True))
    return dates, table


def describe(table: pd.DataFrame) -> list[str]:
    """Plain-language sentences for each change point."""
    out = []
    for wk, g in table.groupby("change week"):
        parts = []
        for _, r in g.head(3).iterrows():
            arrow = "rose" if r["after"] > r["before"] else "fell"
            parts.append(f"{r['measure']} {arrow} from {r['before']:.1f} to {r['after']:.1f}")
        out.append(f"Around <b>{pd.Timestamp(wk):%d %b %Y}</b>: " + "; ".join(parts) + ".")
    return out
