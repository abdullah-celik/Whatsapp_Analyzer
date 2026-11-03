"""Multivariate Hawkes process: who triggers whom.

Each person's turns (consecutive messages) are events in time. A Hawkes process says the rate at which
person i writes is

    rate_i(t) = mu_i * g(hour of t)  +  sum over earlier turns m:  alpha[i, who(m)] * beta * exp(-beta (t - t_m))

  * mu_i * g(...)  "spontaneous" writing, with a daily rhythm g (people don't write at 4 am)
  * alpha[i, j]     branching ratio: how many turns of i are triggered, on average, by ONE turn of j
                    alpha[B, A] = 0.8 -> each time A writes, B answers with 0.8 turns on average
                    alpha[A, A]       -> A writing again on their own after writing (follow-ups)
  * beta            how fast the triggering effect fades (reported as a half-life)

Fitted by maximum likelihood (exact likelihood, exponential kernel, shared decay chosen by search).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar


@dataclass
class HawkesFit:
    people: list[str]
    mu_per_day: np.ndarray           # (D,) average spontaneous turns per day
    alpha: np.ndarray                # (D, D) branching ratios, alpha[i, j]: turns of i triggered by one turn of j
    half_life_min: float
    loglik: float
    loglik_poisson: float
    n_events: np.ndarray             # (D,)
    triggered_share: np.ndarray      # (D,) share of each person's turns explained by triggering

    @property
    def lr_stat(self) -> float:
        return 2 * (self.loglik - self.loglik_poisson)

    def matrix(self) -> pd.DataFrame:
        return pd.DataFrame(self.alpha, index=self.people, columns=self.people)


# ------------------------------------------------------------------ core

def _decayed_sums(t: np.ndarray, u: np.ndarray, D: int, beta: float) -> np.ndarray:
    """S[k, j] = sum over earlier events m (m < k) of type j of exp(-beta (t_k - t_m)). Vectorised in blocks."""
    N = len(t)
    S = np.zeros((N, D))
    onehot = np.zeros((N, D))
    onehot[np.arange(N), u] = 1.0
    carry = np.zeros(D)
    span = 400.0 / beta
    start = 0
    t_ref = t[0] if N else 0.0
    while start < N:
        end = np.searchsorted(t, t_ref + span, side="right")
        end = max(end, start + 1)
        tb = t[start:end] - t_ref
        contrib = onehot[start:end] * np.exp(beta * tb)[:, None]
        prefix = np.cumsum(contrib, axis=0) - contrib          # exclusive prefix sums
        S[start:end] = np.exp(-beta * tb)[:, None] * (carry + prefix)
        total = carry + contrib.sum(axis=0)
        if end < N:
            t_next = t[end]
            carry = np.exp(-beta * (t_next - t_ref)) * total
            t_ref = t_next
        start = end
    return S


def _diurnal(t_hours_abs: np.ndarray, t0_hour_of_day: float) -> tuple[np.ndarray, callable]:
    """Hour-of-day profile g (mean 1) estimated from the events themselves."""
    hod = np.floor((t_hours_abs + t0_hour_of_day) % 24).astype(int)
    counts = np.bincount(hod, minlength=24).astype(float)
    smooth = (np.roll(counts, 1) + 2 * counts + np.roll(counts, -1)) / 4 + 0.5
    g = smooth / smooth.mean()
    g = np.maximum(g, 0.05)

    def integral(T: float) -> float:
        full_days, rest = divmod(T, 24.0)
        val = full_days * g.sum()
        h = t0_hour_of_day
        r = rest
        while r > 1e-12:
            step = min(r, 1 - (h % 1) if h % 1 else 1.0)
            val += g[int(h) % 24] * step
            h = (h + step) % 24
            r -= step
        return val
    return g, integral


def _fit_fixed_beta(t, u, D, T, beta, g_ev, G_int, init=None):
    S = _decayed_sums(t, u, D, beta)
    E = np.array([np.sum(1 - np.exp(-beta * (T - t[u == j]))) for j in range(D)])
    idx = [np.flatnonzero(u == i) for i in range(D)]
    params0 = init if init is not None else np.concatenate([np.full(D, 0.5 * len(t) / (D * G_int)), np.full(D * D, 0.1)])

    def nll(theta):
        mu = theta[:D]
        A = theta[D:].reshape(D, D)
        val, grad_mu, grad_A = 0.0, np.zeros(D), np.zeros((D, D))
        for i in range(D):
            k = idx[i]
            lam = mu[i] * g_ev[k] + beta * S[k] @ A[i]
            lam = np.maximum(lam, 1e-300)
            val += np.log(lam).sum() - mu[i] * G_int - A[i] @ E
            inv = 1 / lam
            grad_mu[i] = (g_ev[k] * inv).sum() - G_int
            grad_A[i] = beta * (S[k] * inv[:, None]).sum(axis=0) - E
        return -val, -np.concatenate([grad_mu, grad_A.ravel()])

    bounds = [(1e-9, None)] * D + [(0.0, 50.0)] * (D * D)
    res = minimize(nll, params0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-10})
    return -res.fun, res.x


def fit(times: pd.Series, who: pd.Series, people: list[str] | None = None, diurnal: bool = True,
        half_life_bounds_min: tuple[float, float] = (0.5, 24 * 60)) -> HawkesFit:
    """times: event timestamps; who: person per event."""
    order = np.argsort(times.to_numpy())
    ts = pd.to_datetime(times.to_numpy()[order])
    names = who.to_numpy()[order]
    people = people or list(pd.Series(names).value_counts().index)
    pidx = {p: i for i, p in enumerate(people)}
    keep = np.array([n in pidx for n in names])
    ts, names = ts[keep], names[keep]
    D = len(people)
    t0 = ts[0]
    t = (ts - t0).total_seconds().to_numpy() / 3600.0
    # tiny jitter keeps identical timestamps (minute resolution) in a strict order
    t = t + np.arange(len(t)) * 1e-6
    u = np.array([pidx[n] for n in names])
    T = t[-1] + 1e-3
    if diurnal:
        h0 = t0.hour + t0.minute / 60
        g, integ = _diurnal(t, h0)
        g_ev = g[np.floor((t + h0) % 24).astype(int)]
        G_int = integ(T)
    else:
        g_ev, G_int = np.ones(len(t)), T
    n_ev = np.bincount(u, minlength=D)

    def neg_ll_logbeta(lb):
        ll, _ = _fit_fixed_beta(t, u, D, T, np.exp(lb), g_ev, G_int)
        return -ll

    # beta = ln2 / half-life; search over half-lives on a log grid, then refine
    hl_grid = np.geomspace(half_life_bounds_min[0], half_life_bounds_min[1], 9) / 60.0
    lbs = np.log(np.log(2) / hl_grid)
    vals = [neg_ll_logbeta(lb) for lb in lbs]
    j = int(np.argmin(vals))
    lo, hi = lbs[max(j - 1, 0)], lbs[min(j + 1, len(lbs) - 1)]
    if lo > hi:
        lo, hi = hi, lo
    if hi > lo:
        r = minimize_scalar(neg_ll_logbeta, bounds=(lo, hi), method="bounded", options={"xatol": 0.02})
        best_lb = r.x if r.fun <= vals[j] else lbs[j]
    else:
        best_lb = lbs[j]
    beta = float(np.exp(best_lb))
    ll, theta = _fit_fixed_beta(t, u, D, T, beta, g_ev, G_int)
    mu, A = theta[:D], theta[D:].reshape(D, D)

    # Poisson (no triggering) baseline with the same daily rhythm
    mu0 = n_ev / G_int
    ll0 = float(sum(np.log(np.maximum(mu0[u] * g_ev, 1e-300))) - n_ev.sum())
    triggered = (A @ n_ev) / np.maximum(n_ev, 1)
    return HawkesFit(people=people, mu_per_day=mu * 24, alpha=A, half_life_min=float(np.log(2) / beta * 60),
                     loglik=float(ll), loglik_poisson=ll0, n_events=n_ev,
                     triggered_share=np.clip(triggered, 0, 1))


def fit_turns(turns: pd.DataFrame, people: list[str], max_people: int = 8, **kw) -> HawkesFit:
    """Fit on turn starts. In big groups, everyone outside the top `max_people` is merged into 'others'."""
    who = turns["author"].where(turns["author"].isin(people[:max_people]), "others")
    ppl = [p for p in people[:max_people]] + (["others"] if (who == "others").any() else [])
    return fit(turns["start"], who, ppl, **kw)


def fit_by_period(turns: pd.DataFrame, people: list[str], freq: str = "QS", min_events: int = 150,
                  max_people: int = 4) -> pd.DataFrame:
    """Branching ratios per period (default: quarter). Long table: period, from, to, alpha."""
    rows = []
    for period, d in turns.groupby(turns["start"].dt.to_period(freq[0]).dt.start_time):
        if len(d) < min_events or d["author"].nunique() < 2:
            continue
        try:
            f = fit_turns(d, people, max_people=max_people)
        except Exception:
            continue
        for i, a in enumerate(f.people):
            for j, b in enumerate(f.people):
                if i != j:
                    rows.append({"period": period, "responder": a, "trigger": b, "alpha": f.alpha[i, j]})
    return pd.DataFrame(rows)


def describe(f: HawkesFit) -> list[str]:
    out = []
    D = len(f.people)
    for i, a in enumerate(f.people):
        for j, b in enumerate(f.people):
            if i != j and (D <= 3 or f.alpha[i, j] >= 0.15):
                out.append(f"Each turn by <b>{b}</b> triggers on average <b>{f.alpha[i, j]:.2f}</b> turns by {a}.")
    for i, a in enumerate(f.people):
        out.append(f"{a}: {f.triggered_share[i] * 100:.0f}% of turns are reactions, "
                   f"{f.mu_per_day[i]:.2f} spontaneous turns per day, "
                   f"self-follow-up ratio {f.alpha[i, i]:.2f}.")
    out.append(f"The triggering effect halves every {f.half_life_min:.0f} minutes. "
               f"Model fit vs. 'no triggering': likelihood-ratio statistic {f.lr_stat:,.0f} "
               f"({'very strong' if f.lr_stat > 50 else 'weak'} evidence of triggering).")
    return out


# ------------------------------------------------------------------ simulation (for tests)

def simulate(mu: np.ndarray, alpha: np.ndarray, beta: float, T: float, seed: int = 0):
    """Ogata thinning. Times in hours. Returns (times, types)."""
    rng = np.random.default_rng(seed)
    D = len(mu)
    t, times, types = 0.0, [], []
    state = np.zeros(D)        # decayed sum per source type at current time
    while True:
        lam_bar = mu.sum() + beta * (alpha @ state).sum()
        w = rng.exponential(1 / lam_bar)
        state = state * np.exp(-beta * w)
        t += w
        if t > T:
            break
        lam = mu + beta * alpha @ state
        if rng.uniform() * lam_bar <= lam.sum():
            k = rng.choice(D, p=lam / lam.sum())
            times.append(t)
            types.append(k)
            state[k] += 1
    return np.array(times), np.array(types)
