"""Checks for the statistical models: correctness on known/simulated data, and that everything runs."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wa_analyzer import analysis as A
from wa_analyzer import interest as I
from wa_analyzer.loading import load_filtered
from wa_analyzer.parser import parse_text
from wa_analyzer.stats import aggregates as G
from wa_analyzer.stats import changepoints as CP
from wa_analyzer.stats import hawkes as H
from wa_analyzer.stats import prompts as P
from wa_analyzer.stats import style as ST
from wa_analyzer.stats import survival as SV

FAKE = Path(__file__).parent / "fake_chats"


@pytest.fixture(scope="module")
def pair():
    if not (FAKE / "truth.json").exists():
        from make_fake_chats import main
        main(FAKE)
    s = A.Settings()
    m = A.prepare(load_filtered(FAKE / "pair_android_24h_dmy.txt"), s)
    t = A.build_turns(m)
    c = A.build_conversations(m, t, s)
    cats = I.WordClassifier().apply(m)
    people = list(m["author"].value_counts().index)
    return s, m, t, c, cats, people


def test_effort_from_aggregates_matches_direct(pair):
    s, m, t, c, cats, people = pair
    agg = G.build_aggregates(m, t, c, cats, s, people)
    shares, score = G.effort_shares(agg)
    assert score.sum() == pytest.approx(100, abs=0.2)
    # per-month point estimates are in range and the simulated swap of interest is visible
    mon = G.monthly_effort(agg, B=50)
    first = mon[mon["month"] == mon["month"].min()].set_index("person")["score"]
    last = mon[mon["month"] == mon["month"].max()].set_index("person")["score"]
    assert first[people[0]] > 60 or first[people[1]] > 60
    assert np.sign(first[people[0]] - 50) != np.sign(last[people[0]] - 50)
    assert ((mon["low"] <= mon["score"] + 1e-9) & (mon["score"] <= mon["high"] + 1e-9)).all()


def test_ci_table(pair):
    s, m, t, c, cats, people = pair
    agg = G.build_aggregates(m, t, c, cats, s, people)
    per, diff = G.ci_table(agg, B=300)
    assert (per["low"] <= per["high"]).all()
    assert diff is not None and "verdict" in diff.columns


def test_weighted_median():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    conv = np.array([0, 1, 2, 3])
    W = np.array([[1, 1, 1, 1], [0, 0, 0, 5], [5, 0, 0, 0]], dtype=float)
    assert list(G._weighted_median(v, conv, W)) == [2.0, 4.0, 1.0]


def test_hawkes_recovers_simulation():
    mu = np.array([0.25, 0.15])
    alpha = np.array([[0.05, 0.5], [0.65, 0.1]])
    beta = np.log(2) / (15 / 60)
    t, k = H.simulate(mu, alpha, beta, T=6000, seed=7)
    ts = pd.Series(pd.Timestamp("2024-01-01") + pd.to_timedelta(t, unit="h"))
    who = pd.Series(np.where(k == 0, "A", "B"))
    f = H.fit(ts, who, ["A", "B"], diurnal=False)
    assert np.abs(f.alpha - alpha).max() < 0.08
    assert f.half_life_min == pytest.approx(15, rel=0.25)
    assert f.lr_stat > 100


def test_decayed_sums_match_loop():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 200, 800))
    u = rng.integers(0, 3, 800)
    for beta in (0.1, 5.0, 60.0):
        S = H._decayed_sums(t, u, 3, beta)
        ref = np.zeros_like(S)
        st = np.zeros(3)
        for i in range(len(t)):
            if i:
                st = st * np.exp(-beta * (t[i] - t[i - 1]))
            ref[i] = st
            st = st.copy()
            st[u[i]] += 1
        assert np.allclose(S, ref, atol=1e-9)


def test_changepoint_finds_the_swap(pair):
    s, m, t, c, cats, people = pair
    agg = G.build_aggregates(m, t, c, cats, s, people)
    w = CP.weekly_series(m, t, agg, people)
    dates, table = CP.detect(w)
    assert dates, "the simulated swap of interest should be detected"
    # simulation swaps interest around the middle of the 6 months (early April)
    assert any(pd.Timestamp("2025-02-15") <= d <= pd.Timestamp("2025-05-15") for d in dates)


def test_prompts_next_reply_and_censoring():
    raw = ("01/03/2025, 10:00 - A: hi\n01/03/2025, 10:10 - B: hey\n01/03/2025, 10:11 - A: how are you\n"
           "03/03/2025, 12:00 - A: hello??\n03/03/2025, 12:30 - B: sorry\n")
    s = A.Settings()
    m = A.prepare(parse_text(raw), s)
    t = A.build_turns(m)
    p = P.build_prompts(m, t, cap_hours=24)
    # A's 10:11 message: B only answers 2 days later -> censored at 24 h
    row = p[p["end"] == pd.Timestamp("2025-03-01 10:11")].iloc[0]
    assert row["replied"] == 0 and row["wait_h"] == pytest.approx(24)
    first = p.iloc[0]
    assert first["replied"] == 1 and first["wait_h"] == pytest.approx(10 / 60)
    assert p.iloc[-1]["replied"] == 0  # last turn: end of data


def test_survival_runs(pair):
    s, m, t, c, cats, people = pair
    pr = P.build_prompts(m, t, cats)
    curves, summ = SV.kaplan_meier(pr, people)
    assert len(summ) == 2 and curves["waiting_pct"].between(0, 100).all()
    tab, conc, n = SV.cox_model(pr, people)
    assert tab is not None and 0.5 < conc < 1


def test_mirroring_detects_copying():
    rng = np.random.default_rng(1)
    lines, t0 = [], pd.Timestamp("2025-01-01 10:00")
    for i in range(600):
        a_emoji = rng.random() < 0.5
        b_emoji = a_emoji if rng.random() < 0.8 else not a_emoji   # B copies A's emoji use
        ts = t0 + pd.Timedelta(minutes=10 * i)
        lines.append(f"{ts:%d/%m/%Y, %H:%M} - A: message {i}{' 😊' if a_emoji else ''}")
        lines.append(f"{ts + pd.Timedelta(minutes=2):%d/%m/%Y, %H:%M} - B: answer {i}{' 😂' if b_emoji else ''}")
    s = A.Settings(gap_hours=0.1)
    m = A.prepare(parse_text("\n".join(lines)), s)
    t = A.build_turns(m)
    feats = ST.message_features(m)
    res = ST.mirroring(m, t, feats, ["A", "B"])
    b = res["table"].query("person == 'B' and feature == 'Emoji'").iloc[0]
    assert b["mirroring"] > 40 and b["low"] > 0


def test_full_pipeline_pair_and_group():
    from wa_analyzer.pipeline import Options, analyze
    from wa_analyzer.report import render_html
    from wa_analyzer.sections import build_sections
    for f in ("pair_ios_12h_mdy", "group_android_24h_dmy"):
        r = analyze(load_filtered(FAKE / f"{f}.txt"), f, A.Settings(), Options(bootstrap=200))
        assert not r.errors, r.errors
        html = render_html(r, build_sections(r))
        assert "Turning points" in html and "Who triggers whom" in html
        if f.startswith("group"):
            assert r.network is not None and "network" in html
        assert r.predict is not None
