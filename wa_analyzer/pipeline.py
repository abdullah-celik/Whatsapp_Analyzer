"""Run every analysis once and keep the results together (used by the HTML report and the dashboard)."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from . import analysis as A
from . import interest as I

PALETTE = ["#2a6fdb", "#e0663a", "#1d9a6c", "#a64ac9", "#d4a017", "#c2185b", "#00838f", "#6d4c41",
           "#5c6bc0", "#7cb342", "#8d6e63", "#78909c"]


@dataclass
class Options:
    stats: bool = True                 # run the statistics (CIs, survival, change points, Hawkes, ...)
    bootstrap: int = 1000              # bootstrap replicates for confidence intervals
    seed: int = 0
    survival_cap_hours: float = 48.0   # longer waits count as "no reply"
    predict_hours: float = 1.0         # reply-prediction target: answered within this many hours
    lexicon: list[str] | None = None   # custom word lists
    ai_dir: str | None = None          # folder with AI results (from `python -m wa_analyzer.ai ...`)
    classifier: str | None = None      # trained message classifier (from `python -m wa_analyzer.ai train`)
    device: str = "auto"               # for applying the trained classifier: auto | cpu | cuda
    max_people: int = 12


@dataclass
class Results:
    title: str
    settings: A.Settings
    options: Options
    df: pd.DataFrame
    m: pd.DataFrame = None
    turns: pd.DataFrame = None
    convs: pd.DataFrame = None
    summary: pd.DataFrame = None
    people: list[str] = field(default_factory=list)
    people_all: list[str] = field(default_factory=list)
    is_group: bool = False
    colors: dict = field(default_factory=dict)
    cats: pd.DataFrame = None
    labels: dict = field(default_factory=dict)
    rates: pd.DataFrame = None
    distinct: dict = None
    agg: Any = None
    shares: pd.DataFrame = None
    score: pd.Series = None
    monthly: pd.DataFrame = None
    ci_per: pd.DataFrame = None
    ci_diff: pd.DataFrame = None
    prompts: pd.DataFrame = None
    km_curves: pd.DataFrame = None
    km_summary: pd.DataFrame = None
    cox: tuple = None
    weekly: pd.DataFrame = None
    change_dates: list = None
    change_table: pd.DataFrame = None
    hawkes: Any = None
    hawkes_periods: pd.DataFrame = None
    style: dict = None
    mirror: dict = None
    predict: dict = None
    network: dict = None
    ai: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def analyze(df: pd.DataFrame, title: str, s: A.Settings | None = None, o: Options | None = None,
            progress: Callable[[str], None] | None = None) -> Results:
    s = s or A.Settings()
    o = o or Options()
    say = progress or (lambda msg: None)
    r = Results(title=title, settings=s, options=o, df=df)

    def step(name: str, fn: Callable[[], None]):
        say(name)
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # keep going: one failing analysis must not kill the report
            r.errors[name] = f"{type(e).__name__}: {e}"
            if progress:
                say(f"  ! {name} failed: {e}")
            r.errors[name + " (details)"] = traceback.format_exc(limit=3)
        r.timings[name] = time.time() - t0

    # ---------------------------------------------------------------- core
    say("Preparing messages")
    r.m = A.prepare(df, s)
    if r.m.empty:
        raise ValueError("The chat contains no messages from people.")
    r.turns = A.build_turns(r.m)
    r.convs = A.build_conversations(r.m, r.turns, s)
    r.summary = A.people_summary(r.m, r.turns, r.convs, s)
    r.people_all = list(r.summary.index)
    r.people = r.people_all[:o.max_people]
    r.is_group = len(r.people_all) > 2
    r.colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(r.people_all)}

    def words():
        clf = I.WordClassifier(o.lexicon)
        r.cats = clf.apply(r.m)
        r.labels = clf.labels
        r.rates = I.category_rates(r.m, r.cats).reindex(r.people)
        r.distinct = I.distinctive_words(r.m)
    step("Word classifier", words)

    from .stats import aggregates as G

    def effort():
        r.agg = G.build_aggregates(r.m, r.turns, r.convs, r.cats, s, r.people_all)
        r.shares, r.score = G.effort_shares(r.agg)
    step("Effort score", effort)

    if o.stats:
        def cis():
            r.ci_per, r.ci_diff = G.ci_table(r.agg, B=o.bootstrap, seed=o.seed)
            r.monthly = G.monthly_effort(r.agg, B=max(200, o.bootstrap // 3), seed=o.seed)
        step("Confidence intervals (bootstrap)", cis)

        from .stats import prompts as P
        from .stats import survival as SV

        def surv():
            r.prompts = P.build_prompts(r.m, r.turns, r.cats, cap_hours=o.survival_cap_hours)
            r.km_curves, r.km_summary = SV.kaplan_meier(r.prompts, r.people[:8], o.survival_cap_hours)
            r.cox = SV.cox_model(r.prompts, r.people)
        step("Reply-time survival model", surv)

        from .stats import changepoints as CP

        def changes():
            r.weekly = CP.weekly_series(r.m, r.turns, r.agg, r.people_all)
            r.change_dates, r.change_table = CP.detect(r.weekly)
        step("Change-point detection", changes)

        from .stats import hawkes as H

        def hawkes():
            r.hawkes = H.fit_turns(r.turns, r.people_all, max_people=8)
            r.hawkes_periods = H.fit_by_period(r.turns, r.people_all, max_people=2 if not r.is_group else 4)
        step("Hawkes model (who triggers whom)", hawkes)

        from .stats import style as ST

        def style():
            feats = ST.message_features(r.m, r.cats)
            r.style = ST.style_matching(r.m, r.people, feats)
            r.mirror = ST.mirroring(r.m, r.turns, feats, r.people)
        step("Style matching & mirroring", style)

        from .stats import predict as PR

        def predict():
            if r.prompts is not None:
                r.predict = PR.run(r.prompts, r.m, r.people_all, horizon_h=o.predict_hours, seed=o.seed)
        step("Reply prediction model", predict)

        if r.is_group:
            from .stats import network as N
            step("Group network", lambda: setattr(r, "network", N.analyse(r.turns, r.m, r.people_all)))
    else:
        r.monthly = G.monthly_effort(r.agg, B=50, seed=o.seed)

    if o.ai_dir and Path(o.ai_dir).exists():
        from .ai import results as AR
        step("Loading AI results", lambda: r.ai.update(AR.load(o.ai_dir, r.m)))
    clf_path = o.classifier or r.ai.get("classifier_path")
    if clf_path and not o.classifier:
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is None:
            say("  (trained classifier found, but the AI packages are not installed: skipped)")
            clf_path = None
    if clf_path:
        def trained():
            from .ai.classifier import apply_trained
            pred, labels = apply_trained(clf_path, r.m, device=o.device)
            r.ai["trained_classifier"] = {"predictions": pred, "labels": labels}
        step("Applying the trained message classifier", trained)
    return r
