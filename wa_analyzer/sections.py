"""Turn analysis results into report sections (figures, tables, text).

The same sections are rendered by the HTML report (report.py) and the Streamlit dashboard.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import analysis as A
from . import interest as I
from .pipeline import Results


# ------------------------------------------------------------------ building blocks

@dataclass
class Fig:
    fig: go.Figure
    height: int = 360


@dataclass
class Table:
    df: pd.DataFrame
    fmt: str = "{:,.1f}"
    index: bool = True


@dataclass
class Html:
    html: str


@dataclass
class Grid:
    items: list


@dataclass
class Section:
    id: str
    tab: str
    title: str
    note: str = ""
    blocks: list = field(default_factory=list)


DATE_TICKS = [dict(dtickrange=[None, 86400000 * 27], value="%d %b %Y"),
              dict(dtickrange=[86400000 * 27, None], value="%b %Y")]


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    # date axes: show days when ticks are closer than a month, so labels never repeat
    fig.update_xaxes(tickformatstops=DATE_TICKS)
    fig.update_layout(height=height, margin=dict(l=40, r=20, t=50, b=40), template="plotly_white",
                      font=dict(family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=12),
                      legend=dict(orientation="h", y=-0.22))
    return fig


def fmt_minutes(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    if x < 1:
        return f"{x * 60:.0f} sec"
    if x < 90:
        return f"{x:.0f} min"
    if x < 48 * 60:
        return f"{x / 60:.1f} h"
    return f"{x / 1440:.1f} days"


def esc(s) -> str:
    return html.escape(str(s))


def _ci(est, lo, hi, digits=1) -> str:
    f = f"{{:.{digits}f}}"
    return f"{f.format(est)}  ({f.format(lo)}–{f.format(hi)})"


def _band(fig, x, lo, hi, color, name):
    xs = list(x) + list(x)[::-1]
    ys = list(hi) + list(lo)[::-1]
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=_rgba(color, 0.15), line=dict(width=0), mode="lines",
                             hoverinfo="skip", showlegend=False, name=name))


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ------------------------------------------------------------------ sections

def build_sections(r: Results) -> list[Section]:
    out = [key_findings(r), overview(r), activity(r), streaks(r), response_times(r)]
    out.append(survival(r))
    out.append(interest(r))
    out.append(statistics(r))
    out.append(change_points(r))
    out.append(hawkes(r))
    out.append(style(r))
    out.append(prediction(r))
    if r.is_group:
        out.append(network(r))
    out.append(words(r))
    out.append(typical_words(r))
    if r.ai:
        from .ai.sections import ai_sections
        out.extend(ai_sections(r))
    if r.errors:
        msgs = [f"<li><b>{esc(k)}</b>: {esc(v)}</li>" for k, v in r.errors.items() if not k.endswith("(details)")]
        out.append(Section("errors", "Overview", "Analyses that could not run",
                           "These parts were skipped; the rest of the report is fine.", [Html("<ul>" + "".join(msgs) + "</ul>")]))
    return [s for s in out if s is not None and s.blocks]


def key_findings(r: Results) -> Section:
    f = []
    summ, people = r.summary, r.people_all
    top = summ["messages"]
    f.append(f"<b>{esc(top.index[0])}</b> writes the most: {top.iloc[0]:,} messages ({top.iloc[0] / top.sum() * 100:.0f}% of all).")
    med = summ["median_response_min"].replace(0, np.nan).dropna()
    med = med[summ["replies"].reindex(med.index) >= 5]
    if len(med) >= 2 and med.min() < med.max():
        f.append(f"Fastest typical reply: <b>{esc(med.idxmin())}</b> ({fmt_minutes(med.min())}); "
                 f"slowest: <b>{esc(med.idxmax())}</b> ({fmt_minutes(med.max())}).")
    st = summ["conversations_started"]
    if st.sum() > 0:
        f.append(f"<b>{esc(st.idxmax())}</b> starts most conversations ({st.max() / st.sum() * 100:.0f}%).")
    if r.streaks is not None and r.streaks.longest is not None:
        lg = r.streaks.longest
        txt = (f"Longest streak: <b>{int(lg['days'])} days</b> in a row ({_period(lg['start'], lg['end'])}) "
               f"on which {_streak_rule(r.streaks, r.is_group)}")
        if r.streaks.current:
            txt += f"; the current streak is <b>{r.streaks.current} days</b>"
        f.append(txt + ".")
    if r.score is not None:
        fair = 100 / len(people)
        lead = r.score.idxmax()
        word = "puts the most effort into the chat" if r.is_group else "shows more interest / effort"
        txt = f"Overall, <b>{esc(lead)}</b> {word}: effort score {r.score.max():.0f}% (equal share: {fair:.0f}%)"
        if r.ci_diff is not None and "Effort score (% share)" in r.ci_diff.index:
            d = r.ci_diff.loc["Effort score (% share)"]
            txt += f" — statistically: <i>{esc(d['verdict'])}</i> (95% interval of the difference {d['low']:+.1f} to {d['high']:+.1f} points)"
        f.append(txt + ".")
    if r.monthly is not None and not r.monthly.empty and r.monthly["month"].nunique() >= 4:
        piv = r.monthly.pivot(index="month", columns="person", values="score")
        n = min(3, len(piv) // 2)
        early, late = piv.iloc[:n].mean(), piv.iloc[-n:].mean()
        ch = (late - early).dropna()
        if not ch.empty and ch.abs().max() >= 5:
            p = ch.abs().idxmax()
            f.append(f"Biggest change over time: <b>{esc(p)}</b>'s effort share has {'risen' if ch[p] > 0 else 'dropped'} "
                     f"from {early[p]:.0f}% (first {n} months) to {late[p]:.0f}% (last {n} months).")
    if r.change_dates:
        f.append("The dynamic shifted clearly around " + ", ".join(f"<b>{pd.Timestamp(d):%d %b %Y}</b>" for d in r.change_dates)
                 + " (see <i>Turning points</i>).")
    if r.hawkes is not None and len(r.hawkes.people) == 2:
        a, b = r.hawkes.people
        f.append(f"Responsiveness (Hawkes): each turn by {esc(a)} triggers {r.hawkes.alpha[1, 0]:.2f} turns by {esc(b)}; "
                 f"each turn by {esc(b)} triggers {r.hawkes.alpha[0, 1]:.2f} by {esc(a)}.")
    if r.cox and r.cox[0] is not None:
        sig = r.cox[0][(r.cox[0]["meaning"] != "no clear effect") & ~r.cox[0]["factor"].str.startswith("Time trend")]
        if len(sig):
            top_f = sig.iloc[(sig["hazard ratio"].apply(lambda x: abs(np.log(x)))).argmax()]
            f.append(f"Strongest factor for reply speed: <i>{esc(top_f['factor'])}</i> → {esc(top_f['meaning'])}.")
    return Section("findings", "Overview", "Key findings", "", [Html("<ul>" + "".join(f"<li>{x}</li>" for x in f) + "</ul>")])


def overview(r: Results) -> Section:
    s = r.settings
    ov = r.summary.copy()
    ov["median_response"] = ov["median_response_min"].map(fmt_minutes)
    quick_col = f"replies_under_{int(s.quick_reply_minutes)}min_pct"
    ov = ov[["messages", "words", "avg_words_per_text", "media", "questions", "active_days", "median_response",
             quick_col, "conversations_started", "late_replies", "conversations_ended",
             "left_without_answer", "followups_without_reply", "deleted"]]
    ov.columns = ["Messages", "Words", "Words / message", "Media", "Questions", "Active days", "Typical reply time",
                  f"Replies < {int(s.quick_reply_minutes)} min (%)", "Conversations started", "Late replies",
                  "Last word in conversation", "Started but got no answer", "Wrote again without reply", "Deleted"]
    return Section("overview", "Overview", "Overview", "", [Table(ov)])


def activity(r: Results) -> Section:
    mc = r.m[r.m["author"].isin(r.people)]
    act = A.activity_tables(mc)
    bm = act["by_month"].reset_index().melt(id_vars="month", var_name="person", value_name="messages")
    f1 = px.line(bm, x="month", y="messages", color="person", markers=True, color_discrete_map=r.colors,
                 title="Messages per month")
    bh = act["by_hour"].reset_index().melt(id_vars="hour", var_name="person", value_name="messages")
    f2 = px.bar(bh, x="hour", y="messages", color="person", barmode="group", color_discrete_map=r.colors,
                title="When do people write? (hour of day)")
    bw = act["by_weekday"].reset_index().melt(id_vars="weekday", var_name="person", value_name="messages")
    f3 = px.bar(bw, x="weekday", y="messages", color="person", barmode="group", color_discrete_map=r.colors,
                title="Messages by weekday")
    hm = act["heatmap"]
    f4 = px.imshow(hm.values, x=list(hm.columns), y=[str(i) for i in hm.index], color_continuous_scale="Blues",
                   aspect="auto", labels=dict(x="hour", y="", color="messages"), title="Activity heatmap (all people)")
    return Section("activity", "Activity", "Who writes, and when", "", [Fig(f1), Grid([Fig(f2), Fig(f3)]), Fig(f4, 320)])


def _period(start, end) -> str:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return f"{start:%d %b %Y}" if start == end else f"{start:%d %b %Y} – {end:%d %b %Y}"


def _streak_rule(st, is_group: bool) -> str:
    if st.min_people == 1:
        return "anyone wrote"
    return f"at least {st.min_people} people wrote" if is_group else "both of you wrote"


def _tile(value: str, label: str, sub: str = "") -> str:
    sub = f"<div style='color:#8c959f;font-size:12px'>{sub}</div>" if sub else ""
    return (f"<div style='border:1px solid #eaeef2;border-radius:10px;padding:10px 14px'>"
            f"<div style='font-size:26px;font-weight:600;color:#075e54'>{value}</div>"
            f"<div style='font-size:14px'>{label}</div>{sub}</div>")


def streaks(r: Results) -> Section | None:
    st = r.streaks
    if st is None or st.longest is None:
        return None
    top, span = st.longest, st.span_days
    n_talked = len(st.talked_days)
    cur_sub = "still going at the end of the export" if st.current else f"nobody talked on {st.last_day:%d %b %Y}"
    tiles = "".join([
        _tile(f"{int(top['days'])} days", "Longest streak", _period(top["start"], top["end"])),
        _tile(f"{st.current} days", "Current streak", cur_sub),
        _tile(f"{n_talked:,} / {span:,}", "Days talked", f"{n_talked / span * 100:.0f}% of the chat's days"),
        _tile(f"{int((st.runs['days'] >= 7).sum())}", "Streaks of a week or more"),
    ])
    blocks = [Html("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;"
                   f"margin-bottom:8px'>{tiles}</div>")]

    # calendar: one column per week, one row per weekday
    days = pd.date_range(st.first_day, st.last_day, freq="D")
    writers = r.m.groupby("date")["author"].unique()
    talked = set(st.talked_days)
    # categories: (label, colour); one-sided days get the colour of the person who wrote
    cats = [("no messages", "#ebedf0"), (f"talked ({_streak_rule(st, r.is_group)})", "#075e54")]
    cat_of = {}

    def category(label, color):
        if label not in cat_of:
            cat_of[label] = len(cats)
            cats.append((label, color))
        return cat_of[label]

    level, detail = [], []
    for d in days:
        who = list(writers.get(d, []))
        if d in talked:
            level.append(1)
            detail.append("talked: " + ", ".join(who))
        elif not who:
            level.append(0)
            detail.append("no messages")
        elif len(who) == 1:
            p = who[0]
            level.append(category(f"only {p}" if p in r.people else "only someone else",
                                  r.colors[p] if p in r.people else "#9aa4ae"))
            detail.append(f"only {who[0]} wrote")
        else:  # group chat with streak_min_people > 2: several wrote, but too few
            level.append(category("too few people", "#d6c08f"))
            detail.append("only " + ", ".join(who) + " wrote")
    n = len(cats)
    week = days - pd.to_timedelta(days.weekday, unit="D")
    cal = pd.DataFrame({"week": week, "weekday": days.weekday, "level": level,
                        "hover": [f"{d:%a %d %b %Y}: {t}" for d, t in zip(days, detail)]})
    z = cal.pivot(index="weekday", columns="week", values="level").reindex(range(7))
    hover = cal.pivot(index="weekday", columns="week", values="hover").reindex(range(7)).fillna("")
    scale = []
    for i, (_, color) in enumerate(cats):
        scale += [[i / n, color], [(i + 1) / n, color]]
    fig = go.Figure(go.Heatmap(
        z=z.values, x=list(z.columns), y=[d[:3] for d in A.WEEKDAYS], text=hover.values, hoverinfo="text",
        colorscale=scale, zmin=-0.5, zmax=n - 0.5, showscale=False, xgap=2, ygap=2, showlegend=False))
    for label, color in cats:  # heatmaps have no legend: add one marker per category
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(symbol="square", size=12, color=color)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Every day of the chat (hover a day to see who wrote)")
    blocks.append(Fig(fig, 300))

    tb = st.runs.head(10).copy()
    tb.index = range(1, len(tb) + 1)
    tb.index.name = "#"
    tb = tb.rename(columns={"start": "From", "end": "To", "days": "Days in a row"})[["Days in a row", "From", "To"]]
    tables = [Html("<h3>Longest streaks</h3>"), Table(tb)]
    if r.is_group or st.min_people > 1:
        tables += [Html("<h3>Each person: longest run of days they wrote</h3>"),
                   Table(st.per_person.reindex([p for p in r.people if p in st.per_person.index]))]
    blocks += tables
    return Section("streaks", "Activity", "Daily streaks",
                   f"A streak is a run of consecutive calendar days on which {_streak_rule(st, r.is_group)} "
                   f"(change with <code>--streak-min-people</code>). Days are cut at midnight; the current streak is "
                   f"measured against the last day of the export ({st.last_day:%d %b %Y}).", blocks)


def response_times(r: Results) -> Section:
    people, turns = r.people, r.turns
    replies = turns[turns["prev_author"].notna() & turns["author"].isin(people)].copy()
    replies["response (minutes)"] = replies["response_min"].clip(lower=0.1)
    f5 = px.box(replies, x="author", y="response (minutes)", color="author", color_discrete_map=r.colors, log_y=True,
                points=False, category_orders={"author": people},
                title="Reply times inside conversations (log scale; line = typical reply time)")
    ticks = [0.1, 0.5, 1, 5, 15, 60, 180, 600, 1440]
    f5.update_yaxes(tickvals=ticks, ticktext=["6 s", "30 s", "1 min", "5 min", "15 min", "1 h", "3 h", "10 h", "24 h"],
                    title="reply time")
    f5.update_layout(showlegend=False)
    bins = [0, 1, 5, 15, 60, 180, 24 * 60 * 10]
    labels = ["< 1 min", "1–5 min", "5–15 min", "15–60 min", "1–3 h", "> 3 h"]
    replies["bucket"] = pd.cut(replies["response_min"], bins=bins, labels=labels, include_lowest=True)
    dist = (pd.crosstab(replies["author"], replies["bucket"], normalize="index") * 100).round(1)
    dist = dist.reset_index().melt(id_vars="author", var_name="reply time", value_name="% of replies")
    f6 = px.bar(dist, x="author", y="% of replies", color="reply time", title="How quickly does each person reply?",
                category_orders={"author": people, "reply time": labels}, color_discrete_sequence=px.colors.sequential.Viridis)
    blocks = [Fig(f5), Fig(f6)]
    if r.is_group:
        rm = A.reply_matrix(turns).reindex(index=people, columns=people).fillna(0)
        f7 = px.imshow(rm.values, x=list(rm.columns), y=list(rm.index), color_continuous_scale="Oranges", text_auto=True,
                       labels=dict(x="replies to …", y="who replies", color="replies"), title="Who replies to whom")
        rt = A.reply_time_matrix(turns).reindex(index=people, columns=people)
        f8 = px.imshow(rt.values, x=list(rt.columns), y=list(rt.index), color_continuous_scale="RdYlGn_r", text_auto=True,
                       labels=dict(x="replying to …", y="who replies", color="minutes"), title="Typical reply time (minutes) by pair")
        blocks.append(Grid([Fig(f7, 420), Fig(f8, 420)]))
    return Section("responses", "Replies", "Response times",
                   f"A reply is the first message after someone else's message within the same conversation "
                   f"(silences longer than {r.settings.gap_hours:g} h start a new conversation).", blocks)


def survival(r: Results) -> Section | None:
    if r.km_curves is None or r.km_curves.empty:
        return None
    cap = r.options.survival_cap_hours
    fig = go.Figure()
    for who, d in r.km_curves.groupby("who", sort=False):
        color = r.colors.get(d["prompt_author"].iloc[0], "#555")
        other = who.split(" replying")[0] if " replying" in who else d["prompt_author"].iloc[0]
        color = r.colors.get(other, color)
        _band(fig, d["minutes"], d["low"], d["high"], color, who)
        fig.add_trace(go.Scatter(x=d["minutes"], y=d["waiting_pct"], name=who, line=dict(color=color, width=2.5)))
    fig.update_xaxes(type="log", title="time since the message", tickvals=[1, 5, 15, 60, 180, 600, 1440, 2880],
                     ticktext=["1 min", "5 min", "15 min", "1 h", "3 h", "10 h", "24 h", "48 h"])
    fig.update_yaxes(title="% of messages still waiting", range=[0, 100])
    fig.update_layout(title="How long messages wait for an answer (Kaplan–Meier, shaded = 95% interval)")
    tab = r.km_summary.set_index("who").copy()
    for c in ["typical wait (median)"]:
        tab[c] = [f"{fmt_minutes(a)} ({fmt_minutes(b)}–{fmt_minutes(c2)})" for a, b, c2 in
                  zip(r.km_summary["typical wait (median)"], r.km_summary["median low"], r.km_summary["median high"])]
    tab = tab.drop(columns=["median low", "median high"])
    blocks = [Fig(fig, 420), Table(tab)]
    if r.cox and r.cox[0] is not None:
        ct, conc, n = r.cox
        ct = ct.copy()
        ff = go.Figure()
        ct_sorted = ct.sort_values("hazard ratio")
        colors = ["#1d9a6c" if lo > 1 else "#c2185b" if hi < 1 else "#8c959f"
                  for lo, hi in zip(ct_sorted["low"], ct_sorted["high"])]
        ff.add_trace(go.Scatter(x=ct_sorted["hazard ratio"], y=ct_sorted["factor"], mode="markers",
                                marker=dict(size=10, color=colors),
                                error_x=dict(type="data", symmetric=False,
                                             array=ct_sorted["high"] - ct_sorted["hazard ratio"],
                                             arrayminus=ct_sorted["hazard ratio"] - ct_sorted["low"]),
                                showlegend=False))
        ff.add_vline(x=1, line_dash="dash", line_color="#888")
        ff.update_xaxes(type="log", title="hazard ratio (> 1 = faster replies, < 1 = slower)",
                        tickvals=[0.1, 0.2, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 5, 10],
                        ticktext=["0.1", "0.2", "0.33", "0.5", "0.67", "1", "1.5", "2", "3", "5", "10"])
        ff.update_layout(title="What makes replies come faster or slower (Cox model, 95% intervals)")
        ct_disp = ct.set_index("factor")[["hazard ratio", "low", "high", "p", "meaning"]]
        blocks += [Fig(ff, max(360, 28 * len(ct) + 120)),
                   Html(f"<p class='note'>Model quality: concordance {conc:.2f} on {n:,} messages "
                        "(0.5 = no better than chance, 1 = perfect ordering of reply times). Green = clear effect.</p>"),
                   Table(ct_disp, "{:,.3f}")]
    return Section("survival", "Replies", "How long do messages wait? (survival analysis)",
                   f"Every turn waits until someone else writes. Waits longer than {cap:g} h count as 'not answered' "
                   "and are kept in the analysis as unfinished waits (censored) instead of being dropped. Here a reply is "
                   "whatever the other person writes next, also the next day.", blocks)


def interest(r: Results) -> Section | None:
    if r.score is None:
        return None
    people_all, fair = r.people_all, 100 / len(r.people_all)
    sc = r.score.reindex(people_all).reset_index()
    sc.columns = ["person", "score"]
    err_lo = err_hi = None
    if r.ci_per is not None and not r.ci_per.empty:
        e = r.ci_per[r.ci_per["measure"] == "Effort score (% share)"].set_index("person")
        sc["lo"] = sc["person"].map(e["low"])
        sc["hi"] = sc["person"].map(e["high"])
        err_lo, err_hi = sc["score"] - sc["lo"], sc["hi"] - sc["score"]
    sc = sc.head(r.options.max_people)
    f9 = go.Figure(go.Bar(x=sc["person"], y=sc["score"], marker_color=[r.colors[p] for p in sc["person"]],
                          text=sc["score"].round(0), textposition="inside",
                          error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo) if err_lo is not None else None))
    f9.add_hline(y=fair, line_dash="dash", annotation_text="equal share")
    f9.update_layout(title="Overall effort score (% share, bars = 95% interval)")
    sh = r.shares[[p for p in r.people if p in r.shares.columns]]
    f10 = px.imshow(sh.values, x=list(sh.columns), y=list(sh.index), color_continuous_scale="RdBu",
                    zmin=0, zmax=100 if not r.is_group else max(2 * fair, np.nanmax(sh.values) if sh.size else 1),
                    color_continuous_midpoint=fair, text_auto=".0f", aspect="auto",
                    title="Share of each signal (%) · blue = more than equal share")
    blocks = [Grid([Fig(f9, 380), Fig(f10, 520)])]
    if r.monthly is not None and not r.monthly.empty and r.monthly["month"].nunique() >= 2:
        f11 = go.Figure()
        for p in r.people:
            d = r.monthly[r.monthly["person"] == p]
            if d.empty:
                continue
            _band(f11, d["month"], d["low"], d["high"], r.colors[p], p)
            f11.add_trace(go.Scatter(x=d["month"], y=d["score"], name=p, mode="lines+markers",
                                     line=dict(color=r.colors[p], width=2.5)))
        f11.add_hline(y=fair, line_dash="dash")
        for d0 in (r.change_dates or []):
            f11.add_vline(x=pd.Timestamp(d0), line_dash="dot", line_color="#d4a017")
        f11.update_layout(title="Effort score month by month (shaded = 95% interval; dotted = detected turning point)")
        f11.update_xaxes(title="month")
        blocks.append(Fig(f11, 400))
    title = "Engagement in the group" if r.is_group else "Who is more interested?"
    return Section("interest", "Interest", title,
                   "The score averages each person's share of: starting conversations, writing again without a reply, questions, "
                   "words written, reply speed, quick replies, affection, asking about the other, plans, compliments, laughter, "
                   "good morning / good night, enthusiasm, few dry replies and few negative words. Signals, not proof.", blocks)


def statistics(r: Results) -> Section | None:
    if r.ci_per is None or r.ci_per.empty:
        return None
    t = r.ci_per.copy()
    t["value (95% interval)"] = [_ci(e, l, h) for e, l, h in zip(t["estimate"], t["low"], t["high"])]
    piv = t.pivot(index="measure", columns="person", values="value (95% interval)")
    piv = piv.reindex([m for m in t["measure"].unique()])[[p for p in r.people if p in piv.columns]]
    blocks = [Table(piv)]
    note = (f"Intervals come from a cluster bootstrap: whole conversations are resampled {r.options.bootstrap:,} times, "
            "because messages inside one conversation are not independent. ")
    if r.ci_diff is not None:
        d = r.ci_diff.copy()
        a, b = r.people_all[0], r.people_all[1]
        d[f"difference ({a} − {b})"] = [_ci(e, l, h) for e, l, h in zip(d["difference"], d["low"], d["high"])]
        pcol = [c for c in d.columns if c.startswith("P(")][0]
        d[pcol] = (d[pcol] * 100).round(0).astype(int).astype(str) + "%"
        d = d[[f"difference ({a} − {b})", pcol, "verdict"]]
        blocks.append(Html("<h3>Are the differences real?</h3>"))
        blocks.append(Table(d))
        note += ("'Clear difference' = the 95% interval of the difference excludes zero. 'P(... higher)' is the share "
                 "of bootstrap samples in which the first person's value was higher.")
    return Section("statistics", "Interest", "How sure are these numbers? (confidence intervals)", note, blocks)


def change_points(r: Results) -> Section | None:
    if r.weekly is None or r.weekly.empty:
        return None
    w = r.weekly
    p0 = r.people_all[0]
    cols = [c for c in w.columns if "effort score" in c][:3] + [c for c in w.columns if "reply time" in c][:3]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=["Effort score", "Typical reply time (min, log scale)", "Messages per week"])
    for c in cols:
        person = c.split(":")[0]
        row = 1 if "effort" in c else 2
        fig.add_trace(go.Scatter(x=w.index, y=w[c], name=c, mode="lines+markers", marker=dict(size=4),
                                 line=dict(color=r.colors.get(person, "#555"), dash="solid" if row == 1 else "dot")),
                      row=row, col=1)
    fig.add_trace(go.Bar(x=w.index, y=w["Messages per week"], name="Messages per week", marker_color="#9aa5b1"), row=3, col=1)
    fig.update_yaxes(type="log", row=2, col=1, tickvals=[0.5, 1, 2, 5, 10, 30, 60, 180, 600, 1440],
                     ticktext=["30 s", "1 min", "2", "5", "10", "30", "1 h", "3 h", "10 h", "24 h"])
    for d in (r.change_dates or []):
        fig.add_vline(x=pd.Timestamp(d), line_dash="dash", line_color="#d4a017", line_width=2)
    fig.update_layout(title="Weekly measures with detected turning points (orange lines)")
    blocks = [Fig(fig, 640)]
    if r.change_dates:
        from .stats.changepoints import describe
        blocks.insert(0, Html("<ul>" + "".join(f"<li>{x}</li>" for x in describe(r.change_table)) + "</ul>"))
        t = r.change_table.copy()
        t["change week"] = pd.to_datetime(t["change week"]).dt.strftime("%d %b %Y")
        blocks.append(Table(t.set_index("change week"), "{:,.2f}"))
    else:
        blocks.insert(0, Html("<p>No clear turning point: the weekly measures vary, but no shift is larger than normal week-to-week noise.</p>"))
    return Section("changes", "Dynamics", "Turning points (change-point detection)",
                   "Weekly series are standardised and searched for points where their average level shifts (exact optimal partitioning with a penalty per change, the problem the PELT algorithm solves). "
                   "'size (sd)' = how big the jump was, in standard deviations of that measure.", blocks)


def hawkes(r: Results) -> Section | None:
    h = r.hawkes
    if h is None:
        return None
    from .stats.hawkes import describe
    mat = h.matrix()
    fig = px.imshow(mat.values, x=[f"after {p}" for p in mat.columns], y=list(mat.index), text_auto=".2f",
                    color_continuous_scale="Purples", aspect="auto",
                    labels=dict(color="turns triggered"), title="Turns triggered: row person reacting to column person")
    blocks = [Html("<ul>" + "".join(f"<li>{x}</li>" for x in describe(h)) + "</ul>"), Fig(fig, 300 + 40 * len(mat))]
    hp = r.hawkes_periods
    if hp is not None and not hp.empty and hp["period"].nunique() >= 2:
        hp = hp.copy()
        hp["pair"] = hp["responder"] + " reacting to " + hp["trigger"]
        hp["period"] = pd.PeriodIndex(hp["period"], freq="Q").astype(str)
        f2 = px.line(hp, x="period", y="alpha", color="pair", markers=True,
                     title="Responsiveness per quarter (turns triggered by one turn of the other)")
        f2.update_xaxes(title="quarter")
        blocks.append(Fig(f2))
    return Section("hawkes", "Dynamics", "Who triggers whom (Hawkes process)",
                   "Each turn can trigger reactions that fade over time. The model separates <i>spontaneous</i> writing "
                   "(with a daily rhythm) from writing <i>triggered</i> by the other person, and estimates how many reactions "
                   "one turn causes on average.", blocks)


def style(r: Results) -> Section | None:
    if not r.style and not r.mirror:
        return None
    blocks = []
    st = r.style or {}
    if "overall" in st and not st["overall"].empty:
        blocks.append(Table(st["overall"].set_index("pair"), "{:.3f}"))
    if "monthly" in st and not st["monthly"].empty:
        mm = st["monthly"].melt(id_vars="month", var_name="measure", value_name="similarity")
        f = px.line(mm, x="month", y="similarity", color="measure", markers=True,
                    title="How similar your writing styles are, month by month (1 = identical)")
        f.update_yaxes(range=[0, 1])
        f.update_xaxes(title="month")
        blocks.append(Fig(f))
    mi = r.mirror or {}
    if "table" in mi and not mi["table"].empty:
        t = mi["table"]
        f2 = px.bar(t, x="feature", y="mirroring", color="person", barmode="group", color_discrete_map=r.colors,
                    error_y=t["high"] - t["mirroring"], error_y_minus=t["mirroring"] - t["low"],
                    title="Mirroring: how much more often a reply uses a feature when the other just used it (percentage points)")
        blocks.append(Fig(f2, 420))
    if "summary" in mi and not mi["summary"].empty:
        blocks.append(Grid([Table(mi["summary"]), Table(mi["echo"]) if not mi["echo"].empty else Html("")]))
    if "quarterly" in mi and not mi["quarterly"].empty and mi["quarterly"]["quarter"].nunique() >= 2:
        q = mi["quarterly"].copy()
        q["quarter"] = pd.PeriodIndex(q["quarter"], freq="Q").astype(str)
        f3 = px.line(q, x="quarter", y="average mirroring", color="person", markers=True,
                     color_discrete_map=r.colors, title="Average mirroring per quarter")
        f3.update_xaxes(title="quarter")
        blocks.append(Fig(f3))
    if "function_words" in st:
        blocks.append(Html("<h3>Function words (% of words)</h3>"))
        blocks.append(Table(st["function_words"], "{:.2f}"))
    return Section("style", "Dynamics", "Style matching & mirroring",
                   "<b>Style matching</b> compares how often each person uses small function words (pronouns, negations, "
                   "particles like <i>yani, işte, bi</i> …) and style features; people in tune converge. <b>Mirroring</b> "
                   "asks whether a reply picks up what the other just did (emoji, laughter, stretched letters …), compared "
                   "within the same month so that slow changes over time do not count.", blocks)


def prediction(r: Results) -> Section | None:
    p = r.predict
    if not p:
        return None
    met = p["metrics"]
    blocks = [Html(f"<p>Task: predict whether a message is answered within <b>{p['horizon_h']:g} h</b>. "
                   f"Trained on {p['n_train']:,} messages, tested on the last {p['n_test']:,} "
                   f"(from {pd.Timestamp(p['test_from']):%d %b %Y}). Base rate: {p['base_rate'] * 100:.0f}% answered.</p>"),
              Table(met, "{:.3f}")]
    imp = p["importance"].head(12).iloc[::-1]
    f = px.bar(imp, x="importance", y="feature", orientation="h", error_x="sd",
               title=f"What matters most ({p['best_model']}; drop in AUC when the feature is shuffled)")
    blocks.append(Fig(f, 420))
    if p.get("shap") is not None:
        s = p["shap"].head(12).iloc[::-1]
        blocks.append(Fig(px.bar(s, x="mean |SHAP|", y="feature", orientation="h", title="SHAP importance (gradient boosting)"), 420))
    eff = p["effects"]
    eff = pd.concat([eff.head(5), eff.tail(5)]).drop_duplicates().set_index("feature")
    blocks.append(Html("<h3>Readable effects (logistic regression)</h3><p class='note'>Odds ratio per standard deviation: "
                       "above 1 = more likely to get a quick answer.</p>"))
    blocks.append(Table(eff, "{:.2f}"))
    return Section("prediction", "Prediction", "Will this message get a reply? (prediction model)",
                   "Features are only what was known when the message was sent (no look into the future). AUC 0.5 = guessing, "
                   "1.0 = perfect. Compare with the baselines to see how much the model adds.", blocks)


def network(r: Results) -> Section | None:
    n = r.network
    if not n:
        return None
    G, pos, nodes = n["graph"], n["positions"], n["nodes"]
    fig = go.Figure()
    maxw = max([d["weight"] for _, _, d in G.edges(data=True)] or [1])
    seen = set()
    for a, b, d in G.edges(data=True):
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        w = d["weight"] + (G[b][a]["weight"] if G.has_edge(b, a) else 0)
        fig.add_trace(go.Scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
                                 line=dict(width=1 + 8 * w / (2 * maxw), color="rgba(120,120,140,0.45)"),
                                 hoverinfo="text", text=f"{a} ↔ {b}: {w} replies", showlegend=False))
    comm_colors = px.colors.qualitative.Set2
    fig.add_trace(go.Scatter(
        x=[pos[p][0] for p in nodes.index], y=[pos[p][1] for p in nodes.index], mode="markers+text",
        text=list(nodes.index), textposition="top center",
        marker=dict(size=12 + 30 * np.sqrt(nodes["messages"] / max(nodes["messages"].max(), 1)),
                    color=[comm_colors[(g - 1) % len(comm_colors)] for g in nodes["sub-group"]],
                    line=dict(width=1, color="#333")),
        hovertext=[f"{p}: {int(x)} messages" for p, x in nodes["messages"].items()], hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=f"Reply network (node size = messages, colour = sub-group; {n['n_communities']} sub-group(s))")
    blocks = [Fig(fig, 520), Table(nodes, "{:,.3f}")]
    if not n["pairs"].empty:
        blocks.append(Table(n["pairs"].head(20).set_index("pair"), "{:.2f}"))
    att = n["attention_monthly"]
    if len(att) >= 2:
        a2 = att.reset_index().melt(id_vars="month", var_name="person", value_name="% of replies received")
        f2 = px.area(a2, x="month", y="% of replies received", color="person", color_discrete_map=r.colors,
                     title="Who gets the attention (share of replies received per month)")
        f2.update_xaxes(title="month")
        blocks.append(Fig(f2))
    return Section("network", "Group", "The group as a network",
                   "PageRank = attention from people who themselves get attention; betweenness = bridging sub-groups; "
                   "balance = how even each pair's replies are (1 = equal).", blocks)


def words(r: Results) -> Section | None:
    if r.rates is None:
        return None
    rates = r.rates.rename(columns=r.labels).reset_index()
    rates = rates.rename(columns={rates.columns[0]: "person"}).melt(id_vars="person", var_name="category", value_name="% of messages")
    f12 = px.bar(rates, x="category", y="% of messages", color="person", barmode="group", color_discrete_map=r.colors,
                 title="Word classifier: % of each person's messages in each category")
    blocks = [Fig(f12, 420)]
    keep = r.m["author"].isin(r.people)
    cm = I.category_monthly(r.m[keep], r.cats.loc[keep])
    if not cm.empty and cm["month"].nunique() >= 2:
        active = cm.groupby("category")["rate"].sum()
        cm = cm[cm["category"].isin(active[active > 0].index)].rename(columns={"author": "person"})
        cm["category"] = cm["category"].map(r.labels)
        f13 = px.line(cm, x="month", y="rate", color="person", facet_col="category", facet_col_wrap=3,
                      color_discrete_map=r.colors, markers=True, title="Word categories over time (% of messages)")
        f13.update_yaxes(matches=None, title=None, showticklabels=True)
        f13.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        blocks.append(Fig(f13, 760))
    return Section("words", "Words", "What people write: word classifier",
                   "Words, phrases and emojis matched against the editable word lists <code>lexicon_tr.json</code> (Turkish "
                   "with slang and spoken forms) and <code>lexicon.json</code> (English, Italian, Spanish, German, French). "
                   "A message counts once per category.", blocks)


def typical_words(r: Results) -> Section | None:
    if not r.distinct:
        return None
    cells = []
    for p in r.people:
        items = r.distinct.get(p, [])[:15]
        lis = "".join(f"<li>{esc(w)} <span class='muted'>×{n}</span></li>" for w, n, _ in items) or "<li class='muted'>not enough text</li>"
        cells.append(f"<div class='card'><h3 style='color:{r.colors[p]}'>{esc(p)}</h3><ol>{lis}</ol></div>")
    return Section("typical", "Words", "Words typical for each person",
                   "Words each person uses much more often than the others (works in any language).",
                   [Html("<div class='cards'>" + "".join(cells) + "</div>")])
