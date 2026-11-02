"""Build a single, offline HTML report."""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from . import analysis as A
from . import interest as I

PALETTE = ["#2a6fdb", "#e0663a", "#1d9a6c", "#a64ac9", "#d4a017", "#c2185b", "#00838f", "#6d4c41",
           "#5c6bc0", "#7cb342", "#8d6e63", "#78909c"]
MAX_PEOPLE_IN_CHARTS = 12


def _fig(fig: go.Figure, height=360) -> str:
    for ax in fig.layout:
        if ax.startswith("xaxis") and fig.layout[ax].title and fig.layout[ax].title.text == "month":
            fig.layout[ax].update(dtick="M1", tickformat="%b %Y")
    fig.update_layout(height=height, margin=dict(l=40, r=20, t=40, b=40), template="plotly_white",
                      font=dict(family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=12),
                      legend=dict(orientation="h", y=-0.2))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def _table(df: pd.DataFrame, fmt: str = "{:,.1f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else (f"{int(v):,}" if float(v).is_integer() else fmt.format(v))
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    body = "".join(
        "<tr><th>" + html.escape(str(i)) + "</th>" + "".join(f"<td>{f(v)}</td>" for v in row) + "</tr>"
        for i, row in zip(df.index, df.itertuples(index=False))
    )
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _fmt_minutes(x: float) -> str:
    if x is None or np.isnan(x):
        return "–"
    if x < 1:
        return f"{x * 60:.0f} sec"
    if x < 90:
        return f"{x:.0f} min"
    return f"{x / 60:.1f} h"


def build_report(df: pd.DataFrame, title: str, s: A.Settings, lexicon_path=None) -> str:
    m = A.prepare(df, s)
    if m.empty:
        raise ValueError("The chat contains no messages from people.")
    turns = A.build_turns(m)
    convs = A.build_conversations(m, turns, s)
    summary = A.people_summary(m, turns, convs, s)
    people_all = list(summary.index)
    people = people_all[:MAX_PEOPLE_IN_CHARTS]
    is_group = len(people_all) > 2
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(people_all)}
    mc = m[m["author"].isin(people)]

    clf = I.WordClassifier(lexicon_path)
    cats = clf.apply(m)
    rates = I.category_rates(m, cats).reindex(people)
    shares, score = I.effort_balance(m, turns, convs, cats, s, people_all)
    monthly_score = I.effort_balance_monthly(m, turns, convs, cats, s, people_all)
    distinct = I.distinctive_words(m)

    sec = []

    # ------------------------------------------------------------ key findings
    findings = []
    top = summary["messages"]
    findings.append(f"<b>{html.escape(top.index[0])}</b> writes the most: {top.iloc[0]:,} messages "
                    f"({top.iloc[0] / top.sum() * 100:.0f}% of all).")
    med = summary["median_response_min"].replace(0, np.nan).dropna()
    rep = summary["replies"]
    med = med[rep.reindex(med.index) >= 5]
    if len(med) >= 2:
        findings.append(f"Fastest typical reply: <b>{html.escape(med.idxmin())}</b> ({_fmt_minutes(med.min())}); "
                        f"slowest: <b>{html.escape(med.idxmax())}</b> ({_fmt_minutes(med.max())}).")
    st = summary["conversations_started"]
    if st.sum() > 0:
        findings.append(f"<b>{html.escape(st.idxmax())}</b> starts most conversations "
                        f"({st.max() / st.sum() * 100:.0f}%).")
    fair = 100 / len(people_all)
    lead = score.idxmax()
    word = "puts the most effort into the chat" if is_group else "shows more interest / effort"
    findings.append(f"Overall, <b>{html.escape(lead)}</b> {word}: effort score {score.max():.0f}% "
                    f"(an equal share would be {fair:.0f}%).")
    if len(monthly_score) >= 4:
        n = min(3, len(monthly_score) // 2)
        early, late = monthly_score.iloc[:n].mean(), monthly_score.iloc[-n:].mean()
        change = (late - early).dropna()
        if not change.empty and change.abs().max() >= 5:
            p = change.abs().idxmax()
            direction = "risen" if change[p] > 0 else "dropped"
            findings.append(f"Biggest change over time: <b>{html.escape(p)}</b>'s effort share has {direction} "
                            f"from {early[p]:.0f}% (first {n} months) to {late[p]:.0f}% (last {n} months).")
    sec.append("<section><h2>Key findings</h2><ul>" + "".join(f"<li>{x}</li>" for x in findings) + "</ul></section>")

    # ------------------------------------------------------------ overview
    ov = summary.copy()
    ov["median_response"] = ov["median_response_min"].map(_fmt_minutes)
    quick_col = f"replies_under_{int(s.quick_reply_minutes)}min_pct"
    ov = ov[["messages", "words", "avg_words_per_text", "media", "questions", "active_days", "median_response",
             quick_col, "conversations_started", "late_replies", "conversations_ended",
             "left_without_answer", "followups_without_reply", "deleted"]]
    ov.columns = ["Messages", "Words", "Words / message", "Media", "Questions", "Active days", "Typical reply time",
                  f"Replies < {int(s.quick_reply_minutes)} min (%)", "Conversations started", "Late replies",
                  "Last word in conversation", "Started but got no answer", "Wrote again without reply", "Deleted"]
    sec.append("<section><h2>Overview</h2>" + _table(ov) + "</section>")

    # ------------------------------------------------------------ activity
    act = A.activity_tables(mc)
    bm = act["by_month"].reset_index().melt(id_vars="month", var_name="person", value_name="messages")
    f1 = px.line(bm, x="month", y="messages", color="person", markers=True, color_discrete_map=colors,
                 title="Messages per month")
    bh = act["by_hour"].reset_index().melt(id_vars="hour", var_name="person", value_name="messages")
    f2 = px.bar(bh, x="hour", y="messages", color="person", barmode="group", color_discrete_map=colors,
                title="When do people write? (hour of day)")
    bw = act["by_weekday"].reset_index().melt(id_vars="weekday", var_name="person", value_name="messages")
    f3 = px.bar(bw, x="weekday", y="messages", color="person", barmode="group", color_discrete_map=colors,
                title="Messages by weekday")
    hm = act["heatmap"]
    f4 = px.imshow(hm.values, x=list(hm.columns), y=[str(i) for i in hm.index], color_continuous_scale="Blues",
                   aspect="auto", labels=dict(x="hour", y="", color="messages"), title="Activity heatmap (all people)")
    sec.append("<section><h2>Who writes, and when</h2>" + _fig(f1) + '<div class="grid">' + _fig(f2) + _fig(f3)
               + "</div>" + _fig(f4, 320) + "</section>")

    # ------------------------------------------------------------ response times
    replies = turns[turns["prev_author"].notna() & turns["author"].isin(people)].copy()
    replies["response (minutes)"] = replies["response_min"].clip(lower=0.1)
    f5 = px.box(replies, x="author", y="response (minutes)", color="author", color_discrete_map=colors, log_y=True,
                points=False, category_orders={"author": people},
                title="Reply times (log scale; the line in each box is the typical reply time)")
    ticks = [0.1, 0.5, 1, 5, 15, 60, 180, 600, 1440]
    f5.update_yaxes(tickvals=ticks, ticktext=["6 s", "30 s", "1 min", "5 min", "15 min", "1 h", "3 h", "10 h", "24 h"],
                    title="reply time")
    f5.update_layout(showlegend=False)
    bins = [0, 1, 5, 15, 60, 180, 24 * 60 * 10]
    labels = ["< 1 min", "1–5 min", "5–15 min", "15–60 min", "1–3 h", "> 3 h"]
    replies["bucket"] = pd.cut(replies["response_min"], bins=bins, labels=labels, include_lowest=True)
    dist = (pd.crosstab(replies["author"], replies["bucket"], normalize="index") * 100).round(1)
    dist = dist.reset_index().melt(id_vars="author", var_name="reply time", value_name="% of replies")
    f6 = px.bar(dist, x="author", y="% of replies", color="reply time", title="How quickly does each person reply?", category_orders={"author": people, "reply time": labels},
                color_discrete_sequence=px.colors.sequential.Viridis)
    parts = [_fig(f5), _fig(f6)]
    if is_group:
        rm = A.reply_matrix(turns).reindex(index=people, columns=people).fillna(0)
        f7 = px.imshow(rm.values, x=list(rm.columns), y=list(rm.index), color_continuous_scale="Oranges", text_auto=True,
                       labels=dict(x="replies to …", y="who replies", color="replies"), title="Who replies to whom")
        rt = A.reply_time_matrix(turns).reindex(index=people, columns=people)
        f8 = px.imshow(rt.values, x=list(rt.columns), y=list(rt.index), color_continuous_scale="RdYlGn_r", text_auto=True,
                       labels=dict(x="replying to …", y="who replies", color="minutes"),
                       title="Typical reply time (minutes) by pair")
        parts.append('<div class="grid">' + _fig(f7, 420) + _fig(f8, 420) + "</div>")
    sec.append(f"<section><h2>Response times</h2><p class='note'>A reply is the first message after someone else's "
               f"message within the same conversation (silences longer than {s.gap_hours:g} h start a new "
               f"conversation).</p>" + "".join(parts) + "</section>")

    # ------------------------------------------------------------ interest / effort
    t = "Engagement in the group" if is_group else "Who is more interested?"
    sc = score.reindex(people_all).reset_index()
    sc.columns = ["person", "score"]
    f9 = px.bar(sc.head(MAX_PEOPLE_IN_CHARTS), x="person", y="score", color="person", color_discrete_map=colors,
                title="Overall effort score (% share)", text_auto=".0f")
    f9.add_hline(y=fair, line_dash="dash", annotation_text="equal share")
    f9.update_layout(showlegend=False)
    sh = shares[people].copy()
    f10 = px.imshow(sh.values, x=list(sh.columns), y=list(sh.index), color_continuous_scale="RdBu",
                    zmin=0, zmax=100 if not is_group else max(2 * fair, sh.values.max() if sh.size else 1),
                    color_continuous_midpoint=fair, text_auto=".0f", aspect="auto",
                    title="Share of each signal (%)  ·  blue = more than equal share")
    parts = ['<div class="grid">' + _fig(f9, 380) + _fig(f10, 480) + "</div>"]
    if len(monthly_score) >= 2:
        ms = monthly_score[[p for p in people if p in monthly_score.columns]]
        ms = ms.reset_index().rename(columns={"index": "month"}).melt(id_vars="month", var_name="person", value_name="score")
        f11 = px.line(ms, x="month", y="score", color="person", markers=True, color_discrete_map=colors,
                      title="Effort score month by month (months with few messages are skipped)")
        f11.add_hline(y=fair, line_dash="dash")
        parts.append(_fig(f11))
    sec.append(f"<section><h2>{t}</h2><p class='note'>The score averages the person's share of: starting "
               "conversations, writing again without a reply, questions, words written, reply speed, quick replies, "
               "affection, asking about the other, making plans, compliments, laughter, good morning / good night "
               "messages, enthusiasm, few dry replies and few negative words. These are signals, not proof — someone busy may reply slowly and still care a lot."
               "</p>" + "".join(parts) + "</section>")

    # ------------------------------------------------------------ word classifier
    labels_map = clf.labels
    r = rates.rename(columns=labels_map).reset_index().rename(columns={"author": "person"})
    r = r.rename(columns={r.columns[0]: "person"}).melt(id_vars="person", var_name="category", value_name="% of messages")
    f12 = px.bar(r, x="category", y="% of messages", color="person", barmode="group", color_discrete_map=colors,
                 title="Word classifier: % of each person's messages in each category")
    cm = I.category_monthly(m[m["author"].isin(people)], cats.loc[m["author"].isin(people)])
    parts = [_fig(f12, 420)]
    if not cm.empty and cm["month"].nunique() >= 2:
        active = cm.groupby("category")["rate"].sum()
        cm = cm[cm["category"].isin(active[active > 0].index)].rename(columns={"author": "person"})
        cm["category"] = cm["category"].map(labels_map)
        f13 = px.line(cm, x="month", y="rate", color="person", facet_col="category", facet_col_wrap=3,
                      color_discrete_map=colors, markers=True, title="Word categories over time (% of messages)")
        f13.update_yaxes(matches=None, title=None, showticklabels=True)
        f13.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        parts.append(_fig(f13, 720))
    sec.append("<section><h2>What people write: word classifier</h2><p class='note'>Words, phrases and emojis are "
               "matched against editable word lists: <code>lexicon_tr.json</code> (Turkish, including slang, abbreviations, "
               "spellings without Turkish letters and spoken forms like <i>seviom</i>, <i>napıyon</i>, <i>gelicem</i>) and "
               "<code>lexicon.json</code> (English, Italian, Spanish, German, French). A message counts once per category.</p>" + "".join(parts) + "</section>")

    # ------------------------------------------------------------ distinctive words
    cells = []
    for p in people:
        items = distinct.get(p, [])[:15]
        lis = "".join(f"<li>{html.escape(w)} <span class='muted'>×{n}</span></li>" for w, n, _ in items) or "<li class='muted'>not enough text</li>"
        cells.append(f"<div class='card'><h3 style='color:{colors[p]}'>{html.escape(p)}</h3><ol>{lis}</ol></div>")
    sec.append("<section><h2>Words typical for each person</h2><p class='note'>Words each person uses much more "
               "often than the others (works in any language).</p><div class='cards'>" + "".join(cells) + "</div></section>")

    # ------------------------------------------------------------ page
    start, end = m["timestamp"].min(), m["timestamp"].max()
    meta = (f"{len(m):,} messages · {len(people_all)} people · {start:%d %b %Y} – {end:%d %b %Y} · "
            f"{'group chat' if is_group else 'one-to-one chat'} · {len(convs):,} conversations")
    if len(people_all) > MAX_PEOPLE_IN_CHARTS:
        meta += f" · charts show the {MAX_PEOPLE_IN_CHARTS} most active people"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} – WhatsApp analysis</title>
<script>{get_plotlyjs()}</script>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1f2328}}
header{{background:#075e54;color:#fff;padding:28px 32px}} header h1{{margin:0 0 6px;font-size:26px}}
header p{{margin:0;opacity:.85}} main{{max-width:1200px;margin:0 auto;padding:16px}}
section{{background:#fff;border-radius:12px;padding:20px 24px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
h2{{margin-top:0;font-size:20px}} .note{{color:#57606a;font-size:14px;max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}} .grid>div{{min-width:0}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{padding:6px 10px;border-bottom:1px solid #eaeef2;text-align:right;white-space:nowrap}}
thead th{{background:#f6f8fa;position:sticky;top:0}} tbody th{{text-align:left}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}}
.card{{border:1px solid #eaeef2;border-radius:10px;padding:10px 14px}} .card h3{{margin:4px 0;font-size:15px}}
.card ol{{margin:0;padding-left:20px;font-size:14px;line-height:1.6}} .muted{{color:#8c959f;font-size:12px}}
li{{margin:4px 0}} code{{background:#f6f8fa;padding:1px 5px;border-radius:4px}}
footer{{text-align:center;color:#8c959f;font-size:12px;padding:20px}}
</style></head><body>
<header><h1>{html.escape(title)}</h1><p>{meta}</p></header>
<main>{''.join(sec)}</main>
<script>window.addEventListener("load",()=>document.querySelectorAll(".plotly-graph-div").forEach(d=>Plotly.Plots.resize(d)));</script>
<footer>Generated locally on {datetime.now():%d %b %Y %H:%M}. Nothing was uploaded.</footer>
</body></html>"""
