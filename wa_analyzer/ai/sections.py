"""Report sections for the AI results."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..sections import Fig, Grid, Html, Section, Table, _band, esc


def ai_sections(r) -> list[Section]:
    ai = r.ai
    out = []
    if "sentiment" in ai:
        out.append(sentiment_section(r, ai["sentiment"], ai.get("sentiment_meta", {})))
    if "reader_conversations" in ai:
        out.append(reader_section(r, ai))
    if "topics" in ai:
        out.append(topics_section(r, ai["topics"], ai["topic_docs"]))
    if "trained_classifier" in ai or "classifier_metrics" in ai or "accuracy" in ai:
        out.append(classifier_section(r, ai))
    return [s for s in out if s is not None]


def sentiment_section(r, s: pd.DataFrame, meta: dict) -> Section | None:
    s = s[s["author"].isin(r.people)].copy()
    if s.empty:
        return None
    blocks = []
    g = s.groupby(["month", "author"])["polarity"].agg(["mean", "std", "size"]).reset_index()
    g = g[g["size"] >= 10]
    g["lo"] = g["mean"] - 1.96 * g["std"] / np.sqrt(g["size"])
    g["hi"] = g["mean"] + 1.96 * g["std"] / np.sqrt(g["size"])
    fig = go.Figure()
    for p in r.people:
        d = g[g["author"] == p]
        if d.empty:
            continue
        _band(fig, d["month"], d["lo"], d["hi"], r.colors[p], p)
        fig.add_trace(go.Scatter(x=d["month"], y=d["mean"], name=p, mode="lines+markers", line=dict(color=r.colors[p], width=2.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="#888")
    fig.update_layout(title="Average tone per month (−1 negative … +1 positive; shaded = 95% interval)")
    fig.update_xaxes(title="month")
    blocks.append(Fig(fig, 400))

    dist = (pd.crosstab(s["author"], s["sentiment"], normalize="index") * 100).reindex(columns=["negative", "neutral", "positive"]).fillna(0)
    d2 = dist.reset_index().melt(id_vars="author", var_name="sentiment", value_name="% of turns")
    f2 = px.bar(d2, x="author", y="% of turns", color="sentiment", barmode="stack",
                color_discrete_map={"negative": "#c2185b", "neutral": "#b0b7c3", "positive": "#1d9a6c"},
                title="Tone of each person's turns")
    items = [Fig(f2)]
    if "emotion" in s:
        emo = (pd.crosstab(s["author"], s["emotion"], normalize="index") * 100)
        d3 = emo.reset_index().melt(id_vars="author", var_name="emotion", value_name="% of turns")
        f3 = px.bar(d3, x="author", y="% of turns", color="emotion", barmode="stack",
                    color_discrete_map={"joy": "#d4a017", "sadness": "#5c6bc0", "anger": "#c2185b", "fear": "#6d4c41"},
                    title="Strongest emotion per turn")
        items.append(Fig(f3))
    blocks.append(Grid(items))

    # emotional contagion: does a reply follow the tone of the message it answers?
    t = r.turns.set_index("turn_id")[["author", "prev_author"]]
    pol = s.drop_duplicates("turn_id").set_index("turn_id")["polarity"]
    d = t.join(pol.rename("cur"), how="inner")
    d["prev"] = pol.reindex(d.index - 1).to_numpy()
    d = d[d["prev_author"].notna() & d["prev"].notna()]
    rows = []
    for p in r.people[:6]:
        x = d[d["author"] == p]
        if len(x) < 30:
            continue
        cur, prv = x["cur"].to_numpy(), x["prev"].to_numpy()

        def avg(mask):
            return cur[mask].mean() if mask.sum() >= 5 else np.nan
        rows.append({"person": p, "tone after a positive message": avg(prv > 0.3),
                     "tone after a neutral message": avg(np.abs(prv) <= 0.3),
                     "tone after a negative message": avg(prv < -0.3),
                     "correlation with previous tone": np.corrcoef(cur, prv)[0, 1]})
    if rows:
        blocks.append(Html("<h3>Emotional contagion</h3><p class='note'>Average tone of a person's reply depending on the "
                           "tone of the message they answered. A strong correlation = they mirror the other's mood.</p>"))
        blocks.append(Table(pd.DataFrame(rows).set_index("person"), "{:+.2f}"))
    models = ", ".join(x for x in [meta.get("sentiment_model"), meta.get("turkish_model"), meta.get("emotion_model")] if x)
    return Section("ai_sentiment", "AI", "Tone and emotions (AI model)",
                   f"Each turn was read by transformer models that understand context and negation "
                   f"({esc(models) or 'see sentiment_meta.json'}; device: {esc(meta.get('device', '?'))}). "
                   "Tone = probability positive − probability negative; for Turkish turns averaged with the Turkish model.", blocks)


def reader_section(r, ai: dict) -> Section | None:
    conv, sc = ai["reader_conversations"], ai.get("reader_scores")
    blocks = []
    if ai.get("reader_overall"):
        blocks.append(Html(f"<p style='font-size:15px;line-height:1.5'><b>Overall:</b> {esc(ai['reader_overall'])}</p>"))
    if sc is not None and not sc.empty:
        sc = sc[sc["person"].isin(r.people)]
        avg = sc.groupby("person")[["warmth", "engagement", "interest"]].mean().reindex([p for p in r.people if p in set(sc["person"])])
        blocks.append(Table(avg.round(2), "{:.2f}"))
        mon = sc.groupby(["month", "person"])[["warmth", "engagement", "interest"]].mean().reset_index()
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Warmth", "Engagement", "Interest in the other"], shared_yaxes=True)
        for j, col in enumerate(["warmth", "engagement", "interest"], 1):
            for p in r.people:
                d = mon[mon["person"] == p]
                if d.empty:
                    continue
                fig.add_trace(go.Scatter(x=d["month"], y=d[col], name=p, mode="lines+markers", showlegend=(j == 1),
                                         line=dict(color=r.colors[p], width=2)), row=1, col=j)
        fig.update_yaxes(range=[1, 5])
        fig.update_layout(title="LLM ratings per month (1–5, averaged over the conversations read)")
        blocks.append(Fig(fig, 380))
    items = []
    drv = conv["drives"].value_counts(normalize=True).mul(100).rename_axis("drives").reset_index(name="% of conversations")
    items.append(Fig(px.bar(drv, x="drives", y="% of conversations", title="Who drives the conversations")))
    tone = (pd.crosstab(conv["month"], conv["tone"], normalize="index") * 100).reset_index().melt(id_vars="month", var_name="tone", value_name="%")
    ft = px.bar(tone, x="month", y="%", color="tone", title="Tone of conversations per month",
                color_discrete_map={"affectionate": "#e91e63", "friendly": "#1d9a6c", "playful": "#d4a017", "neutral": "#b0b7c3",
                                    "practical": "#5c6bc0", "tense": "#e0663a", "conflict": "#b71c1c", "cold": "#455a64"})
    ft.update_xaxes(title="month")
    items.append(Fig(ft))
    blocks.append(Grid(items))
    ms = ai.get("reader_months")
    if ms is not None and not ms.empty:
        t = ms.copy()
        t["month"] = t["month"].dt.strftime("%b %Y")
        blocks.append(Html("<h3>Month by month</h3>"))
        blocks.append(Table(t.set_index("month"), index=True))
    recent = conv.sort_values("start", ascending=False).head(25).copy()
    recent["date"] = recent["start"].dt.strftime("%d %b %Y")
    blocks.append(Html("<h3>Latest conversations read</h3>"))
    blocks.append(Table(recent.set_index("date")[["tone", "drives", "conflict", "summary"]]))
    meta = ai.get("reader_meta", {})
    return Section("ai_reader", "AI", "What the local LLM sees in your conversations",
                   f"{len(conv):,} conversations were read by {esc(meta.get('model', 'a local model'))} via Ollama "
                   f"({'all' if meta.get('per_month') is None else 'up to ' + str(meta.get('per_month')) + ' per month'}). "
                   "LLM judgements are subjective and can be wrong: use them alongside the measured statistics.", blocks)


def topics_section(r, topics: pd.DataFrame, docs: pd.DataFrame) -> Section | None:
    blocks = [Table(topics.set_index("name")[["keywords", "size", "example"]])]
    pal = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Set3
    tcol = {n: pal[i % len(pal)] for i, n in enumerate(topics.sort_values("size", ascending=False)["name"])}
    share = pd.crosstab(docs["month"], docs["topic_name"], normalize="index") * 100
    if len(share) >= 2:
        d = share.reset_index().melt(id_vars="month", var_name="topic", value_name="% of conversation chunks")
        f = px.area(d, x="month", y="% of conversation chunks", color="topic", title="Topics over time",
                    color_discrete_map=tcol, category_orders={"topic": list(tcol)})
        f.update_xaxes(title="month")
        blocks.append(Fig(f, 460))
    rows = []
    for a, t in zip(docs["authors"], docs["topic_name"]):
        try:
            w = json.loads(a) if isinstance(a, str) else a
        except Exception:
            continue
        for p, x in w.items():
            if p in r.people:
                rows.append({"person": p, "topic": t, "w": x})
    items = []
    if rows:
        pt = pd.DataFrame(rows).groupby(["person", "topic"])["w"].sum()
        pt = (pt / pt.groupby(level=0).transform("sum") * 100).reset_index(name="% of their words")
        items.append(Fig(px.bar(pt, x="person", y="% of their words", color="topic", title="Who talks about what",
                                    color_discrete_map=tcol, category_orders={"topic": list(tcol)}), 460))
    smp = docs.sample(min(len(docs), 3000), random_state=0)
    items.append(Fig(px.scatter(smp, x="x", y="y", color="topic_name", hover_data=["month"], opacity=0.6,
                                color_discrete_map=tcol, category_orders={"topic_name": list(tcol)},
                                title="Topic map (each dot = a conversation chunk; close = similar content)"), 460))
    blocks.append(Grid(items))
    return Section("ai_topics", "AI", "What you talk about (topic model)",
                   "Conversations were turned into multilingual sentence embeddings and grouped by meaning; keywords are the words "
                   "that set each topic apart.", blocks)


def classifier_section(r, ai: dict) -> Section | None:
    blocks = []
    if "accuracy" in ai:
        blocks.append(Html("<h3>How accurate are the labels? (checked by you)</h3><p class='note'>Against the messages you reviewed "
                           "with <code>python -m wa_analyzer.ai review</code>. F1: 1 = perfect.</p>"))
        blocks.append(Table(ai["accuracy"].set_index(ai["accuracy"].columns[0]), "{:.2f}"))
    if "classifier_metrics" in ai:
        blocks.append(Html("<h3>Trained classifier quality</h3><p class='note'>5-fold cross-validation against the LLM labels "
                           "(your reviews take priority), compared with the word lists on the same messages.</p>"))
        blocks.append(Table(ai["classifier_metrics"].set_index(ai["classifier_metrics"].columns[0]), "{:.2f}"))
    tc = ai.get("trained_classifier")
    if tc is not None:
        pred = tc["predictions"]
        is_text = r.m["kind"] == "text"
        rows = []
        for c in pred.columns:
            for p in r.people:
                sel = is_text & (r.m["author"] == p)
                if sel.sum() == 0:
                    continue
                rows.append({"category": c.replace("_", " "), "person": p, "method": "trained classifier",
                             "% of messages": pred.loc[sel, c].mean() * 100})
                if c in r.cats:
                    rows.append({"category": c.replace("_", " "), "person": p, "method": "word lists",
                                 "% of messages": r.cats.loc[sel, c].mean() * 100})
        d = pd.DataFrame(rows)
        f = px.bar(d, x="category", y="% of messages", color="person", barmode="group", pattern_shape="method",
                   color_discrete_map=r.colors, title="Trained classifier vs word lists (% of each person's messages)")
        blocks.append(Fig(f, 460))
    if not blocks:
        return None
    return Section("ai_classifier", "AI", "LLM-labelled classifier & accuracy check",
                   "A local LLM labelled a sample of your messages; a small fast model was trained on those labels and applied "
                   "to every message.", blocks)
