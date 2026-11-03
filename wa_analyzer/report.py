"""Render the analysis sections as one offline HTML file."""
from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from . import analysis as A
from .pipeline import Options, Results, analyze
from .sections import Fig, Grid, Html, Section, Table, build_sections, style_fig


def table_html(df: pd.DataFrame, fmt: str = "{:,.1f}", index: bool = True) -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "–"
            return f"{int(v):,}" if float(v).is_integer() and abs(v) >= 1 else fmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, pd.Timestamp):
            return f"{v:%d %b %Y}"
        return html.escape(str(v))
    cols = list(df.columns)
    head = ("<th>" + html.escape(str(df.index.name or "")) + "</th>" if index else "") + \
        "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    rows = []
    for i, row in zip(df.index, df.itertuples(index=False)):
        cells = "".join(f"<td>{f(v)}</td>" for v in row)
        rows.append("<tr>" + (f"<th>{html.escape(str(i))}</th>" if index else "") + cells + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def block_html(b) -> str:
    if isinstance(b, Fig):
        fig = style_fig(b.fig, b.height)
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
    if isinstance(b, Table):
        return table_html(b.df, b.fmt, b.index)
    if isinstance(b, Html):
        return b.html
    if isinstance(b, Grid):
        return '<div class="grid">' + "".join(f"<div>{block_html(x)}</div>" for x in b.items) + "</div>"
    return ""


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1f2328}
header{background:#075e54;color:#fff;padding:28px 32px} header h1{margin:0 0 6px;font-size:26px}
header p{margin:0;opacity:.85} main{max-width:1200px;margin:0 auto;padding:16px}
nav{max-width:1200px;margin:0 auto;padding:10px 16px 0;display:flex;flex-wrap:wrap;gap:6px}
nav a{font-size:13px;background:#fff;border:1px solid #d0d7de;border-radius:14px;padding:3px 10px;color:#075e54;text-decoration:none}
section{background:#fff;border-radius:12px;padding:20px 24px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
h2{margin-top:0;font-size:20px} h3{font-size:16px;margin:18px 0 6px} .note{color:#57606a;font-size:14px;max-width:950px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px} .grid>div{min-width:0}
.tw{overflow-x:auto;margin:8px 0} table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:6px 10px;border-bottom:1px solid #eaeef2;text-align:right;white-space:nowrap}
thead th{background:#f6f8fa;position:sticky;top:0} tbody th{text-align:left}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.card{border:1px solid #eaeef2;border-radius:10px;padding:10px 14px} .card h3{margin:4px 0;font-size:15px}
.card ol{margin:0;padding-left:20px;font-size:14px;line-height:1.6} .muted{color:#8c959f;font-size:12px}
li{margin:4px 0} code{background:#f6f8fa;padding:1px 5px;border-radius:4px}
footer{text-align:center;color:#8c959f;font-size:12px;padding:20px}
"""


def meta_line(r: Results) -> str:
    m = r.m
    start, end = m["timestamp"].min(), m["timestamp"].max()
    meta = (f"{len(m):,} messages · {len(r.people_all)} people · {start:%d %b %Y} – {end:%d %b %Y} · "
            f"{'group chat' if r.is_group else 'one-to-one chat'} · {len(r.convs):,} conversations")
    if len(r.people_all) > len(r.people):
        meta += f" · charts show the {len(r.people)} most active people"
    return meta


def render_html(r: Results, sections: list[Section]) -> str:
    nav = "".join(f'<a href="#{s.id}">{html.escape(s.title.split(" (")[0])}</a>' for s in sections)
    body = []
    for s in sections:
        note = f"<p class='note'>{s.note}</p>" if s.note else ""
        body.append(f"<section id='{s.id}'><h2>{html.escape(s.title)}</h2>{note}{''.join(block_html(b) for b in s.blocks)}</section>")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(r.title)} – WhatsApp analysis</title>
<script>{get_plotlyjs()}</script>
<style>{CSS}</style></head><body>
<header><h1>{html.escape(r.title)}</h1><p>{meta_line(r)}</p></header>
<nav>{nav}</nav>
<main>{''.join(body)}</main>
<script>window.addEventListener("load",()=>document.querySelectorAll(".plotly-graph-div").forEach(d=>Plotly.Plots.resize(d)));</script>
<footer>Generated locally on {datetime.now():%d %b %Y %H:%M}. Nothing was uploaded.</footer>
</body></html>"""


def build_report(df: pd.DataFrame, title: str, s: A.Settings, lexicon_path=None, options: Options | None = None,
                 progress=None) -> str:
    o = options or Options()
    if lexicon_path is not None:
        o.lexicon = lexicon_path if isinstance(lexicon_path, list) else [lexicon_path]
    r = analyze(df, title, s, o, progress=progress)
    return render_html(r, build_sections(r))
