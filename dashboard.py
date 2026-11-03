"""Local dashboard.   Start with:   streamlit run dashboard.py

Everything runs on your computer; uploaded files are only read by this local app.
"""
from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from wa_analyzer.analysis import Settings  # noqa: E402
from wa_analyzer.loading import default_ai_dir, load_filtered  # noqa: E402
from wa_analyzer.pipeline import Options, analyze  # noqa: E402
from wa_analyzer.report import meta_line, render_html  # noqa: E402
from wa_analyzer.sections import Fig, Grid, Html, Table, build_sections, style_fig  # noqa: E402

st.set_page_config(page_title="WhatsApp Analyzer", page_icon="💬", layout="wide")
TAB_ORDER = ["Overview", "Activity", "Replies", "Interest", "Dynamics", "Prediction", "Group", "Words", "AI"]
HERE = Path(__file__).parent


# ------------------------------------------------------------------ helpers

def show_plot(fig, height):
    fig = style_fig(fig, height)
    try:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def show_table(df: pd.DataFrame):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(3)
    try:
        st.dataframe(d, width="stretch")
    except TypeError:
        st.dataframe(d, use_container_width=True)


def show_block(b):
    if isinstance(b, Fig):
        show_plot(b.fig, b.height)
    elif isinstance(b, Table):
        show_table(b.df)
    elif isinstance(b, Html):
        if b.html.strip():
            st.markdown(b.html, unsafe_allow_html=True)
    elif isinstance(b, Grid):
        cols = st.columns(len(b.items))
        for col, item in zip(cols, b.items):
            with col:
                show_block(item)


def candidate_chats() -> list[str]:
    found = []
    for folder in [HERE / "chats", Path.home()]:
        if folder.exists():
            for p in sorted(folder.glob("*")):
                if p.suffix.lower() in (".txt", ".zip") and p.stat().st_size > 1000 and "requirements" not in p.name:
                    found.append(str(p))
    return found


@st.cache_data(show_spinner=False)
def cached_load(path: str, mtime: float, date_order, since, until, exclude):
    return load_filtered(path, date_order, since, until, list(exclude))


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_analysis(path: str, mtime: float, date_order, since, until, exclude, settings_t, options_t, ai_mtime):
    df = cached_load(path, mtime, date_order, since, until, exclude)
    s = Settings(*settings_t)
    o = Options(**dict(options_t))
    title = Path(path).stem.replace("WhatsApp Chat with ", "").replace("_", " ")
    return analyze(df, title, s, o)


def ai_folder_mtime(d: Path) -> float:
    if not d.exists():
        return 0.0
    return max([p.stat().st_mtime for p in d.glob("*.csv")] + [p.stat().st_mtime for p in d.glob("*.joblib")] + [0.0])


# ------------------------------------------------------------------ sidebar

st.sidebar.title("💬 WhatsApp Analyzer")
st.sidebar.caption("Runs fully on your computer.")

chats = candidate_chats()
choice = st.sidebar.selectbox("Chat export", ["(type a path or upload)"] + chats, index=1 if chats else 0)
typed = st.sidebar.text_input("…or path to a .txt / .zip", value="" if choice.startswith("(") else choice)
up = st.sidebar.file_uploader("…or drop the export here", type=["txt", "zip"])
if up is not None:
    tmp = Path(tempfile.gettempdir()) / f"wa_upload_{up.name}"
    tmp.write_bytes(up.getvalue())
    typed = str(tmp)

path = typed.strip().strip('"')
if not path or not Path(path).expanduser().exists():
    st.title("WhatsApp chat analyzer")
    st.info("Choose a chat export in the sidebar (a .txt or .zip exported from WhatsApp, without media).")
    st.stop()
path = str(Path(path).expanduser())
mtime = Path(path).stat().st_mtime

with st.sidebar.expander("Filters", expanded=False):
    date_order = st.selectbox("Date order", ["auto", "DMY", "MDY", "YMD"], index=0)
    date_order = None if date_order == "auto" else date_order
    raw = cached_load(path, mtime, date_order, None, None, ())
    lo, hi = raw["timestamp"].min().date(), raw["timestamp"].max().date()
    rng = st.date_input("Period", value=(lo, hi), min_value=lo, max_value=hi)
    since, until = (str(rng[0]), str(rng[1])) if isinstance(rng, (tuple, list)) and len(rng) == 2 else (None, None)
    people_all = list(raw.loc[raw["kind"] != "system", "author"].value_counts().index)
    exclude = tuple(st.multiselect("Leave out", people_all))

with st.sidebar.expander("Settings", expanded=False):
    gap = st.slider("New conversation after a silence of (hours)", 1.0, 24.0, 6.0, 0.5)
    followup = st.slider("Follow-up after (minutes without answer)", 5, 180, 30, 5)
    late = st.slider("Late reply window (hours)", 6, 72, 36, 2)
    quick = st.slider("Quick reply limit (minutes)", 1, 30, 5, 1)
    run_stats = st.checkbox("Statistical models", value=True,
                            help="Confidence intervals, survival model, change points, Hawkes, style, prediction, network")
    boot = st.select_slider("Bootstrap samples", [200, 500, 1000, 2000], value=1000)
    pred_h = st.select_slider("Prediction: answered within (hours)", [0.25, 0.5, 1.0, 2.0, 6.0], value=1.0)
    cap = st.select_slider("Survival: give up waiting after (hours)", [12, 24, 48, 72, 168], value=48)

ai_dir = default_ai_dir(path)
use_ai = st.sidebar.checkbox(f"Include AI results ({ai_dir.name})", value=ai_dir.exists(), disabled=not ai_dir.exists(),
                             help="Create them in the 'Run local AI' panel below or with python -m wa_analyzer.ai")

settings_t = (gap, float(late), float(followup), float(quick))
options_t = tuple(sorted({"stats": run_stats, "bootstrap": int(boot), "predict_hours": float(pred_h),
                          "survival_cap_hours": float(cap), "ai_dir": str(ai_dir) if use_ai else None}.items()))

# ------------------------------------------------------------------ run local AI from the dashboard

with st.sidebar.expander("Run local AI", expanded=False):
    st.caption("Models run on this computer. First use downloads them once. On CPU this can take a while; "
               "results are cached so you can stop and continue.")
    device = st.radio("Device", ["auto", "cpu", "cuda"], horizontal=True)
    ollama_model = st.text_input("Ollama model", value="qwen2.5:3b" if device == "cpu" else "qwen2.5:7b")
    per_month = st.number_input("LLM: conversations per month", 1, 100, 4 if device == "cpu" else 12)
    max_turns = st.number_input("Sentiment: max turns (0 = all)", 0, 500000, 0, step=1000)
    jobs = st.multiselect("What to run", ["sentiment", "topics", "read"], default=[])
    if st.button("Run", disabled=not jobs):
        from wa_analyzer.analysis import build_conversations, build_turns, prepare
        df_ai = cached_load(path, mtime, date_order, since, until, exclude)
        s_ai = Settings(*settings_t)
        m_ai = prepare(df_ai, s_ai)
        convs_ai = build_conversations(m_ai, build_turns(m_ai), s_ai)
        logbox = st.empty()
        lines: list[str] = []

        def log(msg):
            lines.append(str(msg))
            logbox.code("\n".join(lines[-12:]))
        try:
            if "sentiment" in jobs:
                from wa_analyzer.ai import sentiment as S
                S.run(m_ai, ai_dir, device=device, max_turns=int(max_turns) or None, log=log)
            if "topics" in jobs:
                from wa_analyzer.ai import topics as T
                T.run(m_ai, ai_dir, device=device, log=log)
            if "read" in jobs:
                from wa_analyzer.ai import reader as R
                from wa_analyzer.ai.ollama import Ollama
                llm = Ollama(model=ollama_model, device="cpu" if device == "cpu" else "auto")
                R.run(m_ai, convs_ai, ai_dir, llm, per_month=int(per_month),
                      max_chars=4000 if device == "cpu" else 8000, log=log)
            st.success("Done. Tick 'Include AI results' above.")
        except ImportError as e:
            st.error(f"Missing package: {e}. Run bash setup_cpu.sh or bash setup_gpu.sh first.")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

with st.sidebar.expander("Test the word classifier", expanded=False):
    txt = st.text_input("Message", value="seni seviom aşkm 😘")
    if txt:
        from wa_analyzer.lexicon_engine import WordClassifier, looks_like_question
        clf = WordClassifier()
        found = clf.explain(txt)
        st.write({clf.labels[k]: ", ".join(dict.fromkeys(v)) for k, v in found.items()} or "(no category)")
        st.caption(f"question: {'yes' if looks_like_question(txt) else 'no'}")

# ------------------------------------------------------------------ main page

t0 = time.time()
with st.spinner("Analysing…"):
    r = cached_analysis(path, mtime, date_order, since, until, exclude, settings_t, options_t,
                        ai_folder_mtime(ai_dir) if use_ai else 0.0)
sections = build_sections(r)

st.title(r.title)
st.caption(meta_line(r))
c1, c2 = st.columns([1, 4])
with c1:
    st.download_button("Download HTML report", render_html(r, sections).encode("utf-8"),
                       file_name=f"{Path(path).stem}_report.html", mime="text/html")

tabs_present = [t for t in TAB_ORDER if any(s.tab == t for s in sections)]
for tab, name in zip(st.tabs(tabs_present), tabs_present):
    with tab:
        for s in [s for s in sections if s.tab == name]:
            st.subheader(s.title)
            if s.note:
                st.markdown(f"<p style='color:#57606a;font-size:0.9rem'>{s.note}</p>", unsafe_allow_html=True)
            for b in s.blocks:
                show_block(b)
            st.divider()
