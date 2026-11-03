"""Load saved AI results (CSV files in the '<chat>_ai' folder) for the report and the dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _csv(path: Path, dates=("start", "month", "timestamp")) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for c in dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load(ai_dir: str | Path, m: pd.DataFrame | None = None) -> dict:
    d = Path(ai_dir)
    out: dict = {"dir": str(d)}
    lo = m["timestamp"].min() if m is not None else None
    hi = m["timestamp"].max() if m is not None else None

    def clip(df, col="start"):
        if df is None or lo is None or col not in df:
            return df
        return df[(df[col] >= lo) & (df[col] <= hi)].reset_index(drop=True)

    s = clip(_csv(d / "sentiment.csv"))
    if s is not None and len(s):
        out["sentiment"] = s
        meta = d / "sentiment_meta.json"
        out["sentiment_meta"] = json.loads(meta.read_text()) if meta.exists() else {}
    t, td = _csv(d / "topics.csv"), clip(_csv(d / "topic_docs.csv"))
    if t is not None and td is not None and len(td):
        out["topics"], out["topic_docs"] = t, td
    rc = clip(_csv(d / "reader_conversations.csv"))
    if rc is not None and len(rc):
        out["reader_conversations"] = rc
        out["reader_scores"] = clip(_csv(d / "reader_scores.csv"))
        rm = _csv(d / "reader_months.csv")
        if rm is not None and lo is not None:
            rm = rm[(rm["month"] >= lo.to_period("M").to_timestamp()) & (rm["month"] <= hi)]
        out["reader_months"] = rm
        ov = d / "reader_overall.txt"
        out["reader_overall"] = ov.read_text(encoding="utf-8") if ov.exists() else ""
        meta = d / "reader_meta.json"
        out["reader_meta"] = json.loads(meta.read_text()) if meta.exists() else {}
    for name in ("classifier_metrics", "accuracy"):
        df = _csv(d / f"{name}.csv", dates=())
        if df is not None and len(df):
            out[name] = df
    if (d / "message_classifier.joblib").exists():
        out["classifier_path"] = str(d / "message_classifier.joblib")
    return out
