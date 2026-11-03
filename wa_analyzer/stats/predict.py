"""Reply prediction: will this message get an answer within H hours?

  * One row per turn ("prompt"), features known at the moment it was sent (no peeking into the future:
    rolling reply statistics only use earlier messages whose outcome was already known).
  * Time-ordered split: the model is trained on the first 80% of the chat and tested on the last 20%.
  * Two models: logistic regression (readable effects) and gradient boosting (more flexible).
  * Explanations: permutation importance on the test period, SHAP values if `shap` is installed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

SKLEARN_OK = True
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    SKLEARN_OK = False

FEATURE_LABELS = {
    "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)", "weekend": "Weekend", "night": "Night",
    "log_words": "Message length", "n_msgs": "Messages in the turn", "has_question": "Contains a question",
    "has_media": "Media", "first_in_conv": "Opens a conversation", "turn_index": "Position in conversation",
    "log_gap_before": "Time since previous turn", "recent_median_wait": "Recent typical reply time",
    "recent_reply_rate": "Recent reply rate", "activity_24h": "Messages in the last 24 h",
    "trend": "Time trend",
}


def build_features(prompts: pd.DataFrame, m: pd.DataFrame, people: list[str], horizon_h: float,
                   window: int = 20) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    p = prompts.sort_values("end").reset_index(drop=True)
    # drop prompts whose outcome is unknown at the horizon (end of export)
    known = (p["replied"] == 1) | (p["wait_h"] >= horizon_h)
    p = p[known].reset_index(drop=True)
    y = ((p["replied"] == 1) & (p["wait_h"] <= horizon_h)).astype(int)

    X = pd.DataFrame(index=p.index)
    X["hour_sin"] = np.sin(2 * np.pi * p["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * p["hour"] / 24)
    X["weekend"] = p["weekend"]
    X["night"] = p["night"]
    X["log_words"] = np.log1p(p["words"])
    X["n_msgs"] = p["n_msgs"].clip(upper=20)
    X["has_question"] = p["has_question"]
    X["has_media"] = p["has_media"]
    X["first_in_conv"] = p["first_in_conv"]
    X["turn_index"] = p["turn_index"].clip(upper=50)
    X["log_gap_before"] = np.log1p(p["gap_before_h"])
    for a in people[1:8]:
        X[f"by::{a}"] = (p["author"] == a).astype(int)
    for c in [c for c in p.columns if c.startswith("cat_")]:
        if p[c].sum() >= 20:
            X[c] = p[c]
    X["trend"] = (p["end"] - p["end"].min()).dt.total_seconds() / 86400 / 365.25

    # rolling, leakage-free reply history for messages by the same author
    med = np.full(len(p), np.nan)
    rate = np.full(len(p), np.nan)
    resolved_at = p["end"] + pd.to_timedelta(p["wait_h"], unit="h")
    for a, idx in p.groupby("author").groups.items():
        idx = np.asarray(idx)
        ends = p.loc[idx, "end"].to_numpy()
        res = resolved_at.loc[idx].to_numpy()
        waits = p.loc[idx, "wait_h"].to_numpy()
        ys = y.loc[idx].to_numpy()
        for k in range(len(idx)):
            lo = max(0, k - 4 * window)
            prev = np.arange(lo, k)
            prev = prev[res[prev] <= ends[k]][-window:]
            if len(prev) >= 3:
                med[idx[k]] = np.median(waits[prev])
                rate[idx[k]] = ys[prev].mean()
    X["recent_median_wait"] = np.log1p(pd.Series(med).fillna(np.nanmedian(med) if np.isfinite(med).any() else 1))
    X["recent_reply_rate"] = pd.Series(rate).fillna(y.mean())
    ts = m["timestamp"].sort_values().to_numpy()
    e = p["end"].to_numpy()
    X["activity_24h"] = np.searchsorted(ts, e, side="right") - np.searchsorted(ts, e - np.timedelta64(24, "h"))
    return X, y, p["end"]


def run(prompts: pd.DataFrame, m: pd.DataFrame, people: list[str], horizon_h: float = 1.0,
        test_share: float = 0.2, seed: int = 0) -> dict | None:
    if not SKLEARN_OK:
        raise ImportError("pip install scikit-learn")
    X, y, when = build_features(prompts, m, people, horizon_h)
    n = len(X)
    if n < 300 or y.nunique() < 2:
        return None
    cut = int(n * (1 - test_share))
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    if ytr.nunique() < 2 or yte.nunique() < 2:
        return None

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         l2_regularization=1.0, early_stopping=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logit.fit(Xtr, ytr)
        gbm.fit(Xtr, ytr)

    def scores(model):
        pr = model.predict_proba(Xte)[:, 1]
        return {"ROC AUC": roc_auc_score(yte, pr), "Average precision": average_precision_score(yte, pr),
                "Brier score": brier_score_loss(yte, pr), "Accuracy": ((pr >= 0.5) == yte).mean()}

    base_rate = float(ytr.mean())
    metrics = pd.DataFrame({
        "Logistic regression": scores(logit),
        "Gradient boosting": scores(gbm),
        "Baseline: recent reply rate only": {
            "ROC AUC": roc_auc_score(yte, Xte["recent_reply_rate"]),
            "Average precision": average_precision_score(yte, Xte["recent_reply_rate"]),
            "Brier score": brier_score_loss(yte, Xte["recent_reply_rate"].clip(0, 1)),
            "Accuracy": ((Xte["recent_reply_rate"] >= 0.5) == yte).mean()},
        "Baseline: always the average": {
            "ROC AUC": 0.5, "Average precision": float(yte.mean()),
            "Brier score": brier_score_loss(yte, np.full(len(yte), base_rate)),
            "Accuracy": max(yte.mean(), 1 - yte.mean())},
    }).T

    best = gbm if metrics.loc["Gradient boosting", "ROC AUC"] >= metrics.loc["Logistic regression", "ROC AUC"] else logit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=seed)
    imp = pd.DataFrame({"feature": [label(c) for c in X.columns], "importance": pi.importances_mean,
                        "sd": pi.importances_std}).sort_values("importance", ascending=False)

    coefs = logit[-1].coef_[0]
    effects = pd.DataFrame({"feature": [label(c) for c in X.columns],
                            "odds ratio per SD": np.exp(coefs)}).sort_values("odds ratio per SD", ascending=False)

    shap_df = None
    try:  # optional
        import shap
        explainer = shap.TreeExplainer(gbm)
        sv = explainer.shap_values(Xte)
        sv = sv[1] if isinstance(sv, list) else sv
        shap_df = pd.DataFrame({"feature": [label(c) for c in X.columns],
                                "mean |SHAP|": np.abs(sv).mean(axis=0)}).sort_values("mean |SHAP|", ascending=False)
    except Exception:
        pass

    return {"metrics": metrics, "importance": imp, "effects": effects, "shap": shap_df,
            "n_train": len(Xtr), "n_test": len(Xte), "base_rate": base_rate, "horizon_h": horizon_h,
            "test_from": when.iloc[cut], "best_model": "Gradient boosting" if best is gbm else "Logistic regression"}


def label(c: str) -> str:
    if c.startswith("by::"):
        return f"Written by {c[4:]}"
    if c.startswith("cat_"):
        return "Category: " + c[4:].replace("_", " ")
    return FEATURE_LABELS.get(c, c)
