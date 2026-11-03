"""Style matching and mirroring.

Two complementary ideas from conversation research:

1. Language Style Matching (LSM; Ireland & Pennebaker): people who are in tune use small "function words"
   (pronouns, negations, conjunctions, discourse particles, intensifiers ...) at similar rates.
   LSM per category = 1 - |p1 - p2| / (p1 + p2); averaged over categories. 1 = identical style.
   Here extended with a multilingual (Turkish / English / Italian) word list.

2. Mirroring / coordination (Danescu-Niculescu-Mizil et al.): does someone's reply pick up a feature of
   the message they answer? For a feature f (emoji, laughter, question, '!', stretched letters ...):
   mirroring = P(reply has f | message had f) - P(reply has f | message did not have f)
   Positive = the person adapts to the other. The difference is computed within each month and then
   averaged (stratified), so slow changes over time (both people getting warmer) are not mistaken for
   mirroring.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..lexicon_engine import key, tokens

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F000-\U0001F02F\U0001F0A0-\U0001F1FF☀-➿⬀-⯿⌀-⏿❤☺]")
STRETCH_RE = re.compile(r"([^\W\d_])\1\1", re.U)
CAPS_RE = re.compile(r"\b[^\W\d_]{3,}\b", re.U)

FUNCTION_WORDS = {
    "pronouns": "ben sen o biz siz onlar beni seni onu bizi sizi bana sana ona bize size benim senin onun bizim sizin "
                "kendim kendin i you me my mine your yours we us our he she him her they them "
                "io tu lui lei noi voi loro me te mi ti ci vi mio mia tuo tua",
    "negations": "değil degil deil yok hayır hayir hiç hic yo not no never nothing don't dont can't cant won't isn't "
                 "non mai niente nessuno nemmeno neanche",
    "conjunctions": "ve ama fakat ancak çünkü cunku ya veya yoksa ki hem ise and but because or so if then "
                    "e ed ma però pero perché perche quindi oppure se anche",
    "discourse": "işte iste yani bi hani şey sey aslında aslinda zaten acaba artık artik well like just actually "
                 "anyway basically literally cioè cioe tipo allora dai boh comunque insomma praticamente",
    "intensifiers": "çok cok az biraz hep her en daha bayağı bayagi epey cidden gerçekten very so really much too "
                    "totally super molto tanto troppo poco sempre proprio davvero veramente",
    "questions": "ne neden niye nasıl nasil nerede nerde kim hangi kaç kac mı mi mu mü what why how where who which "
                 "when cosa che perché come dove chi quale quando",
    "articles_prepositions": "the a an of in on at to for with from il lo la i gli le un una uno di da in con su per tra fra",
}
FW_KEYS = {cat: {key(w) for w in words.split()} for cat, words in FUNCTION_WORDS.items()}

ABBREVIATIONS = {key(w) for w in (
    "tmm tm slm mrb nbr nslsn cnm aşkm tşk tşkler kib sçs knk grşz grsz eyw eyv bgn yrn nrd nzmn kb aeo arv inş "
    "ok okk lol lmao xd omg idk btw brb tbh imo pls plz thx ty u ur np "
    "tvb tvtb cmq nn xke xké xk x qnd cm tt ke qlc qlcs tvb grz").split()}

FEATURES = {
    "emoji": "Emoji",
    "laughter": "Laughter",
    "question": "Question",
    "exclaim": "Exclamation mark",
    "ellipsis": "Ellipsis (...)",
    "stretch": "Stretched letters (canımmm)",
    "caps": "CAPITALS",
    "abbrev": "Texting abbreviations",
    "long": "Long message (8+ words)",
}


def message_features(m: pd.DataFrame, cats: pd.DataFrame | None = None) -> pd.DataFrame:
    """0/1 style features per message (text messages only; others are 0)."""
    is_text = m["kind"] == "text"
    txt = m["text"].where(is_text, "")
    f = pd.DataFrame(index=m.index)
    f["emoji"] = txt.str.contains(EMOJI_RE)
    f["laughter"] = cats["laughter"].astype(bool) if cats is not None and "laughter" in cats else False
    f["question"] = m["is_question"]
    f["exclaim"] = txt.str.contains("!", regex=False)
    f["ellipsis"] = txt.str.contains(r"\.\.\.|…", regex=True)
    f["stretch"] = txt.map(lambda x: bool(STRETCH_RE.search(x)))
    f["caps"] = txt.map(lambda s: any(w.isupper() for w in CAPS_RE.findall(s)))
    f["abbrev"] = txt.map(lambda s: any(key(tok) in ABBREVIATIONS for tok in tokens(s)))
    f["long"] = m["words"] >= 8
    return (f & is_text.to_numpy()[:, None]).astype(int)


def function_word_rates(m: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """% of words in each function-word category, per group (e.g. month, author)."""
    is_text = m["kind"] == "text"
    d = m.loc[is_text, group_cols + ["text"]].copy()
    toks = d["text"].map(lambda s: [key(t) for t in tokens(s)])
    d["n"] = toks.map(len)
    for cat, keys in FW_KEYS.items():
        d[cat] = toks.map(lambda ts, k=keys: sum(t in k for t in ts))
    g = d.groupby(group_cols)[["n"] + list(FW_KEYS)].sum()
    rates = g[list(FW_KEYS)].div(g["n"].replace(0, np.nan), axis=0) * 100
    rates["words"] = g["n"]
    return rates


def _lsm(a: pd.Series, b: pd.Series) -> float:
    vals = []
    for cat in FW_KEYS:
        pa, pb = a.get(cat, np.nan), b.get(cat, np.nan)
        if np.isfinite(pa) and np.isfinite(pb) and (pa + pb) > 0:
            vals.append(1 - abs(pa - pb) / (pa + pb + 1e-4))
    return float(np.mean(vals)) if vals else np.nan


def style_matching(m: pd.DataFrame, people: list[str], feats: pd.DataFrame, min_words: int = 150) -> dict:
    """LSM overall and per month (pairs), plus style-feature similarity per month."""
    out = {}
    rates = function_word_rates(m, ["author"])
    pairs = [(people[0], people[1])] if len(people) == 2 else \
        [(a, b) for i, a in enumerate(people[:6]) for b in people[i + 1:6]]
    out["overall"] = pd.DataFrame([{"pair": f"{a} ↔ {b}", "LSM": _lsm(rates.loc[a], rates.loc[b])}
                                   for a, b in pairs if a in rates.index and b in rates.index])
    out["function_words"] = rates.drop(columns="words").reindex(people[:6]).round(2)
    if len(people) >= 2:
        a, b = people[0], people[1]
        mon = function_word_rates(m, ["month", "author"])
        rows = []
        f = feats.copy()
        f["month"], f["author"] = m["month"], m["author"]
        frates = f[f["author"].isin([a, b])].groupby(["month", "author"]).mean()
        for month in sorted(m["month"].unique()):
            try:
                ra, rb = mon.loc[(month, a)], mon.loc[(month, b)]
            except KeyError:
                continue
            if ra["words"] < min_words or rb["words"] < min_words:
                continue
            fa, fb = frates.loc[(month, a)], frates.loc[(month, b)]
            sim = np.nanmean([1 - abs(fa[c] - fb[c]) / (fa[c] + fb[c]) for c in FEATURES if (fa[c] + fb[c]) > 0])
            rows.append({"month": month, "Function-word matching (LSM)": _lsm(ra, rb),
                         "Style-feature similarity": sim})
        out["monthly"] = pd.DataFrame(rows)
    return out


def _turn_table(m: pd.DataFrame, feats: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    f = feats.groupby(m["turn_id"]).max()
    t = turns.set_index("turn_id")[["author", "conv_id", "prev_author", "start", "words"]].join(f)
    texts = m[m["kind"] == "text"].groupby("turn_id")["text"].apply(lambda s: " ".join(s))
    t["text"] = texts.reindex(t.index).fillna("")
    return t


def mirroring(m: pd.DataFrame, turns: pd.DataFrame, feats: pd.DataFrame, people: list[str],
              min_n: int = 20) -> dict:
    """Per responder and feature: P(f in reply | f in prompt) - P(f in reply | f not in prompt), with 95% CI."""
    t = _turn_table(m, feats, turns)
    prev = t.shift(1)
    pairs = t[t["prev_author"].notna()].copy()
    pp = prev.loc[pairs.index]
    pairs["month"] = pairs["start"].dt.to_period("M")
    rows = []
    for r in people[:8]:
        sel = pairs["author"] == r
        for f, label in FEATURES.items():
            had_all = pp.loc[sel, f] == 1
            res = _stratified_diff(had_all, pairs.loc[sel, f], pairs.loc[sel, "month"])
            if res is None or res["n1"] < min_n or res["n0"] < min_n:
                continue
            rows.append({"person": r, "feature": label, "mirroring": res["diff"] * 100,
                         "low": (res["diff"] - 1.96 * res["se"]) * 100, "high": (res["diff"] + 1.96 * res["se"]) * 100,
                         "when other used it (%)": res["p1"] * 100, "otherwise (%)": res["p0"] * 100, "n": res["n1"]})
    table = pd.DataFrame(rows)
    summary = table.groupby("person")["mirroring"].mean().rename("average mirroring").to_frame() if len(table) else pd.DataFrame()

    # echo: reusing an emoji / a content word from the message being answered
    def emojis(s):
        return set(EMOJI_RE.findall(s))

    def words(s):
        return {k for k in (key(x) for x in tokens(s)) if len(k) >= 4}

    echo_rows = []
    for r in people[:8]:
        sel = pairs[pairs["author"] == r]
        if len(sel) < min_n:
            continue
        prompts = pp.loc[sel.index, "text"]
        e_has = prompts.map(lambda s: bool(emojis(s)))
        e_reuse = [bool(emojis(a) & emojis(b)) for a, b in zip(sel["text"][e_has], prompts[e_has])]
        w_reuse = [bool(words(a) & words(b)) for a, b in zip(sel["text"], prompts)]
        lw = np.log1p(sel["words"].to_numpy(dtype=float))
        lp = np.log1p(pp.loc[sel.index, "words"].to_numpy(dtype=float))
        corr = pd.Series(lw).corr(pd.Series(lp), method="spearman") if len(lw) > 10 else np.nan
        echo_rows.append({"person": r,
                          "reuses an emoji from the message (%)": np.mean(e_reuse) * 100 if e_reuse else np.nan,
                          "reuses a word from the message (%)": np.mean(w_reuse) * 100,
                          "length matching (Spearman r)": corr})
    echo = pd.DataFrame(echo_rows).set_index("person") if echo_rows else pd.DataFrame()

    # quarterly average mirroring per person
    pairs["q"] = pairs["start"].dt.to_period("Q").dt.start_time
    qrows = []
    for (q, r), sel_idx in pairs.groupby(["q", "author"]).groups.items():
        if r not in people[:8] or len(sel_idx) < 3 * min_n:
            continue
        vals = []
        for f in FEATURES:
            res = _stratified_diff(pp.loc[sel_idx, f] == 1, pairs.loc[sel_idx, f], pairs.loc[sel_idx, "month"])
            if res is not None and res["n1"] >= 10 and res["n0"] >= 10:
                vals.append(res["diff"] * 100)
        if vals:
            qrows.append({"quarter": q, "person": r, "average mirroring": np.mean(vals)})
    return {"table": table, "summary": summary, "echo": echo, "quarterly": pd.DataFrame(qrows)}


def _stratified_diff(had: pd.Series, rep: pd.Series, strata: pd.Series, min_cell: int = 3):
    """Mantel-Haenszel style risk difference: P(rep | had) - P(rep | not had), within strata, pooled."""
    num = den = var = 0.0
    n1 = n0 = 0
    p1s = p0s = 0.0
    for _, idx in had.groupby(strata).groups.items():
        h, r = had.loc[idx], rep.loc[idx]
        a, b = int(h.sum()), int((~h).sum())
        if a < min_cell or b < min_cell:
            continue
        q1, q0 = r[h].mean(), r[~h].mean()
        w = a * b / (a + b)
        num += w * (q1 - q0)
        den += w
        var += w ** 2 * (q1 * (1 - q1) / a + q0 * (1 - q0) / b)
        n1, n0 = n1 + a, n0 + b
        p1s += q1 * a
        p0s += q0 * b
    if den == 0:
        return None
    return {"diff": num / den, "se": np.sqrt(var) / den, "n1": n1, "n0": n0,
            "p1": p1s / max(n1, 1), "p0": p0s / max(n0, 1)}
