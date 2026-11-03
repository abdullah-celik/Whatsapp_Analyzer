"""Topic modelling: what do you talk about, and how does it change over time?

1. Conversations (long ones split into chunks) are turned into vectors with a multilingual sentence-embedding
   model (paraphrase-multilingual-MiniLM-L12-v2: Turkish, Italian, English and 50+ more languages).
2. The vectors are clustered with k-means (number of topics chosen automatically by silhouette score).
3. Each topic gets keywords via class-based TF-IDF (the words that set it apart from the other topics),
   and optionally a short name written by your local LLM (Ollama).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..lexicon_engine import key, tokens
from . import device as D
from .text import topic_documents

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

STOPWORDS = {key(w) for w in (
    # Turkish
    "ve bir bu şu o ne çok da de mi mı mu mü ki ya ama için gibi daha en ben sen biz siz onlar bana sana beni seni "
    "benim senin var yok değil olarak olan oldu olur şey bi yani işte zaten hani her hiç kadar sonra önce şimdi "
    "bugün yarın evet hayır tamam tmm ok okey bunu şunu onu böyle şöyle öyle nasıl neden niye nerede ile ise "
    "artık hep biraz sadece bile ben de sende bende hadi aynen peki ha he hı yaa ya da ama ki mi "
    # English
    "the a an and or but if so to of in on at for with is are was were be been it this that you i me my your we "
    "they he she him her them not no yes ok okay just like do does did have has had will would can could what "
    "when where how why who there here then than too very also im dont its get got go going "
    # Italian
    "il lo la i gli le un una uno di da in con su per tra fra e ed o ma che chi non si sì no io tu lui lei noi "
    "voi loro mi ti ci vi è sono era ho hai ha abbiamo anche come dove quando perché perche più piu molto poi "
    "ok va bene però pero allora cosa questo questa quello quella del della dei delle al alla nel nella"
).split()}


def embed(texts: list[str], model_name: str, device: str, log=print) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()
    t0 = time.time()
    E = model.encode(texts, batch_size=D.batch_size(device, "embed"), show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=True)
    log(f"  embedded {len(texts):,} texts in {time.time() - t0:.0f}s")
    return E.astype(np.float32)


def choose_k(E: np.ndarray, k_min: int = 5, k_max: int = 18, seed: int = 0) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = len(E)
    k_max = max(k_min, min(k_max, n // 15))
    if n < 3 * k_min:
        return max(2, n // 5)
    rng = np.random.default_rng(seed)
    sample = E[rng.choice(n, size=min(n, 3000), replace=False)]
    best, best_s = k_min, -1
    for k in range(k_min, k_max + 1):
        lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(sample)
        s = silhouette_score(sample, lab, metric="cosine")
        if s > best_s:
            best, best_s = k, s
    return best


def ctfidf_keywords(docs: pd.DataFrame, labels: np.ndarray, top: int = 8) -> dict[int, list[str]]:
    counts: dict[int, Counter] = {}
    surface: dict[str, Counter] = {}
    for text, lab in zip(docs["text"], labels):
        c = counts.setdefault(int(lab), Counter())
        for raw in tokens(text):
            k = key(raw)
            if len(k) < 3 or k in STOPWORDS or k.isdigit():
                continue
            c[k] += 1
            surface.setdefault(k, Counter())[raw] += 1
    total_words = sum(sum(c.values()) for c in counts.values()) or 1
    avg = total_words / max(len(counts), 1)
    freq = Counter()
    for c in counts.values():
        freq.update(c)
    out = {}
    for lab, c in counts.items():
        n = sum(c.values()) or 1
        scores = {w: (cnt / n) * math.log(1 + avg / freq[w]) for w, cnt in c.items() if cnt >= 2}
        best = sorted(scores, key=scores.get, reverse=True)[:top]
        out[lab] = [surface[w].most_common(1)[0][0] for w in best]
    return out


def run(m: pd.DataFrame, out_dir: str | Path, device: str = "auto", n_topics: int | None = None,
        embed_model: str = EMBED_MODEL, ollama=None, seed: int = 0, log=print) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = D.resolve(device)
    log(f"Device: {D.describe(dev)}")
    docs = topic_documents(m)
    if len(docs) < 30:
        raise ValueError(f"Only {len(docs)} conversation chunks with enough text: too little for topic modelling.")
    log(f"{len(docs):,} conversation chunks. Loading {embed_model} ...")
    E = embed(docs["text"].tolist(), embed_model, dev, log)
    k = n_topics or choose_k(E, seed=seed)
    log(f"Clustering into {k} topics ...")
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(E)
    kw = ctfidf_keywords(docs, labels)
    xy = PCA(n_components=2, random_state=seed).fit_transform(E)

    topics = []
    for t in range(k):
        sel = labels == t
        centroid = E[sel].mean(axis=0)
        sims = E[sel] @ centroid
        examples = docs.loc[sel, "text"].iloc[np.argsort(-sims)[:3]].tolist()
        name = ", ".join(kw.get(t, [])[:3]) or f"topic {t + 1}"
        if ollama is not None:
            try:
                name = ollama.name_topic(kw.get(t, []), examples) or name
            except Exception as e:  # naming is optional
                log(f"  (topic naming with Ollama failed: {e})")
        topics.append({"topic": t, "name": name, "keywords": ", ".join(kw.get(t, [])), "size": int(sel.sum()),
                       "example": examples[0][:300] if examples else ""})
    tdf = pd.DataFrame(topics).sort_values("size", ascending=False)
    docs = docs.assign(topic=labels, x=xy[:, 0], y=xy[:, 1])
    docs["topic_name"] = docs["topic"].map(tdf.set_index("topic")["name"])
    docs["authors"] = docs["authors"].map(json.dumps)
    docs.drop(columns=["text"]).to_csv(out_dir / "topic_docs.csv", index=False)
    tdf.to_csv(out_dir / "topics.csv", index=False)
    (out_dir / "topics_meta.json").write_text(json.dumps({"embed_model": embed_model, "k": int(k), "device": dev,
                                                          "docs": len(docs), "created": pd.Timestamp.now().isoformat()}, indent=1))
    log(f"Saved {k} topics to {out_dir / 'topics.csv'}")
    return {"topics": tdf, "docs": docs}
