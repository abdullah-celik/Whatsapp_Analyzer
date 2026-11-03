"""Group chats as a network: who replies to whom.

Nodes = people, directed edge B -> A weighted by how often B replies to A.
  * attention received (weighted in-degree): how often others answer you
  * PageRank: being answered by people who are themselves answered a lot
  * betweenness: bridging people between sub-groups
  * reciprocity: how balanced each pair's replies are
  * communities (Louvain): sub-groups that mostly talk to each other
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NX_OK = True
try:
    import networkx as nx
except Exception:  # pragma: no cover
    NX_OK = False


def reply_graph(turns: pd.DataFrame, people: list[str] | None = None):
    r = turns[turns["prev_author"].notna() & (turns["prev_author"] != turns["author"])]
    if people is not None:
        r = r[r["author"].isin(people) & r["prev_author"].isin(people)]
    w = r.groupby(["author", "prev_author"]).size()
    G = nx.DiGraph()
    G.add_nodes_from(people or sorted(set(r["author"]) | set(r["prev_author"])))
    for (a, b), n in w.items():
        G.add_edge(a, b, weight=int(n))      # a replies to b
    return G


def analyse(turns: pd.DataFrame, m: pd.DataFrame, people: list[str], max_people: int = 30) -> dict:
    if not NX_OK:
        raise ImportError("pip install networkx")
    ppl = people[:max_people]
    G = reply_graph(turns, ppl)
    # attention flows towards the person being replied to: reverse edges for PageRank
    R = G.reverse(copy=True)
    pr = nx.pagerank(R, weight="weight") if G.number_of_edges() else {p: 1 / len(ppl) for p in ppl}
    D = nx.DiGraph()
    for a, b, d in G.edges(data=True):
        D.add_edge(a, b, dist=1.0 / d["weight"])
    btw = nx.betweenness_centrality(D, weight="dist") if D.number_of_edges() else {}
    U = nx.Graph()
    U.add_nodes_from(ppl)
    for a, b, d in G.edges(data=True):
        w = U[a][b]["weight"] + d["weight"] if U.has_edge(a, b) else d["weight"]
        U.add_edge(a, b, weight=w)
    try:
        comms = nx.community.louvain_communities(U, weight="weight", seed=0) if U.number_of_edges() else [set(ppl)]
    except Exception:
        comms = [set(ppl)]
    comm_of = {p: i + 1 for i, c in enumerate(sorted(comms, key=len, reverse=True)) for p in c}

    msgs = m["author"].value_counts()
    rows = []
    for p in ppl:
        rows.append({"person": p, "messages": int(msgs.get(p, 0)),
                     "replies received": int(G.in_degree(p, weight="weight")),
                     "replies given": int(G.out_degree(p, weight="weight")),
                     "PageRank (attention)": pr.get(p, 0.0), "betweenness (bridge)": btw.get(p, 0.0),
                     "sub-group": comm_of.get(p, 0)})
    nodes = pd.DataFrame(rows).set_index("person")
    nodes["answered per message"] = nodes["replies received"] / nodes["messages"].replace(0, np.nan)

    pair_rows = []
    for i, a in enumerate(ppl):
        for b in ppl[i + 1:]:
            ab = G[a][b]["weight"] if G.has_edge(a, b) else 0
            ba = G[b][a]["weight"] if G.has_edge(b, a) else 0
            if ab + ba >= 5:
                pair_rows.append({"pair": f"{a} ↔ {b}", f"replies": ab + ba,
                                  "balance": min(ab, ba) / max(ab, ba) if max(ab, ba) else np.nan,
                                  "more active replier": a if ab > ba else b if ba > ab else "equal"})
    pairs = pd.DataFrame(pair_rows).sort_values("replies", ascending=False) if pair_rows else pd.DataFrame()

    # attention over time: share of all replies received, per month
    r = turns[turns["prev_author"].notna() & turns["prev_author"].isin(ppl)].copy()
    r["month"] = r["start"].dt.to_period("M").dt.to_timestamp()
    att = pd.crosstab(r["month"], r["prev_author"], normalize="index") * 100
    pos = nx.spring_layout(U, weight="weight", seed=1, k=1.5 / np.sqrt(max(len(ppl), 1)))
    return {"graph": G, "nodes": nodes, "pairs": pairs, "attention_monthly": att, "positions": pos,
            "n_communities": len(comms)}
