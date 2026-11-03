"""AI pipeline tests. Ollama is replaced by a mock server; transformer tests use tiny local models and are
skipped if PyTorch / transformers are not installed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wa_analyzer import analysis as A
from wa_analyzer.loading import load_filtered

FAKE = Path(__file__).parent / "fake_chats"


@pytest.fixture(scope="module")
def chat():
    if not (FAKE / "truth.json").exists():
        from make_fake_chats import main
        main(FAKE)
    s = A.Settings()
    m = A.prepare(load_filtered(FAKE / "pair_android_24h_dmy.txt"), s)
    t = A.build_turns(m)
    c = A.build_conversations(m, t, s)
    return m, t, c


@pytest.fixture(scope="module")
def ollama_url():
    from mock_ollama import start
    srv, url = start()
    yield url
    srv.shutdown()


def hashing_embed(texts):
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=512, analyzer="char_wb", ngram_range=(2, 4), alternate_sign=False, norm="l2")
    return hv.transform(texts).toarray().astype(np.float32)


def test_ollama_client(ollama_url):
    from wa_analyzer.ai.ollama import Ollama, OllamaError
    llm = Ollama(model="qwen2.5:3b", host=ollama_url, device="cpu")
    assert llm.available()
    llm.check()
    with pytest.raises(OllamaError):
        Ollama(model="not-installed", host=ollama_url).check()
    assert not Ollama(host="http://127.0.0.1:9").available()


def test_reader_label_review_train(chat, ollama_url, tmp_path):
    from mock_ollama import Handler
    from wa_analyzer.ai import classifier as C
    from wa_analyzer.ai import labeling as L
    from wa_analyzer.ai import reader as R
    from wa_analyzer.ai.ollama import Ollama
    m, t, c = chat
    llm = Ollama(model="qwen2.5:3b", host=ollama_url, device="cpu")
    res = R.run(m, c, tmp_path, llm, per_month=2, log=lambda *_: None)
    assert len(res["conversations"]) == 12
    assert set(res["scores"]["person"]) == set(m["author"].unique())   # first names matched to full names
    assert not res["months"].empty and res["overall"]
    # cpu device -> Ollama asked to keep the model on CPU
    assert Handler.calls[-1]["options"]["num_gpu"] == 0
    # second run is served from the cache
    n_calls = len(Handler.calls)
    R.run(m, c, tmp_path, llm, per_month=2, log=lambda *_: None)
    assert len(Handler.calls) == n_calls

    labs = L.label(m, tmp_path, llm, n=250, log=lambda *_: None)
    assert len(labs) > 100 and labs["affection"].sum() > 5
    answers = iter(["", "1", "", "s"] + [""] * 50)
    acc = L.review(tmp_path, n=30, seed=0, inp=lambda _: next(answers), out=lambda *_: None)
    assert acc is not None and "LLM F1" in acc.columns
    met = C.train(tmp_path, embed_fn=hashing_embed, min_pos=10, log=lambda *_: None)
    assert (met["model F1"].dropna() > 0.5).all()
    pred, labels = C.apply_trained(tmp_path / "message_classifier.joblib", m, embed_fn=hashing_embed)
    assert pred.shape[0] == len(m) and pred.to_numpy().max() == 1


def test_ai_results_in_report(chat, ollama_url, tmp_path):
    from wa_analyzer.ai import reader as R
    from wa_analyzer.ai.ollama import Ollama
    from wa_analyzer.pipeline import Options, analyze
    from wa_analyzer.report import render_html
    from wa_analyzer.sections import build_sections
    m, t, c = chat
    R.run(m, c, tmp_path, Ollama(model="qwen2.5:3b", host=ollama_url), per_month=2, log=lambda *_: None)
    # fake sentiment + topics files in the expected format
    from wa_analyzer.ai.text import turn_texts
    tt = turn_texts(m)
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=len(tt))
    s = tt.drop(columns="text").assign(**{"sent:negative": p[:, 0], "sent:neutral": p[:, 1], "sent:positive": p[:, 2]})
    s["polarity"] = s["sent:positive"] - s["sent:negative"]
    s["sentiment"] = np.array(["negative", "neutral", "positive"])[p.argmax(1)]
    s["emotion"] = rng.choice(["joy", "sadness", "anger", "fear"], len(s))
    s.to_csv(tmp_path / "sentiment.csv", index=False)
    from wa_analyzer.ai.text import topic_documents
    docs = topic_documents(m)
    docs["topic"] = rng.integers(0, 3, len(docs))
    docs["topic_name"] = docs["topic"].map({0: "plans", 1: "love", 2: "daily life"})
    docs["x"], docs["y"] = rng.normal(size=len(docs)), rng.normal(size=len(docs))
    docs["authors"] = docs["authors"].map(json.dumps)
    docs.drop(columns="text").to_csv(tmp_path / "topic_docs.csv", index=False)
    pd.DataFrame({"topic": [0, 1, 2], "name": ["plans", "love", "daily life"], "keywords": ["a", "b", "c"],
                  "size": [1, 2, 3], "example": ["x", "y", "z"]}).to_csv(tmp_path / "topics.csv", index=False)
    r = analyze(load_filtered(FAKE / "pair_android_24h_dmy.txt"), "t", A.Settings(),
                Options(stats=False, ai_dir=str(tmp_path)))
    assert {"sentiment", "topics", "reader_conversations"} <= set(r.ai)
    html = render_html(r, build_sections(r))
    for title in ("Tone and emotions", "What the local LLM sees", "What you talk about"):
        assert title in html


def _tiny_models(tmp: Path):
    """Build tiny random BERT classifiers + a tiny sentence-transformer locally (no download)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyzçğıöşü0123456789"):
        vocab[ch] = 5 + i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")
    paths = {}
    for name, labels in [("sent", ["negative", "neutral", "positive"]), ("tr", ["negatif", "pozitif"]),
                         ("emo", ["anger", "fear", "joy", "sadness"])]:
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, num_labels=len(labels), id2label=dict(enumerate(labels)),
                         label2id={l: i for i, l in enumerate(labels)}, max_position_embeddings=512)
        mdl = BertForSequenceClassification(cfg)
        d = tmp / name
        mdl.save_pretrained(d)
        fast.save_pretrained(d)
        paths[name] = str(d)
    from sentence_transformers import SentenceTransformer, models as stm
    from transformers import BertModel
    d = tmp / "embed"
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=512)).save_pretrained(d / "bert")
    fast.save_pretrained(d / "bert")
    word = stm.Transformer(str(d / "bert"), max_seq_length=128)
    pool = stm.Pooling(word.get_word_embedding_dimension())
    SentenceTransformer(modules=[word, pool]).save(str(d / "st"))
    paths["embed"] = str(d / "st")
    return paths


def test_sentiment_and_topics_with_tiny_models(chat, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("sentence_transformers")
    from wa_analyzer.ai import sentiment as S
    from wa_analyzer.ai import topics as T
    m, t, c = chat
    paths = _tiny_models(tmp_path / "models")
    res = S.run(m, tmp_path / "ai", device="cpu", sentiment_model=paths["sent"], turkish_model=paths["tr"],
                emotion_model=paths["emo"], max_turns=400, log=lambda *_: None)
    assert len(res) == 400 and res["polarity"].between(-1, 1).all()
    assert {"sent:positive", "tr:positive", "emo:joy"} <= set(res.columns)
    assert res["tr:positive"].notna().sum() > 0          # Turkish turns went through the Turkish model
    out = T.run(m, tmp_path / "ai", device="cpu", n_topics=4, embed_model=paths["embed"], log=lambda *_: None)
    assert len(out["topics"]) == 4
