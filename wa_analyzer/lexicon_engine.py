"""Word classifier engine: text normalisation, automatic spelling variants and matching.

Every word in the text and every lexicon entry is turned into a *key* the same way:
  1. lower-case (Turkish İ handled), apostrophes removed      n'aber   -> naber
  2. accents / Turkish letters folded to plain ASCII           aşkım    -> askim,  perché -> perche
  3. w -> v                                                    sewiyorum -> seviyorum
  4. repeated letters collapsed                                canımmmm -> canim, tamaaaam -> tamam
So "AŞKIMMM", "askimm" and "aşkım" are all the same key.

For Turkish lexicons, every entry additionally gets automatic variants (see turkish_variants):
  * ğ dropped (bebeğim -> bebeim), ş written as j (aşkım -> ajkım)
  * spoken verb endings: seviyorum -> seviyom / seviom / sevyom / seviyorm / seviyrm ...,
    yapıyorsun -> yapıyon / yapıyosun, geliyor -> geliyo / gelio,
    geleceğim -> gelicem / gelcem / gelecem, arayacağım -> arıcam / aracam
  * vowel dropping: canım -> canm / cnm, seviyorum -> svyrm, özledim -> özledm / özldm
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_LEXICONS = [HERE / "lexicon.json", HERE / "lexicon_tr.json"]

TOKEN_RE = re.compile(r"<3|[\w']+", re.U)
VOWELS = set("aeiou")
_FOLD = str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ç": "c", "ö": "o", "ü": "u", "â": "a", "î": "i", "û": "u",
                       "ß": "ss", "æ": "ae", "ø": "o", "œ": "oe", "w": "v"})


def lower(text: str) -> str:
    t = unicodedata.normalize("NFC", text).replace("İ", "i").replace("I", "ı" if False else "i")
    return t.casefold().replace("̇", "")


def squeeze(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s)


def fold(s: str) -> str:
    s = s.translate(_FOLD)
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


@lru_cache(maxsize=200_000)
def key(word: str) -> str:
    """Matching key for a single word."""
    w = lower(word).replace("'", "").replace("’", "")
    return squeeze(fold(w))


@lru_cache(maxsize=200_000)
def exact_key(word: str) -> str:
    """Key for '=' entries: only lower-cased (no folding, no squeezing)."""
    return lower(word).replace("'", "").replace("’", "")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(lower(text).replace("’", "'"))


# ------------------------------------------------------------------ Turkish variants

def _drop_last_vowel(w: str) -> str | None:
    idx = [i for i, c in enumerate(w) if c in VOWELS and i > 0]
    if not idx or len(w) < 4:
        return None
    i = idx[-1]
    return w[:i] + w[i + 1:]


def _skeleton(w: str) -> str | None:
    if len(w) < 4:
        return None
    s = w[0] + "".join(c for c in w[1:] if c not in VOWELS)
    return s if len(s) >= 3 and s != w else None


def _verb_variants(w: str) -> set[str]:
    """Spoken/texting forms of Turkish verb endings. `w` is already folded (ı->i, ü->u ...)."""
    out = set()
    m = re.match(r"^(.*?)([aeiou]?)yorum$", w)          # seviyorum
    if m and m.group(1):
        st, v = m.groups()
        out |= {st + v + "yom", st + v + "om", st + "yom", st + v + "yorm", st + v + "yrm", st + "yrm",
                st + v + "orum", st + "yorum", st + v + "yom"}
    m = re.match(r"^(.*?)([aeiou]?)yorsun$", w)         # yapıyorsun
    if m and m.group(1):
        st, v = m.groups()
        out |= {st + v + "yon", st + v + "on", st + v + "yosun", st + v + "osun", st + v + "yosn", st + v + "yorsn",
                st + "yon", st + "yosun"}
    m = re.match(r"^(.*?)([aeiou]?)yor$", w)            # geliyor
    if m and m.group(1):
        st, v = m.groups()
        out |= {st + v + "yo", st + v + "o", st + "yo"}
    m = re.match(r"^(.*?)([aeiou]?)yoruz$", w)          # geliyoruz
    if m and m.group(1):
        st, v = m.groups()
        out |= {st + v + "yoz", st + v + "oz", st + v + "yruz"}
    m = re.match(r"^(.*?)(y?)([ea])c[ea]g?(im|iz|sin|k)$", w)   # geleceğim / arayacağız / geleceksin
    if m and m.group(1):
        st, y, vow, end = m.groups()
        st2 = st[:-1] if y and len(st) > 2 else st      # özle-yeceğim -> özl-icem
        h = vow
        tail = {"im": "m", "iz": "z", "sin": "n", "k": "k"}[end]
        out |= {st2 + "ic" + h + tail, st + "c" + h + tail, st + y + vow + "c" + h + tail,
                st2 + "c" + h + tail, st + y + "c" + h + tail}
    return {o for o in out if len(o) >= 3}


def turkish_variants(entry: str) -> set[str]:
    """All spellings (as keys) generated for one Turkish entry (word or phrase)."""
    base = lower(entry)
    pre = {base, base.replace("ğ", ""), base.replace("ş", "j"), base.replace("ğ", "").replace("ş", "j")}
    words_variants = []
    for p in pre:
        words = [squeeze(fold(x.replace("'", ""))) for x in p.split()]
        words_variants.append(words)
    out = set()
    for words in words_variants:
        per_word = []
        for i, w in enumerate(words):
            vs = {w} | _verb_variants(w)
            d = _drop_last_vowel(w)
            if d:
                vs.add(d)
            s = _skeleton(w)
            if s:
                vs.add(s)
            per_word.append({squeeze(v) for v in vs})
        if len(words) == 1:
            out |= per_word[0]
        else:
            # phrase: original words, every word changed in the same way, and written together
            out.add(" ".join(words))
            last_variants = per_word[-1]
            for lv in last_variants:
                out.add(" ".join(words[:-1] + [lv]))
            skel = [(_skeleton(w) or w) for w in words]
            out.add(" ".join(skel))
            out.add(squeeze("".join(words)))
            out.add(squeeze("".join(skel)))
    return {o for o in out if o}


# ------------------------------------------------------------------ lexicon

class Lexicon:
    """Merged word lists from one or more JSON files."""

    def __init__(self, paths=None):
        paths = [Path(p) for p in (paths or DEFAULT_LEXICONS)]
        self.categories: dict[str, dict] = {}
        self.never: set[str] = set()
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            turkish = data.get("language") == "tr"
            self.never |= {key(w) for w in data.get("never_match", [])}
            for name, cat in data["categories"].items():
                c = self.categories.setdefault(name, {
                    "label": cat.get("label", name), "interest": cat.get("interest", 0),
                    "whole": bool(cat.get("whole_message")), "words": set(), "exact": set(),
                    "prefixes": set(), "phrases": set(), "emojis": set(), "except": set(),
                    "sources": {},
                })
                c["whole"] = c["whole"] or bool(cat.get("whole_message"))
                c["emojis"] |= set(cat.get("emojis", []))
                for ph in cat.get("except", []):
                    c["except"].add(" ".join(key(w.rstrip("*")) + ("*" if w.endswith("*") else "") for w in ph.split()))
                for entry in cat.get("words", []):
                    self._add(c, entry, turkish)
        for c in self.categories.values():
            c["words"] -= self.never
            c["phrases"] -= self.never
            c["prefixes"] = tuple(sorted(c["prefixes"]))
            c["phrase_re"] = _compile_phrases(c["phrases"])
            c["except_re"] = _compile_phrases(c["except"])

    def _add(self, c, entry: str, turkish: bool):
        entry = entry.strip()
        if not entry:
            return
        if entry.startswith("="):
            c["exact"].add(exact_key(entry[1:]))
            c["sources"].setdefault(exact_key(entry[1:]), entry)
            return
        if entry.endswith("*") and " " not in entry:
            stem = entry[:-1]
            stems = {key(stem)}
            if turkish:
                stems |= {key(lower(stem).replace("ğ", "")), key(lower(stem).replace("ş", "j"))}
            c["prefixes"] |= {s for s in stems if len(s) >= 3}
            return
        variants = turkish_variants(entry) if turkish else {" ".join(key(w) for w in entry.split())}
        if c["whole"]:
            c["words"] |= variants
            for v in variants:
                c["sources"].setdefault(v, entry)
            return
        for v in variants:
            (c["phrases"] if " " in v else c["words"]).add(v)
            c["sources"].setdefault(v, entry)

    @property
    def labels(self) -> dict[str, str]:
        return {k: v["label"] for k, v in self.categories.items()}


def _compile_phrases(phrases) -> re.Pattern | None:
    if not phrases:
        return None
    parts = []
    for ph in sorted(phrases, key=len, reverse=True):
        words = ph.split()
        parts.append(r"\s".join(re.escape(w[:-1]) + r"\w*" if w.endswith("*") else re.escape(w) for w in words))
    return re.compile(r"(?<!\S)(?:" + "|".join(parts) + r")(?!\S)")


LAUGH_RE = re.compile(
    r"^(?:a?h?(?:ha|ah){2,}h?|(?:he){2,}h?|(?:hi){2,}h?|j?(?:ja|aj){2,}|(?:je){2,}|(?:ji){2,}|(?:xd)+|k{3,}"
    r"|(?:ks|sk|sj|js|hs|sh|jd|dj|kd|dk|jk|kj|hj|jh){2,}[a-z]?)$")
HOME_ROW = set("asdfghjklş")


def is_laugh_token(t: str, raw: str) -> bool:
    if LAUGH_RE.match(t) or LAUGH_RE.match(raw):
        return True
    # keyboard smash typed as laughter: "ajsjsjsj", "asdfghjkl", "skdjsjdk", "ahsjdjdjs"
    if len(raw) >= 6 and raw.isalpha() and set(raw) <= HOME_ROW:
        vowels = sum(ch in "aeiou" for ch in raw)
        bigrams = [raw[i:i + 2] for i in range(len(raw) - 1)]
        repeated = len(bigrams) - len(set(bigrams)) >= 2
        if vowels / len(raw) <= 0.2 and ("j" in raw or repeated or raw.startswith("asdf")):
            return True
    return False


TR_QUESTION_RE = re.compile(r"(?:^|\s)m[ıiuü](?:y[ıiuü]m|s[ıiuü]n|y[ıiuü]z|s[ıiuü]n[ıiuü]z|yd[ıiuü]|yd[ıiuü]n|yd[ıiuü]m|yken)?(?=\s|$|[?!.,])")
TR_QUESTION_ATTACHED = re.compile(r"\w{2,}m[ıiuü](?:s[ıiuü]n|y[ıiuü]m|y[ıiuü]z|s[ıiuü]n[ıiuü]z)\b")


def looks_like_question(text: str) -> bool:
    """'?' in any script, or a Turkish question particle (mı/mi/mu/mü, misin, musun ...)."""
    if re.search(r"[?¿؟？]", text):
        return True
    t = lower(text)
    m = TR_QUESTION_RE.search(t)
    if m:
        particle = m.group(0).strip()
        # bare 'mi' is also Italian ("mi manchi"): require Turkish letters elsewhere in the message
        if particle != "mi" or re.search(r"[ışğçöü]", t):
            return True
    if TR_QUESTION_ATTACHED.search(t):
        return True
    ks = [key(x) for x in tokens(text)]
    if ks and ks[0] in TR_QUESTION_FIRST:
        return True
    if ks and ks[0] == "ne" and any(k.endswith(_NE_VERB_ENDINGS) for k in ks[1:3]):
        return True  # "ne yapıyosun", "ne dedin", "ne zaman gelcen"
    return any(k in TR_QUESTION_WORDS for k in ks)


# Turkish question words that start a question, and question forms that are questions anywhere
TR_QUESTION_FIRST = {key(w) for w in
                     "neden niye niçin nasıl nerede nerde nereye nereden nerden kim kime kimi kimle kiminle "
                     "hangi hangisi kaç kaçta kaçıncı".split()}
_NE_VERB_ENDINGS = ("yon", "yosun", "yorsun", "yosn", "cen", "can", "caksin", "ceksin", "din", "dun", "tin", "tun",
                    "zaman")
TR_QUESTION_WORDS = {key(w) for w in
                     "nasılsın nasılsınız naslsn nslsn naber nbr nabr napıyon napiyon napıyosun napıyorsun "
                     "napion naptın nörüyon neredesin nerdesin nrd nrdsn nzmn noldu nooldu".split()}


class WordClassifier:
    """Marks, per message, which categories it contains (0/1 per category)."""

    def __init__(self, lexicon_paths=None):
        if isinstance(lexicon_paths, (str, Path)):
            lexicon_paths = [lexicon_paths]
        self.lex = Lexicon(lexicon_paths)
        self.categories = self.lex.categories

    @property
    def labels(self):
        return self.lex.labels

    def _match(self, text: str, explain: bool = False):
        raw = tokens(text)
        keys = [key(t) for t in raw]
        exacts = {exact_key(t) for t in raw}
        joined = " ".join(keys)
        whole = " ".join(k for k in keys if k)
        res, why = {}, {}
        for name, c in self.categories.items():
            found = []
            if c["whole"]:
                if any(is_laugh_token(k, r) for k, r in zip(keys, raw)):
                    pass
                elif whole and (whole in c["words"] or whole in c["exact"] or " ".join(exact_key(t) for t in raw) in c["exact"]):
                    found.append(c["sources"].get(whole, whole))
                elif not whole and c["emojis"] and len(text.strip()) <= 4 and any(e in text for e in c["emojis"]):
                    found.append(text.strip())
            else:
                txt = c["except_re"].sub(" ", joined) if c["except_re"] else joined
                ks = txt.split()
                for k in ks:
                    if k in self.lex.never:
                        continue
                    if k in c["words"]:
                        found.append(c["sources"].get(k, k))
                    elif c["prefixes"] and k.startswith(c["prefixes"]):
                        found.append(k)
                found += [c["sources"].get(e, e) for e in exacts & c["exact"]]
                if c["phrase_re"]:
                    for m in c["phrase_re"].finditer(txt):
                        found.append(c["sources"].get(m.group(0), m.group(0)))
                found += [e for e in c["emojis"] if e in text]
                if name == "laughter":
                    found += [r for r, k in zip(raw, keys) if is_laugh_token(k, r) and r not in found]
            res[name] = 1 if found else 0
            if found:
                why[name] = found
        return (res, why) if explain else res

    def classify(self, text: str) -> dict[str, int]:
        return self._match(text)

    def explain(self, text: str) -> dict[str, list[str]]:
        return self._match(text, explain=True)[1]

    def apply(self, m):
        """DataFrame (one row per message) with a 0/1 column per category."""
        import pandas as pd
        names = list(self.categories)
        rows = [self.classify(t) if k == "text" else dict.fromkeys(names, 0) for t, k in zip(m["text"], m["kind"])]
        return pd.DataFrame(rows, index=m.index, columns=names)
