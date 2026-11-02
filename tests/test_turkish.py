"""Turkish slang, misspellings and false-positive checks for the word classifier."""
import pytest

from wa_analyzer.lexicon_engine import WordClassifier, looks_like_question

clf = WordClassifier()

POSITIVE = [
    # affection: proper, no Turkish letters, stretched, abbreviated, spoken forms
    ("seni seviyorum", "affection"), ("SENİ SEVİYORUMMM", "affection"), ("seni seviom", "affection"),
    ("seviyom seni", "affection"), ("sevyom", "affection"), ("sni svyrm", "affection"), ("sçs", "affection"),
    ("aşkım", "affection"), ("askimmm", "affection"), ("aşkm", "affection"), ("ajkım", "affection"),
    ("canım", "affection"), ("cnm", "affection"), ("canimmm", "affection"), ("bebeim", "affection"),
    ("özledim", "affection"), ("ozledimmm", "affection"), ("özledm", "affection"), ("özlicem seni", "affection"),
    ("muck", "affection"), ("mujk", "affection"), ("öpüyorum", "affection"), ("iyiki varsın", "affection"),
    ("hayatım", "affection"), ("birtanem", "affection"), ("sarılmak istiyorum", "affection"),
    # asking about the other
    ("nbr", "care_questions"), ("naber", "care_questions"), ("nslsn", "care_questions"),
    ("nasilsin", "care_questions"), ("napıyon", "care_questions"), ("napiyosun", "care_questions"),
    ("ne yapıyosun", "care_questions"), ("iyimisin", "care_questions"), ("nerdesin", "care_questions"),
    ("eve vardın mı", "care_questions"), ("yemek yedin mi", "care_questions"), ("kib", "care_questions"),
    # plans
    ("buluşalım mı", "plans"), ("bulusalim", "plans"), ("müsaitmisin", "plans"), ("gelcen mi", "plans"),
    ("görüşürüz", "plans"), ("grşz", "plans"), ("arıcam", "plans"), ("kahve içelim", "plans"),
    # compliments / enthusiasm
    ("çok güzelsin", "compliments"), ("guzelsin", "compliments"), ("tatlısın", "compliments"),
    ("yakışıklı", "compliments"), ("harikasın", "compliments"), ("maşallah", "enthusiasm"), ("efsane", "enthusiasm"),
    # thanks / apology
    ("tşk", "gratitude"), ("tşkler", "gratitude"), ("tesekkurler", "gratitude"), ("saol", "gratitude"),
    ("eyw", "gratitude"), ("eline sağlık", "gratitude"),
    ("kusura bakma", "apology"), ("özür dilerim", "apology"), ("ozur dilerim", "apology"), ("pardon", "apology"),
    # laughter
    ("hahahaha", "laughter"), ("ajsjsjsj", "laughter"), ("sjsjsj", "laughter"), ("ksksksk", "laughter"),
    ("asdfghjkl", "laughter"), ("skdjsjdk", "laughter"), ("koptum", "laughter"), ("öldüm gülmekten", "laughter"),
    ("😂😂", "laughter"),
    # good morning / night
    ("günaydın", "good_morning_night"), ("gunaydinnn", "good_morning_night"), ("iyi geceler", "good_morning_night"),
    ("iyi gclr", "good_morning_night"), ("tatlı rüyalar", "good_morning_night"),
    # negative
    ("boşver", "negative"), ("bosver", "negative"), ("yoruldum", "negative"), ("meşgulüm", "negative"),
    ("bakarız", "negative"), ("rahat bırak beni", "negative"),
    # dry replies (only when the whole message is this)
    ("tmm", "dry_reply"), ("tamaaam", "dry_reply"), ("ok", "dry_reply"), ("peki", "dry_reply"),
    ("hı", "dry_reply"), ("aynen", "dry_reply"), ("👍", "dry_reply"),
]


@pytest.mark.parametrize("text,cat", POSITIVE)
def test_positive(text, cat):
    assert clf.classify(text)[cat] == 1, (text, clf.explain(text))


NEGATIVE = [
    # ordinary words that must not trigger anything (or not the given category)
    ("seviye", "affection"), ("sevinç", "affection"), ("Özlem geldi mi", "affection"),
    ("canım sıkılıyor", "affection"), ("canım yandı", "affection"), ("can I ask you", "affection"),
    ("kış geldi", "affection"), ("hakan geldi", "laughter"), ("small glass", "laughter"),
    ("başta zordu", "negative"), ("bela", "compliments"), ("afet", "apology"),
    ("tamam ama yarın konuşalım mı", "dry_reply"), ("kkkkk", "dry_reply"),
    ("the way you", "enthusiasm"), ("bugün", "plans"), ("özürlü", "apology"),
]


@pytest.mark.parametrize("text,cat", NEGATIVE)
def test_no_false_positive(text, cat):
    assert clf.classify(text)[cat] == 0, (text, clf.explain(text))


@pytest.mark.parametrize("text,expected", [
    ("iyi misin", True), ("geliyor musun", True), ("nasılsın", True), ("ne yapıyosun", True),
    ("neden gelmedin", True), ("tamam mı", True), ("iyimisin", True),
    ("ne güzel", False), ("mi manchi", False), ("tamam", False), ("bugün geldim", False),
])
def test_turkish_questions(text, expected):
    assert looks_like_question(text) == expected
