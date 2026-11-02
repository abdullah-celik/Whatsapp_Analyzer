import json
from pathlib import Path

import pandas as pd
import pytest

from wa_analyzer.parser import load_chat, parse_text

FAKE = Path(__file__).parent / "fake_chats"
FORMATS = ["android_24h_dmy", "android_12h_mdy", "android_dotted", "ios_24h_dmy", "ios_12h_mdy", "iso"]


@pytest.fixture(scope="session")
def truth():
    if not (FAKE / "truth.json").exists():
        from make_fake_chats import main
        main(FAKE)
    return json.loads((FAKE / "truth.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("chat", ["pair", "group"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_formats_match_truth(truth, chat, fmt):
    df = load_chat(FAKE / f"{chat}_{fmt}.txt")
    people = df[df["kind"] != "system"].reset_index(drop=True)
    exp = truth[chat]
    assert len(people) == len(exp)
    assert (df["kind"] == "system").sum() >= 1
    for (ts, author, text, kind), (_, row) in zip(exp, people.iterrows()):
        ts = pd.Timestamp(ts)
        assert row["author"] == author
        assert abs((row["timestamp"] - ts).total_seconds()) < 60
        if kind == "text":
            assert row["kind"] == "text"
            assert row["text"] == text
        else:
            assert row["kind"] != "text"


def test_zip(truth):
    df = load_chat(FAKE / "pair_ios_export.zip")
    assert (df["kind"] != "system").sum() == len(truth["pair"])


def test_ambiguous_dates_resolved_by_order():
    raw = "1/2/24, 10:00 - A: hi\n1/3/24, 10:00 - B: hey\n2/3/24, 10:00 - A: yo\n"
    # DMY: 1 Feb, 1 Mar, 2 Mar (increasing) ; MDY: Jan 2, Jan 3, Feb 3 (also increasing) -> both fine
    df = parse_text(raw)
    assert df["timestamp"].is_monotonic_increasing


def test_multiline_and_edit():
    raw = "[05/01/24, 10:00:00] A: line one\nline two\n[05/01/24, 10:01:00] B: fixed <This message was edited>\n"
    df = parse_text(raw)
    assert df.loc[0, "text"] == "line one\nline two"
    assert df.loc[1, "text"] == "fixed" and df.loc[1, "edited"]
