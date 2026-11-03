from wa_analyzer import analysis as A
from wa_analyzer.parser import parse_text

RAW = """01/03/2025, 10:00 - Ann: hi, are you there?
01/03/2025, 10:45 - Ann: hello??
01/03/2025, 10:50 - Bob: sorry, busy
01/03/2025, 10:52 - Ann: ok
01/03/2025, 20:00 - Ann: good night?
02/03/2025, 08:00 - Bob: morning! just saw this
02/03/2025, 08:01 - Ann: :)
03/03/2025, 09:00 - Ann: you there
04/03/2025, 09:00 - Ann: hello again
"""


def test_conversation_logic():
    s = A.Settings()
    m = A.prepare(parse_text(RAW), s)
    t = A.build_turns(m)
    c = A.build_conversations(m, t, s)
    p = A.people_summary(m, t, c, s)
    assert len(c) == 5
    # Bob's morning message answers Ann's unanswered question -> late reply, not a new start
    assert p.loc["Bob", "late_replies"] == 1 and p.loc["Bob", "conversations_started"] == 0
    assert p.loc["Ann", "conversations_started"] == 4
    # Ann wrote again after 45 min without answer + started again after an unanswered day
    assert p.loc["Ann", "followups_without_reply"] == 2
    assert p.loc["Ann", "left_without_answer"] == 3
    assert p.loc["Bob", "median_response_min"] == 5
    assert p.loc["Ann", "median_response_min"] == 1.5


def test_daily_streaks():
    m = A.prepare(parse_text(RAW), A.Settings())
    # both wrote on 1 + 2 March; only Ann on 3 + 4 March
    both = A.daily_streaks(m, min_people=2)
    assert both.longest["days"] == 2
    assert both.longest["start"].day == 1 and both.longest["end"].day == 2
    assert both.current == 0  # Bob did not write on the last day
    anyone = A.daily_streaks(m, min_people=1)
    assert anyone.longest["days"] == 4 and anyone.current == 4
    assert anyone.span_days == 4
    assert both.per_person.loc["Ann", "longest streak (days)"] == 4
    assert both.per_person.loc["Bob", "longest streak (days)"] == 2


def test_daily_streaks_gap():
    raw = """01/03/2025, 10:00 - Ann: hi
01/03/2025, 10:01 - Bob: hi
02/03/2025, 23:59 - Ann: late
03/03/2025, 00:01 - Bob: early
05/03/2025, 10:00 - Ann: back
05/03/2025, 10:05 - Bob: yes
"""
    st = A.daily_streaks(A.prepare(parse_text(raw), A.Settings()), 2)
    # 2 March only Ann, 3 March only Bob -> neither counts; 4 March silent
    assert list(st.runs["days"]) == [1, 1]
    assert st.current == 1
