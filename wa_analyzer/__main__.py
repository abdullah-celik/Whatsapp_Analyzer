"""Command line:  python -m wa_analyzer CHAT.txt|CHAT.zip [-o report.html] [options]"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .analysis import Settings
from .loading import default_ai_dir, load_filtered
from .pipeline import Options
from .report import build_report


def main(argv=None):
    p = argparse.ArgumentParser(prog="wa_analyzer", description="Analyse a WhatsApp chat export locally.")
    p.add_argument("chat", nargs="?", help="exported chat: .txt or .zip")
    p.add_argument("-o", "--output", help="output HTML file (default: <chat name>_report.html)")
    p.add_argument("--title", help="report title (default: file name)")
    p.add_argument("--gap-hours", type=float, default=6.0, help="silence that starts a new conversation (default 6)")
    p.add_argument("--late-reply-hours", type=float, default=36.0,
                   help="answer to an unanswered message within this time counts as a late reply, not a new start (default 36)")
    p.add_argument("--followup-minutes", type=float, default=30.0,
                   help="writing again after this long with no answer counts as a follow-up (default 30)")
    p.add_argument("--streak-min-people", type=int, default=2,
                   help="a day counts for a streak when at least this many people wrote (default 2 = both; 1 = any message)")
    p.add_argument("--date-order", choices=["DMY", "MDY", "YMD"], help="force date order if auto-detection is wrong")
    p.add_argument("--since", help="only use messages from this date (YYYY-MM-DD)")
    p.add_argument("--until", help="only use messages up to this date (YYYY-MM-DD)")
    p.add_argument("--exclude", nargs="*", default=[], help="people to leave out (e.g. bots)")
    p.add_argument("--lexicon", nargs="+", help="word-list JSON file(s) to use instead of the built-in ones "
                   "(lexicon.json + lexicon_tr.json)")
    p.add_argument("--explain", metavar="TEXT", nargs="+",
                   help='show how the word classifier reads a message, e.g. --explain "seni seviom aşkm"')
    p.add_argument("--csv", help="also save the parsed messages to this CSV file")
    st = p.add_argument_group("statistics")
    st.add_argument("--no-stats", action="store_true",
                    help="skip the statistical models (faster): confidence intervals, survival, change points, ...")
    st.add_argument("--bootstrap", type=int, default=1000, help="bootstrap samples for confidence intervals (default 1000)")
    st.add_argument("--predict-hours", type=float, default=1.0,
                    help="reply-prediction target: answered within this many hours (default 1)")
    st.add_argument("--survival-cap-hours", type=float, default=48.0,
                    help="waits longer than this count as 'no reply' in the survival model (default 48)")
    st.add_argument("--seed", type=int, default=0, help="random seed")
    ai = p.add_argument_group("AI results (create them first with: python -m wa_analyzer.ai ...)")
    ai.add_argument("--ai-dir", help="folder with AI results (default: '<chat name>_ai' next to the chat, if it exists)")
    ai.add_argument("--no-ai", action="store_true", help="do not include AI results even if they exist")
    ai.add_argument("--classifier", help="trained message classifier (.joblib) to add to the report")
    p.add_argument("--open", action="store_true", help="open the report in the browser")
    a = p.parse_args(argv)
    if a.explain:
        from .lexicon_engine import WordClassifier, looks_like_question
        clf = WordClassifier(a.lexicon)
        for text in a.explain:
            found = clf.explain(text)
            print(f"\n{text}")
            print(f"  question: {'yes' if looks_like_question(text) else 'no'}")
            for cat, hits in found.items():
                print(f"  {clf.labels[cat]:<28} {', '.join(dict.fromkeys(hits))}")
            if not found:
                print("  (no category)")
        return 0
    if not a.chat:
        p.error("give a chat file (or use --explain)")

    chat = Path(a.chat)
    df = load_filtered(chat, a.date_order, a.since, a.until, a.exclude)
    people = df.loc[df["kind"] != "system", "author"].value_counts()
    print(f"Parsed {len(df):,} lines · {people.sum():,} messages from {len(people)} people "
          f"(date order {df.attrs.get('date_order')})")
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"Saved messages to {a.csv}")

    s = Settings(gap_hours=a.gap_hours, late_reply_hours=a.late_reply_hours, followup_minutes=a.followup_minutes,
                 streak_min_people=a.streak_min_people)
    if people.sum():
        from .analysis import daily_streaks, prepare
        sk = daily_streaks(prepare(df, s), s.streak_min_people)
        if sk.longest is not None:
            lg = sk.longest
            print(f"Longest streak: {int(lg['days'])} days ({lg['start']:%d %b %Y} – {lg['end']:%d %b %Y}) · "
                  f"current: {sk.current} days")
        else:
            print("No streak: there is no day on which enough people wrote")
    title = a.title or chat.stem.replace("WhatsApp Chat with ", "").replace("_", " ")
    out = Path(a.output or chat.with_name(chat.stem + "_report.html"))
    ai_dir = None if a.no_ai else (a.ai_dir or (str(default_ai_dir(chat)) if default_ai_dir(chat).exists() else None))
    if ai_dir:
        print(f"Including AI results from {ai_dir}")
    o = Options(stats=not a.no_stats, bootstrap=a.bootstrap, seed=a.seed, predict_hours=a.predict_hours,
                survival_cap_hours=a.survival_cap_hours, lexicon=a.lexicon, ai_dir=ai_dir, classifier=a.classifier)
    html = build_report(df, title, s, options=o, progress=lambda msg: print("  " + msg))
    out.write_text(html, encoding="utf-8")
    print(f"Report written to {out.resolve()}")
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
