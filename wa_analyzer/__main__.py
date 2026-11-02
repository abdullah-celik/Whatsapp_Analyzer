"""Command line:  python -m wa_analyzer CHAT.txt|CHAT.zip [-o report.html] [options]"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .analysis import Settings
from .parser import load_chat
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
    p.add_argument("--date-order", choices=["DMY", "MDY", "YMD"], help="force date order if auto-detection is wrong")
    p.add_argument("--since", help="only use messages from this date (YYYY-MM-DD)")
    p.add_argument("--until", help="only use messages up to this date (YYYY-MM-DD)")
    p.add_argument("--exclude", nargs="*", default=[], help="people to leave out (e.g. bots)")
    p.add_argument("--lexicon", nargs="+", help="word-list JSON file(s) to use instead of the built-in ones "
                   "(lexicon.json + lexicon_tr.json)")
    p.add_argument("--explain", metavar="TEXT", nargs="+",
                   help='show how the word classifier reads a message, e.g. --explain "seni seviom aşkm"')
    p.add_argument("--csv", help="also save the parsed messages to this CSV file")
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
    df = load_chat(chat, date_order=a.date_order)
    if a.since:
        df = df[df["timestamp"] >= a.since]
    if a.until:
        df = df[df["timestamp"] < __import__("pandas").Timestamp(a.until) + __import__("pandas").Timedelta(days=1)]
    if a.exclude:
        df = df[~df["author"].isin(a.exclude)]
    people = df.loc[df["kind"] != "system", "author"].value_counts()
    print(f"Parsed {len(df):,} lines · {people.sum():,} messages from {len(people)} people "
          f"(date order {df.attrs.get('date_order')})")
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"Saved messages to {a.csv}")

    s = Settings(gap_hours=a.gap_hours, late_reply_hours=a.late_reply_hours, followup_minutes=a.followup_minutes)
    title = a.title or chat.stem.replace("WhatsApp Chat with ", "").replace("_", " ")
    out = Path(a.output or chat.with_name(chat.stem + "_report.html"))
    out.write_text(build_report(df, title, s, a.lexicon), encoding="utf-8")
    print(f"Report written to {out.resolve()}")
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
