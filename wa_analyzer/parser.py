"""Parse WhatsApp chat exports (.txt or .zip) from Android and iOS into a DataFrame.

Handles:
  * Android:  "31/12/23, 21:05 - Name: text"   and  "12/31/23, 9:05 PM - Name: text"
  * iOS:      "[31/12/23, 21:05:33] Name: text" and "[12/31/23, 9:05:33 PM] Name: text"
  * Date separators / . -, 2- or 4-digit years, year-first dates (2023-12-31)
  * Day/month order is detected automatically (or forced with date_order=)
  * Multi-line messages, invisible direction marks, system messages, media placeholders
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

# Invisible characters WhatsApp sprinkles into exports
_INVISIBLE = "‎‏‪‫‬⁦⁧⁨⁩﻿"
_SPACES = "   "

HEADER_RE = re.compile(
    r"^\[?"
    r"(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4})"
    r",?\s+"
    r"(?P<time>\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)"
    r"(?:\s?(?P<ampm>[AaPp]\.?\s?[Mm]\.?))?"
    r"\]?"
    r"(?:\s+-\s+|\s+)"
    r"(?P<rest>.*)$"
)
AUTHOR_RE = re.compile(r"^(?P<author>[^:]{1,60}?):\s(?P<text>.*)$", re.S)

# Media / special placeholders (English exports). Extend if your phone uses another language.
MEDIA_PATTERNS = {
    "image": [r"^image omitted$", r"\.(jpg|jpeg|png|webp|heic) \(file attached\)$", r"PHOTO.*\.(jpg|jpeg|png)>?$"],
    "video": [r"^video omitted$", r"\.(mp4|mov|3gp) \(file attached\)$", r"VIDEO.*\.mp4>?$"],
    "audio": [r"^audio omitted$", r"\.(opus|m4a|mp3|aac) \(file attached\)$", r"AUDIO.*\.(opus|m4a)>?$", r"^PTT-"],
    "sticker": [r"^sticker omitted$", r"\.webp \(file attached\)$", r"STICKER.*\.webp>?$"],
    "gif": [r"^GIF omitted$", r"GIF.*\.mp4>?$"],
    "document": [r"^document omitted$", r"\.(pdf|docx?|xlsx?|pptx?|zip|txt) \(file attached\)$", r"^.* • \d+ pages? document omitted$"],
    "contact": [r"^Contact card omitted$", r"\.vcf \(file attached\)$"],
    "media": [r"^<Media omitted>$", r"^<attached: .*>$", r"\(file attached\)$"],
    "location": [r"^location: https?://", r"^Location: https?://"],
    "poll": [r"^POLL:"],
    "view_once": [r"^null$", r"^view once (photo|video|voice message) omitted$"],
}
_MEDIA_RES = {k: [re.compile(p, re.I) for p in v] for k, v in MEDIA_PATTERNS.items()}
DELETED_RE = re.compile(r"^(This message was deleted|You deleted this message)\.?$", re.I)
EDITED_RE = re.compile(r"\s*<This message was edited>\s*$", re.I)
SYSTEM_HINTS = re.compile(
    r"(end-to-end encrypted|created group|created this group|added|removed|left$|joined using|"
    r"changed (the |this )?(group|subject|description|icon|settings)|changed their phone number|"
    r"security code|disappearing messages|is a contact|pinned a message|now an admin|"
    r"Missed (voice|video) call|Voice call|Video call|turned (on|off))",
    re.I,
)


def _clean(s: str) -> str:
    for ch in _INVISIBLE:
        s = s.replace(ch, "")
    for ch in _SPACES:
        s = s.replace(ch, " ")
    return s


def read_export(path: str | Path) -> str:
    """Return the raw chat text from a .txt or a .zip export."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            txts = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not txts:
                raise ValueError("No .txt chat file found inside the zip")
            name = next((n for n in txts if Path(n).name == "_chat.txt"), txts[0])
            raw = z.read(name)
    else:
        raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode chat file")


def _detect_date_order(dates: list[str]) -> str:
    """Return 'YMD', 'DMY' or 'MDY'."""
    parts = [re.split(r"[./-]", d) for d in dates]
    if parts and len(parts[0][0]) == 4:
        return "YMD"
    a = [int(p[0]) for p in parts]
    b = [int(p[1]) for p in parts]
    if max(a, default=0) > 12:
        return "DMY"
    if max(b, default=0) > 12:
        return "MDY"
    # Ambiguous (all values <= 12): choose the order with fewest backwards jumps in time
    def backwards(order):
        ts = [_to_date(d, order) for d in dates]
        return sum(1 for x, y in zip(ts, ts[1:]) if y < x)
    return min(("DMY", "MDY"), key=backwards)


def _to_date(d: str, order: str) -> tuple[int, int, int]:
    p = [int(x) for x in re.split(r"[./-]", d)]
    if order == "YMD":
        y, m, dd = p
    elif order == "DMY":
        dd, m, y = p
    else:
        m, dd, y = p
    if y < 100:
        y += 2000
    return (y, m, dd)


def _to_minutes_seconds(t: str, ampm: str | None) -> tuple[int, int, int]:
    p = [int(x) for x in re.split(r"[:.]", t)]
    h, mi = p[0], p[1]
    s = p[2] if len(p) > 2 else 0
    if ampm:
        pm = ampm.lower().startswith("p")
        if h == 12:
            h = 0
        if pm:
            h += 12
    return h, mi, s


def classify_media(text: str) -> str | None:
    t = text.strip()
    for kind, res in _MEDIA_RES.items():
        if any(r.search(t) for r in res):
            return kind
    return None


def parse_text(raw: str, date_order: str | None = None) -> pd.DataFrame:
    """Parse raw export text into a DataFrame with one row per message.

    Columns: timestamp, author, text, kind ('text', 'system', 'deleted', or a media kind), edited
    """
    lines = raw.splitlines()
    entries = []  # [date, time, ampm, rest, had_lrm_text]
    for line in lines:
        clean = _clean(line)
        m = HEADER_RE.match(clean)
        if m:
            entries.append([m["date"], m["time"], m["ampm"], m["rest"], line])
        elif entries:
            entries[-1][3] += "\n" + clean
        # lines before the first header are ignored
    if not entries:
        raise ValueError("No WhatsApp messages recognised. Is this a WhatsApp chat export?")

    order = date_order or _detect_date_order([e[0] for e in entries])
    rows = []
    for d, t, ampm, rest, original in entries:
        y, mo, dd = _to_date(d, order)
        h, mi, s = _to_minutes_seconds(t, ampm)
        ts = pd.Timestamp(year=y, month=mo, day=dd, hour=h, minute=mi, second=s)
        am = AUTHOR_RE.match(rest)
        if am:
            author, text = am["author"].strip(), am["text"]
            # iOS marks system lines (and attachments) with an invisible mark before the text
            after_colon = original.split(": ", 1)[1] if ": " in original else ""
            lrm_text = after_colon.startswith("‎")
        else:
            author, text, lrm_text = None, rest, False
        rows.append((ts, author, text, lrm_text))

    df = pd.DataFrame(rows, columns=["timestamp", "author", "text", "lrm"])
    df["edited"] = df["text"].str.contains(EDITED_RE)
    df["text"] = df["text"].str.replace(EDITED_RE, "", regex=True)

    kinds = []
    for author, text, lrm in zip(df["author"], df["text"], df["lrm"]):
        if author is None:
            kinds.append("system")
            continue
        if DELETED_RE.match(text.strip()):
            kinds.append("deleted")
            continue
        media = classify_media(text)
        if media:
            kinds.append(media)
        elif lrm and SYSTEM_HINTS.search(text):
            kinds.append("system")  # iOS group-name "author" lines
        else:
            kinds.append("text")
    df["kind"] = kinds

    # Authors that only ever produce system-looking lines are not people (e.g. iOS group name)
    people = df[df["kind"] != "system"]["author"].dropna().unique()
    df.loc[~df["author"].isin(people), "kind"] = "system"

    # Text with a colon in a system line can be mistaken for "author: text"; demote
    # authors that look like sentences and appear only once.
    counts = df[df["kind"] != "system"]["author"].value_counts()
    suspicious = [a for a, c in counts.items() if c == 1 and (len(a.split()) > 4 or '"' in a)]
    df.loc[df["author"].isin(suspicious), "kind"] = "system"

    df = df.drop(columns="lrm").sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.attrs["date_order"] = order
    return df


def load_chat(path: str | Path, date_order: str | None = None) -> pd.DataFrame:
    return parse_text(read_export(path), date_order=date_order)
