"""Generate fake WhatsApp exports in every common format, with known ground truth.

Usage:  python tests/make_fake_chats.py OUTPUT_DIR
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

LRM = "‎"
NNBSP = " "

LINES = {
    "keen": [
        "Ciao amore come stai? 😘", "Günaydın canım ☀️ nasılsın?", "I miss you so much ❤️", "Cosa fai stasera?",
        "Domani ci vediamo? 😍", "Hahaha sei troppo divertente 😂", "Bu akşam buluşalım mı?", "Sei bellissima oggi",
        "Did you sleep well? 🥰", "Ti penso sempre 💕", "Hey are you free this weekend?", "Ne yapıyorsun şu an?",
        "Grazie mille per ieri ❤️", "Che bello! raccontami tutto", "Ahahah no dai 🤣",
        "Ho visto questa cosa e ho pensato a te\nguarda che carina", "When are you free? I'd love to see you",
        "slm aşkm nbr 😘", "seni çok seviommm", "özledimmm cnm", "napıyon bi tanem", "yarın müsaitsen buluşak mı",
        "ksksksks çok komiksin", "günaydınnn ☀️", "iyi geceler tatlı rüyalar 🌙", "eve vardın mı?",
    ],
    "cool": [
        "ok", "Tamam", "va bene", "👍", "si", "hmm", "Sono stanca oggi", "busy now, later", "Bene, tu?",
        "Haha", "Ok ci sentiamo", "boh", "Yarın bakarız", "sure", "Iyi", "Lascia stare 🙄", "Fine",
        "Sì domani forse", "Grazie",
        "tmm", "ok", "hı", "bakarız", "boşver", "yoruldum ya", "peki", "meşgulüm sonra yazarım",
    ],
}
GROUP_LINES = [
    "Chi viene stasera?", "Io ci sono!", "Ben de geliyorum 😂", "Ok", "Who's bringing the drinks?", "hahaha",
    "Sorry I'm late", "Bu fotoğraf çok güzel", "Ci vediamo alle 8", "👍", "Che bello!", "No dai, domani?",
    "Teşekkürler herkese ❤️", "Anyone free on Saturday?", "Lol", "Vengo anche io\nma arrivo tardi",
]
MEDIA = ["image", "video", "audio", "sticker"]


def simulate_pair(seed=1, days=180):
    """A writes a lot early on and cools down; B is the opposite."""
    rnd = random.Random(seed)
    a, b = "Abdullah Çelik", "Giulia"
    msgs = []
    t = datetime(2025, 1, 1, 9, 0)
    end = t + timedelta(days=days)
    while t < end:
        progress = (t - datetime(2025, 1, 1)).days / days
        a_keen = 1 - progress  # A's interest falls over time
        starter = a if rnd.random() < 0.3 + 0.5 * a_keen else b
        cur = starter
        n_turns = rnd.randint(2, 10)
        for _ in range(n_turns):
            keen = a_keen if cur == a else 1 - a_keen
            n = rnd.randint(1, 4) if rnd.random() < keen else 1
            for _ in range(n):
                pool = LINES["keen"] if rnd.random() < keen else LINES["cool"]
                r = rnd.random()
                if r < 0.06:
                    msgs.append((t, cur, None, rnd.choice(MEDIA)))
                elif r < 0.07:
                    msgs.append((t, cur, None, "deleted"))
                else:
                    msgs.append((t, cur, rnd.choice(pool), "text"))
                t += timedelta(seconds=rnd.randint(5, 90))
            cur = b if cur == a else a
            keen_next = a_keen if cur == a else 1 - a_keen
            # keen people answer fast, others slowly
            delay = rnd.expovariate(1 / (1 + 40 * (1 - keen_next) ** 2))
            t += timedelta(minutes=delay)
        t += timedelta(hours=rnd.uniform(7, 30))
        if 1 <= t.hour < 7:
            t += timedelta(hours=7)
    return msgs


def simulate_group(seed=2, days=60):
    rnd = random.Random(seed)
    people = ["Marco", "Elif", "Sara", "Abdullah Çelik", "Tom"]
    weights = [5, 3, 2, 4, 1]
    msgs = []
    t = datetime(2025, 3, 1, 18, 0)
    end = t + timedelta(days=days)
    while t < end:
        for _ in range(rnd.randint(3, 15)):
            p = rnd.choices(people, weights)[0]
            if rnd.random() < 0.08:
                msgs.append((t, p, None, "image"))
            else:
                msgs.append((t, p, rnd.choice(GROUP_LINES), "text"))
            t += timedelta(minutes=rnd.expovariate(1 / 6))
        t += timedelta(hours=rnd.uniform(8, 48))
    return msgs


# ---------------------------------------------------------------- renderers

def _media_text(kind, style):
    if kind == "deleted":
        return "This message was deleted"
    if style == "android":
        return "<Media omitted>"
    return f"{LRM}{kind} omitted" if kind != "sticker" else f"{LRM}sticker omitted"


def render(msgs, fmt, system=None):
    out = []
    style = "ios" if fmt.startswith("ios") else "android"
    for ts, kind_line in (system or []):
        out.append(_header(ts, fmt) + kind_line)
    for i, (ts, author, text, kind) in enumerate(msgs):
        body = text if kind == "text" else _media_text(kind, style)
        if kind == "text" and i % 37 == 5:
            body += " <This message was edited>"
        prefix = LRM if (style == "ios" and kind not in ("text",)) else ""
        out.append(prefix + _header(ts, fmt) + f"{author}: {body}")
    return "\n".join(out) + "\n"


def _header(ts, fmt):
    if fmt == "android_24h_dmy":
        return ts.strftime("%d/%m/%Y, %H:%M - ")
    if fmt == "android_12h_mdy":
        return f"{ts.month}/{ts.day}/{ts.strftime('%y')}, {ts.strftime('%I').lstrip('0')}:{ts.strftime('%M')}{NNBSP}{ts.strftime('%p')} - "
    if fmt == "android_dotted":
        return ts.strftime("%d.%m.%y, %H:%M - ")
    if fmt == "ios_24h_dmy":
        return ts.strftime("[%d/%m/%y, %H:%M:%S] ")
    if fmt == "ios_12h_mdy":
        return f"[{ts.month}/{ts.day}/{ts.strftime('%y')}, {ts.strftime('%I').lstrip('0')}:{ts.strftime('%M:%S')}{NNBSP}{ts.strftime('%p')}] "
    if fmt == "iso":
        return ts.strftime("%Y-%m-%d, %H:%M - ")
    raise ValueError(fmt)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair = simulate_pair()
    group = simulate_group()
    # Android drops seconds; keep truth at minute precision for android formats
    truth = {}
    for fmt in ["android_24h_dmy", "android_12h_mdy", "android_dotted", "ios_24h_dmy", "ios_12h_mdy", "iso"]:
        sysl_pair = [(pair[0][0], "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.")]
        if fmt.startswith("ios"):
            sysl_pair = [(pair[0][0], f"Giulia: {LRM}Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.")]
            sysl_group = [(group[0][0], f"Weekend crew: {LRM}Marco created group “Weekend crew”"),
                          (group[0][0], f"Weekend crew: {LRM}Marco added Elif")]
        else:
            sysl_group = [(group[0][0], "Marco created group \"Weekend crew\""), (group[0][0], "Marco added Elif"),
                          (group[0][0], "Marco changed the subject from \"Crew\" to \"Weekend: crew\"")]
        (out / f"pair_{fmt}.txt").write_text(render(pair, fmt, sysl_pair), encoding="utf-8")
        (out / f"group_{fmt}.txt").write_text(render(group, fmt, sysl_group), encoding="utf-8")
    truth["pair"] = [(ts.isoformat(), a, t, k) for ts, a, t, k in pair]
    truth["group"] = [(ts.isoformat(), a, t, k) for ts, a, t, k in group]
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")
    # an iOS-style zip export
    import zipfile
    with zipfile.ZipFile(out / "pair_ios_export.zip", "w") as z:
        z.writestr("_chat.txt", render(pair, "ios_24h_dmy"))
    print(f"wrote fake chats to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fake_chats")
