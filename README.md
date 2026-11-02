# WhatsApp Chat Analyzer (local)

Analyses an exported WhatsApp chat on your own computer and creates one HTML report.
Works for one-to-one and group chats, Android and iPhone exports, and messages in any language.
Nothing is uploaded anywhere.

## 1. Export a chat from WhatsApp

- **iPhone:** open the chat → tap the name at the top → *Export Chat* → **Without Media** → you get a `.zip`.
- **Android:** open the chat → ⋮ → *More* → *Export chat* → **Without media** → you get a `.txt` (sometimes zipped).

Use the `.txt` or the `.zip` directly; no need to unzip. Keep your phone in English if you can.
The chat itself can be in any language, but the placeholders like "image omitted" are recognised in English.

## 2. Install (once)

Requires Python 3.10+.

```bash
cd wa_analyzer
pip install -r requirements.txt
```

## 3. Run

```bash
python -m wa_analyzer "WhatsApp Chat with Giulia.zip" --open
```

The report is saved next to the chat as `<chat name>_report.html` and opened in your browser.

Useful options:

| Option | What it does |
|---|---|
| `-o report.html` | choose the output file |
| `--since 2025-01-01 --until 2025-06-30` | analyse only a period |
| `--gap-hours 6` | silence that starts a new conversation (default 6 h) |
| `--followup-minutes 30` | writing again after this long with no answer counts as a follow-up |
| `--late-reply-hours 36` | answering an unanswered message within this time is a *late reply*, not a new start |
| `--exclude "Meta AI"` | leave people/bots out |
| `--date-order DMY` | force the date order if detection is wrong (DMY, MDY, YMD) |
| `--lexicon my_words.json` | use your own word lists |
| `--csv messages.csv` | also save all parsed messages as a table |

## What the report shows

1. **Key findings**: a few sentences summarising the chat.
2. **Overview**: messages, words, media, questions, typical reply time, conversations started and ended, late replies, messages that got no answer, follow-ups without a reply.
3. **Who writes, and when**: per month, hour of day and weekday, plus a heatmap.
4. **Response times**: distribution per person; for groups, *who replies to whom* and reply time by pair.
5. **Who is more interested / Engagement in the group**: an effort score per person, with the share of every signal and how the score changes month by month.
6. **Word classifier**: affection, asking about the other, plans, compliments, thanks, apologies, laughter, negative/distant words and dry one-word replies, per person and over time.
7. **Words typical for each person**: words each person uses much more often than the others (works in any language).

### How the measures are defined

- **Conversation**: messages with no silence longer than `--gap-hours`.
- **Reply time**: from someone's last message to the other person's first answer, inside the same conversation.
- **Started a conversation**: wrote first after a long silence. If the previous conversation ended with someone else's unanswered message or question, it counts as a **late reply** instead.
- **Effort score**: the average of each person's share of the interest signals. With two people, 50 % is balanced; in a group, an equal share is 100 % divided by the number of people. It is a set of signals, not proof.

## The word classifier (Turkish-optimised)

Every message is checked for these categories: affection, asking about the other, plans, compliments,
enthusiasm, thanks, apologies, laughter, good morning / good night, negative / distant, and dry one-word replies.
A message counts once per category, and the report shows the % of each person's messages in each category.

Word lists:
- `wa_analyzer/lexicon_tr.json` — Turkish, ~1,000 entries including texting abbreviations
  (slm, nbr, cnm, tşk, kib, sçs, muck, grşz, eyw, tmm ...).
- `wa_analyzer/lexicon.json` — English, Italian, Spanish, German, French.

Spellings are matched automatically, so you only write each word once, normally:

| You write in the list | Also matched automatically |
|---|---|
| `aşkım` | askim, aşkımmm, ASKIMM, ajkım, aşkm, askm |
| `seviyorum` | seviyom, seviom, sevyom, seviyorm, seviyrm, svyrm, sevıyorum, seviyorummm |
| `ne yapıyorsun` | ne yapıyosun, ne yapıyon, neyapiyorsun ... |
| `özleyeceğim` | özlicem, özleycem, özleyecem |
| `bebeğim` | bebeim, bebegim |
| `günaydın` | gunaydin, günaydınnnn, gnydn |

How: letters are compared without Turkish characters (ı=i, ş=s, ğ=g, ç=c, ö=o, ü=u), stretched letters
are collapsed (canımmmm = canım), and Turkish entries get spoken verb endings (-iyom, -iyon, -icem) and
vowel-dropped forms. Laughter also recognises keyboard smashes (ajsjsjs, sksksk, asdfghjkl). Questions are
detected by `?` and Turkish question particles (mı/mi/mu/mü, misin, musun ...) and question words.

Tips:
- Check how a message is read: `python3 -m wa_analyzer --explain "seni seviom aşkm" "tmm"`
- `word*` matches every word starting with `word`; `=word` matches only exactly as written.
- `except` (in a category) lists phrases that must not count, e.g. `canım sıkıldı` is not affection.
- `never_match` lists spellings that must never count because they are ordinary words
  (e.g. *bela*, *afet*, *sal*, *op*). The lists were checked against the 50,000 most common Turkish,
  English and Italian words to remove such clashes.
- Use your own lists: `--lexicon my_tr.json my_other.json`.

## Tests

```bash
python tests/make_fake_chats.py tests/fake_chats
PYTHONPATH=.:tests python -m pytest tests
```

The fake chats cover Android and iPhone exports, 12- and 24-hour clocks, day/month and month/day order, dotted and ISO dates, group system messages, multi-line and edited messages, deleted messages and media.
