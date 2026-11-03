# Whatsapp_Analyzer

**WhatsApp chat analyzer — fully local**

Analyses an exported WhatsApp chat on your own computer: statistics, statistical models and local AI.
Works for one-to-one and group chats, Android and iPhone exports, any chat language (Turkish-optimised word lists).
Nothing is uploaded anywhere: AI models run on your CPU or NVIDIA GPU, the LLM runs in Ollama on your machine.

## 1. Install

CPU only (no NVIDIA GPU):
```bash
bash setup_cpu.sh
```
With an NVIDIA GPU:
```bash
bash setup_gpu.sh
```
Both create a `.venv` folder and finish with a check (`python -m wa_analyzer.ai doctor`). Afterwards, in every new
terminal: `source .venv/bin/activate`.

Only want the report and statistics (no AI)? `pip install -r requirements.txt` is enough.
Optional: `pip install -r requirements-extra.txt` adds SHAP explanations to the prediction model.

**Ollama** (local LLM, for `read`, `label` and topic names):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b     # CPU: small and fast enough
ollama pull qwen2.5:7b     # GPU (6 GB+ VRAM): better judgements
```
Other models work too (`--model llama3.2:3b`, `--model gemma3:4b`, ...).

## 2. Export a chat

iPhone: chat → name at the top → *Export Chat* → **Without Media** (.zip).
Android: chat → ⋮ → *More* → *Export chat* → **Without media** (.txt).
Keep the phone in English if you can (the chat itself can be in any language).

## 3. Report (command line)
```bash
python -m wa_analyzer ~/giulia.txt --open
```
Writes `giulia_report.html` next to the chat. If AI results exist in `giulia_ai/`, they are included automatically.

| Option | |
|---|---|
| `--since 2025-01-01 --until 2025-06-30` | only this period |
| `--exclude "Meta AI"` | leave people/bots out |
| `--gap-hours 6` | silence that starts a new conversation |
| `--no-stats` | skip the statistical models (faster) |
| `--bootstrap 1000` | samples for confidence intervals |
| `--predict-hours 1` | prediction target: answered within N hours |
| `--survival-cap-hours 48` | waits longer than this count as "no reply" |
| `--no-ai` / `--ai-dir DIR` | leave out / choose AI results |
| `--explain "seni seviom" "tmm"` | show how the word classifier reads a message |

## 4. Dashboard
```bash
streamlit run dashboard.py
```
Opens in your browser (local only). Pick a chat (from `chats/`, your home folder, a path, or drag & drop), filter by
period and people, change settings, run the local AI from the sidebar, and download the HTML report.

## 5. Local AI
```bash
python -m wa_analyzer.ai doctor                       # check PyTorch, GPU, Ollama
python -m wa_analyzer.ai sentiment ~/giulia.txt       # tone + emotions (transformers)
python -m wa_analyzer.ai topics ~/giulia.txt          # topics over time
python -m wa_analyzer.ai read ~/giulia.txt            # LLM reads conversations
python -m wa_analyzer.ai label ~/giulia.txt           # LLM labels a sample of messages
python -m wa_analyzer.ai review ~/giulia.txt          # YOU check the labels -> accuracy
python -m wa_analyzer.ai train ~/giulia.txt           # train a fast classifier on the labels
python -m wa_analyzer ~/giulia.txt --open             # report with everything
```
All commands take `--device auto|cpu|cuda` (default: GPU if available). Results are cached in `giulia_ai/`:
stop with Ctrl+C at any time and run the same command again to continue.

| Step | What | CPU (rough) | NVIDIA GPU (rough) |
|---|---|---|---|
| sentiment | XLM-R sentiment + Turkish BERT + XLM-EMO emotions on every turn | 30–60 turns/s per model; use `--max-turns 5000` | 500+ turns/s |
| topics | multilingual MiniLM embeddings + k-means + keywords | a few minutes | < 1 min |
| read | Ollama reads conversations (`--per-month`, default 4 CPU / 12 GPU) | 20–60 s per conversation | 2–5 s |
| label | Ollama labels messages in batches of 15 (`--n`, default 600 CPU / 2000 GPU) | ~1–3 s per message | fast |
| train | embeddings + logistic regression per category, 5-fold CV | < 1 min | seconds |

Models (downloaded once from Hugging Face, then cached locally):
`cardiffnlp/twitter-xlm-roberta-base-sentiment`, `savasy/bert-base-turkish-sentiment-cased`,
`MilaNLProc/xlm-emo-t`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Change them with
`--sentiment-model`, `--turkish-model`, `--emotion-model` (`none` to skip).

## What is in the report

**Descriptive**: key findings, overview per person, activity by month / hour / weekday, reply times,
who replies to whom (groups), word classifier categories, typical words per person.

**Statistics**
- *Confidence intervals*: cluster bootstrap (whole conversations resampled) for every main measure, plus
  "is the difference real?" for two-person chats.
- *Effort score* per person and per month, with 95% bands.
- *Survival analysis of reply times*: Kaplan–Meier curves (unanswered messages kept as censored) and a Cox model:
  which factors make replies faster or slower (hazard ratios with intervals).
- *Turning points*: change-point detection (optimal partitioning, as in PELT) on weekly effort, reply times, affection, dry replies, volume.
- *Who triggers whom*: multivariate Hawkes process with a daily rhythm: spontaneous vs. triggered writing,
  branching ratios (reactions per turn), decay half-life, per quarter.
- *Style matching & mirroring*: function-word style matching (LSM) per month; mirroring of emoji, laughter,
  questions, stretched letters ... (stratified by month), echo of emojis/words, length matching.
- *Reply prediction*: will a message be answered within N hours? Leakage-free features, time-ordered test set,
  logistic regression + gradient boosting vs. baselines, permutation importance (and SHAP if installed).
- *Group network* (groups): reply graph, PageRank, betweenness, sub-groups (Louvain), attention over time.

**Local AI**: tone and emotions over time, emotional contagion; LLM ratings of warmth / engagement / interest
per month, who drives conversations, tone, monthly and overall summaries; topics over time and per person;
LLM-labelled classifier vs. word lists, and accuracy checked by you.

## The word classifier (Turkish-optimised)
Word lists in `wa_analyzer/lexicon_tr.json` (Turkish, ~900 entries with slang and abbreviations) and
`wa_analyzer/lexicon.json` (English, Italian, Spanish, German, French). Spellings without Turkish letters,
stretched letters (canımmm), spoken forms (seviom, napıyon, gelicem) and vowel-dropped forms (cnm, svyrm)
are matched automatically. See `--explain`. `word*` = prefix, `=word` = exact only, `except` = phrases that
must not count, `never_match` = spellings that clash with ordinary words.

## Privacy
Chats, reports, AI folders, CSV/JSONL and model files are in `.gitignore`. The AI folders contain message text
(caches, labels): never publish them.

## Tests
```bash
python tests/make_fake_chats.py tests/fake_chats
PYTHONPATH=.:tests python -m pytest tests
```
Fake chats in every export format; statistical models checked on simulated data with known answers (e.g. the
Hawkes fit recovers the true branching ratios); Ollama replaced by a mock server; transformer code tested with
tiny local models.

## Project structure
```
wa_analyzer/
  parser.py, loading.py        reading exports
  analysis.py                  conversations, turns, replies, summaries
  lexicon_engine.py, interest.py, lexicon*.json   word classifier
  stats/                       aggregates+bootstrap, prompts, survival, changepoints, hawkes, style, predict, network
  ai/                          device, sentiment, topics, ollama, reader, labeling, classifier, results, sections
  pipeline.py                  runs everything -> Results
  sections.py, report.py       report sections -> HTML
dashboard.py                   Streamlit app (same sections)
```
