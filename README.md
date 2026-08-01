# WhatsApp Chat Analysis

A CLI tool (with an optional web UI) for analyzing WhatsApp chat exports: statistics, plots, sentiment, anomaly detection, and phrase search.

## Features

- **Multi-format parsing**: auto-detects Android and iOS export styles, 12h/24h clocks, and day-first/month-first dates
- **Group chats**: works with any number of participants, not just two
- **Message classification**: text, media, voice notes, edited messages
- **Statistics**: response times, activity by hour/weekday, conversation starters, message bursts, milestones
- **Emoji analysis** and **common phrases** (bigrams/trigrams)
- **Phrase and domain tracking**: track any phrase or link domain, repeatable
- **Sentiment analysis** (optional): any Hugging Face model, labels read from the model's own output
- **Anomaly detection**: flags unusual-activity days via `IsolationForest`
- **Plots**: message counts, activity heatmaps, messages-over-time — shown or saved as PNG
- **Export**: full stats as JSON, flat tables as CSV
- **Phrase search**: one-shot (`--phrase`) or interactive

## Installation

```bash
git clone https://github.com/SabrinaCiccolo/whatsapp-chat-analysis.git
cd whatsapp-chat-analysis
python3 -m venv venv
venv/bin/pip install -e .
```

For sentiment analysis (optional, heavy — pulls in `transformers` + `torch`):

```bash
venv/bin/pip install -e ".[sentiment]"
```

## Exporting your WhatsApp chat

**Android**: open the chat → ⋮ menu → More → Export chat → "Without media".
**iPhone**: open the chat → contact name → Export Chat → "Without Media".

## Usage

```bash
whatsapp-analysis analyze chat.txt
```

This parses the file, auto-detects its format, prints a full stats report, saves 3 plots and `stats.json` to `outputs/`.

Common flags:

```bash
# Restrict/order to specific senders (auto-detected from the file otherwise)
whatsapp-analysis analyze chat.txt --users Alice Bob

# Track custom phrases and link domains (repeatable)
whatsapp-analysis analyze chat.txt --track-phrase "ti amo" --track-domain meet.google.com

# Skip plots, export CSV instead of JSON
whatsapp-analysis analyze chat.txt --plots none --export csv

# Show plots interactively instead of saving them
whatsapp-analysis analyze chat.txt --plots show

# Force date order or line format if auto-detection guesses wrong
whatsapp-analysis analyze chat.txt --date-order dmy --line-format ios_bracket

# Sentiment analysis (needs the [sentiment] extra)
whatsapp-analysis analyze chat.txt --sentiment --sentiment-model distilbert-base-uncased-finetuned-sst-2-english
```

Search a chat for a phrase:

```bash
# One-shot
whatsapp-analysis search chat.txt --phrase "good morning"

# Interactive (prompts repeatedly until you press Enter with nothing)
whatsapp-analysis search chat.txt
```

Run `whatsapp-analysis analyze --help` or `whatsapp-analysis search --help` for the full flag list.

## Web UI

A Streamlit app for the same analysis, without the CLI:

```bash
venv/bin/pip install -e ".[ui]"
venv/bin/streamlit run src/whatsapp_analysis/app.py
```

Upload a chat export in the browser, adjust filters/users in the sidebar, and see stats,
plots, and phrase search update live.

### Config file

Any flag can instead be set in a JSON file, with CLI flags always taking precedence:

```bash
whatsapp-analysis analyze chat.txt --config myconfig.json
```

```json
{
  "burst_threshold": 15,
  "conversation_gap_hours": 6,
  "track_phrases": ["ti amo"]
}
```

## Troubleshooting

**"Could not confidently detect the chat line format"** — the export doesn't match the known Android/iOS patterns. Pass `--line-format android_dash` or `--line-format ios_bracket` explicitly.

**Dates look wrong** — pass `--date-order mdy` or `--date-order dmy` to override the auto-detected order.

**Sentiment analysis skipped** — install the optional extra: `pip install -e ".[sentiment]"`.

## Development

```bash
venv/bin/pip install -e ".[dev]"
venv/bin/pytest
```

## Privacy

Chat data is processed **locally**. Nothing is sent anywhere, except a one-time model download from Hugging Face if you use `--sentiment`.
