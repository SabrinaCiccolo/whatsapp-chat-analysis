# WhatsApp Chat Analysis

A comprehensive Python tool for analyzing WhatsApp chat exports with statistics, visualizations, sentiment analysis, and more.

## Features

- **Message Classification**: Categorizes messages into text, media, voice notes, and edited messages
- **Statistical Analysis**: 
  - Response times between users
  - Message frequency by hour and day
  - Weekday vs weekend activity
  - Conversation starters and message bursts
- **Emoji Analysis**: Track most-used emojis overall and by user
- **Common Phrases**: Identifies frequently used bigrams and trigrams
- **Custom Love Expression Tracking**: Search for any phrase in any language (e.g., "ti amo", "I love you", "te quiero")
- **Specific Link Tracking**: Monitor mentions of specific domains (e.g., Instagram, TikTok, Google Meet)
- **Sentiment Analysis**: AI-powered sentiment detection (requires optional dependencies)
- **Anomaly Detection**: Identifies unusual activity patterns using machine learning
- **Interactive Visualizations**:
  - Message distribution charts
  - Activity heatmaps by hour and day
  - Message frequency over time
- **Phrase Search**: Interactive search for specific words or phrases

## Requirements

### Core Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn emoji
```

### Optional (for Sentiment Analysis)
```bash
pip install transformers torch
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SabrinaCiccolo/whatsapp-chat-analysis.git
cd whatsapp-chat-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Export Your WhatsApp Chat

**On Android:**
1. Open the chat
2. Tap the three dots menu → More → Export chat
3. Choose "Without media" for faster processing
4. Save the `.txt` file

**On iPhone:**
1. Open the chat
2. Tap the contact name → Export Chat
3. Choose "Without Media"
4. Save the `.txt` file

### 2. Configure the Script

Edit the configuration section at the top of `whatsapp_analysis.py`:

```python
# ============================================================================
# CONFIGURATION
# ============================================================================

CHAT_FILE = "chat.txt"  # Path to your exported chat file
USER1_NAME = "Alice"     # First user's name (as it appears in the chat)
USER2_NAME = "Bob"       # Second user's name

# Love expression to search for (any phrase in any language)
LOVE_EXPRESSION = "I love you"  # Examples: "ti amo", "te quiero", "je t'aime"

# Specific link domain to track
SPECIFIC_LINK_DOMAIN = "instagram.com"  # Examples: "tiktok.com", "zoom.us"

# Analysis parameters
BURST_THRESHOLD = 10              # Consecutive messages to consider a "burst"
CONVERSATION_GAP_HOURS = 4        # Hours of silence = new conversation
SENTIMENT_SAMPLE_SIZE = 2000      # Max messages for sentiment analysis
SHOW_PLOTS = True                 # Set to False to skip visualizations
```

### 3. Run the Analysis

```bash
python whatsapp_analysis.py
```

The script will:
- Parse your chat file
- Generate comprehensive statistics
- Create visualizations (if enabled)
- Prompt for an interactive phrase search at the end

## Output

The script produces:

1. **Console Output**: Detailed statistics including:
   - Quick overview (date range, total messages, avg per day)
   - Message type breakdown
   - Most active hours
   - Weekday vs weekend activity
   - Response times
   - Conversation starters and milestones
   - Emoji analysis
   - Common phrases
   - Love expression frequency
   - Sentiment analysis (if available)
   - Anomaly detection results

2. **Visualizations** (3 figures):
   - Total messages and message types by user
   - Activity heatmaps (hour × day of week)
   - Message frequency over time

3. **Interactive Search**: Search for any phrase and see:
   - Total occurrences
   - Breakdown by user
   - Recent examples
   - First/last occurrence
   - Most active day and hour

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CHAT_FILE` | Path to WhatsApp export file | `"chat.txt"` |
| `USER1_NAME` | First user's display name | - |
| `USER2_NAME` | Second user's display name | - |
| `LOVE_EXPRESSION` | Phrase to track (any language) | `"ti amo"` |
| `SPECIFIC_LINK_DOMAIN` | Domain to monitor | `"meet.google.com"` |
| `BURST_THRESHOLD` | Messages for burst detection | `10` |
| `CONVERSATION_GAP_HOURS` | Gap defining new conversation | `4` |
| `SENTIMENT_SAMPLE_SIZE` | Max messages for sentiment | `2000` |
| `SHOW_PLOTS` | Enable/disable visualizations | `True` |

## Internationalization

The tool is language-agnostic and works with chats in any language:

- **Love Expression**: Set to any phrase in your language
  ```python
  LOVE_EXPRESSION = "te quiero"      # Spanish
  LOVE_EXPRESSION = "je t'aime"      # French
  LOVE_EXPRESSION = "ich liebe dich" # German
  LOVE_EXPRESSION = "我爱你"          # Chinese
  ```

- **Sentiment Analysis**: Currently configured for Italian. To use other languages, change the model:
  ```python
  # For English
  sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
  
  # For Spanish
  sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
  ```

## Troubleshooting

### "Chat file not found"
- Ensure the `CHAT_FILE` path is correct
- Check that the file is in the same directory as the script, or use an absolute path

### Sentiment Analysis Issues
- Install optional dependencies: `pip install transformers torch`
- The script will skip sentiment analysis if dependencies are missing

### Date Parsing Errors
- The script expects WhatsApp's default format: `MM/DD/YY, HH:MM - Name: Message`
- If your format differs, adjust the `MESSAGE_PATTERN` regex

### Memory Issues with Large Chats
- Reduce `SENTIMENT_SAMPLE_SIZE`
- Set `SHOW_PLOTS = False`
- Process the chat in smaller date ranges

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Sentiment analysis powered by [Hugging Face Transformers](https://huggingface.co/transformers/)
- Visualization built with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- Anomaly detection using [scikit-learn](https://scikit-learn.org/)

## Privacy Note

This tool processes chat data **locally on your computer**. No data is sent to external servers (except when downloading the sentiment analysis model for the first time). Always respect privacy when analyzing shared conversations.
