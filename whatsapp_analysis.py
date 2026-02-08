import re
import pandas as pd
import numpy as np
from datetime import datetime
import emoji
from collections import Counter
import warnings
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

CHAT_FILE = "chat.txt"
USER1_NAME = "Alice"
USER2_NAME = "Bob"

# Love expression to search for (set this to any phrase in any language)
# Examples: "ti amo", "I love you", "te quiero", "je t'aime", "ich liebe dich"
LOVE_EXPRESSION = "ti amo"

# Specific link domain to track (set this to any domain you want to track)
# Examples: "meet.google.com", "instagram.com", "tiktok.com", "zoom.us"
SPECIFIC_LINK_DOMAIN = "meet.google.com"

# Analysis parameters
BURST_THRESHOLD = 10  # Consecutive messages to consider a "burst"
CONVERSATION_GAP_HOURS = 4  # Hours of silence = new conversation
SENTIMENT_SAMPLE_SIZE = 2000  # Max messages for sentiment analysis
SHOW_PLOTS = True  # Set to False to skip visualizations

# ============================================================================
# INPUT VALIDATION
# ============================================================================

if not os.path.exists(CHAT_FILE):
    raise FileNotFoundError(f"Chat file not found: {CHAT_FILE}")

if not LOVE_EXPRESSION:
    print("WARNING: LOVE_EXPRESSION is empty, love analysis will be skipped")

if not SPECIFIC_LINK_DOMAIN:
    print("WARNING: SPECIFIC_LINK_DOMAIN is empty, specific link tracking disabled")

# ============================================================================
# REGEX PATTERNS (pre-compiled for efficiency)
# ============================================================================

MESSAGE_PATTERN = re.compile(r'(\d{1,2}/\d{1,2}/\d{2}),\s(\d{2}:\d{2})\s-\s([^:]*):\s(.*)')
LINK_PATTERN = re.compile(r'http[s]?://', re.IGNORECASE)
WORD_CLEAN_PATTERN = re.compile(r'[^\w\s]')

# Create dynamic patterns based on configuration
LOVE_EXPRESSION_PATTERN = re.compile(rf'\b{re.escape(LOVE_EXPRESSION)}\b', re.IGNORECASE) if LOVE_EXPRESSION else None
SPECIFIC_LINK_PATTERN = re.compile(re.escape(SPECIFIC_LINK_DOMAIN), re.IGNORECASE) if SPECIFIC_LINK_DOMAIN else None

# Day order for heatmaps
DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_MAPPING = {i: day for i, day in enumerate(DAY_ORDER)}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_emojis(text):
    """Count emojis in text."""
    if pd.isna(text):
        return 0
    return len([c for c in str(text) if c in emoji.EMOJI_DATA])


def extract_emojis(text):
    """Extract all emojis from text."""
    if pd.isna(text):
        return []
    return [c for c in str(text) if c in emoji.EMOJI_DATA]


def extract_ngrams(messages, n):
    """Extract n-grams from messages."""
    ngrams = []
    for text in messages.fillna(''):
        text_clean = text.lower()
        text_clean = WORD_CLEAN_PATTERN.sub(' ', text_clean)
        words = text_clean.split()
        
        for i in range(len(words) - n + 1):
            if all(len(words[i+j]) > 1 for j in range(n)):
                ngrams.append(' '.join(words[i:i+n]))
    
    return ngrams


def to_minutes(x):
    """Convert timedelta to minutes."""
    if pd.isna(x) or x == 0:
        return 0
    return x.total_seconds() / 60 if hasattr(x, 'total_seconds') else float(x)


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_chat_file(filepath):
    """Parse WhatsApp chat file into a list of message dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    messages = []
    current_message = None
    
    for line in content.split('\n'):
        match = MESSAGE_PATTERN.match(line)
        
        if match:
            if current_message:
                messages.append(current_message)
            
            date_str, time_str, sender, message = match.groups()
            current_message = {
                'date': date_str,
                'time': time_str,
                'sender': sender.strip(),
                'message': message.strip()
            }
        else:
            if current_message:
                current_message['message'] += '\n' + line
    
    if current_message:
        messages.append(current_message)
    
    return messages


def add_datetime_features(df):
    """Add datetime-related features to dataframe."""
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%y %H:%M')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_name'] = df['datetime'].dt.day_name()
    df['date_only'] = df['datetime'].dt.date
    df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday=5, Sunday=6
    return df.sort_values('datetime').reset_index(drop=True)


def classify_messages(df):
    """Classify different types of messages and add features."""
    # Message type classification
    df['is_media'] = df['message'].str.contains(r'<Media omitted>', regex=True, na=False)
    df['is_view_once_voice'] = df['message'].str.contains(r'<View once voice message omitted>', regex=True, na=False)
    df['has_edit_marker'] = df['message'].str.contains(r' <This message was edited>', regex=True, na=False)
    df['clean_message'] = df['message'].str.replace(' <This message was edited>', '', regex=False)
    
    # Text message features
    text_mask = ~df['is_media'] & ~df['is_view_once_voice']
    df.loc[text_mask, 'message_length'] = df.loc[text_mask, 'clean_message'].str.len()
    df.loc[text_mask, 'word_count'] = df.loc[text_mask, 'clean_message'].str.split().str.len()
    df.loc[text_mask, 'emoji_count'] = df.loc[text_mask, 'clean_message'].apply(count_emojis)
    df['emoji_count'] = df['emoji_count'].fillna(0)
    
    # Link detection
    df['has_link'] = df['clean_message'].str.contains(LINK_PATTERN, na=False)
    if SPECIFIC_LINK_PATTERN:
        df['is_specific_link'] = df['clean_message'].str.contains(SPECIFIC_LINK_PATTERN, na=False)
    else:
        df['is_specific_link'] = False
    
    return df


def detect_bursts(df, threshold=BURST_THRESHOLD):
    """Detect message bursts (consecutive messages by same sender)."""
    df['burst_id'] = 0
    df['is_burst'] = False
    
    burst_id = 0
    current_sender = None
    consecutive_count = 0
    burst_start = 0
    
    for idx, row in df.iterrows():
        if row['sender'] == current_sender:
            consecutive_count += 1
        else:
            if consecutive_count > threshold:
                df.loc[burst_start:idx-1, 'is_burst'] = True
                df.loc[burst_start:idx-1, 'burst_id'] = burst_id
                burst_id += 1
            
            current_sender = row['sender']
            consecutive_count = 1
            burst_start = idx
    
    if consecutive_count > threshold:
        df.loc[burst_start:, 'is_burst'] = True
        df.loc[burst_start:, 'burst_id'] = burst_id
    
    return df


def calculate_response_times(df):
    """Calculate response times between users."""
    df['time_diff'] = df['datetime'].diff()
    df['sender_changed'] = df['sender'] != df['sender'].shift(1)
    df['response_time'] = df.apply(lambda x: x['time_diff'] if x['sender_changed'] else pd.NaT, axis=1)
    return df


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("WHATSAPP CHAT ANALYSIS")
print("="*80)
print(f"File: {CHAT_FILE}")
print(f"Users: {USER1_NAME}, {USER2_NAME}")

# Parse chat file
print("\n" + "="*80)
print("PARSING CHAT")
print("="*80)

print("Parsing messages...", end='', flush=True)
messages = parse_chat_file(CHAT_FILE)
df = pd.DataFrame(messages)
print(" ✓")
print(f"Parsed {len(df):,} total messages")

# Add features
print("\n" + "="*80)
print("CLASSIFYING MESSAGES")
print("="*80)

print("Adding datetime features...", end='', flush=True)
df = add_datetime_features(df)
print(" ✓")

print("Classifying message types...", end='', flush=True)
df = classify_messages(df)
print(" ✓")

print("Detecting message bursts...", end='', flush=True)
df = detect_bursts(df)
print(" ✓")

print("Calculating response times...", end='', flush=True)
df = calculate_response_times(df)
print(" ✓")

# Create separate dataframes for different message types
df_simple = df[~df['has_edit_marker'] & ~df['is_media'] & ~df['is_view_once_voice']].copy()
df_media = df[df['is_media']].copy()
df_voice_once = df[df['is_view_once_voice']].copy()
df_edited = df[df['has_edit_marker']].copy()

print(f"\nTotal messages: {len(df):,}")
print(f"  Simple messages (no edits): {len(df_simple):,}")
print(f"  Media: {len(df_media):,}")
print(f"  View-once voice: {len(df_voice_once):,}")
print(f"  Edited messages: {len(df_edited):,}")

# ============================================================================
# QUICK OVERVIEW
# ============================================================================

print("\n" + "="*80)
print("QUICK OVERVIEW")
print("="*80)
print(f"Date range: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
total_days = (df['datetime'].max() - df['datetime'].min()).days
print(f"Total days: {total_days}")
print(f"Total messages: {len(df):,}")
if total_days > 0:
    print(f"Messages per day: {len(df)/total_days:.1f}")

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "="*80)
print("STATISTICS")
print("="*80)

for user in [USER1_NAME, USER2_NAME]:
    user_df = df[df['sender'] == user]
    user_simple = df_simple[df_simple['sender'] == user]
    user_text = user_df[~user_df['is_media'] & ~user_df['is_view_once_voice']]
    
    print(f"\n{user}:")
    print(f"  Total messages: {len(user_df):,}")
    print(f"  Simple: {len(user_simple):,} | Media: {len(df_media[df_media['sender']==user]):,} | Voice: {len(df_voice_once[df_voice_once['sender']==user]):,} | Edited: {len(df_edited[df_edited['sender']==user]):,}")
    
    if len(user_text) > 0:
        print(f"  Total words: {user_text['word_count'].sum():,}")
        print(f"  Average message length: {user_text['message_length'].mean():.1f} characters")
        print(f"  Average words per message: {user_text['word_count'].mean():.1f}")
        print(f"  Total emojis: {user_text['emoji_count'].sum():,}")
        if SPECIFIC_LINK_DOMAIN:
            print(f"  Links sent: {user_text['has_link'].sum():,} ({SPECIFIC_LINK_DOMAIN}: {user_text['is_specific_link'].sum()})")
        else:
            print(f"  Links sent: {user_text['has_link'].sum():,}")

# Most active hours
print("\n" + "-"*80)
print("MOST ACTIVE HOURS")
print("-"*80)

for user in [USER1_NAME, USER2_NAME]:
    user_hours = df[df['sender'] == user]['hour'].value_counts().sort_index()
    if len(user_hours) > 0:
        peak_hour = user_hours.idxmax()
        print(f"{user}: Most active at {peak_hour}:00 ({user_hours.max()} messages)")

# Weekday vs Weekend
print("\n" + "-"*80)
print("WEEKDAY VS WEEKEND ACTIVITY")
print("-"*80)

for user in [USER1_NAME, USER2_NAME]:
    user_df = df[df['sender'] == user]
    weekday = len(user_df[~user_df['is_weekend']])
    weekend = len(user_df[user_df['is_weekend']])
    total = len(user_df)
    
    if total > 0:
        print(f"{user}:")
        print(f"  Weekday: {weekday:,} ({weekday/total*100:.1f}%)")
        print(f"  Weekend: {weekend:,} ({weekend/total*100:.1f}%)")

# Response times
print("\n" + "-"*80)
print("RESPONSE TIMES")
print("-"*80)

response_df = df[df['sender_changed'] == True].copy()
response_df['response_minutes'] = response_df['response_time'].dt.total_seconds() / 60

for user in [USER1_NAME, USER2_NAME]:
    user_responses = response_df[response_df['sender'] == user]['response_minutes'].dropna()
    
    if len(user_responses) > 0:
        non_instant = user_responses[user_responses > 0.016]
        mean_resp = user_responses.mean()
        median_resp = non_instant.median() if len(non_instant) > 0 else user_responses.median()
        
        print(f"{user}: Average {mean_resp:.1f} min | Median {median_resp:.1f} min")

# Conversation starters
print("\n" + "-"*80)
print(f"CONVERSATION STARTERS (after {CONVERSATION_GAP_HOURS}+ hour gap)")
print("-"*80)

conversation_break = pd.Timedelta(hours=CONVERSATION_GAP_HOURS)
df['is_conversation_start'] = (df['datetime'].diff() > conversation_break) | (df['datetime'].diff().isna())
starters = df[df['is_conversation_start']]['sender'].value_counts()

for user, count in starters.items():
    pct = (count / starters.sum()) * 100
    print(f"{user}: {count:,} times ({pct:.1f}%)")

# Bursts
if df['is_burst'].sum() > 0:
    print("\n" + "-"*80)
    print(f"MESSAGE BURSTS (>{BURST_THRESHOLD} consecutive messages)")
    print("-"*80)
    
    bursts_df = df[df['is_burst']].copy()
    burst_by_user = bursts_df.groupby('sender')['burst_id'].nunique()
    
    for user, count in burst_by_user.items():
        total_burst_msgs = len(bursts_df[bursts_df['sender'] == user])
        print(f"{user}: {count} bursts ({total_burst_msgs:,} messages)")
    
    print("\nExamples:")
    unique_bursts = bursts_df['burst_id'].unique()
    for burst_id in np.random.choice(unique_bursts, min(3, len(unique_bursts)), replace=False):
        burst_msgs = bursts_df[bursts_df['burst_id'] == burst_id]
        first_msg = burst_msgs.iloc[0]
        duration = (burst_msgs.iloc[-1]['datetime'] - first_msg['datetime']).total_seconds() / 60
        print(f"  {first_msg['sender']}: {len(burst_msgs)} messages in {duration:.1f} min on {first_msg['datetime'].strftime('%Y-%m-%d %H:%M')}")

# Conversation milestones
print("\n" + "-"*80)
print("CONVERSATION MILESTONES")
print("-"*80)

first_day_msgs = df[df['date_only'] == df['date_only'].min()]
print(f"First day ({df['datetime'].min().strftime('%B %d, %Y')}): {len(first_day_msgs)} messages")

if len(df) >= 1000:
    milestone_1k = df.iloc[999]
    print(f"1,000th message: {milestone_1k['datetime'].strftime('%B %d, %Y')} by {milestone_1k['sender']}")

if len(df) >= 10000:
    milestone_10k = df.iloc[9999]
    print(f"10,000th message: {milestone_10k['datetime'].strftime('%B %d, %Y')} by {milestone_10k['sender']}")

# ============================================================================
# EMOJI ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("EMOJI ANALYSIS")
print("="*80)

all_emojis = []
for msg in df_simple['clean_message']:
    all_emojis.extend(extract_emojis(msg))

print("\nTop 10 emojis overall:")
for em, count in Counter(all_emojis).most_common(10):
    print(f"  {em} : {count:,} times")

print("\nTop 5 by user:")
for user in [USER1_NAME, USER2_NAME]:
    user_msgs = df_simple[df_simple['sender'] == user]['clean_message']
    user_emojis = []
    for msg in user_msgs:
        user_emojis.extend(extract_emojis(msg))
    
    print(f"\n{user}:")
    for em, count in Counter(user_emojis).most_common(5):
        print(f"  {em} : {count:,}")

# ============================================================================
# COMMON PHRASES
# ============================================================================

print("\n" + "="*80)
print("COMMON PHRASES")
print("="*80)

bigrams = extract_ngrams(df_simple['clean_message'], 2)
trigrams = extract_ngrams(df_simple['clean_message'], 3)

print("\nTop 15 two-word phrases:")
for phrase, count in Counter(bigrams).most_common(15):
    print(f"  '{phrase}': {count:,}")

print("\nTop 15 three-word phrases:")
for phrase, count in Counter(trigrams).most_common(15):
    print(f"  '{phrase}': {count:,}")

# ============================================================================
# LOVE EXPRESSIONS
# ============================================================================

if LOVE_EXPRESSION and LOVE_EXPRESSION_PATTERN:
    print("\n" + "="*80)
    print("LOVE EXPRESSIONS")
    print("="*80)

    love_msgs = df_simple[df_simple['clean_message'].str.contains(LOVE_EXPRESSION_PATTERN, na=False)]

    print(f"\nMessages containing '{LOVE_EXPRESSION}': {len(love_msgs):,}")

    if len(love_msgs) > 0:
        love_by_user = love_msgs.groupby('sender').size()
        print("\nBy user:")
        for user, count in love_by_user.items():
            pct = (count / len(love_msgs)) * 100
            print(f"  {user}: {count:,} ({pct:.1f}%)")
        
        # Find variants (words around the expression)
        variants = []
        for msg in love_msgs['clean_message']:
            # Search for the expression with surrounding words
            pattern = rf'\b\w*\s*{re.escape(LOVE_EXPRESSION)}\s*\w*\s*\w*'
            matches = re.finditer(pattern, str(msg), re.IGNORECASE)
            for match in matches:
                variants.append(match.group().strip().lower())
        
        print("\nTop 15 variants:")
        for variant, count in Counter(variants).most_common(15):
            print(f"  '{variant}': {count:,}")
        
        print("\nRandom examples:")
        for _, row in love_msgs.sample(min(5, len(love_msgs))).iterrows():
            print(f"  [{row['datetime'].strftime('%Y-%m-%d')}] {row['sender']}: {row['clean_message'][:70]}...")

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

print("\n" + "="*80)
print("SENTIMENT ANALYSIS")
print("="*80)

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

if not TRANSFORMERS_AVAILABLE:
    print("\nSentiment analysis skipped")
    print("  Install with: pip install transformers torch")
else:
    try:
        print("Loading sentiment model...", end='', flush=True)
        sentiment_pipeline = pipeline("sentiment-analysis", model="neuraly/bert-base-italian-cased-sentiment", device=-1)
        print(" ✓")
        
        # Filter valid messages for sentiment analysis
        valid_messages = df_simple[
            (df_simple['clean_message'].str.len() > 10) &
            (df_simple['word_count'] > 1) &
            (~df_simple['clean_message'].str.contains(LINK_PATTERN, na=False))
        ].copy()
        
        print(f"Total valid messages for sentiment analysis: {len(valid_messages):,}")
        
        if len(valid_messages) == 0:
            print("No valid messages for sentiment analysis.")
        else:
            min_date = valid_messages['datetime'].min()
            max_date = valid_messages['datetime'].max()
            total_days = (max_date - min_date).days + 1
            
            print(f"Conversation spans {total_days} days")
            
            # Determine sample size
            sample_size = min(SENTIMENT_SAMPLE_SIZE, int(len(valid_messages) * 0.3))
            if sample_size < 500:
                sample_size = min(500, len(valid_messages))
            
            print(f"\nRandomly sampling {sample_size:,} messages for sentiment analysis...")
            
            # Stratified sampling by user
            sampled_df = pd.DataFrame()
            for user in [USER1_NAME, USER2_NAME]:
                user_messages = valid_messages[valid_messages['sender'] == user]
                user_sample_size = min(sample_size // 2, len(user_messages))
                
                if len(user_messages) > 0 and user_sample_size > 0:
                    if len(user_messages) <= user_sample_size:
                        user_sample = user_messages
                    else:
                        user_sample = user_messages.sample(n=user_sample_size, random_state=42)
                    
                    sampled_df = pd.concat([sampled_df, user_sample])
            
            # Add remaining samples if needed
            if len(sampled_df) < sample_size:
                remaining_needed = sample_size - len(sampled_df)
                remaining_messages = valid_messages[~valid_messages.index.isin(sampled_df.index)]
                if len(remaining_messages) > 0:
                    additional_sample = remaining_messages.sample(
                        n=min(remaining_needed, len(remaining_messages)),
                        random_state=42
                    )
                    sampled_df = pd.concat([sampled_df, additional_sample])
            
            print(f"Analyzing {len(sampled_df):,} sampled messages ({len(sampled_df)/len(valid_messages)*100:.1f}% of valid messages)")
            
            texts = sampled_df['clean_message'].fillna('').tolist()
            sentiments = []
            
            batch_size = 16
            print(f"\nProcessing {len(texts):,} messages in batches of {batch_size}...")
            
            for i in range(0, len(texts), batch_size):
                batch = [t[:512] for t in texts[i:i+batch_size]]
                sentiments.extend(sentiment_pipeline(batch))
                
                if (i // batch_size) % 10 == 0:
                    progress = min(i + batch_size, len(texts))
                    print(f"  Processed {progress:,} of {len(texts):,} messages...")
            
            # Overall sentiment distribution
            sentiment_counts = Counter([s['label'] for s in sentiments])
            total = len(sentiments)
            
            print("\n" + "-"*80)
            print("OVERALL SENTIMENT DISTRIBUTION:")
            print("-"*80)
            for label, count in sorted(sentiment_counts.items()):
                percentage = (count / total) * 100
                print(f"  {label}: {count:,} messages ({percentage:.1f}%)")
            
            # Sentiment by user
            print("\n" + "-"*80)
            print("SENTIMENT BY USER:")
            print("-"*80)
            
            sampled_df['sentiment'] = [s['label'] for s in sentiments]
            sampled_df['sentiment_score'] = [s['score'] for s in sentiments]
            
            for user in [USER1_NAME, USER2_NAME]:
                user_sentiments = sampled_df[sampled_df['sender'] == user]['sentiment']
                if len(user_sentiments) > 0:
                    user_counts = Counter(user_sentiments)
                    user_total = len(user_sentiments)
                    
                    print(f"\n{user}:")
                    for label in sorted(user_counts.keys()):
                        count = user_counts[label]
                        percentage = (count / user_total) * 100
                        print(f"  {label}: {count:,} ({percentage:.1f}%)")
                    
                    avg_score = sampled_df[sampled_df['sender'] == user]['sentiment_score'].mean()
                    print(f"  Average confidence score: {avg_score:.3f}")
            
            # Examples of each sentiment
            print("\n" + "-"*80)
            print("EXAMPLES OF EACH SENTIMENT:")
            print("-"*80)
            
            for sentiment in ['positive', 'neutral', 'negative']:
                examples = sampled_df[sampled_df['sentiment'] == sentiment].head(2)
                if len(examples) > 0:
                    print(f"\n{sentiment.upper()} examples:")
                    for idx, row in examples.iterrows():
                        preview = row['clean_message'][:80] + "..." if len(row['clean_message']) > 80 else row['clean_message']
                        print(f"  [{row['datetime'].strftime('%Y-%m-%d')}] {row['sender']}: {preview}")
    
    except ImportError:
        print("\nSentiment analysis skipped")
        print("  Install with: pip install transformers torch")
    except Exception as e:
        print(f"\nSentiment analysis failed: {type(e).__name__}")
        print(f"  Error: {str(e)[:100]}")

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

print("\n" + "="*80)
print("ANOMALY DETECTION")
print("="*80)

daily_stats = df.groupby('date_only').agg({
    'message': 'count',
    'response_time': 'mean',
    'word_count': 'sum'
}).fillna(0)

daily_stats.columns = ['message_count', 'avg_response_time', 'total_words']
daily_stats['avg_response_time'] = daily_stats['avg_response_time'].apply(to_minutes)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(daily_stats)

iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=1)
daily_stats['anomaly'] = iso_forest.fit_predict(features_scaled)
daily_stats['anomaly_score'] = iso_forest.score_samples(features_scaled)

anomalies = daily_stats[daily_stats['anomaly'] == -1].sort_values('anomaly_score')

print(f"\nFound {len(anomalies)} anomalous days (unusual activity patterns):\n")
print(anomalies.head(10).to_string())

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

first = df.iloc[0]
last = df.iloc[-1]
total_days = (df['datetime'].max() - df['datetime'].min()).days

print(f"\nConversation period:")
print(f"  First message: {first['datetime'].strftime('%B %d, %Y')} by {first['sender']}")
print(f"  Last message: {last['datetime'].strftime('%B %d, %Y')} by {last['sender']}")
print(f"  Total days: {total_days}")

print(f"\nTotal messages: {len(df):,}")
print(f"  Average per day: {len(df)/max(total_days,1):.1f}")

msg_counts = df.groupby('sender').size().sort_values(ascending=False)
leader = msg_counts.index[0]
leader_pct = (msg_counts.iloc[0] / msg_counts.sum()) * 100
print(f"\nMost active: {leader} ({leader_pct:.1f}% of messages)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

if SHOW_PLOTS:
    # Figure 1: Message counts
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    message_counts = df['sender'].value_counts()
    ax1.bar(message_counts.index, message_counts.values, color=['#FF932E', '#FFEE2E'])
    ax1.set_title('Total Messages by User', fontsize=14, fontweight='bold')
    ax1.set_xlabel('User')
    ax1.set_ylabel('Number of Messages')
    for i, (user, count) in enumerate(message_counts.items()):
        ax1.text(i, count + 5, f'{count:,}', ha='center', fontweight='bold')

    # Message types by user
    simple1 = len(df_simple[df_simple['sender'] == USER1_NAME])
    media1 = len(df_media[df_media['sender'] == USER1_NAME])
    voice1 = len(df_voice_once[df_voice_once['sender'] == USER1_NAME])
    edited1 = len(df_edited[df_edited['sender'] == USER1_NAME])

    simple2 = len(df_simple[df_simple['sender'] == USER2_NAME])
    media2 = len(df_media[df_media['sender'] == USER2_NAME])
    voice2 = len(df_voice_once[df_voice_once['sender'] == USER2_NAME])
    edited2 = len(df_edited[df_edited['sender'] == USER2_NAME])

    bar_width = 0.35
    index = [0, 1, 2, 3]

    ax2.bar(index, [simple1, media1, voice1, edited1], bar_width, label=USER1_NAME, color='#FF932E')
    ax2.bar([i + bar_width for i in index], [simple2, media2, voice2, edited2], bar_width, label=USER2_NAME, color='#FFEE2E')

    ax2.set_xlabel('Message Type')
    ax2.set_ylabel('Count')
    ax2.set_title('Message Types by User')
    ax2.set_xticks([i + bar_width/2 for i in index])
    ax2.set_xticklabels(['Simple', 'Media', 'View-once Voice', 'Edited'])
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Figure 2: Activity heatmaps
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

    # User 1 heatmap
    user1_df = df[df['sender'] == USER1_NAME]
    heatmap_data1 = user1_df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    heatmap_data1.index = heatmap_data1.index.map(DAY_MAPPING)
    heatmap_data1 = heatmap_data1.reindex(DAY_ORDER)

    sns.heatmap(heatmap_data1, cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Messages'}, linewidths=0.5, linecolor='gray')
    ax3.set_title(f'{USER1_NAME} - Activity by Hour/Day', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('')

    # User 2 heatmap
    user2_df = df[df['sender'] == USER2_NAME]
    heatmap_data2 = user2_df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    heatmap_data2.index = heatmap_data2.index.map(DAY_MAPPING)
    heatmap_data2 = heatmap_data2.reindex(DAY_ORDER)

    sns.heatmap(heatmap_data2, cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Messages'}, linewidths=0.5, linecolor='gray')
    ax4.set_title(f'{USER2_NAME} - Activity by Hour/Day', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('')

    plt.tight_layout()
    plt.show()

    # Figure 3: Messages over time
    plt.figure(figsize=(12, 6))

    user1_counts = df[df['sender'] == USER1_NAME].groupby('date_only').size()
    user2_counts = df[df['sender'] == USER2_NAME].groupby('date_only').size()

    if len(user1_counts) > 0:
        dates1 = list(user1_counts.index)
        plt.plot(dates1, list(user1_counts.values), label=USER1_NAME, marker='o', markersize=3,
                 linewidth=2, color="#FF932E")
    if len(user2_counts) > 0:
        dates2 = list(user2_counts.index)
        plt.plot(dates2, list(user2_counts.values), label=USER2_NAME, marker='s', markersize=3,
                 linewidth=2, color="#FFEE2E")

    plt.title('Message Frequency Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Messages per Day')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("\n[Plots skipped - SHOW_PLOTS is set to False]")

# ============================================================================
# SEARCH FUNCTIONALITY
# ============================================================================

print("\n" + "="*80)
print("SEARCH FOR SPECIFIC PHRASES")
print("="*80)

search_string = input("\nEnter a word or phrase to search for in messages: ").strip()

if search_string:
    print(f"\nSearching for: '{search_string}'")
    
    search_mask = df['clean_message'].str.contains(search_string, case=False, na=False)
    messages_with_string = df[search_mask]
    
    total_count = len(messages_with_string)
    print(f"\nTotal messages containing '{search_string}': {total_count:,}")
    
    print("\nMessages by user:")
    for user in [USER1_NAME, USER2_NAME]:
        user_count = len(messages_with_string[messages_with_string['sender'] == user])
        if total_count > 0:
            percentage = (user_count / total_count) * 100
            print(f"  {user}: {user_count:,} ({percentage:.1f}%)")
        else:
            print(f"  {user}: {user_count:,}")
    
    if total_count > 0:
        print(f"\nShowing {min(10, total_count)} examples:")
        print("-" * 80)
        
        examples = messages_with_string.sort_values('datetime', ascending=False).head(10)
        
        for idx, row in examples.iterrows():
            message_preview = row['clean_message']
            if len(message_preview) > 100:
                message_preview = message_preview[:97] + "..."
            
            message_preview = message_preview.replace('\n', ' ')
            
            print(f"[{row['datetime'].strftime('%Y-%m-%d %H:%M')}] {row['sender']}:")
            print(f"  {message_preview}")
            print("-" * 80)
    
    if total_count > 0:
        print(f"\nAdditional statistics:")
        
        first_msg = messages_with_string.sort_values('datetime').iloc[0]
        last_msg = messages_with_string.sort_values('datetime', ascending=False).iloc[0]
        
        print(f"  First occurrence: {first_msg['datetime'].strftime('%B %d, %Y')}")
        print(f"  Last occurrence: {last_msg['datetime'].strftime('%B %d, %Y')}")
        
        days_counts = messages_with_string.groupby('date_only').size().sort_values(ascending=False)
        if len(days_counts) > 0:
            top_day = days_counts.index[0]
            top_count = days_counts.iloc[0]
            print(f"  Day with most mentions: {top_day} ({top_count} times)")
        
        if 'hour' in messages_with_string.columns:
            hour_counts = messages_with_string.groupby('hour').size()
            if len(hour_counts) > 0:
                most_active_hour = hour_counts.idxmax()
                print(f"  Most common hour: {most_active_hour}:00 ({hour_counts.max()} times)")
        
        total_messages = len(df)
        percentage_of_all = (total_count / total_messages) * 100
        print(f"  Percentage of all messages: {percentage_of_all:.2f}%")
else:
    print("\nNo search string entered. Skipping search.")
