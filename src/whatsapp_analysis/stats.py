from __future__ import annotations

import random
import re
from collections import Counter

import numpy as np
import pandas as pd

from .features import extract_emojis, extract_ngrams

INSTANT_RESPONSE_THRESHOLD_MINUTES = 0.016  # ~1 second


def split_message_types(df):
    """Split the full dataframe into simple/media/voice/edited subsets."""
    df_simple = df[~df['has_edit_marker'] & ~df['is_media'] & ~df['is_view_once_voice']]
    df_media = df[df['is_media']]
    df_voice = df[df['is_view_once_voice']]
    df_edited = df[df['has_edit_marker']]
    return df_simple, df_media, df_voice, df_edited


def compute_overview(df):
    total_days = (df['datetime'].max() - df['datetime'].min()).days
    return {
        'date_start': df['datetime'].min(),
        'date_end': df['datetime'].max(),
        'total_days': total_days,
        'total_messages': len(df),
        'messages_per_day': len(df) / total_days if total_days > 0 else float(len(df)),
    }


def compute_message_type_counts(df, df_simple, df_media, df_voice, df_edited, users):
    per_user = {}
    for user in users:
        per_user[user] = {
            'total': int((df['sender'] == user).sum()),
            'simple': int((df_simple['sender'] == user).sum()),
            'media': int((df_media['sender'] == user).sum()),
            'voice': int((df_voice['sender'] == user).sum()),
            'edited': int((df_edited['sender'] == user).sum()),
        }
    return {
        'total': len(df),
        'simple': len(df_simple),
        'media': len(df_media),
        'voice': len(df_voice),
        'edited': len(df_edited),
        'per_user': per_user,
    }


def compute_user_stats(df_simple, users):
    stats = {}
    for user in users:
        user_text = df_simple[df_simple['sender'] == user]
        if len(user_text) == 0:
            stats[user] = None
            continue
        stats[user] = {
            'total_words': int(user_text['word_count'].sum()),
            'avg_message_length': float(user_text['message_length'].mean()),
            'avg_words_per_message': float(user_text['word_count'].mean()),
            'total_emojis': int(user_text['emoji_count'].sum()),
            'links_sent': int(user_text['has_link'].sum()),
        }
    return stats


def compute_active_hours(df, users):
    result = {}
    for user in users:
        user_hours = df[df['sender'] == user]['hour'].value_counts().sort_index()
        if len(user_hours) == 0:
            result[user] = None
            continue
        result[user] = {'peak_hour': int(user_hours.idxmax()), 'peak_count': int(user_hours.max())}
    return result


def compute_weekday_weekend(df, users):
    result = {}
    for user in users:
        user_df = df[df['sender'] == user]
        total = len(user_df)
        if total == 0:
            result[user] = None
            continue
        weekday = int((~user_df['is_weekend']).sum())
        weekend = int(user_df['is_weekend'].sum())
        result[user] = {
            'weekday': weekday,
            'weekend': weekend,
            'weekday_pct': weekday / total * 100,
            'weekend_pct': weekend / total * 100,
        }
    return result


def compute_response_times(df, users):
    response_df = df[df['sender_changed']].copy()
    response_df['response_minutes'] = response_df['response_time'].dt.total_seconds() / 60

    result = {}
    for user in users:
        user_responses = response_df[response_df['sender'] == user]['response_minutes'].dropna()
        if len(user_responses) == 0:
            result[user] = None
            continue
        non_instant = user_responses[user_responses > INSTANT_RESPONSE_THRESHOLD_MINUTES]
        median_resp = non_instant.median() if len(non_instant) > 0 else user_responses.median()
        result[user] = {
            'mean_minutes': float(user_responses.mean()),
            'median_minutes': float(median_resp),
        }
    return result


def compute_conversation_starters(df, gap_hours=4.0):
    gap = pd.Timedelta(hours=gap_hours)
    diffs = df['datetime'].diff()
    is_start = (diffs > gap) | diffs.isna()
    starters = df.loc[is_start, 'sender'].value_counts()
    total = starters.sum()

    return {
        user: {'count': int(count), 'pct': float(count / total * 100)}
        for user, count in starters.items()
    }


def compute_bursts_summary(df, threshold=10, n_examples=3, seed=None):
    bursts_df = df[df['is_burst']]
    if len(bursts_df) == 0:
        return {'per_user': {}, 'examples': [], 'threshold': threshold}

    per_user = {
        user: {'burst_count': int(group['burst_id'].nunique()), 'message_count': int(len(group))}
        for user, group in bursts_df.groupby('sender')
    }

    rng = np.random.default_rng(seed)
    unique_bursts = bursts_df['burst_id'].unique()
    chosen = rng.choice(unique_bursts, min(n_examples, len(unique_bursts)), replace=False)

    examples = []
    for burst_id in chosen:
        burst_msgs = bursts_df[bursts_df['burst_id'] == burst_id]
        first_msg = burst_msgs.iloc[0]
        duration_minutes = (burst_msgs.iloc[-1]['datetime'] - first_msg['datetime']).total_seconds() / 60
        examples.append({
            'sender': first_msg['sender'],
            'message_count': len(burst_msgs),
            'duration_minutes': duration_minutes,
            'start': first_msg['datetime'],
        })

    return {'per_user': per_user, 'examples': examples, 'threshold': threshold}


def compute_milestones(df):
    first_day = df['date_only'].min()
    milestones = {
        'first_day': first_day,
        'first_day_count': int((df['date_only'] == first_day).sum()),
    }

    if len(df) >= 1000:
        row = df.iloc[999]
        milestones['1000th'] = {'datetime': row['datetime'], 'sender': row['sender']}
    if len(df) >= 10000:
        row = df.iloc[9999]
        milestones['10000th'] = {'datetime': row['datetime'], 'sender': row['sender']}

    return milestones


def compute_emoji_stats(df_simple, users, top_overall=10, top_per_user=5):
    all_emojis = []
    for msg in df_simple['clean_message']:
        all_emojis.extend(extract_emojis(msg))

    per_user = {}
    for user in users:
        user_emojis = []
        for msg in df_simple[df_simple['sender'] == user]['clean_message']:
            user_emojis.extend(extract_emojis(msg))
        per_user[user] = Counter(user_emojis).most_common(top_per_user)

    return {'overall': Counter(all_emojis).most_common(top_overall), 'per_user': per_user}


def compute_common_phrases(df_simple, top_n=15):
    bigrams = extract_ngrams(df_simple['clean_message'], 2)
    trigrams = extract_ngrams(df_simple['clean_message'], 3)
    return {
        'bigrams': Counter(bigrams).most_common(top_n),
        'trigrams': Counter(trigrams).most_common(top_n),
    }


def compute_phrase_tracking(df_simple, phrase, top_variants=15, n_examples=5, seed=None):
    """Generalizes the old single hardcoded LOVE_EXPRESSION tracking to any phrase."""
    pattern = re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE)
    matches = df_simple[df_simple['clean_message'].str.contains(pattern, na=False)]

    if len(matches) == 0:
        return {'phrase': phrase, 'count': 0, 'by_user': {}, 'variants': [], 'examples': []}

    by_user = {
        user: {'count': int(count), 'pct': float(count / len(matches) * 100)}
        for user, count in matches.groupby('sender').size().items()
    }

    variant_pattern = re.compile(rf'\b\w*\s*{re.escape(phrase)}\s*\w*\s*\w*', re.IGNORECASE)
    variants = []
    for msg in matches['clean_message']:
        variants.extend(m.group().strip().lower() for m in variant_pattern.finditer(str(msg)))

    rng = random.Random(seed)
    example_idx = rng.sample(list(matches.index), min(n_examples, len(matches)))
    examples = [
        {
            'datetime': matches.loc[idx, 'datetime'],
            'sender': matches.loc[idx, 'sender'],
            'preview': str(matches.loc[idx, 'clean_message'])[:70],
        }
        for idx in example_idx
    ]

    return {
        'phrase': phrase,
        'count': len(matches),
        'by_user': by_user,
        'variants': Counter(variants).most_common(top_variants),
        'examples': examples,
    }


def compute_link_domain_stats(df_simple, users, domains):
    result = {}
    for domain in domains:
        pattern = re.compile(re.escape(domain), re.IGNORECASE)
        is_match = df_simple['clean_message'].str.contains(pattern, na=False)
        per_user = {user: int(is_match[df_simple['sender'] == user].sum()) for user in users}
        result[domain] = {'total': int(is_match.sum()), 'per_user': per_user}
    return result


def compute_summary(df):
    first, last = df.iloc[0], df.iloc[-1]
    total_days = (df['datetime'].max() - df['datetime'].min()).days

    msg_counts = df.groupby('sender').size().sort_values(ascending=False)
    leader = msg_counts.index[0]
    leader_pct = float(msg_counts.iloc[0] / msg_counts.sum() * 100)

    return {
        'first_message': {'datetime': first['datetime'], 'sender': first['sender']},
        'last_message': {'datetime': last['datetime'], 'sender': last['sender']},
        'total_days': total_days,
        'avg_per_day': len(df) / max(total_days, 1),
        'leader': leader,
        'leader_pct': leader_pct,
    }


def compute_all_stats(df, users, track_domains=None, track_phrases=None,
                       conversation_gap_hours=4.0, burst_threshold=10, seed=None):
    """Run the full analysis pipeline over an already-featurized dataframe and
    return one nested dict, used for both console reporting and export."""
    df_simple, df_media, df_voice, df_edited = split_message_types(df)

    stats = {
        'overview': compute_overview(df),
        'message_types': compute_message_type_counts(df, df_simple, df_media, df_voice, df_edited, users),
        'user_stats': compute_user_stats(df_simple, users),
        'active_hours': compute_active_hours(df, users),
        'weekday_weekend': compute_weekday_weekend(df, users),
        'response_times': compute_response_times(df, users),
        'conversation_starters': compute_conversation_starters(df, conversation_gap_hours),
        'bursts': compute_bursts_summary(df, burst_threshold, seed=seed),
        'milestones': compute_milestones(df),
        'emojis': compute_emoji_stats(df_simple, users),
        'phrases': compute_common_phrases(df_simple),
        'summary': compute_summary(df),
    }

    if track_domains:
        stats['domains'] = compute_link_domain_stats(df_simple, users, track_domains)

    if track_phrases:
        stats['tracked_phrases'] = [
            compute_phrase_tracking(df_simple, phrase, seed=seed) for phrase in track_phrases
        ]

    return stats


def print_report(stats, users):
    """Render a compute_all_stats() dict as console text, generalized to N users."""
    ov = stats['overview']
    print("\n" + "=" * 80)
    print("QUICK OVERVIEW")
    print("=" * 80)
    print(f"Date range: {ov['date_start'].strftime('%Y-%m-%d')} to {ov['date_end'].strftime('%Y-%m-%d')}")
    print(f"Total days: {ov['total_days']}")
    print(f"Total messages: {ov['total_messages']:,}")
    print(f"Messages per day: {ov['messages_per_day']:.1f}")

    mt = stats['message_types']
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"\nTotal messages: {mt['total']:,}")
    print(f"  Simple: {mt['simple']:,} | Media: {mt['media']:,} | Voice: {mt['voice']:,} | Edited: {mt['edited']:,}")

    for user in users:
        counts = mt['per_user'].get(user)
        if counts is None:
            continue
        print(f"\n{user}:")
        print(f"  Total messages: {counts['total']:,}")
        print(f"  Simple: {counts['simple']:,} | Media: {counts['media']:,} | Voice: {counts['voice']:,} | Edited: {counts['edited']:,}")
        user_stat = stats['user_stats'].get(user)
        if user_stat:
            print(f"  Total words: {user_stat['total_words']:,}")
            print(f"  Average message length: {user_stat['avg_message_length']:.1f} characters")
            print(f"  Average words per message: {user_stat['avg_words_per_message']:.1f}")
            print(f"  Total emojis: {user_stat['total_emojis']:,}")
            print(f"  Links sent: {user_stat['links_sent']:,}")

    print("\n" + "-" * 80)
    print("MOST ACTIVE HOURS")
    print("-" * 80)
    for user in users:
        info = stats['active_hours'].get(user)
        if info:
            print(f"{user}: Most active at {info['peak_hour']}:00 ({info['peak_count']} messages)")

    print("\n" + "-" * 80)
    print("WEEKDAY VS WEEKEND ACTIVITY")
    print("-" * 80)
    for user in users:
        info = stats['weekday_weekend'].get(user)
        if info:
            print(f"{user}:")
            print(f"  Weekday: {info['weekday']:,} ({info['weekday_pct']:.1f}%)")
            print(f"  Weekend: {info['weekend']:,} ({info['weekend_pct']:.1f}%)")

    print("\n" + "-" * 80)
    print("RESPONSE TIMES")
    print("-" * 80)
    for user in users:
        info = stats['response_times'].get(user)
        if info:
            print(f"{user}: Average {info['mean_minutes']:.1f} min | Median {info['median_minutes']:.1f} min")

    print("\n" + "-" * 80)
    print("CONVERSATION STARTERS")
    print("-" * 80)
    for user, info in stats['conversation_starters'].items():
        print(f"{user}: {info['count']:,} times ({info['pct']:.1f}%)")

    bursts = stats['bursts']
    if bursts['per_user']:
        print("\n" + "-" * 80)
        print(f"MESSAGE BURSTS (>{bursts['threshold']} consecutive messages)")
        print("-" * 80)
        for user, info in bursts['per_user'].items():
            print(f"{user}: {info['burst_count']} bursts ({info['message_count']:,} messages)")
        if bursts['examples']:
            print("\nExamples:")
            for ex in bursts['examples']:
                print(f"  {ex['sender']}: {ex['message_count']} messages in {ex['duration_minutes']:.1f} min "
                      f"on {ex['start'].strftime('%Y-%m-%d %H:%M')}")

    ms = stats['milestones']
    print("\n" + "-" * 80)
    print("CONVERSATION MILESTONES")
    print("-" * 80)
    print(f"First day ({ms['first_day']}): {ms['first_day_count']} messages")
    if '1000th' in ms:
        print(f"1,000th message: {ms['1000th']['datetime'].strftime('%B %d, %Y')} by {ms['1000th']['sender']}")
    if '10000th' in ms:
        print(f"10,000th message: {ms['10000th']['datetime'].strftime('%B %d, %Y')} by {ms['10000th']['sender']}")

    print("\n" + "=" * 80)
    print("EMOJI ANALYSIS")
    print("=" * 80)
    print("\nTop 10 emojis overall:")
    for em, count in stats['emojis']['overall']:
        print(f"  {em} : {count:,} times")
    print("\nTop 5 by user:")
    for user in users:
        print(f"\n{user}:")
        for em, count in stats['emojis']['per_user'].get(user, []):
            print(f"  {em} : {count:,}")

    print("\n" + "=" * 80)
    print("COMMON PHRASES")
    print("=" * 80)
    print("\nTop 15 two-word phrases:")
    for phrase, count in stats['phrases']['bigrams']:
        print(f"  '{phrase}': {count:,}")
    print("\nTop 15 three-word phrases:")
    for phrase, count in stats['phrases']['trigrams']:
        print(f"  '{phrase}': {count:,}")

    for tracked in stats.get('tracked_phrases', []):
        print("\n" + "=" * 80)
        print(f"TRACKED PHRASE: '{tracked['phrase']}'")
        print("=" * 80)
        print(f"\nMessages containing '{tracked['phrase']}': {tracked['count']:,}")
        if tracked['count'] > 0:
            print("\nBy user:")
            for user, info in tracked['by_user'].items():
                print(f"  {user}: {info['count']:,} ({info['pct']:.1f}%)")
            print("\nTop variants:")
            for variant, count in tracked['variants']:
                print(f"  '{variant}': {count:,}")
            print("\nRandom examples:")
            for ex in tracked['examples']:
                print(f"  [{ex['datetime'].strftime('%Y-%m-%d')}] {ex['sender']}: {ex['preview']}...")

    if 'domains' in stats:
        print("\n" + "-" * 80)
        print("TRACKED DOMAINS")
        print("-" * 80)
        for domain, info in stats['domains'].items():
            print(f"{domain}: {info['total']:,} total")
            for user, count in info['per_user'].items():
                print(f"  {user}: {count:,}")

    sentiment = stats.get('sentiment')
    if sentiment is not None:
        print("\n" + "=" * 80)
        print("SENTIMENT ANALYSIS")
        print("=" * 80)
        if sentiment.get('valid_message_count', 0) == 0:
            print("\nNo valid messages for sentiment analysis.")
        else:
            print(f"\nModel: {sentiment['model']}")
            print(f"Valid messages: {sentiment['valid_message_count']:,}")
            print(f"Sampled: {sentiment['sample_count']:,}")

            print("\n" + "-" * 80)
            print("OVERALL SENTIMENT DISTRIBUTION:")
            print("-" * 80)
            for label in sentiment['labels']:
                count = sentiment['overall_counts'][label]
                pct = count / sentiment['overall_total'] * 100 if sentiment['overall_total'] else 0
                print(f"  {label}: {count:,} messages ({pct:.1f}%)")

            print("\n" + "-" * 80)
            print("SENTIMENT BY USER:")
            print("-" * 80)
            for user, info in sentiment['per_user'].items():
                print(f"\n{user}:")
                for label in sentiment['labels']:
                    count = info['counts'][label]
                    pct = count / info['total'] * 100 if info['total'] else 0
                    print(f"  {label}: {count:,} ({pct:.1f}%)")
                print(f"  Average confidence score: {info['avg_confidence']:.3f}")

            print("\n" + "-" * 80)
            print("EXAMPLES OF EACH SENTIMENT:")
            print("-" * 80)
            for label, examples in sentiment['examples'].items():
                if examples:
                    print(f"\n{label.upper()} examples:")
                    for ex in examples:
                        print(f"  [{ex['datetime'].strftime('%Y-%m-%d')}] {ex['sender']}: {ex['preview']}")

    if 'anomalies' in stats:
        anomalies = stats['anomalies']
        print("\n" + "=" * 80)
        print("ANOMALY DETECTION")
        print("=" * 80)
        print(f"\nFound {anomalies['anomaly_count']} anomalous days (unusual activity patterns):\n")
        print(anomalies['top_anomalies'].to_string())

    summary = stats['summary']
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nConversation period:")
    print(f"  First message: {summary['first_message']['datetime'].strftime('%B %d, %Y')} by {summary['first_message']['sender']}")
    print(f"  Last message: {summary['last_message']['datetime'].strftime('%B %d, %Y')} by {summary['last_message']['sender']}")
    print(f"  Total days: {summary['total_days']}")
    print(f"\nTotal messages: {ov['total_messages']:,}")
    print(f"  Average per day: {summary['avg_per_day']:.1f}")
    print(f"\nMost active: {summary['leader']} ({summary['leader_pct']:.1f}% of messages)")
