from __future__ import annotations

import warnings
from collections import Counter

import pandas as pd

from .features import LINK_PATTERN

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_SAMPLE_SIZE = 2000
MIN_SAMPLE_SIZE = 500
SAMPLE_RATIO = 0.3
BATCH_SIZE = 16
MAX_CHARS = 512


def _select_valid_messages(df_simple):
    return df_simple[
        (df_simple['clean_message'].str.len() > 10)
        & (df_simple['word_count'] > 1)
        & (~df_simple['clean_message'].str.contains(LINK_PATTERN, na=False))
    ]


def _stratified_sample(valid_messages, users, sample_size, random_state=42):
    """Sample roughly evenly across users, topping up from the pool if some users fall short."""
    per_user_target = sample_size // max(len(users), 1)
    sampled = pd.DataFrame()

    for user in users:
        user_messages = valid_messages[valid_messages['sender'] == user]
        n = min(per_user_target, len(user_messages))
        if n > 0:
            sampled = pd.concat([sampled, user_messages.sample(n=n, random_state=random_state)])

    if len(sampled) < sample_size:
        remaining = valid_messages[~valid_messages.index.isin(sampled.index)]
        if len(remaining) > 0:
            extra_n = min(sample_size - len(sampled), len(remaining))
            sampled = pd.concat([sampled, remaining.sample(n=extra_n, random_state=random_state)])

    return sampled


def analyze_sentiment(df_simple, users, model_name=DEFAULT_MODEL, sample_size=DEFAULT_SAMPLE_SIZE,
                       random_state=42, progress=None):
    """Run sentiment inference over a stratified sample of messages.

    Labels are read from the model's own output rather than assumed, so this works
    with binary, 3-class, or star-rating sentiment models without extra config.
    Returns None if `transformers` isn't installed (an optional dependency).
    """
    try:
        from transformers import pipeline
    except ImportError:
        return None

    valid_messages = _select_valid_messages(df_simple)
    if len(valid_messages) == 0:
        return {'model': model_name, 'valid_message_count': 0}

    target_size = min(sample_size, int(len(valid_messages) * SAMPLE_RATIO))
    target_size = max(target_size, min(MIN_SAMPLE_SIZE, len(valid_messages)))

    sampled = _stratified_sample(valid_messages, users, target_size, random_state).copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=-1)

    texts = sampled['clean_message'].fillna('').tolist()
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t[:MAX_CHARS] for t in texts[i:i + BATCH_SIZE]]
        results.extend(sentiment_pipeline(batch))
        if progress:
            progress(min(i + BATCH_SIZE, len(texts)), len(texts))

    sampled['sentiment'] = [r['label'] for r in results]
    sampled['sentiment_score'] = [r['score'] for r in results]

    labels = sorted({r['label'] for r in results})
    overall_counts = Counter(sampled['sentiment'])

    per_user = {}
    for user in users:
        user_rows = sampled[sampled['sender'] == user]
        if len(user_rows) == 0:
            continue
        counts = Counter(user_rows['sentiment'])
        per_user[user] = {
            'counts': {label: counts.get(label, 0) for label in labels},
            'total': len(user_rows),
            'avg_confidence': float(user_rows['sentiment_score'].mean()),
        }

    examples = {}
    for label in labels:
        label_rows = sampled[sampled['sentiment'] == label].head(2)
        examples[label] = [
            {
                'datetime': row['datetime'],
                'sender': row['sender'],
                'preview': (row['clean_message'][:80] + '...') if len(row['clean_message']) > 80 else row['clean_message'],
            }
            for _, row in label_rows.iterrows()
        ]

    return {
        'model': model_name,
        'valid_message_count': len(valid_messages),
        'sample_count': len(sampled),
        'labels': labels,
        'overall_counts': {label: overall_counts.get(label, 0) for label in labels},
        'overall_total': len(sampled),
        'per_user': per_user,
        'examples': examples,
    }
