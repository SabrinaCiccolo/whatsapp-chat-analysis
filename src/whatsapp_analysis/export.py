from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return json.loads(obj.reset_index().to_json(orient='records', date_format='iso'))
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def export_json(stats, output_dir="outputs", filename="stats.json"):
    """Write the full stats dict as JSON. This is the canonical, complete export."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=_json_default, ensure_ascii=False)

    return path


def export_csv(stats, output_dir="outputs"):
    """Write a handful of flat CSV tables. Not a complete export (JSON is) —
    just the parts that are naturally tabular."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = []

    user_rows = [
        {'user': user, **user_stats}
        for user, user_stats in stats['user_stats'].items()
        if user_stats is not None
    ]
    if user_rows:
        path = output_path / 'user_stats.csv'
        pd.DataFrame(user_rows).to_csv(path, index=False)
        saved.append(path)

    emoji_rows = [
        {'scope': 'overall', 'emoji': emoji_char, 'count': count}
        for emoji_char, count in stats['emojis']['overall']
    ]
    for user, items in stats['emojis']['per_user'].items():
        emoji_rows.extend({'scope': user, 'emoji': emoji_char, 'count': count} for emoji_char, count in items)
    if emoji_rows:
        path = output_path / 'emoji_counts.csv'
        pd.DataFrame(emoji_rows).to_csv(path, index=False)
        saved.append(path)

    phrase_rows = [
        {'type': 'bigram', 'phrase': phrase, 'count': count}
        for phrase, count in stats['phrases']['bigrams']
    ]
    phrase_rows.extend(
        {'type': 'trigram', 'phrase': phrase, 'count': count}
        for phrase, count in stats['phrases']['trigrams']
    )
    if phrase_rows:
        path = output_path / 'common_phrases.csv'
        pd.DataFrame(phrase_rows).to_csv(path, index=False)
        saved.append(path)

    top_anomalies = stats.get('anomalies', {}).get('top_anomalies')
    if top_anomalies is not None and len(top_anomalies) > 0:
        path = output_path / 'anomalous_days.csv'
        top_anomalies.to_csv(path)
        saved.append(path)

    return saved


def export_stats(stats, formats, output_dir="outputs"):
    """formats: 'json' | 'csv' | 'both' | 'none'."""
    if formats == "none":
        return []

    saved = []
    if formats in ("json", "both"):
        saved.append(export_json(stats, output_dir))
    if formats in ("csv", "both"):
        saved.extend(export_csv(stats, output_dir))
    return saved
