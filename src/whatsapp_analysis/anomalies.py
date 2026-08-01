from __future__ import annotations

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import to_minutes

DEFAULT_CONTAMINATION = 0.1
DEFAULT_RANDOM_STATE = 42


def compute_daily_stats(df):
    daily_stats = df.groupby('date_only').agg({
        'message': 'count',
        'response_time': 'mean',
        'word_count': 'sum',
    }).fillna(0)
    daily_stats.columns = ['message_count', 'avg_response_time', 'total_words']
    daily_stats['avg_response_time'] = daily_stats['avg_response_time'].apply(to_minutes)
    return daily_stats


def detect_anomalies(df, contamination=DEFAULT_CONTAMINATION, random_state=DEFAULT_RANDOM_STATE, top_n=10):
    """Flag days with unusual message/response/word-count activity via IsolationForest."""
    daily_stats = compute_daily_stats(df)

    if len(daily_stats) < 2:
        return {'daily_stats': daily_stats, 'top_anomalies': daily_stats.iloc[0:0], 'anomaly_count': 0}

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(daily_stats)

    iso_forest = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=1)
    daily_stats = daily_stats.copy()
    daily_stats['anomaly'] = iso_forest.fit_predict(features_scaled)
    daily_stats['anomaly_score'] = iso_forest.score_samples(features_scaled)

    anomalies = daily_stats[daily_stats['anomaly'] == -1].sort_values('anomaly_score')

    return {
        'daily_stats': daily_stats,
        'top_anomalies': anomalies.head(top_n),
        'anomaly_count': len(anomalies),
    }
