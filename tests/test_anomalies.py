import pandas as pd

from whatsapp_analysis.anomalies import compute_daily_stats, detect_anomalies


def _synthetic_df(n_days=15, spike_day_messages=200):
    rows = []
    for day in range(n_days):
        count = spike_day_messages if day == n_days // 2 else 10
        for i in range(count):
            rows.append({
                'date_only': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                'message': f'msg {i}',
                'response_time': pd.Timedelta(minutes=1),
                'word_count': 3,
            })
    return pd.DataFrame(rows)


def test_compute_daily_stats_shape():
    df = _synthetic_df()
    daily = compute_daily_stats(df)
    assert list(daily.columns) == ['message_count', 'avg_response_time', 'total_words']
    assert daily['message_count'].max() == 200


def test_detect_anomalies_flags_spike_day():
    df = _synthetic_df()
    result = detect_anomalies(df)
    assert result['anomaly_count'] >= 1
    assert result['top_anomalies']['message_count'].max() == 200


def test_detect_anomalies_handles_single_day_gracefully():
    df = _synthetic_df(n_days=1, spike_day_messages=5)
    result = detect_anomalies(df)
    assert result['anomaly_count'] == 0
