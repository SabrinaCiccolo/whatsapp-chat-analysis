import json

import pandas as pd

from whatsapp_analysis.parsing import parse_chat_file, add_datetime_features
from whatsapp_analysis.features import classify_messages, detect_bursts, calculate_response_times
from whatsapp_analysis.stats import compute_all_stats
from whatsapp_analysis.export import export_json, export_csv, export_stats
from whatsapp_analysis.anomalies import detect_anomalies


def _stats(fixture_path):
    messages = parse_chat_file(fixture_path("chat_basic.txt"))
    df = pd.DataFrame(messages)
    df = add_datetime_features(df)
    df = classify_messages(df)
    df = detect_bursts(df)
    df = calculate_response_times(df)
    return compute_all_stats(df, ['Alice', 'Bob'], seed=42)


def test_export_json_writes_valid_file(fixture_path, tmp_path):
    stats = _stats(fixture_path)
    path = export_json(stats, output_dir=tmp_path)

    data = json.loads(path.read_text())
    assert data['overview']['total_messages'] == 6


def test_export_csv_writes_tables(fixture_path, tmp_path):
    stats = _stats(fixture_path)
    saved = export_csv(stats, output_dir=tmp_path)

    names = {p.name for p in saved}
    assert 'user_stats.csv' in names
    assert 'emoji_counts.csv' in names
    assert 'common_phrases.csv' in names


def test_export_json_serializes_anomalies_dataframe(fixture_path, tmp_path):
    stats = _stats(fixture_path)
    df = pd.DataFrame(parse_chat_file(fixture_path("chat_basic.txt")))
    df = add_datetime_features(df)
    df = classify_messages(df)
    df = detect_bursts(df)
    df = calculate_response_times(df)
    stats['anomalies'] = detect_anomalies(df)

    path = export_json(stats, output_dir=tmp_path)
    data = json.loads(path.read_text())
    assert 'anomalies' in data
    assert isinstance(data['anomalies']['top_anomalies'], list)


def test_export_stats_none_writes_nothing(fixture_path, tmp_path):
    stats = _stats(fixture_path)
    saved = export_stats(stats, "none", output_dir=tmp_path)
    assert saved == []
    assert list(tmp_path.iterdir()) == []
