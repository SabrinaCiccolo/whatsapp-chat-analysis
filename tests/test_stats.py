import pandas as pd
import pytest

from whatsapp_analysis.parsing import parse_chat_file, add_datetime_features
from whatsapp_analysis.features import classify_messages, detect_bursts, calculate_response_times
from whatsapp_analysis.stats import (
    split_message_types,
    compute_overview,
    compute_user_stats,
    compute_active_hours,
    compute_conversation_starters,
    compute_emoji_stats,
    compute_phrase_tracking,
    compute_summary,
    compute_all_stats,
)


@pytest.fixture
def featurized_df(fixture_path):
    messages = parse_chat_file(fixture_path("chat_basic.txt"))
    df = pd.DataFrame(messages)
    df = add_datetime_features(df)
    df = classify_messages(df)
    df = detect_bursts(df)
    df = calculate_response_times(df)
    return df


def test_compute_overview(featurized_df):
    overview = compute_overview(featurized_df)
    assert overview['total_messages'] == 6


def test_compute_user_stats(featurized_df):
    df_simple, _, _, _ = split_message_types(featurized_df)
    stats = compute_user_stats(df_simple, ['Alice', 'Bob'])
    assert stats['Alice']['total_emojis'] == 0
    assert stats['Bob']['total_emojis'] == 1


def test_compute_active_hours(featurized_df):
    result = compute_active_hours(featurized_df, ['Alice', 'Bob'])
    assert result['Alice']['peak_hour'] == 9


def test_compute_conversation_starters_first_message_always_a_start(featurized_df):
    starters = compute_conversation_starters(featurized_df, gap_hours=4.0)
    assert sum(v['count'] for v in starters.values()) >= 1
    assert 'Alice' in starters  # Alice sent the very first message


def test_compute_emoji_stats(featurized_df):
    df_simple, _, _, _ = split_message_types(featurized_df)
    result = compute_emoji_stats(df_simple, ['Alice', 'Bob'])
    assert result['overall'] == [('😄', 1)]


def test_compute_phrase_tracking_generalizes_love_expression(featurized_df):
    df_simple, _, _, _ = split_message_types(featurized_df)
    result = compute_phrase_tracking(df_simple, "thanks")
    assert result['count'] == 1
    assert result['by_user']['Alice']['count'] == 1


def test_compute_phrase_tracking_no_matches(featurized_df):
    df_simple, _, _, _ = split_message_types(featurized_df)
    result = compute_phrase_tracking(df_simple, "nonexistent phrase")
    assert result['count'] == 0
    assert result['examples'] == []


def test_compute_summary_leader(featurized_df):
    summary = compute_summary(featurized_df)
    assert summary['leader'] in ('Alice', 'Bob')


def test_compute_all_stats_smoke(featurized_df):
    stats = compute_all_stats(
        featurized_df, ['Alice', 'Bob'],
        track_domains=['example.com'], track_phrases=['thanks'], seed=42,
    )
    assert stats['overview']['total_messages'] == 6
    assert 'domains' in stats
    assert 'tracked_phrases' in stats
