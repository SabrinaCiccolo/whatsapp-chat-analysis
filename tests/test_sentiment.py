import builtins

import pandas as pd
import pytest

from whatsapp_analysis.sentiment import analyze_sentiment, _select_valid_messages, _stratified_sample


def test_select_valid_messages_filters_short_and_link_messages():
    df = pd.DataFrame([
        {'clean_message': 'ok', 'word_count': 1},
        {'clean_message': 'this is a long enough message', 'word_count': 6},
        {'clean_message': 'check http://example.com out today', 'word_count': 5},
    ])
    valid = _select_valid_messages(df)
    assert len(valid) == 1
    assert valid.iloc[0]['clean_message'] == 'this is a long enough message'


def test_stratified_sample_covers_all_users():
    rows = [{'sender': 'Alice', 'clean_message': f'msg {i}'} for i in range(20)]
    rows += [{'sender': 'Bob', 'clean_message': f'msg {i}'} for i in range(5)]
    df = pd.DataFrame(rows)

    sample = _stratified_sample(df, ['Alice', 'Bob'], sample_size=10, random_state=1)
    assert 'Alice' in sample['sender'].values
    assert 'Bob' in sample['sender'].values


def test_analyze_sentiment_returns_none_without_transformers(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'transformers':
            raise ImportError("no transformers installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    df = pd.DataFrame({'clean_message': ['hello there friend'], 'word_count': [3], 'sender': ['Alice']})
    result = analyze_sentiment(df, ['Alice'])
    assert result is None
