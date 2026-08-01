import pandas as pd

from whatsapp_analysis.parsing import parse_chat_file, add_datetime_features
from whatsapp_analysis.features import classify_messages, detect_bursts, calculate_response_times
from whatsapp_analysis.search import search_phrase, print_search_result, run_interactive_search


def _featurized_df(fixture_path, name="chat_basic.txt"):
    messages = parse_chat_file(fixture_path(name))
    df = pd.DataFrame(messages)
    df = add_datetime_features(df)
    df = classify_messages(df)
    df = detect_bursts(df)
    df = calculate_response_times(df)
    return df


def test_search_phrase_finds_match(fixture_path):
    df = _featurized_df(fixture_path)
    result = search_phrase(df, "example.com", ['Alice', 'Bob'])
    assert result['total'] == 1
    assert result['by_user']['Alice']['count'] == 1
    assert result['by_user']['Bob']['count'] == 0
    assert len(result['examples']) == 1


def test_search_phrase_no_match(fixture_path):
    df = _featurized_df(fixture_path)
    result = search_phrase(df, "nonexistent", ['Alice', 'Bob'])
    assert result['total'] == 0
    assert result['examples'] == []


def test_search_phrase_is_literal_not_regex(fixture_path):
    df = _featurized_df(fixture_path)
    # "." in "example.com" must be treated literally, not as regex "any char"
    result = search_phrase(df, "examplexcom", ['Alice', 'Bob'])
    assert result['total'] == 0


def test_print_search_result_no_crash_on_empty(capsys):
    print_search_result({'phrase': 'xyz', 'total': 0, 'by_user': {}, 'examples': []})
    out = capsys.readouterr().out
    assert "No messages found" in out


def test_run_interactive_search_exits_on_empty_input(fixture_path, capsys):
    df = _featurized_df(fixture_path)
    inputs = iter(["thanks", ""])
    run_interactive_search(df, ['Alice', 'Bob'], input_func=lambda _: next(inputs))
    out = capsys.readouterr().out
    assert "Total messages containing 'thanks': 1" in out
    assert "Exiting search." in out
