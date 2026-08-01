import json

from whatsapp_analysis.cli import main


def test_analyze_runs_end_to_end(fixture_path, capsys):
    main(["analyze", str(fixture_path("chat_basic.txt")), "--plots", "none", "--export", "none"])
    out = capsys.readouterr().out
    assert "Total messages: 6" in out
    assert "Alice" in out
    assert "Bob" in out


def test_analyze_respects_users_filter(fixture_path, capsys):
    main([
        "analyze", str(fixture_path("chat_group_3users.txt")),
        "--users", "Alice", "Bob", "--plots", "none", "--export", "none",
    ])
    out = capsys.readouterr().out
    assert "Carol" not in out


def test_analyze_saves_plots(fixture_path, tmp_path, capsys):
    main([
        "analyze", str(fixture_path("chat_group_3users.txt")),
        "--output-dir", str(tmp_path), "--export", "none",
    ])
    out = capsys.readouterr().out
    assert "Saved plots:" in out
    saved = sorted(p.name for p in tmp_path.glob("*.png"))
    assert saved == ["fig1_message_counts.png", "fig2_activity_heatmaps.png", "fig3_messages_over_time.png"]


def test_analyze_exports_json_and_csv(fixture_path, tmp_path, capsys):
    main([
        "analyze", str(fixture_path("chat_group_3users.txt")),
        "--output-dir", str(tmp_path), "--plots", "none", "--export", "both",
    ])
    out = capsys.readouterr().out
    assert "Exported:" in out
    assert (tmp_path / "stats.json").exists()
    assert (tmp_path / "user_stats.csv").exists()

    data = json.loads((tmp_path / "stats.json").read_text())
    assert data['overview']['total_messages'] == 5


def test_search_one_shot_phrase(fixture_path, capsys):
    main(["search", str(fixture_path("chat_basic.txt")), "--phrase", "example.com"])
    out = capsys.readouterr().out
    assert "Total messages containing 'example.com': 1" in out


def test_config_file_layers_under_cli_flags(fixture_path, tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"burst_threshold": 1}))

    main([
        "analyze", str(fixture_path("chat_basic.txt")),
        "--config", str(config_path), "--plots", "none", "--export", "none",
    ])
    out = capsys.readouterr().out
    assert "Total messages: 6" in out
