from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import AnalysisConfig, load_config
from .parsing import parse_chat_file, add_datetime_features, LINE_FORMATS
from .features import classify_messages, detect_bursts, calculate_response_times
from .stats import compute_all_stats, print_report, split_message_types
from .plotting import render_plots
from .export import export_stats
from .anomalies import detect_anomalies
from .sentiment import analyze_sentiment, DEFAULT_MODEL as DEFAULT_SENTIMENT_MODEL
from .search import search_phrase, print_search_result, run_interactive_search


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="whatsapp-analysis", description="Analyze WhatsApp chat exports."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_args(sub):
        sub.add_argument("chat_file", help="Path to the WhatsApp chat export .txt file")
        sub.add_argument(
            "--users", nargs="+", default=None, help="Restrict/order analysis to these sender names"
        )
        sub.add_argument("--date-order", choices=["auto", "mdy", "dmy"], default="auto")
        sub.add_argument(
            "--line-format", choices=[fmt.name for fmt in LINE_FORMATS], default=None
        )
        sub.add_argument("--config", dest="config_path", default=None, help="Path to a JSON config file")
        sub.add_argument("--output-dir", default="outputs")

    analyze = subparsers.add_parser("analyze", help="Run the analysis pipeline")
    add_shared_args(analyze)
    analyze.add_argument("--burst-threshold", type=int, default=10)
    analyze.add_argument("--conversation-gap-hours", type=float, default=4.0)
    analyze.add_argument(
        "--track-phrase", dest="track_phrases", action="append", default=[],
        help="Phrase to track, e.g. 'ti amo' (repeatable)",
    )
    analyze.add_argument(
        "--track-domain", dest="track_domains", action="append", default=[],
        help="Link domain to track, e.g. meet.google.com (repeatable)",
    )
    analyze.add_argument("--plots", choices=["show", "save", "none"], default="save")
    analyze.add_argument("--export", choices=["json", "csv", "both", "none"], default="json")
    analyze.add_argument(
        "--no-anomalies", dest="anomalies", action="store_false", default=True,
        help="Skip anomaly detection",
    )
    analyze.add_argument(
        "--sentiment", action="store_true", default=False,
        help="Run sentiment analysis (requires the [sentiment] extra: transformers + torch)",
    )
    analyze.add_argument("--sentiment-model", default=DEFAULT_SENTIMENT_MODEL)

    search = subparsers.add_parser("search", help="Search messages for a phrase")
    add_shared_args(search)
    search.add_argument("--phrase", default=None, help="Phrase to search for (omit for interactive mode)")

    return parser, {"analyze": analyze, "search": search}


def _config_from_args(args):
    cfg = AnalysisConfig(chat_file=args.chat_file)
    cfg.users = args.users
    cfg.date_order = args.date_order
    cfg.line_format = args.line_format
    cfg.output_dir = args.output_dir
    if hasattr(args, "burst_threshold"):
        cfg.burst_threshold = args.burst_threshold
    if hasattr(args, "conversation_gap_hours"):
        cfg.conversation_gap_hours = args.conversation_gap_hours
    cfg.track_phrases = getattr(args, "track_phrases", None) or []
    cfg.track_domains = getattr(args, "track_domains", None) or []
    if hasattr(args, "plots"):
        cfg.plots = args.plots
    if hasattr(args, "export"):
        cfg.export = args.export
    if hasattr(args, "anomalies"):
        cfg.anomalies = args.anomalies
    if hasattr(args, "sentiment"):
        cfg.sentiment = args.sentiment
    if hasattr(args, "sentiment_model"):
        cfg.sentiment_model = args.sentiment_model
    return cfg


def _load_dataframe(cfg: AnalysisConfig) -> pd.DataFrame:
    messages = parse_chat_file(cfg.chat_file, line_format=cfg.line_format)
    if not messages:
        raise ValueError(f"No messages parsed from {cfg.chat_file}")

    df = pd.DataFrame(messages)
    df = add_datetime_features(df, date_order=cfg.date_order)
    df = classify_messages(df)
    df = detect_bursts(df, threshold=cfg.burst_threshold if hasattr(cfg, "burst_threshold") else 10)
    df = calculate_response_times(df)

    if cfg.users:
        df = df[df['sender'].isin(cfg.users)].reset_index(drop=True)

    return df


def _resolve_users(df, cfg):
    return cfg.users if cfg.users else sorted(df['sender'].unique())


def run_analyze(args):
    cfg = _config_from_args(args)
    df = _load_dataframe(cfg)
    users = _resolve_users(df, cfg)

    print(f"File: {cfg.chat_file}")
    print(f"Users: {', '.join(users)}")

    stats = compute_all_stats(
        df,
        users,
        track_domains=cfg.track_domains,
        track_phrases=cfg.track_phrases,
        conversation_gap_hours=cfg.conversation_gap_hours,
        burst_threshold=cfg.burst_threshold,
    )
    if cfg.anomalies:
        stats['anomalies'] = detect_anomalies(df)

    if cfg.sentiment:
        df_simple, _, _, _ = split_message_types(df)
        sentiment_result = analyze_sentiment(df_simple, users, model_name=cfg.sentiment_model,
                                              sample_size=cfg.sentiment_sample_size)
        if sentiment_result is None:
            print("\nSentiment analysis skipped: install with `pip install -e .[sentiment]`")
        else:
            stats['sentiment'] = sentiment_result

    print_report(stats, users)

    saved_plots = render_plots(df, stats, users, mode=cfg.plots, output_dir=cfg.output_dir)
    if saved_plots:
        print("\nSaved plots:")
        for path in saved_plots:
            print(f"  {path}")

    saved_exports = export_stats(stats, cfg.export, cfg.output_dir)
    if saved_exports:
        print("\nExported:")
        for path in saved_exports:
            print(f"  {path}")


def run_search(args):
    cfg = _config_from_args(args)
    df = _load_dataframe(cfg)
    users = _resolve_users(df, cfg)

    if args.phrase:
        print_search_result(search_phrase(df, args.phrase, users))
    else:
        run_interactive_search(df, users)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", dest="config_path", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser, subparsers_by_name = _build_parser()

    if pre_args.config_path:
        config_overrides = load_config(pre_args.config_path)
        for sub in subparsers_by_name.values():
            sub.set_defaults(**config_overrides)

    args = parser.parse_args(argv)

    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "search":
        run_search(args)


if __name__ == "__main__":
    main()
