from __future__ import annotations

MAX_EXAMPLES = 10


def search_phrase(df, phrase, users, max_examples=MAX_EXAMPLES):
    """Search an already-featurized dataframe for a literal phrase (case-insensitive)."""
    mask = df['clean_message'].str.contains(phrase, case=False, na=False, regex=False)
    matches = df[mask]
    total = len(matches)

    if total == 0:
        return {'phrase': phrase, 'total': 0, 'by_user': {}, 'examples': []}

    by_user = {
        user: {
            'count': int((matches['sender'] == user).sum()),
            'pct': float((matches['sender'] == user).sum() / total * 100),
        }
        for user in users
    }

    examples_df = matches.sort_values('datetime', ascending=False).head(max_examples)
    examples = [
        {
            'datetime': row['datetime'],
            'sender': row['sender'],
            'preview': _preview(row['clean_message']),
        }
        for _, row in examples_df.iterrows()
    ]

    day_counts = matches.groupby('date_only').size().sort_values(ascending=False)
    top_day = (str(day_counts.index[0]), int(day_counts.iloc[0])) if len(day_counts) else None

    hour_counts = matches.groupby('hour').size()
    top_hour = (int(hour_counts.idxmax()), int(hour_counts.max())) if len(hour_counts) else None

    return {
        'phrase': phrase,
        'total': total,
        'by_user': by_user,
        'examples': examples,
        'first_occurrence': matches['datetime'].min(),
        'last_occurrence': matches['datetime'].max(),
        'top_day': top_day,
        'top_hour': top_hour,
        'pct_of_all': float(total / len(df) * 100) if len(df) else 0.0,
    }


def _preview(message, limit=100):
    text = str(message).replace('\n', ' ')
    return text[:limit - 3] + '...' if len(text) > limit else text


def print_search_result(result):
    print(f"\nSearching for: '{result['phrase']}'")
    print(f"\nTotal messages containing '{result['phrase']}': {result['total']:,}")

    if result['total'] == 0:
        print(f"\nNo messages found containing '{result['phrase']}'.")
        return

    print("\nMessages by user:")
    for user, info in result['by_user'].items():
        print(f"  {user}: {info['count']:,} ({info['pct']:.1f}%)")

    print(f"\nShowing {len(result['examples'])} examples:")
    print("-" * 80)
    for ex in result['examples']:
        print(f"[{ex['datetime'].strftime('%Y-%m-%d %H:%M')}] {ex['sender']}:")
        print(f"  {ex['preview']}")
        print("-" * 80)

    print("\nAdditional statistics:")
    print(f"  First occurrence: {result['first_occurrence'].strftime('%B %d, %Y')}")
    print(f"  Last occurrence: {result['last_occurrence'].strftime('%B %d, %Y')}")
    if result['top_day']:
        print(f"  Day with most mentions: {result['top_day'][0]} ({result['top_day'][1]} times)")
    if result['top_hour']:
        print(f"  Most common hour: {result['top_hour'][0]}:00 ({result['top_hour'][1]} times)")
    print(f"  Percentage of all messages: {result['pct_of_all']:.2f}%")


def run_interactive_search(df, users, input_func=input):
    """Prompt for phrases and print results, looping until the user presses Enter with none."""
    while True:
        phrase = input_func("\nEnter a word or phrase to search for (or press Enter to exit): ").strip()
        if not phrase:
            print("\nExiting search.")
            return
        print_search_result(search_phrase(df, phrase, users))
