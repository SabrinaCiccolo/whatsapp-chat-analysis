from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
MAX_HEATMAP_PANELS = 6

plt.style.use('seaborn-v0_8-darkgrid')


def build_user_colors(users):
    palette = sns.color_palette('husl', len(users))
    return dict(zip(users, palette))


def plot_message_counts(df, message_type_counts, users, user_colors):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    message_counts = df['sender'].value_counts().reindex(users).fillna(0)
    ax1.bar(users, message_counts.values, color=[user_colors[u] for u in users])
    ax1.set_title('Total Messages by User', fontsize=14, fontweight='bold')
    ax1.set_xlabel('User')
    ax1.set_ylabel('Number of Messages')
    max_count = max(message_counts.values.max(), 1)
    for i, count in enumerate(message_counts.values):
        ax1.text(i, count + max_count * 0.01, f'{int(count):,}', ha='center', fontweight='bold')

    categories = ['Simple', 'Media', 'View-once Voice', 'Edited']
    bar_width = 0.8 / len(users)
    index = range(len(categories))
    for i, user in enumerate(users):
        counts = message_type_counts['per_user'].get(user, {})
        values = [counts.get('simple', 0), counts.get('media', 0), counts.get('voice', 0), counts.get('edited', 0)]
        offsets = [x + i * bar_width for x in index]
        ax2.bar(offsets, values, bar_width, label=user, color=user_colors[user])

    ax2.set_xlabel('Message Type')
    ax2.set_ylabel('Count')
    ax2.set_title('Message Types by User')
    ax2.set_xticks([x + bar_width * (len(users) - 1) / 2 for x in index])
    ax2.set_xticklabels(categories)
    ax2.legend()

    fig.tight_layout()
    return fig


def plot_activity_heatmaps(df, users, max_panels=MAX_HEATMAP_PANELS):
    panel_users = users
    if len(users) > max_panels:
        panel_users = df['sender'].value_counts().index[:max_panels].tolist()
        logger.warning(
            "%d users exceeds heatmap panel cap of %d; showing top %d most active: %s",
            len(users), max_panels, max_panels, ", ".join(panel_users),
        )

    fig, axes = plt.subplots(1, len(panel_users), figsize=(6 * len(panel_users), 6), squeeze=False)

    for ax, user in zip(axes[0], panel_users):
        user_df = df[df['sender'] == user]
        heatmap_data = user_df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        heatmap_data.index = heatmap_data.index.map(dict(enumerate(DAY_ORDER)))
        heatmap_data = heatmap_data.reindex(DAY_ORDER)

        sns.heatmap(
            heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Messages'},
            linewidths=0.5, linecolor='gray',
        )
        ax.set_title(f'{user} - Activity by Hour/Day', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('')

    fig.tight_layout()
    return fig


def plot_messages_over_time(df, users, user_colors):
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, user in enumerate(users):
        counts = df[df['sender'] == user].groupby('date_only').size()
        if len(counts) == 0:
            continue
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(
            list(counts.index), list(counts.values), label=user, marker=marker,
            markersize=3, linewidth=2, color=user_colors[user],
        )

    ax.set_title('Message Frequency Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Messages per Day')
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_plots(df, stats, users, mode="save", output_dir="outputs"):
    """Build all figures. mode is 'show', 'save', or 'none'. Returns saved file paths (empty for show/none)."""
    if mode == "none":
        return []

    user_colors = build_user_colors(users)
    figures = {
        'fig1_message_counts': plot_message_counts(df, stats['message_types'], users, user_colors),
        'fig2_activity_heatmaps': plot_activity_heatmaps(df, users),
        'fig3_messages_over_time': plot_messages_over_time(df, users, user_colors),
    }

    if mode == "show":
        plt.show()
        for fig in figures.values():
            plt.close(fig)
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, fig in figures.items():
        path = output_path / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    return saved
