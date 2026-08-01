from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from whatsapp_analysis.parsing import parse_chat_file, add_datetime_features, LINE_FORMATS
from whatsapp_analysis.features import classify_messages, detect_bursts, calculate_response_times
from whatsapp_analysis.stats import compute_all_stats
from whatsapp_analysis.plotting import (
    build_user_colors, plot_message_counts, plot_activity_heatmaps, plot_messages_over_time,
)
from whatsapp_analysis.anomalies import detect_anomalies
from whatsapp_analysis.search import search_phrase

st.set_page_config(page_title="WhatsApp Chat Analysis", layout="wide")
st.title("WhatsApp Chat Analysis")

uploaded = st.sidebar.file_uploader("Chat export (.txt)", type="txt")
if not uploaded:
    st.info("Upload a WhatsApp chat export (.txt) to begin.")
    st.stop()

line_format_choice = st.sidebar.selectbox("Line format", ["auto"] + [fmt.name for fmt in LINE_FORMATS])
date_order = st.sidebar.selectbox("Date order", ["auto", "mdy", "dmy"])

tmp_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    line_format = None if line_format_choice == "auto" else line_format_choice
    messages = parse_chat_file(tmp_path, line_format=line_format)
finally:
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

if not messages:
    st.error(f"No messages parsed from '{uploaded.name}'.")
    st.stop()

df = pd.DataFrame(messages)
df = add_datetime_features(df, date_order=date_order)

all_users = sorted(df['sender'].unique())
users = st.sidebar.multiselect("Users", all_users, default=all_users)
burst_threshold = st.sidebar.number_input("Burst threshold", min_value=1, value=10)
conversation_gap_hours = st.sidebar.number_input("Conversation gap (hours)", min_value=0.5, value=4.0)
track_phrase = st.sidebar.text_input("Track phrase (optional)")
track_domain = st.sidebar.text_input("Track domain (optional)")
run_anomalies = st.sidebar.checkbox("Anomaly detection", value=True)

if not users:
    st.warning("Select at least one user.")
    st.stop()

df = classify_messages(df)
df = detect_bursts(df, threshold=burst_threshold)
df = calculate_response_times(df)
df = df[df['sender'].isin(users)].reset_index(drop=True)

stats = compute_all_stats(
    df, users,
    track_domains=[track_domain] if track_domain else None,
    track_phrases=[track_phrase] if track_phrase else None,
    conversation_gap_hours=conversation_gap_hours,
    burst_threshold=burst_threshold,
)
if run_anomalies:
    stats['anomalies'] = detect_anomalies(df)

overview = stats['overview']
summary = stats['summary']
c1, c2, c3, c4 = st.columns(4)
c1.metric("Messages", f"{overview['total_messages']:,}")
c2.metric("Days", overview['total_days'])
c3.metric("Messages/day", f"{overview['messages_per_day']:.1f}")
c4.metric("Most active", f"{summary['leader']} ({summary['leader_pct']:.0f}%)")

st.subheader("Per-user stats")
rows = []
for user in users:
    row = {'user': user, **stats['message_types']['per_user'].get(user, {})}
    row.update(stats['user_stats'].get(user) or {})
    rows.append(row)
st.dataframe(pd.DataFrame(rows).set_index('user'), use_container_width=True)

st.subheader("Activity")
user_colors = build_user_colors(users)
st.pyplot(plot_message_counts(df, stats['message_types'], users, user_colors))
st.pyplot(plot_activity_heatmaps(df, users))
st.pyplot(plot_messages_over_time(df, users, user_colors))

col_emoji, col_phrases = st.columns(2)
with col_emoji:
    st.subheader("Top emojis")
    if stats['emojis']['overall']:
        st.dataframe(pd.DataFrame(stats['emojis']['overall'], columns=['emoji', 'count']), use_container_width=True)
    else:
        st.caption("No emojis found.")
with col_phrases:
    st.subheader("Common phrases")
    st.dataframe(pd.DataFrame(stats['phrases']['bigrams'][:10], columns=['phrase', 'count']), use_container_width=True)

for tracked in stats.get('tracked_phrases', []):
    st.subheader(f"Tracked phrase: '{tracked['phrase']}'")
    st.write(f"{tracked['count']:,} messages")
    if tracked['by_user']:
        st.dataframe(pd.DataFrame(tracked['by_user']).T, use_container_width=True)

if 'domains' in stats:
    st.subheader("Tracked domains")
    for domain, info in stats['domains'].items():
        st.write(f"**{domain}**: {info['total']:,} total — {info['per_user']}")

if 'anomalies' in stats and stats['anomalies']['anomaly_count'] > 0:
    st.subheader("Anomalous days")
    st.dataframe(stats['anomalies']['top_anomalies'], use_container_width=True)

st.subheader("Search")
query = st.text_input("Search phrase")
if query:
    result = search_phrase(df, query, users)
    st.write(f"Found {result['total']:,} message(s)")
    if result['examples']:
        st.dataframe(pd.DataFrame(result['examples']), use_container_width=True)
