# import random
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import streamlit as st


# def load_history(csv_path: str, _mtime: float) -> pd.DataFrame:
#     """
#     Read history.csv, parse dates, compute missing duration,
#     derive hour/date/length columns, and return DataFrame.
#     """
#     df = pd.read_csv(
#         csv_path,
#         parse_dates=["timestamp", "start_time", "end_time"],
#         dtype={"task": str, "assistant": str},
#         na_filter=False,
#     )

#     if df.empty:
#         return df

#     # 1) Compute duration if missing
#     if "duration" not in df.columns or df["duration"].isna().all():
#         df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
#     else:
#         df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

#     # 2) Drop rows missing any critical column
#     df = df.dropna(
#         subset=["task", "assistant", "generated_code", "start_time", "end_time", "duration"]
#     )

#     # 3) Fill missing timestamps with start_time
#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(df["start_time"])

#     # 4) Derive additional columns for charts
#     df["hour"] = df["timestamp"].dt.floor("h")
#     df["date"] = df["timestamp"].dt.date
#     df["length"] = df["generated_code"].astype(str).str.len()

#     if "accuracy_rating" in df.columns:
#         df["accuracy_rating"] = pd.to_numeric(df["accuracy_rating"], errors="coerce")
#     else:
#         df["accuracy_rating"] = np.nan

#     return df


# # --------------------------- Streamlit Setup -----------------------------
# st.set_page_config(
#     page_title="Smart AI Coding Assistant Dashboard",
#     layout="wide",
#     page_icon="🤖",
# )

# st.markdown(
#     """
#     <style>
#     .stApp {
#         background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #0f2027);
#         background-size: 400% 400%;
#         animation: gradientBG 20s ease infinite;
#     }
#     @keyframes gradientBG {
#         0%   { background-position: 0% 50%; }
#         50%  { background-position: 100% 50%; }
#         100% { background-position: 0% 50%; }
#     }
#     h1, h2, h3, h4 {
#         color: #ffffff !important;
#         text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
#     }
#     .css-1d391kg .stMetricLabel, .css-1d391kg .stMetricValue {
#         color: #ffffff !important;
#     }
#     .stDataFrame, .stTable {
#         color: #ffffff !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.title("🤖 Smart AI Coding Assistant Dashboard")

# # ——— Load Data ——————————————————————————————————————————————————————————
# DATA_PATH = Path(__file__).parent.parent / "data" / "history.csv"
# if not DATA_PATH.exists():
#     st.error(
#         "⚠️ Cannot find data/history.csv. Please run your watcher to generate at least one row."
#     )
#     st.stop()

# file_mtime = DATA_PATH.stat().st_mtime
# df = load_history(str(DATA_PATH), file_mtime)

# if df.empty:
#     st.warning(
#         "No valid rows found in history.csv. Make a change in scripts/demo.py so the watcher generates data."
#     )
#     st.stop()

# # ——— Sidebar Metrics & Filtering —————————————————————————————————————————
# with st.sidebar:
#     st.header("📊 Metrics Summary")

#     total_requests = len(df)
#     unique_tasks = df["task"].nunique()
#     avg_length = df["length"].mean() if total_requests > 0 else 0.0
#     avg_duration = df["duration"].mean() if "duration" in df.columns else 0.0
#     med_duration = df["duration"].median() if "duration" in df.columns else 0.0
#     max_duration = df["duration"].max() if "duration" in df.columns else 0.0

#     rated_df = df.dropna(subset=["accuracy_rating"])
#     if not rated_df.empty:
#         success_pct = (rated_df["accuracy_rating"] > 0).sum() / len(rated_df) * 100
#     else:
#         success_pct = np.nan

#     st.metric("Total Requests", total_requests)
#     st.metric("Unique Tasks", unique_tasks)
#     st.metric("Avg. Code Length", f"{avg_length:.0f} chars")
#     st.metric("Avg. Duration", f"{avg_duration:.3f} s")
#     st.metric("Median Duration", f"{med_duration:.3f} s")
#     st.metric("Max Duration", f"{max_duration:.3f} s")
#     if not np.isnan(success_pct):
#         st.metric("Success % (rating > 0)", f"{success_pct:.1f}%")
#     else:
#         st.metric("Success % (rating > 0)", "N/A")

#     st.markdown("---")
#     st.write("**Filter by Assistant(s)**")
#     all_assistants = sorted(df["assistant"].unique())
#     selected_assistants = st.multiselect(
#         "Select assistant(s)",
#         all_assistants,
#         default=all_assistants,
#     )

# if not selected_assistants:
#     st.warning("Please select at least one assistant before viewing charts.")
#     st.stop()

# filtered = df[df["assistant"].isin(selected_assistants)].copy()

# # ——— Main Content: Overview Charts —————————————————————————————————————
# st.markdown("## 📈 Overview Charts", unsafe_allow_html=True)

# # ─── 1. Requests per Assistant ────────────────────────────────────────────────
# st.subheader("Requests per Assistant")

# fig_requests = px.histogram(
#     filtered,
#     x="assistant",
#     color="assistant",
#     barmode="group",
#     template="plotly_dark",
#     color_discrete_sequence=["#8FBCE6", "#1F77B4"],  # light-blue, dark-blue
#     labels={"assistant": "Assistant", "count": "Requests"},
# )
# fig_requests.update_layout(
#     title_text="Number of Requests by Assistant",
#     yaxis=dict(
#         title="Requests",
#         range=[0, filtered["assistant"].value_counts().max() * 1.2],
#     ),
#     legend_title_text="Assistant",
#     height=400,
# )
# fig_requests.update_traces(texttemplate="%{y}", textposition="outside")
# st.plotly_chart(fig_requests, use_container_width=True)

# # ─── 2. Task Diversity by Assistant ─────────────────────────────────────────
# st.subheader("Task Diversity by Assistant")

# # Build DataFrame of (assistant, task) pairs, drop duplicates
# unique_tasks_df = filtered[["assistant", "task"]].drop_duplicates()

# # Use a histogram to count unique tasks per assistant
# fig_task_div = px.histogram(
#     unique_tasks_df,
#     x="assistant",
#     histfunc="count",
#     color="assistant",
#     template="plotly_dark",
#     color_discrete_sequence=["#8FBCE6", "#1F77B4"],  # same two blues
#     labels={"assistant": "Assistant", "count": "Unique Tasks"},
# )
# max_unique = unique_tasks_df["assistant"].value_counts().max()
# fig_task_div.update_layout(
#     title_text="Unique Tasks per Assistant",
#     yaxis=dict(
#         title="Unique Tasks",
#         range=[0, max_unique * 1.2 if max_unique > 0 else 1],
#     ),
#     legend_title_text="Assistant",
#     height=350,
# )
# fig_task_div.update_traces(texttemplate="%{y}", textposition="outside")
# st.plotly_chart(fig_task_div, use_container_width=True)

# # ─── 3. Average Duration per Assistant ───────────────────────────────────────
# st.subheader("Average Duration per Assistant")

# # Convert duration (seconds) → milliseconds
# filtered["duration_ms"] = filtered["duration"] * 1000.0

# # Use a histogram with histfunc="avg" to compute average duration_ms per assistant
# fig_avg_dur = px.histogram(
#     filtered,
#     x="assistant",
#     y="duration_ms",
#     histfunc="avg",
#     color="assistant",
#     template="plotly_dark",
#     color_discrete_sequence=["#8FBCE6", "#1F77B4"],  # same two blues
#     labels={"assistant": "Assistant", "duration_ms": "Avg Duration (ms)"},
# )
# avg_vals = filtered.groupby("assistant")["duration_ms"].mean()
# max_ms = avg_vals.max() if not avg_vals.empty else 1.0
# fig_avg_dur.update_layout(
#     title_text="Average Generation Time (ms) by Assistant",
#     yaxis=dict(title="Avg Duration (ms)", range=[0, max_ms * 1.2]),
#     legend_title_text="Assistant",
#     height=350,
# )
# fig_avg_dur.update_traces(texttemplate="%{y:.1f} ms", textposition="outside")
# st.plotly_chart(fig_avg_dur, use_container_width=True)

# # ─── 4. Task Diversity Word-Scatter (Matplotlib) ─────────────────────────────
# st.subheader("Task Diversity Word-Scatter")

# # Drop duplicates so each task appears only once
# unique_tasks_scatter = filtered[["assistant", "task"]].drop_duplicates().reset_index(drop=True)

# # Generate random positions
# unique_tasks_scatter["x"] = [random.uniform(0.1, 0.9) for _ in range(len(unique_tasks_scatter))]
# unique_tasks_scatter["y"] = [random.uniform(0.1, 0.9) for _ in range(len(unique_tasks_scatter))]

# # Use the same two shades of blue
# assistant_to_color = {
#     "Copilot": "#8FBCE6",   # light-blue
#     "Windsurf": "#1F77B4",  # darker-blue
# }

# plt.figure(figsize=(8, 6), facecolor="#0f2027")
# ax = plt.gca()
# ax.set_facecolor("#0f2027")

# for _, row in unique_tasks_scatter.iterrows():
#     ax.text(
#         row["x"],
#         row["y"],
#         row["task"],
#         ha="center",
#         va="center",
#         fontsize=16,
#         color=assistant_to_color.get(row["assistant"], "#FFFFFF"),
#         fontweight="bold",
#         alpha=0.8,
#     )

# # Hide axes
# plt.xticks([])
# plt.yticks([])
# for spine in ["top", "right", "left", "bottom"]:
#     ax.spines[spine].set_visible(False)

# plt.title(
#     "🌐 Task Diversity Word-Scatter",
#     color="#FFFFFF",
#     fontsize=20,
#     pad=20
# )
# st.pyplot(plt.gcf())

# # ─── 5. Code Length Distribution ───────────────────────────────────────────────
# st.subheader("Code Length Distribution")
# fig_code_len = px.box(
#     filtered,
#     x="assistant",
#     y="length",
#     color="assistant",
#     points="all",
#     title="Distribution of Generated Code Length",
#     template="plotly_dark",
#     color_discrete_sequence=["#8FBCE6", "#1F77B4"],
#     labels={"assistant": "Assistant", "length": "Code Length (chars)"},
# )
# st.plotly_chart(fig_code_len, use_container_width=True)

# assistants = sorted(filtered["assistant"].unique())
# rated_df = filtered.dropna(subset=["accuracy_rating"])

# # ─── 6. Success % Calculation ─────────────────────────────────────────────────
# st.subheader("Success % (Rating > 0)")

# # Compute success_pct only if there are ratings; otherwise build zeros
# if not rated_df.empty:
#     success_df = (
#         rated_df
#         .groupby("assistant")["accuracy_rating"]
#         .apply(lambda col: (col > 0).sum() / len(col) * 100)
#         .reset_index(name="success_pct")
#     )
# else:
#     # If no ratings yet, show zero‐percent bars for each assistant
#     success_df = pd.DataFrame({
#         "assistant": assistants,
#         "success_pct": [0.0] * len(assistants),
#     })

# # Build a Plotly bar chart for Success %
# fig_success = px.bar(
#     success_df,
#     x="assistant",
#     y="success_pct",
#     color="assistant",
#     template="plotly_dark",
#     color_discrete_sequence=["#8FBCE6", "#1F77B4"],  # light‐blue for Copilot, dark‐blue for Windsurf
#     labels={"assistant": "Assistant", "success_pct": "Success %"},
# )

# # Force y‐axis 0→100, hide legend box, set height
# fig_success.update_layout(
#     title_text="Success Percentage by Assistant",
#     yaxis=dict(range=[0, 100]),
#     legend_title_text="Assistant",
#     showlegend=False,
#     height=350,
# )

# # Show numeric labels on top of each bar, formatted to one decimal
# fig_success.update_traces(
#     texttemplate="%{y:.1f}%",
#     textposition="outside",
# )

# st.plotly_chart(fig_success, use_container_width=True)

# # ─── 7. Accuracy Rating Boxplot ─────────────────────────────────────────────────
# st.subheader("Accuracy Rating Boxplot")

# if not rated_df.empty:
#     # If ratings exist, show a normal boxplot (with “points=all” to scatter individual points)
#     fig_box = px.box(
#         rated_df,
#         x="assistant",
#         y="accuracy_rating",
#         color="assistant",
#         template="plotly_dark",
#         color_discrete_sequence=["#8FBCE6", "#1F77B4"],
#         labels={"assistant": "Assistant", "accuracy_rating": "Accuracy Rating (0–5)"},
#         points="all",
#     )
#     fig_box.update_layout(
#         title_text="Distribution of Accuracy Ratings by Assistant",
#         showlegend=False,
#         height=350,
#     )
#     st.plotly_chart(fig_box, use_container_width=True)
# else:
#     # If there are no ratings yet, create a “dummy” DataFrame so Plotly can still render
#     dummy = pd.DataFrame({
#         "assistant": assistants,
#         "accuracy_rating": [None] * len(assistants),
#     })
#     fig_box = px.box(
#         dummy,
#         x="assistant",
#         y="accuracy_rating",
#         color="assistant",
#         template="plotly_dark",
#         color_discrete_sequence=["#8FBCE6", "#1F77B4"],
#         labels={"assistant": "Assistant", "accuracy_rating": "Accuracy Rating (0–5)"},
#     )
#     fig_box.update_layout(
#         title_text="Accuracy Rating Boxplot (No Ratings Yet)",
#         showlegend=False,
#         height=350,
#     )
#     st.plotly_chart(fig_box, use_container_width=True)

# # ─── 8. Usage Over Time (Hourly) ─────────────────────────────────────────────────
# st.subheader("Usage Over Time (Hourly)")

# fig_hourly = px.histogram(
#     filtered,
#     x="hour",                      # the hourly timestamp (floored to the hour)
#     color="assistant",             # split bars by assistant
#     histfunc="count",              # count how many rows in each (hour,assistant) bin
#     barmode="group",               # place Copilot and Windsurf bars side by side for each hour
#     template="plotly_dark",
#     color_discrete_sequence=["#8FBCE6", "#1F77B4"],  # light-blue for Copilot, dark-blue for Windsurf
#     labels={
#         "hour": "Hour",
#         "count": "Requests",
#         "assistant": "Assistant",
#     },
# )

# # Determine a 20% headroom for the y-axis
# y_max = filtered.groupby("hour")["assistant"].count().max()
# if y_max is None or np.isnan(y_max):
#     y_max = 1
# else:
#     y_max = y_max * 1.2  # add 20% headroom

# fig_hourly.update_layout(
#     title_text="Number of Requests per Hour by Assistant",
#     xaxis_title="Hour",
#     yaxis_title="Requests",
#     yaxis=dict(range=[0, y_max]),
#     legend_title_text="Assistant",
#     height=400,
# )

# # Optionally, show the count above each bar
# fig_hourly.update_traces(
#     texttemplate="%{y}",
#     textposition="outside",
# )

# st.plotly_chart(fig_hourly, use_container_width=True)

# # ─── 9. Heatmap: Runs per Day per Assistant ────────────────────────────────────
# st.subheader("Heatmap: Runs per Day per Assistant")
# pivot = (
#     filtered.groupby(["date", "assistant"])
#     .size()
#     .reset_index(name="count")
#     .pivot(index="assistant", columns="date", values="count")
#     .fillna(0)
# )
# if not pivot.empty:
#     plt.figure(figsize=(12, 4))
#     plt.imshow(pivot, aspect="auto", cmap="YlGnBu")
#     plt.yticks(range(len(pivot.index)), pivot.index, color="white")
#     plt.xticks(
#         range(len(pivot.columns)),
#         [d.strftime("%Y-%m-%d") for d in pivot.columns],
#         rotation=45,
#         ha="right",
#         color="white",
#     )
#     plt.colorbar(label="Number of Runs")
#     st.pyplot(plt.gcf())
# else:
#     st.info("Not enough data to build heatmap.")

# st.markdown("---")

# # ------------------ MANUAL ACCURACY RATING ------------------
# st.subheader("✍️ Manual Accuracy Rating")
# editable_df = filtered[
#     ["timestamp", "task", "assistant", "generated_code", "accuracy_rating"]
# ].copy()
# editable_df["accuracy_rating"] = editable_df["accuracy_rating"].fillna("")

# # For maximum compatibility, use st.dataframe instead of st.data_editor
# st.dataframe(editable_df, use_container_width=True)

# # ------------------ EXPERIMENTAL ASSISTANT SCORING ------------------
# st.subheader("🔍 Experimental Assistant Scoring")
# if "accuracy_rating" in filtered.columns and filtered[
#     "accuracy_rating"
# ].notnull().any():
#     try:
#         score_df = (
#             filtered.dropna(subset=["accuracy_rating"])
#             .groupby("assistant")["accuracy_rating"]
#             .mean()
#             .reset_index(name="avg_accuracy")
#             .sort_values("avg_accuracy", ascending=False)
#         )
#         fig_score = px.bar(
#             score_df,
#             x="assistant",
#             y="avg_accuracy",
#             color="assistant",
#             template="plotly_dark",
#             color_discrete_sequence=["#8FBCE6", "#1F77B4"],
#             labels={"assistant": "Assistant", "avg_accuracy": "Avg Accuracy"},
#         )
#         fig_score.update_layout(
#             title_text="Average Accuracy Rating by Assistant",
#             showlegend=False,
#             height=350,
#         )
#         fig_score.update_traces(texttemplate="%{y:.2f}", textposition="outside")
#         st.plotly_chart(fig_score, use_container_width=True)
#     except Exception:
#         st.warning("Unable to compute accuracy scores (ensure numeric ratings).")
# else:
#     st.info("No accuracy ratings to compute experimental scoring.")

# st.markdown(
#     """
#     <style>
#     .future-list {
#         color: #ffffff;
#         font-style: italic;
#     }
#     </style>
#     <div class="future-list">
#     🔜 Future Additions:<br>
#     • Auto‐tag success/failure via AI heuristics<br>
#     • Code comment readability analysis<br>
#     • Assistant‐generated documentation quality check<br>
#     • Animated agent avatars and interactive tooltips
#     </div>
#     """,
#     unsafe_allow_html=True,
# )






import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def load_history(csv_path: str, _mtime: float) -> pd.DataFrame:
    """
    Read history.csv, parse dates, compute missing duration,
    derive hour/date/length columns, and return DataFrame.
    """
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp", "start_time", "end_time"],
        dtype={"task": str, "assistant": str},
        na_filter=False,
    )

    if df.empty:
        return df

    # 1) Compute duration if missing
    if "duration" not in df.columns or df["duration"].isna().all():
        df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    else:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    # 2) Drop rows missing any critical column
    df = df.dropna(
        subset=["task", "assistant", "generated_code", "start_time", "end_time", "duration"]
    )

    # 3) Fill missing timestamps with start_time
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(df["start_time"])

    # 4) Derive additional columns for charts
    df["hour"] = df["timestamp"].dt.floor("h")
    df["date"] = df["timestamp"].dt.date
    df["length"] = df["generated_code"].astype(str).str.len()

    if "accuracy_rating" in df.columns:
        df["accuracy_rating"] = pd.to_numeric(df["accuracy_rating"], errors="coerce")
    else:
        df["accuracy_rating"] = np.nan

    return df


# --------------------------- Streamlit Setup -----------------------------
st.set_page_config(
    page_title="Smart AI Coding Assistant Dashboard",
    layout="wide",
    page_icon="🤖",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #0f2027);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
    }
    @keyframes gradientBG {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .css-1d391kg .stMetricLabel, .css-1d391kg .stMetricValue {
        color: #ffffff !important;
    }
    .stDataFrame, .stTable {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Smart AI Coding Assistant Dashboard")

# ——— Load Data ——————————————————————————————————————————————————————————
DATA_PATH = Path(__file__).parent.parent / "data" / "history.csv"
if not DATA_PATH.exists():
    st.error(
        "⚠️ Cannot find data/history.csv. Please run your watcher to generate at least one row."
    )
    st.stop()

file_mtime = DATA_PATH.stat().st_mtime
df = load_history(str(DATA_PATH), file_mtime)

if df.empty:
    st.warning(
        "No valid rows found in history.csv. Make a change in scripts/demo.py so the watcher generates data."
    )
    st.stop()

# ——— Sidebar Metrics & Filtering —————————————————————————————————————————
with st.sidebar:
    st.header("📊 Metrics Summary")

    total_requests = len(df)
    unique_tasks = df["task"].nunique()
    avg_length = df["length"].mean() if total_requests > 0 else 0.0
    avg_duration = df["duration"].mean() if "duration" in df.columns else 0.0
    med_duration = df["duration"].median() if "duration" in df.columns else 0.0
    max_duration = df["duration"].max() if "duration" in df.columns else 0.0

    rated_df = df.dropna(subset=["accuracy_rating"])
    if not rated_df.empty:
        success_pct = (rated_df["accuracy_rating"] > 0).sum() / len(rated_df) * 100
    else:
        success_pct = np.nan

    st.metric("Total Requests", total_requests)
    st.metric("Unique Tasks", unique_tasks)
    st.metric("Avg. Code Length", f"{avg_length:.0f} chars")
    st.metric("Avg. Duration", f"{avg_duration:.3f} s")
    st.metric("Median Duration", f"{med_duration:.3f} s")
    st.metric("Max Duration", f"{max_duration:.3f} s")
    if not np.isnan(success_pct):
        st.metric("Success % (rating > 0)", f"{success_pct:.1f}%")
    else:
        st.metric("Success % (rating > 0)", "N/A")

    st.markdown("---")
    st.write("**Filter by Assistant(s)**")
    all_assistants = sorted(df["assistant"].unique())
    selected_assistants = st.multiselect(
        "Select assistant(s)",
        all_assistants,
        default=all_assistants,
    )

if not selected_assistants:
    st.warning("Please select at least one assistant before viewing charts.")
    st.stop()

filtered = df[df["assistant"].isin(selected_assistants)].copy()

# ——— Main Content: Overview Charts —————————————————————————————————————
st.markdown("## 📈 Overview Charts", unsafe_allow_html=True)

# ─── 1. Requests per Assistant ────────────────────────────────────────────────
st.subheader("Requests per Assistant")

fig_requests = px.histogram(
    filtered,
    x="assistant",
    color="assistant",
    barmode="group",
    template="plotly_dark",
    color_discrete_sequence=["#8FBCE6", "#1F77B4"],
    labels={"assistant": "Assistant", "count": "Requests"},
)
fig_requests.update_layout(
    title_text="Number of Requests by Assistant",
    yaxis=dict(
        title="Requests",
        range=[0, filtered["assistant"].value_counts().max() * 1.2],
    ),
    legend_title_text="Assistant",
    height=400,
)
fig_requests.update_traces(texttemplate="%{y}", textposition="outside")
st.plotly_chart(fig_requests, use_container_width=True)

# ─── 2. Task Diversity by Assistant ─────────────────────────────────────────
st.subheader("Task Diversity by Assistant")

# Build DataFrame of (assistant, task) pairs, drop duplicates
unique_tasks_df = filtered[["assistant", "task"]].drop_duplicates()

# Use a histogram to count unique tasks per assistant
fig_task_div = px.histogram(
    unique_tasks_df,
    x="assistant",
    histfunc="count",
    color="assistant",
    template="plotly_dark",
    color_discrete_sequence=["#8FBCE6", "#1F77B4"],
    labels={"assistant": "Assistant", "count": "Unique Tasks"},
)
max_unique = unique_tasks_df["assistant"].value_counts().max()
fig_task_div.update_layout(
    title_text="Unique Tasks per Assistant",
    yaxis=dict(
        title="Unique Tasks",
        range=[0, max_unique * 1.2 if max_unique > 0 else 1],
    ),
    legend_title_text="Assistant",
    height=350,
)
fig_task_div.update_traces(texttemplate="%{y}", textposition="outside")
st.plotly_chart(fig_task_div, use_container_width=True)

# ─── 3. Average Duration per Assistant ───────────────────────────────────────
st.subheader("Average Duration per Assistant")

# Convert duration (seconds) → milliseconds
filtered["duration_ms"] = filtered["duration"] * 1000.0

# Use a histogram with histfunc="avg" to compute average duration_ms per assistant
fig_avg_dur = px.histogram(
    filtered,
    x="assistant",
    y="duration_ms",
    histfunc="avg",
    color="assistant",
    template="plotly_dark",
    color_discrete_sequence=["#8FBCE6", "#1F77B4"],
    labels={"assistant": "Assistant", "duration_ms": "Avg Duration (ms)"},
)
avg_vals = filtered.groupby("assistant")["duration_ms"].mean()
max_ms = avg_vals.max() if not avg_vals.empty else 1.0
fig_avg_dur.update_layout(
    title_text="Average Generation Time (ms) by Assistant",
    yaxis=dict(title="Avg Duration (ms)", range=[0, max_ms * 1.2]),
    legend_title_text="Assistant",
    height=350,
)
fig_avg_dur.update_traces(texttemplate="%{y:.1f} ms", textposition="outside")
st.plotly_chart(fig_avg_dur, use_container_width=True)

# ─── 4. Task Diversity Word-Scatter (Matplotlib) ─────────────────────────────
st.subheader("Task Diversity Word-Scatter")

# Drop duplicates so each task appears only once
unique_tasks_scatter = filtered[["assistant", "task"]].drop_duplicates().reset_index(drop=True)

# Generate random positions
unique_tasks_scatter["x"] = [random.uniform(0.1, 0.9) for _ in range(len(unique_tasks_scatter))]
unique_tasks_scatter["y"] = [random.uniform(0.1, 0.9) for _ in range(len(unique_tasks_scatter))]

# Use the same two shades of blue
assistant_to_color = {
    "Copilot": "#8FBCE6",
    "Windsurf": "#1F77B4",
}

plt.figure(figsize=(8, 6), facecolor="#0f2027")
ax = plt.gca()
ax.set_facecolor("#0f2027")

for _, row in unique_tasks_scatter.iterrows():
    ax.text(
        row["x"],
        row["y"],
        row["task"],
        ha="center",
        va="center",
        fontsize=16,
        color=assistant_to_color.get(row["assistant"], "#FFFFFF"),
        fontweight="bold",
        alpha=0.8,
    )

# Hide axes
plt.xticks([])
plt.yticks([])
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

plt.title(
    "🌐 Task Diversity Word-Scatter",
    color="#FFFFFF",
    fontsize=20,
    pad=20
)
st.pyplot(plt.gcf())

# ─── 5. Code Length Distribution ─────────────────────────────────────────────
st.subheader("Code Length Distribution")
fig_code_len = px.box(
    filtered,
    x="assistant",
    y="length",
    color="assistant",
    points="all",
    template="plotly_dark",
    color_discrete_sequence=["#8FBCE6", "#1F77B4"],
    labels={"assistant": "Assistant", "length": "Code Length (chars)"},
    title="Distribution of Generated Code Length",
)
st.plotly_chart(fig_code_len, use_container_width=True)

assistants = sorted(filtered["assistant"].unique())

# ─── 6. Success % Calculation ─────────────────────────────────────────────────
st.subheader("Success % (Rating > 0)")

# Compute success_pct only if there are ratings; otherwise build zeros
if not rated_df.empty:
    success_df = (
        rated_df
        .groupby("assistant")["accuracy_rating"]
        .apply(lambda col: (col > 0).sum() / len(col) * 100)
        .reset_index(name="success_pct")
    )
else:
    # If no ratings yet, show zero‐percent bars for each assistant
    success_df = pd.DataFrame({
        "assistant": assistants,
        "success_pct": [0.0] * len(assistants),
    })

# Build a Plotly bar chart for Success %
fig_success = px.bar(
    success_df,
    x="assistant",
    y="success_pct",
    color="assistant",
    template="plotly_dark",
    color_discrete_sequence=["#8FBCE6", "#1F77B4"],
    labels={"assistant": "Assistant", "success_pct": "Success %"},
)
fig_success.update_layout(
    title_text="Success Percentage per Assistant",
    yaxis=dict(range=[0, 100]),
    legend_title_text="Assistant",
    height=350,
)
fig_success.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
st.plotly_chart(fig_success, use_container_width=True)

# ─── 7. Accuracy Rating Boxplot ───────────────────────────────────────────────
st.subheader("Accuracy Rating Boxplot")

if not rated_df.empty:
    # If ratings exist, show a normal boxplot with points
    fig_box = px.box(
        filtered.dropna(subset=["accuracy_rating"]),
        x="assistant",
        y="accuracy_rating",
        color="assistant",
        template="plotly_dark",
        color_discrete_sequence=["#8FBCE6", "#1F77B4"],
        labels={"assistant": "Assistant", "accuracy_rating": "Accuracy Rating (0–5)"},
        points="all",
        title="Distribution of Accuracy Ratings",
    )
    fig_box.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_box, use_container_width=True)
else:
    # If there are no ratings yet, create a dummy DataFrame so Plotly still renders an empty boxplot
    dummy = pd.DataFrame({
        "assistant": assistants,
        "accuracy_rating": [None] * len(assistants),
    })
    fig_box = px.box(
        dummy,
        x="assistant",
        y="accuracy_rating",
        color="assistant",
        template="plotly_dark",
        color_discrete_sequence=["#8FBCE6", "#1F77B4"],
        labels={"assistant": "Assistant", "accuracy_rating": "Accuracy Rating (0–5)"},
        title="Distribution of Accuracy Ratings",
    )
    fig_box.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_box, use_container_width=True)

# ─── 8. Usage Over Time (Hourly) ───────────────────────────────────────────────
st.subheader("Usage Over Time (Hourly)")

fig_hourly = px.histogram(
    filtered,
    x="hour",
    color="assistant",
    histfunc="count",
    barmode="group",
    template="plotly_dark",
    color_discrete_sequence=["#8FBCE6", "#1F77B4"],
    labels={"hour": "Hour", "count": "Requests", "assistant": "Assistant"},
)
y_max = filtered.groupby("hour")["assistant"].count().max()
y_max = y_max * 1.2 if not np.isnan(y_max) else 1.0

fig_hourly.update_layout(
    title_text="Number of Requests per Hour by Assistant",
    xaxis_title="Hour",
    yaxis=dict(title="Requests", range=[0, y_max]),
    legend_title_text="Assistant",
    height=400,
)
fig_hourly.update_traces(texttemplate="%{y}", textposition="outside")
st.plotly_chart(fig_hourly, use_container_width=True)

# ─── 9. Heatmap: Runs per Day per Assistant ───────────────────────────────────
st.subheader("Heatmap: Runs per Day per Assistant")
pivot = (
    filtered.groupby(["date", "assistant"])
    .size()
    .reset_index(name="count")
    .pivot(index="assistant", columns="date", values="count")
    .fillna(0)
)
if not pivot.empty:
    plt.figure(figsize=(12, 4), facecolor="#0f2027")
    ax = plt.gca()
    im = ax.imshow(pivot, aspect="auto", cmap="YlGnBu")
    plt.yticks(range(len(pivot.index)), pivot.index, color="white")
    plt.xticks(
        range(len(pivot.columns)),
        [d.strftime("%Y-%m-%d") for d in pivot.columns],
        rotation=45,
        ha="right",
        color="white",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.title("Requests per Day per Assistant", color="white", pad=20)
    st.pyplot(plt.gcf())
else:
    st.info("Not enough data to build heatmap.")

st.markdown("---")

# ------------------ MANUAL ACCURACY RATING ------------------
st.subheader("✍️ Manual Accuracy Rating")
editable_df = filtered[
    ["timestamp", "task", "assistant", "generated_code", "accuracy_rating"]
].copy()
editable_df["accuracy_rating"] = editable_df["accuracy_rating"].fillna("")

# For maximum compatibility, use st.dataframe instead of st.data_editor
st.dataframe(editable_df, use_container_width=True)

# ------------------ EXPERIMENTAL ASSISTANT SCORING ------------------
st.subheader("🔍 Experimental Assistant Scoring")
if "accuracy_rating" in filtered.columns and filtered[
    "accuracy_rating"
].notnull().any():
    try:
        score_df = (
            filtered.dropna(subset=["accuracy_rating"])
            .groupby("assistant")["accuracy_rating"]
            .mean()
            .reset_index(name="avg_accuracy")
            .sort_values("avg_accuracy", ascending=False)
        )
        fig_score = px.bar(
            score_df,
            x="assistant",
            y="avg_accuracy",
            color="assistant",
            template="plotly_dark",
            color_discrete_sequence=["#8FBCE6", "#1F77B4"],
            labels={"assistant": "Assistant", "avg_accuracy": "Avg Accuracy"},
        )
        fig_score.update_layout(
            title_text="Average Accuracy Rating by Assistant",
            showlegend=False,
            height=350,
        )
        fig_score.update_traces(texttemplate="%{y:.2f}", textposition="outside")
        st.plotly_chart(fig_score, use_container_width=True)
    except Exception:
        st.warning("Unable to compute accuracy scores (ensure numeric ratings).")
else:
    st.info("No accuracy ratings to compute experimental scoring.")

st.markdown(
    """
    <style>
    .future-list {
        color: #ffffff;
        font-style: italic;
    }
    </style>
    <div class="future-list">
    🔜 Future Additions:<br>
    • Auto‐tag success/failure via AI heuristics<br>
    • Code comment readability analysis<br>
    • Assistant‐generated documentation quality check<br>
    • Animated agent avatars and interactive tooltips
    </div>
    """,
    unsafe_allow_html=True,
)
