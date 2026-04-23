import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="LMS Course Progress Dashboard", layout="wide")

# =========================
# CONFIG
# =========================
CSV_PATH = "dashboard_dataset.csv"
PASS_MARK = 50
AT_RISK_THRESHOLD = 40

# =========================
# LOAD DATA
# =========================
@st.cache_data

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize column names lightly
    df.columns = [c.strip() for c in df.columns]

    text_cols = [
        "course_name", "fullname", "email", "section_name",
        "week_name", "week_label", "status"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
            )

    numeric_cols = [
        "course_id", "user_id", "cmid", "completed", "week_order",
        "course_completed", "avg_score", "best_score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize binary fields
    if "completed" in df.columns:
        df["completed"] = df["completed"].fillna(0).astype(int)
    if "course_completed" in df.columns:
        df["course_completed"] = df["course_completed"].fillna(0).astype(int)

    # Standardize score/status fields
    if "best_score" in df.columns:
        df["best_score"] = df["best_score"].round(2)
    if "avg_score" in df.columns:
        df["avg_score"] = df["avg_score"].round(2)

    if "status" in df.columns:
        df["status"] = (
            df["status"]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"Nan": pd.NA})
        )

    # Week sorting fallback
    if "week_order" in df.columns:
        df["week_order"] = pd.to_numeric(df["week_order"], errors="coerce")

    if "week_label" not in df.columns and "week_name" in df.columns:
        df["week_label"] = df["week_name"]

    return df


raw_df = load_data(CSV_PATH)

# One row per learner per course for learner-level KPIs
learner_df = (
    raw_df.groupby(["course_id", "course_name", "user_id", "fullname", "email"], dropna=False)
    .agg(
        total_weeks=("week_label", "nunique"),
        weeks_completed=("completed", "sum"),
        course_completed=("course_completed", "max"),
        avg_score=("avg_score", "max"),
        best_score=("best_score", "max"),
        status=("status", "max"),
    )
    .reset_index()
)

learner_df["progress_pct"] = np.where(
    learner_df["total_weeks"] > 0,
    (learner_df["weeks_completed"] / learner_df["total_weeks"]) * 100,
    0,
)
learner_df["is_active"] = learner_df["weeks_completed"] > 0
learner_df["at_risk"] = learner_df["best_score"].fillna(-1) < AT_RISK_THRESHOLD
learner_df["at_risk"] = learner_df["at_risk"] & learner_df["best_score"].notna()


def safe_options(data: pd.DataFrame, col: str):
    if col not in data.columns:
        return ["All"]
    vals = data[col].dropna().astype(str).str.strip().unique().tolist()
    vals = sorted([v for v in vals if v])
    return ["All"] + vals


# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filter Dashboard")

course_options = safe_options(raw_df, "course_name")
selected_course = st.sidebar.selectbox("Course", course_options)

course_filtered = raw_df.copy()
if selected_course != "All":
    course_filtered = course_filtered[course_filtered["course_name"] == selected_course]

week_options = ["All"]
if "week_label" in course_filtered.columns:
    week_table = (
        course_filtered[["week_label", "week_order"]]
        .drop_duplicates()
        .sort_values(["week_order", "week_label"], na_position="last")
    )
    week_options += week_table["week_label"].dropna().tolist()
selected_week = st.sidebar.selectbox("Week", week_options)

status_options = safe_options(course_filtered, "status")
selected_status = st.sidebar.selectbox("Quiz Status", status_options)

completion_options = ["All", "Completed Course", "Not Completed Course"]
selected_completion = st.sidebar.selectbox("Course Completion", completion_options)

activity_options = ["All", "Active Learners", "Inactive Learners"]
selected_activity = st.sidebar.selectbox("Learner Activity", activity_options)

# =========================
# APPLY FILTERS
# =========================
filtered_raw = raw_df.copy()
filtered_learners = learner_df.copy()

if selected_course != "All":
    filtered_raw = filtered_raw[filtered_raw["course_name"] == selected_course]
    filtered_learners = filtered_learners[filtered_learners["course_name"] == selected_course]

if selected_status != "All":
    filtered_raw = filtered_raw[filtered_raw["status"] == selected_status]
    filtered_learners = filtered_learners[filtered_learners["status"] == selected_status]

if selected_completion == "Completed Course":
    filtered_raw = filtered_raw[filtered_raw["course_completed"] == 1]
    filtered_learners = filtered_learners[filtered_learners["course_completed"] == 1]
elif selected_completion == "Not Completed Course":
    filtered_raw = filtered_raw[filtered_raw["course_completed"] == 0]
    filtered_learners = filtered_learners[filtered_learners["course_completed"] == 0]

if selected_activity == "Active Learners":
    filtered_learners = filtered_learners[filtered_learners["is_active"]]
    filtered_raw = filtered_raw.merge(
        filtered_learners[["course_id", "user_id"]],
        on=["course_id", "user_id"],
        how="inner"
    )
elif selected_activity == "Inactive Learners":
    filtered_learners = filtered_learners[~filtered_learners["is_active"]]
    filtered_raw = filtered_raw.merge(
        filtered_learners[["course_id", "user_id"]],
        on=["course_id", "user_id"],
        how="inner"
    )

if selected_week != "All":
    filtered_raw = filtered_raw[filtered_raw["week_label"] == selected_week]

# =========================
# TITLE
# =========================
st.title("LMS Course Progress Dashboard")
st.markdown("Track course completion, weekly engagement, quiz performance, and at-risk learners.")

# =========================
# KPI SECTION
# =========================
total_learners = filtered_learners["user_id"].nunique()
completed_courses = int(filtered_learners["course_completed"].sum()) if not filtered_learners.empty else 0
completion_rate = (completed_courses / total_learners * 100) if total_learners else 0
active_learners = int(filtered_learners["is_active"].sum()) if not filtered_learners.empty else 0
avg_progress = filtered_learners["progress_pct"].mean() if not filtered_learners.empty else 0

attempted_scores = filtered_learners[filtered_learners["best_score"].notna()].copy()
pass_count = int((attempted_scores["best_score"] >= PASS_MARK).sum()) if not attempted_scores.empty else 0
pass_rate = (pass_count / len(attempted_scores) * 100) if len(attempted_scores) else 0
avg_score = attempted_scores["best_score"].mean() if not attempted_scores.empty else 0
at_risk_count = int(filtered_learners["at_risk"].sum()) if not filtered_learners.empty else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Learners", f"{total_learners:,}")
c2.metric("Course Completion Rate", f"{completion_rate:.1f}%")
c3.metric("Active Learners", f"{active_learners:,}")
c4.metric("Average Progress", f"{avg_progress:.1f}%")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Pass Rate", f"{pass_rate:.1f}%")
c6.metric("Average Best Score", f"{avg_score:.1f}%")
c7.metric("At-Risk Learners", f"{at_risk_count:,}")
c8.metric("Students Attempted Quiz", f"{len(attempted_scores):,}")

# =========================
# WEEKLY COMPLETION TREND
# =========================
st.subheader("Weekly Completion Rate")

if not filtered_raw.empty:
    weekly_completion = (
        filtered_raw.groupby(["week_order", "week_label"], dropna=False)
        .agg(
            total_learners=("user_id", "nunique"),
            learners_completed=("completed", "sum")
        )
        .reset_index()
        .sort_values(["week_order", "week_label"], na_position="last")
    )
    weekly_completion["weekly_completion_rate_pct"] = np.where(
        weekly_completion["total_learners"] > 0,
        (weekly_completion["learners_completed"] / weekly_completion["total_learners"]) * 100,
        0,
    )

    fig_weekly = px.line(
        weekly_completion,
        x="week_label",
        y="weekly_completion_rate_pct",
        markers=True,
        title="Weekly Completion Rate (%)"
    )
    fig_weekly.update_layout(yaxis_title="Completion Rate (%)", xaxis_title="Week")
    st.plotly_chart(fig_weekly, use_container_width=True)
    st.dataframe(weekly_completion, use_container_width=True, hide_index=True)
else:
    st.info("No weekly data available for the selected filters.")

# =========================
# DROPOFF RATE
# =========================
st.subheader("Weekly Drop-off")

if not filtered_raw.empty:
    dropoff_df = weekly_completion.copy()
    dropoff_df["prev_week_completion_rate_pct"] = dropoff_df["weekly_completion_rate_pct"].shift(1)
    dropoff_df["dropoff_pct"] = (
        dropoff_df["prev_week_completion_rate_pct"] - dropoff_df["weekly_completion_rate_pct"]
    ).round(1)

    fig_dropoff = px.bar(
        dropoff_df,
        x="week_label",
        y="dropoff_pct",
        title="Drop-off from Previous Week (%)",
        text="dropoff_pct"
    )
    fig_dropoff.update_traces(textposition="outside")
    fig_dropoff.update_layout(yaxis_title="Drop-off (%)", xaxis_title="Week")
    st.plotly_chart(fig_dropoff, use_container_width=True)
    st.dataframe(dropoff_df, use_container_width=True, hide_index=True)
else:
    st.info("No drop-off data available for the selected filters.")

# =========================
# QUIZ PERFORMANCE
# =========================
st.subheader("Quiz Performance")
col_left, col_right = st.columns(2)

with col_left:
    if not attempted_scores.empty:
        fig_score = px.histogram(
            attempted_scores,
            x="best_score",
            nbins=20,
            title="Best Score Distribution"
        )
        fig_score.update_layout(xaxis_title="Best Score (%)", yaxis_title="Learners")
        st.plotly_chart(fig_score, use_container_width=True)
    else:
        st.info("No quiz score data available for the selected filters.")

with col_right:
    status_df = (
        filtered_learners["status"]
        .fillna("No Status")
        .value_counts()
        .rename_axis("status")
        .reset_index(name="count")
    )
    if not status_df.empty:
        fig_status = px.pie(
            status_df,
            names="status",
            values="count",
            hole=0.4,
            title="Quiz Status Split"
        )
        st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.info("No status data available for the selected filters.")

# =========================
# PROGRESS BANDS
# =========================
st.subheader("Learner Progress Bands")

if not filtered_learners.empty:
    band_df = filtered_learners.copy()
    band_df["progress_band"] = pd.cut(
        band_df["progress_pct"],
        bins=[-0.1, 0, 25, 50, 75, 100],
        labels=["0%", "1–25%", "26–50%", "51–75%", "76–100%"]
    )
    band_summary = (
        band_df["progress_band"]
        .value_counts(dropna=False)
        .rename_axis("progress_band")
        .reset_index(name="learners")
    )
    fig_band = px.bar(
        band_summary,
        x="progress_band",
        y="learners",
        text="learners",
        title="Learners by Progress Band"
    )
    fig_band.update_traces(textposition="outside")
    st.plotly_chart(fig_band, use_container_width=True)
else:
    st.info("No learner progress data available for the selected filters.")

# =========================
# AT-RISK TABLE
# =========================
st.subheader("At-Risk Learners")

at_risk_df = filtered_learners[filtered_learners["at_risk"]].copy()
at_risk_df = at_risk_df.sort_values(["course_name", "best_score", "progress_pct"], ascending=[True, True, True])

if not at_risk_df.empty:
    st.dataframe(
        at_risk_df[[
            "course_name", "fullname", "email", "weeks_completed", "total_weeks",
            "progress_pct", "best_score", "status", "course_completed"
        ]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No at-risk learners found for the selected filters.")

# =========================
# LEARNER TABLE
# =========================
st.subheader("Learner Summary")

learner_display = filtered_learners.copy().sort_values(["course_name", "fullname"])
if not learner_display.empty:
    st.dataframe(
        learner_display[[
            "course_name", "fullname", "email", "weeks_completed", "total_weeks",
            "progress_pct", "best_score", "status", "course_completed"
        ]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No learner data available for the selected filters.")

# =========================
# DOWNLOADS
# =========================
st.subheader("Download Filtered Data")

csv_learners = filtered_learners.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Download learner summary CSV",
    data=csv_learners,
    file_name="filtered_learner_summary.csv",
    mime="text/csv",
)

csv_weeks = filtered_raw.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Download weekly progress CSV",
    data=csv_weeks,
    file_name="filtered_weekly_progress.csv",
    mime="text/csv",
)
