# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import textwrap

st.set_page_config(
    page_title="NJFP LMS Learning Analytics Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    return pd.read_csv("undp_lms_dataset.csv")

final_table = load_data()

# =========================
# REQUIRED COLUMNS
# =========================

required_cols = [
    "course_id", "course_name", "user_id", "fullname", "email",
    "section_name", "week_name", "week_number", "cmid", "completed",
    "total_quizzes", "attempted_quizzes", "avg_score", "course_completed"
]

missing_cols = [col for col in required_cols if col not in final_table.columns]

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# =========================
# CLEAN DATA
# =========================

final_table["completed"] = final_table["completed"].fillna(0).astype(int)
final_table["course_completed"] = final_table["course_completed"].fillna(0).astype(int)
final_table["total_quizzes"] = final_table["total_quizzes"].fillna(0)
final_table["attempted_quizzes"] = final_table["attempted_quizzes"].fillna(0)
final_table["avg_score"] = final_table["avg_score"].fillna(0)
final_table["week_number"] = pd.to_numeric(final_table["week_number"], errors="coerce")

# =========================
# ROW-LEVEL KPI COLUMNS
# =========================

final_table["quiz_attempt_rate"] = np.where(
    final_table["total_quizzes"] > 0,
    final_table["attempted_quizzes"] / final_table["total_quizzes"],
    0
)

final_table["week_active"] = (
    (final_table["completed"] == 1) |
    (final_table["attempted_quizzes"] > 0)
).astype(int)

final_table["week_at_risk"] = (
    (final_table["completed"] == 0) &
    (final_table["attempted_quizzes"] == 0)
).astype(int)

final_table["engagement_score"] = (
    (final_table["completed"] * 0.4) +
    ((final_table["attempted_quizzes"] > 0).astype(int) * 0.3) +
    ((final_table["avg_score"] / 100) * 0.3)
)

final_table["learning_effectiveness"] = (
    final_table["completed"] * (final_table["avg_score"] / 100)
)

# =========================
# HELPER FUNCTIONS
# =========================

def wrap_label(text, width=15):
    return "<br>".join(textwrap.wrap(str(text), width=width))

def build_overall_learner_summary(df):
    learner_df = (
        df.groupby("user_id")
        .agg(
            weeks_completed=("completed", "sum"),
            quizzes_attempted=("attempted_quizzes", "sum"),
            course_completed=("course_completed", "max"),
            avg_score=("avg_score", lambda x: x[x > 0].mean()),
            engagement_score=("engagement_score", "mean")
        )
        .reset_index()
    )

    learner_df["is_active"] = (
        (learner_df["weeks_completed"] > 0) |
        (learner_df["quizzes_attempted"] > 0)
    ).astype(int)

    learner_df["at_risk"] = (
        (learner_df["weeks_completed"] == 0) &
        (learner_df["quizzes_attempted"] == 0)
    ).astype(int)

    return learner_df


def build_course_learner_summary(df):
    learner_df = (
        df.groupby(["course_id", "course_name", "user_id"])
        .agg(
            weeks_completed=("completed", "sum"),
            quizzes_attempted=("attempted_quizzes", "sum"),
            course_completed=("course_completed", "max"),
            avg_score=("avg_score", "mean"),
            engagement_score=("engagement_score", "mean")
        )
        .reset_index()
    )

    learner_df["is_active"] = (
        (learner_df["weeks_completed"] > 0) |
        (learner_df["quizzes_attempted"] > 0)
    ).astype(int)

    learner_df["at_risk"] = (
        (learner_df["weeks_completed"] == 0) &
        (learner_df["quizzes_attempted"] == 0)
    ).astype(int)

    return learner_df

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.title("Filters")

course_options = ["All Courses"] + sorted(final_table["course_name"].dropna().unique())

selected_course = st.sidebar.selectbox(
    "Select Course",
    options=course_options
)

if selected_course == "All Courses":
    filtered_df = final_table.copy()
else:
    filtered_df = final_table[final_table["course_name"] == selected_course].copy()

week_options = ["All Weeks"] + sorted(filtered_df["week_number"].dropna().unique())

selected_week = st.sidebar.selectbox(
    "Select Week",
    options=week_options
)

if selected_week != "All Weeks":
    filtered_df = filtered_df[filtered_df["week_number"] == selected_week].copy()

learner_status = st.sidebar.selectbox(
    "Learner Status",
    options=["All Learners", "Active Learners", "At-Risk Learners"]
)

learner_filter_df = build_overall_learner_summary(filtered_df)

if learner_status == "Active Learners":
    active_ids = learner_filter_df.loc[
        learner_filter_df["is_active"] == 1, "user_id"
    ]
    filtered_df = filtered_df[filtered_df["user_id"].isin(active_ids)].copy()

elif learner_status == "At-Risk Learners":
    at_risk_ids = learner_filter_df.loc[
        learner_filter_df["at_risk"] == 1, "user_id"
    ]
    filtered_df = filtered_df[filtered_df["user_id"].isin(at_risk_ids)].copy()

completion_status = st.sidebar.selectbox(
    "Course Completion Status",
    options=["All", "Completed", "Not Completed"]
)

if completion_status == "Completed":
    filtered_df = filtered_df[filtered_df["course_completed"] == 1].copy()

elif completion_status == "Not Completed":
    filtered_df = filtered_df[filtered_df["course_completed"] == 0].copy()

search_text = st.sidebar.text_input("Search Learner Name or Email")

if search_text:
    filtered_df = filtered_df[
        filtered_df["fullname"].str.contains(search_text, case=False, na=False) |
        filtered_df["email"].str.contains(search_text, case=False, na=False)
    ].copy()

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# =========================
# TITLE
# =========================

st.title("NJFP LMS Learning Analytics Dashboard")
st.caption("Dashboard showing learner engagement, weekly completion, quiz performance, and at-risk learners.")

# =========================
# KPI CARDS
# =========================

learner_summary_filtered = build_overall_learner_summary(filtered_df)

total_learners = learner_summary_filtered["user_id"].nunique()
active_learners = learner_summary_filtered["is_active"].sum()
at_risk_learners = learner_summary_filtered["at_risk"].sum()

course_completion_rate = learner_summary_filtered["course_completed"].mean() * 100
avg_engagement_score = learner_summary_filtered["engagement_score"].mean() * 100

attempted_rows = filtered_df[
    filtered_df["attempted_quizzes"] > 0
]

if not attempted_rows.empty:
    avg_quiz_score = (
        (attempted_rows["avg_score"] * attempted_rows["attempted_quizzes"]).sum()
        / attempted_rows["attempted_quizzes"].sum()
    )
else:
    avg_quiz_score = 0


col1, col2, col3 = st.columns(3)

col1.metric("Total Learners", f"{total_learners:,}")
col2.metric("Active Learners", f"{active_learners:,}")
col3.metric("At-Risk Learners", f"{at_risk_learners:,}")

col4, col5, col6 = st.columns(3)

col4.metric("Course Completion Rate", f"{course_completion_rate:.1f}%")
col5.metric("Avg Engagement Score", f"{avg_engagement_score:.1f}%")
col6.metric(
    "Avg Quiz Score",
    f"{avg_quiz_score:.1f}%" if not np.isnan(avg_quiz_score) else "0.0%"
)

st.divider()

# =========================
# TOTAL USERS BY COURSE
# =========================

total_users_by_course = (
    filtered_df
    .groupby("course_name")
    .agg(total_users=("user_id", "nunique"))
    .reset_index()
    .sort_values("total_users", ascending=False)
)

total_users_by_course["course_name_wrapped"] = total_users_by_course["course_name"].apply(wrap_label)

fig_total_users = px.bar(
    total_users_by_course,
    x="course_name_wrapped",
    y="total_users",
    title="Total Users by Course",
    text="total_users",
    labels={
        "course_name_wrapped": "Course",
        "total_users": "Total Users"
    }
)

fig_total_users.update_traces(textposition="outside")

fig_total_users.update_layout(
    xaxis_tickangle=0,
    margin=dict(t=60, b=120),
    yaxis_title="Total Users",
    xaxis_title="Course"
)

st.plotly_chart(fig_total_users, use_container_width=True)

# =========================
# WEEKLY COMPLETION RATE
# =========================

weekly_completion = (
    filtered_df
    .groupby(["course_name", "week_number"])
    .agg(
        total_users=("user_id", "nunique"),
        completed_users=("completed", "sum")
    )
    .reset_index()
)

weekly_completion["completion_rate"] = (
    weekly_completion["completed_users"] /
    weekly_completion["total_users"] * 100
)

fig_weekly_completion = px.line(
    weekly_completion,
    x="week_number",
    y="completion_rate",
    color="course_name",
    markers=True,
    title="Weekly Completion Rate by Course",
    labels={
        "week_number": "Week",
        "completion_rate": "Completion Rate (%)",
        "course_name": "Course"
    }
)

fig_weekly_completion.update_layout(
    xaxis=dict(dtick=1),
    yaxis=dict(range=[0, 100])
)

st.plotly_chart(fig_weekly_completion, use_container_width=True)

# =========================
# ACTIVE USERS BY COURSE
# =========================

active_by_course = (
    build_course_learner_summary(filtered_df)
    .groupby("course_name")
    .agg(
        total_learners=("user_id", "nunique"),
        active_learners=("is_active", "sum")
    )
    .reset_index()
)

active_by_course["active_rate"] = (
    active_by_course["active_learners"] /
    active_by_course["total_learners"] * 100
)

active_by_course["course_name_wrapped"] = active_by_course["course_name"].apply(wrap_label)

fig_active = px.bar(
    active_by_course,
    x="course_name_wrapped",
    y="active_rate",
    title="Active Learner Rate by Course",
    text=active_by_course["active_rate"].round(1),
    labels={
        "course_name_wrapped": "Course",
        "active_rate": "Active Learner Rate (%)"
    }
)

fig_active.update_traces(textposition="outside")

fig_active.update_layout(
    yaxis=dict(range=[0, 100]),
    margin=dict(t=60, b=120),
    xaxis_title="Course",
    yaxis_title="Active Learner Rate (%)"
)

st.plotly_chart(fig_active, use_container_width=True)

# =========================
# AT-RISK LEARNERS BY COURSE
# =========================

risk_by_course = (
    build_course_learner_summary(filtered_df)
    .groupby("course_name")
    .agg(
        at_risk_learners=("at_risk", "sum")
    )
    .reset_index()
)

risk_by_course["course_name_wrapped"] = risk_by_course["course_name"].apply(wrap_label)

fig_risk = px.bar(
    risk_by_course,
    x="course_name_wrapped",
    y="at_risk_learners",
    title="At-Risk Learners by Course",
    text="at_risk_learners",
    labels={
        "course_name_wrapped": "Course",
        "at_risk_learners": "At-Risk Learners"
    }
)

fig_risk.update_traces(textposition="outside")

fig_risk.update_layout(
    margin=dict(t=60, b=120),
    xaxis_title="Course",
    yaxis_title="At-Risk Learners"
)

st.plotly_chart(fig_risk, use_container_width=True)

# =========================
# AVERAGE QUIZ SCORE BY WEEK
# =========================

score_by_week = (
    filtered_df
    .replace({"avg_score": {0: np.nan}})
    .groupby(["course_name", "week_number"])
    .agg(
        average_score=("avg_score", "mean")
    )
    .reset_index()
)

fig_score = px.line(
    score_by_week,
    x="week_number",
    y="average_score",
    color="course_name",
    markers=True,
    title="Average Quiz Score by Week",
    labels={
        "week_number": "Week",
        "average_score": "Average Score (%)",
        "course_name": "Course"
    }
)

fig_score.update_layout(
    xaxis=dict(dtick=1),
    yaxis=dict(range=[0, 100])
)

st.plotly_chart(fig_score, use_container_width=True)

# =========================
# ENGAGEMENT SCORE BY COURSE
# =========================

engagement_by_course = (
    build_course_learner_summary(filtered_df)
    .groupby("course_name")
    .agg(
        avg_engagement_score=("engagement_score", "mean")
    )
    .reset_index()
)

engagement_by_course["avg_engagement_score"] *= 100
engagement_by_course["course_name_wrapped"] = engagement_by_course["course_name"].apply(wrap_label)

fig_engagement = px.bar(
    engagement_by_course,
    x="course_name_wrapped",
    y="avg_engagement_score",
    title="Average Engagement Score by Course",
    text=engagement_by_course["avg_engagement_score"].round(1),
    labels={
        "course_name_wrapped": "Course",
        "avg_engagement_score": "Engagement Score (%)"
    }
)

fig_engagement.update_traces(textposition="outside")

fig_engagement.update_layout(
    yaxis=dict(range=[0, 100]),
    margin=dict(t=60, b=120),
    xaxis_title="Course",
    yaxis_title="Engagement Score (%)"
)

st.plotly_chart(fig_engagement, use_container_width=True)

# =========================
# LEARNER TABLE
# =========================

st.subheader("Learner-Level Progress Table")

display_cols = [
    "course_name", "fullname", "email", "week_number",
    "completed", "total_quizzes", "attempted_quizzes",
    "avg_score", "course_completed", "engagement_score",
    "week_at_risk"
]

display_df = filtered_df[display_cols].sort_values(
    ["course_name", "fullname", "week_number"]
).reset_index(drop=True)

display_df.insert(0, "S/N", display_df.index + 1)

display_df["week_at_risk"] = display_df["week_at_risk"].map({
    1: "Yes",
    0: "No"
})

display_df = display_df.rename(columns={
    "course_name": "Course",
    "fullname": "Full Name",
    "email": "Email",
    "week_number": "Week",
    "completed": "Week Completed",
    "total_quizzes": "Total Quizzes",
    "attempted_quizzes": "Attempted Quizzes",
    "avg_score": "Average Score",
    "course_completed": "Course Completed",
    "engagement_score": "Engagement Score",
    "week_at_risk": "At Risk"
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# =========================
# DOWNLOAD DATA
# =========================

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="njfp_lms_dashboard_data.csv",
    mime="text/csv"
)

# =========================
# KPI DOCUMENTATION
# =========================

with st.expander("KPI Documentation"):
    st.markdown("""
## KPI Documentation

### 1. Total Learners
**Meaning:**  
The total number of unique learners enrolled in the selected course(s).

**Formula:**  
`Total Learners = Count of unique user_id`

---

### 2. Active Learners
**Meaning:**  
Learners who showed activity in the LMS by completing at least one week or attempting at least one quiz.

**Formula:**  
A learner is active if:

`total weeks completed > 0 OR total quizzes attempted > 0`

---

### 3. At-Risk Learners
**Meaning:**  
Learners who have not completed any week and have not attempted any quiz. These learners may need follow-up or support.

**Formula:**  
A learner is at risk if:

`total weeks completed = 0 AND total quizzes attempted = 0`

---

### 4. Course Completion Rate
**Meaning:**  
The percentage of learners who completed the course.

**Formula:**  
`Course Completion Rate = Completed Learners / Total Learners × 100`

---

### 5. Weekly Completion Rate
**Meaning:**  
Shows the percentage of learners who completed each week.

**Formula:**  
`Weekly Completion Rate = Users who completed the week / Total users in that week × 100`

---

### 6. Quiz Attempt Rate
**Meaning:**  
Shows the proportion of quizzes attempted by learners.

**Formula:**  
`Quiz Attempt Rate = Attempted Quizzes / Total Quizzes`

---

### 7. Average Quiz Score
**Meaning:**  
The average percentage score achieved by learners in quizzes.

**Formula:**  
`Average Quiz Score = Mean of avg_score`

---

### 8. Engagement Score
**Meaning:**  
A weighted score that combines weekly completion, quiz participation, and quiz performance.

**Formula:**  

`Engagement Score = (Completion × 40%) + (Quiz Attempt Activity × 30%) + (Quiz Score × 30%)`

Where:
- Completion = 1 if learner completed the week, otherwise 0
- Quiz Attempt Activity = 1 if learner attempted any quiz, otherwise 0
- Quiz Score = average quiz score divided by 100

**Interpretation:**
- 80% and above = Highly engaged
- 50% to 79% = Moderately engaged
- Below 50% = Low engagement / needs attention

---

### 9. Learning Effectiveness
**Meaning:**  
Shows whether learners are not just completing activities, but also performing well in quizzes.

**Formula:**  
`Learning Effectiveness = Completion × Average Quiz Score`

A learner who completes a week but scores low may need academic support.  
A learner who scores well but does not complete may need engagement support.
""")