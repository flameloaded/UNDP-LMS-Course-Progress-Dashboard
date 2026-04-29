import os
import time
import re
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("MOODLE_BASE_URL")
TOKEN = os.getenv("MOODLE_TOKEN")

REQUEST_DELAY = 0.08
TIMEOUT = 60

include_ids = [5,7,17,18,19,20,21,22]




def call_moodle(wsfunction, **kwargs):
    params = {
        "wstoken": TOKEN,
        "moodlewsrestformat": "json",
        "wsfunction": wsfunction,
    }
    params.update(kwargs)

    response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and data.get("exception"):
        raise Exception(f"{wsfunction} failed: {data.get('message')}")

    time.sleep(REQUEST_DELAY)
    return data


def safe_call(wsfunction, **kwargs):
    try:
        return call_moodle(wsfunction, **kwargs), None
    except Exception as e:
        return None, str(e)


def classify_module(name):
    if pd.isna(name):
        return "other"

    n = str(name).strip().lower()

    if "quiz" in n:
        return "quiz"

    if (
        re.search(r"mark\s+week\s*\d+\s+as\s+complete", n)
        or re.search(r"mark this as complete after finishing all week\s*\d+\s+materials", n)
    ):
        return "week_completion"

    if re.fullmatch(r"week\s*\d+", n):
        return "week_label"

    if "complete" in n:
        return "other_completion"

    return "other"


def extract_week_number(name):
    if pd.isna(name):
        return np.nan

    match = re.search(r"week\s*(\d+)", str(name).strip().lower())
    return int(match.group(1)) if match else np.nan


def fetch_courses():
    courses = call_moodle("core_course_get_courses")
    courses_df = pd.DataFrame(courses)

    courses_df = courses_df[courses_df["id"] != 1].copy()

    courses_df = courses_df.rename(columns={
        "id": "course_id",
        "fullname": "course_name",
        "shortname": "course_shortname"
    })

    # Exclude unwanted courses
    exclude_ids = [1, 2, 3, 4, 6, 12, 13, 14, 16, 23, 28]
    courses_df = courses_df[~courses_df["course_id"].isin(exclude_ids)].copy()

    # Step 2: Include ONLY what you want (whitelist)
    include_ids = [5, 7, 17, 18, 19, 20, 21, 22]
    courses_df = courses_df[courses_df["course_id"].isin(include_ids)].copy()
    
    return courses_df


def fetch_users(courses_df):
    rows = []

    for _, course in courses_df.iterrows():
        course_id = int(course["course_id"])

        users_data, err = safe_call(
            "core_enrol_get_enrolled_users",
            courseid=course_id
        )

        if err or not isinstance(users_data, list):
            print(f"Skipping users for course {course_id}: {err}")
            continue

        for user in users_data:
            rows.append({
                "course_id": course_id,
                "course_name": course["course_name"],
                "user_id": user.get("id"),
                "fullname": user.get("fullname"),
                "email": user.get("email"),
                "username": user.get("username"),
                "suspended": user.get("suspended"),
            })

    return pd.DataFrame(rows).drop_duplicates(subset=["course_id", "user_id"])


def fetch_quizzes(courses_df):
    course_ids = courses_df["course_id"].dropna().astype(int).tolist()

    params = {
        "wstoken": TOKEN,
        "moodlewsrestformat": "json",
        "wsfunction": "mod_quiz_get_quizzes_by_courses"
    }

    for i, cid in enumerate(course_ids):
        params[f"courseids[{i}]"] = cid

    response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    quizzes = payload.get("quizzes", [])

    quizzes_df = pd.DataFrame([{
        "course_id": q.get("course"),
        "quiz_id": q.get("id"),
        "quiz_name": q.get("name"),
        "grade_max": q.get("grademax"),
    } for q in quizzes])

    return quizzes_df


def fetch_quiz_results(quizzes_df, users_df):
    users_by_course = {
        course_id: group.reset_index(drop=True)
        for course_id, group in users_df.groupby("course_id")
    }

    records = []

    for _, quiz in quizzes_df.iterrows():
        course_id = int(quiz["course_id"])
        quiz_id = int(quiz["quiz_id"])

        enrolled_users = users_by_course.get(course_id)

        if enrolled_users is None or enrolled_users.empty:
            continue

        print(f"Processing quiz: {quiz['quiz_name']}")

        for _, user in enrolled_users.iterrows():
            grade_data, err = safe_call(
                "mod_quiz_get_user_best_grade",
                quizid=quiz_id,
                userid=int(user["user_id"])
            )

            best_grade = None

            if isinstance(grade_data, dict):
                best_grade = grade_data.get("grade")

            records.append({
                "course_id": course_id,
                "course_name": user["course_name"],
                "quiz_id": quiz_id,
                "quiz_name": quiz["quiz_name"],
                "user_id": user["user_id"],
                "fullname": user["fullname"],
                "email": user["email"],
                "best_grade": best_grade,
                "grade_max": 10,
            })

    quiz_df = pd.DataFrame(records)

    quiz_df["best_grade"] = pd.to_numeric(quiz_df["best_grade"], errors="coerce")
    quiz_df["grade_max"] = pd.to_numeric(quiz_df["grade_max"], errors="coerce")

    quiz_df["grade_percent"] = np.where(
        quiz_df["best_grade"].notna() & quiz_df["grade_max"].gt(0),
        (quiz_df["best_grade"] / quiz_df["grade_max"]) * 100,
        np.nan
    )

    quiz_df["attempted"] = quiz_df["grade_percent"].notna()

    return quiz_df


def fetch_contents(courses_df):
    rows = []

    for course_id in courses_df["course_id"].dropna().astype(int).unique():
        data, err = safe_call("core_course_get_contents", courseid=course_id)

        if err:
            print(f"Skipping contents for course {course_id}: {err}")
            continue

        for section in data:
            for mod in section.get("modules", []):
                rows.append({
                    "course_id": course_id,
                    "section_id": section.get("id"),
                    "section_name": section.get("name"),
                    "section_num": section.get("section"),
                    "cmid": mod.get("id"),
                    "module_name": mod.get("name"),
                    "modname": mod.get("modname"),
                    "completion": mod.get("completion"),
                })

    contents_df = pd.DataFrame(rows)

    contents_df["module_name_clean"] = contents_df["module_name"].astype(str).str.strip()
    contents_df["module_type"] = contents_df["module_name_clean"].apply(classify_module)
    contents_df["week_number"] = contents_df["section_name"].apply(extract_week_number)

    return contents_df


def fetch_course_completion(users_df):
    rows = []

    for _, user in users_df.iterrows():
        data, err = safe_call(
            "core_completion_get_course_completion_status",
            courseid=int(user["course_id"]),
            userid=int(user["user_id"])
        )

        if err:
            continue

        completion = data.get("completionstatus", {}) if isinstance(data, dict) else {}

        rows.append({
            "course_id": user["course_id"],
            "user_id": user["user_id"],
            "course_completed": 1 if completion.get("completed") else 0,
            "timecompleted": completion.get("timecompleted"),
        })

    return pd.DataFrame(rows)


def fetch_week_progress(users_df, week_completion_df):
    rows = []

    for _, user in users_df.iterrows():
        course_id = int(user["course_id"])
        user_id = int(user["user_id"])

        data, err = safe_call(
            "core_completion_get_activities_completion_status",
            courseid=course_id,
            userid=user_id
        )

        if err:
            continue

        statuses = data.get("statuses", []) if isinstance(data, dict) else []
        status_map = {s.get("cmid"): s.get("state", 0) for s in statuses}

        course_weeks = week_completion_df[week_completion_df["course_id"] == course_id]

        for _, wk in course_weeks.iterrows():
            state = status_map.get(wk["cmid"], 0)

            rows.append({
                "course_id": course_id,
                "course_name": user["course_name"],
                "user_id": user_id,
                "fullname": user["fullname"],
                "email": user["email"],
                "section_name": wk["section_name"],
                "week_name": wk["module_name"],
                "week_number": wk["week_number"],
                "cmid": wk["cmid"],
                "week_completed": 1 if state == 1 else 0,
            })

    return pd.DataFrame(rows)


def build_dataset():
    print("Fetching courses...", flush=True)
    courses_df = fetch_courses()

    print("Fetching users...", flush=True)
    users_df = fetch_users(courses_df)

    print("Fetching quizzes...", flush=True)
    quizzes_df = fetch_quizzes(courses_df)

    print("Fetching quiz results...", flush=True)
    quiz_df = fetch_quiz_results(quizzes_df, users_df)

    os.makedirs("data", exist_ok=True)
    quiz_df.to_csv("data/quiz_results.csv", index=False)

    print("Fetching course contents...", flush=True)
    contents_df = fetch_contents(courses_df)

    week_completion_df = contents_df[
        contents_df["module_type"] == "week_completion"
    ].copy()

    print("Fetching course completion...", flush=True)
    course_completion_df = fetch_course_completion(users_df)

    print("Fetching weekly progress...", flush=True)
    week_progress_df = fetch_week_progress(users_df, week_completion_df)

    quiz_week_summary = (
        quiz_df
        .merge(
            contents_df[["course_id", "module_name", "week_number"]],
            left_on=["course_id", "quiz_name"],
            right_on=["course_id", "module_name"],
            how="left"
        )
        .groupby(["course_id", "user_id", "week_number"], dropna=False)
        .agg(
            total_quizzes=("quiz_id", "nunique"),
            attempted_quizzes=("attempted", "sum"),
            avg_score=("grade_percent", "mean")
        )
        .reset_index()
    )

    final_table = week_progress_df.merge(
        quiz_week_summary,
        on=["course_id", "user_id", "week_number"],
        how="left"
    )

    final_table = final_table.merge(
        course_completion_df[["course_id", "user_id", "course_completed"]],
        on=["course_id", "user_id"],
        how="left"
    )

    final_table["total_quizzes"] = final_table["total_quizzes"].fillna(0)
    final_table["attempted_quizzes"] = final_table["attempted_quizzes"].fillna(0)
    final_table["avg_score"] = final_table["avg_score"].fillna(0)
    final_table["course_completed"] = final_table["course_completed"].fillna(0)

    final_table["at_risk"] = np.where(
        (final_table["week_completed"] == 0) & (final_table["attempted_quizzes"] == 0),
        "Yes",
        "No"
    )

    final_table.insert(0, "S/N", range(1, len(final_table) + 1))

    return final_table


OUTPUT_FILE = "data/undp_lms_dataset.csv"
BACKUP_FILE = "data/undp_lms_dataset_backup.csv"
TEMP_FILE = "data/temp_dataset.csv"

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    try:
        df = build_dataset()

        # keep old file as backup before replacing
        if os.path.exists(OUTPUT_FILE):
            old_df = pd.read_csv(OUTPUT_FILE)
            old_df.to_csv(BACKUP_FILE, index=False)

        # write new data safely
        df.to_csv(TEMP_FILE, index=False)
        os.replace(TEMP_FILE, OUTPUT_FILE)

        print("✅ Data updated successfully")

    except Exception as e:
        print("❌ Pipeline failed:", e)
        print("⚠️ Old dataset was not replaced.")