import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
import io

st.set_page_config(page_title="MedShift — MVP", page_icon="🩺", layout="wide")

st.title("🩺 MedShift — Duty Scheduling (MVP)")
st.caption("Focus: working algorithm. UI kept minimal on purpose.")

# -----------------------------
# Sidebar — Inputs
# -----------------------------
st.sidebar.header("⚙️ Setup")

with st.sidebar.expander("📅 Schedule Window", expanded=True):
    start_date = st.date_input("Start date", value=datetime.today().date())
    weeks = st.number_input("Number of weeks", min_value=1, max_value=12, value=4)
    days = weeks * 7
    end_date = start_date + timedelta(days=days - 1)

with st.sidebar.expander("🧑‍⚕️ Interns & quotas", expanded=True):
    st.markdown("Upload a CSV. Columns: **first_name,last_name,email,group,max_shifts**")
    default_interns_csv = """first_name,last_name,email,group,max_shifts
Dana,Levi,dana@example.com,A,6
Noa,Cohen,noa@example.com,A,6
Yossi,Bar,yossi@example.com,B,6
Amit,Meliches,amit@example.com,B,6
"""
    interns_file = st.file_uploader("Interns CSV", type=["csv"], key="interns")
    if interns_file:
        interns_df = pd.read_csv(interns_file)
    else:
        interns_df = pd.read_csv(io.StringIO(default_interns_csv))
    st.dataframe(interns_df, use_container_width=True)

with st.sidebar.expander("🚫 Unavailability (optional)", expanded=False):
    st.markdown("Upload a CSV with columns: **email,date** (YYYY-MM-DD)")
    default_unavail_csv = """email,date
dana@example.com,2025-09-03
noa@example.com,2025-09-10
"""
    unavail_file = st.file_uploader("Unavailability CSV", type=["csv"], key="unavail")
    if unavail_file:
        unavail_df = pd.read_csv(unavail_file, parse_dates=["date"])
    else:
        unavail_df = (
            pd.read_csv(io.StringIO(default_unavail_csv), parse_dates=["date"])
            if st.checkbox("Use sample unavailability")
            else pd.DataFrame(columns=["email", "date"])
        )
    if not unavail_df.empty:
        st.dataframe(unavail_df, use_container_width=True)

with st.sidebar.expander("🏷️ Group quotas (optional)", expanded=False):
    st.markdown("Upload CSV with columns: **group,target_shifts**")
    default_group_csv = """group,target_shifts
A,16
B,12
"""
    group_file = st.file_uploader("Group quotas CSV", type=["csv"], key="group_quotas")
    if group_file:
        group_quota_df = pd.read_csv(group_file)
    else:
        group_quota_df = pd.read_csv(io.StringIO(default_group_csv)) if st.checkbox("Use sample group quotas") else pd.DataFrame(columns=["group","target_shifts"])
    if not group_quota_df.empty:
        st.dataframe(group_quota_df, use_container_width=True)

group_targets = {}
if 'group_quota_df' in locals() and not group_quota_df.empty:
    for _, r in group_quota_df.iterrows():
        if pd.notna(r.get("group")) and pd.notna(r.get("target_shifts")):
            try:
                group_targets[str(r["group"])] = int(r["target_shifts"])
            except Exception:
                pass

with st.sidebar.expander("📜 Rules", expanded=True):
    st.caption("One 26h duty per day. Rest: Duty on D ⇒ no duties on D+1,D+2 (earliest D+3).")
    max_shifts_per_day = 1
    min_days_between_shifts = 3
    respect_groups = st.checkbox("Balance by seniority groups", value=True)
    lock_weekends = st.checkbox("Prefer not to repeat same intern on weekend", value=True)

# -----------------------------
# Helper structures
# -----------------------------
def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

calendar_days = [d for d in daterange(start_date, end_date)]

if "max_shifts" not in interns_df.columns:
    interns_df["max_shifts"] = 9999
interns_df["email"] = interns_df["email"].astype(str).str.strip().str.lower()

unavail = set()
if not unavail_df.empty:
    unavail_df["email"] = unavail_df["email"].astype(str).str.strip().str.lower()
    unavail = {(row.email, pd.to_datetime(row.date).date()) for _, row in unavail_df.iterrows()}

schedule_df = pd.DataFrame()

# -----------------------------
# Scheduler
# -----------------------------
def build_schedule():
    interns = interns_df.copy()
    interns["assigned"] = 0
    interns["last_day"] = pd.NaT

    if respect_groups and "group" in interns.columns:
        groups = {g: interns[interns.group == g].reset_index(drop=True) for g in interns.group.fillna("_").unique()}
        group_cycle = deque(sorted(groups.keys()))
    else:
        groups = {"all": interns}
        group_cycle = deque(["all"])

    assignments = []
    last_day_map = {row.email: None for _, row in interns.iterrows()}
    weekend_last = defaultdict(lambda: None)
    group_assigned = defaultdict(int)

    def next_pool():
        g = group_cycle[0]
        group_cycle.rotate(-1)
        return g, groups[g]

    for day in calendar_days:
        year, week_idx, _ = day.isocalendar()
        for slot in range(max_shifts_per_day):
            tried_groups = 0
            chosen = None
            while tried_groups < len(group_cycle) and chosen is None:
                g_name, pool = next_pool()
                candidates = []
                for _, r in pool.iterrows():
                    email = r.email
                    if r.assigned >= r.max_shifts:
                        continue
                    if (email, day) in unavail:
                        continue
                    last = last_day_map[email]
                    if last is not None and (day - last).days < min_days_between_shifts:
                        continue
                    if lock_weekends and day.weekday() >= 5:
                        prev_week_pick = weekend_last[(year, week_idx)]
                        if prev_week_pick == email:
                            continue
                    rest_days = 999 if last is None else (day - last).days
                    name_for_sort = r.get("last_name", r.get("first_name", ""))
                    gname = r["group"] if "group" in r and pd.notna(r["group"]) else None
                    if gname and gname in group_targets:
                        deficit = group_targets[gname] - group_assigned[gname]
                    else:
                        deficit = 0
                    candidates.append((-deficit, r.assigned, -rest_days, str(name_for_sort), email))
                if candidates:
                    candidates.sort()
                    chosen_email = candidates[0][4]
                    chosen = chosen_email
                    if gname:
                        group_assigned[gname] += 1
                    interns.loc[interns.email == chosen_email, "assigned"] += 1
                    interns.loc[interns.email == chosen_email, "last_day"] = pd.to_datetime(day)
                    if "group" in interns.columns and respect_groups:
                        groups[g_name].loc[groups[g_name].email == chosen_email, "assigned"] += 1
                        groups[g_name].loc[groups[g_name].email == chosen_email, "last_day"] = pd.to_datetime(day)
                    last_day_map[chosen_email] = day
                    if day.weekday() >= 5:
                        weekend_last[(year, week_idx)] = chosen_email
                    assignments.append({"date": day, "slot": slot + 1, "email": chosen_email})
                else:
                    tried_groups += 1
            if chosen is None:
                assignments.append({"date": day, "slot": slot + 1, "email": None})

    out = pd.DataFrame(assignments)
    join_cols = [c for c in ["email", "first_name", "last_name", "group"] if c in interns_df.columns]
    out = out.merge(interns_df[join_cols], how="left", on="email")
    out.sort_values(["date", "slot"], inplace=True)
    return out

# -----------------------------
# Validation
# -----------------------------
def validate_schedule(df, min_rest_days=3, max_per_day=1):
    issues = []
    if df.empty:
        return issues
    day_counts = df.groupby("date").size()
    for day, cnt in day_counts.items():
        if cnt > max_per_day:
            issues.append(f"More than {max_per_day} assignment on {day}.")
    df2 = df.dropna(subset=["email"]).copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.date
    by_email = df2.groupby("email")["date"].apply(lambda s: sorted(set(s)))
    for email, dates in by_email.items():
        for i, d in enumerate(dates):
            for j in range(i+1, len(dates)):
                gap = (dates[j] - d).days
                if gap < min_rest_days:
                    issues.append(f"Rest rule violated for {email}: {d} -> {dates[j]} (gap {gap}d)")
    return issues

WEEKEND_DAYS = {4,5}  # Fri,Sat
def weekend_issues(df):
    issues = []
    if df.empty:
        return issues
    df2 = df.dropna(subset=["email"]).copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.date
    df2["weekday"] = pd.to_datetime(df2["date"]).dt.weekday
    df2["iso_year_week"] = pd.to_datetime(df2["date"]).dt.isocalendar().year.astype(str) + "-" + \
                           pd.to_datetime(df2["date"]).dt.isocalendar().week.astype(str)
    for (email, yw), g in df2.groupby(["email","iso_year_week"]):
        wk = g[g["weekday"].isin(WEEKEND_DAYS)]
        if len(wk) >= 2:
            days_str = ", ".join(sorted(str(d) for d in wk["date"].unique()))
            issues.append(f"Weekend repeat for {email} in week {yw}: {days_str}")
    return issues

# -----------------------------
# Main CTA
# -----------------------------
colA, colB = st.columns([1,1])
with colA:
    if st.button("🧮 Run Scheduler"):
        schedule_df = build_schedule()
        st.session_state["schedule_df"] = schedule_df

schedule_df = st.session_state.get("schedule_df", pd.DataFrame())

if not schedule_df.empty:
    problems = validate_schedule(schedule_df, min_rest_days=3, max_per_day=1)
    problems += weekend_issues(schedule_df)
    if problems:
        st.error("Validation issues found:")
        for p in problems:
            st.write("• " + p)
    else:
        st.success("Validation passed: one duty/day, 3-day rest, and weekend respected.")

    st.subheader("📅 Calendar view")
    pivot = schedule_df.copy()
    pivot["assignee"] = pivot.apply(
        lambda r: f"{r['first_name']} {r['last_name']}".strip() if pd.notna(r.get("email")) else "—",
        axis=1,
    )
    calendar = pivot.pivot_table(index="date", columns="slot", values="assignee", aggfunc=lambda x: " / ".join([str(i) for i in x]))
    calendar.columns = [f"Slot {i}" for i in calendar.columns]
    st.dataframe(calendar, use_container_width=True)

    st.subheader("📋 Flat table")
    st.dataframe(schedule_df, use_container_width=True)

    st.subheader("📈 Load per intern")
    counts = (
        schedule_df.dropna(subset=["email"])
        .groupby(["email","first_name","last_name"])
        .size()
        .reset_index(name="shifts")
    )
    st.dataframe(counts.sort_values("shifts", ascending=False), use_container_width=True)

    if group_targets:
        st.subheader("🎯 Group quota check")
        actual_by_group = (
            schedule_df.dropna(subset=["email"])
            .groupby("group")
            .size()
            .rename("actual")
            .to_frame()
        )
        target_df = pd.DataFrame({"target": pd.Series(group_targets, dtype="Int64")})
        compare = target_df.join(actual_by_group, how="left").fillna(0).astype({"actual":"int"})
        compare["delta"] = compare["actual"] - compare["target"]
        st.dataframe(compare, use_container_width=True)

    csv_buf = io.StringIO()
    schedule_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download CSV", data=csv_buf.getvalue(), file_name=f"medshift_schedule_{start_date}_{end_date}.csv", mime="text/csv")
