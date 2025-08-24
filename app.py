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
    st.markdown("Upload a CSV or edit inline. Columns: **first_name,last_name,email,group,max_shifts**")
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

with st.sidebar.expander("📜 Rules", expanded=True):
    max_shifts_per_day = st.number_input("Max shifts per day (coverage target)", 1, 10, 2)
    min_days_between_shifts = st.number_input("Min rest days between shifts", 0, 7, 1)
    respect_groups = st.checkbox(
        "Balance by seniority groups",
        value=True,
        help="Alternate between groups where possible.",
    )
    lock_weekends = st.checkbox("Prefer not to repeat same intern on weekend", value=True)

# -----------------------------
# Helper structures
# -----------------------------
def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

# precompute calendar
calendar_days = [d for d in daterange(start_date, end_date)]

# Index interns
if "max_shifts" not in interns_df.columns:
    interns_df["max_shifts"] = 9999
interns_df["email"] = interns_df["email"].astype(str).str.strip().str.lower()

# Build unavailability set
unavail = set()
if not unavail_df.empty:
    unavail_df["email"] = unavail_df["email"].astype(str).str.strip().str.lower()
    unavail = {(row.email, pd.to_datetime(row.date).date()) for _, row in unavail_df.iterrows()}

# State to show after run
schedule_df = pd.DataFrame()

# -----------------------------
# Greedy-balanced scheduler
# -----------------------------
def build_schedule():
    interns = interns_df.copy()
    interns["assigned"] = 0
    interns["last_day"] = pd.NaT

    # Split pools by group for fairness (optional)
    if respect_groups and "group" in interns.columns:
        groups = {g: interns[interns.group == g].reset_index(drop=True) for g in interns.group.fillna("_").unique()}
        group_cycle = deque(sorted(groups.keys()))
    else:
        groups = {"all": interns}
        group_cycle = deque(["all"])

    assignments = []  # list of {date, slot, email}

    # Helper: pick next candidate list (group alternation)
    def next_pool():
        g = group_cycle[0]
        group_cycle.rotate(-1)
        return g, groups[g]

    # Track last assignment per intern for rest spacing
    last_day_map = {row.email: None for _, row in interns.iterrows()}

    # Weekend memory to avoid same person Sat+Sun repeatedly
    weekend_last = defaultdict(lambda: None)  # key: (year, week_idx) -> email

    for day in calendar_days:
        year, week_idx, _ = day.isocalendar()
        for slot in range(max_shifts_per_day):
            # Assemble candidate pool
            tried_groups = 0
            chosen = None
            while tried_groups < len(group_cycle) and chosen is None:
                g_name, pool = next_pool()

                # Filter availability & capacity
                candidates = []
                for _, r in pool.iterrows():
                    email = r.email

                    # capacity
                    if r.assigned >= r.max_shifts:
                        continue
                    # unavailability
                    if (email, day) in unavail:
                        continue
                    # rest rule
                    last = last_day_map[email]
                    if last is not None and (day - last).days < min_days_between_shifts:
                        continue
                    # weekend rule
                    if lock_weekends and day.weekday() >= 5:  # Sat/Sun
                        prev_week_pick = weekend_last[(year, week_idx)]
                        if prev_week_pick == email:
                            continue

                    # candidate score: fewer assigned first, then longest rest, then name tie-breaker
                    rest_days = 999 if last is None else (day - last).days
                    name_for_sort = r.get("last_name", r.get("first_name", ""))
                    candidates.append((r.assigned, -rest_days, str(name_for_sort), email))

                if candidates:
                    candidates.sort()  # lowest assigned, then longest rest (since negative), then alpha
                    chosen_email = candidates[0][3]
                    chosen = chosen_email

                    # mutate global interns & pools
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

            # if nobody found after scanning all groups, leave slot empty
            if chosen is None:
                assignments.append({"date": day, "slot": slot + 1, "email": None})

    out = pd.DataFrame(assignments)

    # Join intern names safely (only columns that exist)
    join_cols = [c for c in ["email", "first_name", "last_name", "group"] if c in interns_df.columns]
    out = out.merge(interns_df[join_cols], how="left", on="email")
    out.sort_values(["date", "slot"], inplace=True)
    return out

# -----------------------------
# Main CTA
# -----------------------------
colA, colB = st.columns([1, 1])
with colA:
    if st.button("🧮 Run Scheduler"):
        schedule_df = build_schedule()
        st.session_state["schedule_df"] = schedule_df

# Show results if available
schedule_df = st.session_state.get("schedule_df", pd.DataFrame())

if not schedule_df.empty:
    st.success("Schedule created! Scroll for views & export.")

    # Calendar view (pivot)
    pivot = schedule_df.copy()
    pivot["assignee"] = pivot.apply(
        lambda r: f"{r['first_name']} {r['last_name']}".strip()
        if pd.notna(r.get("email"))
        else "—",
        axis=1,
    )
    calendar = pivot.pivot_table(
        index="date",
        columns="slot",
        values="assignee",
        aggfunc=lambda x: " / ".join([str(i) for i in x]),
    )
    calendar.columns = [f"Slot {i}" for i in calendar.columns]

    st.subheader("📅 Calendar view")
    st.dataframe(calendar, use_container_width=True)

    st.subheader("📋 Flat table")
    st.dataframe(schedule_df, use_container_width=True)

    # Per-intern counts
    st.subheader("📈 Load per intern")
    counts = (
        schedule_df.dropna(subset=["email"])
        .groupby(["email", "first_name", "last_name"])
        .size()
        .reset_index(name="shifts")
    )
    st.dataframe(counts.sort_values("shifts", ascending=False), use_container_width=True)

    # Export
    csv_buf = io.StringIO()
    schedule_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download CSV",
        data=csv_buf.getvalue(),
        file_name=f"medshift_schedule_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

# -----------------------------
# Notes / TODOs
# -----------------------------
with st.expander("📝 TODO / next steps"):
    st.markdown(
        """
- Hard constraints to add: minimum per-intern monthly shifts, forbid certain weekdays per intern, team pairings.
- Soft constraints scoring (future): prefer evenly balanced weekends, seniority mix per day, avoid consecutive nights.
- Replace CSVs with a database (Supabase/Postgres) and auth.
- Add role-based UI later (Chief vs Intern).
- Add audit log & manual overrides.
"""
    )
