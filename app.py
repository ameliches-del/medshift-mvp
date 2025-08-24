import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import io
import calendar as pycal

st.set_page_config(page_title="MedShift — MVP", page_icon="🩺", layout="wide")
st.title("🩺 MedShift — Duty Scheduling (MVP)")
st.caption("Focus: algorithm first. UI is intentionally minimal.")

# -----------------------------
# Sidebar — Inputs
# -----------------------------
st.sidebar.header("⚙️ Setup")

with st.sidebar.expander("📅 Calendar Month", expanded=True):
    today = datetime.today()
    year = st.number_input("Year", min_value=2023, max_value=2100, value=today.year, step=1)
    month = st.selectbox("Month", options=list(range(1,13)), format_func=lambda m: f"{m:02d}", index=today.month-1)
    # Monthly window
    start_date = datetime(int(year), int(month), 1).date()
    last_day = pycal.monthrange(int(year), int(month))[1]
    end_date = datetime(int(year), int(month), last_day).date()
    st.write(f"Scheduling window: **{start_date} → {end_date}**")

with st.sidebar.expander("🧑‍⚕️ Interns & quotas", expanded=True):
    st.markdown("Upload a CSV. Columns: **first_name,last_name,email,group,max_shifts**")
    default_interns_csv = """first_name,last_name,email,group,max_shifts
Dana,Levi,dana@example.com,חדשים,10
Noa,Cohen,noa@example.com,לפני שלב א,10
Yossi,Bar,yossi@example.com,אחרי שלב א,10
Amit,Meliches,amit@example.com,חדשים,10
"""
    f = st.file_uploader("Interns CSV", type=["csv"], key="interns")
    if f:
        interns_df = pd.read_csv(f)
    else:
        interns_df = pd.read_csv(io.StringIO(default_interns_csv))
    st.dataframe(interns_df, use_container_width=True)

with st.sidebar.expander("🚫 Unavailability (optional)", expanded=False):
    st.markdown("Upload a CSV with columns: **email,date** (YYYY-MM-DD)")
    f = st.file_uploader("Unavailability CSV", type=["csv"], key="unavail")
    if f:
        unavail_df = pd.read_csv(f, parse_dates=["date"])
    else:
        unavail_df = pd.DataFrame(columns=["email","date"])
    if not unavail_df.empty:
        st.dataframe(unavail_df, use_container_width=True)

with st.sidebar.expander("🏷️ Group quotas (optional)", expanded=False):
    st.markdown("Upload CSV with columns: **group,target_shifts**  (targets for THIS month)")
    f = st.file_uploader("Group quotas CSV", type=["csv"], key="group_quotas")
    if f:
        group_quota_df = pd.read_csv(f)
    else:
        group_quota_df = pd.DataFrame(columns=["group","target_shifts"])
    if not group_quota_df.empty:
        st.dataframe(group_quota_df, use_container_width=True)

with st.sidebar.expander("📅 Holidays (optional)", expanded=False):
    st.markdown("Upload CSV with column: **date** (YYYY-MM-DD)")
    f = st.file_uploader("Holidays CSV", type=["csv"], key="holidays")
    if f:
        holidays_df = pd.read_csv(f, parse_dates=["date"])
    else:
        holidays_df = pd.DataFrame(columns=["date"])

# Map quotas
group_targets = {}
if 'group_quota_df' in locals() and not group_quota_df.empty:
    for _, r in group_quota_df.iterrows():
        if pd.notna(r.get("group")) and pd.notna(r.get("target_shifts")):
            group_targets[str(r["group"])] = int(r["target_shifts"])

with st.sidebar.expander("📜 Rules", expanded=True):
    st.caption("One 26h duty per day. Rest rule: Duty on D ⇒ no duties on D+1, D+2 (earliest D+3).")
    max_shifts_per_day = 1  # fixed by spec
    min_days_between_shifts = 3
    lock_weekends = st.checkbox("Avoid same intern twice on the same ISO-weekend", value=True)

# -----------------------------
# Helpers & precomputed sets
# -----------------------------
WEEKEND_DAYS = {4, 5}  # Fri(4), Sat(5)  (Mon=0..Sun=6)

def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

calendar_days = [d for d in daterange(start_date, end_date)]

if "max_shifts" not in interns_df.columns:
    interns_df["max_shifts"] = 9999
interns_df["email"] = interns_df["email"].astype(str).str.strip().str.lower()

unavail = set()
if 'unavail_df' in locals() and not unavail_df.empty:
    unavail_df["email"] = unavail_df["email"].astype(str).str.strip().str.lower()
    unavail = {(row.email, pd.to_datetime(row.date).date()) for _, row in unavail_df.iterrows()}

# -----------------------------
# Scheduler
# -----------------------------
def build_schedule():
    interns = interns_df.copy()
    interns["assigned"] = 0
    interns["last_day"] = pd.NaT

    # overflow priority: higher target ⇒ higher priority
    group_order = sorted(group_targets.keys(), key=lambda g: group_targets[g], reverse=True) if group_targets else []
    group_rank = {g: i for i, g in enumerate(group_order)}  # 0 = highest

    assignments = []
    last_day_map = {row.email: None for _, row in interns.iterrows()}
    weekend_last = defaultdict(lambda: None)  # (year, week) -> email
    group_assigned = defaultdict(int)

    def leading_group():
        if not group_targets:
            return None
        deficits = {g: group_targets[g] - group_assigned[g] for g in group_targets}
        return max(deficits, key=lambda g: deficits[g])  # largest deficit

    PREF_WED_FRI = {2, 4}  # Wed/Fri

    def all_targets_met():
        if not group_targets:
            return True
        for g, t in group_targets.items():
            if group_assigned[g] < t:
                return False
        return True

    # ----- PRE-PASS: fair distribution of weekend/holiday (one per intern first) -----
    hol_set = set(pd.to_datetime(holidays_df["date"]).dt.date) if not holidays_df.empty else set()
    special_days = [d for d in calendar_days if (d.weekday() in WEEKEND_DAYS) or (d in hol_set)]

    def intern_group(email):
        try:
            row = interns.loc[interns.email == email].iloc[0]
            return row.get("group") if pd.notna(row.get("group")) else None
        except Exception:
            return None

    def has_special(email):
        return any(a["email"] == email and ((a["date"].weekday() in WEEKEND_DAYS) or (a["date"] in hol_set)) for a in assignments)

    for day in special_days:
        # already filled? (one slot/day)
        if any(a["date"] == day and a["email"] is not None for a in assignments):
            continue
        year, week_idx, _ = day.isocalendar()

        cand = []
        for _, r in interns.iterrows():
            email = r.email
            if has_special(email):
                continue
            if r.assigned >= r.max_shifts:
                continue
            if (email, day) in unavail:
                continue
            last = last_day_map[email]
            if last is not None and (day - last).days < min_days_between_shifts:
                continue
            if lock_weekends and day.weekday() in WEEKEND_DAYS and weekend_last[(year, week_idx)] == email:
                continue

            gname = intern_group(email)
            target = group_targets.get(gname, 0)
            rest_days = 999 if last is None else (day - last).days
            cand.append((-target, r.assigned, -rest_days, str(r.get("last_name", r.get("first_name",""))), email))

        if cand:
            cand.sort()
            chosen_email = cand[0][4]
            assignments.append({"date": day, "slot": 1, "email": chosen_email})
            last_day_map[chosen_email] = day
            interns.loc[interns.email == chosen_email, "assigned"] += 1
            gname = intern_group(chosen_email)
            if gname:
                group_assigned[gname] += 1
            if day.weekday() in WEEKEND_DAYS:
                weekend_last[(year, week_idx)] = chosen_email

    # ----- MAIN LOOP: always fill the day (no empty days) -----
    for day in calendar_days:
        if any(a["date"] == day and a["email"] is not None for a in assignments):
            continue

        year, week_idx, _ = day.isocalendar()
        lead = leading_group()

        def collect_candidates(relax_weekend=False, relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=False):
            cands = []
            for _, r in interns.iterrows():
                email = r.email
                # availability
                if not ignore_unavail and (email, day) in unavail:
                    continue
                # caps
                if not ignore_caps and r.assigned >= r.max_shifts:
                    continue

                last = last_day_map[email]
                # rest rule
                if not relax_rest and last is not None and (day - last).days < min_days_between_shifts:
                    continue
                # weekend repetition
                if not relax_weekend and lock_weekends and day.weekday() in WEEKEND_DAYS and weekend_last[(year, week_idx)] == email:
                    continue

                gname = r.get("group") if pd.notna(r.get("group")) else None
                deficit = max(0, group_targets.get(gname, 0) - group_assigned[gname]) if gname else 0

                # Hard targets phase: if not all met, require deficit>0
                if not overflow and not all_targets_met() and deficit == 0:
                    continue

                rest_days = 999 if last is None else (day - last).days
                name_for_sort = r.get("last_name", r.get("first_name", ""))

                if not overflow:
                    # still filling targets
                    wed_fri_bonus = 0.5 if (lead and gname == lead and day.weekday() in PREF_WED_FRI and deficit > 0) else 0.0
                    score = deficit + wed_fri_bonus
                else:
                    # overflow: by target rank (higher first)
                    score = 100.0 - (group_rank.get(gname, 99))

                cands.append((-score, r.assigned, -rest_days, str(name_for_sort), email))
            return cands

        # Try in stages until we must fill:
        stages = [
            # 1) strict targets / or overflow strict
            dict(relax_weekend=False, relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=False),
            dict(relax_weekend=False, relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=True),
            # 2) relax weekend repetition
            dict(relax_weekend=True,  relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=False),
            dict(relax_weekend=True,  relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=True),
            # 3) relax rest
            dict(relax_weekend=True,  relax_rest=True,  ignore_unavail=False, ignore_caps=False, overflow=False),
            dict(relax_weekend=True,  relax_rest=True,  ignore_unavail=False, ignore_caps=False, overflow=True),
            # 4) ignore caps
            dict(relax_weekend=True,  relax_rest=True,  ignore_unavail=False, ignore_caps=True,  overflow=True),
            # 5) last resort: ignore unavailability too (to guarantee a duty every day)
            dict(relax_weekend=True,  relax_rest=True,  ignore_unavail=True,  ignore_caps=True,  overflow=True),
        ]

        chosen_email = None
        for params in stages:
            cands = collect_candidates(**params)
            if cands:
                cands.sort()
                chosen_email = cands[0][4]
                break

        # Must always assign someone
        if chosen_email is None:
            # absolute last fallback: pick least assigned globally
            tmp = interns.sort_values(["assigned"]).iloc[0]
            chosen_email = tmp.email

        # assign
        assignments.append({"date": day, "slot": 1, "email": chosen_email})
        last_day_map[chosen_email] = day
        interns.loc[interns.email == chosen_email, "assigned"] += 1

        # counters
        row = interns.loc[interns.email == chosen_email].iloc[0]
        gname = row.get("group") if pd.notna(row.get("group")) else None
        if gname:
            # increment towards target as long as we're in month; overflow handled naturally
            group_assigned[gname] += 1
        if day.weekday() in WEEKEND_DAYS:
            weekend_last[(year, week_idx)] = chosen_email

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
    # one per day
    day_counts = df.groupby("date")["email"].apply(lambda s: s.notna().sum())
    for day, cnt in day_counts.items():
        if cnt != max_per_day:
            issues.append(f"Day {day} must have exactly {max_per_day} duty; got {cnt}.")
    # rest rule
    df2 = df.dropna(subset=["email"]).copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.date
    by_email = df2.groupby("email")["date"].apply(lambda s: sorted(list(s)))
    for email, dates in by_email.items():
        for i, d in enumerate(dates):
            for j in range(i+1, len(dates)):
                gap = (dates[j] - d).days
                if gap < min_rest_days:
                    issues.append(f"Rest rule violated for {email}: {d} -> {dates[j]} (gap {gap}d)")
    return issues

def weekend_issues(df):
    issues = []
    if df.empty:
        return issues
    df2 = df.dropna(subset=["email"]).copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.date
    df2["weekday"] = pd.to_datetime(df2["date"]).dt.weekday
    df2["iso_year_week"] = pd.to_datetime(df2["date"]).dt.isocalendar().year.astype(str) + "-" + \
                           pd.to_datetime(df2["date"]).dt.isocalendar().week.astype(str)
    wkend = df2[df2["weekday"].isin(WEEKEND_DAYS)]
    for (email, yw), g in wkend.groupby(["email","iso_year_week"]):
        if len(g) >= 2:
            days_str = ", ".join(sorted(str(d) for d in g["date"].unique()))
            issues.append(f"Weekend repeat for {email} in week {yw}: {days_str}")
    return issues

def must_have_weekend_or_holiday(df, holidays_df):
    issues = []
    if df.empty:
        return issues
    hol_set = set(pd.to_datetime(holidays_df["date"]).dt.date) if not holidays_df.empty else set()
    df2 = df.dropna(subset=["email"]).copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.date
    df2["weekday"] = pd.to_datetime(df2["date"]).dt.weekday
    by_email = df2.groupby("email")
    for email, g in by_email:
        did_weekend = any(w in WEEKEND_DAYS for w in g["weekday"])
        did_holiday = any(d in hol_set for d in g["date"])
        if not (did_weekend or did_holiday):
            issues.append(f"{email} has no weekend or holiday duty in this window")
    return issues

# -----------------------------
# Main CTA
# -----------------------------
if st.button("🧮 Run Scheduler"):
    schedule_df = build_schedule()
    st.session_state["schedule_df"] = schedule_df

schedule_df = st.session_state.get("schedule_df", pd.DataFrame())

if not schedule_df.empty:
    problems = validate_schedule(schedule_df, min_rest_days=3, max_per_day=1)
    problems += weekend_issues(schedule_df)
    problems += must_have_weekend_or_holiday(schedule_df, holidays_df)
    if problems:
        st.error("Validation issues found:")
        for p in problems:
            st.write("• " + p)
    else:
        st.success("Validation passed: exactly one duty per day, 3-day rest (unless relaxed to avoid gaps), weekend respected, and everyone has weekend/holiday if feasible.")

    # 🎯 Group quota check (target vs actual)
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

    # 📅 Calendar
    st.subheader("📅 Calendar view")
    pivot = schedule_df.copy()
    pivot["assignee"] = pivot.apply(
        lambda r: f"{r['first_name']} {r['last_name']}".strip() if pd.notna(r.get("email")) else "—",
        axis=1,
    )
    calendar = pivot.pivot_table(index="date", columns="slot", values="assignee",
                                 aggfunc=lambda x: " / ".join([str(i) for i in x]))
    calendar.columns = [f"Slot {i}" for i in calendar.columns]
    st.dataframe(calendar, use_container_width=True)

    # 📋 Flat table
    st.subheader("📋 Flat table")
    st.dataframe(schedule_df, use_container_width=True)

    # 📈 Load per intern
    st.subheader("📈 Load per intern")
    counts = (
        schedule_df.dropna(subset=["email"])
        .groupby(["email","first_name","last_name"])
        .size()
        .reset_index(name="shifts")
    )
    st.dataframe(counts.sort_values("shifts", ascending=False), use_container_width=True)

    # ⬇️ Export
    csv_buf = io.StringIO()
    schedule_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download CSV", data=csv_buf.getvalue(),
                       file_name=f"medshift_schedule_{start_date}_{end_date}.csv", mime="text/csv")
