import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import io
import calendar as pycal

st.set_page_config(page_title="MedShift — MVP", page_icon="🩺", layout="wide")
st.title("🩺 MedShift — Duty Scheduling (MVP)")
st.caption("Algorithm-first MVP: monthly window, one duty/day, 3-day rest, weekend/holiday fairness.")

# -----------------------------
# Sidebar — Inputs
# -----------------------------
st.sidebar.header("⚙️ Setup")

# Monthly window
with st.sidebar.expander("📅 Calendar Month", expanded=True):
    today = datetime.today()
    year = st.number_input("Year", min_value=2023, max_value=2100, value=today.year, step=1)
    month = st.selectbox("Month", options=list(range(1, 13)), index=today.month - 1, format_func=lambda m: f"{m:02d}")
    start_date = datetime(int(year), int(month), 1).date()
    last_day = pycal.monthrange(int(year), int(month))[1]
    end_date = datetime(int(year), int(month), last_day).date()
    st.write(f"Scheduling window: **{start_date} → {end_date}**")

with st.sidebar.expander("🧑‍⚕️ Interns (CSV)", expanded=True):
    st.markdown("Columns: **first_name,last_name,email,group,max_shifts**")
    sample_interns = """first_name,last_name,email,group,max_shifts
Adi,Hadash,adi.hadash@example.com,חדשים,10
Dana,Achrei,dana.achrei@example.com,אחרי שלב א,10
Omri,Achrei,omri.achrei@example.com,אחרי שלב א,10
Lior,Lifnei,lior.lifnei@example.com,לפני שלב א,10
"""
    f = st.file_uploader("Interns CSV", type=["csv"], key="interns")
    interns_df = pd.read_csv(f) if f else pd.read_csv(io.StringIO(sample_interns))
    st.dataframe(interns_df, use_container_width=True)

with st.sidebar.expander("🚫 Unavailability (optional)", expanded=False):
    st.markdown("Columns: **email,date** (YYYY-MM-DD)")
    f = st.file_uploader("Unavailability CSV", type=["csv"], key="unavail")
    unavail_df = pd.read_csv(f, parse_dates=["date"]) if f else pd.DataFrame(columns=["email", "date"])
    if not unavail_df.empty:
        st.dataframe(unavail_df, use_container_width=True)

with st.sidebar.expander("🏷️ Group minimums (CSV)", expanded=True):
    st.markdown("Columns (choose one of the first two): **group,min_per_intern**  _or_ **group,target_shifts**  (treated as min_per_intern for backward-compat). Optional: **priority** (higher = gets overflow first).")
    sample_quotas = """group,min_per_intern,priority
חדשים,4,3
לפני שלב א,3,2
אחרי שלב א,2,1
"""
    f = st.file_uploader("Group minimums CSV", type=["csv"], key="group_quotas")
    group_quota_df = pd.read_csv(f) if f else pd.read_csv(io.StringIO(sample_quotas))
    st.dataframe(group_quota_df, use_container_width=True)

with st.sidebar.expander("📅 Holidays (optional)", expanded=False):
    st.markdown("Columns: **date** (YYYY-MM-DD)")
    f = st.file_uploader("Holidays CSV", type=["csv"], key="holidays")
    holidays_df = pd.read_csv(f, parse_dates=["date"]) if f else pd.DataFrame(columns=["date"])

# Rules
with st.sidebar.expander("📜 Rules", expanded=True):
    st.caption("One 26h duty/day. Rest: duty on D ⇒ no duties on D+1,D+2 (earliest D+3).")
    max_shifts_per_day = 1
    min_days_between_shifts = 3
    lock_weekends = st.checkbox("Avoid same intern twice on same ISO-weekend", value=True)

# -----------------------------
# Precompute / clean
# -----------------------------
WEEKEND_DAYS = {4, 5}  # Fri, Sat (Mon=0..Sun=6)
calendar_days = [start_date + timedelta(n) for n in range((end_date - start_date).days + 1)]

if "max_shifts" not in interns_df.columns:
    interns_df["max_shifts"] = 9999
interns_df["email"] = interns_df["email"].astype(str).str.strip().str.lower()

unavail = set()
if not unavail_df.empty:
    unavail_df["email"] = unavail_df["email"].astype(str).str.strip().str.lower()
    unavail = {(r.email, pd.to_datetime(r.date).date()) for _, r in unavail_df.iterrows()}

# group → min per intern, and priority for overflow
group_min = {}
group_priority = {}
for _, r in group_quota_df.iterrows():
    g = str(r.get("group")).strip()
    if not g or g == "nan":
        continue
    if "min_per_intern" in r and pd.notna(r["min_per_intern"]):
        group_min[g] = int(r["min_per_intern"])
    elif "target_shifts" in r and pd.notna(r["target_shifts"]):
        group_min[g] = int(r["target_shifts"])  # backward-compat: interpret as min per intern
    if "priority" in r and pd.notna(r["priority"]):
        group_priority[g] = float(r["priority"])
# default priority = min value (lowest) if not provided
if group_min and not group_priority:
    # rank by min_per_intern (higher min => higher priority)
    sorted_g = sorted(group_min.keys(), key=lambda x: group_min[x], reverse=True)
    group_priority = {g: float(len(sorted_g) - i) for i, g in enumerate(sorted_g)}

# -----------------------------
# Scheduler
# -----------------------------
def build_schedule():
    interns = interns_df.copy()
    interns["assigned"] = 0
    interns["last_day"] = pd.NaT

    # helpers
    def igroup(email):
        try:
            row = interns.loc[interns.email == email].iloc[0]
            return row.get("group") if pd.notna(row.get("group")) else None
        except Exception:
            return None

    # per-intern minimum need: need[email] = (min_per_intern(group) - assigned[email]) if positive
    def personal_deficit(email):
        g = igroup(email)
        need = max(0, group_min.get(g, 0) - int(interns.loc[interns.email == email, "assigned"].iloc[0]))
        return need

    assignments = []
    last_day_map = {row.email: None for _, row in interns.iterrows()}
    weekend_last = defaultdict(lambda: None)  # (year, week) -> email

    # --- PRE-PASS: one weekend/holiday per intern if possible, prefer those with highest personal deficit
    hol_set = set(pd.to_datetime(holidays_df["date"]).dt.date) if not holidays_df.empty else set()
    special_days = [d for d in calendar_days if (d.weekday() in WEEKEND_DAYS) or (d in hol_set)]

    def has_special(email):
        return any(a["email"] == email and ((a["date"].weekday() in WEEKEND_DAYS) or (a["date"] in hol_set)) for a in assignments)

    for day in special_days:
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
            need = personal_deficit(email)
            g = igroup(email)
            prio = group_priority.get(g, 0.0)
            rest_days = 999 if last is None else (day - last).days
            # prefer: higher personal deficit → higher group priority → less assigned → more rest
            cand.append((-need, -prio, r.assigned, -rest_days, str(r.get("last_name", r.get("first_name",""))), email))
        if cand:
            cand.sort()
            chosen = cand[0][5]
            assignments.append({"date": day, "slot": 1, "email": chosen})
            last_day_map[chosen] = day
            interns.loc[interns.email == chosen, "assigned"] += 1
            if day.weekday() in WEEKEND_DAYS:
                weekend_last[(year, week_idx)] = chosen

    # ---- MAIN LOOP: always fill the day (no empty days)
    for day in calendar_days:
        if any(a["date"] == day and a["email"] is not None for a in assignments):
            continue
        year, week_idx, _ = day.isocalendar()

        def collect(relax_weekend=False, relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=False):
            c = []
            for _, r in interns.iterrows():
                email = r.email
                if not ignore_unavail and (email, day) in unavail:
                    continue
                if not ignore_caps and r.assigned >= r.max_shifts:
                    continue
                last = last_day_map[email]
                if not relax_rest and last is not None and (day - last).days < min_days_between_shifts:
                    continue
                if not relax_weekend and lock_weekends and day.weekday() in WEEKEND_DAYS and weekend_last[(year, week_idx)] == email:
                    continue

                need = personal_deficit(email)  # per-intern deficit to minimum
                g = igroup(email)
                prio = group_priority.get(g, 0.0)
                # HARD MINIMUMS: if we have people still below minimum, ignore those already meeting theirs
                if not overflow and need == 0:
                    continue

                rest_days = 999 if last is None else (day - last).days
                name_for_sort = r.get("last_name", r.get("first_name", ""))
                if not overflow:
                    # stage: fill personal minimums first. Score by: larger need, then higher group priority
                    score = need + 0.001 * prio
                else:
                    # overflow stage: ignore need; allocate by group priority
                    score = 100.0 + prio
                c.append((-score, r.assigned, -rest_days, str(name_for_sort), email))
            return c

        stages = [
            # meet personal minimums strictly
            dict(relax_weekend=False, relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=False),
            # overflow strictly
            dict(relax_weekend=False, relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=True),
            # relax weekend repetition
            dict(relax_weekend=True,  relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=False),
            dict(relax_weekend=True,  relax_rest=False, ignore_unavail=False, ignore_caps=False, overflow=True),
            # relax rest
            dict(relax_weekend=True,  relax_rest=True,  ignore_unavail=False, ignore_caps=False, overflow=False),
            dict(relax_weekend=True,  relax_rest=True,  ignore_unavail=False, ignore_caps=False, overflow=True),
            # ignore caps
            dict(relax_weekend=True,  relax_rest=True,  ignore_unavail=False, ignore_caps=True,  overflow=True),
            # last resort: ignore unavailability too
            dict(relax_weekend=True,  relax_rest=True,  ignore_unavail=True,  ignore_caps=True,  overflow=True),
        ]

        chosen = None
        for params in stages:
            cands = collect(**params)
            if cands:
                cands.sort()
                chosen = cands[0][4]
                break
        if chosen is None:
            # absolute fallback: pick least assigned globally
            chosen = interns.sort_values(["assigned"]).iloc[0].email

        assignments.append({"date": day, "slot": 1, "email": chosen})
        last_day_map[chosen] = day
        interns.loc[interns.email == chosen, "assigned"] += 1
        if day.weekday() in WEEKEND_DAYS:
            weekend_last[(year, week_idx)] = chosen

    out = pd.DataFrame(assignments)
    # join info for display
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
    # exactly one per day
    counts = df.groupby("date")["email"].apply(lambda s: s.notna().sum())
    for d, c in counts.items():
        if c != max_per_day:
            issues.append(f"Day {d} must have exactly {max_per_day} duty; got {c}.")
    # rest rule
    df2 = df.dropna(subset=["email"]).copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.date
    by_email = df2.groupby("email")["date"].apply(lambda s: sorted(list(s)))
    for email, dates in by_email.items():
        for i in range(len(dates)-1):
            gap = (dates[i+1] - dates[i]).days
            if gap < min_rest_days:
                issues.append(f"Rest rule violated for {email}: {dates[i]} -> {dates[i+1]} (gap {gap}d)")
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
    wk = df2[df2["weekday"].isin(WEEKEND_DAYS)]
    for (email, yw), g in wk.groupby(["email","iso_year_week"]):
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
        st.success("Validation passed (exactly one duty/day; 3-day rest unless relaxed to avoid gaps).")

    # 📊 Summary: per-group totals vs minimum-per-intern requirement
    if group_min:
        st.subheader("🎯 Group check")
        per_group = schedule_df.dropna(subset=["email"]).groupby("group").size().rename("actual").to_frame()
        per_group["min_per_intern"] = per_group.index.map(lambda g: group_min.get(g, 0))
        st.dataframe(per_group, use_container_width=True)

    # 📅 Calendar
    st.subheader("📅 Calendar view")
    pivot = schedule_df.copy()
    pivot["assignee"] = pivot.apply(
        lambda r: f"{r['first_name']} {r['last_name']}".strip() if pd.notna(r.get("email")) else "—",
        axis=1,
    )
    cal = pivot.pivot_table(index="date", columns="slot", values="assignee",
                            aggfunc=lambda x: " / ".join([str(i) for i in x]))
    cal.columns = [f"Slot {i}" for i in cal.columns]
    st.dataframe(cal, use_container_width=True)

    # Flat table
    st.subheader("📋 Flat table")
    st.dataframe(schedule_df, use_container_width=True)

    # Load per intern
    st.subheader("📈 Load per intern")
    counts = (schedule_df.dropna(subset=["email"])
              .groupby(["email","first_name","last_name","group"])
              .size().reset_index(name="shifts"))
    st.dataframe(counts.sort_values(["group","shifts","last_name"], ascending=[True, False, True]),
                 use_container_width=True)

    # Export
    csv_buf = io.StringIO()
    schedule_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download CSV", data=csv_buf.getvalue(),
                       file_name=f"medshift_{year}-{month:02d}.csv", mime="text/csv")
