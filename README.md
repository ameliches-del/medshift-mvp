# MedShift — Duty Scheduling (Streamlit MVP)

This repo helps you deploy the MVP **without paying**. It pairs with the `app.py` that's in your ChatGPT Canvas.

## Files in this folder
- `requirements.txt` — Python deps for Streamlit Cloud
- `interns_template.csv` — sample interns file you can edit
- `unavailability_template.csv` — sample optional unavailability file
- `app.py` — **you will create this** by copying the code from the ChatGPT Canvas named “MedShift — Minimal Scheduling App (Streamlit MVP)”.

---

## Step 1 — Create a new GitHub repo (via web UI)
1. Go to GitHub → **New repository**.
2. Name it, e.g. `medshift-mvp`.
3. Create the repo (public is fine for now).

## Step 2 — Add the files
1. In your new repo, click **Add file → Upload files**.
2. Upload:
   - `requirements.txt`
   - `interns_template.csv` (optional, for quick testing)
   - `unavailability_template.csv` (optional)
3. Create a new file called **`app.py`** and **paste the code** from the ChatGPT Canvas (“MedShift — Minimal Scheduling App (Streamlit MVP)”).

> Tip: You can also upload `app.py` as a file if you saved it locally.

## Step 3 — (Optional) Run locally
If you have Python installed:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL that Streamlit prints.

## Step 4 — Deploy (Free) on Streamlit Community Cloud
1. Go to `https://share.streamlit.io` and sign in with GitHub.
2. Click **New app**, choose your `medshift-mvp` repo, branch (main), and file path `app.py`.
3. Click **Deploy** — your app will build and go live at a public URL.

## Step 5 — Use the app
1. In the sidebar:
   - Pick a **Start date** and **Number of weeks**.
   - Upload **Interns CSV** (use the template CSV here or your own).
   - Optionally upload **Unavailability CSV** (email + date).
   - Set rules: coverage per day, rest days, group balancing, weekend rule.
2. Click **Run Scheduler**.
3. Review: Calendar view, flat table, per-intern load.
4. Click **Download CSV** to export the schedule.

## CSV formats
### `interns_template.csv`
Columns:
- `first_name`, `last_name` — strings
- `email` — unique identifier per intern
- `group` — seniority or team bucket (e.g. A/B)
- `max_shifts` — maximum number of shifts during the selected period

### `unavailability_template.csv` (optional)
Columns:
- `email` — must match an intern in the interns CSV
- `date` — `YYYY-MM-DD`

## Troubleshooting
- **No assignments on some days**: maybe constraints too tight or no available interns (check unavailability and rest days).
- **Uneven load**: reduce `min rest days`, increase coverage, or adjust `max_shifts`.
- **Weekend repeats**: try enabling the weekend lock rule.

When you're happy with the algorithm, we can switch data to **Supabase** and build admin screens in **Lovable**.
