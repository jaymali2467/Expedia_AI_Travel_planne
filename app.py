"""
Expedia AI Travel Companion — Hackathon UI (Roam design system)
================================================================
Single-file Streamlit app. Fixes vs. previous version:

1. FLIGHT CARDS RENDERED AS BLACK CODE BLOCKS
   Root cause: the card HTML f-strings were indented inside the for-loop.
   Markdown treats lines starting with 4+ spaces as <pre> code blocks, so
   st.markdown printed the HTML *source* in a dark box instead of rendering it.
   Fix: html() helper collapses all indentation/newlines before st.markdown.

2. INVISIBLE PLOTLY CHART
   Root cause: st.plotly_chart(theme="streamlit") inherits the app's DARK theme
   (white text) while CSS forces a light background -> white-on-white.
   Fix: theme=None + template="plotly_white" + explicit font/grid colors.

3. DARK SELECTBOXES / THEME FIGHTING
   Root cause: BaseWeb dropdown menus render in a portal OUTSIDE .stApp, so
   .stApp-scoped CSS never reaches them; CSS alone can't beat the theme engine.
   Fix: app bootstraps .streamlit/config.toml with base="light" (the canonical
   fix) AND ships defense-in-depth CSS covering portals + color-scheme.

4. SECURITY: API key is no longer hardcoded. Reads GROQ_API_KEY from env or
   st.secrets; falls back to heuristic weight inference so the demo never dies.

Run:  streamlit run app.py
Note: on the VERY first launch the config.toml is written fresh — if the page
      still opens dark, refresh once (or restart) and it will lock to light.
"""

import calendar
import itertools
import json
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0. THEME BOOTSTRAP — must happen before Streamlit renders anything.
#    config.toml is the only 100%-reliable way to force light mode; CSS below
#    is defense-in-depth for the first-ever launch.
# ─────────────────────────────────────────────────────────────────────────────
_CONFIG = """[theme]
base = "light"
primaryColor = "#006CE4"
backgroundColor = "#f7f7f9"
secondaryBackgroundColor = "#ffffff"
textColor = "#1f1f1f"
font = "sans serif"

[client]
toolbarMode = "minimal"
"""
try:
    os.makedirs(".streamlit", exist_ok=True)
    _cfg_path = os.path.join(".streamlit", "config.toml")
    if not os.path.exists(_cfg_path) or open(_cfg_path).read() != _CONFIG:
        with open(_cfg_path, "w") as f:
            f.write(_CONFIG)
except Exception:
    pass  # read-only FS etc. — CSS fallback still applies

st.set_page_config(
    page_title="Expedia AI Travel Companion",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM CLIENT — key from env / secrets only. Never hardcode.
# ─────────────────────────────────────────────────────────────────────────────
def get_groq_client():
    """Returns (client_or_None, status_message)."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception as e:
            return None, ("secrets.toml exists but could not be read — check TOML "
                          "syntax (the key value must be in \"quotes\")"
                          if os.path.exists(".streamlit/secrets.toml")
                          else "no key found in GROQ_API_KEY env var or .streamlit/secrets.toml")
    if not key:
        return None, "no key found in GROQ_API_KEY env var or .streamlit/secrets.toml"
    try:
        from groq import Groq
        return Groq(api_key=key), "key loaded"
    except ImportError:
        return None, "groq package not installed — run: pip install groq"
    except Exception as e:
        return None, f"client init failed: {str(e)[:120]}"

client, LLM_STATUS = get_groq_client()


def llm_health():
    """One live ping per session so the sidebar can show WHY AI is on/off."""
    if client is None:
        return False, LLM_STATUS
    if "llm_health" not in st.session_state:
        try:
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "ping"}], max_tokens=4)
            st.session_state["llm_health"] = (True, "llama-3.3-70b via Groq — connected")
        except Exception as e:
            st.session_state["llm_health"] = (False, f"API call failed: {str(e)[:150]}")
    return st.session_state["llm_health"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. GLOBAL CSS — Expedia "Roam" tokens. Covers .stApp AND BaseWeb portals.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

    :root { color-scheme: light only; }

    /* ── Canvas ─────────────────────────────────────────────── */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #f7f7f9 !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }

    /* ── Typography ─────────────────────────────────────────── */
    html, body, .stApp, .stApp p, .stApp span, .stApp div,
    .stApp label, .stApp li {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        color: #1f1f1f;
    }
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        color: #1f1f1f !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    /* Streamlit draws UI icons (sidebar collapse arrows etc.) as Material
       Symbols LIGATURES — text like "keyboard_double_arrow_right" that the
       icon font renders as an arrow. The global font override above hijacks
       those spans, showing the raw text. Streamlit marks every ligature icon
       span with translate="no" across versions, so exempt those plus all
       known icon containers. The @import at the top guarantees the font is
       actually loaded. */
    span[translate="no"],
    [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stExpanderToggleIcon"],
    span[class*="material-symbols"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: 400 !important;
        font-feature-settings: 'liga';
        -webkit-font-feature-settings: 'liga';
    }

    /* ── Sidebar ────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e3e3e5 !important;
    }
    [data-testid="stSidebar"] * { color: #1f1f1f; }

    /* ── Buttons ────────────────────────────────────────────── */
    .stButton > button {
        background-color: #006CE4 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 24px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 0.65rem 1.25rem !important;
        transition: background-color .15s ease !important;
        box-shadow: none !important;
    }
    .stButton > button:hover  { background-color: #0056b8 !important; }
    .stButton > button:active { background-color: #004a9e !important; }
    .stButton > button p, .stButton > button span { color: #ffffff !important; }

    /* ── Selectboxes (control) ──────────────────────────────── */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #c7c7cc !important;
        border-radius: 10px !important;
        min-height: 46px;
    }
    .stSelectbox [data-baseweb="select"] * { color: #1f1f1f !important; }
    .stSelectbox [data-baseweb="select"] svg { fill: #616161 !important; }
    .stSelectbox label { color: #616161 !important; font-weight: 600; font-size: 13px !important; }

    /* ── Selectbox dropdown MENU — renders in a portal OUTSIDE .stApp ── */
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul {
        background-color: #ffffff !important;
        border: 1px solid #e3e3e5 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li * { color: #1f1f1f !important; }
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"] {
        background-color: #e8f0fe !important;
    }

    /* ── Alerts / spinner text ──────────────────────────────── */
    [data-testid="stAlert"] * , .stSpinner * { color: #1f1f1f !important; }

    /* ── Reusable components ────────────────────────────────── */
    .xp-header {
        display: flex; align-items: center; gap: 14px; margin-bottom: 4px;
    }
    .xp-logo {
        font-size: 26px; font-weight: 800; color: #003B95; letter-spacing: -0.5px;
    }
    .xp-logo .xp-dot { color: #EF3346; }
    .xp-subtitle { font-size: 15px; color: #616161; margin-top: 2px; }

    .xp-panel {
        background: #ffffff; border: 1px solid #e3e3e5; border-radius: 14px;
        padding: 20px 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }

    .ai-insight-box {
        background: linear-gradient(135deg, #e8f0fe 0%, #ffffff 65%);
        border: 1px solid #d5e3fb;
        border-left: 4px solid #006CE4;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0,60,150,0.06);
        margin: 4px 0 24px 0;
        line-height: 1.55; font-size: 14.5px; color: #1f1f1f;
    }
    .ai-insight-box .ai-title {
        font-weight: 700; color: #003B95; font-size: 15px;
        display: block; margin-bottom: 6px;
    }

    /* ── Flight card ────────────────────────────────────────── */
    .flight-card {
        background: #ffffff;
        border: 1px solid #e3e3e5;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        display: flex; align-items: stretch; gap: 20px;
        transition: box-shadow .18s ease, border-color .18s ease;
    }
    .flight-card:hover { box-shadow: 0 6px 16px rgba(0,0,0,0.08); border-color: #006CE4; }
    .flight-card.top { border: 2px solid #006CE4; box-shadow: 0 4px 14px rgba(0,108,228,0.12); }

    .fc-main { flex: 1 1 auto; display: flex; flex-direction: column; gap: 10px; min-width: 0; }

    .fc-badges { display: flex; gap: 8px; flex-wrap: wrap; }
    .badge { font-size: 11.5px; font-weight: 700; padding: 3px 10px; border-radius: 999px; display: inline-block; }
    .badge-best   { background: #007a3d; color: #ffffff !important; }
    .badge-pref   { background: #fdf3d8; color: #7a5c00 !important; border: 1px solid #f0dfa6; }
    .badge-scarce { background: #fdecec; color: #b3261e !important; border: 1px solid #f6cfcf; }
    .badge-warn   { background: #fff4e5; color: #9a5b00 !important; border: 1px solid #f5dfc0; }
    .badge-self   { background: #eef2f7; color: #3b4a5a !important; border: 1px solid #d8e0ea; }
    .badge-hol    { background: #efe9fb; color: #5b21b6 !important; border: 1px solid #ddd0f5; }

    .fc-airline { font-size: 14px; font-weight: 600; color: #1f1f1f; display: flex; align-items: center; gap: 8px; }
    .fc-cabin { font-size: 12px; color: #616161; font-weight: 500; background: #f1f1f4; padding: 2px 8px; border-radius: 6px; }

    .fc-times { display: flex; align-items: center; gap: 18px; }
    .fc-time-block { display: flex; flex-direction: column; min-width: 64px; }
    .fc-time { font-size: 21px; font-weight: 700; color: #1f1f1f; line-height: 1.15; }
    .fc-date { font-size: 11px; color: #9a9a9e; }
    .fc-code { font-size: 12px; color: #616161; font-weight: 600; }

    .fc-duration { flex: 1 1 auto; max-width: 220px; display: flex; flex-direction: column; align-items: center; }
    .fc-dur-text { font-size: 12px; color: #616161; margin-bottom: 5px; font-weight: 600; }
    .fc-line { width: 100%; height: 2px; background: #c7c7cc; position: relative; }
    .fc-line::before, .fc-line::after {
        content: ''; position: absolute; top: -3px; width: 8px; height: 8px;
        border-radius: 50%; background: #c7c7cc;
    }
    .fc-line::before { left: 0; } .fc-line::after { right: 0; }
    .fc-line .fc-stopdot {
        position: absolute; left: 50%; top: -4px; width: 10px; height: 10px;
        border-radius: 50%; background: #ffffff; border: 2px solid #616161; transform: translateX(-50%);
    }
    .fc-stops { font-size: 12px; margin-top: 5px; font-weight: 600; }
    .fc-stops.direct { color: #007a3d; }
    .fc-stops.onestop { color: #b45309; }

    .fc-meta { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: #616161; }
    .fc-meta b { color: #1f1f1f; }

    .fc-price {
        flex: 0 0 150px; border-left: 1px solid #e3e3e5;
        display: flex; flex-direction: column; justify-content: center; align-items: flex-end;
        padding-left: 22px;
    }
    .fc-price .amount { font-size: 28px; font-weight: 800; color: #1f1f1f; line-height: 1.1; }
    .fc-price .sub { font-size: 12px; color: #616161; }
    .fc-price .score { font-size: 11px; color: #9a9a9e; margin-top: 6px; }

    /* ── Sidebar profile components ─────────────────────────── */
    .profile-row { display: flex; justify-content: space-between; align-items: baseline;
                   padding: 7px 0; border-bottom: 1px dashed #ececef; font-size: 13.5px; }
    .profile-row .k { color: #616161; }
    .profile-row .v { color: #1f1f1f; font-weight: 700; text-align: right; }

    .history-quote {
        background: #f7f7f9; border-left: 3px solid #c7c7cc; border-radius: 0 8px 8px 0;
        padding: 10px 12px; font-size: 12.5px; font-style: italic; color: #4a4a4e;
        line-height: 1.5; margin-top: 6px;
    }

    .weight-row { margin: 10px 0 2px 0; }
    .weight-label { display: flex; justify-content: space-between; font-size: 12.5px;
                    font-weight: 600; color: #1f1f1f; margin-bottom: 4px; }
    .weight-label .wv { color: #006CE4; }
    .weight-track { width: 100%; height: 8px; background: #e9e9ec; border-radius: 999px; overflow: hidden; }
    .weight-fill { height: 100%; background: linear-gradient(90deg, #006CE4, #003B95); border-radius: 999px; }

    /* ── Trade-off strip ────────────────────────────────────── */
    /* ── Journey mode (round trip / multi-city) ─────────────── */
    .journey-head {
        background: #003B95; color: #ffffff !important; border-radius: 14px;
        padding: 16px 22px; margin-bottom: 14px;
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 10px;
    }
    .journey-head * { color: #ffffff !important; }
    .jh-route { font-size: 19px; font-weight: 800; letter-spacing: .01em; }
    .jh-meta { font-size: 13px; opacity: .92; }
    .jh-price { font-size: 24px; font-weight: 800; text-align: right; }
    .jh-price .sub { display: block; font-size: 11.5px; font-weight: 500; opacity: .85; }

    .leg-badge { background: #e8f0fe; color: #003B95 !important;
                 border: 1px solid #c9dcfa; }

    .stay-divider {
        display: flex; align-items: center; gap: 12px;
        margin: 2px 6px 14px 6px; color: #616161; font-size: 12.5px; font-weight: 600;
    }
    .stay-divider::before, .stay-divider::after {
        content: ''; flex: 1; border-top: 1px dashed #c7c7cc;
    }

    .tradeoff-strip {
        display: flex; gap: 12px; margin: 0 0 18px 0; flex-wrap: wrap;
    }
    .to-cell {
        flex: 1 1 200px; background: #ffffff; border: 1px solid #e3e3e5;
        border-radius: 12px; padding: 12px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .to-cell.pick { border: 2px solid #006CE4; background: #f4f8fe; }
    .to-kicker { font-size: 11px; font-weight: 700; letter-spacing: .04em;
                 text-transform: uppercase; color: #616161; }
    .to-cell.pick .to-kicker { color: #006CE4; }
    .to-main { font-size: 19px; font-weight: 800; color: #1f1f1f; margin: 2px 0; }
    .to-sub { font-size: 12px; color: #616161; }

    .empty-hint {
        text-align: center; padding: 56px 20px; color: #616161; font-size: 15px;
        background: #ffffff; border: 1px dashed #d5d5d9; border-radius: 14px;
    }
    .empty-hint .big { font-size: 34px; display: block; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render raw HTML safely through st.markdown.
    CRITICAL: collapses all newlines + leading indentation so Markdown can
    never interpret indented lines as a <pre> code block (the black-box bug)."""
    st.markdown(re.sub(r"\n\s*", "", markup), unsafe_allow_html=True)


def safe_date(s):
    """Parse a date string defensively. LLMs sometimes emit impossible calendar
    dates (e.g. 2025-02-29 in a non-leap year); clamp the day to the month's
    real end instead of crashing. Returns datetime.date or None."""
    if not s or str(s).lower() in ("none", "null", "nan"):
        return None
    try:
        return pd.Timestamp(str(s)).date()
    except Exception:
        m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(s))
        if not m:
            return None
        y, mo, d = map(int, m.groups())
        mo = min(max(mo, 1), 12)
        last = calendar.monthrange(y, mo)[1]
        return pd.Timestamp(year=y, month=mo, day=min(max(d, 1), last)).date()


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Loads ONLY the official case dataset from the data/ folder."""
    paths = {"flights": os.path.join("data", "flights_data.csv"),
             "users": os.path.join("data", "user_data.csv")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        st.error(f"Required dataset file(s) not found: {', '.join(missing)}. "
                 "Place them in the data/ folder next to app.py and reload.")
        st.stop()
    users_df = pd.read_csv(paths["users"])
    flights_df = pd.read_csv(paths["flights"])
    return users_df, flights_df



# ─────────────────────────────────────────────────────────────────────────────
# 4. AI PREFERENCE INFERENCE  (structured priors -> LLM refinement -> fallback)
# ─────────────────────────────────────────────────────────────────────────────
def heuristic_weights(user_row) -> dict:
    """Priors from STRUCTURED fields so the app is meaningful even without an LLM."""
    ps = str(user_row.get("price_sensitivity", "medium")).lower()
    dp = str(user_row.get("direct_preference", "moderate")).lower()
    budget = {"none": 0.1, "low": 0.3, "medium": 0.6, "high": 1.0}.get(ps, 0.5)
    time_w = {"none": 0.2, "moderate": 0.5, "strong": 0.85}.get(dp, 0.5)
    comfort = round(min(1.0, 1.1 - budget), 2)
    return {"budget_weight": budget, "time_weight": time_w, "comfort_weight": comfort}


@st.cache_data(show_spinner=False)
def infer_preferences(user_id: str, raw_history: str, price_sens: str,
                      direct_pref: str, llm_on: bool = True) -> dict:
    priors = heuristic_weights({"price_sensitivity": price_sens, "direct_preference": direct_pref})
    if client is None:
        return priors
    prompt = f"""You are an AI travel profiler for Expedia. Combine the structured
signals AND the raw history below into a JSON object with exactly three keys:
'budget_weight', 'time_weight', 'comfort_weight' (floats 0.0-1.0).

Structured: price_sensitivity={price_sens}, direct_preference={direct_pref}
Prior estimate from structure alone: {json.dumps(priors)}
Raw history: "{raw_history}"

Adjust the priors using the raw history (e.g. 'broke student, cheapest only'
=> budget_weight 1.0 and time_weight near 0.1). Return ONLY valid JSON."""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, response_format={"type": "json_object"},
        )
        out = json.loads(resp.choices[0].message.content)
        return {k: float(np.clip(float(out.get(k, priors[k])), 0.0, 1.0))
                for k in ("budget_weight", "time_weight", "comfort_weight")}
    except Exception:
        return priors


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROUTING + RANKING  (unchanged core: time-dependent graph, robust min-max)
# ─────────────────────────────────────────────────────────────────────────────
REGION_HINTS = {  # last-resort keyless mapping; filtered against the dataset
    "asia": ["NRT", "HND", "SIN", "BKK", "HKG", "ICN", "KUL", "DPS", "PEK", "PVG", "DEL", "BOM"],
    "europe": ["LHR", "CDG", "FCO", "AMS", "FRA", "IST", "MAD", "BCN", "MUC", "ZRH"],
    "america": ["JFK", "LAX", "SFO", "ORD", "MIA", "YYZ", "MEX", "GRU", "GIG"],
    "africa": ["CPT", "JNB", "CAI", "NBO"],
    "australia": ["SYD", "MEL"], "oceania": ["SYD", "MEL", "AKL"],
    "japan": ["NRT", "HND"], "india": ["DEL", "BOM", "MAA"],
}


@st.cache_data(show_spinner=False)
def parse_travel_request(request: str, home_airport: str, airports_json: str,
                         today_iso: str, flex_days: int, llm_on: bool = True) -> dict:
    """Parse a free-text request (like the benchmark prompts) into
    origin / destination / trip type / date windows. Origin defaults to HOME
    ('I need to get from home to Tokyo' -> home_airport -> NRT).
    Falls back to city/region matching when no LLM is available.
    NOTE: llm_on is part of the cache key so a keyless parse can't be
    served from cache after the key starts working."""
    airports = json.loads(airports_json)  # {code: city}

    def heuristic():
        text = request.lower()
        tokens = set(re.findall(r"[a-z]+", text))
        dest, extra = None, []
        for code, city in airports.items():
            if not isinstance(city, str) or not city:
                continue
            city_l = city.lower()
            first = city_l.split()[0]
            hit = (city_l in text or code.lower() in tokens
                   or (len(first) >= 3 and first in tokens))
            if not hit or code == home_airport:
                continue
            if dest is None:
                dest = code
            else:
                extra.append(code)
        pool = []
        if dest is None:  # region-level request like "Asia trip"
            for word, codes in REGION_HINTS.items():
                if word in tokens:
                    avail = [c for c in codes if c in airports and c != home_airport]
                    if avail:
                        dest, extra, pool = avail[0], avail[1:3], avail[:6]
                        break
        round_trip = any(k in text for k in ("round trip", "return", "back ", "back,",
                                             "and back", "there and back"))
        dur = None
        wordnum = {"one": 1, "a": 1, "two": 2, "couple": 2, "three": 3,
                   "few": 3, "four": 4, "five": 5, "six": 6}
        m = re.search(r"(\d+|one|a|two|couple|three|few|four|five|six)\s*"
                      r"(?:of\s*)?(day|week|month)s?", text)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else wordnum[m.group(1)]
            dur = n * {"day": 1, "week": 7, "month": 30}[m.group(2)]
        trip_type = "round_trip" if round_trip else ("multi_city" if extra else "one_way")
        return {"origin": home_airport, "destination": dest,
                "date_from": None, "date_to": None, "additional_stops": extra,
                "trip_type": trip_type,
                "return_date_from": None, "return_date_to": None,
                "trip_duration_days": dur,
                "return_home": trip_type != "one_way",
                "order_flexible": True, "region_pool": pool,
                "interpretation": "Parsed by keyword match (AI offline)."}

    if client is None or not llm_on:
        return heuristic()

    prompt = f"""You are a flight-search query parser. Today is {today_iso}.
The traveler's HOME airport is {home_airport}. Words like "home" mean {home_airport}.
Available airports (code=city): {"; ".join(f"{k}={v}" for k, v in airports.items())}

Parse this request into STRICT JSON with keys:
- "origin": airport code (default "{home_airport}" if unstated)
- "destination": airport code of the FIRST/main destination, or null if none found.
  If the request names a REGION or COUNTRY (e.g. "Asia", "Europe", "Japan") instead
  of a specific city, choose the most popular destination airport in that region
  FROM THE LIST ABOVE as "destination", and put 1-2 other plausible airports from
  the same region into "additional_stops". Never return null when a region is named.
- "additional_stops": list of further destination codes for multi-city requests (else [])
- "date_from", "date_to": ISO dates "YYYY-MM-DD" for the departure window, or null if
  no timing is implied. Interpret relative phrases from today ({today_iso}):
  "next month" = the whole next calendar month; "summer" = Jun 1 to Aug 31;
  "around the holidays" = Dec 15 to Jan 5; a weekday like "Tuesday" = the next
  such weekday. If a single exact date, set date_from = date_to. For open-ended
  phrasing like "after May 31st" or "from June onwards", set date_from to the
  day the window opens and date_to to null. NEVER return a window entirely
  before today ({today_iso}): seasonal or relative words refer to their NEXT
  occurrence (e.g. "summer" asked in November = next year's Jun 1 - Aug 31).
- "interpretation": one short sentence restating what you understood.
- "trip_type": "one_way", "round_trip" (they mention returning/coming back), or
  "multi_city" (several destinations in one journey).
- "return_date_from", "return_date_to": ISO dates for the RETURN departure window
  when trip_type is "round_trip" (e.g. "back Thursday" = the Thursday after the
  outbound date), else null.
- "trip_duration_days": integer TOTAL length of the whole journey in days when the
  user indicates one — "about three weeks" or "I have three weeks of flexibility"
  means the trip must fit in ~21 days; "a 10 day trip" = 10. This is separate from
  the date window: "any three weeks between May and June" means date_from/date_to
  span May-June but trip_duration_days = 21. Null if no duration is implied.
- "return_home": true when the trip implies ending back where it started — tours,
  vacations, "in one journey", round trips. False only for clearly one-way moves.
- "order_flexible": false ONLY if the user fixes the visiting sequence ("first X,
  then Y"); a plain list like "London + Paris + Rome" is flexible = true.
- "region_pool": when a REGION or country was named, up to 6 airport codes from
  the list above in that region (candidate substitutes); else [].
CRITICAL: all dates must be REAL calendar dates (February has 28 days in
non-leap years; use the month's actual last day).

Only use airport codes from the list above. Request: "{request}"
Return ONLY the JSON object."""
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, response_format={"type": "json_object"})
        out = json.loads(r.choices[0].message.content)
        if out.get("origin") not in airports:
            out["origin"] = home_airport
        if out.get("destination") not in airports:
            out["destination"] = None
        out["additional_stops"] = [c for c in (out.get("additional_stops") or [])
                                   if c in airports and c != out.get("destination")]
        if out.get("trip_type") not in ("one_way", "round_trip", "multi_city"):
            out["trip_type"] = "multi_city" if out["additional_stops"] else "one_way"
        for k in ("return_date_from", "return_date_to"):
            out.setdefault(k, None)
        try:
            out["trip_duration_days"] = int(out.get("trip_duration_days")) \
                if out.get("trip_duration_days") else None
        except Exception:
            out["trip_duration_days"] = None
        out["return_home"] = bool(out.get("return_home",
                                          out.get("trip_type") != "one_way"))
        out["order_flexible"] = bool(out.get("order_flexible", True))
        out["region_pool"] = [c for c in (out.get("region_pool") or []) if c in airports]
        # Single fixed date + user's flexibility -> widen the window
        df_, dt_ = safe_date(out.get("date_from")), safe_date(out.get("date_to"))
        if df_ and df_ == dt_ and flex_days:
            out["date_from"] = (df_ - timedelta(days=flex_days)).strftime("%Y-%m-%d")
            out["date_to"] = (df_ + timedelta(days=flex_days)).strftime("%Y-%m-%d")
        return out
    except Exception:
        return heuristic()


def optimize_flights(origin, dest, user, w, flights_df, date_from=None, date_to=None,
                     enforce_max_layover=True):
    df = flights_df.copy()
    df["dep_time"] = pd.to_datetime(df["departure_utc"])
    if "arrival_utc" in df.columns:
        df["arr_time"] = pd.to_datetime(df["arrival_utc"])
    else:
        df["arr_time"] = df["dep_time"] + pd.to_timedelta(df["duration_minutes"], unit="m")

    # ── Date window filter (applies to itinerary DEPARTURE) ────────────
    # Filter direct itineraries and first legs only; connecting second
    # legs may depart after date_to and that's fine.
    def in_window(frame):
        if frame.empty or (date_from is None and date_to is None):
            return frame
        m = pd.Series(True, index=frame.index)
        lo, hi = safe_date(date_from), safe_date(date_to)
        if lo is not None:
            m &= frame["dep_time"].dt.date >= lo
        if hi is not None:
            m &= frame["dep_time"].dt.date <= hi
        return frame[m]

    direct = df[(df["origin"] == origin) & (df["destination"] == dest)].copy()
    if not direct.empty:
        direct["route_stops"] = direct.get("stops", 0)
        direct["route_layover"] = direct.get("layover_minutes", 0).fillna(0) \
            if "layover_minutes" in direct.columns else 0
        direct["via"] = direct.get("layover_airports", "")
        # Enforce the user's layover constraint on pre-built itineraries too;
        # when relaxed, keep a 24h sanity ceiling so absurd routings never rank
        _lay_cap = float(user["max_layover_minutes"]) if enforce_max_layover else 1440.0
        direct = direct[direct["route_layover"] <= _lay_cap]
        direct = in_window(direct)
        direct["is_self_transfer"] = False

    leg1 = in_window(df[(df["origin"] == origin) & (df["destination"] != dest) & (df.get("stops", 0) == 0)].copy())
    leg2 = df[(df["destination"] == dest) & (df["origin"] != origin) & (df.get("stops", 0) == 0)].copy()
    conn = pd.merge(leg1, leg2, left_on="destination", right_on="origin", suffixes=("_1", "_2"))
    if not conn.empty:
        conn["route_layover"] = (conn["dep_time_2"] - conn["arr_time_1"]).dt.total_seconds() / 60.0
        _cap = float(user["max_layover_minutes"]) if enforce_max_layover else 1440.0
        conn = conn[(conn["route_layover"] >= 45) &
                    (conn["route_layover"] <= _cap)].copy()
    if not conn.empty:
        conn["flight_id"] = conn["flight_id_1"] + " + " + conn["flight_id_2"]
        conn["airline_name"] = np.where(conn["airline_name_1"] == conn["airline_name_2"],
                                        conn["airline_name_1"],
                                        conn["airline_name_1"] + " / " + conn["airline_name_2"])
        conn["airline_code"] = conn.get("airline_code_1", "")
        conn["departure_utc"] = conn["departure_utc_1"]
        conn["dep_time"] = conn["dep_time_1"]
        conn["duration_minutes"] = (conn["duration_minutes_1"] + conn["duration_minutes_2"]
                                    + conn["route_layover"]).astype(int)
        conn["price"] = conn["price_1"] + conn["price_2"]
        conn["route_stops"] = 1
        conn["via"] = conn["destination_1"]
        conn["origin"], conn["destination"] = origin, dest
        conn["cabin_class"] = conn.get("cabin_class_1", "Economy")
        conn["seats_available"] = conn[["seats_available_1", "seats_available_2"]].min(axis=1) \
            if "seats_available_1" in conn.columns else np.nan
        conn["baggage_included"] = conn.get("baggage_included_1", np.nan)
        conn["on_time_performance"] = conn[["on_time_performance_1", "on_time_performance_2"]].mean(axis=1) \
            if "on_time_performance_1" in conn.columns else np.nan
        conn["is_holiday_season"] = conn.get("is_holiday_season_1", False)
        conn["is_self_transfer"] = True

    frames = [f for f in (direct, conn) if not f.empty]
    if not frames:
        return pd.DataFrame()
    cols = ["flight_id", "airline_name", "airline_code", "origin", "destination",
            "departure_utc", "dep_time", "duration_minutes", "price",
            "route_stops", "route_layover", "via", "cabin_class",
            "seats_available", "baggage_included", "on_time_performance",
            "is_holiday_season", "is_self_transfer"]
    allx = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)

    a, b, g = w.get("budget_weight", .5), w.get("time_weight", .5), w.get("comfort_weight", .5)

    # ── Dollar-grounded generalized cost ────────────────────────────────
    # WHY NOT MIN-MAX: normalized weights act on relative position inside
    # the candidate pool's range, so one expensive outlier stretches the
    # price range and flattens a real $50 saving to ~0.04 units — which is
    # how a broke-student profile ended up NOT getting the cheapest flight.
    # Instead we convert time into dollars via an implicit Value-of-Time
    # derived from the weights, and rank by total effective cost in $.
    vot_per_hour = float(np.clip(35.0 * b / max(a, 0.15), 0.0, 250.0))
    stop_cost = {"strong": 150.0, "moderate": 50.0, "none": 0.0}.get(
        str(user["direct_preference"]).lower(), 50.0) * (0.5 + g)

    allx["effective_cost"] = (
        allx["price"]
        + vot_per_hour * allx["duration_minutes"] / 60.0
        + stop_cost * allx["route_stops"]
    )
    allx["vot_per_hour"] = vot_per_hour
    allx["optimization_score"] = allx["effective_cost"]  # kept for chart/compat
    return allx.sort_values("effective_cost").head(10).reset_index(drop=True)


def plan_journey(seq, user, w, flights_df, first_window=(None, None),
                 return_window=(None, None), trip_duration_days=None):
    """Greedy multi-leg planner (round trips & multi-city chains).
    - Picks each leg by best per-user effective cost, departing after the
      previous arrival + stay (time-feasible chaining).
    - trip_duration_days caps the WHOLE journey: stays are allocated evenly
      within the budget and every leg must depart before the hard deadline
      (first departure + duration). "3 weeks between May and June" = any
      window, but the trip itself fits in ~21 days.
    - Legs with no options in their window auto-extend the search end in
      +14-day steps (max 3 steps, never past the duration deadline), reported
      transparently — nearest options first, never a silent multi-month jump.
    Returns (list_of_top_rows, failed_pair_or_None, notes)."""
    legs, notes = [], []
    n_stays = max(0, len(seq) - 2)
    if trip_duration_days and n_stays > 0:
        stay_days = max(1, int(trip_duration_days) // (n_stays + 1))
    else:
        stay_days = 2
    deadline = None  # tz-aware, set after the first leg is chosen
    date_from, date_to = first_window

    def _try(o, d, lo, hi):
        r = optimize_flights(o, d, user, w, flights_df, date_from=lo, date_to=hi)
        if r.empty:  # relax layover cap before touching the dates
            r = optimize_flights(o, d, user, w, flights_df, date_from=lo,
                                 date_to=hi, enforce_max_layover=False)
        return r

    for i, (o, d) in enumerate(zip(seq[:-1], seq[1:])):
        is_last = (i == len(seq) - 2)
        if is_last and (return_window[0] or return_window[1]):
            date_from, date_to = return_window
        if deadline is not None:
            dl_iso = deadline.strftime("%Y-%m-%d")
            date_to = dl_iso if date_to is None else min(str(date_to), dl_iso)

        res = _try(o, d, date_from, date_to)

        # Biweekly progressive widening: extend the window END in +14d steps
        # (nearest first). Never unbounded, never past the deadline.
        base_end = safe_date(date_to or date_from)
        step = 0
        while res.empty and base_end is not None and step < 3:
            step += 1
            new_hi = base_end + timedelta(days=14 * step)
            if deadline is not None:
                new_hi = min(new_hi, deadline.date())
            res = _try(o, d, date_from, new_hi.isoformat())
            if not res.empty:
                notes.append(f"Leg {i + 1} ({o} → {d}): no options in the requested "
                             f"window — extended the search by {14 * step} days "
                             f"(departs {res.iloc[0]['dep_time'].date()}).")
            if deadline is not None and new_hi >= deadline.date():
                break
        if res.empty:
            return legs, (o, d), notes

        top = res.iloc[0]
        legs.append(top)
        if deadline is None and trip_duration_days:
            deadline = top["dep_time"] + pd.Timedelta(days=int(trip_duration_days))
        # Next leg departs after arrival + stay, inside a 7-day search window
        arr = top["dep_time"] + pd.Timedelta(minutes=int(top["duration_minutes"]))
        nxt = (arr + pd.Timedelta(days=stay_days)).date()
        date_from = nxt.strftime("%Y-%m-%d")
        date_to = (nxt + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return legs, None, notes


def plan_multicity(origin, stops, user, w, flights_df, first_window=(None, None),
                   return_window=(None, None), trip_duration_days=None,
                   return_home=True, order_flexible=True, region_pool=None):
    """Multi-city orchestrator on top of plan_journey.
    - order_flexible: tries ALL visit orderings (exact for ≤3 stops = 6 perms)
      and keeps the cheapest COMPLETE journey by total effective cost.
    - return_home: appends the origin as the final destination (tours end home).
    - region_pool: when destinations came from a region ask ("Asia"), cities are
      fungible — if no ordering completes, substitute pool cities that are
      actually reachable from the origin, and say so.
    Returns (legs, failed_pair_or_None, notes, chosen_order)."""
    stops = [s for s in dict.fromkeys(stops) if s != origin]
    region_pool = [c for c in (region_pool or []) if c != origin]

    def orders_of(stop_list):
        if order_flexible and 1 < len(stop_list) <= 3:
            return list(itertools.permutations(stop_list))
        return [tuple(stop_list)]

    def run(stop_list):
        best = None
        orders = orders_of(stop_list)
        for order in orders:
            seq = [origin] + list(order) + ([origin] if return_home else [])
            legs, failed, notes = plan_journey(
                seq, user, w, flights_df, first_window=first_window,
                return_window=return_window, trip_duration_days=trip_duration_days)
            total = sum(float(l["effective_cost"]) for l in legs) if legs else float("inf")
            key = (failed is not None, -len(legs), total)
            if best is None or key < best["key"]:
                best = dict(key=key, legs=legs, failed=failed, notes=notes,
                            order=list(order), n_orders=len(orders))
        return best

    best = run(stops)

    # Region substitution: the asked cities were the AI's guess, not the user's
    # requirement — swap in reachable ones before giving up on a full journey.
    if best["failed"] is not None and region_pool:
        lo = first_window[0]
        _end = safe_date(first_window[1] or first_window[0])
        hi = (_end + timedelta(days=42)).isoformat() if _end else None
        reachable = []
        for c in region_pool:
            if c in reachable:
                continue
            probe = optimize_flights(origin, c, user, w, flights_df,
                                     date_from=lo, date_to=hi)
            if probe.empty:
                probe = optimize_flights(origin, c, user, w, flights_df,
                                         date_from=lo, date_to=hi,
                                         enforce_max_layover=False)
            if not probe.empty:
                reachable.append(c)
            if len(reachable) >= max(2, min(len(stops), 3)):
                break
        sub = reachable[:max(1, min(len(stops) or 2, 3))]
        if sub and set(sub) != set(stops):
            cand = run(sub)
            if cand["key"] < best["key"]:
                cand["notes"].insert(0, f"The initially suggested cities weren't "
                                        f"reachable from {origin} in this period — "
                                        f"substituted reachable ones: "
                                        f"{', '.join(sub)}.")
                best = cand

    if order_flexible and len(best["order"]) > 1 and best["failed"] is None \
            and best["n_orders"] > 1:
        best["notes"].insert(0, "Visit order optimized: "
                             + " → ".join(best["order"])
                             + f" — cheapest complete journey of "
                               f"{best['n_orders']} possible orderings.")
    return best["legs"], best["failed"], best["notes"], best["order"]


def generate_journey_explanation(user, w, legs, notes, span_days, total_price):
    fallback = ("This journey chains the best-value legs for your inferred "
                "preferences; the per-leg cards below show details and any "
                "compromises the planner had to make.")
    if client is None or not legs:
        return fallback
    vot = float(legs[0].get("vot_per_hour", 0) or 0)
    umax = float(user.get("max_layover_minutes", 1e9) or 1e9)
    leg_lines = []
    for i, l in enumerate(legs, 1):
        flags = []
        if int(l.get("route_stops") or 0) > 0 and float(l.get("route_layover") or 0) > umax:
            flags.append(f"layover {fmt_dur(l['route_layover'])} exceeds their "
                         f"{int(umax)}-min preference")
        if bool(l.get("is_holiday_season")):
            flags.append("holiday-season pricing")
        s = l.get("seats_available")
        if pd.notna(s) and float(s) <= 3:
            flags.append(f"only {int(s)} seats left")
        leg_lines.append(f"Leg {i}: {l['origin']}->{l['destination']}, "
                         f"{l['airline_name']}, ${l['price']:,.0f}, "
                         f"{fmt_dur(l['duration_minutes'])}, "
                         f"{int(l['route_stops'])} stop(s)"
                         + (f" [{'; '.join(flags)}]" if flags else ""))
    prompt = f"""You are Expedia's AI Travel Assistant summarizing a MULTI-LEG journey.
User history: "{user.get('raw_history', '')}"
Weights — budget {w.get('budget_weight')}, time {w.get('time_weight')}, comfort {w.get('comfort_weight')};
implicit Value-of-Time ${vot:.0f}/hour.
Journey: {span_days}-day trip, ${total_price:,.0f} total per traveler.
{chr(10).join(leg_lines)}
Planner notes: {' '.join(notes) if notes else 'none'}
Write 3-4 friendly sentences: why this journey fits this traveler, honestly
acknowledging any compromises or planner adjustments listed above (apologetic
but constructive), plus one practical thing to expect (seasonal pricing, seat
scarcity, or the extended dates). Rely ONLY on the facts above. No greeting.
Format prices like $2,512 and durations like 27h 57m — never raw decimals or
minute counts."""
    try:
        r = client.chat.completions.create(model="llama-3.3-70b-versatile",
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.3)
        return r.choices[0].message.content.strip()
    except Exception:
        return fallback


def generate_llm_explanation(user, w, top, alts, relax_note=None):
    fallback = ("This flight offers the best balance of price, total travel time and "
                "stops for your inferred preference weights.")
    if client is None:
        return fallback
    alt_text = ""
    if not alts.empty:
        cheap = alts.loc[alts["price"].idxmin()]
        fast = alts.loc[alts["duration_minutes"].idxmin()]

        def _delta(a):
            """Signed, factual comparison vs the pick — impossible to invert."""
            dp = float(a["price"]) - float(top["price"])
            dt = int(a["duration_minutes"]) - int(top["duration_minutes"])
            parts = []
            if abs(dp) >= 1:
                parts.append(f"${abs(dp):,.0f} {'MORE expensive' if dp > 0 else 'cheaper'}")
            if abs(dt) >= 5:
                parts.append(f"{fmt_dur(abs(dt))} {'LONGER' if dt > 0 else 'shorter'}")
            return " and ".join(parts) if parts else "essentially identical"

        if top["flight_id"] != cheap["flight_id"]:
            alt_text += (f"The lowest-priced alternative ({cheap['airline_name']}) is "
                         f"{_delta(cheap)} than the pick. ")
        if top["flight_id"] != fast["flight_id"] and cheap["flight_id"] != fast["flight_id"]:
            alt_text += (f"The quickest alternative ({fast['airline_name']}) is "
                         f"{_delta(fast)} than the pick. ")
        if not alt_text:
            alt_text = "No alternative beats the pick on price or duration. "
    prompt = f"""You are Expedia's AI Travel Assistant.
User history: "{user.get('raw_history', '')}"
Engine weights — budget: {w.get('budget_weight')}, time: {w.get('time_weight')}, comfort: {w.get('comfort_weight')}.
The engine converts weights into an implicit Value-of-Time of ${top.get('vot_per_hour', 0):.0f}/hour
and ranks by total effective cost (price + time valued in dollars + stop penalty).
Recommended: {top['airline_name']}, ${top['price']:,.0f}, {fmt_dur(top['duration_minutes'])}, {int(top['route_stops'])} stop(s), effective cost ${top.get('effective_cost', 0):,.0f}.
Alternatives: {alt_text or 'none materially better on any axis.'}
Write a concise, friendly 3-sentence explanation of WHY the engine chose this flight,
explicitly tying the trade-off to the weights and the user's history. Rely ONLY on the
alternatives context given, and repeat each comparison EXACTLY as stated above —
never flip cheaper/more-expensive or shorter/longer. No greeting. Format numbers for reading: prices with a
$ sign and thousands separators rounded to whole dollars ($2,512 — never $2512.04),
and durations in hours and minutes (27h 57m — never 1677 minutes).{'''
IMPORTANT: no option satisfied the user's maximum-layover preference, so this
recommendation involves a longer connection than they'd like. Acknowledge that
compromise apologetically but constructively in your first sentence.'''
    if relax_note in ("layover", "both") else ''}"""
    try:
        r = client.chat.completions.create(model="llama-3.3-70b-versatile",
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0.3)
        return r.choices[0].message.content.strip()
    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# 6. UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_dur(m):
    m = int(m); return f"{m // 60}h {m % 60:02d}m"


def render_flight_card(flight, user, is_top=False, leg_label=None):
    dep = pd.to_datetime(flight["departure_utc"])
    arr = dep + pd.to_timedelta(int(flight["duration_minutes"]), unit="m")
    stops = int(flight["route_stops"])

    badges = []
    if leg_label:
        badges.append(f'<span class="badge leg-badge">{leg_label}</span>')
    if is_top:
        badges.append('<span class="badge badge-best">✨ Best for You</span>')
    pref = str(user.get("preferred_airlines", "") or "")
    if pref and str(flight.get("airline_code", "")) in [p.strip() for p in pref.split(";") if p.strip()]:
        badges.append('<span class="badge badge-pref">★ Your preferred airline</span>')
    seats = flight.get("seats_available")
    if pd.notna(seats) and float(seats) <= 3:
        badges.append(f'<span class="badge badge-scarce">Only {int(seats)} seats left</span>')
    _lay = float(flight.get("route_layover") or 0)
    _umax = float(user.get("max_layover_minutes", 1e9) or 1e9)
    if stops > 0 and _lay > _umax:
        badges.append(f'<span class="badge badge-warn">⏱ Layover exceeds your '
                      f'{int(_umax)}-min preference</span>')
    if bool(flight.get("is_self_transfer")):
        badges.append('<span class="badge badge-self">🔀 Self-transfer · '
                      'separate tickets, engine-built connection</span>')
    if bool(flight.get("is_holiday_season")):
        badges.append('<span class="badge badge-hol">Holiday-season pricing</span>')
    badges_html = f'<div class="fc-badges">{"".join(badges)}</div>' if badges else ""

    if stops == 0:
        stops_html = '<span class="fc-stops direct">Direct</span>'
        stopdot = ""
    else:
        via = str(flight.get("via") or "").replace(";", ", ")
        lay = fmt_dur(flight.get("route_layover", 0))
        label = "1 stop" if stops == 1 else f"{stops} stops"
        via_part = f" · {via}" if via and via != "nan" else ""
        stops_html = f'<span class="fc-stops onestop">{label}{via_part} · {lay} layover</span>'
        stopdot = '<span class="fc-stopdot"></span>'

    meta = []
    if pd.notna(flight.get("on_time_performance")):
        meta.append(f"<span>On-time <b>{int(flight['on_time_performance'])}%</b></span>")
    bag = flight.get("baggage_included")
    if pd.notna(bag):
        meta.append(f"<span>{'🧳 Bags included' if bool(bag) else 'Carry-on only fare'}</span>")
    meta_html = f'<div class="fc-meta">{"".join(meta)}</div>' if meta else ""

    cabin = flight.get("cabin_class") or "Economy"
    date_str = dep.strftime("%a %d %b %Y")

    html(f"""
    <div class="flight-card {'top' if is_top else ''}">
      <div class="fc-main">
        {badges_html}
        <div class="fc-airline">✈️ {flight['airline_name']}
          <span class="fc-cabin">{cabin}</span>
        </div>
        <div class="fc-times">
          <div class="fc-time-block">
            <span class="fc-time">{dep.strftime('%H:%M')}</span>
            <span class="fc-code">{flight['origin']}</span>
            <span class="fc-date">{date_str}</span>
          </div>
          <div class="fc-duration">
            <span class="fc-dur-text">{fmt_dur(flight['duration_minutes'])}</span>
            <div class="fc-line">{stopdot}</div>
            {stops_html}
          </div>
          <div class="fc-time-block">
            <span class="fc-time">{arr.strftime('%H:%M')}</span>
            <span class="fc-code">{flight['destination']}</span>
            <span class="fc-date">{arr.strftime('%a %d %b %Y')}</span>
          </div>
        </div>
        {meta_html}
      </div>
      <div class="fc-price">
        <span class="amount">${flight['price']:,.0f}</span>
        <span class="sub">per traveler</span>
        <span class="score">effective cost ${flight['effective_cost']:,.0f} · time @ ${flight['vot_per_hour']:.0f}/hr</span>
      </div>
    </div>""")


def render_tradeoff_strip(results, top):
    """Benchmark requirement: surface the cost-vs-time trade-off EXPLICITLY
    (e.g. 'cheapest $X / fastest $Y saving Z hrs'), not just in LLM prose."""
    cheapest = results.loc[results["price"].idxmin()]
    fastest = results.loc[results["duration_minutes"].idxmin()]

    def delta(f):
        dp = f["price"] - top["price"]
        dt = int(f["duration_minutes"] - top["duration_minutes"])
        parts = []
        if abs(dp) >= 1:
            parts.append(f"${abs(dp):,.0f} {'cheaper' if dp < 0 else 'pricier'}")
        if abs(dt) >= 5:
            parts.append(f"{fmt_dur(abs(dt))} {'faster' if dt < 0 else 'slower'}")
        return " · ".join(parts) if parts else "same as our pick"

    pick_note = []
    if top["flight_id"] == cheapest["flight_id"]:
        pick_note.append("also the cheapest")
    if top["flight_id"] == fastest["flight_id"]:
        pick_note.append("also the fastest")
    pick_sub = ", ".join(pick_note) if pick_note else \
        f"best effective cost at ${top['effective_cost']:,.0f}"

    html(f"""
    <div class="tradeoff-strip">
      <div class="to-cell pick">
        <div class="to-kicker">✨ Our pick</div>
        <div class="to-main">${top['price']:,.0f} · {fmt_dur(top['duration_minutes'])}</div>
        <div class="to-sub">{top['airline_name']} — {pick_sub}</div>
      </div>
      <div class="to-cell">
        <div class="to-kicker">Cheapest</div>
        <div class="to-main">${cheapest['price']:,.0f} · {fmt_dur(cheapest['duration_minutes'])}</div>
        <div class="to-sub">{cheapest['airline_name']} — {delta(cheapest)}</div>
      </div>
      <div class="to-cell">
        <div class="to-kicker">Fastest</div>
        <div class="to-main">${fastest['price']:,.0f} · {fmt_dur(fastest['duration_minutes'])}</div>
        <div class="to-sub">{fastest['airline_name']} — {delta(fastest)}</div>
      </div>
    </div>""")


def render_weight_bar(label, value):
    pct = int(round(float(value) * 100))
    html(f"""
    <div class="weight-row">
      <div class="weight-label"><span>{label}</span><span class="wv">{value:.1f}</span></div>
      <div class="weight-track"><div class="weight-fill" style="width:{pct}%"></div></div>
    </div>""")


def render_tradeoff_chart(options, top_id):
    opts = options.copy()
    opts["is_top"] = opts["flight_id"] == top_id
    fig = go.Figure()
    alt = opts[~opts["is_top"]]
    top = opts[opts["is_top"]]
    hover = ("<b>%{customdata[0]}</b><br>Price $%{y:,.0f}<br>"
             "Duration %{x:.0f} min<br>Stops %{customdata[1]}<extra></extra>")
    fig.add_trace(go.Scatter(
        x=alt["duration_minutes"], y=alt["price"], mode="markers", name="Alternatives",
        marker=dict(size=11, color="#b8b8bd", line=dict(width=1, color="#9a9a9e")),
        customdata=alt[["airline_name", "route_stops"]], hovertemplate=hover))
    fig.add_trace(go.Scatter(
        x=top["duration_minutes"], y=top["price"], mode="markers", name="✨ Top pick",
        marker=dict(size=18, color="#006CE4", line=dict(width=2, color="#003B95")),
        customdata=top[["airline_name", "route_stops"]], hovertemplate=hover))
    fig.update_layout(
        template="plotly_white",           # ← explicit light template
        height=420,
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                  color="#1f1f1f", size=13),
        xaxis=dict(title="Total duration (minutes)", showgrid=True, gridcolor="#e9e9ec",
                   zeroline=False, linecolor="#c7c7cc", tickfont=dict(color="#616161")),
        yaxis=dict(title="Price (USD)", showgrid=True, gridcolor="#e9e9ec",
                   zeroline=False, linecolor="#c7c7cc", tickfont=dict(color="#616161"),
                   tickprefix="$"),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#1f1f1f")),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    # theme=None is CRITICAL: theme="streamlit" (default) would re-apply the
    # app's dark theme and turn all chart text white-on-white.
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# 7. APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    users_df, flights_df = load_data()

    airport_map = {}
    if "origin_city" in flights_df.columns:
        airport_map.update(dict(zip(flights_df["origin"], flights_df["origin_city"])))
    if "destination_city" in flights_df.columns:
        airport_map.update(dict(zip(flights_df["destination"], flights_df["destination_city"])))

    def fmt_airport(code):
        city = airport_map.get(code, "")
        return f"{city} ({code})" if isinstance(city, str) and city else str(code)

    # ── Sidebar: Travel Profile ─────────────────────────────────────────────
    with st.sidebar:
        html('<div class="xp-logo" style="font-size:22px;">expedia<span class="xp-dot">.</span></div>')
        st.markdown("### 👤 Travel Profile")
        selected = st.selectbox("Switch simulated user", users_df["user_id"].tolist())
        user = users_df[users_df["user_id"] == selected].iloc[0]

        pref_air = str(user.get("preferred_airlines", "") or "—").replace(";", ", ")
        html(f"""
        <div>
          <div class="profile-row"><span class="k">🏠 Home</span><span class="v">{fmt_airport(user['home_airport'])}</span></div>
          <div class="profile-row"><span class="k">⏱️ Max layover</span><span class="v">{int(user['max_layover_minutes'])} min</span></div>
          <div class="profile-row"><span class="k">💸 Price sensitivity</span><span class="v">{str(user['price_sensitivity']).title()}</span></div>
          <div class="profile-row"><span class="k">🎯 Direct preference</span><span class="v">{str(user['direct_preference']).title()}</span></div>
          <div class="profile-row"><span class="k">⭐ Preferred airlines</span><span class="v">{pref_air}</span></div>
        </div>""")

        st.markdown("#### 📝 Raw history log")
        html(f'<div class="history-quote">{str(user.get("raw_history", ""))}</div>')

        st.markdown("#### 🧠 AI-inferred weights")
        weights = infer_preferences(str(user["user_id"]), str(user.get("raw_history", "")),
                                    str(user.get("price_sensitivity", "medium")),
                                    str(user.get("direct_preference", "moderate")),
                                    llm_on=(client is not None))
        render_weight_bar("Budget focus", weights["budget_weight"])
        render_weight_bar("Time focus", weights["time_weight"])
        render_weight_bar("Comfort focus", weights["comfort_weight"])
        ok, msg = llm_health()
        st.caption(("🟢 AI engine: " if ok else "🔴 AI offline — heuristic mode. ") + msg)

    # ── Header ──────────────────────────────────────────────────────────────
    html("""
    <div class="xp-header">
      <span class="xp-logo">expedia<span class="xp-dot">.</span></span>
      <div>
        <h2 style="margin:0; font-size:24px;">AI Travel Companion</h2>
        <div class="xp-subtitle">Personalized flight routing, ranked by your inferred preferences</div>
      </div>
    </div>""")

    # ── Ask in plain language (consumes benchmark-style prompts) ───────────
    html('<div class="xp-panel" style="padding-bottom:6px; margin-bottom:0;">'
         '<h4 style="margin:0 0 8px 0;">💬 Ask your companion</h4></div>')
    a1, a2 = st.columns([5, 1], vertical_alignment="bottom")
    with a1:
        nl_request = st.text_input(
            "Describe your trip in your own words",
            placeholder='e.g. "I need to get from home to Tokyo next month, what do you suggest?"')
    with a2:
        ask = st.button("Ask AI", use_container_width=True)

    # Dataset date coverage — constrains the picker & clamps parsed windows.
    # SIMULATED TODAY: the case dataset lives in a fixed historical window, so
    # relative phrases in prompts ("next month", "a Tuesday") are resolved
    # against a simulated current date anchored to the dataset's first
    # departure — a documented assumption, shown openly in the UI below.
    _dep_dates = pd.to_datetime(flights_df["departure_utc"]).dt.date
    data_min, data_max = _dep_dates.min(), _dep_dates.max()
    # April sits in the densest part of most seasonal datasets (clear of the
    # Jan/Feb sparse edge), so relative dates land on months with real flight
    # coverage more often. Clamped to actual data bounds for safety.
    _april_anchor = pd.Timestamp(year=data_min.year, month=4, day=1).date()
    sim_today = min(max(_april_anchor, data_min), data_max)
    # Optional pin via env var or secrets.toml (SIM_TODAY = "YYYY-MM-DD"),
    # e.g. the date recommended by find_sim_today.py. Clamped to coverage.
    _ov = os.environ.get("SIM_TODAY", "")
    if not _ov:
        try:
            _ov = st.secrets.get("SIM_TODAY", "")
        except Exception:
            _ov = ""
    _ovd = safe_date(_ov)
    if _ovd:
        sim_today = min(max(_ovd, data_min), data_max)
    st.caption(f"🗓️ Simulated today: **{sim_today.strftime('%a %d %b %Y')}** — relative "
               f"dates like “next month” are resolved against this fixed reference point "
               f"(chosen in the denser part of the dataset's {data_min} → {data_max} "
               f"coverage), since the case data lives in a historical window rather "
               f"than real time.")

    if ask and nl_request.strip():
        flex_days = int(user.get("date_flexibility_days", 0) or 0)
        parsed = parse_travel_request(
            nl_request.strip(), str(user["home_airport"]),
            json.dumps({k: (v if isinstance(v, str) else "") for k, v in airport_map.items()}),
            sim_today.strftime("%Y-%m-%d"), flex_days, llm_on=(client is not None))
        if parsed.get("destination"):
            st.session_state["origin_sel"] = parsed["origin"]
            st.session_state["dest_sel"] = parsed["destination"]
            st.session_state["extra_stops"] = parsed.get("additional_stops", [])
            st.session_state["interpretation"] = parsed.get("interpretation", "")
            st.session_state["trip_type"] = parsed.get("trip_type", "one_way")
            rf, rt = safe_date(parsed.get("return_date_from")), safe_date(parsed.get("return_date_to"))
            st.session_state["return_window"] = (rf.isoformat() if rf else None,
                                                 rt.isoformat() if rt else None)
            st.session_state["trip_duration_days"] = parsed.get("trip_duration_days")
            st.session_state["return_home"] = parsed.get("return_home", True)
            st.session_state["order_flexible"] = parsed.get("order_flexible", True)
            st.session_state["region_pool"] = parsed.get("region_pool", [])
            st.session_state["auto_search"] = True
            # Clamp the parsed window into dataset coverage for the picker.
            # One-sided windows are completed, not discarded: "after May 31st"
            # becomes a 30-day window so results never silently jump months.
            rng = ()
            pf, pt = safe_date(parsed.get("date_from")), safe_date(parsed.get("date_to"))
            if pf and not pt:
                pt = min(data_max, pf + timedelta(days=30))
                st.session_state["interpretation"] += (
                    f" Showing the 30 days from {pf} — widen the date picker for more.")
            if pt and not pf:
                pf = data_min
            # PAST-PROOFING: a window entirely before simulated today rolls
            # forward by whole years (summer stays summer); partial overlaps
            # are floored at simulated today. The engine never books the past.
            if pf and pt and pt < sim_today:
                for _ in range(3):
                    if pt >= sim_today:
                        break
                    pf = (pd.Timestamp(pf) + pd.DateOffset(years=1)).date()
                    pt = (pd.Timestamp(pt) + pd.DateOffset(years=1)).date()
                if pt >= sim_today:
                    st.session_state["interpretation"] += (
                        f" That period is in the past relative to simulated today — "
                        f"interpreted as its next occurrence ({pf} → {pt}).")
            if pf and pt:
                lo, hi = max(data_min, sim_today, pf), min(data_max, pt)
                if lo <= hi:
                    rng = (lo, hi)
            # No dates at all: default to the UPCOMING window from simulated
            # today — never search the whole dataset (or the past) silently.
            # A stated duration widens the window ("three weeks" -> ~35 days).
            if not rng and not (pf or pt):
                dur = int(parsed.get("trip_duration_days") or 0)
                span = max(30, dur + 14)
                lo = max(data_min, sim_today)
                hi = min(data_max, lo + timedelta(days=span))
                if lo <= hi:
                    rng = (lo, hi)
                    st.session_state["interpretation"] += (
                        f" No dates given — searching the upcoming {span} days "
                        f"from {sim_today}.")
                else:
                    st.session_state["interpretation"] += (
                        f" (Requested dates fall outside the dataset's "
                        f"{data_min}–{data_max} coverage, so all dates are shown.)")
            st.session_state["date_range_input"] = rng
        else:
            # A failed parse must not leave stale results from a previous one
            for k in ("interpretation", "extra_stops", "trip_type", "return_window",
                      "trip_duration_days", "return_home", "order_flexible",
                      "region_pool"):
                st.session_state.pop(k, None)
            st.warning("I couldn't identify a destination airport in that request — "
                       "please pick one below or mention a city from the dataset.")

    if st.session_state.get("interpretation"):
        st.caption(f"🧭 {st.session_state['interpretation']}")
    trip_type = st.session_state.get("trip_type", "one_way")
    extra_stops = st.session_state.get("extra_stops", [])
    if trip_type == "multi_city" and extra_stops:
        stops_txt = " → ".join(fmt_airport(c) for c in extra_stops)
        st.info(f"🗺️ Multi-city journey: planning all legs, continuing to {stops_txt}. "
                "Legs are chained greedily by effective cost.")
    elif trip_type == "round_trip":
        st.info("🔁 Round trip detected: planning outbound and return legs.")

    # ── Search panel ────────────────────────────────────────────────────────
    def _clear_dates():
        st.session_state["date_range_input"] = ()

    def _apply_window(lo, hi):
        st.session_state["date_range_input"] = (lo, hi) if lo and hi else ()
        st.session_state["auto_search"] = True

    with st.container():
        html('<div class="xp-panel" style="padding-bottom:6px; margin-bottom:0;"><h4 style="margin:0 0 8px 0;">Where to?</h4></div>')
        c1, c2, c3, c4 = st.columns([3, 3, 3, 2], vertical_alignment="bottom")
        origins = sorted(flights_df["origin"].unique().tolist())
        # Validate any parsed origin before the widget renders
        if st.session_state.get("origin_sel") not in origins:
            st.session_state["origin_sel"] = (user["home_airport"]
                                              if user["home_airport"] in origins else origins[0])
        with c1:
            origin = st.selectbox("Leaving from", origins, key="origin_sel",
                                  format_func=fmt_airport)
        dests = sorted([d for d in flights_df["destination"].unique() if d != origin])
        if st.session_state.get("dest_sel") not in dests:
            st.session_state["dest_sel"] = dests[0]
        with c2:
            destination = st.selectbox("Going to", dests, key="dest_sel",
                                       format_func=fmt_airport)
        with c3:
            # Sanitize any stored range against the sim-today floor first,
            # or date_input raises on out-of-bounds session values
            _r = st.session_state.get("date_range_input", ())
            if isinstance(_r, (tuple, list)) and _r:
                _ok = [d for d in _r if d and sim_today <= d <= data_max]
                st.session_state["date_range_input"] = tuple(_ok) if len(_ok) == len(_r) else ()
            if "date_range_input" not in st.session_state:
                st.session_state["date_range_input"] = ()
            date_range = st.date_input(
                "Departure window (optional)", key="date_range_input",
                min_value=sim_today, max_value=data_max,
                help=f"Searchable window: {sim_today} (simulated today) to {data_max}. "
                     "Leave empty for any upcoming date. "
                     "Filled automatically when you ask in plain language.")
        with c4:
            search = st.button("🔍 Search with AI", use_container_width=True)

    # Derive the active window from the picker (0, 1 or 2 dates selected)
    date_from = date_to = None
    if isinstance(date_range, (tuple, list)):
        if len(date_range) >= 1 and date_range[0]:
            date_from = date_range[0].strftime("%Y-%m-%d")
        if len(date_range) == 2 and date_range[1]:
            date_to = date_range[1].strftime("%Y-%m-%d")
    elif date_range:  # single date object
        date_from = date_to = date_range.strftime("%Y-%m-%d")

    if date_from or date_to:
        w1, w2 = st.columns([5, 1], vertical_alignment="center")
        with w1:
            st.caption(f"📅 Filtering departures: **{date_from or 'any'} → {date_to or date_from}**")
        with w2:
            st.button("✕ Clear dates", use_container_width=True, on_click=_clear_dates)

    if st.session_state.pop("auto_search", False):
        search = True

    # HARD FLOOR: no search path may look before simulated today — a traveler
    # cannot board a flight in the past. Applies to journeys, single searches,
    # the "all dates" button, and coverage probes alike.
    if date_from is None or (safe_date(date_from) or sim_today) < sim_today:
        date_from = sim_today.isoformat()

    st.markdown("")

    if not search:
        html("""
        <div class="empty-hint">
          <span class="big">🧭</span>
          Pick an origin and destination, then hit <b>Search with AI</b> —
          the engine will route, rank and explain the best flights for this traveler.
        </div>""")
        return

    # ── JOURNEY MODE: round trips & multi-city chains (IRCTC-style) ────────
    if trip_type in ("round_trip", "multi_city"):
        stops = [destination] + [c for c in extra_stops if c not in (origin, destination)]
        ret_from, ret_to = st.session_state.get("return_window", (None, None))
        dur_days = st.session_state.get("trip_duration_days")
        return_home = bool(st.session_state.get("return_home", True)) \
            or trip_type == "round_trip"
        order_flex = bool(st.session_state.get("order_flexible", True)) \
            and trip_type == "multi_city"
        region_pool = st.session_state.get("region_pool") or []
        with st.spinner("AI is chaining your journey — orderings, legs, trade-offs…"):
            legs, failed, jnotes, order = plan_multicity(
                origin, stops, user, weights, flights_df,
                first_window=(date_from, date_to), return_window=(ret_from, ret_to),
                trip_duration_days=dur_days, return_home=return_home,
                order_flexible=order_flex, region_pool=region_pool)
        for _n in jnotes:
            st.info(f"📅 {_n}")
        if failed and not legs:
            st.error(f"Couldn't find any itinerary for the first leg "
                     f"{fmt_airport(failed[0])} → {fmt_airport(failed[1])} in any "
                     f"tried ordering within this window.")
            # Nearest-departure rescue: find the earliest future departure from
            # the origin toward ANY of the requested stops and offer that window
            near = None
            for c in dict.fromkeys([failed[1]] + stops):
                probe = optimize_flights(origin, c, user, weights, flights_df,
                                         date_from=sim_today.isoformat(),
                                         enforce_max_layover=False)
                if not probe.empty:
                    d0 = probe["dep_time"].dt.date.min()
                    if near is None or d0 < near[0]:
                        near = (d0, c)
            if near:
                lo, hi = near[0], min(data_max, near[0] + timedelta(days=30))
                st.warning(f"The nearest departure from {fmt_airport(origin)} toward "
                           f"{fmt_airport(near[1])} is **{lo}** — want me to plan the "
                           f"journey around that date instead?")
                st.button(f"🔍 Plan around {lo} instead", on_click=_apply_window,
                          args=(lo, hi))
            else:
                st.caption("This route combination has no future departures in the "
                           "dataset at all — a data-coverage gap, not a preference issue.")
            return
        if failed:
            st.warning(f"Planned {len(legs)} leg(s), but no feasible itinerary for "
                       f"{fmt_airport(failed[0])} → {fmt_airport(failed[1])} after the "
                       f"previous arrival — showing the partial journey.")

        total_price = sum(float(l["price"]) for l in legs)
        first_dep = legs[0]["dep_time"]
        last_arr = legs[-1]["dep_time"] + pd.Timedelta(minutes=int(legs[-1]["duration_minutes"]))
        span_days = max(1, (last_arr.date() - first_dep.date()).days)

        j_expl = generate_journey_explanation(user, weights, legs, jnotes,
                                              span_days, total_price)
        html(f"""
        <div class="ai-insight-box">
          <span class="ai-title">✨ Expedia AI Insight</span>
          {j_expl}
        </div>""")

        # Route text from the ACTUAL planned legs (reflects chosen order/subs)
        route_txt = " → ".join([fmt_airport(l["origin"]) for l in legs]
                               + [fmt_airport(legs[-1]["destination"])])
        html(f"""
        <div class="journey-head">
          <div>
            <div class="jh-route">{'🔁' if trip_type == 'round_trip' else '🗺️'} {route_txt}</div>
            <div class="jh-meta">{len(legs)} flight leg(s) · {span_days} day journey{f' · target ≤ {int(dur_days)} days' if dur_days else ''} ·
              departs {first_dep.strftime('%a %d %b %Y')}</div>
          </div>
          <div class="jh-price">${total_price:,.0f}
            <span class="sub">total per traveler · time valued @ ${legs[0]['vot_per_hour']:.0f}/hr</span>
          </div>
        </div>""")

        for i, leg in enumerate(legs):
            render_flight_card(leg, user, is_top=(i == 0),
                               leg_label=f"Leg {i + 1} · {leg['origin']} → {leg['destination']}")
            if i < len(legs) - 1:
                arr = leg["dep_time"] + pd.Timedelta(minutes=int(leg["duration_minutes"]))
                nxt = legs[i + 1]["dep_time"]
                stay = nxt - arr
                stay_txt = f"{stay.days}d {stay.seconds // 3600}h" if stay.days else f"{stay.seconds // 3600}h"
                html(f'<div class="stay-divider">🏨 Stay in '
                     f'{fmt_airport(leg["destination"])} — {stay_txt}</div>')

        st.caption("Legs are chosen greedily by per-user effective cost with time-feasible "
                   "chaining (each leg departs after the previous arrival + stay). "
                   "Jointly-optimal multi-leg search is on the roadmap.")
        return

    with st.spinner("AI is analyzing routes, constraints and trade-offs…"):
        relax_note = None
        # Tier 1: user's dates + layover cap (strict)
        results = optimize_flights(origin, destination, user, weights, flights_df,
                                   date_from=date_from, date_to=date_to)
        # Tier 2: keep the dates, relax the layover cap — the traveler's dates
        # matter more than their connection-length comfort
        if results.empty:
            results = optimize_flights(origin, destination, user, weights, flights_df,
                                       date_from=date_from, date_to=date_to,
                                       enforce_max_layover=False)
            if not results.empty:
                relax_note = "layover"
        # Empty window: DON'T silently search other dates. Probe only the ~30
        # days after the window and ask the user before searching there.
        ext_offer, all_offer = None, None
        if results.empty and (date_from or date_to):
            _end = safe_date(date_to or date_from)
            ext_lo = _end + timedelta(days=1)
            ext_hi = min(data_max, ext_lo + timedelta(days=29))
            if ext_lo <= data_max:
                probe = optimize_flights(origin, destination, user, weights, flights_df,
                                         date_from=ext_lo.isoformat(),
                                         date_to=ext_hi.isoformat())
                if probe.empty:
                    probe = optimize_flights(origin, destination, user, weights,
                                             flights_df, date_from=ext_lo.isoformat(),
                                             date_to=ext_hi.isoformat(),
                                             enforce_max_layover=False)
                if not probe.empty:
                    ext_offer = (ext_lo, ext_hi, len(probe),
                                 probe["dep_time"].dt.date.min())
            if ext_offer is None:
                coverage = optimize_flights(origin, destination, user, weights,
                                            flights_df, date_from=sim_today.isoformat(),
                                            enforce_max_layover=False)
                if not coverage.empty:
                    rd = coverage["dep_time"].dt.date
                    all_offer = (rd.min(), rd.max())

    if results.empty and ext_offer:
        lo, hi, n, first = ext_offer
        st.warning(f"No departures between **{date_from}** and **{date_to or date_from}** "
                   f"on this route — but there are **{n} option(s) in the following "
                   f"month** (earliest {first}). Want me to search that window?")
        st.button(f"🔍 Search {lo} → {hi} instead", on_click=_apply_window, args=(lo, hi))
        return
    if results.empty and all_offer:
        nearest = all_offer[0]
        near_hi = min(data_max, nearest + timedelta(days=30))
        st.warning(f"No flights between **{date_from}** and **{date_to or date_from}** "
                   f"on this route, nor in the following month. The **nearest "
                   f"departure is {nearest}** (route flies {all_offer[0]} → "
                   f"{all_offer[1]}) — plan around that?")
        st.button(f"🔍 Search {nearest} → {near_hi} instead",
                  on_click=_apply_window, args=(nearest, near_hi))
        return

    _umax = int(user["max_layover_minutes"])
    if relax_note == "layover":
        st.warning(f"😔 Sorry — this route has no direct flights and no connections "
                   f"within your {_umax}-minute max layover in the selected window. "
                   f"Showing the best available options; the marked flights need a "
                   f"longer connection than you'd prefer.")

    if results.empty:
        st.error(f"No valid itineraries from {fmt_airport(origin)} to {fmt_airport(destination)} "
                 f"within a {int(user['max_layover_minutes'])}-minute max layover. "
                 "Try another route or a user with looser constraints.")
        return

    top = results.iloc[0]
    alternatives = results.iloc[1:]

    # ── AI insight ──────────────────────────────────────────────────────────
    explanation = generate_llm_explanation(user, weights, top, alternatives,
                                           relax_note=relax_note)
    html(f"""
    <div class="ai-insight-box">
      <span class="ai-title">✨ Expedia AI Insight</span>
      {explanation}
    </div>""")

    # ── Explicit cost-vs-time trade-off ─────────────────────────────────────
    render_tradeoff_strip(results, top)

    # ── Flight cards ────────────────────────────────────────────────────────
    st.markdown("### Top recommended flights")
    for i in range(min(3, len(results))):
        render_flight_card(results.iloc[i], user, is_top=(i == 0))

    # ── Trade-off chart ─────────────────────────────────────────────────────
    st.markdown("### 📊 Route trade-off landscape")
    st.caption("Each dot is a valid itinerary. Down-left is cheaper **and** faster; "
               "the blue dot is what the engine picked for this traveler's weights.")
    render_tradeoff_chart(results, top["flight_id"])


if __name__ == "__main__":
    main()