import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go
import time

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    color: #e2e8f0;
}

.stApp {
    background: #050810;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,255,200,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(127,90,240,0.08) 0%, transparent 55%),
        repeating-linear-gradient(0deg, transparent, transparent 59px, rgba(0,255,200,0.025) 60px),
        repeating-linear-gradient(90deg, transparent, transparent 59px, rgba(0,255,200,0.025) 60px);
}

[data-testid="stSidebar"] {
    background: rgba(5,8,16,0.97) !important;
    border-right: 1px solid rgba(0,255,200,0.12) !important;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

#MainMenu, footer, header { visibility: hidden; }
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }

/* Hide Streamlit's own sidebar toggle */
[data-testid="collapsedControl"] { display: none !important; }
button[kind="header"] { display: none !important; }
.st-emotion-cache-1cypcdb { display: none !important; }

/* Hamburger button — small, square, neon */
div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button {
    padding: 6px 12px !important;
    min-height: 0 !important;
    height: 36px !important;
    width: 44px !important;
    font-size: 18px !important;
    letter-spacing: 0 !important;
    border-radius: 10px !important;
}

.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
}
.stat-label { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #475569; margin-bottom: 10px; }
.stat-value { font-family: 'Orbitron', monospace; font-size: 44px; font-weight: 900; line-height: 1; }
.stat-pos { color: #00ffcc; text-shadow: 0 0 24px rgba(0,255,200,0.5); }
.stat-neg { color: #ff4d6d; text-shadow: 0 0 24px rgba(255,77,109,0.5); }
.stat-neu { color: #7f5af0; text-shadow: 0 0 24px rgba(127,90,240,0.5); }

.result-badge { display: inline-block; padding: 10px 28px; border-radius: 50px; font-family: 'Orbitron', monospace; font-size: 18px; font-weight: 700; letter-spacing: 2px; margin: 12px 0; }
.badge-pos { background: rgba(0,255,200,0.1);  border: 1px solid rgba(0,255,200,0.6);  color: #00ffcc; box-shadow: 0 0 20px rgba(0,255,200,0.2); }
.badge-neg { background: rgba(255,77,109,0.1); border: 1px solid rgba(255,77,109,0.6); color: #ff4d6d; box-shadow: 0 0 20px rgba(255,77,109,0.2); }
.badge-neu { background: rgba(127,90,240,0.1); border: 1px solid rgba(127,90,240,0.6); color: #a78bfa; box-shadow: 0 0 20px rgba(127,90,240,0.2); }

.conf-bar-wrap { margin: 16px 0; background: rgba(255,255,255,0.05); border-radius: 50px; height: 10px; overflow: hidden; }
.conf-bar-fill  { height: 100%; border-radius: 50px; }

.history-chip { display: inline-block; padding: 4px 14px; border-radius: 50px; font-size: 12px; font-weight: 600; margin: 3px; letter-spacing: 1px; }
.chip-pos { background: rgba(0,255,200,0.12);  color: #00ffcc; border: 1px solid rgba(0,255,200,0.3);  }
.chip-neg { background: rgba(255,77,109,0.12); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.3); }
.chip-neu { background: rgba(127,90,240,0.12); color: #a78bfa; border: 1px solid rgba(127,90,240,0.3); }

.neon-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,200,0.3), transparent); margin: 20px 0; }
.section-header { font-family: 'Orbitron', monospace; font-size: 10px; letter-spacing: 4px; text-transform: uppercase; color: #00ffcc; margin-bottom: 4px; }

.sidebar-logo    { font-family: 'Orbitron', monospace; font-size: 20px; font-weight: 900; background: linear-gradient(90deg,#00ffcc,#7f5af0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding: 8px 0 4px 0; }
.sidebar-version { font-size: 10px; letter-spacing: 3px; color: #1e293b; text-transform: uppercase; margin-bottom: 20px; }

.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,255,200,0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
}
.stTextArea textarea:focus { border-color: rgba(0,255,200,0.5) !important; }

.stButton > button {
    background: linear-gradient(135deg, rgba(0,255,200,0.12), rgba(127,90,240,0.12)) !important;
    border: 1px solid rgba(0,255,200,0.35) !important;
    color: #00ffcc !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    padding: 10px 28px !important;
    border-radius: 50px !important;
    transition: all 0.25s !important;
}
.stButton > button:hover { box-shadow: 0 0 28px rgba(0,255,200,0.18) !important; transform: translateY(-1px) !important; }

.stDownloadButton > button {
    background: rgba(127,90,240,0.1) !important;
    border: 1px solid rgba(127,90,240,0.4) !important;
    color: #a78bfa !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    border-radius: 50px !important;
}

.stSelectbox > div > div { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(0,255,200,0.2) !important; border-radius: 10px !important; color: #e2e8f0 !important; }
[data-testid="stFileUploader"] { border: 1px dashed rgba(0,255,200,0.2) !important; border-radius: 12px !important; }

.empty-state { text-align: center; padding: 70px 20px; }
.empty-icon  { font-size: 56px; margin-bottom: 16px; display: block; }
.empty-text  { font-family: 'Orbitron', monospace; font-size: 14px; letter-spacing: 3px; text-transform: uppercase; color: #334155; }
.empty-sub   { font-size: 13px; color: #1e293b; margin-top: 8px; }

.welcome-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(38px, 6vw, 76px);
    font-weight: 900;
    background: linear-gradient(90deg, #00ffcc 0%, #7f5af0 50%, #00ffcc 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
    line-height: 1.1;
    margin-bottom: 16px;
}
@keyframes shimmer { 0% { background-position: 0% center; } 100% { background-position: 200% center; } }
.welcome-sub  { font-size: 18px; color: #475569; letter-spacing: 1px; margin-bottom: 36px; }
.feature-pill { display: inline-block; padding: 6px 18px; border-radius: 50px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09); font-size: 13px; margin: 4px; color: #64748b; }

.nav-locked { opacity: 0.3; padding: 10px 14px; font-size: 14px; color: #475569; }

.locked-banner { border: 1px solid rgba(127,90,240,0.3); background: rgba(127,90,240,0.06); border-radius: 14px; padding: 40px; text-align: center; margin-top: 40px; }
.locked-icon   { font-size: 44px; display: block; margin-bottom: 14px; }
.locked-title  { font-family: 'Orbitron', monospace; font-size: 15px; letter-spacing: 3px; color: #a78bfa; margin-bottom: 10px; }
.locked-sub    { font-size: 13px; color: #475569; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
LABEL_MAP = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
COLORS    = {"POSITIVE": "#00ffcc", "NEUTRAL": "#7f5af0", "NEGATIVE": "#ff4d6d"}
BADGE_CSS = {"POSITIVE": "badge-pos", "NEUTRAL": "badge-neu", "NEGATIVE": "badge-neg"}
CHIP_CSS  = {"POSITIVE": "chip-pos",  "NEUTRAL": "chip-neu",  "NEGATIVE": "chip-neg"}
STAT_CSS  = {"POSITIVE": "stat-pos",  "NEUTRAL": "stat-neu",  "NEGATIVE": "stat-neg"}
ICONS     = {"POSITIVE": "🟢", "NEUTRAL": "🟣", "NEGATIVE": "🔴"}

def decode(label):
    return LABEL_MAP.get(label, label)

def conf_bar(score, sentiment):
    pct   = int(score * 100)
    color = COLORS[sentiment]
    return f"""
    <div class='conf-bar-wrap'>
      <div class='conf-bar-fill' style='width:{pct}%;background:linear-gradient(90deg,{color}55,{color});'></div>
    </div>
    <div style='font-size:12px;color:#475569;letter-spacing:2px;text-align:right;margin-top:6px;'>
        CONFIDENCE &nbsp;·&nbsp; <span style='color:{color};font-weight:700;'>{pct}%</span>
    </div>"""


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
    )


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k, v in {"started": False, "history": [], "page": "Analyze"}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
#  WELCOME SCREEN
# ─────────────────────────────────────────────
if not st.session_state.started:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="welcome-title">Sentiment<br>Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="welcome-sub">Decode emotion. Visualize truth. In real-time.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='margin-bottom:32px;'>
            <span class='feature-pill'>⚡ RoBERTa NLP</span>
            <span class='feature-pill'>📊 Live Charts</span>
            <span class='feature-pill'>📂 Bulk CSV</span>
            <span class='feature-pill'>💾 Export</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex;gap:16px;margin-bottom:36px;'>
            <div style='flex:1;padding:18px;border-radius:14px;background:rgba(0,255,200,0.05);border:1px solid rgba(0,255,200,0.15);'>
                <div style='font-size:22px;margin-bottom:8px;'>✍️</div>
                <div style='font-size:11px;letter-spacing:2px;color:#00ffcc;font-family:Orbitron,monospace;margin-bottom:4px;'>STEP 1</div>
                <div style='font-size:13px;color:#475569;'>Analyze any text for sentiment</div>
            </div>
            <div style='flex:1;padding:18px;border-radius:14px;background:rgba(127,90,240,0.05);border:1px solid rgba(127,90,240,0.15);'>
                <div style='font-size:22px;margin-bottom:8px;'>📊</div>
                <div style='font-size:11px;letter-spacing:2px;color:#7f5af0;font-family:Orbitron,monospace;margin-bottom:4px;'>STEP 2</div>
                <div style='font-size:13px;color:#475569;'>Dashboard unlocks with your data</div>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button("⚡  INITIALIZE SYSTEM", use_container_width=True):
            st.session_state.started = True
            st.session_state.page   = "Analyze"
            st.rerun()
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR  — uses st.button instead of
#  st.radio to eliminate keyboard-stuck bug
# ─────────────────────────────────────────────
# ── Hamburger + slide-down nav drawer ──
has_data = len(st.session_state.history) > 0
menu = st.session_state.page

if "nav_open" not in st.session_state:
    st.session_state.nav_open = False

# Top bar row
bar_left, bar_mid, bar_right = st.columns([1, 8, 2])
with bar_left:
    if st.button("☰", key="hamburger"):
        st.session_state.nav_open = not st.session_state.nav_open
        st.rerun()
with bar_mid:
    st.markdown("<div style='font-family:Orbitron,monospace;font-size:16px;font-weight:900;background:linear-gradient(90deg,#00ffcc,#7f5af0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding-top:8px;'>⚡ SentimentIQ</div>", unsafe_allow_html=True)
with bar_right:
    total = len(st.session_state.history)
    if total:
        st.markdown(f"<div style='text-align:right;font-size:12px;color:#475569;padding-top:10px;'>{total} analyzed</div>", unsafe_allow_html=True)

st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(0,255,200,0.3),transparent);margin-bottom:8px;'></div>", unsafe_allow_html=True)

# Slide-down nav drawer — only renders when open
if st.session_state.nav_open:
    st.markdown("""
    <div style='background:rgba(5,8,16,0.97);border:1px solid rgba(0,255,200,0.15);
                border-radius:16px;padding:16px;margin-bottom:20px;'>
        <div style='font-size:10px;letter-spacing:4px;color:#00ffcc;
                    font-family:Orbitron,monospace;margin-bottom:12px;'>NAVIGATION</div>
    </div>
    """, unsafe_allow_html=True)

    n1, n2, n3, n4 = st.columns(4)
    with n1:
        if st.button("✍️  Analyze", key="dn_analyze", use_container_width=True):
            st.session_state.page = "Analyze"
            st.session_state.nav_open = False
            st.rerun()
    with n2:
        if st.button("📂  Dataset", key="dn_dataset", use_container_width=True):
            st.session_state.page = "Dataset"
            st.session_state.nav_open = False
            st.rerun()
    with n3:
        if has_data:
            if st.button("📊  Dashboard", key="dn_dashboard", use_container_width=True):
                st.session_state.page = "Dashboard"
                st.session_state.nav_open = False
                st.rerun()
        else:
            st.markdown("<div style='text-align:center;padding:9px;font-size:13px;color:#334155;border:1px solid rgba(255,255,255,0.05);border-radius:50px;'>📊 🔒</div>", unsafe_allow_html=True)
    with n4:
        if st.button("↺  Reset", key="dn_reset", use_container_width=True):
            st.session_state.history = []
            st.session_state.page = "Analyze"
            st.session_state.nav_open = False
            st.rerun()

    # Session stats row
    if st.session_state.history:
        pos = sum(1 for h in st.session_state.history if h["sentiment"] == "POSITIVE")
        neg = sum(1 for h in st.session_state.history if h["sentiment"] == "NEGATIVE")
        neu = len(st.session_state.history) - pos - neg
        st.markdown(f"""
        <div style='display:flex;gap:20px;padding:10px 4px 4px 4px;font-size:13px;'>
            <span style='color:#00ffcc;'>🟢 Positive &nbsp;<b>{pos}</b></span>
            <span style='color:#ff4d6d;'>🔴 Negative &nbsp;<b>{neg}</b></span>
            <span style='color:#a78bfa;'>🟣 Neutral &nbsp;<b>{neu}</b></span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.05);margin:12px 0 20px 0;'></div>", unsafe_allow_html=True)

mport streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go
import time

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    color: #e2e8f0;
}

.stApp {
    background: #050810;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,255,200,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(127,90,240,0.08) 0%, transparent 55%),
        repeating-linear-gradient(0deg, transparent, transparent 59px, rgba(0,255,200,0.025) 60px),
        repeating-linear-gradient(90deg, transparent, transparent 59px, rgba(0,255,200,0.025) 60px);
}

[data-testid="stSidebar"] {
    background: rgba(5,8,16,0.97) !important;
    border-right: 1px solid rgba(0,255,200,0.12) !important;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

#MainMenu, footer, header { visibility: hidden; }
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }

/* Hide the sidebar collapse/expand toggle arrow button */
[data-testid="collapsedControl"] { display: none !important; }
button[kind="header"] { display: none !important; }
.st-emotion-cache-1cypcdb { display: none !important; }

.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
}
.stat-label { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #475569; margin-bottom: 10px; }
.stat-value { font-family: 'Orbitron', monospace; font-size: 44px; font-weight: 900; line-height: 1; }
.stat-pos { color: #00ffcc; text-shadow: 0 0 24px rgba(0,255,200,0.5); }
.stat-neg { color: #ff4d6d; text-shadow: 0 0 24px rgba(255,77,109,0.5); }
.stat-neu { color: #7f5af0; text-shadow: 0 0 24px rgba(127,90,240,0.5); }

.result-badge { display: inline-block; padding: 10px 28px; border-radius: 50px; font-family: 'Orbitron', monospace; font-size: 18px; font-weight: 700; letter-spacing: 2px; margin: 12px 0; }
.badge-pos { background: rgba(0,255,200,0.1);  border: 1px solid rgba(0,255,200,0.6);  color: #00ffcc; box-shadow: 0 0 20px rgba(0,255,200,0.2); }
.badge-neg { background: rgba(255,77,109,0.1); border: 1px solid rgba(255,77,109,0.6); color: #ff4d6d; box-shadow: 0 0 20px rgba(255,77,109,0.2); }
.badge-neu { background: rgba(127,90,240,0.1); border: 1px solid rgba(127,90,240,0.6); color: #a78bfa; box-shadow: 0 0 20px rgba(127,90,240,0.2); }

.conf-bar-wrap { margin: 16px 0; background: rgba(255,255,255,0.05); border-radius: 50px; height: 10px; overflow: hidden; }
.conf-bar-fill  { height: 100%; border-radius: 50px; }

.history-chip { display: inline-block; padding: 4px 14px; border-radius: 50px; font-size: 12px; font-weight: 600; margin: 3px; letter-spacing: 1px; }
.chip-pos { background: rgba(0,255,200,0.12);  color: #00ffcc; border: 1px solid rgba(0,255,200,0.3);  }
.chip-neg { background: rgba(255,77,109,0.12); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.3); }
.chip-neu { background: rgba(127,90,240,0.12); color: #a78bfa; border: 1px solid rgba(127,90,240,0.3); }

.neon-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,200,0.3), transparent); margin: 20px 0; }
.section-header { font-family: 'Orbitron', monospace; font-size: 10px; letter-spacing: 4px; text-transform: uppercase; color: #00ffcc; margin-bottom: 4px; }

.sidebar-logo    { font-family: 'Orbitron', monospace; font-size: 20px; font-weight: 900; background: linear-gradient(90deg,#00ffcc,#7f5af0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding: 8px 0 4px 0; }
.sidebar-version { font-size: 10px; letter-spacing: 3px; color: #1e293b; text-transform: uppercase; margin-bottom: 20px; }

.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,255,200,0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
}
.stTextArea textarea:focus { border-color: rgba(0,255,200,0.5) !important; }

.stButton > button {
    background: linear-gradient(135deg, rgba(0,255,200,0.12), rgba(127,90,240,0.12)) !important;
    border: 1px solid rgba(0,255,200,0.35) !important;
    color: #00ffcc !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    padding: 10px 28px !important;
    border-radius: 50px !important;
    transition: all 0.25s !important;
}
.stButton > button:hover { box-shadow: 0 0 28px rgba(0,255,200,0.18) !important; transform: translateY(-1px) !important; }

.stDownloadButton > button {
    background: rgba(127,90,240,0.1) !important;
    border: 1px solid rgba(127,90,240,0.4) !important;
    color: #a78bfa !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    border-radius: 50px !important;
}

.stSelectbox > div > div { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(0,255,200,0.2) !important; border-radius: 10px !important; color: #e2e8f0 !important; }
[data-testid="stFileUploader"] { border: 1px dashed rgba(0,255,200,0.2) !important; border-radius: 12px !important; }

.empty-state { text-align: center; padding: 70px 20px; }
.empty-icon  { font-size: 56px; margin-bottom: 16px; display: block; }
.empty-text  { font-family: 'Orbitron', monospace; font-size: 14px; letter-spacing: 3px; text-transform: uppercase; color: #334155; }
.empty-sub   { font-size: 13px; color: #1e293b; margin-top: 8px; }

.welcome-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(38px, 6vw, 76px);
    font-weight: 900;
    background: linear-gradient(90deg, #00ffcc 0%, #7f5af0 50%, #00ffcc 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
    line-height: 1.1;
    margin-bottom: 16px;
}
@keyframes shimmer { 0% { background-position: 0% center; } 100% { background-position: 200% center; } }
.welcome-sub  { font-size: 18px; color: #475569; letter-spacing: 1px; margin-bottom: 36px; }
.feature-pill { display: inline-block; padding: 6px 18px; border-radius: 50px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09); font-size: 13px; margin: 4px; color: #64748b; }

.nav-locked { opacity: 0.3; padding: 10px 14px; font-size: 14px; color: #475569; }

.locked-banner { border: 1px solid rgba(127,90,240,0.3); background: rgba(127,90,240,0.06); border-radius: 14px; padding: 40px; text-align: center; margin-top: 40px; }
.locked-icon   { font-size: 44px; display: block; margin-bottom: 14px; }
.locked-title  { font-family: 'Orbitron', monospace; font-size: 15px; letter-spacing: 3px; color: #a78bfa; margin-bottom: 10px; }
.locked-sub    { font-size: 13px; color: #475569; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
LABEL_MAP = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
COLORS    = {"POSITIVE": "#00ffcc", "NEUTRAL": "#7f5af0", "NEGATIVE": "#ff4d6d"}
BADGE_CSS = {"POSITIVE": "badge-pos", "NEUTRAL": "badge-neu", "NEGATIVE": "badge-neg"}
CHIP_CSS  = {"POSITIVE": "chip-pos",  "NEUTRAL": "chip-neu",  "NEGATIVE": "chip-neg"}
STAT_CSS  = {"POSITIVE": "stat-pos",  "NEUTRAL": "stat-neu",  "NEGATIVE": "stat-neg"}
ICONS     = {"POSITIVE": "🟢", "NEUTRAL": "🟣", "NEGATIVE": "🔴"}

def decode(label):
    return LABEL_MAP.get(label, label)

def conf_bar(score, sentiment):
    pct   = int(score * 100)
    color = COLORS[sentiment]
    return f"""
    <div class='conf-bar-wrap'>
      <div class='conf-bar-fill' style='width:{pct}%;background:linear-gradient(90deg,{color}55,{color});'></div>
    </div>
    <div style='font-size:12px;color:#475569;letter-spacing:2px;text-align:right;margin-top:6px;'>
        CONFIDENCE &nbsp;·&nbsp; <span style='color:{color};font-weight:700;'>{pct}%</span>
    </div>"""


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
    )


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k, v in {"started": False, "history": [], "page": "Analyze"}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
#  WELCOME SCREEN
# ─────────────────────────────────────────────
if not st.session_state.started:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="welcome-title">Sentiment<br>Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="welcome-sub">Decode emotion. Visualize truth. In real-time.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='margin-bottom:32px;'>
            <span class='feature-pill'>⚡ RoBERTa NLP</span>
            <span class='feature-pill'>📊 Live Charts</span>
            <span class='feature-pill'>📂 Bulk CSV</span>
            <span class='feature-pill'>💾 Export</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex;gap:16px;margin-bottom:36px;'>
            <div style='flex:1;padding:18px;border-radius:14px;background:rgba(0,255,200,0.05);border:1px solid rgba(0,255,200,0.15);'>
                <div style='font-size:22px;margin-bottom:8px;'>✍️</div>
                <div style='font-size:11px;letter-spacing:2px;color:#00ffcc;font-family:Orbitron,monospace;margin-bottom:4px;'>STEP 1</div>
                <div style='font-size:13px;color:#475569;'>Analyze any text for sentiment</div>
            </div>
            <div style='flex:1;padding:18px;border-radius:14px;background:rgba(127,90,240,0.05);border:1px solid rgba(127,90,240,0.15);'>
                <div style='font-size:22px;margin-bottom:8px;'>📊</div>
                <div style='font-size:11px;letter-spacing:2px;color:#7f5af0;font-family:Orbitron,monospace;margin-bottom:4px;'>STEP 2</div>
                <div style='font-size:13px;color:#475569;'>Dashboard unlocks with your data</div>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button("⚡  INITIALIZE SYSTEM", use_container_width=True):
            st.session_state.started = True
            st.session_state.page   = "Analyze"
            st.rerun()
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR  — uses st.button instead of
#  st.radio to eliminate keyboard-stuck bug
# ─────────────────────────────────────────────
# ── Top navigation bar (always visible, no sidebar needed) ──
has_data = len(st.session_state.history) > 0
menu = st.session_state.page

st.markdown("""
<style>
.topnav {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 0 24px 0;
}
.topnav-logo {
    font-family: 'Orbitron', monospace;
    font-size: 18px;
    font-weight: 900;
    background: linear-gradient(90deg,#00ffcc,#7f5af0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-right: 20px;
}
.topnav-sep { width: 1px; height: 28px; background: rgba(255,255,255,0.08); margin: 0 4px; }
.topnav-stat {
    font-size: 12px;
    color: #475569;
    margin-left: auto;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# Top nav with buttons
nav_cols = st.columns([2, 1, 1, 1, 3])
with nav_cols[0]:
    st.markdown("<div style='font-family:Orbitron,monospace;font-size:17px;font-weight:900;background:linear-gradient(90deg,#00ffcc,#7f5af0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding-top:6px;'>⚡ SentimentIQ</div>", unsafe_allow_html=True)
with nav_cols[1]:
    if st.button("✍️  Analyze", key="nav_analyze", use_container_width=True):
        st.session_state.page = "Analyze"
        st.rerun()
with nav_cols[2]:
    if st.button("📂  Dataset", key="nav_dataset", use_container_width=True):
        st.session_state.page = "Dataset"
        st.rerun()
with nav_cols[3]:
    if has_data:
        if st.button("📊  Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()
    else:
        st.markdown("<div style='padding:8px 12px;font-size:13px;color:#334155;border:1px solid rgba(255,255,255,0.05);border-radius:50px;text-align:center;'>📊 🔒</div>", unsafe_allow_html=True)
with nav_cols[4]:
    total = len(st.session_state.history)
    if total:
        pos = sum(1 for h in st.session_state.history if h["sentiment"] == "POSITIVE")
        neg = sum(1 for h in st.session_state.history if h["sentiment"] == "NEGATIVE")
        neu = total - pos - neg
        st.markdown(f"<div style='text-align:right;font-size:12px;color:#475569;padding-top:8px;'>🟢 {pos} &nbsp; 🔴 {neg} &nbsp; 🟣 {neu} &nbsp;·&nbsp; {total} total</div>", unsafe_allow_html=True)
    if st.button("↺ Reset", key="nav_reset"):
        st.session_state.history = []
        st.session_state.page = "Analyze"
        st.rerun()

st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(0,255,200,0.3),transparent);margin-bottom:24px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: ANALYZE
# ─────────────────────────────────────────────
if menu == "Analyze":
    st.markdown('<div class="section-header">Text Analysis</div>', unsafe_allow_html=True)
    st.markdown("### ✍️ Analyze Text")
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    if not has_data:
        st.markdown("""
        <div style='padding:14px 20px;border-radius:12px;background:rgba(0,255,200,0.04);
                    border:1px solid rgba(0,255,200,0.12);margin-bottom:24px;font-size:13px;color:#475569;'>
            👋 &nbsp;Welcome! Enter any text below and hit <b style='color:#00ffcc;'>RUN ANALYSIS</b>.
            The <b style='color:#a78bfa;'>Dashboard</b> will unlock automatically after your first result.
        </div>""", unsafe_allow_html=True)

    col_in, col_out = st.columns([3, 2], gap="large")

    with col_in:
        text = st.text_area(
            "Enter any text — tweet, review, comment, article…",
            height=200,
            placeholder="e.g. 'This product absolutely changed my life!'",
        )
        run = st.button("⚡  RUN ANALYSIS", use_container_width=True)

    with col_out:
        if run and text.strip():
            with st.spinner("Analyzing…"):
                model     = load_model()
                res       = model(text[:512])[0]
            sentiment = decode(res["label"])
            score     = res["score"]

            st.session_state.history.append({
                "text":       text[:80] + ("…" if len(text) > 80 else ""),
                "sentiment":  sentiment,
                "confidence": round(score * 100, 1),
            })

            st.markdown(f"""
            <div style='padding:28px;background:rgba(255,255,255,0.03);border-radius:16px;
                        border:1px solid rgba(255,255,255,0.07);'>
                <div style='font-size:10px;letter-spacing:3px;color:#475569;margin-bottom:12px;'>RESULT</div>
                <div class='result-badge {BADGE_CSS[sentiment]}'>{ICONS[sentiment]}  {sentiment}</div>
                {conf_bar(score, sentiment)}
            </div>""", unsafe_allow_html=True)

            if len(st.session_state.history) == 1:
                st.success("🎉 Dashboard is now unlocked! Find it in the sidebar.")

        elif run:
            st.warning("Please enter some text first.")
        else:
            st.markdown("""
            <div class='empty-state'>
                <span class='empty-icon'>🔍</span>
                <div class='empty-text'>Awaiting Input</div>
                <div class='empty-sub'>Your result will appear here</div>
            </div>""", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Recent History</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-8:]):
            css = CHIP_CSS[item["sentiment"]]
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:12px;padding:10px 4px;
                        border-bottom:1px solid rgba(255,255,255,0.04);'>
                <span class='history-chip {css}'>{item["sentiment"]}</span>
                <span style='font-size:13px;color:#475569;flex:1;overflow:hidden;
                             text-overflow:ellipsis;white-space:nowrap;'>{item["text"]}</span>
                <span style='font-size:12px;color:#334155;'>{item["confidence"]}%</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: DATASET
# ─────────────────────────────────────────────
elif menu == "Dataset":
    st.markdown('<div class="section-header">Bulk Analysis</div>', unsafe_allow_html=True)
    st.markdown("### 📂 Upload Dataset")
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    file = st.file_uploader("Drop a CSV file here", type=["csv"])

    if file:
        df = pd.read_csv(file)
        col_s, col_p = st.columns([2, 1], gap="large")

        with col_s:
            text_col = st.selectbox("Select the text column to analyze", df.columns)
            limit    = st.slider("Rows to analyze", 10, min(200, len(df)), 50)
            run_bulk = st.button("⚡  ANALYZE DATASET", use_container_width=True)

        with col_p:
            st.markdown(f"""
            <div class='stat-card' style='text-align:left;padding:20px;'>
                <div class='stat-label'>Dataset Info</div>
                <div style='font-size:14px;line-height:2.2;color:#64748b;'>
                    <b style='color:#e2e8f0;'>{len(df)}</b> rows &nbsp;·&nbsp;
                    <b style='color:#e2e8f0;'>{len(df.columns)}</b> columns
                </div>
            </div>""", unsafe_allow_html=True)

        if run_bulk:
            model   = load_model()
            rows    = df[text_col].dropna().head(limit).tolist()
            results, scores = [], []
            bar = st.progress(0, text="Analyzing…")
            for i, t in enumerate(rows):
                r = model(str(t)[:512])[0]
                results.append(decode(r["label"]))
                scores.append(round(r["score"] * 100, 1))
                bar.progress((i + 1) / len(rows), text=f"Row {i+1} / {len(rows)}")
            bar.empty()

            out_df = df.head(limit).copy()
            out_df["sentiment"]  = results
            out_df["confidence"] = scores

            for sent, conf, txt in zip(results, scores, rows):
                st.session_state.history.append({
                    "text": str(txt)[:80], "sentiment": sent, "confidence": conf,
                })

            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Results Preview</div>', unsafe_allow_html=True)
            st.dataframe(out_df, use_container_width=True, height=320)

            counts = pd.Series(results).value_counts()
            fig = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker_color=[COLORS.get(l, "#7f5af0") for l in counts.index],
                marker_line_width=0,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748b", family="Syne"),
                margin=dict(l=20, r=20, t=20, b=20), height=220,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig, use_container_width=True)
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("💾  DOWNLOAD RESULTS CSV", csv_bytes,
                               "sentiment_results.csv", "text/csv")
    else:
        st.markdown("""
        <div class='empty-state'>
            <span class='empty-icon'>📂</span>
            <div class='empty-text'>No File Loaded</div>
            <div class='empty-sub'>Upload a CSV to begin bulk analysis</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: DASHBOARD
# ─────────────────────────────────────────────
elif menu == "Dashboard":
    if not has_data:
        st.markdown("""
        <div class='locked-banner'>
            <span class='locked-icon'>🔒</span>
            <div class='locked-title'>Dashboard Locked</div>
            <div class='locked-sub'>Go to <b>Analyze</b> and run at least one prediction to unlock.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Dashboard")
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    history    = st.session_state.history
    sentiments = [h["sentiment"] for h in history]
    total      = len(sentiments)
    counts_map = {s: sentiments.count(s) for s in ["POSITIVE", "NEUTRAL", "NEGATIVE"]}

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    for col, (label, val, css) in zip(
        [c1, c2, c3, c4],
        [("TOTAL", total, "stat-pos"),
         ("POSITIVE", counts_map["POSITIVE"], "stat-pos"),
         ("NEGATIVE", counts_map["NEGATIVE"], "stat-neg"),
         ("NEUTRAL",  counts_map["NEUTRAL"],  "stat-neu")],
    ):
        with col:
            st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-label'>{label}</div>
                <div class='stat-value {css}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown('<div class="section-header">Distribution</div>', unsafe_allow_html=True)
        labels = [k for k, v in counts_map.items() if v > 0]
        values = [counts_map[k] for k in labels]
        colors = [COLORS[k] for k in labels]
        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.65,
            marker=dict(colors=colors, line=dict(color="#050810", width=3)),
            textfont=dict(family="Syne", color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{total}</b><br>TOTAL", x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#e2e8f0", family="Orbitron"),
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color="#64748b", family="Syne"), bgcolor="rgba(0,0,0,0)",
                        orientation="h", x=0.5, xanchor="center", y=-0.05),
            margin=dict(l=20, r=20, t=10, b=20), height=300,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with ch2:
        st.markdown('<div class="section-header">Timeline</div>', unsafe_allow_html=True)
        df_hist = pd.DataFrame(history)
        df_hist["entry_num"] = range(len(df_hist))
        fig_tl = go.Figure()
        for sent in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
            subset = df_hist[df_hist["sentiment"] == sent]
            if subset.empty:
                continue
            fig_tl.add_trace(go.Scatter(
                x=subset["entry_num"], y=[1] * len(subset),
                mode="markers", name=sent,
                marker=dict(color=COLORS[sent], size=14, opacity=0.85,
                            line=dict(color=COLORS[sent], width=1)),
                hovertemplate=f"<b>{sent}</b><br>Entry #%{{x}}<extra></extra>",
            ))
        fig_tl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b", family="Syne"),
            margin=dict(l=20, r=20, t=10, b=20), height=300,
            xaxis=dict(title="Entry #", showgrid=False, color="#334155"),
            yaxis=dict(visible=False),
            legend=dict(font=dict(color="#64748b"), bgcolor="rgba(0,0,0,0)",
                        orientation="h", x=0.5, xanchor="center", y=-0.12),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    # Confidence trend
    conf_items = [h for h in history if "confidence" in h]
    if conf_items:
        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Confidence Trend</div>', unsafe_allow_html=True)
        conf_df = pd.DataFrame([
            {"i": i, "confidence": h["confidence"], "sentiment": h["sentiment"]}
            for i, h in enumerate(conf_items)
        ])
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=conf_df["i"], y=conf_df["confidence"],
            mode="lines+markers",
            line=dict(color="#00ffcc", width=2),
            marker=dict(color=[COLORS[s] for s in conf_df["sentiment"]], size=8,
                        line=dict(color="#050810", width=1)),
            fill="tozeroy", fillcolor="rgba(0,255,200,0.04)",
            hovertemplate="Entry #%{x}<br>Confidence: %{y}%<extra></extra>",
        ))
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b", family="Syne"),
            margin=dict(l=20, r=20, t=10, b=20), height=220,
            xaxis=dict(showgrid=False, color="#334155"),
            yaxis=dict(title="Confidence %", showgrid=True,
                       gridcolor="rgba(255,255,255,0.04)", range=[0, 105]),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    csv_export = pd.DataFrame(history).to_csv(index=False).encode("utf-8")
    st.download_button("💾  EXPORT SESSION HISTORY", csv_export,
                       "sentimentiq_session.csv", "text/csv")
