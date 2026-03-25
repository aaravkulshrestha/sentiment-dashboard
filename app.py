import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px
import time
import io

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
#  GLOBAL CSS  — dark-neon cyberpunk theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@400;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    color: #e2e8f0;
}

/* Streamlit main background */
.stApp {
    background: #050810;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,255,200,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(127,90,240,0.08) 0%, transparent 55%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 59px,
            rgba(0,255,200,0.03) 60px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 59px,
            rgba(0,255,200,0.03) 60px
        );
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(5,8,16,0.95) !important;
    border-right: 1px solid rgba(0,255,200,0.15) !important;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }

/* ── Stat card ── */
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.stat-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 16px;
    padding: 1px;
    background: linear-gradient(135deg, transparent 40%, rgba(0,255,200,0.2));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask-composite: exclude;
    pointer-events: none;
}
.stat-card:hover { border-color: rgba(0,255,200,0.3); }

.stat-label {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 10px;
}
.stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 42px;
    font-weight: 900;
    line-height: 1;
}
.stat-pos { color: #00ffcc; text-shadow: 0 0 20px rgba(0,255,200,0.5); }
.stat-neg { color: #ff4d6d; text-shadow: 0 0 20px rgba(255,77,109,0.5); }
.stat-neu { color: #7f5af0; text-shadow: 0 0 20px rgba(127,90,240,0.5); }

/* ── Result badge ── */
.result-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-family: 'Orbitron', monospace;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 2px;
    margin: 12px 0;
}
.badge-pos {
    background: rgba(0,255,200,0.1);
    border: 1px solid rgba(0,255,200,0.6);
    color: #00ffcc;
    box-shadow: 0 0 20px rgba(0,255,200,0.2);
}
.badge-neg {
    background: rgba(255,77,109,0.1);
    border: 1px solid rgba(255,77,109,0.6);
    color: #ff4d6d;
    box-shadow: 0 0 20px rgba(255,77,109,0.2);
}
.badge-neu {
    background: rgba(127,90,240,0.1);
    border: 1px solid rgba(127,90,240,0.6);
    color: #a78bfa;
    box-shadow: 0 0 20px rgba(127,90,240,0.2);
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    margin: 16px 0;
    background: rgba(255,255,255,0.05);
    border-radius: 50px;
    height: 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 1s ease;
}

/* ── History chip ── */
.history-chip {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px;
    letter-spacing: 1px;
}
.chip-pos { background: rgba(0,255,200,0.15); color: #00ffcc; border: 1px solid rgba(0,255,200,0.3); }
.chip-neg { background: rgba(255,77,109,0.15); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.3); }
.chip-neu { background: rgba(127,90,240,0.15); color: #a78bfa; border: 1px solid rgba(127,90,240,0.3); }

/* ── Section header ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #00ffcc;
    margin-bottom: 4px;
}

/* ── Divider ── */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,200,0.4), transparent);
    margin: 24px 0;
}

/* ── Input textarea ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,255,200,0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    transition: border-color 0.3s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(0,255,200,0.5) !important;
    box-shadow: 0 0 20px rgba(0,255,200,0.08) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,255,200,0.15), rgba(127,90,240,0.15)) !important;
    border: 1px solid rgba(0,255,200,0.4) !important;
    color: #00ffcc !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    padding: 10px 28px !important;
    border-radius: 50px !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,255,200,0.25), rgba(127,90,240,0.25)) !important;
    box-shadow: 0 0 25px rgba(0,255,200,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Sidebar radio ── */
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 14px;
    transition: background 0.2s;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,255,200,0.07);
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,255,200,0.25) !important;
    border-radius: 12px !important;
    background: rgba(0,255,200,0.02) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,255,200,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Welcome screen ── */
.welcome-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(36px, 6vw, 72px);
    font-weight: 900;
    background: linear-gradient(90deg, #00ffcc 0%, #7f5af0 50%, #00ffcc 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    line-height: 1.1;
    margin-bottom: 16px;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.welcome-sub {
    font-size: 18px;
    color: #64748b;
    letter-spacing: 1px;
    margin-bottom: 40px;
}
.feature-pill {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 50px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    font-size: 13px;
    margin: 4px;
    color: #94a3b8;
}

/* ── Plotly chart background ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ── Sidebar logo ── */
.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 20px;
    font-weight: 900;
    background: linear-gradient(90deg, #00ffcc, #7f5af0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 8px 0 20px 0;
    letter-spacing: 1px;
}
.sidebar-version {
    font-size: 10px;
    letter-spacing: 3px;
    color: #334155;
    text-transform: uppercase;
    margin-top: -18px;
    margin-bottom: 24px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #334155;
}
.empty-icon {
    font-size: 56px;
    margin-bottom: 16px;
    opacity: 0.5;
}
.empty-text {
    font-family: 'Orbitron', monospace;
    font-size: 14px;
    letter-spacing: 3px;
    text-transform: uppercase;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(127,90,240,0.1) !important;
    border: 1px solid rgba(127,90,240,0.4) !important;
    color: #a78bfa !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    border-radius: 50px !important;
}
.stDownloadButton > button:hover {
    background: rgba(127,90,240,0.2) !important;
    box-shadow: 0 0 20px rgba(127,90,240,0.2) !important;
}
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
      <div class='conf-bar-fill' style='width:{pct}%;background:linear-gradient(90deg,{color}88,{color});'></div>
    </div>
    <div style='font-size:12px;color:#64748b;letter-spacing:2px;text-align:right;margin-top:4px;'>
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
if "started"  not in st.session_state: st.session_state.started  = False
if "history"  not in st.session_state: st.session_state.history  = []   # list of dicts
if "page"     not in st.session_state: st.session_state.page     = "Dashboard"


# ─────────────────────────────────────────────
#  WELCOME SCREEN
# ─────────────────────────────────────────────
if not st.session_state.started:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_c = st.columns([1, 3, 1])[1]
    with col_c:
        st.markdown('<div class="welcome-title">Sentiment<br>Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="welcome-sub">Decode emotion. Visualize truth. In real-time.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='margin-bottom:36px;'>
            <span class='feature-pill'>⚡ RoBERTa Model</span>
            <span class='feature-pill'>📊 Plotly Charts</span>
            <span class='feature-pill'>📂 Bulk CSV Analysis</span>
            <span class='feature-pill'>💾 Export Results</span>
        </div>""", unsafe_allow_html=True)
        if st.button("⚡  INITIALIZE SYSTEM"):
            st.session_state.started = True
            st.rerun()
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">SentimentIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-version">v2.0 · RoBERTa</div>', unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    menu = st.radio(
        "", ["Dashboard", "Analyze", "Dataset"],
        label_visibility="collapsed",
        index=["Dashboard", "Analyze", "Dataset"].index(st.session_state.page),
    )
    st.session_state.page = menu

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # Mini stats in sidebar
    total = len(st.session_state.history)
    if total:
        pos = sum(1 for h in st.session_state.history if h["sentiment"] == "POSITIVE")
        neg = sum(1 for h in st.session_state.history if h["sentiment"] == "NEGATIVE")
        neu = total - pos - neg

        st.markdown('<div class="section-header">Session Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='padding:12px 0;font-size:13px;line-height:2;'>
            <span style='color:#00ffcc;'>●</span> Positive &nbsp;<b style='color:#00ffcc;float:right'>{pos}</b><br>
            <span style='color:#ff4d6d;'>●</span> Negative &nbsp;<b style='color:#ff4d6d;float:right'>{neg}</b><br>
            <span style='color:#a78bfa;'>●</span> Neutral &nbsp;<b style='color:#a78bfa;float:right'>{neu}</b>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    if st.button("↺  RESET SESSION"):
        st.session_state.history = []
        st.rerun()


# ─────────────────────────────────────────────
#  PAGE: ANALYZE
# ─────────────────────────────────────────────
if menu == "Analyze":
    st.markdown('<div class="section-header">Text Analysis</div>', unsafe_allow_html=True)
    st.markdown("### ✍️ Analyze Text")
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([3, 2], gap="large")

    with col_in:
        text = st.text_area(
            "Enter any text — tweet, review, comment, article excerpt…",
            height=180,
            placeholder="Type or paste your text here…",
            label_visibility="visible",
        )
        run = st.button("⚡  RUN ANALYSIS")

    with col_out:
        if run and text.strip():
            with st.spinner(""):
                model = load_model()
                res   = model(text[:512])[0]

            sentiment = decode(res["label"])
            score     = res["score"]

            st.session_state.history.append({
                "text":      text[:80] + ("…" if len(text) > 80 else ""),
                "sentiment": sentiment,
                "confidence": round(score * 100, 1),
            })

            st.markdown(f"""
            <div style='padding:24px;background:rgba(255,255,255,0.03);border-radius:16px;
                        border:1px solid rgba(255,255,255,0.07);'>
                <div style='font-size:11px;letter-spacing:3px;color:#64748b;margin-bottom:10px;'>RESULT</div>
                <div class='result-badge {BADGE_CSS[sentiment]}'>{ICONS[sentiment]} {sentiment}</div>
                {conf_bar(score, sentiment)}
            </div>""", unsafe_allow_html=True)

        elif run:
            st.warning("Please enter some text first.")
        else:
            st.markdown("""
            <div class='empty-state'>
                <div class='empty-icon'>🔍</div>
                <div class='empty-text'>Awaiting Input</div>
            </div>""", unsafe_allow_html=True)

    # Recent history
    if st.session_state.history:
        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Recent History</div>', unsafe_allow_html=True)
        recent = st.session_state.history[-8:][::-1]
        for item in recent:
            css = CHIP_CSS[item["sentiment"]]
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:12px;padding:10px 0;
                        border-bottom:1px solid rgba(255,255,255,0.04);'>
                <span class='history-chip {css}'>{item['sentiment']}</span>
                <span style='font-size:13px;color:#64748b;flex:1;overflow:hidden;
                             text-overflow:ellipsis;white-space:nowrap;'>{item['text']}</span>
                <span style='font-size:12px;color:#334155;'>{item['confidence']}%</span>
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
            run_bulk = st.button("⚡  ANALYZE DATASET")

        with col_p:
            st.markdown(f"""
            <div class='stat-card' style='text-align:left;'>
                <div class='stat-label'>Dataset Info</div>
                <div style='font-size:14px;line-height:2;color:#94a3b8;'>
                    <b style='color:#e2e8f0;'>{len(df)}</b> rows &nbsp;·&nbsp;
                    <b style='color:#e2e8f0;'>{len(df.columns)}</b> columns
                </div>
            </div>""", unsafe_allow_html=True)

        if run_bulk:
            model = load_model()
            rows  = df[text_col].dropna().head(limit).tolist()
            results, scores = [], []

            progress = st.progress(0, text="Analyzing…")
            for i, t in enumerate(rows):
                r = model(str(t)[:512])[0]
                results.append(decode(r["label"]))
                scores.append(round(r["score"] * 100, 1))
                progress.progress((i + 1) / len(rows), text=f"Analyzing row {i+1}/{len(rows)}…")

            progress.empty()

            out_df = df.head(limit).copy()
            out_df["sentiment"]  = results
            out_df["confidence"] = scores

            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Results Preview</div>', unsafe_allow_html=True)
            st.dataframe(
                out_df.style.applymap(
                    lambda v: f"color: {COLORS.get(v, '#e2e8f0')}",
                    subset=["sentiment"]
                ),
                use_container_width=True,
                height=320,
            )

            # Distribution mini-chart
            counts = pd.Series(results).value_counts()
            fig = go.Figure(go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=[COLORS.get(l, "#7f5af0") for l in counts.index],
                marker_line_width=0,
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Syne"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=220,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Export
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾  DOWNLOAD RESULTS CSV",
                data=csv_bytes,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )

    else:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-icon'>📂</div>
            <div class='empty-text'>No File Loaded</div>
            <div style='font-size:13px;color:#1e293b;margin-top:8px;'>
                Upload a CSV to begin bulk analysis
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: DASHBOARD
# ─────────────────────────────────────────────
elif menu == "Dashboard":
    st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Dashboard")
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-icon'>📊</div>
            <div class='empty-text'>No Data Yet</div>
            <div style='font-size:13px;color:#1e293b;margin-top:8px;'>
                Go to <b>Analyze</b> and run some predictions first
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        history   = st.session_state.history
        sentiments = [h["sentiment"] for h in history]
        counts_map = {s: sentiments.count(s) for s in ["POSITIVE", "NEUTRAL", "NEGATIVE"]}
        total      = len(sentiments)

        # ── KPI cards ──
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        card_data = [
            ("TOTAL ANALYZED", total, "stat-pos"),
            ("POSITIVE",       counts_map["POSITIVE"], "stat-pos"),
            ("NEGATIVE",       counts_map["NEGATIVE"], "stat-neg"),
            ("NEUTRAL",        counts_map["NEUTRAL"],  "stat-neu"),
        ]
        for col, (label, val, css) in zip([c1, c2, c3, c4], card_data):
            with col:
                st.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-label'>{label}</div>
                    <div class='stat-value {css}'>{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts row ──
        ch1, ch2 = st.columns(2, gap="large")

        # Donut chart
        with ch1:
            st.markdown('<div class="section-header">Distribution</div>', unsafe_allow_html=True)
            labels = [k for k, v in counts_map.items() if v > 0]
            values = [counts_map[k] for k in labels]
            colors = [COLORS[k] for k in labels]

            fig_donut = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(colors=colors, line=dict(color="#050810", width=3)),
                textfont=dict(family="Syne", color="#e2e8f0"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            ))
            fig_donut.add_annotation(
                text=f"<b>{total}</b><br><span style='font-size:10px'>TOTAL</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=22, color="#e2e8f0", family="Orbitron"),
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(
                    font=dict(color="#94a3b8", family="Syne"),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="h",
                    x=0.5, xanchor="center", y=-0.05,
                ),
                margin=dict(l=20, r=20, t=10, b=20),
                height=300,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Trend / bar chart
        with ch2:
            st.markdown('<div class="section-header">Timeline</div>', unsafe_allow_html=True)
            df_hist = pd.DataFrame(history)
            df_hist.index.name = "entry"
            df_hist = df_hist.reset_index()

            fig_bar = go.Figure()
            for sent in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
                subset = df_hist[df_hist["sentiment"] == sent]
                fig_bar.add_trace(go.Scatter(
                    x=subset["index"],
                    y=[1] * len(subset),
                    mode="markers",
                    name=sent,
                    marker=dict(
                        color=COLORS[sent],
                        size=14,
                        symbol="circle",
                        line=dict(color=COLORS[sent], width=1),
                        opacity=0.85,
                    ),
                    hovertemplate=f"<b>{sent}</b><br>Entry #%{{x}}<extra></extra>",
                ))

            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Syne"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=300,
                xaxis=dict(
                    title="Entry #",
                    showgrid=False,
                    color="#334155",
                ),
                yaxis=dict(visible=False),
                legend=dict(
                    font=dict(color="#94a3b8"),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="h",
                    x=0.5, xanchor="center", y=-0.12,
                ),
                showlegend=True,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Running average confidence ──
        if any("confidence" in h for h in history):
            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Confidence Trend</div>', unsafe_allow_html=True)

            conf_df = pd.DataFrame([
                {"index": i, "confidence": h.get("confidence", 0), "sentiment": h["sentiment"]}
                for i, h in enumerate(history)
                if "confidence" in h
            ])
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=conf_df["index"],
                y=conf_df["confidence"],
                mode="lines+markers",
                line=dict(color="#00ffcc", width=2),
                marker=dict(
                    color=[COLORS[s] for s in conf_df["sentiment"]],
                    size=8,
                    line=dict(color="#050810", width=1),
                ),
                fill="tozeroy",
                fillcolor="rgba(0,255,200,0.05)",
                hovertemplate="Entry #%{x}<br>Confidence: %{y}%<extra></extra>",
            ))
            fig_line.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Syne"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=220,
                xaxis=dict(showgrid=False, color="#334155"),
                yaxis=dict(
                    title="Confidence %",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.04)",
                    range=[0, 105],
                ),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # ── Export session history ──
        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
        export_df  = pd.DataFrame(history)
        csv_export = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾  EXPORT SESSION HISTORY",
            data=csv_export,
            file_name="sentimentiq_session.csv",
            mime="text/csv",
        )
