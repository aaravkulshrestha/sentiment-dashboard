import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Dashboard", layout="wide")

# ---------------- STYLES ----------------
st.markdown("""
<style>

/* GLOBAL */
html, body, [class*="css"] {
    background-color: #0a0a0a;
    color: white;
    font-family: 'Inter', sans-serif;
}

/* TITLE */
.main-title {
    font-size: 48px;
    font-weight: 700;
    letter-spacing: -1px;
}

/* SUBTEXT */
.sub-text {
    color: #aaa;
    font-size: 18px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 16px;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    border: 1px solid rgba(255,255,255,0.2);
}

/* BUTTON */
.stButton>button {
    background: white;
    color: black;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton>button:hover {
    background: #ddd;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #050505;
    border-right: 1px solid #111;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "started" not in st.session_state:
    st.session_state.started = False

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- WELCOME SCREEN ----------------
if not st.session_state.started:

    st.markdown('<div class="main-title">AI Sentiment Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Understand emotions. Analyze data. Make decisions.</div>', unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        if st.button("Get Started"):
            st.session_state.started = True
            st.rerun()

    st.stop()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚡ AI Panel")
menu = st.sidebar.radio("", ["Dashboard", "Analyze", "Dataset"])

# ---------------- ANALYZE ----------------
if menu == "Analyze":

    st.markdown('<div class="main-title">Analyze Text</div>', unsafe_allow_html=True)

    text = st.text_area("Enter text")

    if st.button("Analyze"):
        res = model(text)[0]

        if res['label'] == "LABEL_0":
            sentiment = "NEGATIVE"
        elif res['label'] == "LABEL_1":
            sentiment = "NEUTRAL"
        else:
            sentiment = "POSITIVE"

        st.session_state.history.append(sentiment)

        st.markdown(f"### Result: {sentiment}")
        st.progress(res['score'])

# ---------------- DATASET ----------------
elif menu == "Dataset":

    st.markdown('<div class="main-title">Dataset Analysis</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        col = st.selectbox("Select column", df.columns)

        if st.button("Analyze Dataset"):
            results = []
            for t in df[col].dropna().head(50):
                r = model(str(t))[0]
                results.append(r['label'])

            df["sentiment"] = results + [""]*(len(df)-len(results))
            st.dataframe(df.head())

# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":

    st.markdown('<div class="main-title">Dashboard</div>', unsafe_allow_html=True)

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history, columns=["sentiment"])
        counts = df['sentiment'].value_counts()

        # CARDS
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f'<div class="card">Positive<br><h2>{counts.get("POSITIVE",0)}</h2></div>', unsafe_allow_html=True)

        with c2:
            st.markdown(f'<div class="card">Negative<br><h2>{counts.get("NEGATIVE",0)}</h2></div>', unsafe_allow_html=True)

        with c3:
            st.markdown(f'<div class="card">Neutral<br><h2>{counts.get("NEUTRAL",0)}</h2></div>', unsafe_allow_html=True)

        # CHARTS
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.bar(counts.index, counts.values)
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            st.pyplot(fig2)

# ---------------- RESET ----------------
if st.sidebar.button("Reset"):
    st.session_state.history = []
    st.rerun()
