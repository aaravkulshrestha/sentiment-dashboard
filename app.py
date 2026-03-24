import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import re

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Dashboard", layout="wide")

# ---------------- STYLES ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0e1117, #1a1f2b);
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}
.big-title {
    font-size: 50px;
    font-weight: bold;
    background: linear-gradient(90deg,#00ffcc,#7f5af0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ---------------- WELCOME ----------------
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown('<div class="big-title">AI Sentiment Intelligence</div>', unsafe_allow_html=True)
    st.write("### Analyze emotions. Visualize insights. Instantly.")

    st.balloons()

    if st.button("🚀 Get Started"):
        st.session_state.started = True
        st.rerun()

    st.stop()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚡ AI System")
menu = st.sidebar.radio("Menu", ["Dashboard", "Analyze", "Dataset"])

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- ANALYZE ----------------
if menu == "Analyze":
    st.title("✍️ Analyze Text")

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
    st.title("📂 Upload Dataset")

    file = st.file_uploader("CSV")

    if file:
        df = pd.read_csv(file)
        col = st.selectbox("Text column", df.columns)

        if st.button("Run Analysis"):
            results = []
            for t in df[col].dropna().head(50):
                r = model(str(t))[0]
                results.append(r['label'])

            df["sentiment"] = results + [""]*(len(df)-len(results))
            st.dataframe(df.head())

# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":
    st.title("📊 Dashboard")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history, columns=["sentiment"])
        counts = df['sentiment'].value_counts()

        # CARDS
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="card">Positive<br><h2>{}</h2></div>'.format(counts.get("POSITIVE",0)), unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">Negative<br><h2>{}</h2></div>'.format(counts.get("NEGATIVE",0)), unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="card">Neutral<br><h2>{}</h2></div>'.format(counts.get("NEUTRAL",0)), unsafe_allow_html=True)

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
