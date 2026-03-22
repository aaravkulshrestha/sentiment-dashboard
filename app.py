import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #ffffff;
}
.stButton>button {
    background: linear-gradient(90deg, #7f5af0, #00ffcc);
    color: black;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
.sidebar .sidebar-content {
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

# ---------------- WELCOME SCREEN ----------------
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:

    st.markdown("""
    <style>
    .center-box {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 90vh;
        text-align: center;
    }
    .big-title {
        font-size: 60px;
        font-weight: bold;
        background: linear-gradient(90deg, #00ffcc, #7f5af0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 20px;
        color: #aaa;
        margin-top: 10px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="center-box">
        <div class="big-title">AI Sentiment Intelligence</div>
        <div class="subtitle">
            Analyze emotions, uncover insights, and visualize data in real-time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.balloons()

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Get Started"):
            st.session_state.started = True
            st.rerun()

    st.stop()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

sentiment_model = load_model()

# ---------------- SIDEBAR (PLUTUS STYLE) ----------------
st.sidebar.title("⚡ AI Dashboard")
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Analyze", "Dataset", "Settings"]
)

# ---------------- SESSION ----------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------- ANALYZE ----------------
if menu == "Analyze":
    st.title("✍️ Analyze Review")

    user_input = st.text_area("Enter your review:")

    if st.button("Analyze"):
        if user_input.strip():
            result = sentiment_model(user_input, truncation=True)[0]
            label = result['label']
            score = result['score']

            if label == "LABEL_0":
                sentiment = "NEGATIVE"
            elif label == "LABEL_1":
                sentiment = "NEUTRAL"
            else:
                sentiment = "POSITIVE"

            st.session_state.history.append({
                "text": user_input,
                "sentiment": sentiment,
                "confidence": score
            })

            if sentiment == "POSITIVE":
                st.success(f"{sentiment}")
            elif sentiment == "NEGATIVE":
                st.error(f"{sentiment}")
            else:
                st.warning(f"{sentiment}")

            st.progress(float(score))
            st.write(f"Confidence: {score:.2f}")

# ---------------- DATASET ----------------
elif menu == "Dataset":
    st.title("📂 Dataset Analysis")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_col = st.selectbox("Select Text Column", df.columns)

        if st.button("Analyze Dataset"):
            results = []

            for text in df[text_col].dropna().head(100):
                result = sentiment_model(str(text), truncation=True)[0]
                label = result['label']

                if label == "LABEL_0":
                    sentiment = "NEGATIVE"
                elif label == "LABEL_1":
                    sentiment = "NEUTRAL"
                else:
                    sentiment = "POSITIVE"

                results.append(sentiment)

            df['sentiment'] = results + [""]*(len(df)-len(results))
            st.dataframe(df.head())

# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":
    st.title("📊 Dashboard")

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        counts = df_hist['sentiment'].value_counts()

        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", counts.get("POSITIVE", 0))
        col2.metric("Negative", counts.get("NEGATIVE", 0))
        col3.metric("Neutral", counts.get("NEUTRAL", 0))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bar Chart")
            fig, ax = plt.subplots()
            ax.bar(counts.index, counts.values)
            st.pyplot(fig)

        with col2:
            st.subheader("Pie Chart")
            fig2, ax2 = plt.subplots()
            ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
            st.pyplot(fig2)

        st.subheader("Trend")
        df_hist['index'] = range(len(df_hist))
        trend = df_hist.groupby(['index', 'sentiment']).size().unstack().fillna(0)
        st.line_chart(trend)

# ---------------- SETTINGS ----------------
elif menu == "Settings":
    st.title("⚙️ Settings")

    st.write("User settings coming soon...")

# ---------------- RESET ----------------
if st.sidebar.button("🗑️ Reset Data"):
    st.session_state.history = []
    st.rerun()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ by Aarav Kulshrestha | AI Project")
