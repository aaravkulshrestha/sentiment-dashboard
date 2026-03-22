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
    color: #00ffcc;
}
.stButton>button {
    background-color: #00ffcc;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
textarea {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

sentiment_model = load_model()

# ---------------- SESSION STATE ----------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------- TITLE ----------------
st.title("🚀 AI Sentiment Analytics Dashboard")
st.markdown("### 🧠 Real-Time AI Sentiment Intelligence")

# ---------------- INPUT SECTION ----------------
st.header("✍️ Analyze Single Review")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Enter your review:")

with col2:
    analyze_btn = st.button("Analyze Review")

if analyze_btn and user_input.strip():
    result = sentiment_model(user_input, truncation=True)[0]
    label = result['label']
    score = result['score']

    # Convert labels
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

    # Output styling
    if sentiment == "POSITIVE":
        st.success(f"{sentiment}")
    elif sentiment == "NEGATIVE":
        st.error(f"{sentiment}")
    else:
        st.warning(f"{sentiment}")

    st.progress(float(score))
    st.write(f"Confidence Score: {score:.2f}")

# ---------------- FILE UPLOAD ----------------
st.header("📂 Upload Dataset")

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

        st.session_state.history += [
            {"text": t, "sentiment": s, "confidence": 1}
            for t, s in zip(df[text_col].dropna().head(100), results)
        ]

# ---------------- ANALYTICS ----------------
st.header("📊 Analytics Dashboard")

if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    counts = df_hist['sentiment'].value_counts()

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", counts.get("POSITIVE", 0))
    col2.metric("Negative", counts.get("NEGATIVE", 0))
    col3.metric("Neutral", counts.get("NEUTRAL", 0))

    # Charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Bar Chart")
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        st.pyplot(fig)

    with col2:
        st.subheader("🥧 Pie Chart")
        fig2, ax2 = plt.subplots()
        ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        st.pyplot(fig2)

    # Trend
    st.subheader("📈 Trend Over Time")
    df_hist['index'] = range(len(df_hist))
    trend = df_hist.groupby(['index', 'sentiment']).size().unstack().fillna(0)
    st.line_chart(trend)

    # ---------------- KEYWORDS ----------------
    st.subheader("🔍 Common Keywords (Negative Reviews)")

    negative_text = " ".join(
        df_hist[df_hist['sentiment'] == "NEGATIVE"]['text']
    )

    words = re.findall(r'\b\w+\b', negative_text.lower())
    common_words = Counter(words).most_common(10)

    if common_words:
        words_df = pd.DataFrame(common_words, columns=["Word", "Count"])
        st.dataframe(words_df)

    # History
    st.subheader("🧾 Full History")
    st.dataframe(df_hist)

# ---------------- RESET ----------------
if st.button("🗑️ Reset All Data"):
    st.session_state.history = []
    st.rerun()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ by Aarav Kulshrestha | AI Capstone Project")
