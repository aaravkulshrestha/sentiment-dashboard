import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import re

# Load model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🚀AI Sentiment Analytics Dashboard")

# ---------------- INPUT SECTION ----------------
st.header("✍️ Analyze Single Review")

user_input = st.text_area("Enter your review:")

if st.button("Analyze Review"):
    if user_input.strip():
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

        # Show result
        if sentiment == "POSITIVE":
            st.success(f"{sentiment} ({score:.2f})")
        elif sentiment == "NEGATIVE":
            st.error(f"{sentiment} ({score:.2f})")
        else:
            st.warning(f"{sentiment} ({score:.2f})")

# ---------------- FILE UPLOAD ----------------
st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    text_col = st.selectbox("Select Text Column", df.columns)

    if st.button("Analyze Dataset"):
        results = []

        for text in df[text_col].dropna().head(100):  # limit for speed
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

    # Bar Chart
    st.subheader("Bar Chart")
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    st.pyplot(fig)

    # Pie Chart
    st.subheader("Pie Chart")
    fig2, ax2 = plt.subplots()
    ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    st.pyplot(fig2)

    # Trend over time
    st.subheader("📈 Trend Over Time")
    df_hist['index'] = range(len(df_hist))
    trend = df_hist.groupby(['index', 'sentiment']).size().unstack().fillna(0)
    st.line_chart(trend)

    # ---------------- KEYWORD EXTRACTION ----------------
    st.subheader("🔍 Common Keywords (Negative Reviews)")

    negative_text = " ".join(
        df_hist[df_hist['sentiment'] == "NEGATIVE"]['text']
    )

    words = re.findall(r'\b\w+\b', negative_text.lower())
    common_words = Counter(words).most_common(10)

    if common_words:
        words_df = pd.DataFrame(common_words, columns=["Word", "Count"])
        st.dataframe(words_df)

    # History table
    st.subheader("🧾 Full History")
    st.dataframe(df_hist)

# Clear
if st.button("🗑️ Reset All Data"):
    st.session_state.history = []
    st.rerun()