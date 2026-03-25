import streamlit as st
import pandas as pd
import plotly.express as px

# Define colors for sentiment visualization
COLORS = {
    "Positive": "#28a745",
    "Neutral": "#0d6efd",
    "Negative": "#dc3545"
}

# Mock sentiment analysis function
def predict_sentiment(text):
    # For demonstration, return a random sentiment
    import random
    return random.choice(["Positive", "Neutral", "Negative"])

# Session state to store analysis history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Dashboard", "Analyze", "Bulk Analysis"])

# Page routing
if page == "Dashboard":
    st.title("Sentiment Analysis Dashboard")
    st.markdown("### Analysis History")
    history_df = pd.DataFrame(st.session_state.history)
    if not history_df.empty:
        fig = px.bar(
            history_df,
            x="Sentiment",
            y="Text",
            color="Sentiment",
            color_discrete_map=COLORS,
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig)
    else:
        st.info("No analysis history yet. Start by analyzing a text or uploading a file.")

elif page == "Analyze":
    st.title("Sentiment Analysis")
    text = st.text_area("Enter text for analysis", "")
    if text:
        sentiment = predict_sentiment(text)
        st.session_state.history.append({"Text": text, "Sentiment": sentiment})
        st.success(f"Sentiment: **{sentiment}**")
        st.markdown(f"### Analysis Result\n\n{sentiment}")
        st.markdown("### History")
        history_df = pd.DataFrame(st.session_state.history)
        fig = px.bar(
            history_df,
            x="Sentiment",
            y="Text",
            color="Sentiment",
            color_discrete_map=COLORS,
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig)

elif page == "Bulk Analysis":
    st.title("Bulk Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload a file (CSV with 'Text' column)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Text" in df.columns:
            results = []
            for _, row in df.iterrows():
                sentiment = predict_sentiment(row["Text"])
                results.append({"Text": row["Text"], "Sentiment": sentiment})
                st.session_state.history.extend(results)
            st.success(f"Processed {len(results)} entries.")
            st.markdown("### Analysis Summary")
            st.markdown(f"<div class='stat-card'>Total Entries: {len(results)}</div>", unsafe_allow_html=True)
            history_df = pd.DataFrame(results)
            fig = px.bar(
                history_df,
                x="Sentiment",
                y="Text",
                color="Sentiment",
                color_discrete_map=COLORS,
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig)
        else:
            st.error("The file must contain a 'Text' column.")

# Add custom CSS for styling
st.markdown("""
<style>
    .stat-card {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .button {
        background-color: #3182ce;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    .button:hover {
        background-color: #2161b5;
    }
</style>
""", unsafe_allow_html=True)
