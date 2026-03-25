import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random

# Define color palette
COLORS = {
    "POSITIVE": "#00ffcc",
    "NEUTRAL": "#f39c12",
    "NEGATIVE": "#e74c3c"
}

# Mock sentiment prediction function
def predict_sentiment(text):
    """Mock sentiment prediction function"""
    return random.choice(["POSITIVE", "NEUTRAL", "NEGATIVE"])

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Analyze", "Bulk Analysis"])

# CSS styling for consistent UI
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #1e293b;
    color: #e2e8f0;
}

.container {
    padding: 20px;
    max-width: 1200px;
    margin: auto;
}

.stat-card {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}

.stat-label {
    color: #e2e8f0;
    font-size: 14px;
    margin-bottom: 5px;
}

.stat-value {
    font-size: 20px;
    font-weight: bold;
    color: #ffffff;
}

.chart-container {
    margin-top: 20px;
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

# Page routing
if page == "Dashboard":
    st.title("Sentiment Analysis Dashboard")
    
    # Display analysis history
    st.markdown("<h3>Analysis History</h3>", unsafe_allow_html=True)
    if st.session_state.history:
        for entry in st.session_state.history:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Text:</div>
                <div class="stat-value">{entry['text']}</div>
                <div class="stat-label">Sentiment:</div>
                <div class="stat-value" style="color: {COLORS[entry['sentiment']]};">
                    {entry['sentiment']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No analysis history found. Start by analyzing some text or uploading a CSV file.")
    
    # Upload CSV for bulk analysis
    uploaded_file = st.file_uploader("Upload CSV for Bulk Analysis", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not df.empty:
            # Process bulk analysis
            results = []
            for _, row in df.iterrows():
                sentiment = predict_sentiment(row['text'])
                results.append({
                    'text': row['text'],
                    'sentiment': sentiment
                })
                st.session_state.history.append({
                    'text': row['text'],
                    'sentiment': sentiment
                })
            
            # Display bar chart
            sentiment_counts = pd.DataFrame(results)['sentiment'].value_counts()
            fig = go.Figure(data=[go.Bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                marker_color=[COLORS[s] for s in sentiment_counts.index]
            )])
            fig.update_layout(
                title="Sentiment Distribution",
                xaxis_title="Sentiment",
                yaxis_title="Count",
                template="plotly_white"
            )
            st.plotly_chart(fig)
    
    # Add separator
    st.markdown("<hr style='border: 1px solid #4a5568;'>", unsafe_allow_html=True)

elif page == "Analyze":
    st.title("Text Sentiment Analysis")
    
    # Input text area
    user_input = st.text_area("Enter text for analysis", "", height=200)
    
    if st.button("Analyze", help="Click to analyze sentiment", class="button"):
        if user_input:
            sentiment = predict_sentiment(user_input)
            st.session_state.history.append({
                'text': user_input,
                'sentiment': sentiment
            })
            
            # Display result
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Input Text:</div>
                <div class="stat-value">{user_input}</div>
                <div class="stat-label">Sentiment:</div>
                <div class="stat-value" style="color: {COLORS[sentiment]};">
                    {sentiment}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze.")

elif page == "Bulk Analysis":
    st.title("Bulk Text Sentiment Analysis")
    
    # Upload CSV for bulk analysis
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not df.empty:
            # Process bulk analysis
            results = []
            for _, row in df.iterrows():
                sentiment = predict_sentiment(row['text'])
                results.append({
                    'text': row['text'],
                    'sentiment': sentiment
                })
                st.session_state.history.append({
                    'text': row['text'],
                    'sentiment': sentiment
                })
            
            # Display bar chart
            sentiment_counts = pd.DataFrame(results)['sentiment'].value_counts()
            fig = go.Figure(data=[go.Bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                marker_color=[COLORS[s] for s in sentiment_counts.index]
            )])
            fig.update_layout(
                title="Sentiment Distribution",
                xaxis_title="Sentiment",
                yaxis_title="Count",
                template="plotly_white"
            )
            st.plotly_chart(fig)
            
            # Display summary
            st.markdown("### Analysis Summary")
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Entries:</div>
                <div class="stat-value">{len(results)}</div>
            </div>
            """, unsafe_allow_html=True)
