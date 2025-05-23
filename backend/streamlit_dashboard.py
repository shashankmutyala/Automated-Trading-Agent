import streamlit as st
import pandas as pd
import os
import sys

# Add backend/analytics to sys.path so we can import modules directly
sys.path.append(os.path.join(os.path.dirname(__file__), "analytics"))

st.title("Crypto Analytics & ML Dashboard")

# --- Analytics Section ---
st.header("Reddit Sentiment Analysis")
if st.button("Run Sentiment Pipeline"):
    from analytics.sentiment_pipeline import fetch_bitcoin_posts, apply_sentiment

    df = fetch_bitcoin_posts(limit=50)
    df = apply_sentiment(df)
    st.write("Sentiment Data (first 10 rows):")
    st.dataframe(df.head(10))
    st.download_button("Download Sentiment CSV", df.to_csv(index=False), "sentiment_output.csv")

st.header("Market ML Model Training & Prediction")
if st.button("Run Market Prediction"):
    from analytics.ml_models import load_real_market_data, train_market_models

    df = load_real_market_data()
    models, reports = train_market_models(df)
    for name, report in reports.items():
        st.subheader(f"{name} Report")
        for metric, values in report.items():
            if isinstance(values, dict):
                st.write(f"**{metric}:**")
                st.json(values)
            else:
                st.write(f"**{metric}:** {values}")

# --- ML Model Results Section (existing code) ---
st.header("ML Model Predictions & Features")

symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
models = ["random_forest", "logistic_regression", "lstm", "pytorch_mlp"]

symbol = st.selectbox("Select Symbol", symbols)
model = st.selectbox("Select Model", models)

pred_file = f"results/{symbol.lower()}_{model}_predictions.csv"
feature_file = f"results/{symbol}_features.txt"
summary_file = "results/model_summary.csv"

# Show model summary if available
if os.path.exists(summary_file):
    st.subheader("Model Summary")
    summary_df = pd.read_csv(summary_file)
    st.write("Columns in summary_df:", summary_df.columns.tolist())
    # Try to find the correct column for symbol
    symbol_col = None
    for col in summary_df.columns:
        if col.lower() == "symbol":
            symbol_col = col
            break
    if symbol_col:
        st.dataframe(summary_df[summary_df[symbol_col] == symbol])
    else:
        st.dataframe(summary_df)
        st.warning("No 'symbol' column found in model_summary.csv.")

# Show feature importance or features if available
if os.path.exists(feature_file):
    st.subheader("Model Features")
    with open(feature_file, "r") as f:
        st.text(f.read())

# Show predictions if available
if os.path.exists(pred_file):
    st.subheader(f"{symbol} {model.replace('_', ' ').title()} Predictions")
    pred_df = pd.read_csv(pred_file)
    st.dataframe(pred_df)
    if "prediction" in pred_df.columns and "timestamp" in pred_df.columns:
        st.line_chart(pred_df.set_index("timestamp")["prediction"])
else:
    st.warning(f"No predictions found for {symbol} with {model}.")
