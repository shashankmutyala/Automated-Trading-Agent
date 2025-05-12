# backend/analytics/main.py

from sentiment_pipeline import fetch_bitcoin_posts, apply_sentiment, save_sentiment_to_csv
from ml_models import load_real_market_data, train_market_models

def run_sentiment_pipeline():
    print("\n📥 Running Reddit Sentiment Analysis...")
    df = fetch_bitcoin_posts(limit=50)
    df = apply_sentiment(df)
    save_sentiment_to_csv(df)
    print(df.head())

def run_market_prediction():
    print("\n📊 Running Market ML Model Training...")
    df = load_real_market_data()
    models, reports = train_market_models(df)
    for name, report in reports.items():
        print(f"\n📈 {name} Report:")
        for metric, values in report.items():
            if isinstance(values, dict):
                print(f"  {metric}:")
                for label, val in values.items():
                    print(f"    {label}: {val}")
            else:
                print(f"  {metric}: {values}")

if __name__ == "__main__":
    run_sentiment_pipeline()
    run_market_prediction()
