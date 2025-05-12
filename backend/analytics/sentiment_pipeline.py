import praw
from datetime import datetime
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def setup_reddit_client():
    return praw.Reddit(
        client_id="3tdVb7PaLXDrBSmj4HJyag",
        client_secret="j_C93ojoVz5ObtWSg0m_5eZih2xsgg",
        user_agent="bitcoin_sentiment_tracker"
    )

def fetch_bitcoin_posts(limit=50):
    reddit = setup_reddit_client()
    posts = reddit.subreddit("bitcoin").new(limit=limit)
    data = []
    for post in posts:
        data.append({
            "title": post.title,
            "timestamp": post.created_utc,
            "datetime": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        })
    return pd.DataFrame(data)

def apply_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["score"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment"] = df["score"].apply(lambda s: "positive" if s > 0.2 else "negative" if s < -0.2 else "neutral")
    return df

def save_sentiment_to_csv(df, filename="sentiment_output.csv"):
    df.to_csv(filename, index=False)
