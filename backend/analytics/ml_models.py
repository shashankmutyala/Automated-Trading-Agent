from price_ingestion import fetch_binance_prices
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_real_market_data():
    """
    Fetch price data from Binance and engineer features for ML model.
    """
    data = fetch_binance_prices(symbol="BTCUSDT", interval="1h", limit=100)
    df = pd.DataFrame(data)

    # Feature Engineering
    df["ma_3"] = df["price"].rolling(3).mean()
    df["ma_7"] = df["price"].rolling(7).mean()
    df["momentum"] = df["price"].diff()

    # Target: whether the price will go up in next interval
    df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)

    return df.dropna()

def train_market_models(df):
    """
    Train multiple ML models to predict next-hour price direction.
    """
    X = df[["ma_3", "ma_7", "momentum"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=200)
    }

    reports = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

    return models, reports
