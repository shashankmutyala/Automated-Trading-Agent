from .price_ingestion import fetch_binance_prices
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
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
    features = ["ma_3", "ma_7", "momentum"]
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Hyperparameter tuning for RandomForest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    # Hyperparameter tuning for GradientBoosting
    gb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 10]
    }
    gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=3, n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    best_gb = gb_grid.best_estimator_

    # Logistic Regression with balanced class weights
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)

    models = {
        "RandomForest": best_rf,
        "GradientBoosting": best_gb,
        "LogisticRegression": lr
    }

    reports = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        reports[name] = classification_report(y_test, y_pred, output_dict=True)
    return models, reports
