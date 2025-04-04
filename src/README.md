# Cryptocurrency Price Prediction Models

This module implements three different predictive models for cryptocurrency price forecasting:

1. **LSTM (Long Short-Term Memory)** - A deep learning model for time series prediction
2. **Linear Regression** - A simple linear model
3. **ARIMA (AutoRegressive Integrated Moving Average)** - A classical time series forecasting model

## Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
# Run the main script to process all data files and train models
python src/price_prediction.py
```

### Advanced Usage

```python
from price_prediction import CryptoPricePredictor

# Initialize predictor
predictor = CryptoPricePredictor(data_dir="data", results_dir="results")

# Load data
data = predictor.load_data()

# Process specific dataset (e.g., binance_btc)
data_key = "binance_btc"
preprocessed_data = predictor.preprocess_data({data_key: data[data_key]})

# Train LSTM model
predictor.train_lstm(data_key, preprocessed_data, epochs=50, batch_size=32)

# Train Linear Regression model
predictor.train_linear_regression(data_key, preprocessed_data)

# Train ARIMA model
predictor.train_arima(data_key, preprocessed_data, order=(5,1,0))

# Compare models
predictor.evaluate_and_compare_models(data_key)

# Visualize predictions
predictor.visualize_predictions(data_key, preprocessed_data)
```

## Data Preprocessing

The preprocessor performs several steps:
1. Resampling to regular time intervals (default: 1 minute)
2. Feature engineering (returns, moving averages, momentum, volatility)
3. Data scaling and sequence creation for LSTM
4. Train/test splitting

## Hyperparameter Tuning

The script includes automated hyperparameter tuning for the LSTM model:
- Window sizes
- LSTM units
- Dropout rates
- Batch sizes

## Model Evaluation

Models are evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score (for Linear Regression)

Results are saved in the `results` directory as JSON files and prediction visualizations. 