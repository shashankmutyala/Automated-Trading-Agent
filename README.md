# Cryptocurrency Market Analysis

This project implements machine learning models for cryptocurrency price prediction and market analysis using historical data from Binance.

## Project Structure

```
├── data/               # Directory for cryptocurrency price data (CSV files)
├── models/             # Saved trained models
├── results/            # Analysis results, visualizations, and processed data
├── src/                # Source code
│   ├── preprocessing.py       # Data preprocessing and feature engineering
│   ├── baseline_model.py      # Implementation of baseline ML models
│   ├── crypto_pipeline.py     # End-to-end pipeline for analysis
│   └── price_prediction.py    # Price prediction implementation
```

## Features

- **Data Preprocessing**: Loading, cleaning, and feature engineering for cryptocurrency data
- **Technical Indicators**: Calculation of various technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- **Trading Signal Generation**: Automatic generation of trading signals based on price movements
- **Model Training**: Implementation of MLP neural network, logistic regression, and random forest models
- **Model Evaluation**: Comprehensive evaluation with accuracy, precision, recall, F1-score, and confusion matrices
- **Visualization**: Visualizations of feature importance, model performance, and trading signals

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### End-to-End Pipeline

Run the complete analysis pipeline with default settings:

```bash
python new_method/crypto_pipeline.py
```

### Command Line Options

```bash
# Process specific cryptocurrencies
python new_method/crypto_pipeline.py --symbols BTCUSDT ETHUSDT

# Reprocess data and retrain models
python new_method/crypto_pipeline.py --reprocess --retrain

# Only run preprocessing step
python new_method/crypto_pipeline.py --preprocess-only

# Only run model training step
python new_method/crypto_pipeline.py --train-only
```

## Data Requirements

The project expects Binance price data for each cryptocurrency in CSV format, stored in the `data/` directory. Each file should contain at least the following columns:

- Open time
- Open
- High
- Low
- Close
- Volume

## Models

### MLP Neural Network
- Multi-layer perceptron with batch normalization and dropout
- Early stopping and learning rate reduction
- Optimized for 3-class classification (Buy/Hold/Sell)

### Logistic Regression
- Multinomial logistic regression
- Class weights to handle imbalanced data
- L2 regularization

### Random Forest
- Ensemble of decision trees
- Feature importance analysis
- Hyperparameter tuning

## Results

Analysis results are stored in the `results/` directory:
- Processed data in CSV format
- Feature visualizations
- Model training history plots
- Confusion matrices
- Model comparison metrics
