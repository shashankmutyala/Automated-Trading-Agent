"""
Cryptocurrency Market Analysis - Data Preprocessing
--------------------------------------------------
This script handles:
1. Loading cryptocurrency data from CSV files
2. Cleaning and preprocessing (handling missing values, normalizing features)
3. Feature engineering (technical indicators - MA, RSI, MACD)
4. Signal generation for supervised learning
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CryptoDataPreprocessor:
    def __init__(self, data_dir="data", results_dir="results"):
        """Initialize the preprocessor"""
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
    def load_data(self, symbol=None):
        """
        Load cryptocurrency data from CSV files
        Optionally filter by symbol
        """
        print(f"Loading cryptocurrency data...")
        all_data = {}
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                filepath = os.path.join(self.data_dir, file)
                df = pd.read_csv(filepath)
                
                # Convert timestamp to datetime if it exists
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                
                # Filter by symbol if specified
                if symbol and 'Symbol' in df.columns:
                    df = df[df['Symbol'] == symbol]
                
                # Store data by file name (without extension)
                if not df.empty:
                    file_key = file.split('.')[0]
                    all_data[file_key] = df
                    print(f"  Loaded {len(df)} records from {file}")
                    
                    # Print time range if timestamp exists
                    if 'Timestamp' in df.columns and len(df) > 0:
                        time_min = df['Timestamp'].min()
                        time_max = df['Timestamp'].max()
                        time_span = time_max - time_min
                        print(f"  Time range: {time_min} to {time_max} ({time_span})")
        
        return all_data
    
    def preprocess_data(self, df):
        """
        Basic preprocessing:
        - Sort by timestamp
        - Remove duplicates
        - Handle missing values
        - Normalize features
        """
        print("Preprocessing data...")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Sort by timestamp if it exists
        if 'Timestamp' in df.columns:
            df = df.sort_values('Timestamp')
            # Set timestamp as index
            df = df.set_index('Timestamp')
        
        # Check for price column (could be 'Price' or 'Close')
        price_col = None
        if 'Price' in df.columns:
            price_col = 'Price'
        elif 'Close' in df.columns:
            price_col = 'Close'
            
        if price_col is None:
            print("Error: No price column found in data")
            return df
        
        # Handle missing values in price
        if df[price_col].isnull().sum() > 0:
            print(f"  Filling {df[price_col].isnull().sum()} missing price values")
            df[price_col] = df[price_col].fillna(method='ffill')
        
        # Normalize price and volume data
        df[f'{price_col}_Normalized'] = (df[price_col] - df[price_col].min()) / (df[price_col].max() - df[price_col].min())
        
        if 'Volume' in df.columns and df['Volume'].min() != df['Volume'].max():
            df['Volume_Normalized'] = (df['Volume'] - df['Volume'].min()) / (df['Volume'].max() - df['Volume'].min())
            
        return df
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators for feature engineering:
        - Moving Averages (SMA, EMA)
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        """
        print("Adding technical indicators...")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Get price column (could be 'Price' or 'Close')
        price_col = None
        if 'Price' in df.columns:
            price_col = 'Price'
        elif 'Close' in df.columns:
            price_col = 'Close'
            
        if price_col is None:
            print("Error: No price column found for creating indicators")
            return df
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                df[f'SMA_{period}'] = df[price_col].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                df[f'EMA_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        if len(df) >= 14:
            # Calculate price changes
            delta = df[price_col].diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss over 14 periods
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(df) >= 26:
            # Calculate EMAs
            ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
            ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
            
            # Calculate MACD and signal line
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Price momentum
        df['Price_Change'] = df[price_col].diff()
        df['Price_Change_Pct'] = df[price_col].pct_change() * 100
        
        # Calculate returns for different periods
        for period in [1, 3, 5]:
            df[f'Return_{period}d'] = df[price_col].pct_change(period) * 100
        
        # Bollinger Bands
        if len(df) >= 20:
            df['BB_Middle'] = df[price_col].rolling(window=20).mean()
            df['BB_Std'] = df[price_col].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
            
            # Calculate Bandwidth and %B
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_B'] = (df[price_col] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators if volume data is available
        if 'Volume' in df.columns:
            # Volume Moving Average
            df['Volume_MA_14'] = df['Volume'].rolling(window=14).mean()
            # Price-volume ratio
            df['Price_Volume_Ratio'] = df[price_col] / (df['Volume'] + 1)  # Add 1 to avoid division by zero
        
        # Drop NaN values after creating indicators
        df = df.dropna()
        print(f"  Created {len(df.columns) - 1} features, {len(df)} rows remain after removing NaN values")
        
        return df
    
    def generate_signals(self, df, n_forward=10, threshold_pct=1.0):
        """
        Generate trading signals based on future price movements
        
        Parameters:
        - df: DataFrame with price data
        - n_forward: Number of periods to look ahead
        - threshold_pct: Percentage threshold for buy/sell signals
        
        Returns:
        - DataFrame with signals (-1 for sell, 0 for hold, 1 for buy)
        """
        print(f"Generating trading signals (n_forward={n_forward}, threshold={threshold_pct}%)...")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Get price column
        price_col = None
        if 'Price' in df.columns:
            price_col = 'Price'
        elif 'Close' in df.columns:
            price_col = 'Close'
            
        if price_col is None:
            print("Error: No price column found for generating signals")
            return df
        
        # Calculate future returns
        df['Future_Return'] = df[price_col].shift(-n_forward) / df[price_col] - 1
        df['Future_Return_Pct'] = df['Future_Return'] * 100
        
        # Generate signals based on threshold
        df['Signal'] = 0  # Default is hold
        df.loc[df['Future_Return_Pct'] > threshold_pct, 'Signal'] = 1  # Buy signal
        df.loc[df['Future_Return_Pct'] < -threshold_pct, 'Signal'] = -1  # Sell signal
        
        # Drop rows where we don't have future data
        df = df.dropna(subset=['Future_Return'])
        
        # Count signals
        buy_count = (df['Signal'] == 1).sum()
        sell_count = (df['Signal'] == -1).sum()
        hold_count = (df['Signal'] == 0).sum()
        total = len(df)
        
        print(f"  Generated {total} signals: {buy_count} buy ({buy_count/total*100:.1f}%), "
              f"{sell_count} sell ({sell_count/total*100:.1f}%), "
              f"{hold_count} hold ({hold_count/total*100:.1f}%)")
        
        return df
    
    def visualize_features(self, df, symbol, output_file=None):
        """
        Visualize key features and their relationship with signals
        
        Parameters:
        - df: DataFrame with features and signals
        - symbol: Symbol for chart title
        - output_file: Optional file path for saving the visualization
        """
        print(f"Visualizing features for {symbol}...")
        
        # Get price column
        price_col = None
        if 'Price' in df.columns:
            price_col = 'Price'
        elif 'Close' in df.columns:
            price_col = 'Close'
            
        if price_col is None:
            print("Error: No price column found for visualization")
            return
        
        # Plot price with signals
        plt.figure(figsize=(14, 7))
        
        # Price chart
        plt.plot(df.index, df[price_col], label='Price', color='blue')
        
        # Mark buy and sell signals
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        
        plt.scatter(buy_signals.index, buy_signals[price_col], 
                  color='green', marker='^', alpha=0.7, s=100, label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals[price_col], 
                  color='red', marker='v', alpha=0.7, s=100, label='Sell Signal')
        
        plt.title(f'{symbol} Price with Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        if output_file:
            plt.savefig(output_file)
        else:
            plt.savefig(os.path.join(self.results_dir, f"{symbol}_signals.png"))
        plt.close()
        
        # Plot key technical indicators
        plt.figure(figsize=(14, 10))
        
        # Subplot 1: Price and MAs
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df[price_col], label='Price')
        
        if 'SMA_20' in df.columns:
            plt.plot(df.index, df['SMA_20'], label='SMA 20')
        if 'EMA_50' in df.columns:
            plt.plot(df.index, df['EMA_50'], label='EMA 50')
            
        plt.title(f'{symbol} Price and Moving Averages')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: RSI
        plt.subplot(2, 1, 2)
        if 'RSI_14' in df.columns:
            plt.plot(df.index, df['RSI_14'], label='RSI 14', color='purple')
            plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
            plt.title('RSI Indicator')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'RSI not available', horizontalalignment='center',
                   verticalalignment='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        # Save figure
        if output_file:
            base_name = os.path.splitext(output_file)[0]
            plt.savefig(f"{base_name}_indicators.png")
        else:
            plt.savefig(os.path.join(self.results_dir, f"{symbol}_indicators.png"))
        plt.close()
        
        print(f"  Saved visualizations to {self.results_dir}/")
    
    def save_processed_data(self, df, symbol, filename=None):
        """
        Save processed data to CSV
        
        Parameters:
        - df: Processed DataFrame
        - symbol: Symbol name
        - filename: Optional custom filename
        """
        if filename is None:
            filename = f"{symbol}_processed.csv"
            
        filepath = os.path.join(self.results_dir, filename)
        df.to_csv(filepath)
        print(f"Saved processed data to {filepath}")
        
        # Also save feature list
        feature_list = [col for col in df.columns if col not in ['Signal', 'Future_Return', 'Future_Return_Pct']]
        feature_filepath = os.path.join(self.results_dir, f"{symbol}_features.txt")
        
        with open(feature_filepath, 'w') as f:
            for feature in feature_list:
                f.write(f"{feature}\n")
                
        print(f"Saved feature list to {feature_filepath}")
        
        return filepath
    
    def process_crypto_data(self, symbol, resample=None, forward_look=10, threshold=1.0):
        """
        Complete processing pipeline:
        1. Load data
        2. Preprocess
        3. Add technical indicators
        4. Generate signals
        5. Visualize
        6. Save processed data
        
        Returns processed DataFrame
        """
        print(f"\n{'='*50}")
        print(f"Processing {symbol} data")
        print(f"{'='*50}")
        
        # Load data
        data_dict = self.load_data(symbol)
        
        if not data_dict:
            print(f"No data found for {symbol}")
            return None
        
        # Get the data for the symbol (first file found)
        df = None
        for key, value in data_dict.items():
            if symbol.lower() in key.lower():
                df = value
                break
                
        if df is None:
            # Just take the first dataset if none matched by name
            df = list(data_dict.values())[0]
        
        # Preprocess
        df_clean = self.preprocess_data(df)
        
        # Check if we have enough data after preprocessing
        if len(df_clean) < 50:
            print(f"Not enough data after preprocessing: {len(df_clean)} rows. Need at least 50.")
            return None
        
        # Add technical indicators
        df_features = self.add_technical_indicators(df_clean)
        
        # Generate signals
        df_signals = self.generate_signals(df_features, n_forward=forward_look, threshold_pct=threshold)
        
        # Visualize
        self.visualize_features(df_signals, symbol)
        
        # Save processed data
        output_file = self.save_processed_data(df_signals, symbol)
        
        return df_signals

def main():
    """Main function to process crypto data"""
    # Initialize preprocessor
    preprocessor = CryptoDataPreprocessor()
    
    # Define symbols to process
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    # Process each symbol
    for symbol in symbols:
        processed_data = preprocessor.process_crypto_data(
            symbol=symbol,
            forward_look=10,  # Look ahead 10 periods
            threshold=0.0001  # Lower threshold to 0.01% to create more class variation
        )
        
        if processed_data is not None:
            print(f"Successfully processed {symbol} data with {len(processed_data)} rows")
        else:
            print(f"Failed to process {symbol} data")
    
    print("\nPreprocessing complete. Data is ready for model training.")

if __name__ == "__main__":
    main() 