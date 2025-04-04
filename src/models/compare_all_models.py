"""
Compare all cryptocurrency prediction models.

This script compares the performance of all models:
- Logistic Regression (baseline)
- Random Forest (baseline)
- MLP (PyTorch)
- LSTM (PyTorch)

It generates a summary table and visualization of performance metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import csv
import glob
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the cryptocurrency symbols
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

def read_pytorch_predictions(symbol, model_type):
    """Read the predictions from the PyTorch model CSV files."""
    file_path = f'results/{symbol}_{model_type}_predictions.csv'
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    
    try:
        df = pd.read_csv(file_path)
        true_values = df['True_Signal'].values
        predictions = df['Predicted_Signal'].values
        
        # Calculate metrics
        accuracy = np.mean(true_values == predictions)
        
        # Calculate precision, recall, and f1 for each class
        precision = {}
        recall = {}
        f1 = {}
        
        for c in sorted(np.unique(true_values)):
            # True positives
            tp = np.sum((true_values == c) & (predictions == c))
            # False positives
            fp = np.sum((true_values != c) & (predictions == c))
            # False negatives
            fn = np.sum((true_values == c) & (predictions != c))
            
            # Calculate metrics
            precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0
        
        # Calculate weighted metrics
        class_counts = {c: np.sum(true_values == c) for c in np.unique(true_values)}
        total_samples = len(true_values)
        
        weighted_precision = sum(precision[c] * class_counts[c] for c in class_counts) / total_samples
        weighted_recall = sum(recall[c] * class_counts[c] for c in class_counts) / total_samples
        weighted_f1 = sum(f1[c] * class_counts[c] for c in class_counts) / total_samples
        
        return {
            'accuracy': accuracy,
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def read_baseline_metrics():
    """Read the baseline metrics from the CSV files."""
    file_path = 'results/model_comparison_test.csv'
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return {}
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        metrics = {}
        
        # Process each row in the CSV
        for _, row in df.iterrows():
            # Extract symbol and model type from the index
            parts = row.iloc[0].split('_')
            symbol = parts[0]
            model_type = '_'.join(parts[1:])
            
            # Extract metrics
            accuracy = row['accuracy']
            precision = row['precision'] if 'precision' in row else None
            recall = row['recall'] if 'recall' in row else None
            f1 = row['f1'] if 'f1' in row else None
            
            # Initialize symbol dict if needed
            if symbol not in metrics:
                metrics[symbol] = {}
            
            # Add metrics
            metrics[symbol][model_type] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return metrics
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}


def generate_comparison_table():
    """Generate a comparison table of all models."""
    baseline_metrics = read_baseline_metrics()
    all_metrics = {}
    
    print("Model comparison for cryptocurrency trade signal prediction:")
    print("=" * 80)
    
    # Create a DataFrame to store the results
    results = []
    
    for symbol in SYMBOLS:
        print(f"\nMetrics for {symbol}:")
        print("-" * 40)
        
        symbol_metrics = {}
        
        # Add baseline metrics if available
        if symbol in baseline_metrics:
            symbol_metrics['logistic_regression'] = baseline_metrics[symbol]['logistic_regression']
            symbol_metrics['random_forest'] = baseline_metrics[symbol]['random_forest']
        
        # Add PyTorch MLP metrics
        mlp_metrics = read_pytorch_predictions(symbol, 'pytorch')
        if mlp_metrics:
            symbol_metrics['mlp'] = mlp_metrics
        
        # Add LSTM metrics
        lstm_metrics = read_pytorch_predictions(symbol, 'lstm')
        if lstm_metrics:
            symbol_metrics['lstm'] = lstm_metrics
        
        # Display metrics
        for model_name, metrics in symbol_metrics.items():
            print(f"{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            if metrics['precision'] is not None:
                print(f"  Precision: {metrics['precision']:.4f}")
            if metrics['recall'] is not None:
                print(f"  Recall:    {metrics['recall']:.4f}")
            if metrics['f1'] is not None:
                print(f"  F1 Score:  {metrics['f1']:.4f}")
            
            # Add to results
            results.append({
                'Symbol': symbol,
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1']
            })
        
        all_metrics[symbol] = symbol_metrics
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv('results/all_models_comparison.csv', index=False)
    print(f"\nComparison table saved to results/all_models_comparison.csv")
    
    return results_df, all_metrics


def plot_comparison(results_df):
    """Create visualizations for model comparison."""
    plt.figure(figsize=(14, 10))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    sns.barplot(x='Symbol', y='Accuracy', hue='Model', data=results_df)
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    
    # Precision comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x='Symbol', y='Precision', hue='Model', data=results_df.dropna(subset=['Precision']))
    plt.title('Precision Comparison')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    
    # Recall comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x='Symbol', y='Recall', hue='Model', data=results_df.dropna(subset=['Recall']))
    plt.title('Recall Comparison')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    
    # F1 Score comparison
    plt.subplot(2, 2, 4)
    sns.barplot(x='Symbol', y='F1 Score', hue='Model', data=results_df.dropna(subset=['F1 Score']))
    plt.title('F1 Score Comparison')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/all_models_comparison.png')
    print(f"Comparison plot saved to results/all_models_comparison.png")


if __name__ == "__main__":
    print(f"Running model comparison on {datetime.now()}")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate comparison table
    results_df, metrics = generate_comparison_table()
    
    # Plot comparison
    plot_comparison(results_df)
    
    print("\nModel comparison completed successfully.") 