"""
Cryptocurrency Trade Signal Prediction using PyTorch LSTM
--------------------------------------------------------
This module implements an LSTM (Long Short-Term Memory) model
using PyTorch for predicting cryptocurrency trading signals.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_length=10):
        """
        Dataset for LSTM sequence data
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature data with shape (num_samples, num_features)
        y : numpy.ndarray
            Target data with shape (num_samples,)
        seq_length : int
            Number of time steps in each sequence
        """
        self.X = X
        self.y = y
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        # Get a sequence of features and the label right after the sequence
        seq_x = self.X[idx: idx + self.seq_length]
        label = self.y[idx + self.seq_length]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3):
        """
        LSTM model for classification
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Number of hidden units in LSTM
        num_layers : int
            Number of LSTM layers
        num_classes : int
            Number of output classes (0=Sell, 1=Hold, 2=Buy)
        """
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, seq_length, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Output logits
        """
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)  
        # Use the output from the last time step for classification
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out


class CryptoLSTMPredictor:
    """
    Class for training and evaluating LSTM models for crypto signal prediction
    """
    def __init__(self, seq_length=10, hidden_dim=64, num_layers=2, learning_rate=0.001, 
                 batch_size=32, epochs=20, patience=5):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        seq_length : int
            Number of time steps in each sequence
        hidden_dim : int
            Number of hidden units in LSTM
        num_layers : int
            Number of LSTM layers
        learning_rate : float
            Learning rate for optimization
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of epochs for training
        patience : int
            Patience for early stopping
        """
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def prepare_data(self, df, test_size=0.2):
        """
        Prepare data for training
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Processed data with features and Signal column
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict
            Dictionary with data loaders and other information
        """
        # Remove non-numeric columns
        non_numeric_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            print(f"Dropping non-numeric columns: {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)
        
        # Drop non-feature columns if they exist
        drop_cols = ['Future_Return', 'Future_Return_Pct']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Separate features and target
        X = df.drop(columns=['Signal'])
        y = df['Signal'].values
        
        # Get feature names
        self.feature_names = X.columns.tolist()
        self.input_dim = len(self.feature_names)
        
        # Determine number of classes in the target
        self.num_classes = len(np.unique(y))
        print(f"Number of classes: {self.num_classes}")
        
        # Signal values are -1, 0, 1
        # Shift to 0, 1, 2 for classification
        unique_signals = np.unique(y)
        print(f"Signal values: {unique_signals}")
        y_mapped = y + 1  # Shift values to make them non-negative
        
        # Print class distribution
        class_counts = np.bincount(y_mapped)
        print(f"Class distribution: {class_counts}")
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequence dataset
        dataset = SequenceDataset(X_scaled, y_mapped, seq_length=self.seq_length)
        print(f"Total sequences: {len(dataset)}")
        
        # Split into train and test sets
        train_size = int(len(dataset) * (1 - test_size))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
        )
        
        # Print dataset sizes
        print(f"Training set: {len(train_dataset)} sequences")
        print(f"Test set: {len(test_dataset)} sequences")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes
        }
    
    def train(self, data):
        """
        Train the model
        
        Parameters:
        -----------
        data : dict
            Dictionary with data loaders and other information
            
        Returns:
        --------
        dict
            Dictionary with training history
        """
        # Get input dimension and number of classes
        input_dim = data['input_dim']
        num_classes = data['num_classes']
        
        # Initialize model
        self.model = LSTMClassifier(
            input_dim=input_dim, 
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=num_classes
        )
        self.model = self.model.to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Get data loaders
        train_loader = data['train_loader']
        test_loader = data['test_loader']
        
        # Initialize variables for early stopping
        best_test_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Initialize history
        history = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': []
        }
        
        # Training loop
        print(f"Starting training for {self.epochs} epochs...")
        start_time = datetime.now()
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Stats
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            # Calculate training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Testing phase
            self.model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Stats
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    test_total += targets.size(0)
                    test_correct += (predicted == targets).sum().item()
            
            # Calculate test loss and accuracy
            test_loss = test_loss / test_total
            test_acc = test_correct / test_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            # Print epoch stats
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Calculate training time
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"Training completed in {training_time}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model
        
        Parameters:
        -----------
        data_loader : torch.utils.data.DataLoader
            DataLoader for evaluation
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert predictions and targets to numpy arrays
        y_pred = np.array(all_preds)
        y_true = np.array(all_targets)
        
        # Map predictions and targets back to original signal values (-1, 0, 1)
        y_pred_mapped = y_pred - 1
        y_true_mapped = y_true - 1
        
        # Use confusion matrix with original classes
        cm = confusion_matrix(y_true_mapped, y_pred_mapped)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_mapped, y_pred_mapped)
        precision = precision_score(y_true_mapped, y_pred_mapped, average='weighted', zero_division=0)
        recall = recall_score(y_true_mapped, y_pred_mapped, average='weighted', zero_division=0)
        f1 = f1_score(y_true_mapped, y_pred_mapped, average='weighted', zero_division=0)
        
        # Get classification report
        cr = classification_report(y_true_mapped, y_pred_mapped, zero_division=0)
        
        # Print summary
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(cr)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred_mapped,
            'true_values': y_true_mapped
        }
    
    def plot_training_history(self, history, symbol, results_dir='results'):
        """
        Plot training history
        
        Parameters:
        -----------
        history : dict
            Dictionary with training history
        symbol : str
            Symbol name for title
        results_dir : str
            Directory to save results
        """
        plt.figure(figsize=(16, 6))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['test_loss'], label='Test Loss')
        plt.title(f'{symbol} - LSTM Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['test_acc'], label='Test Accuracy')
        plt.title(f'{symbol} - LSTM Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"{symbol}_lstm_training_history.png"))
        plt.close()
        
        print(f"Saved training history plot to {results_dir}/{symbol}_lstm_training_history.png")
    
    def plot_confusion_matrix(self, cm, symbol, results_dir='results'):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        cm : numpy.ndarray
            Confusion matrix
        symbol : str
            Symbol name for title
        results_dir : str
            Directory to save results
        """
        plt.figure(figsize=(10, 8))
        
        # Define class labels based on signal values
        class_labels = ['Sell (-1)', 'Hold (0)', 'Buy (1)']
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, 
                   yticklabels=class_labels)
        
        plt.title(f'Confusion Matrix - {symbol} LSTM')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save figure
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"{symbol}_lstm_confusion_matrix.png"))
        plt.close()
        
        print(f"Saved confusion matrix to {results_dir}/{symbol}_lstm_confusion_matrix.png")
    
    def save_model(self, symbol, models_dir='models'):
        """
        Save the trained model
        
        Parameters:
        -----------
        symbol : str
            Symbol name for the model
        models_dir : str
            Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(models_dir, f"{symbol}_lstm.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Save scaler
        scaler_path = os.path.join(models_dir, f"{symbol}_lstm_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'feature_names': self.feature_names,
            'seq_length': self.seq_length
        }
        
        metadata_path = os.path.join(models_dir, f"{symbol}_lstm_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, symbol, models_dir='models'):
        """
        Load a trained model
        
        Parameters:
        -----------
        symbol : str
            Symbol name for the model
        models_dir : str
            Directory containing the model
            
        Returns:
        --------
        bool
            True if model loaded successfully, False otherwise
        """
        # Check if model directory exists
        if not os.path.exists(models_dir):
            print(f"Model directory {models_dir} not found")
            return False
        
        # Load metadata
        metadata_path = os.path.join(models_dir, f"{symbol}_lstm_metadata.pkl")
        
        if not os.path.exists(metadata_path):
            print(f"Model metadata not found at {metadata_path}")
            return False
        
        metadata = joblib.load(metadata_path)
        
        # Set model parameters from metadata
        self.input_dim = metadata['input_dim']
        self.hidden_dim = metadata['hidden_dim']
        self.num_layers = metadata['num_layers']
        self.num_classes = metadata['num_classes']
        self.feature_names = metadata['feature_names']
        self.seq_length = metadata['seq_length']
        
        # Build the model
        self.model = LSTMClassifier(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )
        self.model = self.model.to(self.device)
        
        # Load model weights
        model_path = os.path.join(models_dir, f"{symbol}_lstm.pth")
        
        if not os.path.exists(model_path):
            print(f"Model weights not found at {model_path}")
            return False
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load the scaler
        scaler_path = os.path.join(models_dir, f"{symbol}_lstm_scaler.pkl")
        
        if not os.path.exists(scaler_path):
            print(f"Scaler not found at {scaler_path}")
            return False
        
        self.scaler = joblib.load(scaler_path)
        
        print(f"Model loaded from {model_path}")
        return True


def train_and_evaluate_lstm_model(symbol, processed_data_path, models_dir='models', results_dir='results'):
    """
    Train and evaluate an LSTM model for a cryptocurrency
    
    Parameters:
    -----------
    symbol : str
        Symbol of the cryptocurrency
    processed_data_path : str
        Path to the processed data
    models_dir : str
        Directory to save the model
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Load the processed data
    print(f"Loading processed data from {processed_data_path}")
    df = pd.read_csv(processed_data_path)
    
    # Initialize the predictor
    predictor = CryptoLSTMPredictor(
        seq_length=10,
        hidden_dim=64,
        num_layers=2,
        learning_rate=0.001,
        batch_size=32,
        epochs=20,
        patience=5
    )
    
    # Prepare the data
    data = predictor.prepare_data(df, test_size=0.2)
    
    # Train the model
    print(f"Training LSTM model for {symbol}...")
    history = predictor.train(data)
    
    # Plot training history
    predictor.plot_training_history(history, symbol, results_dir)
    
    # Evaluate on test set
    print(f"Evaluating LSTM model for {symbol}...")
    test_metrics = predictor.evaluate(data['test_loader'])
    
    # Save predictions to CSV
    test_dataset = data['test_dataset']
    y_true = []
    y_pred = []
    
    # Get the model's predictions for the test set
    predictor.model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            inputs, targets = test_dataset[i]
            inputs = inputs.unsqueeze(0).to(predictor.device)  # Add batch dimension
            outputs = predictor.model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Convert to original scale (-1, 0, 1)
            y_true.append(targets.item() - 1)
            y_pred.append(preds.item() - 1)
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'True_Signal': y_true,
        'Predicted_Signal': y_pred
    })
    
    predictions_path = os.path.join(results_dir, f"{symbol}_lstm_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Plot confusion matrix
    predictor.plot_confusion_matrix(test_metrics['confusion_matrix'], symbol, results_dir)
    
    # Save the model
    predictor.save_model(symbol, models_dir)
    
    # Print metrics
    print(f"\nTest metrics for {symbol}:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model for cryptocurrency signal prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Symbol of the cryptocurrency to analyze')
    parser.add_argument('--processed-data-dir', default='results',
                        help='Directory containing processed data')
    parser.add_argument('--models-dir', default='models',
                        help='Directory to save models')
    parser.add_argument('--results-dir', default='results',
                        help='Directory to save results')
    parser.add_argument('--seq-length', type=int, default=10,
                        help='Length of sequence for LSTM')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Path to the processed data
    processed_data_path = os.path.join(args.processed_data_dir, f'{args.symbol}_processed.csv')
    
    if not os.path.exists(processed_data_path):
        print(f"Processed data for {args.symbol} not found at {processed_data_path}")
        sys.exit(1)
    
    # Train and evaluate the model
    train_and_evaluate_lstm_model(args.symbol, processed_data_path, args.models_dir, args.results_dir) 