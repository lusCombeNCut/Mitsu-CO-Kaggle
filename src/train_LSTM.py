"""
Simple LSTM Commodity Prediction Training Script
Basic LSTM implementation for time series prediction - focuses on first target only
"""

import sys
import os
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import spearmanr, pearsonr
    from sklearn.feature_selection import SelectKBest, f_regression
    from utils.competition_metrics import (
        calculate_competition_metrics, print_competition_analysis, 
        get_competition_score, save_competition_results
    )
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")
    print("Install with: pip install torch scikit-learn")
    exit(1)

# Import advanced feature engineering from XGBoost model
try:
    import sys
    sys.path.append('src')
    from train_XGBoost import create_advanced_features, apply_feature_selection
except ImportError:
    print("⚠️ Could not import advanced features from XGBoost model")
    
    def create_advanced_features(data, feature_cols):
        """Fallback basic feature engineering"""
        return data.copy()
    
    def apply_feature_selection(X, y, feature_names, k_best=50):
        """Fallback feature selection"""
        return X, feature_names

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_gpu_availability():
    """Detect if GPU acceleration is available for PyTorch LSTM"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown GPU"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
        
        logger.info(f"✅ GPU detected: {gpu_name}")
        logger.info(f"💾 GPU memory: {gpu_memory:.1f} GB")
        logger.info(f"🔢 GPU count: {gpu_count}")
        
        # Check if we have enough memory for deep LSTM training
        if gpu_memory < 4.0:
            logger.warning(f"⚠️ Limited GPU memory ({gpu_memory:.1f} GB). Consider smaller batch sizes.")
        
        return True
    else:
        logger.info("ℹ️ No GPU detected - using CPU acceleration")
        return False

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences"""
    
    def __init__(self, X, y=None, sequence_length=30):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        # Ensure we return a non-negative length
        raw_length = len(self.X) - self.sequence_length + 1
        return max(0, raw_length)
    
    def __getitem__(self, idx):
        # Validate index bounds
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.__len__()}")
        
        # Ensure we don't exceed array bounds
        end_idx = min(idx + self.sequence_length, len(self.X))
        if end_idx - idx < self.sequence_length:
            raise IndexError(f"Cannot create sequence of length {self.sequence_length} starting at index {idx}")
        
        X_seq = torch.FloatTensor(self.X[idx:end_idx])
        
        if self.y is not None:
            target_idx = idx + self.sequence_length - 1
            if target_idx >= len(self.y):
                raise IndexError(f"Target index {target_idx} out of bounds for target array of length {len(self.y)}")
            
            if self.y.ndim == 1:
                y_val = torch.FloatTensor([self.y[target_idx]])
            else:
                y_val = torch.FloatTensor(self.y[target_idx])
            return X_seq, y_val
        return X_seq

class LSTMModel(nn.Module):
    """Deep Multi-target LSTM model for time series prediction"""
    
    def __init__(self, input_size, output_size=1, hidden_size=50, num_layers=2, dropout_rate=0.2, deep_fc=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.deep_fc = deep_fc
        
        # LSTM layers with increased depth
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Can be made bidirectional for even more depth
        )
        
        # Deep fully connected layers for complex pattern learning
        self.dropout = nn.Dropout(dropout_rate)
        
        if deep_fc:
            # Deep FC architecture for complex multi-target relationships
            fc1_size = max(hidden_size * 2, output_size * 8)  # Much larger first layer
            fc2_size = max(hidden_size, output_size * 4)
            fc3_size = max(hidden_size // 2, output_size * 2)
            
            self.fc_layers = nn.ModuleList([
                nn.Linear(hidden_size, fc1_size),
                nn.Linear(fc1_size, fc2_size),
                nn.Linear(fc2_size, fc3_size),
                nn.Linear(fc3_size, output_size)
            ])
            
            # Batch normalization for training stability
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(fc1_size),
                nn.BatchNorm1d(fc2_size),
                nn.BatchNorm1d(fc3_size)
            ])
        else:
            # Standard architecture
            self.fc1 = nn.Linear(hidden_size, max(25, output_size * 5))
            self.fc2 = nn.Linear(max(25, output_size * 5), output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)  # Better for deep networks
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        if self.deep_fc:
            # Deep fully connected layers with batch norm and residual connections
            out = last_output
            
            for i, (fc_layer, bn_layer) in enumerate(zip(self.fc_layers[:-1], self.batch_norms)):
                residual = out if out.shape[1] == fc_layer.out_features else None
                
                out = fc_layer(out)
                out = bn_layer(out)
                out = self.leaky_relu(out)
                out = self.dropout(out)
                
                # Residual connection if dimensions match
                if residual is not None and residual.shape == out.shape:
                    out = out + residual * 0.2  # Scaled residual
            
            # Final layer without batch norm
            out = self.fc_layers[-1](out)
        else:
            # Standard fully connected layers
            out = self.dropout(last_output)
            out = self.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)
        
        return out

class SimpleLSTMPredictor:
    """Multi-target LSTM predictor using PyTorch with advanced features"""
    
    def __init__(self, sequence_length=30, lstm_units=50, dropout_rate=0.2, learning_rate=0.001, 
                 n_targets=1, use_advanced_features=True, k_best_features=50,
                 num_layers=2, deep_fc=False, disable_early_stopping=False):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_targets = n_targets
        self.use_advanced_features = use_advanced_features
        self.k_best_features = k_best_features
        self.num_layers = num_layers
        self.deep_fc = deep_fc
        self.disable_early_stopping = disable_early_stopping
        # GPU detection and device selection
        self.gpu_available = detect_gpu_availability()
        self.device = torch.device('cuda' if self.gpu_available else 'cpu')
        
        if self.gpu_available:
            # Set optimal GPU settings for LSTM training
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_selector = None
        self.selected_features = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.validation_metrics = None
        
        logger.info(f"🔧 Using device: {self.device}")
        logger.info(f"🎯 Multi-target LSTM: {n_targets} targets")
        logger.info(f"✨ Advanced features: {'Enabled' if use_advanced_features else 'Disabled'}")
        
    def _build_model(self, input_size):
        """Build Deep PyTorch LSTM model"""
        model = LSTMModel(
            input_size=input_size,
            output_size=self.n_targets,
            hidden_size=self.lstm_units,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            deep_fc=self.deep_fc
        ).to(self.device)
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📋 Deep Multi-Target LSTM Model Architecture:")
        logger.info(f"   Input size: {input_size}")
        logger.info(f"   Output size: {self.n_targets}")
        logger.info(f"   Hidden size: {self.lstm_units}")
        logger.info(f"   LSTM layers: {self.num_layers}")
        logger.info(f"   Deep FC layers: {'Yes' if self.deep_fc else 'No'}")
        logger.info(f"   Sequence length: {self.sequence_length}")
        logger.info(f"   Dropout rate: {self.dropout_rate}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Early stopping: {'Disabled' if self.disable_early_stopping else 'Enabled'}")
        logger.info(f"   Device: {self.device}")
        
        if self.gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=1):
        """Train the LSTM model using PyTorch"""
        logger.info(f"🚀 Training PyTorch LSTM model...")
        logger.info(f"   Sequence length: {self.sequence_length}")
        logger.info(f"   LSTM units: {self.lstm_units}")
        logger.info(f"   Training samples: {len(X_train)}")
        
        # Apply feature selection if using advanced features
        if self.use_advanced_features and len(X_train[0]) > self.k_best_features:
            logger.info(f"🎯 Applying feature selection ({len(X_train[0])} → {self.k_best_features} best features)")
            
            # For multi-target, use first target for feature selection
            y_for_selection = y_train if y_train.ndim == 1 else y_train[:, 0]
            
            # Create temporary feature names
            temp_feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            X_train_selected, selected_feature_names = apply_feature_selection(
                X_train, y_for_selection, temp_feature_names, self.k_best_features
            )
            
            # Store selection info for validation and test
            self.feature_selector = SelectKBest(f_regression, k=self.k_best_features)
            self.feature_selector.fit(X_train, y_for_selection)
            self.selected_features = self.feature_selector.get_support()
            
            X_train = X_train_selected
            if X_val is not None:
                X_val = self.feature_selector.transform(X_val)
            
            logger.info(f"✅ Selected {self.k_best_features} most predictive features")
        
        # Scale features and target
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        
        if y_train.ndim == 1:
            y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        else:
            y_train_scaled = self.target_scaler.fit_transform(y_train)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, self.sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        # Handle validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            if y_val.ndim == 1:
                y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            else:
                y_val_scaled = self.target_scaler.transform(y_val)
            
            val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, self.sequence_length)
            
            # Check if validation dataset has enough samples
            if len(val_dataset) <= 0:
                logger.warning(f"⚠️ Validation dataset too small ({len(X_val)} samples, need >{self.sequence_length}). Skipping validation.")
                val_loader = None
            else:
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                logger.info(f"   Validation samples: {len(val_dataset)}")
        
        # Build model
        input_size = X_train_scaled.shape[1]
        self.model = self._build_model(input_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # GPU memory management
        if self.gpu_available:
            torch.cuda.empty_cache()  # Clear GPU memory before training
            logger.info(f"🔥 Using GPU acceleration: {torch.cuda.get_device_name(0)}")
        
        # Training loop with configurable early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15 if not self.disable_early_stopping else epochs + 1  # Disable by setting patience > epochs
        
        if self.disable_early_stopping:
            logger.info(f"🚀 Early stopping DISABLED - will train for full {epochs} epochs")
        else:
            logger.info(f"⏱️ Early stopping enabled with patience={patience}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Handle different output shapes
                if self.n_targets == 1:
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                else:
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                val_batches = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        
                        if self.n_targets == 1:
                            loss = criterion(outputs.squeeze(), batch_y.squeeze())
                        else:
                            loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                self.history['val_loss'].append(avg_val_loss)
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_lstm_model.pth')
                else:
                    patience_counter += 1
                
                if verbose and epoch % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                if patience_counter >= patience and not self.disable_early_stopping:
                    logger.info(f"⏹️ Early stopping triggered at epoch {epoch+1} (patience={patience})")
                    break
            else:
                # GPU memory management
                if self.gpu_available and epoch % 20 == 0:
                    torch.cuda.empty_cache()  # Periodic GPU memory cleanup
                
                if verbose and epoch % 10 == 0:
                    gpu_info = f" | GPU Mem: {torch.cuda.memory_allocated()/1024**3:.1f}GB" if self.gpu_available else ""
                    logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}{gpu_info}")        # Load best model if validation was used
        if val_loader is not None and os.path.exists('best_lstm_model.pth'):
            self.model.load_state_dict(torch.load('best_lstm_model.pth'))
        
        logger.info("✅ PyTorch LSTM training completed!")
        
        # Calculate validation metrics
        if val_loader is not None:
            self.model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = self.model(batch_X)
                    val_predictions.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(batch_y.numpy().flatten())
            
            # Inverse transform
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            
            if self.n_targets == 1:
                val_pred = self.target_scaler.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
                val_true = self.target_scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()
            else:
                # Multi-target metrics - reshape predictions and targets to 2D
                n_samples = len(val_predictions) // self.n_targets
                val_predictions_2d = val_predictions.reshape(n_samples, self.n_targets)
                val_targets_2d = val_targets.reshape(n_samples, self.n_targets)
                
                val_pred = self.target_scaler.inverse_transform(val_predictions_2d)
                val_true = self.target_scaler.inverse_transform(val_targets_2d)
            
            # Calculate comprehensive competition metrics
            target_names = self.target_columns if hasattr(self, 'target_columns') else None
            metrics = calculate_competition_metrics(val_true, val_pred, target_names)
            
            # Log validation metrics
            logger.info(f"📈 Validation Competition Metrics:")
            
            # Always show official competition score if available
            if 'official_competition_score' in metrics:
                logger.info(f"   🎯 OFFICIAL COMPETITION SCORE: {metrics['official_competition_score']:.6f}")
                logger.info(f"   📊 Daily Rank Correlation: {metrics['mean_daily_correlation']:.4f} ± {metrics['std_daily_correlation']:.4f}")
                logger.info(f"   ✅ Positive Days: {metrics['positive_days']}/{metrics['total_days']} ({metrics['positive_day_ratio']*100:.1f}%)")
            
            if self.n_targets == 1:
                logger.info(f"   Spearman correlation: {metrics['spearman_correlation']:.4f}")
                logger.info(f"   Pearson correlation: {metrics['pearson_correlation']:.4f}")
                logger.info(f"   RMSE: {metrics['rmse']:.6f}")
                logger.info(f"   MAE: {metrics['mae']:.6f}")
                return metrics.get('official_competition_score', metrics['spearman_correlation']), metrics['rmse']
            else:
                logger.info(f"   Mean Spearman correlation: {metrics['mean_spearman']:.4f}")
                logger.info(f"   Mean Pearson correlation: {metrics['mean_pearson']:.4f}")
                logger.info(f"   Mean RMSE: {metrics['mean_rmse']:.6f}")
                logger.info(f"   Mean MAE: {metrics['mean_mae']:.6f}")
                
                # Store detailed metrics for later analysis
                self.validation_metrics = metrics
                
                return metrics.get('official_competition_score', metrics['mean_spearman']), metrics['mean_rmse']
        
        return None, None
    
    def predict(self, X):
        """Generate predictions using PyTorch model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Check if we have enough data for prediction
        if len(X_scaled) < self.sequence_length:
            logger.warning(f"Not enough data for prediction. Have {len(X_scaled)} samples, need at least {self.sequence_length}.")
            return np.array([])
        
        # Create dataset for prediction
        pred_dataset = TimeSeriesDataset(X_scaled, None, self.sequence_length)
        
        # Double-check dataset length
        try:
            dataset_length = len(pred_dataset)
            if dataset_length <= 0:
                logger.warning(f"Dataset length is {dataset_length}. Cannot create valid sequences.")
                return np.array([])
        except Exception as e:
            logger.error(f"Error getting dataset length: {e}")
            return np.array([])
        
        # Create data loader with error handling
        try:
            pred_loader = DataLoader(pred_dataset, batch_size=32, shuffle=False)
        except Exception as e:
            logger.error(f"Error creating DataLoader: {e}")
            return np.array([])
        
        # Predict
        self.model.eval()
        predictions = []
        
        try:
            with torch.no_grad():
                for batch_X in pred_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = self.model(batch_X)
                    predictions.extend(outputs.cpu().numpy().flatten())
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return np.array([])
        
        # Inverse transform
        predictions = np.array(predictions)
        
        if self.n_targets == 1:
            pred = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        else:
            # Reshape predictions to 2D for multi-target
            n_samples = len(predictions) // self.n_targets
            predictions_2d = predictions.reshape(n_samples, self.n_targets)
            pred = self.target_scaler.inverse_transform(predictions_2d)
        
        # Pad predictions to match input length (fill first sequence_length with NaN)
        if self.n_targets == 1:
            full_pred = np.full(len(X), np.nan)
            
            # Ensure we don't exceed array bounds
            end_idx = min(self.sequence_length + len(pred), len(full_pred))
            pred_slice = pred[:end_idx - self.sequence_length]
            full_pred[self.sequence_length:end_idx] = pred_slice
        else:
            # Multi-target: create 2D array
            full_pred = np.full((len(X), self.n_targets), np.nan)
            
            # Ensure we don't exceed array bounds
            end_idx = min(self.sequence_length + len(pred), len(full_pred))
            pred_slice = pred[:end_idx - self.sequence_length]
            full_pred[self.sequence_length:end_idx] = pred_slice
        
        return full_pred
    
    def get_training_history(self):
        """Get training history for visualization"""
        return self.history

def load_training_data(data_path: str, n_features: int = 50, n_targets: int = 1, time_fraction: float = 1.0) -> Dict:
    """Load and prepare training dataset with multiple targets support"""
    logger.info(f"Loading training dataset from {data_path}")
    logger.info(f"Using {n_features} features, {n_targets} targets, and {time_fraction:.1%} of time series")
    
    # Load data
    train_data = pd.read_csv(f"{data_path}/train.csv")
    train_labels = pd.read_csv(f"{data_path}/train_labels.csv")
    test_data = pd.read_csv(f"{data_path}/test.csv")
    
    # Use only a fraction of the time series if specified
    if time_fraction < 1.0:
        n_rows = int(len(train_data) * time_fraction)
        logger.info(f"Using first {n_rows} rows out of {len(train_data)} ({time_fraction:.1%})")
        train_data = train_data.head(n_rows).copy()
        train_labels = train_labels.head(n_rows).copy()
        # For test data, also take the proportional fraction
        test_n_rows = int(len(test_data) * time_fraction)
        test_data = test_data.head(test_n_rows).copy()
    
    # Select subset of features (skip date_id, take first n_features)
    feature_cols = [col for col in train_data.columns if col != 'date_id'][:n_features]
    selected_features = ['date_id'] + feature_cols
    
    filtered_train_data = train_data[selected_features].copy()
    filtered_test_data = test_data[selected_features].copy()
    
    # Select multiple targets
    target_cols = [col for col in train_labels.columns if col.startswith('target_')][:n_targets]
    selected_targets = ['date_id'] + target_cols
    
    filtered_train_labels = train_labels[selected_targets].copy()
    
    logger.info(f"Filtered train data shape: {filtered_train_data.shape}")
    logger.info(f"Filtered train labels shape: {filtered_train_labels.shape}")
    logger.info(f"Filtered test data shape: {filtered_test_data.shape}")
    logger.info(f"Selected features: {feature_cols[:5]}..." + (f" (+{len(feature_cols)-5} more)" if len(feature_cols) > 5 else ""))
    logger.info(f"Selected targets: {target_cols}")
    
    return {
        'train_data': filtered_train_data,
        'train_labels': filtered_train_labels,
        'test_data': filtered_test_data,
        'feature_cols': feature_cols,
        'target_cols': target_cols
    }

def create_time_split(data: pd.DataFrame, feature_cols: List[str], target_cols: List[str], 
                     sequence_length: int = 30, validation_days: int = 60) -> Tuple:
    """Create time series split for LSTM training with sequence length consideration"""
    logger.info("Creating time series split...")
    
    # Sort by date_id
    data = data.sort_values('date_id').reset_index(drop=True)
    
    # Prepare features and targets
    X_all = data[feature_cols].values.astype(np.float32)
    if len(target_cols) == 1:
        y_all = data[target_cols[0]].values.astype(np.float32)
    else:
        y_all = data[target_cols].values.astype(np.float32)
    
    # Adaptive validation sizing based on dataset size
    total_rows = len(data)
    min_train_samples = max(50, sequence_length * 3)  # Minimum viable training samples
    
    # Calculate adaptive validation days
    if total_rows < 150:
        # Very small dataset - use minimal validation
        adaptive_validation_days = max(10, total_rows // 8)
    elif total_rows < 300:
        # Small dataset - use smaller validation
        adaptive_validation_days = max(20, min(validation_days, total_rows // 4))
    else:
        # Large dataset - use full validation
        adaptive_validation_days = validation_days
    
    # Ensure we have enough data after adaptive sizing
    min_required = adaptive_validation_days + sequence_length + min_train_samples
    if total_rows < min_required:
        # Last resort - use minimum possible split
        adaptive_validation_days = max(5, total_rows - sequence_length - min_train_samples)
        min_required = adaptive_validation_days + sequence_length + min_train_samples
        
        if total_rows < min_required:
            raise ValueError(
                f"Dataset too small: {total_rows} rows. Need at least {min_required} rows "
                f"(min_train={min_train_samples} + sequence_length={sequence_length} + min_val=5). "
                f"Try: --time-fraction 0.3 or --sequence-length {max(5, total_rows//15)}"
            )
    
    logger.info(f"📊 Adaptive validation: using {adaptive_validation_days} days (requested: {validation_days})")
    
    # Time series split - use last adaptive_validation_days for validation
    train_size = total_rows - adaptive_validation_days
    
    if train_size <= sequence_length:
        raise ValueError(
            f"Training size too small: {train_size} samples (need > {sequence_length}). "
            f"Dataset: {total_rows} rows, validation: {adaptive_validation_days} days. "
            f"Try reducing --sequence-length or increasing --time-fraction"
        )
    
    X_train = X_all[:train_size]
    X_val = X_all[train_size:]
    y_train = y_all[:train_size]
    y_val = y_all[train_size:]
    
    logger.info(f"📅 Time series split:")
    logger.info(f"   Training: {len(X_train)} samples")
    logger.info(f"   Validation: {len(X_val)} samples ({adaptive_validation_days} days)")
    logger.info(f"   Features: {X_train.shape[1]}")
    logger.info(f"   Sequence length: {sequence_length}")
    logger.info(f"   Data efficiency: {(len(X_train) + len(X_val))/total_rows*100:.1f}% of input data used")
    
    # Remove NaN values
    if y_train.ndim == 1:
        train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        val_mask = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val))
    else:
        train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train).any(axis=1))
        val_mask = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val).any(axis=1))
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    
    logger.info(f"   After cleaning: Train {len(X_train)}, Val {len(X_val)} samples")
    logger.info(f"   Sequences after split: Train ~{max(0, len(X_train) - sequence_length)}, Val ~{max(0, len(X_val) - sequence_length)}")
    
    return X_train, X_val, y_train, y_val

def create_comprehensive_visualizations(model: SimpleLSTMPredictor, predictions: np.ndarray, 
                                       validation_metrics: dict, target_cols: list, 
                                       model_name: str = "LSTM") -> None:
    """Create comprehensive visualizations for multi-target LSTM training with individual target analysis"""
    logger.info("Creating comprehensive LSTM training visualizations...")
    
    # Create figures directory
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Overall Training History
    history = model.get_training_history()
    if history and 'train_loss' in history:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Training Overview', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training/Validation Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            epochs_val = range(1, len(history['val_loss']) + 1)
            axes[0, 0].plot(epochs_val, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss Convergence (last 50% of training)
        if len(history['train_loss']) > 20:
            start_idx = len(history['train_loss']) // 2
            late_epochs = epochs[start_idx:]
            late_train_loss = history['train_loss'][start_idx:]
            
            axes[0, 1].plot(late_epochs, late_train_loss, 'g-', label='Training Loss (Late)', linewidth=2)
            if 'val_loss' in history and len(history['val_loss']) > start_idx:
                late_val_loss = history['val_loss'][start_idx:]
                late_val_epochs = range(start_idx + 1, len(history['val_loss']) + 1)
                axes[0, 1].plot(late_val_epochs, late_val_loss, 'orange', label='Validation Loss (Late)', linewidth=2)
            
            axes[0, 1].set_title('Training Convergence (Later Half)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Competition Score Analysis
        if validation_metrics and 'daily_rank_correlations' in validation_metrics:
            daily_corrs = validation_metrics['daily_rank_correlations']
            days = range(1, len(daily_corrs) + 1)
            
            axes[1, 0].plot(days, daily_corrs, 'purple', marker='o', linewidth=2, markersize=4)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Daily Rank Correlations (Validation)')
            axes[1, 0].set_xlabel('Validation Day')
            axes[1, 0].set_ylabel('Rank Correlation')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add performance statistics
            pos_days = sum(1 for x in daily_corrs if x > 0)
            total_days = len(daily_corrs)
            axes[1, 0].text(0.02, 0.98, f'Positive Days: {pos_days}/{total_days} ({pos_days/total_days*100:.1f}%)', 
                           transform=axes[1, 0].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Performance Summary
        axes[1, 1].axis('off')
        
        # Extract key metrics
        official_score = validation_metrics.get('official_competition_score', 0) if validation_metrics else 0
        mean_correlation = validation_metrics.get('mean_daily_correlation', 0) if validation_metrics else 0
        positive_days = validation_metrics.get('positive_days', 0) if validation_metrics else 0
        total_days = validation_metrics.get('total_days', 1) if validation_metrics else 1
        
        summary_text = f"""
{model_name} Training Summary

Model Configuration:
• Targets: {len(target_cols)} commodities
• Sequence Length: {model.sequence_length}
• LSTM Units: {model.lstm_units}
• Layers: {model.num_layers}
• Deep FC: {model.deep_fc}

Performance Metrics:
• Official Score: {official_score:.6f}
• Mean Daily Correlation: {mean_correlation:.4f}
• Positive Days: {positive_days}/{total_days} ({positive_days/total_days*100:.1f}%)

🚨 PERFORMANCE ALERT:
{positive_days/total_days*100:.1f}% positive days indicates
{'RANDOM' if abs(positive_days/total_days - 0.5) < 0.1 else 'SYSTEMATIC'} performance!
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', 
                                facecolor='lightcoral' if abs(positive_days/total_days - 0.5) < 0.1 else 'lightgreen', 
                                alpha=0.8))
        
        plt.tight_layout()
        overview_path = os.path.join(figures_dir, f'{model_name}_training_overview.png')
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training overview saved to: {overview_path}")
        plt.close()
    
    # 2. Individual Target Performance Analysis (Top 20 targets)
    if validation_metrics and 'spearman_correlations' in validation_metrics:
        correlations = validation_metrics['spearman_correlations']
        n_targets_to_show = min(20, len(correlations))
        
        # Sort targets by correlation performance
        target_performance = [(i, corr, target_cols[i] if i < len(target_cols) else f'target_{i}') 
                            for i, corr in enumerate(correlations[:n_targets_to_show])]
        target_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Create performance ranking plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle(f'{model_name} Individual Target Performance (Top 20)', fontsize=16, fontweight='bold')
        
        # Top performers
        indices = [x[0] for x in target_performance]
        corrs = [x[1] for x in target_performance]
        target_names = [x[2] for x in target_performance]
        
        colors = ['green' if c > 0.1 else 'orange' if c > 0 else 'red' for c in corrs]
        bars = ax1.bar(range(len(corrs)), corrs, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Individual Target Correlations (Ranked by Performance)')
        ax1.set_xlabel('Target Rank')
        ax1.set_ylabel('Spearman Correlation')
        ax1.grid(True, alpha=0.3)
        
        # Add target names to bars
        for i, (bar, name, corr) in enumerate(zip(bars, target_names, corrs)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                   f'{name}\n{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                   rotation=45, fontsize=8)
        
        # Correlation distribution
        ax2.hist(correlations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(correlations):.3f}')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Distribution of Target Correlations')
        ax2.set_xlabel('Spearman Correlation')
        ax2.set_ylabel('Number of Targets')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        performance_path = os.path.join(figures_dir, f'{model_name}_target_performance.png')
        plt.savefig(performance_path, dpi=150, bbox_inches='tight')
        logger.info(f"Target performance analysis saved to: {performance_path}")
        plt.close()
    
    # 3. Prediction Analysis
    valid_predictions = predictions[~np.isnan(predictions)] if predictions.ndim == 1 else predictions[~np.isnan(predictions).any(axis=1)]
    
    if len(valid_predictions) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Prediction distribution
        if predictions.ndim == 1:
            axes[0, 0].hist(valid_predictions, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0, 0].set_title('Test Prediction Distribution')
        else:
            # For multi-target, show distribution of all predictions
            all_preds = valid_predictions.flatten()
            axes[0, 0].hist(all_preds, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0, 0].set_title('All Target Predictions Distribution')
        
        axes[0, 0].set_xlabel('Prediction Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Prediction statistics per target (if multi-target)
        if predictions.ndim > 1 and len(target_cols) > 1:
            target_means = []
            target_stds = []
            
            for i in range(min(10, predictions.shape[1])):
                target_preds = predictions[:, i]
                valid_target_preds = target_preds[~np.isnan(target_preds)]
                if len(valid_target_preds) > 0:
                    target_means.append(np.mean(valid_target_preds))
                    target_stds.append(np.std(valid_target_preds))
                else:
                    target_means.append(0)
                    target_stds.append(0)
            
            x_pos = range(len(target_means))
            axes[0, 1].bar(x_pos, target_means, yerr=target_stds, capsize=3, alpha=0.7, color='orange')
            axes[0, 1].set_title('Prediction Statistics by Target (First 10)')
            axes[0, 1].set_xlabel('Target Index')
            axes[0, 1].set_ylabel('Mean ± Std Prediction')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Diagnostic: Check for constant predictions
        if predictions.ndim == 1:
            pred_variance = np.var(valid_predictions)
            axes[1, 0].text(0.5, 0.7, f'Prediction Variance: {pred_variance:.6f}', 
                           ha='center', transform=axes[1, 0].transAxes, fontsize=14,
                           bbox=dict(boxstyle='round', 
                                   facecolor='red' if pred_variance < 1e-6 else 'green', 
                                   alpha=0.8))
            
            if pred_variance < 1e-6:
                axes[1, 0].text(0.5, 0.5, '⚠️ WARNING: Nearly constant predictions!\nModel may not be learning patterns.', 
                               ha='center', transform=axes[1, 0].transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        axes[1, 0].set_title('Prediction Diagnostics')
        axes[1, 0].axis('off')
        
        # Training diagnostics
        diagnostic_text = f"""
TRAINING DIAGNOSTICS:

🔍 Model Analysis:
• Parameters: {sum(p.numel() for p in model.model.parameters()):,}
• Device: {model.device}
• Sequence Length: {model.sequence_length}

📊 Data Analysis:
• Valid Predictions: {len(valid_predictions):,}
• Prediction Range: [{np.min(valid_predictions):.4f}, {np.max(valid_predictions):.4f}]
• Prediction Mean: {np.mean(valid_predictions):.4f}
• Prediction Std: {np.std(valid_predictions):.4f}

🎯 Performance Issues:
{'• Random performance suggests overfitting' if validation_metrics and abs(validation_metrics.get('positive_day_ratio', 0.5) - 0.5) < 0.1 else '• Systematic performance detected'}
{'• Low prediction variance may indicate underfitting' if np.var(valid_predictions) < 1e-4 else '• Normal prediction variance'}
        """
        
        axes[1, 1].text(0.05, 0.95, diagnostic_text, transform=axes[1, 1].transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        prediction_path = os.path.join(figures_dir, f'{model_name}_prediction_analysis.png')
        plt.savefig(prediction_path, dpi=150, bbox_inches='tight')
        logger.info(f"Prediction analysis saved to: {prediction_path}")
        plt.close()
    
    logger.info(f"✅ All visualizations saved to {figures_dir}/ directory")
    
    # 1. Training history
    history = model.get_training_history()
    if history and 'train_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            epochs_val = range(1, len(history['val_loss']) + 1)
            axes[0, 0].plot(epochs_val, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('PyTorch Training Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Learning rate / Loss zoom
        if len(history['train_loss']) > 10:
            # Show last portion of training for detail
            last_epochs = epochs[-20:] if len(epochs) > 20 else epochs
            last_train_loss = history['train_loss'][-20:] if len(history['train_loss']) > 20 else history['train_loss']
            
            axes[0, 1].plot(last_epochs, last_train_loss, 'g-', label='Training Loss (Last 20)', linewidth=2)
            if 'val_loss' in history and len(history['val_loss']) > 0:
                last_val_epochs = min(len(history['val_loss']), 20)
                last_val_loss = history['val_loss'][-last_val_epochs:]
                last_val_epochs_range = range(len(epochs) - last_val_epochs + 1, len(epochs) + 1)
                axes[0, 1].plot(last_val_epochs_range, last_val_loss, 'orange', label='Validation Loss (Last 20)', linewidth=2)
            
            axes[0, 1].set_title('Training Progress (Detailed View)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Need more epochs\nfor detailed view', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
    else:
        axes[0, 0].text(0.5, 0.5, 'No training history available', 
                      ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, 'No training history available', 
                      ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. Prediction distribution
    valid_predictions = predictions[~np.isnan(predictions)]
    if len(valid_predictions) > 0:
        axes[1, 0].hist(valid_predictions, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title(f'Test Prediction Distribution - {model_name}')
        axes[1, 0].set_xlabel('Prediction Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_pred = np.mean(valid_predictions)
        std_pred = np.std(valid_predictions)
        axes[1, 0].axvline(mean_pred, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pred:.4f}')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No valid predictions', 
                      ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 4. Summary statistics
    axes[1, 1].axis('off')
    
    summary_text = f"""
LSTM Training Summary:

Model Configuration:
• Architecture: 2-layer LSTM + Dense
• Sequence Length: {model.sequence_length}
• LSTM Units: {model.lstm_units}
• Dropout Rate: {model.dropout_rate:.1f}

Performance Metrics:
• Targets: {len(target_cols)} commodities
• Official Score: {official_score:.6f}
• Mean Daily Correlation: {mean_correlation:.4f}
• Positive Days: {positive_days}/{total_days} ({positive_days/total_days*100:.1f}%)
• Test Samples: {len(predictions) if hasattr(predictions, '__len__') else 'N/A'}
• Valid Predictions: {len(valid_predictions) if len(valid_predictions) > 0 else 'None'}

Model Details:
• Framework: PyTorch
• Optimizer: Adam
• Loss Function: MSE
• Scaling: StandardScaler (features)
           MinMaxScaler (targets)
    """
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('lstm_training_results.png', dpi=150, bbox_inches='tight')
    logger.info("LSTM visualization saved to lstm_training_results.png")
    plt.show()

def evaluate_on_test_data(model: SimpleLSTMPredictor, test_data: pd.DataFrame, 
                         feature_cols: list, target_cols: list, 
                         use_advanced_features: bool = True) -> dict:
    """Evaluate trained model on test dataset and provide comprehensive analysis"""
    logger.info("\n🧪 TESTING MODEL ON TEST DATASET...")
    
    # Apply advanced features to test data if enabled
    if use_advanced_features:
        logger.info(f"🔬 Creating advanced features for test data...")
        try:
            test_enhanced = create_advanced_features(test_data, feature_cols)
            # Use same feature columns as training
            available_features = [col for col in feature_cols if col in test_enhanced.columns]
            X_test = test_enhanced[available_features].values.astype(np.float32)
            logger.info(f"✨ Test features created: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        except Exception as e:
            logger.warning(f"⚠️ Advanced feature creation for test failed: {e}. Using basic features.")
            basic_features = [col for col in feature_cols if col in test_data.columns]
            X_test = test_data[basic_features].values.astype(np.float32)
    else:
        # Use basic features only
        basic_features = [col for col in feature_cols if col in test_data.columns]
        X_test = test_data[basic_features].values.astype(np.float32)
    
    # Generate predictions
    logger.info(f"🔮 Generating predictions on {len(X_test)} test samples...")
    test_predictions = model.predict(X_test)
    
    # Analyze predictions
    test_analysis = {
        'total_samples': len(X_test),
        'prediction_shape': test_predictions.shape if hasattr(test_predictions, 'shape') else 'scalar',
        'valid_predictions': 0,
        'prediction_stats': {},
        'coverage_analysis': {},
        'potential_issues': []
    }
    
    if len(test_predictions) > 0:
        if test_predictions.ndim == 1:
            valid_mask = ~np.isnan(test_predictions)
            valid_preds = test_predictions[valid_mask]
            
            test_analysis['valid_predictions'] = len(valid_preds)
            if len(valid_preds) > 0:
                test_analysis['prediction_stats'] = {
                    'mean': float(np.mean(valid_preds)),
                    'std': float(np.std(valid_preds)),
                    'min': float(np.min(valid_preds)),
                    'max': float(np.max(valid_preds)),
                    'variance': float(np.var(valid_preds))
                }
        else:
            # Multi-target predictions
            for i, target in enumerate(target_cols):
                if i < test_predictions.shape[1]:
                    target_preds = test_predictions[:, i]
                    valid_mask = ~np.isnan(target_preds)
                    valid_preds = target_preds[valid_mask]
                    
                    if len(valid_preds) > 0:
                        test_analysis['prediction_stats'][target] = {
                            'valid_count': len(valid_preds),
                            'mean': float(np.mean(valid_preds)),
                            'std': float(np.std(valid_preds)),
                            'min': float(np.min(valid_preds)),
                            'max': float(np.max(valid_preds))
                        }
            
            # Overall valid predictions
            all_valid = ~np.isnan(test_predictions).any(axis=1)
            test_analysis['valid_predictions'] = int(np.sum(all_valid))
    
    # Coverage analysis
    sequence_length = model.sequence_length
    expected_predictions = max(0, len(X_test) - sequence_length + 1)
    actual_predictions = test_analysis['valid_predictions']
    
    test_analysis['coverage_analysis'] = {
        'expected_predictions': expected_predictions,
        'actual_predictions': actual_predictions,
        'coverage_ratio': actual_predictions / max(1, expected_predictions),
        'missing_predictions': expected_predictions - actual_predictions
    }
    
    # Identify potential issues
    issues = []
    if actual_predictions == 0:
        issues.append("❌ NO VALID PREDICTIONS: Model failed to generate any predictions")
    elif actual_predictions < expected_predictions * 0.5:
        issues.append(f"⚠️ LOW COVERAGE: Only {actual_predictions}/{expected_predictions} predictions ({actual_predictions/expected_predictions*100:.1f}%)")
    
    if test_analysis['prediction_stats']:
        if isinstance(test_analysis['prediction_stats'], dict):
            if 'variance' in test_analysis['prediction_stats'] and test_analysis['prediction_stats']['variance'] < 1e-6:
                issues.append("⚠️ CONSTANT PREDICTIONS: Model outputs nearly identical values")
        else:
            # Multi-target case - check for low variance targets
            low_variance_targets = []
            for target, stats in test_analysis['prediction_stats'].items():
                if isinstance(stats, dict) and 'std' in stats and stats['std'] < 1e-3:
                    low_variance_targets.append(target)
            if low_variance_targets:
                issues.append(f"⚠️ LOW VARIANCE TARGETS: {len(low_variance_targets)} targets with minimal variation")
    
    test_analysis['potential_issues'] = issues
    
    # Log results
    logger.info(f"\n📊 TEST DATASET EVALUATION RESULTS:")
    logger.info(f"   Total samples: {test_analysis['total_samples']}")
    logger.info(f"   Valid predictions: {test_analysis['valid_predictions']}")
    logger.info(f"   Coverage: {test_analysis['coverage_analysis']['coverage_ratio']*100:.1f}%")
    
    if test_analysis['potential_issues']:
        logger.warning(f"\n⚠️ IDENTIFIED ISSUES:")
        for issue in test_analysis['potential_issues']:
            logger.warning(f"   {issue}")
    
    if test_analysis['prediction_stats']:
        if isinstance(test_analysis['prediction_stats'], dict) and 'mean' in test_analysis['prediction_stats']:
            stats = test_analysis['prediction_stats']
            logger.info(f"   Prediction range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logger.info(f"   Prediction mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return test_analysis

def main():
    """Main LSTM training function with enhanced visualization and test evaluation"""
    parser = argparse.ArgumentParser(description='Simple LSTM Commodity Prediction Training')
    parser.add_argument('--data-path', type=str, default='mitsui-commodity-prediction-challenge',
                       help='Path to data directory')
    parser.add_argument('--n-features', type=int, default=370,
                       help='Number of features to use (default: 370 = all features)')
    parser.add_argument('--n-targets', type=int, default=424,
                       help='Number of targets to predict (default: 424 = all targets)')
    parser.add_argument('--time-fraction', type=float, default=1.0,
                       help='Fraction of time series to use (default: 1.0)')
    parser.add_argument('--sequence-length', type=int, default=30,
                       help='LSTM sequence length (default: 30, auto-reduces for small datasets)')
    parser.add_argument('--lstm-units', type=int, default=50,
                       help='Number of LSTM units (default: 50)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default: 0.2)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--no-advanced-features', action='store_true',
                       help='Skip advanced feature engineering')
    parser.add_argument('--k-best', type=int, default=50,
                       help='Number of best features to select (default: 50)')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save competition evaluation results')
    parser.add_argument('--use-official-metric', action='store_true', default=True,
                       help='Use official rank correlation Sharpe ratio metric')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2, recommended: 3-6 for deeper models)')
    parser.add_argument('--deep-fc', action='store_true',
                       help='Use deep fully connected layers with batch norm and residuals')
    parser.add_argument('--disable-early-stopping', action='store_true',
                       help='Disable early stopping and train for full epochs')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU training even if GPU is available')
    
    args = parser.parse_args()
    
    # Detect GPU availability first
    gpu_available = detect_gpu_availability()
    
    print("🧠 MITSUI DEEP LSTM COMMODITY PREDICTOR")
    print("=" * 55)
    print(f"Data path: {args.data_path}")
    print(f"Features: {args.n_features} ({'ALL FEATURES' if args.n_features >= 370 else 'SUBSET'})")
    print(f"Targets: {args.n_targets} ({'ALL TARGETS' if args.n_targets >= 424 else 'SUBSET'})")
    print(f"Time fraction: {args.time_fraction:.1%}")
    print(f"Architecture: {args.num_layers}-layer LSTM + {'Deep' if args.deep_fc else 'Standard'} FC")
    print(f"Sequence length: {args.sequence_length}")
    print(f"LSTM units: {args.lstm_units}")
    print(f"Epochs: {args.epochs} ({'NO EARLY STOP' if args.disable_early_stopping else 'WITH EARLY STOP'})")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {'🔥 GPU' if gpu_available else '💻 CPU'}")
    print("=" * 55)
    
    start_time = time.time()
    
    try:
        # Load data
        data_dict = load_training_data(args.data_path, args.n_features, args.n_targets, args.time_fraction)
        
        # Auto-adjust sequence length for small datasets
        merged_data = data_dict['train_data'].merge(data_dict['train_labels'], on='date_id', how='inner')
        available_samples = len(merged_data)
        
        if available_samples < 200 and args.sequence_length > 15:
            original_seq = args.sequence_length
            args.sequence_length = min(15, max(5, available_samples // 10))
            logger.warning(f"⚠️ Small dataset ({available_samples} samples): reducing sequence length from {original_seq} to {args.sequence_length}")
        elif available_samples < 500 and args.sequence_length > 25:
            original_seq = args.sequence_length
            args.sequence_length = min(25, max(10, available_samples // 15))
            logger.warning(f"⚠️ Medium dataset ({available_samples} samples): reducing sequence length from {original_seq} to {args.sequence_length}")
        
        # Provide dataset size recommendations
        if available_samples < 200:
            logger.warning(f"📊 Small dataset warning: {available_samples} samples may limit model performance")
            logger.info(f"💡 Recommendations for better results:")
            logger.info(f"   • Increase --time-fraction to 0.3-0.5 (currently {args.time_fraction})")
            logger.info(f"   • Reduce --n-targets to 50-100 for initial testing")
            logger.info(f"   • Use --sequence-length 10-15 for small datasets")
        elif available_samples < 300:
            logger.info(f"📊 Medium dataset: {available_samples} samples - good for testing, consider more data for production")
        
        # Merge data
        merged_data = data_dict['train_data'].merge(
            data_dict['train_labels'], 
            on='date_id', 
            how='inner'
        )
        
        # Remove rows with missing targets
        target_cols = data_dict['target_cols']
        merged_data = merged_data.dropna(subset=target_cols)
        
        # Target statistics summary (condensed)
        logger.info(f"📊 Target statistics: {len(target_cols)} targets loaded successfully")
        
        # Apply advanced feature engineering if enabled
        if not args.no_advanced_features:
            logger.info(f"🔬 Creating advanced Phase 2 features...")
            try:
                enhanced_data = create_advanced_features(merged_data, data_dict['feature_cols'])
                feature_cols = [col for col in enhanced_data.columns if col not in ['date_id'] + target_cols]
                logger.info(f"✨ Advanced features created: {enhanced_data.shape[0]} samples, {len(feature_cols)} features")
                logger.info(f"✨ Feature expansion: {len(data_dict['feature_cols'])} → {len(feature_cols)} ({len(feature_cols)/len(data_dict['feature_cols']):.1f}x)")
                merged_data = enhanced_data
                data_dict['feature_cols'] = feature_cols
            except Exception as e:
                logger.warning(f"⚠️ Advanced feature creation failed: {e}. Using basic features.")
        else:
            logger.info(f"ℹ️ Using basic features only (advanced features disabled)")
        
        # Create time series split
        X_train, X_val, y_train, y_val = create_time_split(
            merged_data, data_dict['feature_cols'], target_cols, 
            sequence_length=args.sequence_length, validation_days=60
        )
        
        # Create and train LSTM model with enhanced architecture
        model = SimpleLSTMPredictor(
            sequence_length=args.sequence_length,
            lstm_units=args.lstm_units,
            dropout_rate=args.dropout,
            learning_rate=args.learning_rate,
            n_targets=args.n_targets,
            use_advanced_features=not args.no_advanced_features,
            k_best_features=args.k_best,
            num_layers=args.num_layers,
            deep_fc=args.deep_fc,
            disable_early_stopping=args.disable_early_stopping
        )
        model.force_cpu = args.force_cpu  # Pass force_cpu flag
        
        correlation, rmse = model.fit(
            X_train, y_train, X_val, y_val, 
            epochs=args.epochs, batch_size=args.batch_size
        )
        
        # Save model
        print(f"\n💾 Saving PyTorch LSTM model...")
        os.makedirs('models', exist_ok=True)
        model_path = 'models/lstm_target0_model.pth'
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'model_config': {
                'input_size': len(data_dict['feature_cols']),
                'output_size': model.n_targets,
                'hidden_size': model.lstm_units,
                'num_layers': 2,
                'dropout_rate': model.dropout_rate
            }
        }, model_path)
        
        # Save scalers
        scaler_path = 'models/lstm_scalers.joblib'
        joblib.dump({
            'feature_scaler': model.feature_scaler,
            'target_scaler': model.target_scaler,
            'sequence_length': model.sequence_length
        }, scaler_path)
        
        logger.info(f"PyTorch LSTM model saved: {model_path}")
        logger.info(f"Scalers saved: {scaler_path}")
        
        # Comprehensive Test Dataset Evaluation
        print(f"\n🧪 COMPREHENSIVE TEST DATASET EVALUATION")
        print("=" * 55)
        test_data = data_dict['test_data']
        
        # Evaluate model on test dataset with detailed analysis
        test_results = evaluate_on_test_data(
            model, test_data, data_dict['feature_cols'], target_cols, 
            use_advanced_features=not args.no_advanced_features
        )
        
        # Extract predictions for compatibility with existing code
        if not args.no_advanced_features:
            try:
                test_enhanced = create_advanced_features(test_data, data_dict['feature_cols'])
                available_features = [col for col in data_dict['feature_cols'] if col in test_enhanced.columns]
                X_test = test_enhanced[available_features].values.astype(np.float32)
            except Exception as e:
                logger.warning(f"⚠️ Using basic features for predictions: {e}")
                basic_features = [col for col in data_dict['feature_cols'] if col in test_data.columns]
                X_test = test_data[basic_features].values.astype(np.float32)
        else:
            basic_features = [col for col in data_dict['feature_cols'] if col in test_data.columns]
            X_test = test_data[basic_features].values.astype(np.float32)
        
        predictions = model.predict(X_test)
        
        # Create predictions DataFrame - handle length mismatches
        date_ids = test_data['date_id'].values
        n_test_samples = len(date_ids)
        
        # Ensure predictions match the test data length
        if predictions is not None and len(predictions) > 0:
            if len(predictions) != n_test_samples:
                # Pad or truncate predictions to match test data length
                if len(predictions) < n_test_samples:
                    # Pad with NaN
                    if predictions.ndim == 1:
                        padded_pred = np.full(n_test_samples, np.nan)
                        padded_pred[:len(predictions)] = predictions
                        predictions = padded_pred
                    else:
                        padded_pred = np.full((n_test_samples, predictions.shape[1]), np.nan)
                        padded_pred[:len(predictions), :] = predictions
                        predictions = padded_pred
                else:
                    # Truncate
                    predictions = predictions[:n_test_samples]
        else:
            # No predictions available - create NaN array
            if args.n_targets == 1:
                predictions = np.full(n_test_samples, np.nan)
            else:
                predictions = np.full((n_test_samples, args.n_targets), np.nan)
        
        # Create predictions DataFrame
        pred_data = {'date_id': date_ids}
        
        if args.n_targets == 1:
            pred_data[f'{target_cols[0]}_pred'] = predictions
        else:
            for i, target_col in enumerate(target_cols):
                if predictions.ndim == 1:
                    pred_data[f'{target_col}_pred'] = predictions
                else:
                    pred_data[f'{target_col}_pred'] = predictions[:, i] if predictions.shape[1] > i else np.full(n_test_samples, np.nan)
        
        pred_df = pd.DataFrame(pred_data)
        
        logger.info(f"Generated {len(pred_df)} test predictions")
        logger.info(f"Valid predictions: {np.sum(~np.isnan(predictions))}/{len(predictions)}")
        
        # Generate comprehensive competition analysis
        if correlation is not None and hasattr(model, 'validation_metrics'):
            model_name = f"LSTM_{args.n_targets}targets_{args.lstm_units}units"
            print_competition_analysis(model.validation_metrics, model_name, detailed=True)
            
            # Save competition results
            results_path = save_competition_results(model.validation_metrics, model_name)
            logger.info(f"Competition results saved to: {results_path}")
        
        # Test set analysis (if we have ground truth)
        logger.info(f"\n📊 Test Set Analysis:")
        if args.n_targets == 1:
            valid_preds = np.sum(~np.isnan(predictions))
            logger.info(f"   Valid predictions: {valid_preds}/{len(predictions)} ({valid_preds/len(predictions)*100:.1f}%)")
            if valid_preds > 0:
                pred_stats = predictions[~np.isnan(predictions)]
                logger.info(f"   Prediction range: [{pred_stats.min():.4f}, {pred_stats.max():.4f}]")
                logger.info(f"   Prediction mean: {pred_stats.mean():.4f}, std: {pred_stats.std():.4f}")
        else:
            for i, target_col in enumerate(target_cols):
                if predictions.ndim == 1:
                    valid_preds = np.sum(~np.isnan(predictions))
                else:
                    valid_preds = np.sum(~np.isnan(predictions[:, i]))
                logger.info(f"   {target_col} - Valid predictions: {valid_preds}/{len(predictions)} ({valid_preds/len(predictions)*100:.1f}%)")
        
        # Create comprehensive visualizations
        if not args.no_plots and correlation is not None:
            model_name = f"LSTM_{args.n_targets}targets_{args.lstm_units}units"
            create_comprehensive_visualizations(
                model, predictions, model.validation_metrics, target_cols, model_name
            )
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 ENHANCED LSTM TRAINING COMPLETED")
        print("=" * 60)
        print(f"✅ Targets: {len(target_cols)} ({', '.join(target_cols[:3])}{'...' if len(target_cols) > 3 else ''})")
        print(f"✅ Model: Multi-target LSTM with {args.lstm_units} units")
        print(f"✅ Advanced features: {'Enabled' if not args.no_advanced_features else 'Disabled'}")
        if hasattr(model, 'validation_metrics') and 'official_competition_score' in model.validation_metrics:
            print(f"✅ OFFICIAL Competition Score: {model.validation_metrics['official_competition_score']:.6f}")
        print(f"✅ Validation Score: {correlation:.4f}" if correlation else "❌ No validation score")
        print(f"✅ Validation RMSE: {rmse:.6f}" if rmse else "❌ No validation RMSE")
        print(f"✅ Test predictions: {len(predictions)} samples")
        print(f"✅ Test evaluation: {'Completed' if 'test_results' in locals() else 'Skipped'}")
        print(f"✅ Training time: {total_time:.2f}s")
        
        if not args.no_plots:
            print(f"\n📊 VISUALIZATION OUTPUTS:")
            print(f"   📁 All charts saved to: figures/ directory")
            print(f"   🔍 Training overview: figures/LSTM_{args.n_targets}targets_{args.lstm_units}units_training_overview.png")
            print(f"   📈 Target performance: figures/LSTM_{args.n_targets}targets_{args.lstm_units}units_target_performance.png")
            print(f"   🔬 Prediction analysis: figures/LSTM_{args.n_targets}targets_{args.lstm_units}units_prediction_analysis.png")
        
        print("=" * 60)
        
        # Competition readiness check using OFFICIAL score
        if correlation and hasattr(model, 'validation_metrics'):
            official_score = model.validation_metrics.get('official_competition_score', correlation)
            
            print(f"\n🎯 Final Competition Assessment (Official Score: {official_score:.6f})")
            
            if official_score > 0.5:
                print("🏆 OUTSTANDING - Exceptional competition performance!")
            elif official_score > 0.2:
                print("� EXCELLENT - Strong competition performance expected!")
            elif official_score > 0.1:
                print("✅ GOOD - Competitive performance expected!")
            elif official_score > 0.0:
                print("⚡ FAIR - May benefit from further optimization.")
            else:
                print("❌ POOR - Requires significant improvement for competition.")
            
            # Additional insights based on daily performance
            if 'positive_day_ratio' in model.validation_metrics:
                positive_ratio = model.validation_metrics['positive_day_ratio']
                if positive_ratio > 0.6:
                    print(f"✨ Consistent daily performance: {positive_ratio*100:.1f}% positive days")
                elif positive_ratio > 0.4:
                    print(f"🔄 Mixed daily performance: {positive_ratio*100:.1f}% positive days")
                else:
                    print(f"⚠️ Inconsistent daily performance: {positive_ratio*100:.1f}% positive days")
        
    except ValueError as e:
        if "Not enough data" in str(e) or "Dataset too small" in str(e):
            logger.error(f"❌ Dataset size error: {e}")
            logger.info(f"💡 Quick fixes:")
            logger.info(f"   • Try: --time-fraction 0.3 (for 30% of data)")
            logger.info(f"   • Try: --sequence-length 10 (shorter sequences)")
            logger.info(f"   • Try: --n-targets 50 (fewer targets for testing)")
            logger.info(f"   • Example: python src/train_LSTM.py --time-fraction 0.3 --n-targets 50 --sequence-length 10")
            return
        else:
            logger.error(f"LSTM training failed: {e}")
            raise
    except ValueError as e:
        if "Dataset too small" in str(e) or "Not enough data" in str(e) or "Training size too small" in str(e):
            logger.error(f"❌ Dataset size error: {e}")
            logger.info(f"💡 Quick fixes:")
            logger.info(f"   • Try: --time-fraction 0.3 (for 30% of data)")
            logger.info(f"   • Try: --sequence-length 10 (shorter sequences)")
            logger.info(f"   • Try: --n-targets 50 (fewer targets for testing)")
            logger.info(f"   • Example: python src/train_LSTM.py --time-fraction 0.3 --n-targets 50 --sequence-length 10")
            return
        else:
            logger.error(f"LSTM training failed: {e}")
            raise
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        raise

if __name__ == "__main__":
    main()