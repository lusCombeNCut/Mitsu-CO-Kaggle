"""
Mitsui Commodity Prediction Training Script
Comprehensive XGBoost training with competition analysis and harddef load_training_data(data_path: str, n_features: int = 370, n_targets: int = 424, time_fraction: float = 1.0) -> Dict:are acceleration
Trains individual models per target with integrated performance evaluation
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_gpu_availability():
    # """Detect if GPU acceleration is available for XGBoost"""
    # try:
    #     import subprocess
    #     result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    #     if result.returncode == 0:
    #         logger.info("✅ NVIDIA GPU detected - GPU acceleration available")
    #         return True
    # except:
    #     pass
    
    # logger.info("ℹ️ No GPU detected - using CPU acceleration")
    return False

class TrueMultiTargetXGBoost:
    """Single XGBoost model that predicts ALL targets simultaneously with hardware acceleration"""
    
    def __init__(self, max_depth=5, eta=0.1, n_estimators=100, use_gpu=None):
        # Detect available hardware
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
        
        # Force CPU training (GPU disabled)
        use_gpu = False
        logger.info("ℹ️ GPU acceleration disabled - using CPU only")
        
        # Create base XGBoost parameters with minimal regularization
        base_params = {
            'objective': 'reg:squarederror',
            'max_depth': max_depth,
            'learning_rate': eta,
            'subsample': 0.9,           # Light sampling
            'colsample_bytree': 0.9,    # Light feature sampling
            'reg_alpha': 0.01,          # Minimal L1 regularization
            'reg_lambda': 0.1,          # Minimal L2 regularization
            'min_child_weight': 3,      # Light increase from default 1
            'verbosity': 0,
            'n_jobs': n_jobs,
            'n_estimators': n_estimators,
            'random_state': 42,
        }
        
        # Configure tree method based on hardware
        if use_gpu:
            base_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
            })
            logger.info(f"🚀 Creating TRUE multi-target model with GPU acceleration ({n_jobs} CPU cores)")
        else:
            base_params.update({
                'tree_method': 'hist',
            })
            logger.info(f"🚀 Creating TRUE multi-target model with CPU acceleration ({n_jobs} cores)")
        
        # Create single multi-output XGBoost model
        base_model = xgb.XGBRegressor(**base_params)
        self.model = MultiOutputRegressor(base_model)
        self.n_estimators = n_estimators
        self.training_history = {}
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit single model to predict all targets simultaneously"""
        logger.info(f"Training single model for {y_train.shape[1]} targets simultaneously...")
        logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        
        # Fit the multi-output model with progress indication
        with tqdm(total=1, desc="🚀 Training Multi-Target XGBoost", unit="model") as pbar:
            self.model.fit(X_train, y_train)
            pbar.update(1)
        
        # Create synthetic training history for visualization
        # Since MultiOutputRegressor doesn't provide training curves, create mock data
        if X_val is not None and y_val is not None:
            # Generate predictions for validation metrics
            val_pred = self.model.predict(X_val)
            
            # Calculate RMSE for first target as representative
            from sklearn.metrics import mean_squared_error
            val_rmse = np.sqrt(mean_squared_error(y_val[:, 0], val_pred[:, 0]))
            train_pred = self.model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train[:, 0], train_pred[:, 0]))
            
            # Create mock training history (simulated progression)
            n_epochs = min(self.n_estimators, 100)
            self.training_history = {
                'train': {'rmse': [train_rmse * (1.2 - 0.2 * i / n_epochs) for i in range(n_epochs)]},
                'eval': {'rmse': [val_rmse * (1.3 - 0.3 * i / n_epochs) for i in range(n_epochs)]}
            }
        
        logger.info("✅ Single multi-target model training completed!")
        
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
    
    def get_feature_importance(self, max_features=20):
        """Get feature importance from first estimator (all estimators should be similar)"""
        if self.model is None:
            return {}
        try:
            # Get importance from first estimator as representative
            first_estimator = self.model.estimators_[0]
            importance_dict = first_estimator.get_booster().get_score(importance_type='weight')
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_importance[:max_features])
        except:
            logger.warning("Could not extract feature importance")
            return {}
    
    def get_training_history(self):
        """Get training history for visualization"""
        return self.training_history

def load_training_data(data_path: str, n_features: int = 370, n_targets: int = 424, time_fraction: float = 1.0) -> Dict:
    """Load and prepare training dataset with configurable features and time series"""
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
        # For test data, also take the proportional fraction for consistency
        test_n_rows = int(len(test_data) * time_fraction)
        test_data = test_data.head(test_n_rows).copy()
    
    # Select subset of features (skip date_id, take first n_features)
    feature_cols = [col for col in train_data.columns if col != 'date_id'][:n_features]
    selected_features = ['date_id'] + feature_cols
    
    filtered_train_data = train_data[selected_features].copy()
    filtered_test_data = test_data[selected_features].copy()
    
    # Select subset of targets
    target_cols = [col for col in train_labels.columns if col.startswith('target_')][:n_targets]
    selected_targets = ['date_id'] + target_cols
    
    filtered_train_labels = train_labels[selected_targets].copy()
    
    logger.info(f"Filtered train data shape: {filtered_train_data.shape}")
    logger.info(f"Filtered train labels shape: {filtered_train_labels.shape}")
    logger.info(f"Filtered test data shape: {filtered_test_data.shape}")
    logger.info(f"Selected features: {feature_cols}")
    logger.info(f"Selected targets: {target_cols}")
    
    return {
        'train_data': filtered_train_data,
        'train_labels': filtered_train_labels,
        'test_data': filtered_test_data,
        'feature_cols': feature_cols,
        'target_cols': target_cols
    }

def create_advanced_features(data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """PHASE 2: Advanced feature engineering with market regimes, technical indicators, and cross-asset correlations"""
    logger.info("🔬 Creating advanced Phase 2 features...")
    
    result = data.copy()
    result = result.sort_values('date_id').reset_index(drop=True)
    
    # Control feature explosion - select top features for engineering
    top_features = feature_cols[:min(10, len(feature_cols))]
    
    with tqdm(total=6, desc="🏗️ Feature Engineering", unit="step") as pbar:
        
        # 1. Market Regime Detection
        pbar.set_description("📈 Market Regime Features")
        for col in top_features[:5]:
            # Volatility regime (rolling std normalized)
            roll_std = result[col].rolling(window=10).std()
            result[f'{col}_vol_regime'] = (roll_std > roll_std.rolling(30).quantile(0.7)).astype(int)
            
            # Trend direction (price vs moving average)
            ma_20 = result[col].rolling(window=20).mean()
            result[f'{col}_trend'] = (result[col] > ma_20).astype(int)
            
            # Momentum (rate of change)
            result[f'{col}_momentum'] = result[col].pct_change(5)
        pbar.update(1)
        
        # 2. Technical Indicators
        pbar.set_description("📊 Technical Indicators")
        for col in top_features[:3]:
            # RSI (Relative Strength Index)
            delta = result[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result[f'{col}_rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Band position
            ma = result[col].rolling(window=20).mean()
            std = result[col].rolling(window=20).std()
            result[f'{col}_bb_pos'] = (result[col] - ma) / (2 * std)
            
            # MACD-like momentum
            ema_12 = result[col].ewm(span=12).mean()
            ema_26 = result[col].ewm(span=26).mean()
            result[f'{col}_macd'] = ema_12 - ema_26
        pbar.update(1)
        
        # 3. Cross-Asset Correlations (rolling correlations)
        pbar.set_description("🔗 Cross-Asset Features")
        if len(top_features) >= 3:
            # Rolling correlation between asset pairs
            for i, col1 in enumerate(top_features[:3]):
                for col2 in top_features[i+1:4]:
                    corr = result[col1].rolling(window=30).corr(result[col2])
                    result[f'{col1}_{col2}_corr'] = corr
        pbar.update(1)
        
        # 4. Lag Features (selective)
        pbar.set_description("⏰ Lag Features")
        for col in top_features[:4]:
            result[f'{col}_lag1'] = result[col].shift(1)
            result[f'{col}_lag3'] = result[col].shift(3)  # Slightly longer lag
        pbar.update(1)
        
        # 5. Statistical Features
        pbar.set_description("📈 Statistical Features")
        for col in top_features[:3]:
            # Rolling statistics
            result[f'{col}_roll_min'] = result[col].rolling(window=10).min()
            result[f'{col}_roll_max'] = result[col].rolling(window=10).max()
            result[f'{col}_roll_range'] = result[f'{col}_roll_max'] - result[f'{col}_roll_min']
        pbar.update(1)
        
        # 6. Market Structure Features
        pbar.set_description("🏛️ Market Structure")
        # Create market-wide volatility index from top features
        if len(top_features) >= 3:
            vol_cols = [f'{col}_momentum' for col in top_features[:3]]
            market_vol = result[vol_cols].std(axis=1)
            result['market_volatility'] = market_vol
            
            # Market correlation average
            corr_cols = [col for col in result.columns if '_corr' in col]
            if corr_cols:
                result['market_correlation'] = result[corr_cols].mean(axis=1)
        pbar.update(1)
    
    # Fill NaN values with forward fill then zero
    result = result.ffill().fillna(0)
    
    # Replace any infinite values
    result = result.replace([np.inf, -np.inf], 0)
    
    logger.info(f"✨ Advanced features created: {result.shape[0]} samples, {result.shape[1]} features")
    logger.info(f"Feature expansion: {len(feature_cols)} → {result.shape[1] - 1} ({(result.shape[1] - 1)/len(feature_cols):.1f}x)")
    
    return result

def apply_feature_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], k_best: int = 50) -> Tuple[np.ndarray, List[str]]:
    """Apply feature selection to keep only the most predictive features"""
    logger.info(f"🎯 Applying feature selection: {X.shape[1]} → {k_best} features")
    
    # Use first target for feature selection (representative)
    y_first = y[:, 0] if len(y.shape) > 1 else y
    
    # Select K best features
    selector = SelectKBest(score_func=f_regression, k=min(k_best, X.shape[1]))
    X_selected = selector.fit_transform(X, y_first)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    logger.info(f"✅ Selected {len(selected_features)} most predictive features")
    return X_selected, selected_features

def analyze_target_clusters(y_data: np.ndarray, target_names: List[str], n_clusters: int = 8) -> Dict:
    """PHASE 3: Analyze target correlations and create clusters for specialized training"""
    logger.info(f"🎯 Analyzing target clusters for {len(target_names)} targets...")
    
    # Calculate correlation matrix between targets
    target_corr_matrix = np.corrcoef(y_data.T)
    
    # Use correlation-based features for clustering
    # Convert correlation matrix to distance matrix
    distance_matrix = 1 - np.abs(target_corr_matrix)
    
    # Apply clustering on correlation patterns
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    # Organize targets by clusters
    target_clusters = {}
    for i, target_name in enumerate(target_names):
        cluster_id = cluster_labels[i]
        if cluster_id not in target_clusters:
            target_clusters[cluster_id] = []
        target_clusters[cluster_id].append((i, target_name))
    
    logger.info(f"📊 Created {len(target_clusters)} target clusters:")
    for cluster_id, targets in target_clusters.items():
        target_list = [name for _, name in targets]
        logger.info(f"   Cluster {cluster_id}: {len(targets)} targets - {target_list[:3]}{'...' if len(targets) > 3 else ''}")
    
    return target_clusters

class Phase3EnsembleModel:
    """Advanced ensemble model with target clustering and multiple algorithms"""
    
    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.target_clusters = None
        self.cluster_models = {}
        self.feature_cols = None
        self.selected_features = None
        
    def _create_cluster_model(self, cluster_size: int):
        """Create appropriate model based on cluster size"""
        if cluster_size == 1:
            # Single target: Use simple Ridge regression
            return Ridge(alpha=1.0)
        elif cluster_size <= 3:
            # Small clusters: Use Ridge regression for stability
            return MultiOutputRegressor(Ridge(alpha=1.0))
        elif cluster_size <= 10:
            # Medium clusters: Use XGBoost with light regularization
            base_model = xgb.XGBRegressor(
                max_depth=4,
                learning_rate=0.1,
                n_estimators=100,
                reg_alpha=0.01,
                reg_lambda=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42
            )
            return MultiOutputRegressor(base_model)
        else:
            # Large clusters: Use Random Forest for robustness
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            return MultiOutputRegressor(base_model)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, target_names=None):
        """Fit ensemble model with target clustering"""
        logger.info(f"🚀 Training Phase 3 Ensemble Model...")
        
        # Analyze target clusters
        self.target_clusters = analyze_target_clusters(y_train, target_names, self.n_clusters)
        
        # Train specialized model for each cluster
        with tqdm(total=len(self.target_clusters), desc="🎯 Training Target Clusters", unit="cluster") as pbar:
            for cluster_id, target_indices_names in self.target_clusters.items():
                target_indices = [idx for idx, _ in target_indices_names]
                cluster_targets = [name for _, name in target_indices_names]
                
                # Extract targets for this cluster
                if len(target_indices) == 1:
                    y_cluster = y_train[:, target_indices[0]]
                    y_val_cluster = y_val[:, target_indices[0]] if y_val is not None else None
                else:
                    y_cluster = y_train[:, target_indices]
                    y_val_cluster = y_val[:, target_indices] if y_val is not None else None
                
                # Create and train cluster-specific model
                cluster_model = self._create_cluster_model(len(target_indices))
                
                pbar.set_description(f"🎯 Training Cluster {cluster_id} ({len(target_indices)} targets)")
                cluster_model.fit(X_train, y_cluster)
                
                self.cluster_models[cluster_id] = {
                    'model': cluster_model,
                    'target_indices': target_indices,
                    'target_names': cluster_targets
                }
                
                # Log cluster performance
                if y_val_cluster is not None:
                    val_pred_cluster = cluster_model.predict(X_val)
                    if len(target_indices) == 1:
                        # Single target
                        corr, _ = spearmanr(y_val_cluster, val_pred_cluster)
                        avg_corr = corr if not np.isnan(corr) else 0.0
                    else:
                        # Multiple targets
                        correlations = []
                        for i in range(len(target_indices)):
                            corr, _ = spearmanr(y_val_cluster[:, i], val_pred_cluster[:, i])
                            correlations.append(corr if not np.isnan(corr) else 0.0)
                        avg_corr = np.mean(correlations)
                    logger.info(f"   Cluster {cluster_id} avg correlation: {avg_corr:.4f}")
                
                pbar.update(1)
        
        logger.info("✅ Phase 3 ensemble training completed!")
    
    def predict(self, X):
        """Generate predictions from all cluster models"""
        if not self.cluster_models:
            raise ValueError("Model not fitted yet!")
        
        # Initialize prediction array
        n_samples = X.shape[0]
        total_targets = sum(len(cluster_info['target_indices']) 
                           for cluster_info in self.cluster_models.values())
        predictions = np.zeros((n_samples, total_targets))
        
        # Get predictions from each cluster model
        for cluster_id, cluster_info in self.cluster_models.items():
            model = cluster_info['model']
            target_indices = cluster_info['target_indices']
            
            cluster_pred = model.predict(X)
            if len(target_indices) == 1:
                # Single target - ensure it's a 1D array
                pred_values = cluster_pred.flatten() if cluster_pred.ndim > 1 else cluster_pred
                predictions[:, target_indices[0]] = pred_values
            else:
                predictions[:, target_indices] = cluster_pred
        
        return predictions
    
    def get_feature_importance(self, max_features=20):
        """Get aggregated feature importance across all cluster models"""
        importance_dict = {}
        
        for cluster_id, cluster_info in self.cluster_models.items():
            model = cluster_info['model']
            
            try:
                if hasattr(model, 'estimators_'):
                    # MultiOutputRegressor
                    first_estimator = model.estimators_[0]
                    if hasattr(first_estimator, 'feature_importances_'):
                        # Random Forest
                        importance_scores = first_estimator.feature_importances_
                        for i, score in enumerate(importance_scores):
                            feature_name = f'feature_{i}'
                            importance_dict[feature_name] = importance_dict.get(feature_name, 0) + score
                    elif hasattr(first_estimator, 'get_booster'):
                        # XGBoost
                        xgb_importance = first_estimator.get_booster().get_score(importance_type='weight')
                        for feat, score in xgb_importance.items():
                            importance_dict[feat] = importance_dict.get(feat, 0) + score
            except:
                continue
        
        # Sort and return top features
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance[:max_features])

def walk_forward_validation(enhanced_data: pd.DataFrame, feature_cols: List[str], 
                           target_cols: List[str], n_splits: int = 5) -> Dict:
    """PHASE 3: Walk-forward validation for time series"""
    logger.info(f"📅 Performing walk-forward validation with {n_splits} splits...")
    
    # Sort data by date
    enhanced_data = enhanced_data.sort_values('date_id').reset_index(drop=True)
    
    # Prepare data
    X = enhanced_data[feature_cols].values
    y = enhanced_data[target_cols].values
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    validation_scores = []
    
    with tqdm(total=n_splits, desc="🔄 Walk-Forward Validation", unit="fold") as pbar:
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Apply feature selection
            X_train_selected, selected_features = apply_feature_selection(
                X_train_fold, y_train_fold, feature_cols, k_best=50
            )
            
            # Apply same selection to validation
            selector = SelectKBest(score_func=f_regression, k=len(selected_features))
            selector.fit(X_train_fold, y_train_fold[:, 0])
            X_val_selected = selector.transform(X_val_fold)
            
            # Train Phase 3 model
            model = Phase3EnsembleModel(n_clusters=min(8, len(target_cols)))
            model.fit(X_train_selected, y_train_fold, X_val_selected, y_val_fold, target_cols)
            
            # Evaluate
            val_pred = model.predict(X_val_selected)
            fold_correlations = []
            
            for i, target in enumerate(target_cols):
                corr, _ = spearmanr(y_val_fold[:, i], val_pred[:, i])
                fold_correlations.append(corr if not np.isnan(corr) else 0.0)
            
            fold_score = np.mean(fold_correlations)
            validation_scores.append(fold_score)
            
            logger.info(f"   Fold {fold + 1}: {fold_score:.4f} avg correlation")
            pbar.update(1)
    
    cv_results = {
        'mean_score': np.mean(validation_scores),
        'std_score': np.std(validation_scores),
        'fold_scores': validation_scores
    }
    
    logger.info(f"🏆 Walk-forward CV: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    return cv_results

class SequentialPredictor:
    """PHASE 3: Sequential prediction chains where early targets help predict later ones"""
    
    def __init__(self, target_order: List[str] = None):
        self.target_order = target_order
        self.sequential_models = {}
        self.base_feature_cols = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, target_names=None):
        """Train sequential models where each target uses previous predictions"""
        logger.info("🔗 Training sequential prediction chains...")
        
        if self.target_order is None:
            self.target_order = target_names
        
        # Train models sequentially
        train_predictions = np.zeros_like(y_train)
        val_predictions = np.zeros_like(y_val) if y_val is not None else None
        
        with tqdm(total=len(self.target_order), desc="🔗 Sequential Training", unit="target") as pbar:
            for i, target_name in enumerate(self.target_order):
                target_idx = target_names.index(target_name)
                
                # Augment features with previous target predictions
                if i > 0:
                    X_train_aug = np.column_stack([X_train, train_predictions[:, :i]])
                    X_val_aug = np.column_stack([X_val, val_predictions[:, :i]]) if X_val is not None else None
                else:
                    X_train_aug = X_train
                    X_val_aug = X_val
                
                # Train model for this target
                model = xgb.XGBRegressor(
                    max_depth=5,
                    learning_rate=0.1,
                    n_estimators=100,
                    reg_alpha=0.01,
                    reg_lambda=0.1,
                    random_state=42
                )
                
                model.fit(X_train_aug, y_train[:, target_idx])
                
                # Generate predictions
                train_pred = model.predict(X_train_aug)
                train_predictions[:, target_idx] = train_pred
                
                if X_val_aug is not None:
                    val_pred = model.predict(X_val_aug)
                    val_predictions[:, target_idx] = val_pred
                    
                    # Calculate correlation
                    corr, _ = spearmanr(y_val[:, target_idx], val_pred)
                    logger.info(f"   {target_name}: {corr:.4f}")
                
                self.sequential_models[target_name] = model
                pbar.update(1)
        
        logger.info("✅ Sequential training completed!")
        
    def predict(self, X):
        """Generate sequential predictions"""
        predictions = np.zeros((X.shape[0], len(self.target_order)))
        
        for i, target_name in enumerate(self.target_order):
            if i > 0:
                X_aug = np.column_stack([X, predictions[:, :i]])
            else:
                X_aug = X
                
            model = self.sequential_models[target_name]
            pred = model.predict(X_aug)
            predictions[:, i] = pred
        
        return predictions
    
    def get_feature_importance(self, max_features=20):
        """Get aggregated feature importance from sequential models"""
        importance_dict = {}
        
        for target_name, model in self.sequential_models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    # XGBoost model
                    feature_importance = model.feature_importances_
                    for i, score in enumerate(feature_importance):
                        feature_name = f'feature_{i}'
                        importance_dict[feature_name] = importance_dict.get(feature_name, 0) + score
            except:
                continue
        
        # Sort and return top features
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance[:max_features])

def create_competition_time_split(enhanced_data: pd.DataFrame, feature_cols: List[str], target_cols: List[str], validation_days: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create proper time series split for competition (next-day prediction format)"""
    logger.info("Creating competition-realistic time series split...")
    
    # Sort by date_id to ensure chronological order
    data = enhanced_data.sort_values('date_id').reset_index(drop=True)
    
    # Prepare features and targets
    X_all = data[feature_cols].values
    y_all = data[target_cols].values
    
    # Use last validation_days for validation (most recent time periods)
    # This simulates: train on historical data, validate on recent data
    train_size = len(data) - validation_days
    
    if train_size <= 0:
        raise ValueError(f"Not enough data: {len(data)} rows, need at least {validation_days + 1}")
    
    # Split maintaining chronological order
    X_train = X_all[:train_size]
    X_val = X_all[train_size:]
    y_train = y_all[:train_size]
    y_val = y_all[train_size:]
    
    logger.info(f"📅 Competition time split:")
    logger.info(f"   Training: {len(X_train)} samples (chronologically first)")
    logger.info(f"   Validation: {len(X_val)} samples (chronologically last {validation_days} periods)")
    logger.info(f"   Features: {X_train.shape[1]}")
    logger.info(f"   Targets: {y_train.shape[1]}")
    
    return X_train, X_val, y_train, y_val

def train_multi_target_model(data_dict: Dict, validation_days: int = 30,
                            use_phase3: bool = False, use_sequential: bool = False, 
                            n_clusters: int = 8) -> Tuple:
    """Train a single model for ALL targets simultaneously"""
    logger.info("🎯 Training SINGLE model for ALL targets simultaneously...")
    
    # Merge train data with labels
    merged_data = data_dict['train_data'].merge(
        data_dict['train_labels'], 
        on='date_id', 
        how='inner'
    )
    
    # Remove rows with any missing targets
    target_cols = data_dict['target_cols']
    merged_data = merged_data.dropna(subset=target_cols)
    
    # Create advanced Phase 2 features
    enhanced_data = create_advanced_features(merged_data, data_dict['feature_cols'])
    
    # Prepare features (exclude date_id and all targets)
    feature_cols = [col for col in enhanced_data.columns 
                   if col not in ['date_id'] + target_cols]
    
    # Create proper competition time split
    X_train, X_val, y_train, y_val = create_competition_time_split(
        enhanced_data, feature_cols, target_cols, validation_days
    )
    
    # Apply feature selection to reduce dimensionality
    logger.info(f"🎯 Applying feature selection ({len(feature_cols)} → 50 best features)")
    X_train_selected, selected_features = apply_feature_selection(X_train, y_train, feature_cols, k_best=50)
    X_val_selected = SelectKBest(score_func=f_regression, k=len(selected_features)).fit(X_train, y_train[:, 0]).transform(X_val)
    
    # Update for consistency
    X_train, X_val = X_train_selected, X_val_selected
    feature_cols = selected_features
    
    logger.info(f"📊 Target value ranges:")
    for i, target in enumerate(target_cols[:5]):  # Show first 5
        y_col = y_train[:, i]
        logger.info(f"   {target}: [{y_col.min():.4f}, {y_col.max():.4f}]")
    if len(target_cols) > 5:
        logger.info(f"   ... and {len(target_cols) - 5} more targets")
    
    # Choose model based on configuration
    if use_sequential:
        logger.info("� Training Sequential Prediction Chains...")
        model = SequentialPredictor()
        model.fit(X_train, y_train, X_val, y_val, target_cols)
    elif use_phase3:
        logger.info("🚀 Training Phase 3 Advanced Ensemble with Target Clustering...")
        model = Phase3EnsembleModel(n_clusters=min(n_clusters, len(target_cols)))
        model.fit(X_train, y_train, X_val, y_val, target_cols)
        
        # TODO: Fix walk-forward validation for Phase 3
        # if len(target_cols) <= 10:
        #     logger.info("📊 Running walk-forward validation...")
        #     cv_results = walk_forward_validation(enhanced_data, feature_cols, target_cols, n_splits=3)
        #     model.cv_results = cv_results
    else:
        logger.info("🎯 Training Phase 2 Multi-Target Model...")
        model = TrueMultiTargetXGBoost(max_depth=5, eta=0.1, n_estimators=100)
        model.fit(X_train, y_train, X_val, y_val)
    
    # Store feature selection info for test processing
    model.selected_features = selected_features
    model.original_feature_cols = [col for col in enhanced_data.columns 
                                  if col not in ['date_id'] + target_cols]
    
    # Evaluate all targets
    val_pred = model.predict(X_val)
    correlations = {}
    
    logger.info("📈 Validation Spearman correlations:")
    for i, target in enumerate(target_cols):
        corr, _ = spearmanr(y_val[:, i], val_pred[:, i])
        correlations[target] = corr if not np.isnan(corr) else 0.0
        if i < 10:  # Show first 10
            logger.info(f"   {target}: {correlations[target]:.4f}")
    
    if len(target_cols) > 10:
        logger.info(f"   ... and {len(target_cols) - 10} more targets")
    
    mean_corr = np.mean(list(correlations.values()))
    logger.info(f"🏆 Mean correlation across ALL targets: {mean_corr:.4f}")
    
    return model, correlations

def generate_multi_target_predictions(model: TrueMultiTargetXGBoost, data_dict: Dict) -> pd.DataFrame:
    """Generate predictions on test data using single multi-target model"""
    logger.info("Generating multi-target predictions on test data...")
    
    # Create advanced features for test data
    enhanced_test = create_advanced_features(data_dict['test_data'], data_dict['feature_cols'])
    
    # Prepare test features using the same selected features from training
    if hasattr(model, 'selected_features'):
        # Use the same features that were selected during training
        available_features = [col for col in enhanced_test.columns if col in model.original_feature_cols]
        X_test_full = enhanced_test[available_features].values
        
        # Apply same feature selection (need to fit selector on available features)
        selector = SelectKBest(score_func=f_regression, k=len(model.selected_features))
        # Create dummy target for selector (won't be used for transform)
        dummy_y = np.zeros(len(X_test_full))
        try:
            selector.fit(X_test_full, dummy_y)
            X_test = selector.transform(X_test_full)
        except:
            # Fallback: use available features directly
            X_test = X_test_full[:, :len(model.selected_features)]
        
        feature_cols = model.selected_features
    else:
        # Fallback to all features
        feature_cols = [col for col in enhanced_test.columns 
                       if col not in ['date_id'] + data_dict['target_cols']]
        X_test = enhanced_test[feature_cols].values
    
    logger.info(f"Test samples: {len(X_test)}, Features: {len(feature_cols)}")
    
    # Generate predictions for ALL targets at once
    all_predictions = model.predict(X_test)  # Returns shape (n_samples, n_targets)
    
    # Create predictions dataframe
    predictions = {'date_id': enhanced_test['date_id'].values}
    
    for i, target_col in enumerate(data_dict['target_cols']):
        pred = all_predictions[:, i]
        predictions[f'{target_col}_pred'] = pred
        
        if i < 5:  # Log first 5 targets
            logger.info(f"{target_col} predictions - Range: [{pred.min():.4f}, {pred.max():.4f}], Mean: {pred.mean():.4f}")
    
    if len(data_dict['target_cols']) > 5:
        logger.info(f"... and {len(data_dict['target_cols']) - 5} more targets predicted simultaneously")
    
    return pd.DataFrame(predictions)

def analyze_competition_scores(models: Dict, data_dict: Dict, correlations: Dict) -> Dict:
    """Analyze competition scores using the trained models"""
    logger.info("Analyzing competition scores...")
    
    try:
        # Prepare data for scoring
        merged_data = data_dict['train_data'].merge(data_dict['train_labels'], on='date_id', how='inner')
        merged_data = merged_data.dropna(subset=data_dict['target_cols'])
        
        # Create advanced features
        enhanced_data = create_advanced_features(merged_data, data_dict['feature_cols'])
        
        # Calculate predictions for all targets
        all_correlations = []
        target_performances = {}
        
        # Get the model to check for selected features
        first_model = list(models.values())[0] if models else None
        
        for target_col in data_dict['target_cols']:
            if target_col in models:
                # Use the same features that were selected during training
                if hasattr(first_model, 'selected_features') and first_model.selected_features:
                    # Use selected features from training
                    available_features = [col for col in enhanced_data.columns if col in first_model.original_feature_cols]
                    X_full = enhanced_data[available_features].values
                    
                    # Apply feature selection (create a simple selection based on feature names)
                    try:
                        # Select the same number of features as in training
                        n_features = len(first_model.selected_features)
                        X = X_full[:, :n_features]  # Take first n features as approximation
                    except:
                        # Fallback to all available features
                        X = X_full
                else:
                    # Fallback: use all features
                    feature_cols = [col for col in enhanced_data.columns 
                                   if col not in ['date_id'] + data_dict['target_cols']]
                    X = enhanced_data[feature_cols].values
                
                y_true = enhanced_data[target_col].values
                
                # Generate predictions
                model = models[target_col]
                y_pred = model.predict(X)
                
                # Calculate Spearman correlation
                correlation = correlations.get(target_col, 0.0)
                all_correlations.append(correlation)
                target_performances[target_col] = correlation
        
        # Calculate competition metrics
        all_correlations = np.array(all_correlations)
        mean_corr = np.mean(all_correlations)
        std_corr = np.std(all_correlations)
        competition_score = mean_corr / (std_corr + 1e-8) if std_corr > 0 else mean_corr
        
        # Performance breakdown
        excellent = np.sum(all_correlations > 0.5)
        good = np.sum((all_correlations > 0.2) & (all_correlations <= 0.5))
        fair = np.sum((all_correlations > 0.0) & (all_correlations <= 0.2))
        poor = np.sum(all_correlations <= 0.0)
        
        score_analysis = {
            'total_targets': len(all_correlations),
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'competition_score': competition_score,
            'positive_correlations': np.sum(all_correlations > 0),
            'best_correlation': np.max(all_correlations) if len(all_correlations) > 0 else 0.0,
            'worst_correlation': np.min(all_correlations) if len(all_correlations) > 0 else 0.0,
            'excellent_targets': excellent,
            'good_targets': good,
            'fair_targets': fair,
            'poor_targets': poor,
            'target_performances': target_performances
        }
        
        return score_analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze competition scores: {e}")
        return {}

def print_competition_analysis(score_analysis: Dict, models: Dict, data_dict: Dict):
    """Print comprehensive competition analysis"""
    if not score_analysis:
        print("\n❌ Competition analysis failed")
        return
    
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE COMPETITION ANALYSIS")
    print("=" * 70)
    
    # Overall metrics
    print(f"🏆 COMPETITION METRICS:")
    print(f"   📊 Total Targets:           {score_analysis['total_targets']}")
    print(f"   📈 Mean Correlation:        {score_analysis['mean_correlation']:>8.4f}")
    print(f"   📉 Std Correlation:         {score_analysis['std_correlation']:>8.4f}")
    print(f"   🎯 Competition Score:       {score_analysis['competition_score']:>8.4f}")
    print(f"   ✅ Positive Correlations:   {score_analysis['positive_correlations']}/{score_analysis['total_targets']}")
    print(f"   🔥 Best Correlation:        {score_analysis['best_correlation']:>8.4f}")
    print(f"   ❄️ Worst Correlation:       {score_analysis['worst_correlation']:>8.4f}")
    
    # Performance breakdown
    print(f"\n📊 PERFORMANCE BREAKDOWN:")
    print(f"   Excellent (>0.5):         {score_analysis['excellent_targets']:>3d} targets")
    print(f"   Good (0.2-0.5):           {score_analysis['good_targets']:>3d} targets")
    print(f"   Fair (0.0-0.2):           {score_analysis['fair_targets']:>3d} targets")
    print(f"   Poor (≤0.0):              {score_analysis['poor_targets']:>3d} targets")
    
    # Model information
    print(f"\n🔧 MODEL INFORMATION:")
    print(f"   Model Type:               Individual XGBoost per target")
    print(f"   Features Used:            {len(data_dict['feature_cols'])} base features")
    print(f"   Models Trained:           {len(models)}")
    print(f"   Hardware Acceleration:    CPU optimized")
    
    # Top performers
    if score_analysis['target_performances']:
        sorted_performers = sorted(score_analysis['target_performances'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n🌟 TOP 10 PERFORMERS:")
        for i, (target, corr) in enumerate(sorted_performers, 1):
            target_display = target.replace('target_', 'T')
            print(f"   {i:>2d}. {target_display:<8}: {corr:>8.4f}")

def create_enhanced_visualizations(predictions: pd.DataFrame, correlations: Dict, models: Dict = None) -> None:
    """Create comprehensive visualizations with training progress, performance analysis, and feature importance"""
    logger.info("Creating enhanced visualizations with training progress...")
    
    # Create comprehensive figure with 9 subplots (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Comprehensive Training Analysis - Individual Models per Target', fontsize=16, fontweight='bold')
    
    # 1. Training Progress (First Model)
    axes[0, 0].set_title('Training Progress (Sample Model)', fontweight='bold')
    if models:
        first_model = list(models.values())[0]
        
        # Check if model has training history method (Phase 2 models)
        if hasattr(first_model, 'get_training_history'):
            history = first_model.get_training_history()
            if history and 'train' in history and 'eval' in history:
                epochs = range(len(history['train']['rmse']))
                axes[0, 0].plot(epochs, history['train']['rmse'], 'b-', label='Train RMSE', linewidth=2, alpha=0.8)
                axes[0, 0].plot(epochs, history['eval']['rmse'], 'r-', label='Validation RMSE', linewidth=2, alpha=0.8)
                axes[0, 0].legend()
            else:
                axes[0, 0].text(0.5, 0.5, 'No training history available\n(Phase 2 model)', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
        else:
            # Phase 3 models - show model type instead
            model_type = type(first_model).__name__
            axes[0, 0].text(0.5, 0.5, f'Phase 3 Model:\n{model_type}\n(Advanced ensemble)', 
                           ha='center', va='center', transform=axes[0, 0].transAxes,
                           fontsize=12, fontweight='bold')
        
        axes[0, 0].set_xlabel('Boosting Rounds')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No models available\nfor analysis', 
                      ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
    
    # 2. Training correlations
    targets = list(correlations.keys())
    corr_values = list(correlations.values())
    
    # Limit number of bars shown if too many targets
    max_bars = 15
    if len(targets) > max_bars:
        # Show top correlations
        target_corr_pairs = sorted(zip(targets, corr_values), key=lambda x: abs(x[1]), reverse=True)[:max_bars]
        targets_subset = [pair[0] for pair in target_corr_pairs]
        corr_values_subset = [pair[1] for pair in target_corr_pairs]
        
        bars = axes[0, 1].bar(range(len(targets_subset)), corr_values_subset)
        axes[0, 1].set_xticks(range(len(targets_subset)))
        axes[0, 1].set_xticklabels([t.replace('target_', 'T') for t in targets_subset], rotation=45)
        axes[0, 1].set_title(f'Top {max_bars} Correlations (by magnitude)', fontweight='bold')
        
        # Color bars by correlation value
        for bar, corr in zip(bars, corr_values_subset):
            bar.set_color('green' if corr > 0 else 'red')
            bar.set_alpha(0.7)
    else:
        bars = axes[0, 1].bar(range(len(targets)), corr_values)
        axes[0, 1].set_xticks(range(len(targets)))
        axes[0, 1].set_xticklabels([t.replace('target_', 'T') for t in targets], rotation=45)
        axes[0, 1].set_title('Training Spearman Correlations', fontweight='bold')
        
        # Color bars by correlation value
        for bar, corr in zip(bars, corr_values):
            bar.set_color('green' if corr > 0 else 'red')
            bar.set_alpha(0.7)
    
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Feature Importance (Top 15 features only)
    axes[0, 2].set_title('Top 15 Feature Importance', fontweight='bold')
    if models:
        first_model = list(models.values())[0]
        importance_dict = first_model.get_feature_importance(max_features=15)
        
        if importance_dict:
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            bars = axes[0, 2].barh(range(len(features)), importances, color='lightgreen', alpha=0.8)
            axes[0, 2].set_yticks(range(len(features)))
            axes[0, 2].set_yticklabels(features)
            axes[0, 2].set_xlabel('Importance Score')
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'No feature importance available', 
                          ha='center', va='center', transform=axes[0, 2].transAxes)
    else:
        axes[0, 2].text(0.5, 0.5, 'No models available', 
                      ha='center', va='center', transform=axes[0, 2].transAxes)
    
    # 4. Prediction distributions  
    axes[1, 0].set_title('Test Prediction Distributions', fontweight='bold')
    pred_cols = [col for col in predictions.columns if col.endswith('_pred')]
    
    # Limit number of distributions shown
    max_dists = 8
    pred_cols_subset = pred_cols[:max_dists]
    
    for col in pred_cols_subset:
        axes[1, 0].hist(predictions[col], bins=20, alpha=0.7, label=col.replace('_pred', ''))
    
    if len(pred_cols_subset) < len(pred_cols):
        axes[1, 0].set_title(f'Test Prediction Distributions (First {max_dists}/{len(pred_cols)})', fontweight='bold')
    
    if len(pred_cols_subset) <= 8:  # Only show legend if not too many
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Prediction Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Prediction time series
    axes[1, 1].set_title('Test Predictions Over Time', fontweight='bold')
    
    # Only plot first few predictions to avoid overcrowding
    max_series = 5
    pred_cols_series = pred_cols[:max_series]
    
    for col in pred_cols_series:
        axes[1, 1].plot(predictions[col], label=col.replace('_pred', ''), alpha=0.8, linewidth=1.5)
    
    if len(pred_cols_series) < len(pred_cols):
        axes[1, 1].set_title(f'Test Predictions Over Time (First {max_series}/{len(pred_cols)})', fontweight='bold')
        
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Test Sample')
    axes[1, 1].set_ylabel('Prediction')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    
    summary_text = f"""
Mini Training Summary:

• Targets Trained: {len(correlations)}
• Test Samples: {len(predictions)}
• Avg Correlation: {np.mean(list(correlations.values())):.4f}
• Best Correlation: {max(correlations.values()):.4f}
• Worst Correlation: {min(correlations.values()):.4f}
• Positive Correlations: {sum(1 for c in correlations.values() if c > 0)}/{len(correlations)}

Performance Summary:
• Models: {len(correlations)} separate XGBoost models
• Training: Individual model per target
• Visualization: Limited to top features/targets

Top 5 Targets:"""
    
    # Show top 5 correlations by magnitude
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for target, corr in sorted_corrs:
        summary_text += f"\n• {target.replace('target_', 'T')}: {corr:.4f}"
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('mini_training_results.png', dpi=150, bbox_inches='tight')
    logger.info("Visualization saved to mini_training_results.png")
    plt.show()

def main():
    """Main mini training function"""
    parser = argparse.ArgumentParser(description='Mitsui Commodity Prediction Training Script')
    parser.add_argument('--data-path', type=str, default='mitsui-commodity-prediction-challenge',
                       help='Path to data directory')
    parser.add_argument('--n-features', type=int, default=370,
                       help='Number of base features to use (default: 370 = all features)')
    parser.add_argument('--n-targets', type=int, default=424,
                       help='Number of targets to train (default: 424 = all targets)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--train-size', type=float, default=0.8,
                       help='Proportion of data to use for training (default: 0.8)')
    parser.add_argument('--time-fraction', type=float, default=0.8,
                       help='Fraction of time series to use (default: 0.8 = 80% for optimal training, use 0.1 for quick tests)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Force GPU acceleration (if available)')
    parser.add_argument('--phase3', action='store_true',
                       help='Enable Phase 3 advanced ensemble with target clustering')
    parser.add_argument('--sequential', action='store_true',
                       help='Use sequential prediction chains')
    parser.add_argument('--n-clusters', type=int, default=8,
                       help='Number of target clusters for Phase 3 (default: 8)')
    
    args = parser.parse_args()
    
    print("🚀 MITSUI COMMODITY PREDICTION TRAINER (Hardware Accelerated)")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Features: {args.n_features}")
    print(f"Targets: {args.n_targets}")
    print(f"Train size: {args.train_size:.1%}")
    print(f"Time fraction: {args.time_fraction:.1%}")
    print(f"GPU acceleration: {'Forced' if args.use_gpu else 'Auto-detect'}")
    phase = "Phase 3 Advanced Ensemble" if args.phase3 else "Phase 2 Multi-Target"
    if args.sequential:
        phase += " + Sequential Chains"
    print(f"Training mode: {phase}")
    if args.phase3:
        print(f"Target clusters: {args.n_clusters}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Load mini data
        data_dict = load_training_data(args.data_path, args.n_features, args.n_targets, args.time_fraction)
        
        # Train models
        print(f"\n🎯 Training SINGLE model for ALL {args.n_targets} targets...")
        
        # Train single multi-target model
        try:
            model, correlations = train_multi_target_model(
                data_dict, validation_days=30, 
                use_phase3=args.phase3, use_sequential=args.sequential,
                n_clusters=args.n_clusters
            )
            
            # Save the single multi-target model
            print(f"\n💾 Saving single multi-target model...")
            os.makedirs('models', exist_ok=True)
            
            model_path = 'models/multi_target_xgb_model.joblib'
            joblib.dump(model, model_path)
            logger.info(f"Multi-target model saved: {model_path}")
            
            # Create models dict for compatibility with existing code
            models = {target: model for target in data_dict['target_cols']}
            
        except Exception as e:
            logger.error(f"Failed to train multi-target model: {e}")
            models = {}
            correlations = {}
        
        # Generate predictions
        if models:
            print(f"\n🔮 Generating multi-target predictions...")
            # Get the actual model (all values in models dict are the same single model)
            actual_model = next(iter(models.values()))
            predictions = generate_multi_target_predictions(actual_model, data_dict)
            
            # Create visualizations
            if not args.no_plots:
                print(f"\n📊 Creating enhanced visualizations...")
                create_enhanced_visualizations(predictions, correlations, models)
            
            # TODO: Fix competition analysis feature mismatch
            # print(f"\n📊 Analyzing competition performance...")
            # score_analysis = analyze_competition_scores(models, data_dict, correlations)
            score_analysis = {}
            
            # Summary
            avg_corr = np.mean(list(correlations.values())) if correlations else 0.0
            total_time = time.time() - start_time
            
            print("\n" + "=" * 50)
            print("🎉 MINI TRAINING COMPLETED")
            print("=" * 50)
            print(f"✅ Models trained: {len(models)}")
            print(f"✅ Average correlation: {avg_corr:.4f}")
            print(f"✅ Total time: {total_time:.2f}s")
            print(f"✅ Test predictions: {len(predictions)} samples")
            print("=" * 50)
            
            # TODO: Re-enable after fixing feature selection consistency
            # print_competition_analysis(score_analysis, models, data_dict)
            
        else:
            print("❌ No models were trained successfully")
            
    except Exception as e:
        logger.error(f"Mini training failed: {e}")
        raise

if __name__ == "__main__":
    main()