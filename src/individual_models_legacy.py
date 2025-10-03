"""
Multi-Target Training Script - Single model for multiple targets with enhanced visualization
Uses a single XGBoost model to predict multiple targets simultaneously with training progress tracking
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
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
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTargetXGBoostRegressor:
    """XGBoost regressor that can handle multiple targets simultaneously with training progress tracking"""
    
    def __init__(self, max_depth=6, eta=0.1, n_estimators=100, enable_categorical=False):
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': max_depth,
            'eta': eta,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'rmse',
            'verbosity': 1,  # Enable some verbosity to track progress
            'enable_categorical': enable_categorical
        }
        self.n_estimators = n_estimators
        self.models = {}  # Will store one model per target
        self.training_history = {}  # Store training metrics
        self.n_targets = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, target_names=None):
        """Fit models for multiple targets with progress tracking"""
        
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
            
        self.n_targets = y_train.shape[1]
        
        if target_names is None:
            target_names = [f'target_{i}' for i in range(self.n_targets)]
        
        logger.info(f"Training multi-target model for {self.n_targets} targets...")
        logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        
        # Store overall training history
        self.training_history = {
            'train_rmse': [],
            'val_rmse': [],
            'train_corr': [],
            'val_corr': [],
            'epochs': []
        }
        
        # Train one model per target but with shared progress tracking
        for i, target_name in enumerate(target_names):
            logger.info(f"Training model for {target_name} ({i+1}/{self.n_targets})")
            
            y_train_single = y_train[:, i]
            y_val_single = y_val[:, i] if y_val is not None else None
            
            dtrain = xgb.DMatrix(X_train, label=y_train_single)
            
            evals = [(dtrain, 'train')]
            eval_history = {}
            
            if X_val is not None and y_val_single is not None:
                dval = xgb.DMatrix(X_val, label=y_val_single)
                evals.append((dval, 'eval'))
            
            # Train with early stopping and evaluation tracking
            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                evals_result=eval_history,
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            self.models[target_name] = model
            
            # Store individual model history for this target
            if 'eval' in eval_history:
                train_rmse = eval_history['train']['rmse']
                val_rmse = eval_history['eval']['rmse']
                
                # Calculate correlations at final epoch
                train_pred = model.predict(dtrain)
                val_pred = model.predict(dval)
                
                train_corr = spearmanr(y_train_single, train_pred)[0]
                val_corr = spearmanr(y_val_single, val_pred)[0]
                
                logger.info(f"  Final RMSE - Train: {train_rmse[-1]:.4f}, Val: {val_rmse[-1]:.4f}")
                logger.info(f"  Final Corr - Train: {train_corr:.4f}, Val: {val_corr:.4f}")
                
                # For the first target, store the training curve for visualization
                if i == 0:
                    self.training_history['train_rmse'] = train_rmse
                    self.training_history['val_rmse'] = val_rmse
                    self.training_history['epochs'] = list(range(len(train_rmse)))
                    self.training_history['train_corr'].append(train_corr)
                    self.training_history['val_corr'].append(val_corr)
        
        logger.info("Multi-target training completed!")
        
    def predict(self, X):
        """Generate predictions for all targets"""
        if not self.models:
            raise ValueError("Model not fitted yet!")
        
        predictions = []
        target_names = list(self.models.keys())
        
        for target_name in target_names:
            model = self.models[target_name]
            dtest = xgb.DMatrix(X)
            pred = model.predict(dtest)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def get_feature_importance(self, max_features=20):
        """Get feature importance from the first model (representative)"""
        if not self.models:
            return {}
        
        first_model = list(self.models.values())[0]
        importance_dict = first_model.get_score(importance_type='weight')
        
        # Sort by importance and limit to top features
        sorted_importance = sorted(importance_dict.items(), 
                                 key=lambda x: x[1], reverse=True)[:max_features]
        
        return dict(sorted_importance)

def load_multi_target_data(data_path: str, n_features: int = 20, n_targets: int = 10, 
                          time_fraction: float = 1.0) -> Dict:
    """Load and prepare data for multi-target training"""
    logger.info(f"Loading multi-target dataset from {data_path}")
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
        test_n_rows = int(len(test_data) * time_fraction)
        test_data = test_data.head(test_n_rows).copy()
    
    # Select subset of features (skip date_id, take first n_features)
    feature_cols = [col for col in train_data.columns if col != 'date_id'][:n_features]
    selected_features = ['date_id'] + feature_cols
    
    multi_train_data = train_data[selected_features].copy()
    multi_test_data = test_data[selected_features].copy()
    
    # Select subset of targets
    target_cols = [col for col in train_labels.columns if col.startswith('target_')][:n_targets]
    selected_targets = ['date_id'] + target_cols
    
    multi_train_labels = train_labels[selected_targets].copy()
    
    logger.info(f"Multi-target train data shape: {multi_train_data.shape}")
    logger.info(f"Multi-target train labels shape: {multi_train_labels.shape}")
    logger.info(f"Multi-target test data shape: {multi_test_data.shape}")
    logger.info(f"Selected features: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Selected features: {feature_cols}")
    logger.info(f"Selected targets: {target_cols[:10]}..." if len(target_cols) > 10 else f"Selected targets: {target_cols}")
    
    return {
        'train_data': multi_train_data,
        'train_labels': multi_train_labels,
        'test_data': multi_test_data,
        'feature_cols': feature_cols,
        'target_cols': target_cols
    }

def create_enhanced_features(data: pd.DataFrame, feature_cols: List[str], max_features_for_engineering: int = 10) -> pd.DataFrame:
    """Create enhanced features with controlled feature engineering"""
    logger.info("Creating enhanced features...")
    
    result = data.copy()
    
    # Sort by date for proper time series operations
    result = result.sort_values('date_id').reset_index(drop=True)
    
    # Limit feature engineering to avoid explosion
    eng_features = feature_cols[:max_features_for_engineering]
    
    # Add lag features (1 and 2 periods) - only for subset
    for col in eng_features[:5]:
        result[f'{col}_lag1'] = result[col].shift(1)
        result[f'{col}_lag2'] = result[col].shift(2)
    
    # Add simple rolling means (5 periods) - only for subset
    for col in eng_features[:3]:
        result[f'{col}_roll5'] = result[col].rolling(window=5).mean()
        result[f'{col}_roll10'] = result[col].rolling(window=10).mean()
    
    # Add some basic interactions between top features
    for i, col1 in enumerate(eng_features[:3]):
        for col2 in eng_features[i+1:4]:
            result[f'{col1}_{col2}_ratio'] = result[col1] / (result[col2] + 1e-8)
    
    # Fill NaN values
    result = result.ffill().fillna(0)
    
    logger.info(f"Enhanced data shape: {result.shape}")
    
    return result

def train_multi_target_model(data_dict: Dict, train_size: float = 0.8) -> Tuple[MultiTargetXGBoostRegressor, Dict]:
    """Train a single multi-target model"""
    logger.info("Training multi-target model...")
    
    # Merge train data with labels
    merged_data = data_dict['train_data'].merge(
        data_dict['train_labels'], 
        on='date_id', 
        how='inner'
    )
    
    # Remove rows with any missing targets
    target_cols = data_dict['target_cols']
    merged_data = merged_data.dropna(subset=target_cols)
    
    # Create enhanced features
    enhanced_data = create_enhanced_features(merged_data, data_dict['feature_cols'])
    
    # Prepare features (exclude date_id and targets)
    feature_cols = [col for col in enhanced_data.columns 
                   if col not in ['date_id'] + target_cols]
    
    X = enhanced_data[feature_cols].values
    y = enhanced_data[target_cols].values
    
    # Time-based split
    split_idx = int(len(enhanced_data) * train_size)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Targets: {len(target_cols)}")
    
    # Train model
    model = MultiTargetXGBoostRegressor(max_depth=6, eta=0.1, n_estimators=200)
    model.fit(X_train, y_train, X_val, y_val, target_names=target_cols)
    
    # Evaluate all targets
    val_pred = model.predict(X_val)
    correlations = {}
    
    for i, target_col in enumerate(target_cols):
        corr, _ = spearmanr(y_val[:, i], val_pred[:, i])
        correlations[target_col] = corr
        logger.info(f"✓ {target_col} - Spearman correlation: {corr:.4f}")
    
    return model, correlations

def create_training_visualizations(model: MultiTargetXGBoostRegressor, correlations: Dict, 
                                 predictions: np.ndarray = None, target_cols: List[str] = None) -> None:
    """Create comprehensive training visualizations"""
    logger.info("Creating enhanced training visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Training Progress (2x2 subplot grid)
    ax1 = plt.subplot(3, 3, 1)
    if model.training_history['epochs']:
        epochs = model.training_history['epochs']
        plt.plot(epochs, model.training_history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
        plt.plot(epochs, model.training_history['val_rmse'], 'r-', label='Val RMSE', linewidth=2)
        plt.title('Training Progress (RMSE)', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Correlation Distribution
    ax2 = plt.subplot(3, 3, 2)
    corr_values = list(correlations.values())
    plt.hist(corr_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(corr_values), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(corr_values):.3f}')
    plt.title('Correlation Distribution Across Targets', fontsize=14, fontweight='bold')
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Top/Bottom Correlations
    ax3 = plt.subplot(3, 3, 3)
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1])
    top_n = 5
    
    # Show top and bottom correlations
    bottom_corrs = sorted_corrs[:top_n]
    top_corrs = sorted_corrs[-top_n:]
    
    labels = [item[0].replace('target_', 'T') for item in bottom_corrs + top_corrs]
    values = [item[1] for item in bottom_corrs + top_corrs]
    colors = ['red'] * top_n + ['green'] * top_n
    
    plt.barh(range(len(labels)), values, color=colors, alpha=0.7)
    plt.yticks(range(len(labels)), labels)
    plt.title(f'Top/Bottom {top_n} Correlations', fontsize=14, fontweight='bold')
    plt.xlabel('Spearman Correlation')
    plt.grid(True, alpha=0.3)
    
    # 4. Feature Importance (Top 15)
    ax4 = plt.subplot(3, 3, 4)
    importance_dict = model.get_feature_importance(max_features=15)
    if importance_dict:
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        plt.barh(range(len(features)), importances, color='lightgreen', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.grid(True, alpha=0.3)
    
    # 5. Prediction Distribution (if available)
    ax5 = plt.subplot(3, 3, 5)
    if predictions is not None:
        plt.hist(predictions.flatten(), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Test Prediction Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    # 6. Correlation vs Target Index
    ax6 = plt.subplot(3, 3, 6)
    target_indices = [int(k.replace('target_', '')) for k in correlations.keys()]
    corr_values = [correlations[f'target_{i}'] for i in target_indices]
    
    plt.scatter(target_indices, corr_values, alpha=0.6, c=corr_values, 
               cmap='RdYlBu_r', s=50)
    plt.colorbar(label='Correlation')
    plt.title('Correlation by Target Index', fontsize=14, fontweight='bold')
    plt.xlabel('Target Index')
    plt.ylabel('Spearman Correlation')
    plt.grid(True, alpha=0.3)
    
    # 7. Performance Summary Text
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    
    summary_text = f"""
Multi-Target Training Summary:

• Total Targets: {len(correlations)}
• Features Used: {len(model.get_feature_importance())}
• Training Method: XGBoost Multi-Target

Performance Statistics:
• Mean Correlation: {np.mean(corr_values):.4f}
• Std Correlation: {np.std(corr_values):.4f}
• Best Correlation: {max(corr_values):.4f}
• Worst Correlation: {min(corr_values):.4f}
• Positive Correlations: {sum(1 for c in corr_values if c > 0)}/{len(corr_values)}

Model Configuration:
• Max Depth: 6
• Learning Rate: 0.1
• Estimators: 200
• Early Stopping: 20 rounds
    """
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 8. Learning Curve Comparison
    ax8 = plt.subplot(3, 3, 8)
    if model.training_history['epochs']:
        # Create a smoothed learning curve
        epochs = model.training_history['epochs']
        train_rmse = model.training_history['train_rmse']
        val_rmse = model.training_history['val_rmse']
        
        plt.plot(epochs, train_rmse, 'b-', alpha=0.7, label='Train RMSE')
        plt.plot(epochs, val_rmse, 'r-', alpha=0.7, label='Val RMSE')
        plt.fill_between(epochs, train_rmse, alpha=0.3, color='blue')
        plt.fill_between(epochs, val_rmse, alpha=0.3, color='red')
        
        plt.title('Learning Curve Detail', fontsize=14, fontweight='bold')
        plt.xlabel('Training Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 9. Correlation Heatmap (sample)
    ax9 = plt.subplot(3, 3, 9)
    if len(correlations) > 1:
        # Create a correlation matrix visualization for first few targets
        sample_size = min(10, len(correlations))
        sample_targets = list(correlations.keys())[:sample_size]
        sample_corrs = [correlations[t] for t in sample_targets]
        
        # Create a simple heatmap-style visualization
        corr_matrix = np.array(sample_corrs).reshape(1, -1)
        im = plt.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
        plt.colorbar(im, label='Correlation')
        plt.title(f'Sample Target Correlations (First {sample_size})', fontsize=14, fontweight='bold')
        plt.ylabel('Correlation')
        plt.xticks(range(sample_size), [t.replace('target_', 'T') for t in sample_targets], rotation=45)
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('multi_target_training_results.png', dpi=150, bbox_inches='tight')
    logger.info("Enhanced visualization saved to multi_target_training_results.png")
    plt.show()

def main():
    """Main multi-target training function"""
    parser = argparse.ArgumentParser(description='Multi-Target Mitsui Training Script')
    parser.add_argument('--data-path', type=str, default='mitsui-commodity-prediction-challenge',
                       help='Path to data directory')
    parser.add_argument('--n-features', type=int, default=20,
                       help='Number of base features to use')
    parser.add_argument('--n-targets', type=int, default=10,
                       help='Number of targets to train')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--train-size', type=float, default=0.8,
                       help='Proportion of data to use for training (default: 0.8)')
    parser.add_argument('--time-fraction', type=float, default=1.0,
                       help='Fraction of time series to use (default: 1.0 = all data)')
    
    args = parser.parse_args()
    
    print("🚀 MULTI-TARGET MITSUI TRAINING SCRIPT")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Features: {args.n_features}")
    print(f"Targets: {args.n_targets}")
    print(f"Train size: {args.train_size:.1%}")
    print(f"Time fraction: {args.time_fraction:.1%}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load multi-target data
        data_dict = load_multi_target_data(args.data_path, args.n_features, 
                                         args.n_targets, args.time_fraction)
        
        # Train single multi-target model
        print(f"\n📚 Training single multi-target model...")
        model, correlations = train_multi_target_model(data_dict, args.train_size)
        
        # Save model
        print(f"\n💾 Saving trained model...")
        os.makedirs('models', exist_ok=True)
        
        model_path = f'models/multi_target_xgb_model.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Multi-target model saved: {model_path}")
        
        # Generate predictions on test data
        print(f"\n🔮 Generating test predictions...")
        enhanced_test = create_enhanced_features(data_dict['test_data'], data_dict['feature_cols'])
        feature_cols = [col for col in enhanced_test.columns if col != 'date_id']
        X_test = enhanced_test[feature_cols].values
        
        test_predictions = model.predict(X_test)
        logger.info(f"Test predictions shape: {test_predictions.shape}")
        
        # Create enhanced visualizations
        if not args.no_plots:
            print(f"\n📊 Creating enhanced visualizations...")
            create_training_visualizations(model, correlations, test_predictions, 
                                         data_dict['target_cols'])
        
        # Summary
        avg_corr = np.mean(list(correlations.values())) if correlations else 0.0
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 MULTI-TARGET TRAINING COMPLETED")
        print("=" * 60)
        print(f"✅ Targets trained: {len(correlations)}")
        print(f"✅ Average correlation: {avg_corr:.4f}")
        print(f"✅ Best correlation: {max(correlations.values()):.4f}")
        print(f"✅ Total time: {total_time:.2f}s")
        print(f"✅ Test predictions: {len(test_predictions)} samples")
        print(f"✅ Model complexity: Single model vs {len(correlations)} separate models")
        
        print("\nTop 10 Target Results:")
        sorted_results = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        for target, corr in sorted_results:
            print(f"  {target}: {corr:.4f}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Multi-target training failed: {e}")
        raise

if __name__ == "__main__":
    main()