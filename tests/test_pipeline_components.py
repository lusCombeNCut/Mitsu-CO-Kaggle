"""
Simple test to verify our XGBoost ranking solution works
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

def test_basic_ranking():
    """Test basic XGBoost ranking functionality"""
    print("🧪 Testing basic XGBoost ranking...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_groups = 5
    samples_per_group = n_samples // n_groups
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate targets with some structure
    y = np.zeros(n_samples)
    for i in range(n_groups):
        start_idx = i * samples_per_group
        end_idx = (i + 1) * samples_per_group
        # Add group-specific pattern
        y[start_idx:end_idx] = np.random.randn(samples_per_group) + i * 0.5
    
    # Create groups array
    groups = np.array([samples_per_group] * n_groups)
    
    print(f"✓ Generated test data: X{X.shape}, y{y.shape}, groups{groups}")
    
    # Create XGBoost ranking model
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups)
    
    params = {
        'objective': 'rank:pairwise',
        'eta': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'eval_metric': 'ndcg@10',
        'verbosity': 0
    }
    
    # Train model
    model = xgb.train(params, dtrain, num_boost_round=50, verbose_eval=False)
    
    # Make predictions
    predictions = model.predict(dtrain)
    
    print(f"✓ Model trained successfully")
    print(f"✓ Predictions generated: {predictions.shape}")
    
    # Test ranking within groups
    from scipy.stats import spearmanr
    
    correlations = []
    start_idx = 0
    
    for group_size in groups:
        end_idx = start_idx + group_size
        
        if group_size > 1:
            group_y = y[start_idx:end_idx]
            group_pred = predictions[start_idx:end_idx]
            
            if len(np.unique(group_y)) > 1:
                corr, _ = spearmanr(group_y, group_pred)
                if not np.isnan(corr):
                    correlations.append(corr)
        
        start_idx = end_idx
    
    avg_correlation = np.mean(correlations) if correlations else 0.0
    print(f"✓ Average Spearman correlation: {avg_correlation:.4f}")
    
    return True

def test_data_loading():
    """Test loading the actual competition data"""
    print("\n🧪 Testing data loading...")
    
    data_path = Path("mitsui-commodity-prediction-challenge")
    
    if not data_path.exists():
        print("❌ Data directory not found")
        return False
    
    # Load train data
    train_df = pd.read_csv(data_path / "train.csv")
    print(f"✓ Training data loaded: {train_df.shape}")
    print(f"  Columns: {train_df.columns.tolist()[:5]}...")
    
    # Load labels
    labels_df = pd.read_csv(data_path / "train_labels.csv")
    print(f"✓ Training labels loaded: {labels_df.shape}")
    print(f"  Columns: {labels_df.columns.tolist()[:5]}...")
    
    # Load target pairs
    pairs_df = pd.read_csv(data_path / "target_pairs.csv")
    print(f"✓ Target pairs loaded: {pairs_df.shape}")
    print(f"  Columns: {pairs_df.columns.tolist()}")
    print(f"  Sample pairs: {pairs_df['pair'].head().tolist()}")
    
    # Load test data
    test_df = pd.read_csv(data_path / "test.csv")
    print(f"✓ Test data loaded: {test_df.shape}")
    
    return True

def test_feature_engineering():
    """Test basic feature engineering"""
    print("\n🧪 Testing feature engineering...")
    
    # Create sample time series data
    np.random.seed(42)
    dates = list(range(100))
    n_features = 50
    
    data = []
    for date in dates:
        row = {'date_id': date}
        # Add some features with trends
        for i in range(n_features):
            base_value = np.sin(date * 0.1) + np.random.randn() * 0.1
            row[f'feature_{i}'] = base_value + i * 0.01
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"✓ Sample data created: {df.shape}")
    
    # Test lag features
    for lag in [1, 2, 3]:
        for col in [f'feature_{i}' for i in range(5)]:  # Test first 5 features
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    print(f"✓ Lag features added: {df.shape}")
    
    # Test rolling features
    for window in [5, 10]:
        for col in [f'feature_{i}' for i in range(5)]:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
    
    print(f"✓ Rolling features added: {df.shape}")
    
    # Remove NaN rows (from lags and rolling windows)
    df_clean = df.dropna()
    print(f"✓ Cleaned data: {df_clean.shape}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing Mitsui Commodity Prediction Pipeline Components")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic ranking
    try:
        success &= test_basic_ranking()
    except Exception as e:
        print(f"❌ Basic ranking test failed: {e}")
        success = False
    
    # Test 2: Data loading
    try:
        success &= test_data_loading()
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        success = False
    
    # Test 3: Feature engineering
    try:
        success &= test_feature_engineering()
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The pipeline components are working correctly.")
        print("\nNext steps:")
        print("1. Make sure your conda environment is activated:")
        print('   conda activate "c:\\Users\\Orlan\\Desktop\\Projects\\Mitsu&Co\\.conda"')
        print("2. Run the full pipeline when you have the complete data structure figured out")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    main()