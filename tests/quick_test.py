"""
Quick Test Script - Fast verification that all components work
Tests core functionality without full training pipeline
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import time
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test 1: Data loading functionality"""
    print("🧪 Test 1: Data Loading")
    print("-" * 30)
    
    try:
        data_path = Path("mitsui-commodity-prediction-challenge")
        
        if not data_path.exists():
            print(f"❌ Data directory not found: {data_path}")
            return False
        
        # Test loading each file
        files_to_test = ["train.csv", "train_labels.csv", "test.csv", "target_pairs.csv"]
        
        for filename in files_to_test:
            file_path = data_path / filename
            if not file_path.exists():
                print(f"❌ File not found: {filename}")
                return False
            
            df = pd.read_csv(file_path)
            print(f"✅ {filename}: {df.shape}")
        
        # Test basic data properties
        train_data = pd.read_csv(data_path / "train.csv")
        train_labels = pd.read_csv(data_path / "train_labels.csv")
        
        print(f"✅ Training data columns: {len(train_data.columns)}")
        print(f"✅ Training labels columns: {len(train_labels.columns)}")
        print(f"✅ Date range: {train_data['date_id'].min()} to {train_data['date_id'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_xgboost_basic():
    """Test 2: XGBoost basic functionality"""
    print("\n🧪 Test 2: XGBoost Basic Functionality")
    print("-" * 40)
    
    try:
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.1  # Simple linear relationship
        
        # Split data
        split_idx = 80
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train XGBoost regressor
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,
            'eta': 0.1,
            'eval_metric': 'rmse',
            'verbosity': 0
        }
        
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=50,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            verbose_eval=False
        )
        
        # Make predictions
        predictions = model.predict(dtest)
        
        # Calculate correlation
        correlation, _ = spearmanr(y_test, predictions)
        
        print(f"✅ Model trained successfully")
        print(f"✅ Test samples: {len(y_test)}")
        print(f"✅ Spearman correlation: {correlation:.4f}")
        print(f"✅ Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        return correlation > 0.8  # Should be high for synthetic data
        
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        return False

def test_feature_engineering():
    """Test 3: Basic feature engineering"""
    print("\n🧪 Test 3: Feature Engineering")
    print("-" * 35)
    
    try:
        # Create time series data
        n_dates = 50
        n_features = 5
        
        data = []
        for date_id in range(n_dates):
            row = {'date_id': date_id}
            for i in range(n_features):
                # Add trend + noise
                value = np.sin(date_id * 0.1) + np.random.randn() * 0.1
                row[f'feature_{i}'] = value
            data.append(row)
        
        df = pd.DataFrame(data)
        print(f"✅ Created synthetic data: {df.shape}")
        
        # Add lag features
        for lag in [1, 2]:
            for col in [f'feature_{i}' for i in range(2)]:  # Only first 2 features
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        print(f"✅ Added lag features: {df.shape}")
        
        # Add rolling features
        for window in [3, 5]:
            for col in [f'feature_{i}' for i in range(2)]:
                df[f'{col}_roll{window}'] = df[col].rolling(window).mean()
        
        print(f"✅ Added rolling features: {df.shape}")
        
        # Handle NaN values
        df_clean = df.fillna(method='ffill').fillna(0)
        n_nan = df_clean.isnull().sum().sum()
        
        print(f"✅ Cleaned data: {df_clean.shape}, NaN values: {n_nan}")
        
        return n_nan == 0
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_mini_pipeline():
    """Test 4: Mini end-to-end pipeline"""
    print("\n🧪 Test 4: Mini Pipeline")
    print("-" * 30)
    
    try:
        data_path = Path("mitsui-commodity-prediction-challenge")
        
        if not data_path.exists():
            print("⚠️  Skipping pipeline test - no data directory")
            return True
        
        # Load small subset of data
        train_data = pd.read_csv(data_path / "train.csv")
        train_labels = pd.read_csv(data_path / "train_labels.csv")
        
        # Take first 100 rows and 10 features
        feature_cols = [col for col in train_data.columns if col != 'date_id'][:10]
        mini_train = train_data[['date_id'] + feature_cols].head(100)
        
        # Take first target
        target_col = [col for col in train_labels.columns if col.startswith('target_')][0]
        mini_labels = train_labels[['date_id', target_col]].head(100)
        
        # Merge
        merged = mini_train.merge(mini_labels, on='date_id', how='inner')
        merged = merged.dropna()
        
        if len(merged) < 20:
            print("⚠️  Not enough clean data for mini pipeline test")
            return True
        
        # Prepare features
        X = merged[feature_cols].fillna(0).values
        y = merged[target_col].values
        
        # Simple train/test split
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train mini model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 2,
            'eta': 0.2,
            'verbosity': 0
        }
        
        model = xgb.train(params, dtrain, num_boost_round=10)
        
        # Predict
        dtest = xgb.DMatrix(X_test)
        predictions = model.predict(dtest)
        
        print(f"✅ Mini pipeline completed")
        print(f"✅ Training samples: {len(X_train)}")
        print(f"✅ Test samples: {len(X_test)}")
        print(f"✅ Features: {X.shape[1]}")
        print(f"✅ Target: {target_col}")
        print(f"✅ Predictions generated: {len(predictions)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mini pipeline test failed: {e}")
        return False

def test_visualization():
    """Test 5: Basic plotting functionality"""
    print("\n🧪 Test 5: Visualization")
    print("-" * 30)
    
    try:
        # Create simple test plot
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(x, y1, label='sin(x)')
        ax.plot(x, y2, label='cos(x)')
        ax.legend()
        ax.set_title('Test Plot')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Test plot created and saved")
        
        # Test if plot file exists
        if Path('test_plot.png').exists():
            print(f"✅ Plot file saved successfully")
            return True
        else:
            print(f"⚠️  Plot file not found")
            return False
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 QUICK SYSTEM VERIFICATION")
    print("=" * 50)
    
    start_time = time.time()
    
    tests = [
        ("Data Loading", test_data_loading),
        ("XGBoost Basic", test_xgboost_basic),
        ("Feature Engineering", test_feature_engineering),
        ("Mini Pipeline", test_mini_pipeline),
        ("Visualization", test_visualization)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your system is ready for training.")
    elif passed > total // 2:
        print(f"\n⚠️  Most tests passed. Check failed tests above.")
    else:
        print(f"\n❌ Multiple tests failed. Check your environment setup.")
    
    print("=" * 50)
    
    # Next steps
    if passed >= total - 1:  # Allow 1 failure
        print("\n🚀 RECOMMENDED NEXT STEPS:")
        print("1. Run mini trainer: python mini_trainer.py")
        print("2. If mini trainer works, try: python mitsui_regression_trainer.py --n-targets 3")
        print("3. For full training: python mitsui_regression_trainer.py")

if __name__ == "__main__":
    main()