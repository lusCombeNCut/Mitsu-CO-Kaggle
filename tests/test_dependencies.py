"""
Simple test script to check if our modules work
"""

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported successfully")
    
    import pandas as pd
    print(f"✓ Pandas {pd.__version__} imported successfully")
    
    # Test basic functionality
    data = pd.DataFrame({
        'date_id': [1, 1, 2, 2],
        'symbol_id': [100, 101, 100, 101],
        'target': [0.1, -0.2, 0.3, -0.1]
    })
    print(f"✓ Basic pandas operations work")
    
    import xgboost as xgb
    print(f"✓ XGBoost {xgb.__version__} imported successfully")
    
    # Test XGBoost ranking
    X = np.random.randn(4, 3)
    y = np.array([0.1, -0.2, 0.3, -0.1])
    groups = np.array([2, 2])  # 2 samples per group
    
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups)
    
    params = {
        'objective': 'rank:pairwise',
        'eta': 0.1,
        'max_depth': 3
    }
    
    model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
    preds = model.predict(dtrain)
    print(f"✓ XGBoost ranking test successful")
    
    print("\n🎉 All core dependencies are working!")
    print("You can now run the full pipeline.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTry running:")
    print("conda install numpy=1.24.4 pandas=2.0.3 -y")
    
except Exception as e:
    print(f"❌ Error: {e}")