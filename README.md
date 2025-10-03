# Mitsui & Co. Commodity Prediction Challenge

A streamlined machine learning solution for commodity price prediction using XGBoost models. This project implements competition-grade metrics, hardware acceleration, and integrated analysis in a clean, simplified architecture.

## 🏗️ Project Structure (Simplified)

```
Mitsu&Co/
├── train_model.py                # 🎯 MAIN TRAINING SCRIPT - Run this to train models
├── src/                          # Core modules
│   ├── train.py                  # TRUE multi-target XGBoost (single model for all targets)
│   └── individual_models_legacy.py # Legacy approach (424 separate models - NOT recommended)
├── utils/                        # Utility functions and helpers
│   ├── feature_engineering.py    # Feature creation and transformation
│   ├── kaggle_score_calculator.py # Competition scoring utilities
│   ├── time_series_cv.py         # Cross-validation utilities
│   └── xgboost_ranker.py         # XGBoost ranking utilities
├── tests/                        # Test suite and examples
├── docs/                         # Documentation
├── models/                       # Saved model artifacts
├── outputs/                      # Generated plots, logs, results
└── mitsui-commodity-prediction-challenge/  # Raw data
```

## 🚀 Quick Start (Simplified Commands)

### Main Training Script (All-in-One Solution)

#### 🚀 Production Training (Recommended)
```bash
# Quick prototyping with competition analysis
python train_model.py --n-features 20 --n-targets 10 --time-fraction 0.2

# Medium scale training with full analysis  
python train_model.py --n-features 100 --n-targets 200 --time-fraction 0.5

# Full scale training (all features and targets) - DEFAULT
python train_model.py

# Quick test with small subset
python train_model.py --time-fraction 0.1 --n-features 10 --n-targets 5
```

#### 📊 Multi-Target Training (Alternative for Experimentation)
```bash
# Fast prototyping with unified model approach
python run_multi_target_trainer.py --n-features 15 --n-targets 10 --time-fraction 0.3

# Experimental multi-target approach
python run_multi_target_trainer.py --n-features 30 --n-targets 100 --time-fraction 0.5
```

## 📊 Key Features

### 🎯 Main Training Engine (`train_model.py`)
- **All-in-One Solution**: Training, scoring, and analysis in a single command
- **Hardware Accelerated**: Auto-detects GPU/CPU configuration for optimal performance
- **Integrated Analysis**: Built-in competition metrics and visualization
- **Production Ready**: Default parameters optimized for full-scale training (370 features, 424 targets)
- **Configurable**: Flexible parameters for development and testing

### 🔧 Core Features
- **XGBoost Models**: Individual optimized models per target
- **Competition Scoring**: Integrated Kaggle-style Spearman correlation analysis  
- **Hardware Optimization**: Auto GPU detection, CPU multi-threading
- **Feature Engineering**: Automated lag features, rolling means, ratios
- **Visualization**: Training results and feature importance plots
- **Model Persistence**: Automatic model saving and loading

### 🗂️ Legacy Code
- **Legacy Individual Models**: Available in `individual_models_legacy.py`
- **Old Architecture**: 424 separate XGBoost models (one per target)
- **Not Recommended**: Much slower and less efficient than true multi-target approach

### 📊 Competition Integration
- **Real Kaggle Metrics**: Spearman correlations with Sharpe-like scoring
- **Performance Breakdown**: Detailed analysis by correlation ranges
- **Model Comparison**: Top performers identification and ranking
- **Validation Focus**: Proper train/validation splits for evaluation

## 💾 Data Overview

- **Training Data**: 1,917 time periods × 370+ features × 424 targets
- **Test Data**: Available for final predictions  
- **Competition Metric**: `mean(daily_spearman_correlations) / std(daily_spearman_correlations)`
- **Time Series**: Commodity price data with engineered features

## 🔧 Configuration Options

### Mini Trainer Parameters
```bash
--n-features      # Number of base features to use (default: 20)
--n-targets       # Number of targets to train (default: 3) 
--time-fraction   # Fraction of time series to use (default: 1.0)
--train-size      # Training/validation split (default: 0.8)
--no-plots        # Skip generating visualizations
```

### Example Configurations
```bash
# Quick test (5 features, 2 targets, 10% data) - Individual models
python run_mini_trainer.py --n-features 5 --n-targets 2 --time-fraction 0.1

# Multi-target approach (single model, better visualization)
python run_multi_target_trainer.py --n-features 20 --n-targets 10 --time-fraction 0.1

# Full scale training (all targets, reduced time series) - Individual models
python run_mini_trainer.py --n-features 10 --n-targets 424 --time-fraction 0.1 --train-size 0.95

# Maximum data usage (20 features, all targets, full time series)
python run_mini_trainer.py --n-features 20 --n-targets 424 --train-size 0.9
```

## 📈 Performance

- **Training Speed**: ~54s for 424 targets with 10% data
- **Memory Efficient**: Processes large datasets incrementally  
- **Correlation Range**: Achieving correlations from -0.76 to +0.68
- **Competition Metric**: Real Kaggle scoring implementation

## 🛠️ Requirements

See `requirements.txt` for full dependencies. Key packages:
- `xgboost >= 2.0`
- `pandas >= 2.0`
- `numpy >= 1.24`  
- `scikit-learn >= 1.3`
- `matplotlib >= 3.7`
- `scipy >= 1.10`

## 📝 Development

The project follows a clean architecture pattern:
- **src/**: Core business logic and models
- **utils/**: Reusable utility functions
- **tests/**: Test suite and examples
- **docs/**: Documentation and guides

All modules support both standalone and packaged execution through flexible import handling.