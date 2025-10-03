# Mitsui & Co. Commodity Prediction Challenge

A streamlined machine learning solution for commodity price prediction using LSTM and XGBoost models. This project implements competition-grade metrics, hardware acceleration, and integrated analysis in a clean, simplified architecture.

## Project Structure

```
Mitsu&Co/
├── src/                         # Core modules
│   ├── train_LSTM.py           # Deep Learning LSTM approach
│   ├── train_XGBoost.py        # Gradient Boosting approach
│   └── train.py                # Training utilities and shared code
├── utils/                       # Utility functions and helpers
│   ├── competition_metrics.py   # Competition scoring utilities
│   └── feature_engineering.py   # Feature creation and transformation
├── tests/                       # Test suite
├── docs/                        # Documentation
├── models/                      # Saved model artifacts
├── figures/                     # Generated visualizations
└── evaluation_results/          # Competition evaluation results
```

## Quick Start

### LSTM Training

```bash
# Quick test with small subset
python src/train_LSTM.py --n-targets 10 --epochs 5 --num-layers 2 --lstm-units 32

# Production training (all targets)
python src/train_LSTM.py --n-targets 424 --epochs 100 --num-layers 4 --deep-fc --disable-early-stopping --lstm-units 192
```

### XGBoost Training

```bash
# Quick prototyping with competition analysis
python src/train_XGBoost.py --n-features 20 --n-targets 10 --time-fraction 0.2

# Full scale training (all features and targets)
python src/train_XGBoost.py
```

## Key Features

### LSTM Model (`train_LSTM.py`)
- Multi-target deep learning approach
- Configurable architecture (layers, units, sequence length)
- Advanced feature engineering
- Comprehensive visualization system
- Early stopping and model checkpointing
- Test dataset evaluation

### XGBoost Model (`train_XGBoost.py`)
- Gradient boosting approach
- Feature importance analysis
- Hardware acceleration (GPU/CPU)
- Automated feature engineering
- Model persistence and evaluation

### Competition Integration
- Spearman correlations with Sharpe-like scoring
- Performance breakdown by correlation ranges
- Model comparison and ranking
- Proper train/validation splits

## Data Overview

- Training Data: 1,917 time periods × 370+ features × 424 targets
- Test Data: 90 time periods for final predictions
- Competition Metric: mean(daily_spearman_correlations) / std(daily_spearman_correlations)
- Type: Time series commodity price data

## Configuration

### LSTM Parameters
```bash
--n-targets              # Number of targets to train (default: 424)
--epochs                 # Training epochs (default: 100)
--num-layers            # LSTM layers (default: 2)
--lstm-units           # Units per layer (default: 64)
--sequence-length      # Time steps to look back (default: 30)
--deep-fc              # Use deep fully connected layers
--disable-early-stopping # Train for full epochs
```

### XGBoost Parameters
```bash
--n-features           # Number of base features (default: 370)
--n-targets           # Number of targets (default: 424)
--time-fraction       # Fraction of time series to use (default: 1.0)
--train-size         # Training/validation split (default: 0.8)
```

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run a test training:
   ```bash
   python src/train_LSTM.py --n-targets 10 --epochs 5
   ```
4. Check generated visualizations in `figures/` directory

## Development

- Use `src/train_LSTM.py` for deep learning approach
- Use `src/train_XGBoost.py` for gradient boosting
- Generated files (models, figures, results) are gitignored
- Tests available in `tests/` directory
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