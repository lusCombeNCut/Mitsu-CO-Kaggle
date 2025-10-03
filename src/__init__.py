"""
Mitsui & Co. Commodity Prediction Challenge - Source Code

This package contains the main modules for the Mitsui commodity prediction challenge:
- Data loading and preprocessing
- Model training and evaluation
- Prediction pipelines
- Competition metric calculation
"""

# Main modules
from .mitsui_data_loader import MitsuiDataLoader
from .mini_trainer import MiniXGBoostRegressor, load_mini_data, train_mini_model
from .real_score_analyzer import load_and_evaluate_actual_model, create_real_score_visualization

__version__ = "1.0.0"
__author__ = "Mitsui Challenge Team"