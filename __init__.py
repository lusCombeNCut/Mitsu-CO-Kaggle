"""
Mitsui & Co. Commodity Prediction Challenge

A comprehensive solution for commodity price prediction using machine learning.
Features XGBoost models, competition metrics, and efficient data processing.
"""

__version__ = "1.0.0"
__author__ = "Mitsui Challenge Team"

# Import main components for easy access
try:
    from src.mitsui_data_loader import MitsuiDataLoader
    from src.mini_trainer import MiniXGBoostRegressor, load_mini_data, train_mini_model
    from src.real_score_analyzer import load_and_evaluate_actual_model
    from utils.kaggle_score_calculator import calculate_kaggle_score
except ImportError:
    # Fallback for when modules haven't been updated yet
    pass