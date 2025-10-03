"""
Utility functions and classes for the Mitsui commodity prediction challenge.

This package contains utility functions for:
- Competition evaluation metrics
- Kaggle score calculation  
- Model performance analysis
"""

# Import main utility functions
from .kaggle_score_calculator import calculate_competition_score
from .competition_metrics import (
    calculate_competition_metrics, 
    print_competition_analysis, 
    get_competition_score, 
    save_competition_results
)

__version__ = "1.0.0"