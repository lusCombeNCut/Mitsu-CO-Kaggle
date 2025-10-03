#!/usr/bin/env python3
"""
Main training script for Mitsui Commodity Prediction Challenge.
Trains XGBoost models on commodity data with integrated competition scoring.
"""

import sys
import os

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from train import main
    main()