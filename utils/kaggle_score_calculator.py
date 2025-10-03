"""
Kaggle Competition Score Visualization
Calculates and displays the actual competition metric clearly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_spearman_per_date(y_true: np.ndarray, y_pred: np.ndarray, 
                                dates: np.ndarray) -> Tuple[List[float], float, float]:
    """
    Calculate Spearman correlation per date (the official competition metric)
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        dates: Date IDs for each sample
        
    Returns:
        List of per-date correlations, mean correlation, std correlation
    """
    unique_dates = np.unique(dates)
    correlations = []
    
    logger.info(f"Calculating Spearman correlation for {len(unique_dates)} dates...")
    
    for date in unique_dates:
        date_mask = dates == date
        date_true = y_true[date_mask]
        date_pred = y_pred[date_mask]
        
        # Need at least 2 samples to compute correlation
        if len(date_true) >= 2 and len(np.unique(date_true)) > 1:
            corr, _ = spearmanr(date_true, date_pred)
            if not np.isnan(corr):
                correlations.append(corr)
            else:
                correlations.append(0.0)
        else:
            correlations.append(0.0)
    
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    
    return correlations, mean_corr, std_corr

def calculate_competition_score(correlations: List[float]) -> float:
    """
    Calculate the competition score (Sharpe-like ratio)
    Score = mean(Spearman correlations) / std(Spearman correlations)
    """
    if len(correlations) == 0:
        return 0.0
    
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    
    if std_corr == 0:
        return 0.0 if mean_corr == 0 else np.inf if mean_corr > 0 else -np.inf
    
    return mean_corr / std_corr

def load_mini_training_results() -> Dict:
    """Load results from mini training if available"""
    
    # Try to find recent training results
    possible_files = [
        "mini_training_results.json",
        "evaluation_results/detailed_analysis_*.json"
    ]
    
    # For now, let's create mock results to demonstrate the visualization
    # In practice, this would load from actual training results
    
    logger.info("Creating example results for demonstration...")
    
    # Simulate results from 3 targets over 50 validation dates
    np.random.seed(42)
    n_dates = 50
    n_targets = 3
    
    results = {}
    
    for target_idx in range(n_targets):
        target_name = f"target_{target_idx}"
        
        # Generate mock predictions and true values
        dates = np.arange(n_dates)
        
        # Simulate some correlation structure
        base_signal = np.sin(np.arange(n_dates) * 0.3) + np.random.randn(n_dates) * 0.1
        true_values = base_signal + np.random.randn(n_dates) * 0.2
        
        # Predictions with varying quality
        correlation_strength = [0.3, 0.6, 0.1][target_idx]  # Different quality per target
        predictions = (base_signal * correlation_strength + 
                      np.random.randn(n_dates) * (1 - correlation_strength))
        
        results[target_name] = {
            'dates': dates,
            'y_true': true_values, 
            'y_pred': predictions,
            'target_name': target_name
        }
    
    return results

def create_competition_score_visualization(results: Dict) -> None:
    """Create comprehensive competition score visualization"""
    
    logger.info("Creating competition score visualization...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Calculate scores for each target
    target_scores = {}
    all_correlations = []
    
    for target_name, data in results.items():
        correlations, mean_corr, std_corr = calculate_spearman_per_date(
            data['y_true'], data['y_pred'], data['dates']
        )
        
        competition_score = calculate_competition_score(correlations)
        
        target_scores[target_name] = {
            'correlations': correlations,
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'competition_score': competition_score,
            'n_dates': len(correlations)
        }
        
        all_correlations.extend(correlations)
    
    # Overall competition score
    overall_score = calculate_competition_score(all_correlations)
    overall_mean = np.mean(all_correlations)
    overall_std = np.std(all_correlations)
    
    # 1. Main Score Display (Large)
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    
    # Create a gauge-like display
    score_color = 'green' if overall_score > 0.5 else 'orange' if overall_score > 0 else 'red'
    
    # Main score circle
    circle = plt.Circle((0.5, 0.5), 0.4, color=score_color, alpha=0.3)
    ax1.add_patch(circle)
    
    # Score text
    ax1.text(0.5, 0.6, f'{overall_score:.3f}', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=36, fontweight='bold', color=score_color)
    
    ax1.text(0.5, 0.35, 'Competition\nScore', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=14, fontweight='bold')
    
    ax1.text(0.5, 0.15, f'μ={overall_mean:.3f}, σ={overall_std:.3f}', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=11, color='gray')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('KAGGLE COMPETITION METRIC', fontsize=16, fontweight='bold', pad=20)
    
    # 2. Score Breakdown by Target
    ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2)
    
    target_names = list(target_scores.keys())
    scores = [target_scores[t]['competition_score'] for t in target_names]
    colors = ['green' if s > 0.5 else 'orange' if s > 0 else 'red' for s in scores]
    
    bars = ax2.bar(target_names, scores, color=colors, alpha=0.7)
    ax2.set_title('Competition Score by Target')
    ax2.set_ylabel('Score (μ/σ)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Correlation Distribution
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
    
    ax3.hist(all_correlations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {overall_mean:.3f}')
    ax3.axvline(overall_mean + overall_std, color='orange', linestyle=':', label=f'+1σ: {overall_mean + overall_std:.3f}')
    ax3.axvline(overall_mean - overall_std, color='orange', linestyle=':', label=f'-1σ: {overall_mean - overall_std:.3f}')
    
    ax3.set_title('Distribution of Daily Spearman Correlations')
    ax3.set_xlabel('Spearman Correlation')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Time Series of Correlations
    ax4 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
    
    for target_name, scores_data in target_scores.items():
        correlations = scores_data['correlations']
        ax4.plot(correlations, label=f'{target_name} (Score: {scores_data["competition_score"]:.3f})', 
                alpha=0.8, linewidth=2)
    
    ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(overall_mean, color='red', linestyle='--', alpha=0.5, label=f'Overall Mean: {overall_mean:.3f}')
    
    ax4.set_title('Daily Spearman Correlations Over Time')
    ax4.set_xlabel('Validation Date')
    ax4.set_ylabel('Spearman Correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary Statistics Table
    ax5 = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    ax5.axis('off')
    
    # Create detailed summary table
    summary_data = []
    summary_data.append(['OVERALL', f'{overall_score:.4f}', f'{overall_mean:.4f}', f'{overall_std:.4f}', f'{len(all_correlations)}'])
    
    for target_name, scores_data in target_scores.items():
        summary_data.append([
            target_name,
            f'{scores_data["competition_score"]:.4f}',
            f'{scores_data["mean_correlation"]:.4f}',
            f'{scores_data["std_correlation"]:.4f}',
            f'{scores_data["n_dates"]}'
        ])
    
    table = ax5.table(
        cellText=summary_data,
        colLabels=['Target', 'Competition Score (μ/σ)', 'Mean Correlation (μ)', 'Std Correlation (σ)', 'N Dates'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.25, 0.2, 0.2, 0.1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Color code the overall row
    for i in range(len(summary_data[0])):
        table[(1, i)].set_facecolor('#E8F4FD')
        table[(1, i)].set_text_props(weight='bold')
    
    ax5.set_title('Detailed Competition Metrics', fontsize=14, fontweight='bold', y=0.9)
    
    # Add interpretation guide
    interpretation_text = """
    INTERPRETATION GUIDE:
    • Competition Score = Mean(Daily Spearman Correlations) / Std(Daily Spearman Correlations)
    • Higher scores are better (consistent positive correlations)
    • Score > 1.0: Excellent (consistent positive correlation)
    • Score 0.5-1.0: Good (moderate positive correlation) 
    • Score 0-0.5: Fair (weak positive correlation)
    • Score < 0: Poor (negative or inconsistent correlation)
    """
    
    fig.text(0.02, 0.02, interpretation_text, fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    plt.savefig('kaggle_competition_score.png', dpi=300, bbox_inches='tight')
    logger.info("Competition score visualization saved to 'kaggle_competition_score.png'")
    
    plt.show()
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏆 KAGGLE COMPETITION SCORE ANALYSIS")
    print("=" * 80)
    print(f"📊 Overall Competition Score: {overall_score:.4f}")
    print(f"📈 Mean Daily Correlation: {overall_mean:.4f}")
    print(f"📉 Std Daily Correlation: {overall_std:.4f}")
    print(f"📅 Total Validation Days: {len(all_correlations)}")
    print()
    
    # Interpretation
    if overall_score > 1.0:
        interpretation = "🎉 EXCELLENT - Very consistent positive correlations!"
    elif overall_score > 0.5:
        interpretation = "👍 GOOD - Moderate positive correlations"
    elif overall_score > 0:
        interpretation = "⚠️ FAIR - Weak positive correlations"
    else:
        interpretation = "❌ POOR - Negative or inconsistent correlations"
    
    print(f"🎯 Model Performance: {interpretation}")
    print()
    
    print("📋 Individual Target Performance:")
    for target_name, scores_data in target_scores.items():
        score = scores_data['competition_score']
        status = "🟢" if score > 0.5 else "🟡" if score > 0 else "🔴"
        print(f"  {status} {target_name}: {score:.4f}")
    
    print("=" * 80)

def main():
    """Main function to create competition score visualization"""
    
    print("📊 KAGGLE COMPETITION SCORE CALCULATOR")
    print("=" * 60)
    print("This script calculates and visualizes the official competition metric:")
    print("Score = Mean(Daily Spearman Correlations) / Std(Daily Spearman Correlations)")
    print("=" * 60)
    
    # Load results (in practice, this would load from your actual training results)
    results = load_mini_training_results()
    
    # Create visualization
    create_competition_score_visualization(results)
    
    print("\n✅ Competition score analysis complete!")
    print("📁 Check 'kaggle_competition_score.png' for detailed visualization")

if __name__ == "__main__":
    main()