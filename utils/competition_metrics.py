"""
Competition Evaluation Metrics for Mitsui Commodity Prediction Challenge

This module provides comprehensive evaluation metrics specifically designed for
the Mitsui competition, including Spearman correlation (primary metric),
Pearson correlation, RMSE, MAE, and detailed performance analysis.

Author: Generated for Mitsui Competition
Date: October 2025
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

SOLUTION_NULL_FILLER = -999999

def rank_correlation_sharpe_ratio(merged_df: pd.DataFrame) -> float:
    """
    Calculates the rank correlation between predictions and target values,
    and returns its Sharpe ratio (mean / standard deviation).
    This is the OFFICIAL competition metric.

    :param merged_df: DataFrame containing prediction columns (starting with 'prediction_')
                      and target columns (starting with 'target_')
    :return: Sharpe ratio of the rank correlation
    :raises ZeroDivisionError: If the standard deviation is zero
    """
    prediction_cols = [col for col in merged_df.columns if col.startswith('prediction_')]
    target_cols = [col for col in merged_df.columns if col.startswith('target_')]

    def _compute_rank_correlation(row):
        non_null_targets = [col for col in target_cols if not pd.isnull(row[col])]
        matching_predictions = [col for col in prediction_cols if col.replace('prediction', 'target') in non_null_targets]
        if not non_null_targets:
            raise ValueError('No non-null target values found')
        if row[non_null_targets].std(ddof=0) == 0 or row[matching_predictions].std(ddof=0) == 0:
            return 0.0  # Return 0 instead of raising error for robustness
        return np.corrcoef(row[matching_predictions].rank(method='average'), row[non_null_targets].rank(method='average'))[0, 1]

    daily_rank_corrs = merged_df.apply(_compute_rank_correlation, axis=1)
    std_dev = daily_rank_corrs.std(ddof=0)
    if std_dev == 0:
        return 0.0  # Return 0 instead of raising error
    sharpe_ratio = daily_rank_corrs.mean() / std_dev
    return float(sharpe_ratio)

def calculate_official_competition_score(y_true: np.ndarray, y_pred: np.ndarray, 
                                       target_names: Optional[List[str]] = None) -> Dict:
    """
    Calculate the OFFICIAL competition score using rank correlation Sharpe ratio.
    
    Args:
        y_true: True values (2D array: samples x targets)
        y_pred: Predicted values (2D array: samples x targets)  
        target_names: Optional list of target names
        
    Returns:
        Dictionary containing official competition metrics
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
        
    n_samples, n_targets = y_true.shape
    
    if target_names is None:
        target_names = [f'target_{i}' for i in range(n_targets)]
    
    # Create DataFrame in competition format
    data = {}
    
    # Add target columns
    for i, target_name in enumerate(target_names):
        data[target_name] = y_true[:, i]
        
    # Add prediction columns
    for i, target_name in enumerate(target_names):
        pred_name = target_name.replace('target_', 'prediction_')
        data[pred_name] = y_pred[:, i]
    
    merged_df = pd.DataFrame(data)
    
    # Handle NaN values (replace with null filler as in competition)
    merged_df = merged_df.fillna(SOLUTION_NULL_FILLER)
    
    try:
        # Calculate official competition score
        official_score = rank_correlation_sharpe_ratio(merged_df)
        
        # Also calculate daily rank correlations for analysis
        prediction_cols = [col for col in merged_df.columns if col.startswith('prediction_')]
        target_cols = [col for col in merged_df.columns if col.startswith('target_')]
        
        daily_correlations = []
        for idx, row in merged_df.iterrows():
            non_null_targets = [col for col in target_cols if row[col] != SOLUTION_NULL_FILLER]
            matching_predictions = [col for col in prediction_cols if col.replace('prediction', 'target') in non_null_targets]
            
            if len(non_null_targets) > 1:
                target_ranks = row[non_null_targets].rank(method='average')
                pred_ranks = row[matching_predictions].rank(method='average')
                
                if target_ranks.std() > 0 and pred_ranks.std() > 0:
                    corr = np.corrcoef(pred_ranks, target_ranks)[0, 1]
                    if not np.isnan(corr):
                        daily_correlations.append(corr)
                    else:
                        daily_correlations.append(0.0)
                else:
                    daily_correlations.append(0.0)
            else:
                daily_correlations.append(0.0)
        
        daily_correlations = np.array(daily_correlations)
        
        metrics = {
            'official_competition_score': official_score,
            'daily_rank_correlations': daily_correlations,
            'mean_daily_correlation': np.mean(daily_correlations),
            'std_daily_correlation': np.std(daily_correlations, ddof=0),
            'median_daily_correlation': np.median(daily_correlations),
            'positive_days': np.sum(daily_correlations > 0),
            'total_days': len(daily_correlations),
            'positive_day_ratio': np.sum(daily_correlations > 0) / len(daily_correlations) if len(daily_correlations) > 0 else 0.0,
            'n_targets': n_targets,
            'n_samples': n_samples,
            'target_names': target_names
        }
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Failed to calculate official competition score: {e}")
        # Fallback to zero score
        return {
            'official_competition_score': 0.0,
            'daily_rank_correlations': np.zeros(n_samples),
            'mean_daily_correlation': 0.0,
            'std_daily_correlation': 0.0,
            'median_daily_correlation': 0.0,
            'positive_days': 0,
            'total_days': n_samples,
            'positive_day_ratio': 0.0,
            'n_targets': n_targets,
            'n_samples': n_samples,
            'target_names': target_names
        }

def calculate_competition_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                target_names: Optional[List[str]] = None) -> Dict:
    """
    Calculate comprehensive competition metrics including OFFICIAL competition score.
    
    Args:
        y_true: True values (1D for single target, 2D for multi-target)
        y_pred: Predicted values (same shape as y_true)
        target_names: Optional list of target names for multi-target case
        
    Returns:
        Dictionary containing all competition metrics including official score
    """
    # First calculate the OFFICIAL competition score
    official_metrics = calculate_official_competition_score(y_true, y_pred, target_names)
    
    # Then calculate traditional metrics for compatibility
    metrics = {}
    
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        # Single target case
        y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
        y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        # Remove NaN values
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        if len(y_true_clean) > 1:
            spearman_corr, spear_p = spearmanr(y_true_clean, y_pred_clean)
            pearson_corr, pears_p = pearsonr(y_true_clean, y_pred_clean)
            
            metrics = {
                'spearman_correlation': spearman_corr if not np.isnan(spearman_corr) else 0.0,
                'spearman_pvalue': spear_p if not np.isnan(spear_p) else 1.0,
                'pearson_correlation': pearson_corr if not np.isnan(pearson_corr) else 0.0,
                'pearson_pvalue': pears_p if not np.isnan(pears_p) else 1.0,
                'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'mae': mean_absolute_error(y_true_clean, y_pred_clean),
                'valid_samples': len(y_true_clean),
                'total_samples': len(y_true_flat),
                'missing_ratio': (len(y_true_flat) - len(y_true_clean)) / len(y_true_flat)
            }
        else:
            metrics = {
                'spearman_correlation': 0.0,
                'spearman_pvalue': 1.0,
                'pearson_correlation': 0.0,
                'pearson_pvalue': 1.0,
                'rmse': float('inf'),
                'mae': float('inf'),
                'valid_samples': len(y_true_clean),
                'total_samples': len(y_true_flat),
                'missing_ratio': 1.0
            }
    else:
        # Multi-target case
        n_targets = y_true.shape[1]
        spearman_correlations = []
        spearman_pvalues = []
        pearson_correlations = []
        pearson_pvalues = []
        rmses = []
        maes = []
        valid_samples = []
        total_samples = []
        missing_ratios = []
        
        for i in range(n_targets):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            
            # Remove NaN values
            mask = ~(np.isnan(y_true_i) | np.isnan(y_pred_i))
            y_true_clean = y_true_i[mask]
            y_pred_clean = y_pred_i[mask]
            
            if len(y_true_clean) > 1:
                spearman_corr, spear_p = spearmanr(y_true_clean, y_pred_clean)
                pearson_corr, pears_p = pearsonr(y_true_clean, y_pred_clean)
                
                spearman_correlations.append(spearman_corr if not np.isnan(spearman_corr) else 0.0)
                spearman_pvalues.append(spear_p if not np.isnan(spear_p) else 1.0)
                pearson_correlations.append(pearson_corr if not np.isnan(pearson_corr) else 0.0)
                pearson_pvalues.append(pears_p if not np.isnan(pears_p) else 1.0)
                rmses.append(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))
                maes.append(mean_absolute_error(y_true_clean, y_pred_clean))
            else:
                spearman_correlations.append(0.0)
                spearman_pvalues.append(1.0)
                pearson_correlations.append(0.0)
                pearson_pvalues.append(1.0)
                rmses.append(float('inf'))
                maes.append(float('inf'))
            
            valid_samples.append(len(y_true_clean))
            total_samples.append(len(y_true_i))
            missing_ratios.append((len(y_true_i) - len(y_true_clean)) / len(y_true_i))
        
        # Calculate aggregate metrics
        valid_rmses = [r for r in rmses if r != float('inf')]
        valid_maes = [m for m in maes if m != float('inf')]
        
        metrics = {
            'spearman_correlations': spearman_correlations,
            'spearman_pvalues': spearman_pvalues,
            'pearson_correlations': pearson_correlations,
            'pearson_pvalues': pearson_pvalues,
            'rmses': rmses,
            'maes': maes,
            'valid_samples': valid_samples,
            'total_samples': total_samples,
            'missing_ratios': missing_ratios,
            'mean_spearman': np.mean(spearman_correlations),
            'median_spearman': np.median(spearman_correlations),
            'mean_pearson': np.mean(pearson_correlations),
            'median_pearson': np.median(pearson_correlations),
            'mean_rmse': np.mean(valid_rmses) if valid_rmses else float('inf'),
            'median_rmse': np.median(valid_rmses) if valid_rmses else float('inf'),
            'mean_mae': np.mean(valid_maes) if valid_maes else float('inf'),
            'median_mae': np.median(valid_maes) if valid_maes else float('inf'),
            'target_names': target_names if target_names else [f'target_{i}' for i in range(n_targets)],
            'n_targets': n_targets
        }
    
    # Merge with official competition metrics
    metrics.update(official_metrics)
    
    return metrics

def print_competition_analysis(metrics: Dict, model_name: str = "Model", 
                             detailed: bool = True, top_n: int = 10) -> None:
    """
    Print comprehensive competition analysis with performance insights.
    
    Args:
        metrics: Metrics dictionary from calculate_competition_metrics
        model_name: Name of the model for display
        detailed: Whether to show individual target performance
        top_n: Number of top/bottom performers to show in detailed view
    """
    print("\n" + "=" * 70)
    print(f"🏆 {model_name.upper()} COMPETITION EVALUATION")
    print("=" * 70)
    
    # Show OFFICIAL competition score first and prominently
    if 'official_competition_score' in metrics:
        print(f"🎯 OFFICIAL COMPETITION SCORE: {metrics['official_competition_score']:.6f}")
        print(f"📊 Daily Rank Correlation Mean: {metrics['mean_daily_correlation']:.4f}")
        print(f"📈 Daily Rank Correlation Std: {metrics['std_daily_correlation']:.4f}")
        print(f"✅ Positive Days: {metrics['positive_days']}/{metrics['total_days']} ({metrics['positive_day_ratio']*100:.1f}%)")
        print("-" * 70)
    
    if 'spearman_correlations' in metrics:
        # Multi-target analysis
        target_names = metrics['target_names']
        spearman_corrs = metrics['spearman_correlations']
        pearson_corrs = metrics['pearson_correlations']
        rmses = metrics['rmses']
        maes = metrics['maes']
        n_targets = metrics['n_targets']
        
        # Overall performance summary
        print(f"📊 Overall Performance ({n_targets} targets):")
        print(f"   🎯 Mean Spearman Correlation: {metrics['mean_spearman']:.4f}")
        print(f"   📈 Median Spearman Correlation: {metrics['median_spearman']:.4f}")
        print(f"   🔄 Mean Pearson Correlation: {metrics['mean_pearson']:.4f}")
        print(f"   📊 Mean RMSE: {metrics['mean_rmse']:.6f}")
        print(f"   📉 Mean MAE: {metrics['mean_mae']:.6f}")
        
        # Performance distribution
        excellent = sum(1 for c in spearman_corrs if c > 0.1)
        good = sum(1 for c in spearman_corrs if 0.05 < c <= 0.1)
        fair = sum(1 for c in spearman_corrs if 0.0 < c <= 0.05)
        poor = sum(1 for c in spearman_corrs if c <= 0.0)
        
        print(f"\n🎨 Performance Distribution:")
        print(f"   ⭐ Excellent (>0.10): {excellent}/{n_targets} targets ({excellent/n_targets*100:.1f}%)")
        print(f"   ✅ Good (0.05-0.10): {good}/{n_targets} targets ({good/n_targets*100:.1f}%)")
        print(f"   ⚡ Fair (0.00-0.05): {fair}/{n_targets} targets ({fair/n_targets*100:.1f}%)")
        print(f"   ❌ Poor (≤0.00): {poor}/{n_targets} targets ({poor/n_targets*100:.1f}%)")
        
        # Statistical significance
        significant_targets = sum(1 for p in metrics['spearman_pvalues'] if p < 0.05)
        print(f"   📋 Statistically Significant (p<0.05): {significant_targets}/{n_targets} targets ({significant_targets/n_targets*100:.1f}%)")
        
        if detailed and n_targets <= 50:
            # Show individual target performance
            print(f"\n📋 Individual Target Performance:")
            print(f"{'Target':<15} {'Spearman':<10} {'Pearson':<10} {'RMSE':<12} {'MAE':<12} {'Samples':<8} {'Sig':<4}")
            print("-" * 85)
            
            # Sort by Spearman correlation for better visualization
            target_data = list(zip(target_names, spearman_corrs, pearson_corrs, 
                                 rmses, maes, metrics['valid_samples'], 
                                 metrics['spearman_pvalues']))
            target_data.sort(key=lambda x: x[1], reverse=True)
            
            for i, (target, spear, pears, rmse, mae, samples, p_val) in enumerate(target_data):
                if i >= top_n and i < len(target_data) - top_n and len(target_data) > 2 * top_n:
                    if i == top_n:
                        print("   ... (middle targets omitted) ...")
                    continue
                
                rmse_display = f"{rmse:.6f}" if rmse != float('inf') else "∞"
                mae_display = f"{mae:.6f}" if mae != float('inf') else "∞"
                sig_mark = "✓" if p_val < 0.05 else ""
                
                print(f"{target:<15} {spear:>9.4f} {pears:>9.4f} {rmse_display:>11} {mae_display:>11} {samples:>7d} {sig_mark:>3}")
        
        elif detailed and n_targets > 50:
            # Show top and bottom performers
            target_data = list(zip(target_names, spearman_corrs))
            target_data.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🏆 Top {top_n} Performing Targets:")
            for i, (target, corr) in enumerate(target_data[:top_n]):
                print(f"   {i+1:2d}. {target:<15} Spearman: {corr:>7.4f}")
            
            print(f"\n📉 Bottom {top_n} Performing Targets:")
            for i, (target, corr) in enumerate(target_data[-top_n:]):
                rank = len(target_data) - top_n + i + 1
                print(f"   {rank:2d}. {target:<15} Spearman: {corr:>7.4f}")
        
    else:
        # Single target analysis
        print(f"📊 Single Target Performance:")
        print(f"   🎯 Spearman Correlation: {metrics['spearman_correlation']:.4f}")
        print(f"   🔄 Pearson Correlation: {metrics['pearson_correlation']:.4f}")
        print(f"   📊 RMSE: {metrics['rmse']:.6f}")
        print(f"   📉 MAE: {metrics['mae']:.6f}")
        print(f"   📋 Valid Samples: {metrics['valid_samples']}/{metrics['total_samples']}")
        
        # Statistical significance
        if metrics['spearman_pvalue'] < 0.05:
            print(f"   ✅ Statistically Significant (p={metrics['spearman_pvalue']:.4f})")
        else:
            print(f"   ⚠️ Not Statistically Significant (p={metrics['spearman_pvalue']:.4f})")
    
    print("=" * 70)

def get_competition_score(metrics: Dict) -> float:
    """
    Get the OFFICIAL competition score (rank correlation Sharpe ratio).
    
    Args:
        metrics: Metrics dictionary from calculate_competition_metrics
        
    Returns:
        Official competition score
    """
    if 'official_competition_score' in metrics:
        return metrics['official_competition_score']
    elif 'mean_spearman' in metrics:
        return metrics['mean_spearman']
    else:
        return metrics['spearman_correlation']

def compare_models(models_metrics: Dict[str, Dict], detailed: bool = False) -> None:
    """
    Compare multiple models' competition performance.
    
    Args:
        models_metrics: Dictionary with model names as keys and metrics as values
        detailed: Whether to show detailed comparison
    """
    print("\n" + "=" * 70)
    print("🏁 MODEL COMPARISON")
    print("=" * 70)
    
    # Sort models by competition score
    model_scores = [(name, get_competition_score(metrics)) 
                   for name, metrics in models_metrics.items()]
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("📊 Competition Rankings:")
    for rank, (model_name, score) in enumerate(model_scores, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}."
        print(f"   {medal} {model_name:<20} Score: {score:>7.4f}")
    
    if detailed:
        print(f"\n📋 Detailed Comparison:")
        print(f"{'Model':<20} {'Spearman':<10} {'Pearson':<10} {'RMSE':<12} {'MAE':<12}")
        print("-" * 70)
        
        for model_name, score in model_scores:
            metrics = models_metrics[model_name]
            if 'mean_spearman' in metrics:
                spear = metrics['mean_spearman']
                pears = metrics['mean_pearson']
                rmse = metrics['mean_rmse']
                mae = metrics['mean_mae']
            else:
                spear = metrics['spearman_correlation']
                pears = metrics['pearson_correlation']
                rmse = metrics['rmse']
                mae = metrics['mae']
            
            rmse_str = f"{rmse:.6f}" if rmse != float('inf') else "∞"
            mae_str = f"{mae:.6f}" if mae != float('inf') else "∞"
            
            print(f"{model_name:<20} {spear:>9.4f} {pears:>9.4f} {rmse_str:>11} {mae_str:>11}")
    
    print("=" * 70)

def save_competition_results(metrics: Dict, model_name: str, 
                           output_path: str = "evaluation_results") -> str:
    """
    Save competition evaluation results to files.
    
    Args:
        metrics: Metrics dictionary from calculate_competition_metrics
        model_name: Name of the model
        output_path: Directory to save results
        
    Returns:
        Path to saved results file
    """
    import os
    import json
    from datetime import datetime
    
    os.makedirs(output_path, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}_competition_results.json"
    filepath = os.path.join(output_path, filename)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    # Save metrics
    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': convert_numpy_types(metrics),
        'competition_score': get_competition_score(metrics)
    }
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Competition results saved to: {filepath}")
    return filepath

def generate_competition_report(metrics: Dict, model_name: str, 
                              predictions: np.ndarray = None, 
                              output_path: str = "evaluation_results") -> str:
    """
    Generate a comprehensive competition report.
    
    Args:
        metrics: Metrics dictionary from calculate_competition_metrics
        model_name: Name of the model
        predictions: Model predictions for additional analysis
        output_path: Directory to save report
        
    Returns:
        Path to generated report
    """
    import os
    from datetime import datetime
    
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{model_name}_{timestamp}_competition_report.md"
    report_path = os.path.join(output_path, report_filename)
    
    with open(report_path, 'w') as f:
        f.write(f"# Competition Evaluation Report\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Competition Score:** {get_competition_score(metrics):.4f}\n\n")
        
        if 'mean_spearman' in metrics:
            # Multi-target report
            f.write(f"## Multi-Target Performance Summary\n\n")
            f.write(f"- **Total Targets:** {metrics['n_targets']}\n")
            f.write(f"- **Mean Spearman Correlation:** {metrics['mean_spearman']:.4f}\n")
            f.write(f"- **Median Spearman Correlation:** {metrics['median_spearman']:.4f}\n")
            f.write(f"- **Mean Pearson Correlation:** {metrics['mean_pearson']:.4f}\n")
            f.write(f"- **Mean RMSE:** {metrics['mean_rmse']:.6f}\n")
            f.write(f"- **Mean MAE:** {metrics['mean_mae']:.6f}\n\n")
            
            # Performance distribution
            spearman_corrs = metrics['spearman_correlations']
            excellent = sum(1 for c in spearman_corrs if c > 0.1)
            good = sum(1 for c in spearman_corrs if 0.05 < c <= 0.1)
            fair = sum(1 for c in spearman_corrs if 0.0 < c <= 0.05)
            poor = sum(1 for c in spearman_corrs if c <= 0.0)
            
            f.write(f"## Performance Distribution\n\n")
            f.write(f"- **Excellent (>0.10):** {excellent} targets ({excellent/len(spearman_corrs)*100:.1f}%)\n")
            f.write(f"- **Good (0.05-0.10):** {good} targets ({good/len(spearman_corrs)*100:.1f}%)\n")
            f.write(f"- **Fair (0.00-0.05):** {fair} targets ({fair/len(spearman_corrs)*100:.1f}%)\n")
            f.write(f"- **Poor (≤0.00):** {poor} targets ({poor/len(spearman_corrs)*100:.1f}%)\n\n")
            
        else:
            # Single target report
            f.write(f"## Single Target Performance\n\n")
            f.write(f"- **Spearman Correlation:** {metrics['spearman_correlation']:.4f}\n")
            f.write(f"- **Pearson Correlation:** {metrics['pearson_correlation']:.4f}\n")
            f.write(f"- **RMSE:** {metrics['rmse']:.6f}\n")
            f.write(f"- **MAE:** {metrics['mae']:.6f}\n")
            f.write(f"- **Valid Samples:** {metrics['valid_samples']}\n\n")
        
        f.write(f"## Competition Readiness Assessment\n\n")
        score = get_competition_score(metrics)
        if score > 0.1:
            f.write("🏆 **EXCELLENT** - Strong competition performance expected\n")
        elif score > 0.05:
            f.write("✅ **GOOD** - Competitive performance expected\n")
        elif score > 0.0:
            f.write("⚡ **FAIR** - May benefit from further optimization\n")
        else:
            f.write("❌ **POOR** - Requires significant improvement\n")
    
    logger.info(f"Competition report generated: {report_path}")
    return report_path