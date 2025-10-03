"""
Test Evaluation and Visualization Module for Mitsui Models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import json
from datetime import datetime

from mitsui_data_loader import MitsuiDataLoader
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEvaluator:
    """
    Comprehensive test evaluation with visualizations and statistics
    """
    
    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def generate_test_features(self, data_loader: MitsuiDataLoader, feature_engineer: FeatureEngineer) -> pd.DataFrame:
        """Generate features for test data using historical training data"""
        logger.info("Generating test features...")
        
        # Combine train and test data for feature generation
        combined_data = pd.concat([data_loader.train_data, data_loader.test_data], ignore_index=True)
        combined_data = combined_data.sort_values('date_id').reset_index(drop=True)
        
        # Generate features on combined data
        combined_features = feature_engineer.generate_features(combined_data)
        
        # Extract test features (keep only test portion)
        train_size = len(data_loader.train_data)
        test_features = combined_features.iloc[train_size:].reset_index(drop=True)
        
        logger.info(f"Test features generated: {test_features.shape}")
        return test_features
    
    def evaluate_single_model(self, model, test_features: pd.DataFrame, target_col: str, 
                             train_correlation: float) -> Dict[str, Any]:
        """Evaluate a single model on test data"""
        
        # Prepare test features (exclude date_id)
        feature_cols = [col for col in test_features.columns if col != 'date_id']
        X_test = test_features[feature_cols].fillna(0).values
        
        # Generate predictions
        predictions = model.predict(X_test)
        
        # Since we don't have true test labels, we'll analyze prediction characteristics
        results = {
            'target_col': target_col,
            'n_test_samples': len(predictions),
            'train_spearman': train_correlation,
            'pred_mean': float(np.mean(predictions)),
            'pred_std': float(np.std(predictions)),
            'pred_min': float(np.min(predictions)),
            'pred_max': float(np.max(predictions)),
            'pred_range': float(np.max(predictions) - np.min(predictions)),
            'predictions': predictions.tolist()
        }
        
        logger.info(f"Test evaluation for {target_col}:")
        logger.info(f"  Test samples: {results['n_test_samples']}")
        logger.info(f"  Prediction range: [{results['pred_min']:.4f}, {results['pred_max']:.4f}]")
        logger.info(f"  Prediction mean ± std: {results['pred_mean']:.4f} ± {results['pred_std']:.4f}")
        
        return results
    
    def create_prediction_plots(self, evaluation_results: List[Dict], save_plots: bool = True) -> None:
        """Create comprehensive visualization plots"""
        logger.info("Creating prediction visualizations...")
        
        n_targets = len(evaluation_results)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Prediction distributions
        plt.subplot(3, 3, 1)
        for i, result in enumerate(evaluation_results):
            predictions = np.array(result['predictions'])
            plt.hist(predictions, bins=30, alpha=0.7, label=f"{result['target_col']}")
        plt.title('Prediction Distributions')
        plt.xlabel('Prediction Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Prediction ranges comparison
        plt.subplot(3, 3, 2)
        target_names = [r['target_col'] for r in evaluation_results]
        pred_means = [r['pred_mean'] for r in evaluation_results]
        pred_stds = [r['pred_std'] for r in evaluation_results]
        
        x_pos = np.arange(len(target_names))
        plt.bar(x_pos, pred_means, yerr=pred_stds, capsize=5)
        plt.title('Prediction Means with Standard Deviations')
        plt.xlabel('Target')
        plt.ylabel('Prediction Value')
        plt.xticks(x_pos, target_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Training correlations
        plt.subplot(3, 3, 3)
        train_corrs = [r['train_spearman'] for r in evaluation_results]
        plt.bar(x_pos, train_corrs)
        plt.title('Training Spearman Correlations')
        plt.xlabel('Target')
        plt.ylabel('Spearman Correlation')
        plt.xticks(x_pos, target_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 4. Prediction time series (if multiple test samples)
        plt.subplot(3, 3, 4)
        for i, result in enumerate(evaluation_results):
            predictions = np.array(result['predictions'])
            plt.plot(predictions, label=f"{result['target_col']}", alpha=0.8)
        plt.title('Prediction Time Series')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Prediction Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Prediction ranges
        plt.subplot(3, 3, 5)
        pred_ranges = [r['pred_range'] for r in evaluation_results]
        plt.bar(x_pos, pred_ranges)
        plt.title('Prediction Ranges (Max - Min)')
        plt.xlabel('Target')
        plt.ylabel('Range')
        plt.xticks(x_pos, target_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. Box plot of predictions
        plt.subplot(3, 3, 6)
        pred_data = [result['predictions'] for result in evaluation_results]
        plt.boxplot(pred_data, labels=target_names)
        plt.title('Prediction Distributions (Box Plot)')
        plt.xlabel('Target')
        plt.ylabel('Prediction Value')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. Correlation vs Prediction Statistics
        plt.subplot(3, 3, 7)
        plt.scatter(train_corrs, pred_stds, s=100, alpha=0.7)
        for i, result in enumerate(evaluation_results):
            plt.annotate(result['target_col'], 
                        (train_corrs[i], pred_stds[i]),
                        xytext=(5, 5), textcoords='offset points')
        plt.title('Training Correlation vs Prediction Std')
        plt.xlabel('Training Spearman Correlation')
        plt.ylabel('Prediction Standard Deviation')
        plt.grid(True, alpha=0.3)
        
        # 8. Summary statistics table
        plt.subplot(3, 3, 8)
        plt.axis('off')
        
        # Create summary table
        summary_data = []
        for result in evaluation_results:
            summary_data.append([
                result['target_col'],
                f"{result['train_spearman']:.3f}",
                f"{result['pred_mean']:.4f}",
                f"{result['pred_std']:.4f}",
                f"{result['pred_range']:.4f}"
            ])
        
        table = plt.table(cellText=summary_data,
                         colLabels=['Target', 'Train Corr', 'Pred Mean', 'Pred Std', 'Pred Range'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Summary Statistics', pad=20)
        
        # 9. Overall statistics
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        # Calculate overall statistics
        avg_correlation = np.mean(train_corrs)
        std_correlation = np.std(train_corrs)
        avg_pred_std = np.mean(pred_stds)
        total_samples = sum(r['n_test_samples'] for r in evaluation_results)
        
        stats_text = f"""
        Overall Performance Summary:
        
        • Targets Evaluated: {len(evaluation_results)}
        • Total Test Samples: {total_samples}
        • Avg Training Correlation: {avg_correlation:.4f} ± {std_correlation:.4f}
        • Avg Prediction Std: {avg_pred_std:.4f}
        • Best Training Correlation: {max(train_corrs):.4f}
        • Worst Training Correlation: {min(train_corrs):.4f}
        
        Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.results_dir / f"prediction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {plot_path}")
        
        plt.show()
        
        return fig
    
    def create_detailed_analysis(self, evaluation_results: List[Dict]) -> Dict[str, Any]:
        """Create detailed statistical analysis"""
        logger.info("Creating detailed statistical analysis...")
        
        # Compile all predictions
        all_predictions = []
        all_train_corrs = []
        
        for result in evaluation_results:
            all_predictions.extend(result['predictions'])
            all_train_corrs.append(result['train_spearman'])
        
        all_predictions = np.array(all_predictions)
        all_train_corrs = np.array(all_train_corrs)
        
        detailed_analysis = {
            'timestamp': datetime.now().isoformat(),
            'n_targets': len(evaluation_results),
            'total_test_samples': len(all_predictions),
            
            # Training performance
            'training_performance': {
                'mean_correlation': float(np.mean(all_train_corrs)),
                'std_correlation': float(np.std(all_train_corrs)),
                'min_correlation': float(np.min(all_train_corrs)),
                'max_correlation': float(np.max(all_train_corrs)),
                'median_correlation': float(np.median(all_train_corrs)),
                'correlations_above_0_1': int(np.sum(all_train_corrs > 0.1)),
                'correlations_above_0_2': int(np.sum(all_train_corrs > 0.2)),
                'correlations_above_0_3': int(np.sum(all_train_corrs > 0.3))
            },
            
            # Prediction characteristics
            'prediction_characteristics': {
                'overall_mean': float(np.mean(all_predictions)),
                'overall_std': float(np.std(all_predictions)),
                'overall_min': float(np.min(all_predictions)),
                'overall_max': float(np.max(all_predictions)),
                'overall_range': float(np.max(all_predictions) - np.min(all_predictions)),
                'percentiles': {
                    '5th': float(np.percentile(all_predictions, 5)),
                    '25th': float(np.percentile(all_predictions, 25)),
                    '50th': float(np.percentile(all_predictions, 50)),
                    '75th': float(np.percentile(all_predictions, 75)),
                    '95th': float(np.percentile(all_predictions, 95))
                }
            },
            
            # Individual target analysis
            'target_analysis': evaluation_results
        }
        
        return detailed_analysis
    
    def save_detailed_analysis(self, analysis: Dict[str, Any]) -> str:
        """Save detailed analysis to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.results_dir / f"detailed_analysis_{timestamp}.json"
        
        with open(file_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Detailed analysis saved to {file_path}")
        return str(file_path)
    
    def print_summary_report(self, analysis: Dict[str, Any]) -> None:
        """Print a comprehensive summary report"""
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST EVALUATION REPORT")
        print("=" * 80)
        
        # Training Performance
        train_perf = analysis['training_performance']
        print(f"\n🎯 TRAINING PERFORMANCE:")
        print(f"   • Average Spearman Correlation: {train_perf['mean_correlation']:.4f} ± {train_perf['std_correlation']:.4f}")
        print(f"   • Best Correlation: {train_perf['max_correlation']:.4f}")
        print(f"   • Worst Correlation: {train_perf['min_correlation']:.4f}")
        print(f"   • Targets with Corr > 0.1: {train_perf['correlations_above_0_1']}/{analysis['n_targets']}")
        print(f"   • Targets with Corr > 0.2: {train_perf['correlations_above_0_2']}/{analysis['n_targets']}")
        print(f"   • Targets with Corr > 0.3: {train_perf['correlations_above_0_3']}/{analysis['n_targets']}")
        
        # Prediction Characteristics
        pred_char = analysis['prediction_characteristics']
        print(f"\n📈 TEST PREDICTION CHARACTERISTICS:")
        print(f"   • Total Test Samples: {analysis['total_test_samples']}")
        print(f"   • Prediction Range: [{pred_char['overall_min']:.4f}, {pred_char['overall_max']:.4f}]")
        print(f"   • Mean ± Std: {pred_char['overall_mean']:.4f} ± {pred_char['overall_std']:.4f}")
        print(f"   • Percentiles (5th, 25th, 50th, 75th, 95th):")
        print(f"     {pred_char['percentiles']['5th']:.4f}, {pred_char['percentiles']['25th']:.4f}, {pred_char['percentiles']['50th']:.4f}, {pred_char['percentiles']['75th']:.4f}, {pred_char['percentiles']['95th']:.4f}")
        
        # Top performing targets
        target_analysis = analysis['target_analysis']
        sorted_targets = sorted(target_analysis, key=lambda x: x['train_spearman'], reverse=True)
        
        print(f"\n🏆 TOP 5 PERFORMING TARGETS (by training correlation):")
        for i, target in enumerate(sorted_targets[:5]):
            print(f"   {i+1}. {target['target_col']}: {target['train_spearman']:.4f}")
        
        print(f"\n📉 BOTTOM 5 PERFORMING TARGETS:")
        for i, target in enumerate(sorted_targets[-5:]):
            print(f"   {i+1}. {target['target_col']}: {target['train_spearman']:.4f}")
        
        print("\n" + "=" * 80)
        print("✅ EVALUATION COMPLETE - Check the generated plots and detailed analysis file!")
        print("=" * 80)


def evaluate_models(models: Dict, data_loader: MitsuiDataLoader, feature_engineer: FeatureEngineer,
                   train_correlations: List[float], target_cols: List[str]) -> None:
    """
    Complete evaluation pipeline for trained models
    """
    logger.info("🔍 Starting comprehensive test evaluation...")
    
    # Initialize evaluator
    evaluator = TestEvaluator()
    
    # Generate test features
    test_features = evaluator.generate_test_features(data_loader, feature_engineer)
    
    # Evaluate each model
    evaluation_results = []
    
    for i, (target_col, model) in enumerate(models.items()):
        logger.info(f"Evaluating {i+1}/{len(models)}: {target_col}")
        
        train_corr = train_correlations[i] if i < len(train_correlations) else 0.0
        result = evaluator.evaluate_single_model(model, test_features, target_col, train_corr)
        evaluation_results.append(result)
    
    # Create visualizations
    logger.info("Creating comprehensive visualizations...")
    evaluator.create_prediction_plots(evaluation_results)
    
    # Create detailed analysis
    detailed_analysis = evaluator.create_detailed_analysis(evaluation_results)
    
    # Save analysis
    analysis_file = evaluator.save_detailed_analysis(detailed_analysis)
    
    # Print summary report
    evaluator.print_summary_report(detailed_analysis)
    
    logger.info("✅ Test evaluation completed successfully!")
    
    return evaluation_results, detailed_analysis