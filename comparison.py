# -*- coding: utf-8 -*-
"""
Model Comparison and Statistical Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
import json
import os


# ============================================================================
# Statistical Tests
# ============================================================================

def paired_ttest(metrics1, metrics2):
    """
    Perform paired t-test

    Args:
        metrics1: list or array of metric values from model 1
        metrics2: list or array of metric values from model 2

    Returns:
        t_statistic, p_value
    """
    metrics1 = np.array(metrics1)
    metrics2 = np.array(metrics2)

    if len(metrics1) != len(metrics2):
        raise ValueError("Metrics arrays must have the same length")

    t_statistic, p_value = stats.ttest_rel(metrics1, metrics2)
    return t_statistic, p_value


def wilcoxon_test(metrics1, metrics2):
    """
    Perform Wilcoxon signed-rank test (non-parametric)

    Args:
        metrics1: list or array of metric values from model 1
        metrics2: list or array of metric values from model 2

    Returns:
        statistic, p_value
    """
    metrics1 = np.array(metrics1)
    metrics2 = np.array(metrics2)

    if len(metrics1) != len(metrics2):
        raise ValueError("Metrics arrays must have the same length")

    statistic, p_value = stats.wilcoxon(metrics1, metrics2)
    return statistic, p_value


def cohen_d(metrics1, metrics2):
    """
    Calculate Cohen's d effect size

    Args:
        metrics1: list or array of metric values from model 1
        metrics2: list or array of metric values from model 2

    Returns:
        effect_size: Cohen's d value
    """
    metrics1 = np.array(metrics1)
    metrics2 = np.array(metrics2)

    mean_diff = np.mean(metrics1) - np.mean(metrics2)
    pooled_std = np.sqrt((np.std(metrics1, ddof=1)**2 + np.std(metrics2, ddof=1)**2) / 2)

    if pooled_std == 0:
        return 0.0

    return mean_diff / pooled_std


# ============================================================================
# Pairwise Model Comparison
# ============================================================================

def pairwise_comparison(all_results, metric_names=None):
    """
    Perform pairwise statistical comparison between all models

    Args:
        all_results: dict of {model_name: {'dice': [values], 'iou': [values], ...}}
        metric_names: list of metrics to compare (if None, use all)

    Returns:
        comparison_results: nested dict with statistical test results
    """
    if metric_names is None:
        # Get all metric names from first model
        first_model = list(all_results.keys())[0]
        metric_names = [k for k in all_results[first_model].keys() if isinstance(all_results[first_model][k], list)]

    model_names = list(all_results.keys())
    comparison_results = {}

    print("\n" + "="*80)
    print("PAIRWISE STATISTICAL COMPARISON")
    print("="*80)

    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            pair_key = (model1, model2)
            comparison_results[pair_key] = {}

            print(f"\n{model1.upper()} vs {model2.upper()}")
            print("-" * 60)

            for metric in metric_names:
                if metric not in all_results[model1] or metric not in all_results[model2]:
                    continue

                values1 = all_results[model1][metric]
                values2 = all_results[model2][metric]

                # Paired t-test
                t_stat, p_value_t = paired_ttest(values1, values2)

                # Wilcoxon test
                w_stat, p_value_w = wilcoxon_test(values1, values2)

                # Effect size
                effect_size = cohen_d(values1, values2)

                # Store results
                comparison_results[pair_key][metric] = {
                    't_statistic': float(t_stat),
                    'p_value_ttest': float(p_value_t),
                    'w_statistic': float(w_stat),
                    'p_value_wilcoxon': float(p_value_w),
                    'cohen_d': float(effect_size),
                    'mean_diff': float(np.mean(values1) - np.mean(values2))
                }

                # Determine significance
                sig = "***" if p_value_t < 0.001 else "**" if p_value_t < 0.01 else "*" if p_value_t < 0.05 else "ns"

                print(f"  {metric:15s}: p={p_value_t:.4f} {sig:3s} | "
                      f"Cohen's d={effect_size:+.3f} | "
                      f"Δ={np.mean(values1) - np.mean(values2):+.4f}")

    return comparison_results


# ============================================================================
# Summary Tables
# ============================================================================

def create_summary_table(all_metrics, save_path=None):
    """
    Create summary table of all metrics

    Args:
        all_metrics: dict of {model_name: {metric_name: value}}
        save_path: path to save CSV (optional)

    Returns:
        DataFrame with summary statistics
    """
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics).T

    # Order columns
    desired_order = ['dice', 'iou', 'sensitivity', 'specificity', 'precision',
                     'accuracy', 'boundary_f1', 'assd', 'auc']
    columns = [col for col in desired_order if col in df.columns]
    df = df[columns]

    # Round values
    df = df.round(4)

    # Add rank for each metric (lower is better for ASSD, higher for others)
    rank_df = df.copy()
    for col in df.columns:
        if col == 'assd':
            rank_df[col] = df[col].rank(ascending=True)
        else:
            rank_df[col] = df[col].rank(ascending=False)

    # Calculate average rank
    df['avg_rank'] = rank_df.mean(axis=1).round(2)

    # Sort by average rank
    df = df.sort_values('avg_rank')

    # Print table
    print("\n" + "="*120)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("="*120)
    print(df.to_string())
    print("="*120)

    # Save to CSV
    if save_path:
        df.to_csv(save_path)
        print(f"\nSummary table saved to {save_path}")

    return df


def create_detailed_table(all_results, metric_names=None, save_path=None):
    """
    Create detailed table with mean ± std for each metric

    Args:
        all_results: dict of {model_name: {'dice': [values], 'iou': [values], ...}}
        metric_names: list of metrics to include
        save_path: path to save CSV

    Returns:
        DataFrame with mean ± std values
    """
    if metric_names is None:
        first_model = list(all_results.keys())[0]
        metric_names = [k for k in all_results[first_model].keys()
                       if isinstance(all_results[first_model][k], list)]

    # Create table data
    table_data = {}

    for model_name, results in all_results.items():
        table_data[model_name] = {}

        for metric in metric_names:
            if metric in results and isinstance(results[metric], list):
                values = results[metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                table_data[model_name][metric] = f"{mean_val:.4f} ± {std_val:.4f}"

    # Convert to DataFrame
    df = pd.DataFrame(table_data).T

    # Print table
    print("\n" + "="*120)
    print("DETAILED METRICS (Mean ± Std)")
    print("="*120)
    print(df.to_string())
    print("="*120)

    # Save to CSV
    if save_path:
        df.to_csv(save_path)
        print(f"\nDetailed table saved to {save_path}")

    return df


# ============================================================================
# Ranking and Best Model Selection
# ============================================================================

def rank_models(all_metrics):
    """
    Rank models based on multiple metrics

    Args:
        all_metrics: dict of {model_name: {metric_name: value}}

    Returns:
        ranking: DataFrame with ranks and scores
    """
    df = pd.DataFrame(all_metrics).T

    # Metrics to consider for ranking
    ranking_metrics = ['dice', 'iou', 'sensitivity', 'boundary_f1', 'auc']
    ranking_metrics = [m for m in ranking_metrics if m in df.columns]

    # Add ASSD if available (invert since lower is better)
    if 'assd' in df.columns:
        df['assd_inv'] = 1.0 / (df['assd'] + 0.01)
        ranking_metrics.append('assd_inv')

    # Calculate ranks for each metric
    ranks = pd.DataFrame(index=df.index)
    for metric in ranking_metrics:
        ranks[f'{metric}_rank'] = df[metric].rank(ascending=False)

    # Calculate average rank
    ranks['avg_rank'] = ranks.mean(axis=1)

    # Calculate normalized score (0-100)
    max_rank = len(df)
    ranks['score'] = 100 * (max_rank - ranks['avg_rank'] + 1) / max_rank

    # Sort by score
    ranks = ranks.sort_values('score', ascending=False)

    print("\n" + "="*80)
    print("MODEL RANKING")
    print("="*80)
    print(ranks[['avg_rank', 'score']].to_string())
    print("="*80)

    return ranks


# ============================================================================
# Save and Load Results
# ============================================================================

def save_results(all_metrics, all_results, comparison_results, save_dir='results'):
    """
    Save all results to JSON files

    Args:
        all_metrics: dict of average metrics
        all_results: dict of detailed results
        comparison_results: dict of statistical comparison results
        save_dir: directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # Save average metrics
    with open(os.path.join(save_dir, 'average_metrics.json'), 'w') as f:
        json.dump(convert_to_serializable(all_metrics), f, indent=2)

    # Save detailed results
    with open(os.path.join(save_dir, 'detailed_results.json'), 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    # Save comparison results
    # Convert tuple keys to strings for JSON
    comparison_json = {}
    for (model1, model2), results in comparison_results.items():
        key = f"{model1}_vs_{model2}"
        comparison_json[key] = convert_to_serializable(results)

    with open(os.path.join(save_dir, 'statistical_comparison.json'), 'w') as f:
        json.dump(comparison_json, f, indent=2)

    print(f"\nAll results saved to {save_dir}/")


def load_results(save_dir='results'):
    """
    Load results from JSON files

    Returns:
        all_metrics, all_results, comparison_results
    """
    with open(os.path.join(save_dir, 'average_metrics.json'), 'r') as f:
        all_metrics = json.load(f)

    with open(os.path.join(save_dir, 'detailed_results.json'), 'r') as f:
        all_results = json.load(f)

    with open(os.path.join(save_dir, 'statistical_comparison.json'), 'r') as f:
        comparison_json = json.load(f)

    # Convert string keys back to tuples
    comparison_results = {}
    for key, results in comparison_json.items():
        model1, model2 = key.split('_vs_')
        comparison_results[(model1, model2)] = results

    return all_metrics, all_results, comparison_results


# ============================================================================
# Generate Report
# ============================================================================

def generate_text_report(all_metrics, all_results, comparison_results,
                        save_path='results/report.txt'):
    """
    Generate a comprehensive text report

    Args:
        all_metrics: dict of average metrics
        all_results: dict of detailed results
        comparison_results: dict of statistical comparison results
        save_path: path to save report
    """
    lines = []

    lines.append("="*80)
    lines.append("MEDICAL IMAGE SEGMENTATION - COMPREHENSIVE EVALUATION REPORT")
    lines.append("="*80)
    lines.append("")

    # 1. Overview
    lines.append("1. OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"Number of models evaluated: {len(all_metrics)}")
    lines.append(f"Models: {', '.join(all_metrics.keys())}")
    lines.append("")

    # 2. Summary metrics
    lines.append("2. SUMMARY METRICS")
    lines.append("-" * 80)
    df = pd.DataFrame(all_metrics).T
    lines.append(df.to_string())
    lines.append("")

    # 3. Ranking
    lines.append("3. MODEL RANKING")
    lines.append("-" * 80)
    ranks = rank_models(all_metrics)
    for idx, (model, row) in enumerate(ranks.iterrows(), 1):
        lines.append(f"{idx}. {model}: Score={row['score']:.2f}, Avg Rank={row['avg_rank']:.2f}")
    lines.append("")

    # 4. Statistical Comparison
    lines.append("4. STATISTICAL SIGNIFICANCE (Paired t-test)")
    lines.append("-" * 80)
    for (model1, model2), results in comparison_results.items():
        lines.append(f"\n{model1} vs {model2}:")
        for metric, stats in results.items():
            p_val = stats['p_value_ttest']
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            lines.append(f"  {metric:15s}: p={p_val:.4f} {sig:3s} | Cohen's d={stats['cohen_d']:+.3f}")
    lines.append("")

    # 5. Key Findings
    lines.append("5. KEY FINDINGS")
    lines.append("-" * 80)

    # Best model for each metric
    for metric in ['dice', 'iou', 'sensitivity', 'boundary_f1']:
        if metric in df.columns:
            best_model = df[metric].idxmax()
            best_value = df[metric].max()
            lines.append(f"Best {metric.upper():15s}: {best_model:20s} ({best_value:.4f})")

    lines.append("")
    lines.append("="*80)
    lines.append("END OF REPORT")
    lines.append("="*80)

    # Write to file
    report_text = '\n'.join(lines)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(report_text)

    print(f"\nComprehensive report saved to {save_path}")
    print("\n" + report_text)

    return report_text


if __name__ == "__main__":
    # Test comparison module
    print("Testing comparison module...")
    print("="*60)

    # Create dummy results
    np.random.seed(42)
    models = ['unet', 'unet_eam', 'transunet', 'transunet_eam']
    metrics_list = ['dice', 'iou', 'sensitivity', 'boundary_f1']

    all_results = {}
    all_metrics = {}

    for i, model in enumerate(models):
        all_results[model] = {}
        all_metrics[model] = {}

        for metric in metrics_list:
            # Simulate increasing performance
            base = 0.70 + i * 0.02
            values = np.random.normal(base, 0.05, 20)
            values = np.clip(values, 0, 1)

            all_results[model][metric] = values.tolist()
            all_metrics[model][metric] = float(np.mean(values))

    # Test functions
    print("\n1. Testing pairwise comparison...")
    comparison_results = pairwise_comparison(all_results)

    print("\n2. Testing summary table...")
    summary_df = create_summary_table(all_metrics)

    print("\n3. Testing ranking...")
    ranks = rank_models(all_metrics)

    print("\n4. Testing save/load...")
    save_results(all_metrics, all_results, comparison_results, 'test_results')

    print("\n5. Testing report generation...")
    generate_text_report(all_metrics, all_results, comparison_results,
                        'test_results/report.txt')

    print("\n✓ All comparison tests passed!")
