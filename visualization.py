# -*- coding: utf-8 -*-
"""
Visualization Functions for Model Comparison and Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd


# ============================================================================
# Color schemes and styles
# ============================================================================

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Color palette for models
MODEL_COLORS = {
    'unet': '#1f77b4',
    'unet_eam': '#ff7f0e',
    'transunet': '#2ca02c',
    'transunet_eam': '#d62728'
}


# ============================================================================
# Training Curves
# ============================================================================

def plot_training_curves(histories, save_path='training_curves.png'):
    """
    Plot training and validation curves for all models

    Args:
        histories: dict of {model_name: {'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}}
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss curves
    ax = axes[0]
    for model_name, history in histories.items():
        color = MODEL_COLORS.get(model_name.lower(), None)
        epochs = range(1, len(history['train_loss']) + 1)

        ax.plot(epochs, history['train_loss'], '--', alpha=0.6, color=color,
                label=f'{model_name} (Train)')
        ax.plot(epochs, history['val_loss'], '-', linewidth=2, color=color,
                label=f'{model_name} (Val)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot accuracy curves
    ax = axes[1]
    for model_name, history in histories.items():
        color = MODEL_COLORS.get(model_name.lower(), None)
        epochs = range(1, len(history['val_acc']) + 1)

        ax.plot(epochs, history['val_acc'], '-', linewidth=2, color=color,
                label=f'{model_name}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


# ============================================================================
# ROC Curves
# ============================================================================

def plot_roc_curves(roc_data, save_path='roc_curves.png'):
    """
    Plot ROC curves for all models

    Args:
        roc_data: dict of {model_name: (fpr, tpr, auc_score)}
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

    # Plot ROC curve for each model
    for model_name, (fpr, tpr, auc_score) in roc_data.items():
        color = MODEL_COLORS.get(model_name.lower(), None)
        ax.plot(fpr, tpr, linewidth=2.5, color=color,
                label=f'{model_name} (AUC = {auc_score:.4f})')

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves - Model Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()


# ============================================================================
# Metrics Comparison
# ============================================================================

def plot_metrics_comparison(all_metrics, save_path='metrics_comparison.png'):
    """
    Plot bar charts comparing all metrics across models

    Args:
        all_metrics: dict of {model_name: {metric_name: value}}
        save_path: path to save figure
    """
    # Metrics to plot
    metrics_to_plot = ['dice', 'iou', 'sensitivity', 'specificity',
                       'precision', 'boundary_f1', 'auc']

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    model_names = list(all_metrics.keys())
    x = np.arange(len(model_names))
    width = 0.6

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Get values for this metric
        values = [all_metrics[model][metric_name] for model in model_names]
        colors = [MODEL_COLORS.get(model.lower(), '#333333') for model in model_names]

        # Create bar chart
        bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Formatting
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

    # Remove extra subplot
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Metrics comparison saved to {save_path}")
    plt.close()


# ============================================================================
# Error Maps
# ============================================================================

def plot_error_maps(image, ground_truth, predictions, model_names,
                    save_path='error_maps.png'):
    """
    Plot error maps showing TP, FP, TN, FN for each model

    Args:
        image: (H, W) input image
        ground_truth: (H, W) ground truth mask
        predictions: dict of {model_name: (H, W) prediction}
        model_names: list of model names to plot
        save_path: path to save figure
    """
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models + 2, figsize=(4 * (n_models + 2), 4))

    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Plot ground truth
    axes[1].imshow(ground_truth, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Define colors for error map
    # TP: Green, FP: Red, TN: Black, FN: Blue
    colors = {
        'TP': [0, 1, 0],      # Green
        'FP': [1, 0, 0],      # Red
        'FN': [0, 0, 1],      # Blue
        'TN': [0, 0, 0]       # Black
    }

    # Plot error maps for each model
    for idx, model_name in enumerate(model_names):
        pred = predictions[model_name]

        # Create error map
        error_map = np.zeros((*pred.shape, 3))

        # True Positives (both predicted and ground truth are 1)
        tp_mask = (pred == 1) & (ground_truth == 1)
        error_map[tp_mask] = colors['TP']

        # False Positives (predicted 1, ground truth 0)
        fp_mask = (pred == 1) & (ground_truth == 0)
        error_map[fp_mask] = colors['FP']

        # False Negatives (predicted 0, ground truth 1)
        fn_mask = (pred == 0) & (ground_truth == 1)
        error_map[fn_mask] = colors['FN']

        # True Negatives (both predicted and ground truth are 0)
        tn_mask = (pred == 0) & (ground_truth == 0)
        error_map[tn_mask] = colors['TN']

        axes[idx + 2].imshow(error_map)
        axes[idx + 2].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx + 2].axis('off')

    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='green', label='True Positive'),
        Rectangle((0, 0), 1, 1, fc='red', label='False Positive'),
        Rectangle((0, 0), 1, 1, fc='blue', label='False Negative'),
        Rectangle((0, 0), 1, 1, fc='black', label='True Negative')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
              bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    print(f"Error maps saved to {save_path}")
    plt.close()


# ============================================================================
# Segmentation Results Visualization
# ============================================================================

def plot_segmentation_results(images, ground_truths, predictions, model_names,
                              num_samples=5, save_path='segmentation_results.png'):
    """
    Plot segmentation results for multiple samples

    Args:
        images: list of (H, W) input images
        ground_truths: list of (H, W) ground truth masks
        predictions: dict of {model_name: list of (H, W) predictions}
        model_names: list of model names
        num_samples: number of samples to visualize
        save_path: path to save figure
    """
    n_models = len(model_names)
    fig, axes = plt.subplots(num_samples, n_models + 2, figsize=(4 * (n_models + 2), 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Plot input image
        axes[i, 0].imshow(images[i], cmap='gray')
        if i == 0:
            axes[i, 0].set_title('Input', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # Plot ground truth
        axes[i, 1].imshow(ground_truths[i], cmap='gray')
        if i == 0:
            axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

        # Plot predictions from each model
        for j, model_name in enumerate(model_names):
            axes[i, j + 2].imshow(predictions[model_name][i], cmap='gray')
            if i == 0:
                axes[i, j + 2].set_title(model_name, fontsize=12, fontweight='bold')
            axes[i, j + 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Segmentation results saved to {save_path}")
    plt.close()


def create_metrics_table(all_metrics, save_path='metrics_table.png'):
    """
    根据所有模型的指标，生成一个总览表格并保存为图片。

    Args:
        all_metrics: dict, 形如 {model_name: {metric_name: value}}
        save_path:   str, 保存路径，例如 'metrics_table.png'
    """

    # ==========================
    # 1. 构造 DataFrame 并整理列顺序
    # ==========================
    if not all_metrics:
        print("create_metrics_table: all_metrics 为空，跳过生成表格。")
        return

    # {model: {metric: value}} -> DataFrame
    # 行：模型名，列：各指标
    df = pd.DataFrame(all_metrics).T

    # 期望的指标顺序（全小写逻辑名）
    desired_order_lower = [
        'dice', 'iou', 'sensitivity', 'specificity',
        'precision', 'accuracy', 'boundary_f1', 'assd', 'auc'
    ]

    # DataFrame 中实际列名可能有大小写差异，比如 'Dice' / 'DICE'
    # 构造从小写 -> 实际列名的映射
    col_lower_to_actual = {col.lower(): col for col in df.columns}

    # 按期望顺序筛选出真正存在的列，并保留原始大小写
    ordered_actual_cols = [
        col_lower_to_actual[k]
        for k in desired_order_lower
        if k in col_lower_to_actual
    ]

    # 如果一个指标都没匹配上，就保持原始列顺序
    if len(ordered_actual_cols) == 0:
        ordered_actual_cols = list(df.columns)

    # 重新按顺序选择列
    df = df[ordered_actual_cols]

    # 备一份数值版本和展示版本
    # 数值部分用于着色；展示部分用于显示文字
    df_numeric = df.astype(float)
    df_display = df_numeric.round(4)

    # NaN 显示为空字符串
    cell_text = df_display.where(~df_numeric.isna(), other='').astype(str).values

    n_rows, n_cols = df_display.shape

    # ==========================
    # 2. 建立 Figure 和 Table
    # ==========================
    # 根据行列数自适应大小
    fig_width = max(8, n_cols * 1.2 + 3)
    fig_height = max(3, n_rows * 0.6 + 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # 列标题：稍微美化一下
    col_labels = [c.replace('_', ' ').title() for c in df_display.columns]

    # 注意：传入 rowLabels 和 colLabels 后，
    # Table 的结构是：
    #   行 0:  列标题行（含左上角空白 + 每个列名）
    #   行 1..n_rows: 每行：第 0 列是行标签，其余列是数据
    table = ax.table(
        cellText=cell_text,
        rowLabels=df_display.index,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # ==========================
    # 3. 给数据单元格上色
    # ==========================
    for row in range(n_rows):      # DataFrame 行索引 0..n_rows-1
        for col in range(n_cols):  # DataFrame 列索引 0..n_cols-1
            value = df_numeric.iloc[row, col]
            col_name_lower = df_numeric.columns[col].lower()

            # 这里 **不要再 +1**，数据单元格是 (row+1, col)
            cell = table[(row + 1, col)]

            # 如果是 NaN，就保持白色背景
            if np.isnan(value):
                cell.set_facecolor((1, 1, 1, 1))
                continue

            # 归一化到 [0, 1]
            if col_name_lower == 'assd':
                # ASSD 越小越好，简单假设上限 10
                norm_value = 1.0 - min(max(float(value), 0.0) / 10.0, 1.0)
            else:
                # 其他指标越大越好
                norm_value = min(max(float(value), 0.0), 1.0)

            color = plt.cm.RdYlGn(norm_value)
            cell.set_facecolor(color)

    # ==========================
    # 4. 表头和行标签样式
    # ==========================
    # 表头（第 0 行，列 0..n_cols-1）
    for col in range(n_cols):
        cell = table[(0, col)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # 行标签（第 1..n_rows 行的列 -1）
    for row in range(n_rows):
        cell = table[(row + 1, -1)]
        cell.set_facecolor('#D9E1F2')
        cell.set_text_props(weight='bold')


    # ==========================
    # 5. 保存图片
    # ==========================
    plt.title('Comprehensive Metrics Comparison',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Metrics table saved to {save_path}")
    plt.close(fig)

# ============================================================================
# Radar Chart for Multi-Metric Comparison
# ============================================================================

def plot_radar_chart(all_metrics, save_path='radar_chart.png'):
    """
    Create radar chart comparing models across multiple metrics

    Args:
        all_metrics: dict of {model_name: {metric_name: value}}
        save_path: path to save figure
    """
    from math import pi

    # Select metrics for radar chart
    metrics = ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'boundary_f1']

    # Number of variables
    num_vars = len(metrics)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot data for each model
    for model_name, metrics_dict in all_metrics.items():
        values = [metrics_dict.get(m, 0) for m in metrics]
        values += values[:1]

        color = MODEL_COLORS.get(model_name.lower(), '#333333')
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)

    # Set y-axis limits
    ax.set_ylim(0, 1)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

    plt.title('Multi-Metric Performance Comparison', size=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Radar chart saved to {save_path}")
    plt.close()


# ============================================================================
# Statistical Significance Visualization
# ============================================================================

def plot_statistical_comparison(pairwise_results, save_path='statistical_comparison.png'):
    """
    Visualize statistical significance of pairwise model comparisons

    Args:
        pairwise_results: dict of {(model1, model2): {'metric': p_value}}
        save_path: path to save figure
    """
    # Extract model pairs and create matrix
    models = sorted(set([m for pair in pairwise_results.keys() for m in pair]))
    metrics = ['dice', 'iou', 'sensitivity', 'boundary_f1']

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for idx, metric in enumerate(metrics):
        # Create p-value matrix
        n_models = len(models)
        p_matrix = np.ones((n_models, n_models))

        for (m1, m2), results in pairwise_results.items():
            if metric in results:
                i, j = models.index(m1), models.index(m2)
                p_matrix[i, j] = results[metric]
                p_matrix[j, i] = results[metric]

        # Plot heatmap
        ax = axes[idx] if len(metrics) > 1 else axes
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)

        # Set ticks
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(models)

        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=9)

        ax.set_title(f'{metric.title()} - P-values', fontsize=12, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Statistical comparison saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Testing visualization functions...")
    print("="*60)

    # Create dummy data for testing
    model_names = ['unet', 'unet_eam', 'transunet', 'transunet_eam']

    # Test training curves
    histories = {}
    for model in model_names:
        histories[model] = {
            'train_loss': np.random.rand(20) * 0.5,
            'val_loss': np.random.rand(20) * 0.4,
            'val_acc': 90 + np.random.rand(20) * 5
        }
    plot_training_curves(histories, 'test_training_curves.png')

    # Test metrics comparison
    all_metrics = {}
    for model in model_names:
        all_metrics[model] = {
            'dice': np.random.rand(),
            'iou': np.random.rand(),
            'sensitivity': np.random.rand(),
            'specificity': np.random.rand(),
            'precision': np.random.rand(),
            'boundary_f1': np.random.rand(),
            'auc': np.random.rand(),
            'assd': np.random.rand() * 2
        }
    plot_metrics_comparison(all_metrics, 'test_metrics_comparison.png')

    # Test ROC curves
    roc_data = {}
    for model in model_names:
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr) + np.random.rand(100) * 0.1
        tpr = np.clip(tpr, 0, 1)
        auc_score = np.trapz(tpr, fpr)
        roc_data[model] = (fpr, tpr, auc_score)
    plot_roc_curves(roc_data, 'test_roc_curves.png')

    print("\n✓ All visualization tests passed!")
