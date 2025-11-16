# -*- coding: utf-8 -*-
"""
Evaluation Metrics for Medical Image Segmentation
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import cv2


# ============================================================================
# Basic Metrics
# ============================================================================

def compute_dice(pred, target, smooth=1.0):
    """
    Compute Dice Score (F1 Score for segmentation)

    Args:
        pred: (H, W) binary prediction
        target: (H, W) binary ground truth
        smooth: smoothing factor to avoid division by zero

    Returns:
        dice: float
    """
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice


def compute_iou(pred, target, smooth=1e-6):
    """
    Compute Intersection over Union (IoU / Jaccard Index)

    Args:
        pred: (H, W) binary prediction
        target: (H, W) binary ground truth

    Returns:
        iou: float
    """
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou


def compute_sensitivity(pred, target, smooth=1e-6):
    """
    Compute Sensitivity (Recall / True Positive Rate)

    Args:
        pred: (H, W) binary prediction
        target: (H, W) binary ground truth

    Returns:
        sensitivity: float
    """
    pred = pred.flatten()
    target = target.flatten()

    true_positive = (pred * target).sum()
    actual_positive = target.sum()

    sensitivity = (true_positive + smooth) / (actual_positive + smooth)

    return sensitivity


def compute_specificity(pred, target, smooth=1e-6):
    """
    Compute Specificity (True Negative Rate)

    Args:
        pred: (H, W) binary prediction
        target: (H, W) binary ground truth

    Returns:
        specificity: float
    """
    pred = pred.flatten()
    target = target.flatten()

    true_negative = ((1 - pred) * (1 - target)).sum()
    actual_negative = (1 - target).sum()

    specificity = (true_negative + smooth) / (actual_negative + smooth)

    return specificity


def compute_precision(pred, target, smooth=1e-6):
    """
    Compute Precision (Positive Predictive Value)

    Args:
        pred: (H, W) binary prediction
        target: (H, W) binary ground truth

    Returns:
        precision: float
    """
    pred = pred.flatten()
    target = target.flatten()

    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()

    precision = (true_positive + smooth) / (predicted_positive + smooth)

    return precision


def compute_accuracy(pred, target):
    """
    Compute Pixel-wise Accuracy

    Args:
        pred: (H, W) binary prediction
        target: (H, W) binary ground truth

    Returns:
        accuracy: float
    """
    pred = pred.flatten()
    target = target.flatten()

    correct = (pred == target).sum()
    total = len(target)

    accuracy = correct / total

    return accuracy


# ============================================================================
# Boundary-based Metrics
# ============================================================================

def compute_boundary_f1(pred, target, threshold=2.0):
    """
    Compute Boundary F1 Score

    Measures how well predicted boundaries match ground truth boundaries.

    Args:
        pred: (H, W) binary prediction
        target: (H, W) binary ground truth
        threshold: distance threshold in pixels

    Returns:
        boundary_f1: float
    """
    # Extract boundaries
    pred_boundary = extract_boundary(pred)
    target_boundary = extract_boundary(target)

    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return 0.0

    # Compute distances
    pred_coords = np.argwhere(pred_boundary > 0)
    target_coords = np.argwhere(target_boundary > 0)

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0

    # Precision: fraction of predicted boundary pixels close to ground truth
    dist_pred_to_target = distance_transform_edt(1 - target_boundary)
    precision_mask = dist_pred_to_target[pred_boundary > 0] <= threshold
    precision = precision_mask.sum() / len(precision_mask) if len(precision_mask) > 0 else 0.0

    # Recall: fraction of ground truth boundary pixels close to prediction
    dist_target_to_pred = distance_transform_edt(1 - pred_boundary)
    recall_mask = dist_target_to_pred[target_boundary > 0] <= threshold
    recall = recall_mask.sum() / len(recall_mask) if len(recall_mask) > 0 else 0.0

    # F1 Score
    if precision + recall == 0:
        return 0.0

    boundary_f1 = 2 * (precision * recall) / (precision + recall)

    return boundary_f1


def compute_assd(pred, target):
    """
    Compute Average Symmetric Surface Distance (ASSD)

    Measures average distance between boundaries.

    Args:
        pred: (H, W) binary prediction
        target: (H, W) binary ground truth

    Returns:
        assd: float (lower is better)
    """
    # Extract boundaries
    pred_boundary = extract_boundary(pred)
    target_boundary = extract_boundary(target)

    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        # If either boundary is empty, return large distance
        return 100.0

    # Compute distance transforms
    dist_pred_to_target = distance_transform_edt(1 - target_boundary)
    dist_target_to_pred = distance_transform_edt(1 - pred_boundary)

    # Get distances at boundary pixels
    distances_pred = dist_pred_to_target[pred_boundary > 0]
    distances_target = dist_target_to_pred[target_boundary > 0]

    # Average symmetric surface distance
    if len(distances_pred) == 0 and len(distances_target) == 0:
        return 0.0

    assd = (distances_pred.sum() + distances_target.sum()) / (len(distances_pred) + len(distances_target))

    return assd


def extract_boundary(mask):
    """
    Extract boundary from binary mask using morphological operations

    Args:
        mask: (H, W) binary mask

    Returns:
        boundary: (H, W) binary boundary mask
    """
    mask_np = mask.astype(np.uint8)

    # Dilate and subtract original to get boundary
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    boundary = dilated - mask_np

    return boundary


# ============================================================================
# ROC and AUC
# ============================================================================

def compute_roc_auc(pred_probs, target):
    """
    Compute ROC curve and AUC

    Args:
        pred_probs: (H, W) probability map [0, 1]
        target: (H, W) binary ground truth

    Returns:
        fpr: false positive rates
        tpr: true positive rates
        auc_score: area under curve
    """
    pred_probs = pred_probs.flatten()
    target = target.flatten()

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(target, pred_probs)

    # Compute AUC
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score


# ============================================================================
# Evaluation on Dataset
# ============================================================================

class MetricsCalculator:
    """
    Calculate comprehensive metrics for segmentation evaluation
    """
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'dice': [],
            'iou': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'accuracy': [],
            'boundary_f1': [],
            'assd': []
        }
        self.all_probs = []
        self.all_targets = []

    def update(self, pred, target, pred_probs=None):
        """
        Update metrics with a new prediction

        Args:
            pred: (H, W) or (B, H, W) binary prediction
            target: (H, W) or (B, H, W) binary ground truth
            pred_probs: (H, W) or (B, H, W) probability map (optional, for ROC)
        """
        # Handle batch dimension
        if len(pred.shape) == 3:
            for i in range(pred.shape[0]):
                self.update(pred[i], target[i],
                           pred_probs[i] if pred_probs is not None else None)
            return

        # Convert to numpy if needed
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        if pred_probs is not None and torch.is_tensor(pred_probs):
            pred_probs = pred_probs.cpu().numpy()

        # Ensure binary
        pred = (pred > 0.5).astype(np.uint8)
        target = target.astype(np.uint8)

        # Compute metrics
        self.metrics['dice'].append(compute_dice(pred, target))
        self.metrics['iou'].append(compute_iou(pred, target))
        self.metrics['sensitivity'].append(compute_sensitivity(pred, target))
        self.metrics['specificity'].append(compute_specificity(pred, target))
        self.metrics['precision'].append(compute_precision(pred, target))
        self.metrics['accuracy'].append(compute_accuracy(pred, target))
        self.metrics['boundary_f1'].append(compute_boundary_f1(pred, target))
        self.metrics['assd'].append(compute_assd(pred, target))

        # Store for ROC calculation
        if pred_probs is not None:
            self.all_probs.append(pred_probs.flatten())
            self.all_targets.append(target.flatten())

    def get_metrics(self):
        """
        Get average metrics

        Returns:
            dict: dictionary of metric names to average values
        """
        results = {}
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                results[metric_name] = np.mean(values)
                results[f'{metric_name}_std'] = np.std(values)
            else:
                results[metric_name] = 0.0
                results[f'{metric_name}_std'] = 0.0

        # Compute ROC AUC if probabilities were provided
        if len(self.all_probs) > 0:
            all_probs = np.concatenate(self.all_probs)
            all_targets = np.concatenate(self.all_targets)
            try:
                results['auc'] = roc_auc_score(all_targets, all_probs)
            except:
                results['auc'] = 0.0
        else:
            results['auc'] = 0.0

        return results

    def get_roc_curve(self):
        """
        Get ROC curve data

        Returns:
            fpr, tpr, auc_score
        """
        if len(self.all_probs) == 0:
            return None, None, 0.0

        all_probs = np.concatenate(self.all_probs)
        all_targets = np.concatenate(self.all_targets)

        return compute_roc_auc(all_probs, all_targets)


# ============================================================================
# Statistical Significance Testing
# ============================================================================

def paired_ttest(metrics1, metrics2, metric_name='dice'):
    """
    Perform paired t-test to compare two models

    Args:
        metrics1: list of metric values from model 1
        metrics2: list of metric values from model 2
        metric_name: name of the metric being compared

    Returns:
        t_statistic, p_value
    """
    from scipy import stats

    # Ensure same length
    assert len(metrics1) == len(metrics2), "Metrics must have same length"

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(metrics1, metrics2)

    return t_statistic, p_value


def wilcoxon_test(metrics1, metrics2):
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to t-test)

    Args:
        metrics1: list of metric values from model 1
        metrics2: list of metric values from model 2

    Returns:
        statistic, p_value
    """
    from scipy import stats

    # Ensure same length
    assert len(metrics1) == len(metrics2), "Metrics must have same length"

    # Perform Wilcoxon test
    statistic, p_value = stats.wilcoxon(metrics1, metrics2)

    return statistic, p_value


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    print("="*60)

    # Create dummy data
    pred = np.random.rand(64, 64) > 0.5
    target = np.random.rand(64, 64) > 0.5
    pred_probs = np.random.rand(64, 64)

    # Test individual metrics
    print(f"Dice Score:     {compute_dice(pred, target):.4f}")
    print(f"IoU:            {compute_iou(pred, target):.4f}")
    print(f"Sensitivity:    {compute_sensitivity(pred, target):.4f}")
    print(f"Specificity:    {compute_specificity(pred, target):.4f}")
    print(f"Precision:      {compute_precision(pred, target):.4f}")
    print(f"Accuracy:       {compute_accuracy(pred, target):.4f}")
    print(f"Boundary F1:    {compute_boundary_f1(pred, target):.4f}")
    print(f"ASSD:           {compute_assd(pred, target):.4f}")

    # Test MetricsCalculator
    print("\n" + "="*60)
    print("Testing MetricsCalculator...")

    calc = MetricsCalculator()
    for _ in range(5):
        pred = np.random.rand(64, 64) > 0.5
        target = np.random.rand(64, 64) > 0.5
        pred_probs = np.random.rand(64, 64)
        calc.update(pred, target, pred_probs)

    results = calc.get_metrics()
    print("\nAverage Metrics:")
    for metric_name, value in results.items():
        if not metric_name.endswith('_std'):
            std = results.get(f'{metric_name}_std', 0)
            print(f"{metric_name:15s}: {value:.4f} ± {std:.4f}")

    print("\n✓ All metrics tests passed!")
