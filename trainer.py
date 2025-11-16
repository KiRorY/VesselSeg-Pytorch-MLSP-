# -*- coding: utf-8 -*-
"""
Training and Evaluation Framework
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import cv2
from PIL import Image

from models import get_model
from losses import get_loss_with_deep_supervision
from metrics import MetricsCalculator


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_raw_drive_data(drive_root_path):
    """
    Load and preprocess raw DRIVE dataset

    Returns:
        images: list of preprocessed images
        masks: list of binary masks
        fovs: list of field-of-view masks
    """
    images, masks, fovs = [], [], []

    train_path = os.path.join(drive_root_path, 'training')

    # Load file paths
    img_paths = sorted(glob.glob(os.path.join(train_path, 'images', '*.tif')))
    mask_paths = sorted(glob.glob(os.path.join(train_path, '1st_manual', '*.gif')))
    fov_paths = sorted(glob.glob(os.path.join(train_path, 'mask', '*_mask.gif')))

    if not (img_paths and mask_paths and fov_paths):
        print(f"Error: Cannot find raw DRIVE files in {train_path}")
        print(f"Please check path: {drive_root_path}")
        print("Ensure it contains 'training/images', 'training/1st_manual', 'training/mask' subfolders.")
        return None, None, None

    print(f"Found {len(img_paths)} training images. Starting preprocessing...")

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for img_p, mask_p, fov_p in tqdm(zip(img_paths, mask_paths, fov_paths), desc="Processing images"):
        # 1. Load RGB image
        img_raw = np.array(Image.open(img_p))  # (H, W, 3)

        # 2. Extract green channel
        img_green = img_raw[:, :, 1]  # (H, W)

        # 3. Apply CLAHE
        img_clahe = clahe.apply(img_green)

        # 4. Z-score normalization
        img_norm = (img_clahe - np.mean(img_clahe)) / (np.std(img_clahe) + 1e-6)

        # 5. Load mask
        mask_raw = np.array(Image.open(mask_p))
        mask_binary = (mask_raw > 128).astype(np.uint8)

        # 6. Load FOV
        fov_raw = np.array(Image.open(fov_p))
        fov_binary = (fov_raw > 128).astype(np.uint8)

        images.append(img_norm)
        masks.append(mask_binary)
        fovs.append(fov_binary)

    print("All images processed and loaded.")
    return images, masks, fovs


def generate_patches(images, masks, fovs, patch_size, n_patches, seed=2021):
    """
    Generate random patches from images

    Args:
        images: list of images
        masks: list of masks
        fovs: list of field-of-view masks
        patch_size: (height, width)
        n_patches: number of patches to generate
        seed: random seed

    Returns:
        patches_data: list of (img_patch, mask_patch) tuples
    """
    patch_h, patch_w = patch_size
    h_border = patch_h // 2
    w_border = patch_w // 2

    # 1. Find all valid center points
    valid_centers = []  # Format: (image_index, y, x)
    for i, (img, fov) in enumerate(zip(images, fovs)):
        h, w = img.shape

        # Find valid pixels in FOV
        coords_y, coords_x = np.where(fov[h_border:h-h_border, w_border:w-w_border] > 0)

        # Adjust coordinates
        coords_y += h_border
        coords_x += w_border

        # Store all valid centers
        for y, x in zip(coords_y, coords_x):
            valid_centers.append((i, y, x))

    print(f"Found {len(valid_centers)} valid patch centers.")
    if not valid_centers:
        raise ValueError("No valid patch centers found. Check FOV masks and patch size.")

    # 2. Randomly sample patches
    patches_data = []
    rng = np.random.default_rng(seed)

    for _ in tqdm(range(n_patches), desc="Generating patches"):
        # Random sample a center
        img_idx, y, x = valid_centers[rng.integers(0, len(valid_centers))]

        # Extract patch
        y_start, x_start = y - h_border, x - w_border
        y_end, x_end = y_start + patch_h, x_start + patch_w

        img_patch = images[img_idx][y_start:y_end, x_start:x_end]
        mask_patch = masks[img_idx][y_start:y_end, x_start:x_end]

        patches_data.append((img_patch, mask_patch))

    return patches_data


class VesselPatchDataset(Dataset):
    """PyTorch Dataset for vessel segmentation patches"""
    def __init__(self, patches_data, augment=False):
        self.patches_data = patches_data
        self.augment = augment

    def __len__(self):
        return len(self.patches_data)

    def __getitem__(self, idx):
        img_patch, mask_patch = self.patches_data[idx]

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:  # Horizontal flip
                img_patch = np.fliplr(img_patch)
                mask_patch = np.fliplr(mask_patch)
            if np.random.rand() > 0.5:  # Vertical flip
                img_patch = np.flipud(img_patch)
                mask_patch = np.flipud(mask_patch)

        # Convert to tensors
        # (H, W) -> (C, H, W)
        img_tensor = torch.from_numpy(img_patch.copy()).unsqueeze(0).float()
        # (H, W)
        mask_tensor = torch.from_numpy(mask_patch.copy()).long()

        return img_tensor, mask_tensor


# ============================================================================
# Training and Validation
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device, calc_metrics=False):
    """
    Validate for one epoch

    Args:
        calc_metrics: if True, calculate comprehensive metrics
    """
    model.eval()
    epoch_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    metrics_calc = MetricsCalculator() if calc_metrics else None

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Handle deep supervision output
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs

            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Get predictions
            probs = torch.softmax(main_output, dim=1)
            preds = torch.argmax(main_output, dim=1)  # (N, H, W)

            # Accuracy
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()

            # Comprehensive metrics
            if calc_metrics:
                # Get vessel class probabilities and predictions
                vessel_probs = probs[:, 1, :, :].cpu().numpy()  # Probability of vessel class
                vessel_preds = preds.cpu().numpy()
                targets = labels.cpu().numpy()

                # Update metrics for each sample in batch
                for i in range(len(vessel_preds)):
                    metrics_calc.update(vessel_preds[i], targets[i], vessel_probs[i])

    avg_loss = epoch_loss / len(loader)
    accuracy = (correct_pixels / total_pixels) * 100.0

    if calc_metrics:
        return avg_loss, accuracy, metrics_calc.get_metrics()
    else:
        return avg_loss, accuracy


# ============================================================================
# Main Training Function
# ============================================================================

def train_model(model_name, config, train_loader, val_loader, device, save_dir='checkpoints'):
    """
    Train a model with given configuration

    Args:
        model_name: name of the model
        config: configuration dictionary
        train_loader: training data loader
        val_loader: validation data loader
        device: torch device
        save_dir: directory to save checkpoints

    Returns:
        model: trained model
        history: training history
        best_metrics: best validation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f'{model_name}_best.pth')

    # Create model
    model = get_model(
        model_name,
        in_channels=config['in_channels'],
        n_classes=config['classes'],
        img_size=config['patch_height'],
        deep_supervision=config.get('deep_supervision', False)
    ).to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    # Loss and optimizer
    if config.get('deep_supervision', False):
        criterion = get_loss_with_deep_supervision(config['loss'])
    else:
        from losses import get_loss
        criterion = get_loss(config['loss'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=0
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    best_metrics = None
    best_epoch = 0
    trigger_times = 0

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    if config['early_stop'] == float('inf'):
        print(f"Early stopping: DISABLED")
    else:
        print(f"Early stopping patience: {config['early_stop']} epochs")
    print(f"{'='*60}\n")
    for epoch in range(config['epochs']):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate (with comprehensive metrics every 5 epochs)
        calc_metrics = (epoch % 5 == 0 or epoch == config['epochs'] - 1)
        if calc_metrics:
            val_loss, val_acc, val_metrics = validate_one_epoch(
                model, val_loader, criterion, device, calc_metrics=True
            )
        else:
            val_loss, val_acc = validate_one_epoch(
                model, val_loader, criterion, device, calc_metrics=False
            )
            val_metrics = None

        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed_time = time.time() - start_time

        # Print progress
        print(f"Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Time: {elapsed_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_metrics:
            print(f"  Dice: {val_metrics['dice']:.4f} | "
                  f"IoU: {val_metrics['iou']:.4f} | "
                  f"Sensitivity: {val_metrics['sensitivity']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_epoch = epoch + 1
            trigger_times = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, model_save_path)
            print(f"  ✓ Best model saved (Val Loss: {best_val_loss:.4f}) - Epoch {best_epoch}")
        else:
            trigger_times += 1
            print(f"  → No improvement for {trigger_times} epoch(s) (Best: {best_val_loss:.4f} at epoch {best_epoch})")

            if trigger_times >= config['early_stop']:
                print(f"\n{'='*60}")
                print(f"⚠ EARLY STOPPING TRIGGERED")
                print(f"{'='*60}")
                print(f"  No improvement for {trigger_times} consecutive epochs")
                print(f"  Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
                print(f"  Training stopped at epoch {epoch+1}/{config['epochs']}")
                print(f"{'='*60}\n")
                break

    print(f"\n{'='*60}")
    print(f"Training completed for {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    if best_metrics:
        print(f"Best Dice: {best_metrics['dice']:.4f}")
        print(f"Best IoU: {best_metrics['iou']:.4f}")
    print(f"Total epochs trained: {epoch + 1}/{config['epochs']}")
    if trigger_times >= config['early_stop'] and config['early_stop'] != float('inf'):
        print(f"Status: Stopped early (patience exhausted)")
    else:
        print(f"Status: Completed all epochs")
    print(f"Model checkpoint: {model_save_path}")
    print(f"{'='*60}\n")

    # Load best model
    checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history, best_metrics


# ============================================================================
# Comprehensive Evaluation
# ============================================================================

def evaluate_model(model, loader, device):
    """
    Comprehensive evaluation of a model

    Returns:
        metrics: dictionary of metrics
        metrics_calc: MetricsCalculator object with detailed results
    """
    model.eval()
    metrics_calc = MetricsCalculator()

    print("Running comprehensive evaluation...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Handle deep supervision
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Get vessel class probabilities
            vessel_probs = probs[:, 1, :, :].cpu().numpy()
            vessel_preds = preds.cpu().numpy()
            targets = labels.cpu().numpy()

            # Update metrics
            for i in range(len(vessel_preds)):
                metrics_calc.update(vessel_preds[i], targets[i], vessel_probs[i])

    metrics = metrics_calc.get_metrics()
    return metrics, metrics_calc


if __name__ == "__main__":
    # Test training framework
    print("Testing training framework...")
    print("="*60)

    # This is just a structure test
    # Actual training should be done through main experiment script
    print("✓ Training framework loaded successfully!")
