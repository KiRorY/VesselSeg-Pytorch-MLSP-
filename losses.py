# -*- coding: utf-8 -*-
"""
Loss Functions for Medical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Dice Loss
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary and multi-class segmentation
    """
    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw logits from model
            targets: (B, H, W) - ground truth labels
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Compute Dice for each class
        dims = (0, 2, 3)  # Batch, Height, Width
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = (probs + targets_one_hot).sum(dim=dims)

        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score

        # Average over classes (excluding background if needed)
        if self.ignore_index is not None:
            mask = torch.ones(num_classes, device=dice_loss.device)
            mask[self.ignore_index] = 0
            dice_loss = (dice_loss * mask).sum() / mask.sum()
        else:
            dice_loss = dice_loss.mean()

        return dice_loss


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss to address class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)

        # Get probabilities
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        pt = (probs * targets_one_hot).sum(dim=1)

        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha

        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


# ============================================================================
# Combined Losses
# ============================================================================

class BCE_DiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class Focal_DiceLoss(nn.Module):
    """Combined Focal Loss and Dice Loss"""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


# ============================================================================
# Deep Supervision Loss
# ============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    Loss function with deep supervision
    Combines losses from multiple decoder stages
    """
    def __init__(self, base_loss, ds_weights=None):
        super().__init__()
        self.base_loss = base_loss
        self.ds_weights = ds_weights or [0.5, 0.3, 0.2]  # Weights for deep supervision outputs

    def forward(self, outputs, targets):
        """
        Args:
            outputs: tuple of (main_output, [ds_output1, ds_output2, ...])
            targets: ground truth labels
        """
        if isinstance(outputs, tuple):
            main_output, ds_outputs = outputs

            # Main loss
            loss = self.base_loss(main_output, targets)

            # Deep supervision losses
            for ds_out, weight in zip(ds_outputs, self.ds_weights):
                # Resize target to match deep supervision output size
                if ds_out.shape[-2:] != targets.shape[-2:]:
                    targets_resized = F.interpolate(
                        targets.unsqueeze(1).float(),
                        size=ds_out.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    targets_resized = targets

                ds_loss = self.base_loss(ds_out, targets_resized)
                loss = loss + weight * ds_loss

            return loss
        else:
            # No deep supervision, just compute main loss
            return self.base_loss(outputs, targets)


# ============================================================================
# Boundary Loss (Advanced)
# ============================================================================

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for better edge segmentation
    Uses distance transform of boundaries
    """
    def __init__(self, theta=5):
        super().__init__()
        self.theta = theta

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        probs = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape

        # Compute distance transforms for boundaries
        boundary_weights = torch.zeros_like(probs)

        for b in range(B):
            target_np = targets[b].cpu().numpy()

            for c in range(C):
                # Get binary mask for this class
                mask = (target_np == c).astype(np.uint8)

                # Compute distance transform
                if mask.sum() > 0:
                    dist = distance_transform_edt(1 - mask) + distance_transform_edt(mask)
                    # Convert to boundary weights
                    weights = np.exp(-dist / self.theta)
                    boundary_weights[b, c] = torch.from_numpy(weights).to(probs.device)

        # Weighted cross entropy
        targets_one_hot = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        loss = -(boundary_weights * targets_one_hot * torch.log(probs + 1e-6)).sum(dim=1).mean()

        return loss


# ============================================================================
# Loss Factory
# ============================================================================

def get_loss(loss_name, **kwargs):
    """Factory function to get loss by name"""
    losses = {
        'ce': nn.CrossEntropyLoss,
        'dice': DiceLoss,
        'focal': FocalLoss,
        'bce_dice': BCE_DiceLoss,
        'focal_dice': Focal_DiceLoss,
        'boundary': BoundaryLoss
    }

    if loss_name.lower() not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from {list(losses.keys())}")

    return losses[loss_name.lower()](**kwargs)


def get_loss_with_deep_supervision(loss_name, **kwargs):
    """Get loss with deep supervision wrapper"""
    base_loss = get_loss(loss_name, **kwargs)
    return DeepSupervisionLoss(base_loss)


if __name__ == "__main__":
    # Test losses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data
    batch_size = 4
    num_classes = 2
    H, W = 64, 64

    logits = torch.randn(batch_size, num_classes, H, W).to(device)
    targets = torch.randint(0, num_classes, (batch_size, H, W)).to(device)

    print("Testing loss functions...")
    print("="*60)

    # Test individual losses
    loss_names = ['ce', 'dice', 'focal', 'bce_dice', 'focal_dice']

    for loss_name in loss_names:
        loss_fn = get_loss(loss_name)
        loss_value = loss_fn(logits, targets)
        print(f"{loss_name.upper():15s}: {loss_value.item():.4f}")

    # Test deep supervision
    print("\n" + "="*60)
    print("Testing Deep Supervision Loss...")

    ds_outputs = [
        torch.randn(batch_size, num_classes, H//2, W//2).to(device),
        torch.randn(batch_size, num_classes, H//4, W//4).to(device)
    ]
    outputs_with_ds = (logits, ds_outputs)

    ds_loss = get_loss_with_deep_supervision('bce_dice')
    loss_value = ds_loss(outputs_with_ds, targets)
    print(f"DS BCE+Dice Loss: {loss_value.item():.4f}")

    print("\n✓ All loss tests passed!")
