# -*- coding: utf-8 -*-
"""
Model Definitions: U-Net, TransUnet, U-Net+EAM, TransUnet+EAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================================
# Basic Building Blocks
# ============================================================================

class ConvBlock(nn.Module):
    """Standard (Conv2D -> BatchNorm -> ReLU) * 2 block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


# ============================================================================
# Edge-Aware Module (EAM)
# ============================================================================

class GaborConv2d(nn.Module):
    """
    Gabor-inspired Directional Convolution
    Generates filters oriented at different angles to detect edges
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, num_orientations=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations

        # Learnable weights for combining oriented filters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # Learnable orientation weights
        self.orientation_weights = nn.Parameter(torch.ones(num_orientations) / num_orientations)

    def forward(self, x):
        # Generate Gabor kernels for different orientations
        gabor_kernels = []
        for i in range(self.num_orientations):
            theta = (i / self.num_orientations) * math.pi
            kernel = self._gabor_kernel(theta, sigma=2.0, lambd=4.0)
            gabor_kernels.append(kernel)

        # Stack and normalize orientation weights
        gabor_stack = torch.stack(gabor_kernels, dim=0)  # (num_orientations, ks, ks)
        weights_normalized = F.softmax(self.orientation_weights, dim=0)

        # Weighted combination of oriented filters
        # Shape: (ks, ks)
        combined_gabor = (gabor_stack * weights_normalized.view(-1, 1, 1)).sum(dim=0)

        # Apply Gabor pattern to learnable weights
        # Broadcast multiply: (out_channels, in_channels, ks, ks) * (ks, ks)
        filters = self.weight * combined_gabor.unsqueeze(0).unsqueeze(0)

        # Apply convolution
        return F.conv2d(x, filters, padding=self.kernel_size//2)

    def _gabor_kernel(self, theta, sigma, lambd):
        """Generate a single Gabor kernel"""
        kernel_size = self.kernel_size
        sigma = sigma.item() if isinstance(sigma, torch.Tensor) else sigma
        lambd = lambd.item() if isinstance(lambd, torch.Tensor) else lambd

        # Create coordinate grid (compatible with all PyTorch versions)
        coord = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)

        # For PyTorch < 1.10, meshgrid doesn't support indexing parameter
        try:
            y, x = torch.meshgrid(coord, coord, indexing='ij')
        except TypeError:
            # Fallback for older PyTorch versions
            x, y = torch.meshgrid(coord, coord)

        # Move to same device as parameters
        y = y.to(self.weight.device)
        x = x.to(self.weight.device)

        # Rotate coordinates
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)

        # Gabor function
        gb = torch.exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2) * \
             torch.cos(2 * math.pi * x_theta / lambd)

        return gb / (gb.abs().sum() + 1e-6)


class ChannelAttention(nn.Module):
    """Channel-wise attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class EdgeAwareModule(nn.Module):
    """
    Edge-Aware Module with Gabor convolutions and channel attention
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Standard convolution path
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Gabor convolution path for edge detection
        self.gabor_path = nn.Sequential(
            GaborConv2d(in_channels, out_channels, kernel_size=5, num_orientations=8),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Channel attention
        self.channel_attention = ChannelAttention(out_channels * 2)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv_feat = self.conv_path(x)
        gabor_feat = self.gabor_path(x)

        # Concatenate features
        combined = torch.cat([conv_feat, gabor_feat], dim=1)

        # Apply channel attention
        attended = self.channel_attention(combined)

        # Fuse features
        output = self.fusion(attended)

        return output


# ============================================================================
# U-Net
# ============================================================================

class UNet(nn.Module):
    """Standard U-Net Architecture"""
    def __init__(self, in_channels, n_classes, deep_supervision=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Final output
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        # Deep supervision outputs (optional)
        if deep_supervision:
            self.ds_out4 = nn.Conv2d(512, n_classes, kernel_size=1)
            self.ds_out3 = nn.Conv2d(256, n_classes, kernel_size=1)
            self.ds_out2 = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        # Final output
        logits = self.out_conv(d1)

        if self.deep_supervision and self.training:
            # Return deep supervision outputs during training
            ds4 = self.ds_out4(d4)
            ds3 = self.ds_out3(d3)
            ds2 = self.ds_out2(d2)
            return logits, [ds4, ds3, ds2]

        return logits


# ============================================================================
# U-Net + EAM
# ============================================================================

class UNet_EAM(nn.Module):
    """U-Net with Edge-Aware Module"""
    def __init__(self, in_channels, n_classes, deep_supervision=True):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        # Encoder (same as U-Net)
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder with EAM
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.eam4 = EdgeAwareModule(512, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.eam3 = EdgeAwareModule(256, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.eam2 = EdgeAwareModule(128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.eam1 = EdgeAwareModule(64, 64)

        # Final output
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        # Deep supervision outputs
        if deep_supervision:
            self.ds_out4 = nn.Conv2d(512, n_classes, kernel_size=1)
            self.ds_out3 = nn.Conv2d(256, n_classes, kernel_size=1)
            self.ds_out2 = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with EAM
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        d4 = self.eam4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        d3 = self.eam3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        d2 = self.eam2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        d1 = self.eam1(d1)

        # Final output
        logits = self.out_conv(d1)

        if self.deep_supervision and self.training:
            ds4 = self.ds_out4(d4)
            ds3 = self.ds_out3(d3)
            ds2 = self.ds_out2(d2)
            return logits, [ds4, ds3, ds2]

        return logits


# ============================================================================
# Transformer Components
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with MHA and FFN"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# TransUnet
# ============================================================================

class TransUnet(nn.Module):
    """TransUnet: Transformer + U-Net hybrid architecture"""
    def __init__(self, in_channels, n_classes, img_size=64, patch_size=1,
                 embed_dim=512, depth=6, num_heads=8, deep_supervision=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.deep_supervision = deep_supervision

        # CNN Encoder (first 3 stages)
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Patch embedding for transformer
        # At this stage, input is (B, 256, H/8, W/8)
        self.patch_embed = nn.Conv2d(256, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        num_patches = (img_size // 8 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Reshape back to 2D
        self.reshape_conv = nn.Conv2d(embed_dim, 512, kernel_size=1)

        # CNN Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)  # 256 from enc3 + 256 from upconv3

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Final output
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        # Deep supervision
        if deep_supervision:
            self.ds_out3 = nn.Conv2d(256, n_classes, kernel_size=1)
            self.ds_out2 = nn.Conv2d(128, n_classes, kernel_size=1)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # CNN Encoder
        e1 = self.enc1(x)  # (B, 64, H, W)
        e2 = self.enc2(self.pool1(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))  # (B, 256, H/4, W/4)

        # Prepare for transformer
        x_pool = self.pool3(e3)  # (B, 256, H/8, W/8)

        # Patch embedding
        B, C, H, W = x_pool.shape
        x_patch = self.patch_embed(x_pool)  # (B, embed_dim, H', W')
        x_flat = x_patch.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Add positional embedding
        x_flat = x_flat + self.pos_embed
        x_flat = self.pos_drop(x_flat)

        # Transformer blocks
        for block in self.transformer_blocks:
            x_flat = block(x_flat)
        x_flat = self.norm(x_flat)

        # Reshape back to 2D
        H_t, W_t = H // self.patch_size, W // self.patch_size
        x_2d = x_flat.transpose(1, 2).reshape(B, -1, H_t, W_t)
        x_2d = self.reshape_conv(x_2d)  # (B, 512, H/8, W/8)

        # CNN Decoder with skip connections
        d3 = self.upconv3(x_2d)  # (B, 256, H/4, W/4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)  # (B, 128, H/2, W/2)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)  # (B, 64, H, W)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        # Final output
        logits = self.out_conv(d1)

        if self.deep_supervision and self.training:
            ds3 = self.ds_out3(d3)
            ds2 = self.ds_out2(d2)
            return logits, [ds3, ds2]

        return logits


# ============================================================================
# TransUnet + EAM
# ============================================================================

class TransUnet_EAM(nn.Module):
    """TransUnet with Edge-Aware Module"""
    def __init__(self, in_channels, n_classes, img_size=64, patch_size=1,
                 embed_dim=512, depth=6, num_heads=8, deep_supervision=True):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.deep_supervision = deep_supervision

        # CNN Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Patch embedding
        self.patch_embed = nn.Conv2d(256, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        num_patches = (img_size // 8 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Reshape back
        self.reshape_conv = nn.Conv2d(embed_dim, 512, kernel_size=1)

        # CNN Decoder with EAM
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.eam3 = EdgeAwareModule(256, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.eam2 = EdgeAwareModule(128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.eam1 = EdgeAwareModule(64, 64)

        # Final output
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        # Deep supervision
        if deep_supervision:
            self.ds_out3 = nn.Conv2d(256, n_classes, kernel_size=1)
            self.ds_out2 = nn.Conv2d(128, n_classes, kernel_size=1)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # CNN Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Prepare for transformer
        x_pool = self.pool3(e3)

        # Patch embedding
        B, C, H, W = x_pool.shape
        x_patch = self.patch_embed(x_pool)
        x_flat = x_patch.flatten(2).transpose(1, 2)

        # Add positional embedding
        x_flat = x_flat + self.pos_embed
        x_flat = self.pos_drop(x_flat)

        # Transformer blocks
        for block in self.transformer_blocks:
            x_flat = block(x_flat)
        x_flat = self.norm(x_flat)

        # Reshape back to 2D
        H_t, W_t = H // self.patch_size, W // self.patch_size
        x_2d = x_flat.transpose(1, 2).reshape(B, -1, H_t, W_t)
        x_2d = self.reshape_conv(x_2d)

        # CNN Decoder with EAM
        d3 = self.upconv3(x_2d)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        d3 = self.eam3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        d2 = self.eam2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        d1 = self.eam1(d1)

        # Final output
        logits = self.out_conv(d1)

        if self.deep_supervision and self.training:
            ds3 = self.ds_out3(d3)
            ds2 = self.ds_out2(d2)
            return logits, [ds3, ds2]

        return logits


# ============================================================================
# Model Factory
# ============================================================================

def get_model(model_name, in_channels=1, n_classes=2, img_size=64, **kwargs):
    """Factory function to get model by name"""
    models = {
        'unet': UNet,
        'unet_eam': UNet_EAM,
        'transunet': TransUnet,
        'transunet_eam': TransUnet_EAM
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    model_class = models[model_name.lower()]

    # Build appropriate kwargs for each model
    if 'transunet' in model_name.lower():
        return model_class(in_channels, n_classes, img_size=img_size, **kwargs)
    else:
        return model_class(in_channels, n_classes, **kwargs)


if __name__ == "__main__":
    # Test all models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 1, 64, 64).to(device)

    models_to_test = ['unet', 'unet_eam', 'transunet', 'transunet_eam']

    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*60}")

        model = get_model(model_name, in_channels=1, n_classes=2, deep_supervision=True).to(device)

        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {params:,}")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
            if isinstance(output, tuple):
                print(f"Output shape: {output[0].shape}")
                print(f"Deep supervision outputs: {[o.shape for o in output[1]]}")
            else:
                print(f"Output shape: {output.shape}")

        print("✓ Model test passed!")
