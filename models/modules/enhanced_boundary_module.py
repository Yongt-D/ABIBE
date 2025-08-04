# models/modules/enhanced_boundary_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class EnhancedBoundaryModule(nn.Module):
    """Advanced boundary enhancement with multi-scale edge processing and attention refinement"""

    def __init__(self, in_channels, mid_channels=64):
        super().__init__()

        # Multi-scale edge detection with dilated convolutions
        self.edge_branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.GELU()
        )

        self.edge_branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.GELU()
        )

        self.edge_branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.GELU()
        )

        # Edge fusion module with channel attention
        self.edge_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels * 3, kernel_size=1),
            nn.Sigmoid()
        )

        self.edge_fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )

        # Residual boundary refinement module
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
        )

        # Feature enhancement with edge awareness
        self.enhance_conv1 = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
            nn.GELU()
        )

        self.enhance_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
            nn.GELU()
        )

        # Adaptive refinement with spatial and channel attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Adaptive refinement gate with controlled initialization
        self.refine_gate = nn.Sequential(
            nn.Conv2d(in_channels + 1, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Boundary-aware feature enhancement gate
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Initialize gate biases
        with torch.no_grad():
            if hasattr(self.refine_gate[-2], 'bias'):
                nn.init.constant_(self.refine_gate[-2].bias, 0.5)
            if hasattr(self.boundary_gate[-2], 'bias'):
                nn.init.constant_(self.boundary_gate[-2].bias, 0.5)

    def forward(self, x):
        # Extract multi-scale edge features
        edge_feat1 = self.edge_branch1(x)
        edge_feat2 = self.edge_branch2(x)
        edge_feat3 = self.edge_branch3(x)

        # Concatenate edge features
        edge_feats = torch.cat([edge_feat1, edge_feat2, edge_feat3], dim=1)

        # Apply channel attention to emphasize important edge features
        channel_weights = self.edge_channel_attention(edge_feats)
        edge_feats = edge_feats * channel_weights

        # Fuse edge features from different scales
        edge_fused = self.edge_fusion(edge_feats)
        edge_map = torch.sigmoid(edge_fused)

        # Apply refinement to edge features for better boundary definition
        refined_edges = self.boundary_refine(edge_feat1)
        refined_edges = edge_feat1 + refined_edges

        # Feature enhancement with edge awareness
        enhanced = self.enhance_conv1(torch.cat([x, edge_map], dim=1))
        enhanced = self.enhance_conv2(enhanced)

        # Channel attention
        channel_attn = self.channel_attention(enhanced)
        enhanced = enhanced * channel_attn

        # Spatial attention - use max and avg pooling across channels
        spatial_max, _ = torch.max(enhanced, dim=1, keepdim=True)
        spatial_avg = torch.mean(enhanced, dim=1, keepdim=True)
        spatial_feats = torch.cat([spatial_max, spatial_avg], dim=1)
        spatial_attn = self.spatial_attention(spatial_feats)
        enhanced = enhanced * spatial_attn

        # Create boundary-aware gate
        boundary_attention = self.boundary_gate(enhanced)

        # Generate adaptive refinement gate
        gate = self.refine_gate(torch.cat([x, edge_map], dim=1))

        # Apply boundary-aware gating mechanism with residual connection
        # Direct edge information helps with precise boundary detection
        boundary_enhanced = x + gate * (enhanced - x) + boundary_attention * edge_map

        return boundary_enhanced, edge_map