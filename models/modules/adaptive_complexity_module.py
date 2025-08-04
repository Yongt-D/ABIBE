# models/modules/adaptive_complexity_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class AdaptiveComplexityModule(nn.Module):
    """Advanced complexity assessment with hierarchical multi-scale analysis"""

    def __init__(self, in_channels):
        super().__init__()

        # Multi-scale pooling pyramid for comprehensive feature aggregation
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.map1 = nn.AdaptiveAvgPool2d(2)  # Medium scale pooling
        self.map2 = nn.AdaptiveAvgPool2d(4)  # Small scale pooling
        self.map3 = nn.AdaptiveAvgPool2d(8)  # Finer scale pooling

        # Global complexity estimation with improved architecture
        self.global_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Local pattern complexity estimation with residual connections
        self.local_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # Multi-scale feature processing
        self.map1_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )

        self.map2_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )

        self.map3_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )

        # Boundary detection branch - critical for building extraction
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Final complexity assessment with boundary awareness
        self.complexity_fusion = nn.Sequential(
            nn.Conv2d(16 * 3 + 1 + 1, 32, kernel_size=1),  # Extra +1 for boundary
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Edge-aware complexity estimator
        self.edge_complexity = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Global complexity assessment
        global_features = self.gap(x)
        global_complexity = self.global_estimator(global_features)

        # Local pattern complexity estimation
        local_features = self.local_estimator(x)

        # Detect boundary regions - critical for building extraction
        boundary_map = self.boundary_detector(local_features)

        # Multi-scale feature aggregation
        map1_features = self.map1(local_features)
        map2_features = self.map2(local_features)
        map3_features = self.map3(local_features)

        # Process multi-scale features
        map1_processed = self.map1_conv(map1_features)
        map2_processed = self.map2_conv(map2_features)
        map3_processed = self.map3_conv(map3_features)

        # Edge complexity estimation - helps with building boundaries
        edge_complexity = self.edge_complexity(local_features)

        # Pool local features to global
        map1_pooled = F.adaptive_avg_pool2d(map1_processed, 1)
        map2_pooled = F.adaptive_avg_pool2d(map2_processed, 1)
        map3_pooled = F.adaptive_avg_pool2d(map3_processed, 1)
        boundary_pooled = F.adaptive_avg_pool2d(boundary_map, 1)

        # Combine global and multi-scale local complexities with boundary awareness
        combined = torch.cat([
            global_complexity,
            map1_pooled,
            map2_pooled,
            map3_pooled,
            boundary_pooled
        ], dim=1)

        refined_complexity = self.complexity_fusion(combined)

        # Create detailed local complexity map for spatial attention
        local_complexity_map = F.interpolate(
            map2_processed,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        # Prepare output with comprehensive complexity information
        return {
            'overall_complexity': refined_complexity,
            'complexity_factor': torch.clamp(refined_complexity, 0.2, 0.8),  # Wider range for adaptation
            'local_complexity_map': local_complexity_map,
            'boundary_map': F.interpolate(boundary_map, size=x.shape[2:], mode='bilinear', align_corners=True),
            'edge_complexity': F.interpolate(edge_complexity, size=x.shape[2:], mode='bilinear', align_corners=True)
        }