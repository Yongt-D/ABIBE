# models/fusion/dynamic_fusion_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DynamicFusionModule(nn.Module):
    """Advanced adaptive fusion with improved gradient flow and stronger feature integration"""

    def __init__(self, feature_dim):
        super().__init__()

        # Feature adaptation with residual connections - simplified to reduce complexity
        self.unet_adapter = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, feature_dim),
        )

        # JanusPro feature adapter - improved to better handle semantic features
        self.janus_adapter = nn.Sequential(
            nn.Conv2d(2048, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, feature_dim),
        )

        # Initial fusion layer - streamlined for better gradient flow
        self.initial_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, feature_dim),
            nn.GELU(),
        )

        # Multi-scale context aggregation with improved skip connections
        self.context_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 4, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.GroupNorm(feature_dim // 16, feature_dim // 4),
                nn.GELU()
            ) for r in [1, 3, 5, 7]
        ])

        # Enhanced skip connections for each context block
        self.skip_connections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 4, kernel_size=1, bias=False),
                nn.GroupNorm(feature_dim // 16, feature_dim // 4),
            ) for _ in range(4)
        ])

        # New: Global context encoding for large-scale building awareness
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )

        # Final fusion with higher capacity
        self.fusion_out = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        )

        # New: Gated fusion mechanism with dynamic calibration
        self.dynamic_gate = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # New: Boundary enhancement with direct connections
        self.boundary_enhancement = nn.Sequential(
            nn.Conv2d(feature_dim + 1, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, feature_dim),
            nn.GELU()
        )

        # New: Calibration module for complexity-aware weighting
        self.calibration = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Improved weight initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, unet_features, janus_features, complexity_info):
        batch_size = unet_features.shape[0]

        # Apply residual adapters with identity connections for UNet features
        unet_res = self.unet_adapter(unet_features)
        unet_feat = unet_features + unet_res  # Explicit residual connection

        # Ensure JanusPro features have correct dimensions
        if janus_features.shape[1] != 2048:
            # More robust dimension handling
            janus_features = F.adaptive_avg_pool1d(
                janus_features.flatten(2).transpose(1, 2),
                output_size=2048
            ).transpose(1, 2).reshape(
                janus_features.shape[0],
                2048,
                janus_features.shape[2],
                janus_features.shape[3]
            )

        # Process JanusPro features with residual connection
        janus_feat = self.janus_adapter(janus_features)

        # Initial concatenation and fusion
        concat_features = torch.cat([unet_feat, janus_feat], dim=1)
        fused = self.initial_fusion(concat_features)

        # Extract and apply global context
        global_ctx = self.global_context(fused)
        fused = fused + global_ctx

        # Multi-scale context blocks with skip connections
        context_outputs = []
        for i, (context_block, skip) in enumerate(zip(self.context_blocks, self.skip_connections)):
            # Enhanced context processing with skip connections
            context_out = context_block(fused)
            skip_out = skip(fused)
            context_outputs.append(context_out + skip_out)

        # Combine context outputs along channel dimension
        context_features = torch.cat(context_outputs, dim=1)

        # Apply dynamic gate for adaptive weighting
        gate_input = torch.cat([unet_feat, janus_feat], dim=1)
        gates = self.dynamic_gate(gate_input)

        # Calculate boundary map from complexity info for enhancement
        boundary_map = complexity_info.get('boundary_map',
                                           complexity_info.get('edge_complexity', None))

        # Fallback if boundary map is unavailable
        if boundary_map is None:
            # Create a simple edge map using Sobel-like operations
            boundary_map = torch.abs(
                F.conv2d(
                    F.pad(fused, [1, 1, 1, 1], mode='reflect'),
                    torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                 dtype=fused.dtype, device=fused.device).view(1, 1, 3, 3).repeat(fused.shape[1], 1, 1,
                                                                                                 1),
                    groups=fused.shape[1]
                )
            ).mean(dim=1, keepdim=True)
            boundary_map = torch.sigmoid(boundary_map)

        # Ensure boundary map has correct dimensions
        if boundary_map.shape[2:] != fused.shape[2:]:
            boundary_map = F.interpolate(
                boundary_map,
                size=fused.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # Boundary-aware enhancement
        boundary_enhanced = self.boundary_enhancement(torch.cat([fused, boundary_map], dim=1))

        # Apply calibration based on feature complexity
        calibration = self.calibration(fused)

        # Combine all features with weighted fusion
        unet_contribution = unet_feat * gates[:, 0:1]
        janus_contribution = janus_feat * gates[:, 1:2]

        # Enhanced fusion with calibrated complexity and boundary awareness
        output = self.fusion_out(context_features) + unet_contribution + janus_contribution + boundary_enhanced

        # Apply final complexity-aware calibration
        output = output * (1.0 + 0.5 * calibration)

        return output