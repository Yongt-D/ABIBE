# models/modules/film_transformation_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FiLMTransformationModule(nn.Module):
    """Advanced Feature-wise Linear Modulation with architectural improvements for building extraction tasks"""

    def __init__(self, feature_dim, text_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.text_dim = text_dim

        # Enhanced text encoder with improved representation capacity
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )

        # Generate FiLM parameters with text-guided feature conditioning
        # More gradual progression of feature dimensions for better gradient flow
        self.gamma_generator = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, feature_dim)
        )

        self.beta_generator = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, feature_dim)
        )

        # Learnable gating mechanism to control modulation strength
        self.modulation_gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Channel attention for feature enhancement
        self.channel_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, feature_dim),
            nn.Sigmoid()
        )

        # Spatial attention for better building boundary awareness
        self.spatial_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Initialize to near-identity transformation
        self._initialize_parameters()

    def _initialize_parameters(self):
        # Initialize modulation parameters for controlled conditioning
        with torch.no_grad():
            # Initialize gamma generator to output values near 1
            if hasattr(self.gamma_generator[-1], 'bias'):
                nn.init.zeros_(self.gamma_generator[-1].bias)

            # Initialize beta generator to output values near 0
            if hasattr(self.beta_generator[-1], 'bias'):
                nn.init.zeros_(self.beta_generator[-1].bias)

            # Set modulation gate to start with moderate strength
            if hasattr(self.modulation_gate[-2], 'bias'):
                nn.init.constant_(self.modulation_gate[-2].bias, 0.0)

    def forward(self, features, text_feats):
        """
        Apply FiLM transformation conditioned on text features

        Args:
            features: [B, C, H, W] visual features
            text_feats: [B, D] text features
        """
        batch_size, C, H, W = features.shape

        # Handle text features with proper error checking
        if text_feats is None:
            # Create default text embeddings for fallback
            text_feats = torch.zeros(batch_size, self.text_dim, device=features.device)

        # Ensure text features have correct shape
        if len(text_feats.shape) == 3:
            text_feats = text_feats.squeeze(1)  # [B, 1, D] -> [B, D]
        elif len(text_feats.shape) == 1:
            text_feats = text_feats.unsqueeze(0)  # [D] -> [1, D]

        # Handle batch size mismatches
        if text_feats.shape[0] != batch_size:
            if text_feats.shape[0] == 1:
                text_feats = text_feats.repeat(batch_size, 1)  # Broadcast to batch
            else:
                text_feats = text_feats[:batch_size]  # Truncate to batch size

        # Encode text features with enhanced representation
        text_encoded = self.text_encoder(text_feats)  # [B, 256]

        # Generate FiLM parameters
        gamma = self.gamma_generator(text_encoded).view(batch_size, C, 1, 1)  # [B, C, 1, 1]
        beta = self.beta_generator(text_encoded).view(batch_size, C, 1, 1)  # [B, C, 1, 1]

        # Generate channel attention weights for selective feature emphasis
        attention = self.channel_attention(text_encoded).view(batch_size, C, 1, 1)  # [B, C, 1, 1]

        # Generate spatial attention for building region emphasis
        spatial_attn = self.spatial_attention(text_encoded).view(batch_size, 1, 1, 1)  # [B, 1, 1, 1]

        # Generate dynamic modulation gate for controlled transformation
        mod_gate = self.modulation_gate(text_encoded).view(batch_size, 1, 1, 1)  # [B, 1, 1, 1]

        # Apply controlled FiLM transformation with carefully bounded parameters
        # 将变换参数范围缩小，减少对原始特征的干扰
        gamma = torch.tanh(gamma) * 0.15 + 1.0  # 从0.3降低到0.1,范围变为[0.9, 1.1]
        beta = torch.tanh(beta) * 0.15  # 从0.2降低到0.1,范围变为[-0.1, 0.1]

        # Apply transformation with adaptive channel attention and gating
        # Channel attention modulates feature importance based on text
        transformed = features * (1.0 + mod_gate * (gamma * attention - 1.0)) + mod_gate * beta

        # Apply spatial attention for building region emphasis with residual connection
        # This helps the model focus on building boundaries more effectively
        enhanced = transformed + spatial_attn * features

        return enhanced