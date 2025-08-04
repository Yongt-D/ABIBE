# models/backbone/enhanced_cross_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)


class EnhancedCrossAttention(nn.Module):
    """Enhanced cross-modal attention with improved multimodal fusion capabilities"""

    def __init__(self, image_dim, text_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads
        assert self.head_dim * num_heads == image_dim, "image_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        # Improved text projection with layer normalization and residual connection
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.GELU()
        )

        # Additional context encoding for better semantic alignment
        self.context_encoder = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.LayerNorm(text_dim // 2),
            nn.GELU(),
            nn.Linear(text_dim // 2, image_dim),
            nn.LayerNorm(image_dim)
        )

        # Enhanced multi-head attention components with separate projections per head
        self.q_proj = nn.Linear(image_dim, image_dim)
        self.k_proj = nn.Linear(image_dim, image_dim)
        self.v_proj = nn.Linear(image_dim, image_dim)

        # Output projection with improved residual gating
        self.out_proj = nn.Linear(image_dim, image_dim)
        self.norm1 = nn.LayerNorm(image_dim)
        self.norm2 = nn.LayerNorm(image_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network for output refinement
        self.ffn = nn.Sequential(
            nn.Linear(image_dim, image_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(image_dim * 4, image_dim),
            nn.Dropout(dropout)
        )

        # Advanced adaptive attention gate with text-guided modulation
        self.adaptive_gate = nn.Sequential(
            nn.Linear(image_dim * 2, image_dim // 2),
            nn.GELU(),
            nn.Linear(image_dim // 2, image_dim // 4),
            nn.GELU(),
            nn.Linear(image_dim // 4, 1),
            nn.Sigmoid()
        )

        # Text-conditional modulation layers
        self.gamma_generator = nn.Sequential(
            nn.Linear(text_dim, image_dim),
            nn.Tanh()  # Use tanh to limit range
        )

        self.beta_generator = nn.Sequential(
            nn.Linear(text_dim, image_dim),
            nn.Tanh()  # Use tanh to limit range
        )

        # Initialize with appropriate scaling
        self._initialize_parameters()

    def _initialize_parameters(self):
        # Special initialization for better gradient flow
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initialize modulation layers to start closer to identity mapping
        for name, p in self.named_parameters():
            if 'gamma_generator' in name or 'beta_generator' in name:
                if p.dim() > 1:  # weights
                    nn.init.zeros_(p)
                else:  # biases
                    if 'gamma' in name:
                        nn.init.ones_(p)
                    else:
                        nn.init.zeros_(p)

    def forward(self, image_feats, text_feats):
        # Handle spatial features - reshape to [B, L, C]
        batch_size = image_feats.size(0)
        B, C, H, W = image_feats.shape
        L = H * W
        image_feats_flat = image_feats.flatten(2).transpose(1, 2)  # [B, L, C]

        # Shape checking and handling for text features
        if len(text_feats.shape) == 2:  # [B, C]
            text_feats = text_feats
        elif len(text_feats.shape) == 3 and text_feats.shape[1] == 1:  # [B, 1, C]
            text_feats = text_feats.squeeze(1)
        else:
            text_feats = text_feats.mean(dim=1)  # Average over sequence dimension if needed

        # Project text features and generate modulation parameters
        text_proj = self.text_projection(text_feats)
        text_context = self.context_encoder(text_feats)

        # Generate conditional modulation parameters with controlled magnitude
        gamma = self.gamma_generator(text_feats).unsqueeze(1)  # [B, 1, C]
        beta = self.beta_generator(text_feats).unsqueeze(1)  # [B, 1, C]

        # Scale parameters to prevent extreme modulation
        gamma = gamma * 0.1 + 1.0  # Center around 1 with small deviation
        beta = beta * 0.1  # Small additive shifts

        # Apply modulation to image features for conditional processing
        # This helps align image features with text semantics before attention
        image_feats_mod = image_feats_flat * gamma + beta

        # Self attention for image with residual connection
        q = self.q_proj(image_feats_mod)  # [B, L, C]
        k = self.k_proj(text_proj.unsqueeze(1))  # [B, 1, C]
        v = self.v_proj(text_context.unsqueeze(1))  # [B, 1, C]

        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]

        # Calculate scaled attention scores with improved numerical stability
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, L, 1]

        # Apply attention masking and softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout1(attn)

        # Apply attention and project back
        out = torch.matmul(attn, v)  # [B, num_heads, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, C)  # [B, L, C]
        out = self.out_proj(out)

        # First residual connection and normalization
        out = self.norm1(image_feats_flat + self.dropout1(out))

        # FFN with second residual connection
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)

        # Calculate adaptive gate for mixing with original features
        # Generate global text-image embedding for dynamic gating
        global_img_embed = torch.mean(image_feats_flat, dim=1)  # [B, C]
        if global_img_embed.shape[1] != text_feats.shape[1]:
            # Option 1: Project text features to match global_img_embed dimension
            if hasattr(self, 'text_projection'):
                text_feats_proj = self.text_projection(text_feats)
            else:
                # Or create a simple linear projection
                text_feats_proj = nn.Linear(text_feats.shape[1], global_img_embed.shape[1]).to(text_feats.device)(
                    text_feats)

            gate_input = torch.cat([global_img_embed, text_feats_proj], dim=1)  # [B, 2*C]
        else:
            gate_input = torch.cat([global_img_embed, text_feats], dim=1)  # [B, 2*C]
        gate_value = self.adaptive_gate(gate_input).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1]

        # Reshape back to spatial form with dynamic gating factor
        out = out.transpose(1, 2).reshape(B, C, H, W)

        # Strong residual connection with adaptive gating
        # This ensures high-quality gradient flow throughout training
        output = image_feats + gate_value * out

        return output