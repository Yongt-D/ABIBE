# models/backbone/simplified_vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=512, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"

        # (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # (B, embed_dim, H', W') -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class Attention(nn.Module):
    """
    Self-attention mechanism
    """

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
        """
        x: [B, N, C]
        Returns: [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, C//num_heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        x: [B, N, C]
        Returns: [B, N, C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    """
    Transformer Block
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        x: [B, N, C]
        Returns: [B, N, C]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class SimplifiedViTSegmentor(nn.Module):
    """
    Simplified Vision Transformer for Segmentation - implemented from scratch without pre-trained weights
    """

    def __init__(self, img_size=512, patch_size=16, in_channels=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., drop_rate=0.1, out_channels=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # Calculate number of patches and output size for decoder
        self.n_patches = (img_size // patch_size) ** 2
        self.patches_per_side = img_size // patch_size

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Add positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))

        # Add dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=drop_rate
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder (simple version for segmentation)
        # Project tokens back to image-like feature map
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)

        # Series of upsampling blocks to get back to original image resolution
        # We need log2(patch_size) upsampling steps to go from patch_size to 1
        num_upsampling = int(math.log2(patch_size))

        # Create upsampling blocks
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim if i == 0 else embed_dim // (2 ** (i - 1)),
                                   embed_dim // (2 ** i),
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim // (2 ** i)),
                nn.ReLU()
            )
            for i in range(1, num_upsampling + 1)
        ])

        # Final convolution to get segmentation map
        self.segmentation_head = nn.Conv2d(
            embed_dim // (2 ** num_upsampling),
            out_channels,
            kernel_size=1
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize patch embedding
        nn.init.kaiming_normal_(self.patch_embed.proj.weight)
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize decoder weights
        for m in self.decoder_blocks.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize segmentation head
        nn.init.kaiming_normal_(self.segmentation_head.weight, mode='fan_out', nonlinearity='sigmoid')
        if self.segmentation_head.bias is not None:
            nn.init.constant_(self.segmentation_head.bias, 0)

    def forward_encoder(self, x):
        """
        Forward through the encoder
        x: [B, C, H, W]
        Returns: [B, N, embed_dim]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, C]

        # Add positional embedding
        x = x + self.pos_embed

        # Apply dropout
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply final norm
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        """
        Forward through the decoder
        x: [B, N, C]
        Returns: [B, out_channels, H, W]
        """
        B, N, C = x.shape

        # Project tokens
        x = self.decoder_embed(x)

        # Reshape to 2D feature map [B, C, H', W']
        # H' and W' are the height and width of the feature map after patch embedding
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Upsampling blocks
        for block in self.decoder_blocks:
            x = block(x)

        # Final segmentation head
        x = self.segmentation_head(x)

        return x

    def forward(self, x):
        """
        Forward pass
        x: [B, C, H, W]
        Returns: [B, out_channels, H, W]
        """
        # Ensure input has the right shape
        if x.size(2) != self.img_size or x.size(3) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)

        # Extract features via encoder
        features = self.forward_encoder(x)

        # Decode features to segmentation map
        logits = self.forward_decoder(features)

        # Ensure output has the right shape
        if logits.size(2) != x.size(2) or logits.size(3) != x.size(3):
            logits = F.interpolate(logits, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        return logits