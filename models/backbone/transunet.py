# models/backbone/transunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
import logging

logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """2D图像到patch嵌入"""

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
        返回: [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入图像尺寸({H}*{W})与模型({self.img_size}*{self.img_size})不匹配"

        # (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # (B, embed_dim, H', W') -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class Attention(nn.Module):
    """多头自注意力机制"""

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


class Mlp(nn.Module):
    """多层感知机"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer编码器块"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码表
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class DecoderBlock(nn.Module):
    """简化的解码器块，专注于稳定性"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 上采样层
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 投影到目标通道数
        self.conv_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 处理跳跃连接
        if skip_channels > 0:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(skip_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 融合卷积
        if skip_channels > 0:
            # 如果有跳跃连接，融合后的通道数是out_channels*2
            fusion_in = out_channels * 2
        else:
            # 如果没有跳跃连接，融合后的通道数是out_channels
            fusion_in = out_channels

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        # 上采样
        x = self.up(x)

        # 投影到目标通道数
        x = self.conv_proj(x)

        # 如果有跳跃连接，进行处理和连接
        if skip is not None:
            # 处理尺寸不匹配
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            # 处理跳跃连接通道数
            skip = self.skip_proj(skip)

            # 连接特征
            x = torch.cat([x, skip], dim=1)

        # 应用融合卷积
        x = self.fusion_conv(x)
        return x


class TransUNet(nn.Module):
    """
    简化版TransUNet，专注于稳定性和可靠性
    """

    def __init__(self, img_size=512, patch_size=16, in_channels=3, out_channels=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1,
                 num_stages=4, skip_attention=True):
        super(TransUNet, self).__init__()

        # 断言输入参数
        assert img_size % patch_size == 0, f"图像尺寸({img_size})必须是块大小({patch_size})的整数倍"
        assert embed_dim % num_heads == 0, f"嵌入维度({embed_dim})必须能被注意力头数量({num_heads})整除"

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_stages = num_stages

        # 初始下采样路径 - 产生跳跃连接
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(  # 第一层: 1/2
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(  # 第二层: 1/4
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(  # 第三层: 1/8
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(  # 第四层: 1/16
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        ])

        # 计算Transformer输入尺寸
        vit_input_size = img_size // 16  # 1/16下采样
        assert vit_input_size * 16 == img_size, f"输入大小({img_size})不能被16整除"

        # Patch Embedding层
        self.patch_embed = PatchEmbed(
            img_size=vit_input_size,
            patch_size=1,  # 已经下采样16倍，只需要1x1的patch
            in_channels=512,
            embed_dim=embed_dim
        )

        # 位置编码和cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, (vit_input_size // 1) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer编码器
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate
        )

        # 从Transformer特征转回2D特征图
        trans_proj_dim = 512  # Transformer投影后的通道数
        self.trans_proj = nn.Linear(embed_dim, trans_proj_dim)

        # 解码器通道配置
        decoder_channels = [512, 256, 128, 64][:num_stages]
        skip_channels = [512, 256, 128, 64][:num_stages]

        # 构建解码器
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_stages):
            if i == 0:
                # 第一个解码器接收Transformer的输出
                self.decoder_blocks.append(
                    DecoderBlock(
                        in_channels=trans_proj_dim,
                        skip_channels=skip_channels[i] if i < len(skip_channels) else 0,
                        out_channels=decoder_channels[i]
                    )
                )
            else:
                # 后续解码器接收前一个解码器的输出
                self.decoder_blocks.append(
                    DecoderBlock(
                        in_channels=decoder_channels[i - 1],
                        skip_channels=skip_channels[i] if i < len(skip_channels) else 0,
                        out_channels=decoder_channels[i]
                    )
                )

        # 最终输出层
        final_channels = decoder_channels[-1]
        self.final_conv = nn.Conv2d(final_channels, out_channels, kernel_size=1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化位置编码和cls token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 初始化线性层和卷积层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        """前向传播编码器部分，提取多层次特征"""
        features = []
        # 通过CNN编码器获取不同尺度特征
        for encoder in self.encoder_layers:
            x = encoder(x)
            features.append(x)  # 存储跳跃连接特征

        # Transformer处理
        x_t = self.patch_embed(x)  # [B, n_patches, embed_dim]

        # 添加cls token
        cls_token = self.cls_token.expand(x_t.shape[0], -1, -1)
        x_t = torch.cat((cls_token, x_t), dim=1)

        # 添加位置编码
        x_t = x_t + self.pos_embed
        x_t = self.pos_drop(x_t)

        # 经过Transformer
        x_t = self.transformer(x_t)

        return x_t, features

    def forward_decoder(self, x, features):
        """前向传播解码器部分"""
        # 移除cls token
        x = x[:, 1:, :]

        # 从序列恢复到特征图
        B, n_patch, C = x.shape
        h = w = int(np.sqrt(n_patch))

        # 投影回特征空间
        x = self.trans_proj(x)
        x = x.permute(0, 2, 1).reshape(B, -1, h, w)

        # 通过解码器块
        decoder_features = []
        for i, decoder in enumerate(self.decoder_blocks):
            # 使用倒序的编码器特征作为跳跃连接
            skip = features[-(i + 1)] if i < len(features) else None
            x = decoder(x, skip)
            decoder_features.append(x)

        # 最终输出
        x = self.final_conv(x)

        return x, decoder_features

    def forward(self, x):
        """完整的前向传播"""
        # 记录原始尺寸
        input_size = x.shape[-2:]

        # 编码器部分
        transformer_out, skip_features = self.forward_encoder(x)

        # 解码器部分
        logits, _ = self.forward_decoder(transformer_out, skip_features)

        # 确保输出与输入尺寸匹配
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)

        return logits