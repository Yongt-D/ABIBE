# models/backbone/enhanced_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class EnhancedUNet(nn.Module):
    """
    优化版U-Net分割网络
    增强了边界处理和特征提取能力
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(EnhancedUNet, self).__init__()
        # 输入参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.target_size = (512, 512)  # 默认输出尺寸

        # UNet编码器部分 (下采样)
        self.inc = self._double_conv(in_channels, base_channels)
        self.down1 = self._down_block(base_channels, base_channels * 2)
        self.down2 = self._down_block(base_channels * 2, base_channels * 4)
        self.down3 = self._down_block(base_channels * 4, base_channels * 8)
        self.down4 = self._down_block(base_channels * 8, base_channels * 16)

        # 解码器部分 (上采样)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels)

        # 输出层
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重，防止梯度消失/爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _double_conv(self, in_channels, out_channels):
        """双卷积块: 卷积 -> BN -> ReLU -> 卷积 -> BN -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _down_block(self, in_channels, out_channels):
        """下采样块: MaxPool -> 双卷积"""
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels)
        )

    def get_bottleneck_feature(self, x):
        """获取编码器瓶颈特征，用于融合"""
        x1 = self.inc(x)  # 第一层特征 [1/1]
        x2 = self.down1(x1)  # 下采样 [1/2]
        x3 = self.down2(x2)  # 下采样 [1/4]
        x4 = self.down3(x3)  # 下采样 [1/8]
        x5 = self.down4(x4)  # 下采样 [1/16]

        # 记录形状以便调试
        logger.debug(f"瓶颈特征形状: {x5.shape}")

        return x5  # 返回瓶颈特征

    def forward_with_injected_features(self, x, injected_features=None):
        """
        带特征注入的前向传播
        x: [B, C, H, W]
        injected_features: [B, C', H', W'] 要注入的特征，将在瓶颈层注入
        Returns: [B, out_channels, H, W]
        """
        # 编码器路径
        x1 = self.inc(x)  # 第一层特征 [1/1]
        x2 = self.down1(x1)  # 下采样 [1/2]
        x3 = self.down2(x2)  # 下采样 [1/4]
        x4 = self.down3(x3)  # 下采样 [1/8]
        x5 = self.down4(x4)  # 下采样 [1/16] - 瓶颈层

        # 如果有注入特征，在瓶颈层融合
        if injected_features is not None:
            # 确保注入特征的尺寸匹配
            if injected_features.shape[2:] != x5.shape[2:]:
                injected_features = F.interpolate(
                    injected_features,
                    size=x5.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            # 如果通道数不匹配，使用1x1卷积调整
            if injected_features.shape[1] != x5.shape[1]:
                conv = nn.Conv2d(injected_features.shape[1], x5.shape[1], kernel_size=1).to(x5.device)
                injected_features = conv(injected_features)

            # 融合特征
            x5 = x5 + injected_features
            logger.debug(f"特征已在瓶颈层注入，形状: {x5.shape}")

        # 解码器路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出层
        logits = self.outc(x)

        # 确保输出尺寸正确
        if logits.shape[-2:] != self.target_size:
            logits = F.interpolate(
                logits,
                size=self.target_size,
                mode='bilinear',
                align_corners=True
            )

        return logits

    def forward(self, x):
        """
        标准前向传播（不注入特征）
        x: [B, C, H, W]
        Returns: [B, out_channels, H, W]
        """
        return self.forward_with_injected_features(x, None)


class UpBlock(nn.Module):
    """改进的上采样块"""

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        # 上采样层
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 拼接后的通道数 = 上采样后的通道 + 跳过连接的通道
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # x1是来自下层的特征，需要上采样
        # x2是来自编码器的跳过连接

        # 先用1x1卷积减少通道数，再上采样
        x1 = self.conv1x1(x1)
        x1 = self.up(x1)

        # 处理可能的尺寸不匹配问题
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 使用padding来调整尺寸
        if diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        # 连接特征
        x = torch.cat([x2, x1], dim=1)

        # 应用卷积
        x = self.conv(x)

        return x