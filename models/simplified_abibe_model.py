# models/simplified_abibe_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

from models.backbone.enhanced_unet import EnhancedUNet
from models.backbone.janus_pro_extractor import JanusProExtractor

logger = logging.getLogger(__name__)


class SimplifiedAttentionMechanism(nn.Module):
    """按照图中的注意力机制实现"""

    def __init__(self, feature_dim, text_dim):
        super().__init__()
        self.feature_dim = feature_dim

        # Query projection
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        # Key和Value projection
        self.k_proj = nn.Linear(text_dim, feature_dim)
        self.v_proj = nn.Linear(text_dim, feature_dim)

        self.scale = feature_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_features, text_features):
        B, C, H, W = image_features.shape

        # 展平图像特征
        image_flat = image_features.flatten(2).transpose(1, 2)  # [B, HW, C]

        # 处理文本特征维度
        if len(text_features.shape) == 3:
            text_features = text_features.squeeze(1)  # [B, 1, D] -> [B, D]

        # 生成Q, K, V
        Q = self.q_proj(image_flat)  # [B, HW, C]
        K = self.k_proj(text_features).unsqueeze(1)  # [B, 1, C]
        V = self.v_proj(text_features).unsqueeze(1)  # [B, 1, C]

        # 计算注意力
        attention = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, HW, 1]
        attention = self.softmax(attention)

        # 应用注意力
        out = torch.matmul(attention, V)  # [B, HW, C]

        # 重新调整形状回到 [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out, attention


class SimplifiedFiLMTransformation(nn.Module):
    """简化的FiLM变换"""

    def __init__(self, feature_dim, text_dim):
        super().__init__()

        # 生成gamma和beta
        self.gamma_proj = nn.Linear(text_dim, feature_dim)
        self.beta_proj = nn.Linear(text_dim, feature_dim)

    def forward(self, features, text_features):
        # 处理文本特征维度
        if len(text_features.shape) == 3:
            text_features = text_features.squeeze(1)

        # 生成gamma和beta
        gamma = self.gamma_proj(text_features).view(-1, features.size(1), 1, 1)
        beta = self.beta_proj(text_features).view(-1, features.size(1), 1, 1)

        # 应用FiLM变换 - 限制gamma的范围避免过度调制
        gamma = torch.tanh(gamma) * 0.1  # 限制在[-0.1, 0.1]范围内
        modulated = features * (1 + gamma) + beta * 0.1

        return modulated


class AdaptiveFusionModule(nn.Module):
    """自适应融合模块，按照图中的设计"""

    def __init__(self, feature_dim):
        super().__init__()

        # 特征门控
        self.unet_gate = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.Sigmoid()
        )

        self.janus_gate = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.Sigmoid()
        )

        # 特征连接后的1x1卷积
        self.concat_conv = nn.Conv2d(feature_dim * 2, feature_dim, 1)

    def forward(self, unet_feat, janus_feat, complexity_factor):
        # 特征门控
        unet_gated = unet_feat * self.unet_gate(unet_feat)
        janus_gated = janus_feat * self.janus_gate(janus_feat)

        # 连接特征
        concat_feat = torch.cat([unet_gated, janus_gated], dim=1)
        fused = self.concat_conv(concat_feat)

        # 根据复杂度调整融合权重
        weighted_fusion = fused * (0.7 + 0.3 * complexity_factor)  # 限制复杂度的影响范围

        return weighted_fusion


class ComplexityAssessmentBranch(nn.Module):
    """复杂度评估分支"""

    def __init__(self, feature_dim):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        # 全局平均池化
        pooled = self.gap(features).flatten(1)
        # MLP评估复杂度
        complexity = self.mlp(pooled).view(-1, 1, 1, 1)
        return complexity


class BoundaryEnhancementModule(nn.Module):
    """边界增强模块"""

    def __init__(self, in_channels):
        super().__init__()

        # 边缘检测器
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        # 边界增强卷积
        self.boundary_conv = nn.Conv2d(in_channels + 1, in_channels, 3, padding=1)

    def forward(self, features):
        # 检测边缘
        edge_map = self.edge_detector(features)
        # 连接原始特征和边缘图
        enhanced = self.boundary_conv(torch.cat([features, edge_map], dim=1))
        return enhanced, edge_map


class SimplifiedABIBE(nn.Module):
    """简化的ABIBE模型，严格按照融合图实现"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = config.get('model', {})

        # 基础参数
        self.in_channels = model_config.get('in_channels', 3)
        self.out_channels = model_config.get('out_channels', 1)
        self.base_channels = model_config.get('base_channels', 64)
        self.feature_dim = self.base_channels * 16  # UNet的瓶颈特征维度

        # 特性开关
        self.enable_janus = model_config.get('enable_janus', False)
        self.use_text = model_config.get('use_text', False)

        # 初始化UNet
        self.unet = EnhancedUNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            base_channels=self.base_channels
        )

        # 如果启用JanusPro
        if self.enable_janus:
            # JanusPro提取器
            janus_model_path = model_config.get('janus_model_path', 'models/janus_pro_1b')
            self.janus_extractor = JanusProExtractor(
                janus_model_path,
                freeze_model=model_config.get('freeze_janus', True)
            )

            # UNet特征投影
            self.unet_projection = nn.Conv2d(self.feature_dim, 384, 1)

            # JanusPro特征投影
            self.janus_projection = nn.Conv2d(2048, 384, 1)

            # 文本特征投影
            self.text_projection = nn.Linear(2048, 512)

            # 注意力机制
            self.attention = SimplifiedAttentionMechanism(384, 512)

            # FiLM变换
            self.film = SimplifiedFiLMTransformation(384, 512)

            # 自适应融合
            self.adaptive_fusion = AdaptiveFusionModule(384)

            # 复杂度评估
            self.complexity_assessment = ComplexityAssessmentBranch(384)

            # 边界增强
            self.boundary_enhancement = BoundaryEnhancementModule(384)

            # 最终输出
            self.output_conv = nn.Conv2d(384, self.out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images, text_features=None):
        """前向传播"""
        # 如果不使用JanusPro，直接返回UNet结果
        if not self.enable_janus:
            return self.unet(images)

        # 保存原始数据类型
        original_dtype = images.dtype
        device = images.device

        # 获取UNet瓶颈特征
        unet_bottleneck = self.unet.get_bottleneck_feature(images)
        unet_features = self.unet_projection(unet_bottleneck)

        # 准备JanusPro的输入 - 始终使用float32
        images_for_janus = images.float()  # 强制转换为float32

        # 提取JanusPro特征
        try:
            janus_output = self.janus_extractor(images_for_janus, text_features)
            janus_features = janus_output['image_features'][-1]  # 最深层特征
            text_feats = janus_output.get('text_features')

            # 将JanusPro特征转换为float32
            janus_features = janus_features.float()

            if text_feats is not None:
                text_feats = text_feats.float()

        except Exception as e:
            logger.error(f"JanusPro特征提取失败: {str(e)}")
            # 直接返回UNet输出
            return self.unet(images)

        # 确保所有特征在相同设备上
        janus_features = janus_features.to(device)
        if text_feats is not None:
            text_feats = text_feats.to(device)

        # 调整JanusPro特征尺寸
        if janus_features.shape[2:] != unet_features.shape[2:]:
            janus_features = F.interpolate(
                janus_features,
                size=unet_features.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # 投影JanusPro特征
        janus_features = self.janus_projection(janus_features)

        # 如果使用文本特征
        if self.use_text and text_feats is not None:
            # 投影文本特征到合适的维度
            text_feats_proj = self.text_projection(text_feats)

            # 注意力机制
            attended_features, _ = self.attention(janus_features, text_feats_proj)
            janus_features = janus_features + attended_features

            # FiLM变换
            janus_features = self.film(janus_features, text_feats_proj)

        # 复杂度评估
        complexity_factor = self.complexity_assessment(unet_features)

        # 自适应融合
        fused_features = self.adaptive_fusion(unet_features, janus_features, complexity_factor)

        # 边界增强
        enhanced_features, edge_map = self.boundary_enhancement(fused_features)

        # 生成输出
        output = self.output_conv(enhanced_features)

        # 调整到原始分辨率
        if output.shape[2:] != images.shape[2:]:
            output = F.interpolate(
                output,
                size=images.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # 确保输出数据类型正确 - 返回与输入相同的类型
        output = output.to(original_dtype)

        return output