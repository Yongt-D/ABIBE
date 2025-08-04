# models/fusion/multimodal_aggregator.py (新建)
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalAggregator(nn.Module):
    def __init__(self, image_dim, text_dim, output_dim):
        super().__init__()

        # 文本特征投影
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

        # 图像特征投影
        self.image_projection = nn.Sequential(
            nn.Conv2d(image_dim, output_dim, kernel_size=1),
            nn.GroupNorm(32, output_dim),
            nn.ReLU()
        )

        # 文本引导的图像特征调制
        self.text_guided_modulation = TextGuidedModulation(output_dim)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(output_dim * 2, output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, output_dim),
            nn.ReLU()
        )

    def forward(self, image_features, text_features):
        batch_size, _, H, W = image_features.shape

        # 投影文本特征
        text_feat = self.text_projection(text_features)

        # 投影图像特征
        image_feat = self.image_projection(image_features)

        # 文本引导调制
        modulated_image = self.text_guided_modulation(image_feat, text_feat)

        # 扩展文本特征以匹配图像尺寸
        text_feat = text_feat.view(batch_size, -1, 1, 1).expand(-1, -1, H, W)

        # 融合特征
        concat_feat = torch.cat([modulated_image, text_feat], dim=1)
        fused_feat = self.fusion(concat_feat)

        return fused_feat


class TextGuidedModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # FiLM生成器
        self.film_generator = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels * 2)
        )

    def forward(self, image_features, text_features):
        # 生成FiLM参数
        film_params = self.film_generator(text_features)
        gamma, beta = torch.chunk(film_params, 2, dim=1)

        # 重塑为[B, C, 1, 1]
        gamma = gamma.view(*gamma.shape, 1, 1)
        beta = beta.view(*beta.shape, 1, 1)

        # 应用FiLM变换
        modulated_features = gamma * image_features + beta

        return modulated_features