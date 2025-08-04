import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

# 导入基础组件
from models.backbone.enhanced_unet import EnhancedUNet

# 导入JanusPro特征提取器
try:
    from models.backbone.janus_pro_extractor import JanusProExtractor
except ImportError as e:
    logging.warning(f"无法导入JanusProExtractor: {str(e)}")

logger = logging.getLogger(__name__)


class FeatureFusionModule(nn.Module):
    """多层级特征融合模块"""

    def __init__(self, unet_channels, janus_channels):
        super(FeatureFusionModule, self).__init__()

        # 调整JanusPro特征通道数以匹配U-Net特征
        self.janus_projection = nn.Sequential(
            nn.Conv2d(janus_channels, unet_channels, kernel_size=1),
            nn.BatchNorm2d(unet_channels),
            nn.ReLU(inplace=True)
        )

        # 注意力生成器
        self.attention_generator = nn.Sequential(
            nn.Conv2d(unet_channels * 2, unet_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(unet_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_channels, unet_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(unet_channels),
            nn.Sigmoid()
        )

        # 输出调整
        self.output_conv = nn.Sequential(
            nn.Conv2d(unet_channels * 2, unet_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(unet_channels),
            nn.ReLU(inplace=True)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, unet_features, janus_features):
        """前向传播"""
        batch_size = unet_features.shape[0]

        # 确保JanusPro特征尺寸与U-Net特征匹配
        if janus_features.shape[2:] != unet_features.shape[2:]:
            janus_features = F.interpolate(
                janus_features,
                size=unet_features.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # 投影JanusPro特征
        projected_janus = self.janus_projection(janus_features)

        # 拼接特征以生成注意力图
        concat_features = torch.cat([unet_features, projected_janus], dim=1)
        attention_map = self.attention_generator(concat_features)

        # 应用注意力
        attended_features = unet_features * attention_map + projected_janus * (1 - attention_map)

        # 最终融合
        final_concat = torch.cat([unet_features, attended_features], dim=1)
        output = self.output_conv(final_concat)

        return output


class BoundaryEnhancementModule(nn.Module):
    """边界增强模块，用于提高边界预测的精确度"""

    def __init__(self, in_channels):
        super().__init__()

        # 边缘检测器 - 使用多尺度特征融合提高鲁棒性
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 边界特化卷积 - 增加BatchNorm提高稳定性
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, 128, kernel_size=3, padding=1),  # 增加通道数
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 增加一层提高容量
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 注意力门控 - 增加平滑参数提高稳定性
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 添加稳定系数 - 控制边界特征的强度
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 初始化为较小值

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模块权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        前向传播
        Args:
            features: 输入特征 [B, C, H, W]
        Returns:
            enhanced_features: 边界增强后的特征 [B, C, H, W]
        """
        # 提取边缘图
        edge_map = self.edge_detector(features)

        # 连接原始特征和边缘图
        concat_features = torch.cat([features, edge_map], dim=1)

        # 边界特化处理
        boundary_features = self.boundary_conv(concat_features)

        # 计算注意力门控
        gate_input = torch.cat([features, boundary_features], dim=1)
        attention_map = self.attention_gate(gate_input)

        # 应用注意力门控进行特征融合，使用学习的稳定系数
        enhanced_features = features + self.alpha * attention_map * boundary_features

        return enhanced_features, edge_map


class FeatureGating(nn.Module):
    """特征门控模块"""

    def __init__(self, unet_channels, janus_channels):
        super().__init__()
        # 确保通道数正确
        total_channels = unet_channels + janus_channels
        self.gate = nn.Sequential(
            nn.Conv2d(total_channels, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, janus_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, unet_features, janus_features):
        # 确保特征尺寸匹配
        if unet_features.shape[2:] != janus_features.shape[2:]:
            janus_features = F.interpolate(
                janus_features,
                size=unet_features.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # 拼接特征
        combined = torch.cat([unet_features, janus_features], dim=1)
        # 生成门控值
        gate_values = self.gate(combined)
        # 应用门控
        gated_features = janus_features * gate_values
        return gated_features


class CrossModalAttention(nn.Module):
    """跨模态注意力机制（Q,K,V结构）"""

    def __init__(self, unet_dim, janus_dim):
        super().__init__()
        self.attention_dim = 256  # 注意力特征维度

        # Q：从U-Net特征映射
        self.query_proj = nn.Conv2d(unet_dim, self.attention_dim, kernel_size=1)
        # K：从JanusPro特征映射
        self.key_proj = nn.Conv2d(janus_dim, self.attention_dim, kernel_size=1)
        # V：从JanusPro特征映射，映射到U-Net维度便于融合
        self.value_proj = nn.Conv2d(janus_dim, unet_dim, kernel_size=1)

        # 注意力缩放因子
        self.scale = self.attention_dim ** -0.5

        # 输出投影，用于最终的融合
        self.output_proj = nn.Conv2d(unet_dim, unet_dim, kernel_size=1)

    def forward(self, unet_feat, janus_feat):
        batch_size = unet_feat.shape[0]

        # 形状调整确保兼容
        if unet_feat.shape[2:] != janus_feat.shape[2:]:
            janus_feat = F.interpolate(janus_feat,
                                       size=unet_feat.shape[2:],
                                       mode='bilinear',
                                       align_corners=True)

        # 生成查询(Q)、键(K)、值(V)
        q = self.query_proj(unet_feat)  # [B, 256, H, W]
        k = self.key_proj(janus_feat)  # [B, 256, H, W]
        v = self.value_proj(janus_feat)  # [B, unet_dim, H, W]

        # 获取空间维度
        B, C, H, W = q.shape

        # 重塑为注意力操作的形状
        q = q.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        k = k.view(B, C, -1)  # [B, C, H*W]
        v = v.view(B, -1, H * W)  # [B, unet_dim, H*W]

        # 计算注意力分数
        attn = torch.bmm(q, k) * self.scale  # [B, H*W, H*W]
        attn = F.softmax(attn, dim=-1)  # 在空间维度上归一化

        # 应用注意力权重
        output = torch.bmm(v, attn.permute(0, 2, 1))  # [B, unet_dim, H*W]
        output = output.view(B, -1, H, W)  # [B, unet_dim, H, W]

        # 最终投影并加入残差连接
        output = self.output_proj(output)
        return unet_feat + output  # 残差连接


class BuildingExtractionModel(nn.Module):
    """
    建筑提取模型
    集成U-Net、JanusPro特征、跨模态注意力机制和边界增强模块
    """

    def __init__(self, config):
        super(BuildingExtractionModel, self).__init__()

        self.config = config
        model_config = config.get('model', {})
        logger.info(f"模型配置: {model_config}")

        # 确定输入通道数和基础通道数
        in_channels = model_config.get('in_channels', 3)  # 默认为RGB输入
        base_channels = model_config.get('base_channels', 64)  # 基础通道数
        out_channels = model_config.get('out_channels', 1)  # 输出通道数，默认为1（二值分割）

        # 初始化基础分割网络
        self.unet = EnhancedUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels
        )
        logger.info(
            f"初始化UNet分割网络: in_channels={in_channels}, out_channels={out_channels}, base_channels={base_channels}")

        # 是否使用JanusPro特征
        self.enable_janus = model_config.get('enable_janus', False)
        self.use_text = model_config.get('use_text', False)

        # 是否启用边界增强
        self.enable_boundary = model_config.get('enable_boundary', True)

        # 初始化边界增强模块
        if self.enable_boundary:
            bottleneck_channels = base_channels * 16
            self.boundary_module = BoundaryEnhancementModule(bottleneck_channels)
            logger.info(f"初始化边界增强模块, 输入通道数: {bottleneck_channels}")
        else:
            self.boundary_module = None

        if self.enable_janus:
            janus_model_path = model_config.get('janus_model_path', 'models/janus_pro_1b')
            if not Path(janus_model_path).exists():
                logger.warning(f"JanusPro模型路径不存在: {janus_model_path}")
                self.enable_janus = False
                self.janus_extractor = None
                self.feature_gating = None
                self.cross_attn = None
                self.feature_fusion_modules = None
            else:
                try:
                    self.janus_extractor = JanusProExtractor(janus_model_path)

                    # 初始化特征门控模块和跨模态注意力 - 注意这里的通道数
                    bottleneck_channels = base_channels * 16  # U-Net瓶颈层通道数
                    janus_channels = 2048  # JanusPro特征通道数

                    # 多层级特征融合模块 - 新增
                    self.feature_fusion1 = FeatureFusionModule(base_channels * 8, 1024)  # 解码器第1层
                    self.feature_fusion2 = FeatureFusionModule(base_channels * 4, 512)  # 解码器第2层
                    self.feature_fusion3 = FeatureFusionModule(base_channels * 2, 256)  # 解码器第3层
                    self.feature_fusion4 = FeatureFusionModule(base_channels, 256)  # 解码器第4层

                    # 特征门控
                    self.feature_gating = FeatureGating(bottleneck_channels, janus_channels)

                    # 跨模态注意力机制
                    self.cross_attn = CrossModalAttention(bottleneck_channels, janus_channels)

                    logger.info(f"初始化JanusPro特征提取器、特征门控、跨模态注意力机制和多层级特征融合模块，"
                                f"U-Net瓶颈通道数: {bottleneck_channels}, JanusPro通道数: {janus_channels}")
                except Exception as e:
                    logger.error(f"初始化JanusPro特征提取器失败: {str(e)}")
                    self.enable_janus = False
                    self.janus_extractor = None
                    self.feature_gating = None
                    self.cross_attn = None
                    self.feature_fusion_modules = None
        else:
            self.janus_extractor = None
            self.feature_gating = None
            self.cross_attn = None
            self.feature_fusion_modules = None
            logger.info("未启用JanusPro特征")

        logger.info(
            f"初始化建筑提取模型完成，JanusPro: {'启用' if self.enable_janus else '禁用'}, "
            f"边界增强: {'启用' if self.enable_boundary else '禁用'}")

    def forward(self, images, text_features=None):
        """
        前向传播
        images: 输入图像 [B, C, H, W]
        text_features: 可选的文本特征 [B, D]
        """
        batch_size, _, height, width = images.shape
        edge_map = None  # 初始化边缘图为None

        try:
            # 提取JanusPro特征（如果启用）
            janus_features = None

            if self.enable_janus and self.janus_extractor is not None:
                try:
                    # 提取所有特征
                    janus_output = self.janus_extractor(images, text_features)

                    # 检查返回是否为字典
                    if isinstance(janus_output, dict):
                        janus_features = janus_output.get('image_features', None)
                        edge_map = janus_output.get('edge_map', None)
                    else:
                        janus_features = janus_output

                except Exception as e:
                    logger.error(f"提取JanusPro特征失败: {str(e)}")
                    import traceback as tb
                    logger.error(tb.format_exc())
                    janus_features = None

            # 多层级特征融合的新实现
            if self.enable_janus and janus_features is not None and isinstance(janus_features, list) and len(
                    janus_features) >= 4:
                # 获取U-Net编码器各层特征
                x1 = self.unet.inc(images)  # 第一层特征 [1/1]
                x2 = self.unet.down1(x1)  # 下采样 [1/2]
                x3 = self.unet.down2(x2)  # 下采样 [1/4]
                x4 = self.unet.down3(x3)  # 下采样 [1/8]
                x5 = self.unet.down4(x4)  # 下采样 [1/16]

                # 应用边界增强（如果启用）
                if self.enable_boundary and self.boundary_module is not None:
                    x5, edge_map_from_boundary = self.boundary_module(x5)
                    if edge_map is None:  # 如果JanusPro没有提供边缘图，使用边界模块生成的
                        edge_map = edge_map_from_boundary

                # 瓶颈层应用跨模态注意力
                x5 = self.cross_attn(x5, janus_features[3])

                # 解码器路径 - 多层级特征融合
                x = self.unet.up1(x5, x4)
                x = self.feature_fusion1(x, janus_features[2])  # 融合第1层

                x = self.unet.up2(x, x3)
                x = self.feature_fusion2(x, janus_features[1])  # 融合第2层

                x = self.unet.up3(x, x2)
                x = self.feature_fusion3(x, janus_features[0])  # 融合第3层

                x = self.unet.up4(x, x1)
                # 最后一层融合 - 如果有合适的特征
                if len(janus_features) > 4:
                    x = self.feature_fusion4(x, janus_features[4])

                # 输出层
                output = self.unet.outc(x)

            else:
                # 如果不使用多层级融合，回退到原有的实现
                # 获取U-Net瓶颈特征
                unet_bottleneck = self.unet.get_bottleneck_feature(images)

                # 如果有JanusPro特征，应用特征门控和注意力机制
                injected_features = None
                if self.enable_janus and janus_features is not None and self.feature_gating is not None and self.cross_attn is not None:
                    try:
                        # 选择最后一层特征
                        if isinstance(janus_features, list):
                            janus_feat = janus_features[-1]
                        else:
                            janus_feat = janus_features

                        # 应用特征门控
                        janus_gated = self.feature_gating(unet_bottleneck, janus_feat)

                        # 应用跨模态注意力
                        attn_features = self.cross_attn(unet_bottleneck, janus_gated)

                        # 应用边界增强（如果启用）
                        if self.enable_boundary and self.boundary_module is not None:
                            enhanced_features, edge_map = self.boundary_module(attn_features)
                            injected_features = enhanced_features
                        else:
                            injected_features = attn_features

                    except Exception as e:
                        logger.error(f"特征融合失败: {str(e)}")
                        import traceback as tb
                        logger.error(tb.format_exc())
                        injected_features = None

                elif self.enable_boundary and self.boundary_module is not None:
                    # 仅使用边界增强
                    enhanced_features, edge_map = self.boundary_module(unet_bottleneck)
                    injected_features = enhanced_features

                # 使用U-Net进行前向传播（带特征注入）
                if injected_features is not None:
                    output = self.unet.forward_with_injected_features(images, injected_features)
                else:
                    output = self.unet(images)

        except Exception as e:
            logger.error(f"前向传播出错: {str(e)}")
            import traceback as tb
            logger.error(tb.format_exc())
            # 引发异常，终止训练
            raise

        # 确保输出尺寸正确
        if output.shape[2:] != (height, width):
            output = F.interpolate(
                output,
                size=(height, width),
                mode='bilinear',
                align_corners=True
            )

        # 返回分割结果和边缘图（如果有）
        if edge_map is not None:
            # 调整边缘图到原始尺寸
            edge_map = F.interpolate(
                edge_map,
                size=(height, width),
                mode='bilinear',
                align_corners=True
            )
            return output, edge_map
        else:
            return output