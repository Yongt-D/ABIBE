# models/backbone/janus_pro_extractor.py
import torch
import torch.nn as nn
from pathlib import Path
from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
from PIL import Image
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class JanusProExtractor(nn.Module):
    """JanusPro视觉语言模型特征提取器"""

    def __init__(self, model_path, feature_layers=[3, 6, 9, 12], freeze_model=True):
        super().__init__()
        self.model_path = Path(model_path)
        self.feature_layers = feature_layers
        self.output_dims = [256, 512, 1024, 2048]  # 每层输出维度
        self.use_text = True  # 添加此属性，表示支持文本特征

        logger.info(f"初始化JanusPro从路径: {str(self.model_path)}")

        # 加载JanusPro处理器和模型
        try:
            self.processor = VLChatProcessor.from_pretrained(str(self.model_path), legacy=False)
            self.model = MultiModalityCausalLM.from_pretrained(str(self.model_path), trust_remote_code=True)
            logger.info("成功加载JanusPro模型和处理器")

            # 将模型移至GPU并设置为评估模式
            self.model = self.model.to(torch.bfloat16).cuda().eval()
            logger.info("JanusPro模型已移至GPU并设置为评估模式")
        except Exception as e:
            logger.error(f"加载JanusPro模型失败: {str(e)}")
            raise

        # 根据参数决定是否冻结
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("JanusPro模型参数已冻结")
        else:
            logger.info("JanusPro模型参数可训练")

        # 特征转换网络
        self.image_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, dim, kernel_size=1),
                nn.GroupNorm(min(32, dim), dim),
                nn.ReLU()
            ) for dim in self.output_dims
        ])

        # 边缘检测器 - 用于边界增强
        self.edge_detector = nn.Sequential(
            nn.Conv2d(self.output_dims[-1], 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 设置内存管理
        self.max_batch_size = 2  # 限制批量大小以防止OOM

    def extract_text_features(self, text):
        """提取文本特征"""
        with torch.no_grad():
            try:
                # 创建对话格式
                conversation = [
                    {"role": "<|User|>", "content": text, "images": []},
                    {"role": "<|Assistant|>", "content": ""}
                ]

                # 使用处理器准备输入
                prepare_inputs = self.processor(
                    conversations=conversation,
                    images=[],
                    force_batchify=True
                ).to(self.model.device)

                # 提取特征
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

                # 处理不同的返回格式
                if isinstance(inputs_embeds, dict):
                    text_features = inputs_embeds.get('input_embeds',
                                                      inputs_embeds.get('text_embeds', inputs_embeds))
                else:
                    text_features = inputs_embeds

                # 平均池化
                text_features = torch.mean(text_features, dim=1)  # [B, D]

                # 确保数据类型是float32
                text_features = text_features.to(torch.float32)

                return text_features

            except Exception as e:
                logger.error(f"文本特征提取出错: {str(e)}")
                # 返回零张量作为回退
                return torch.zeros(1, 2048, device=self.model.device)

    def _tensor_to_pil_images(self, images):
        """将张量转换为PIL图像列表"""
        if isinstance(images, torch.Tensor):
            images_pil = []
            for i in range(images.shape[0]):
                try:
                    # 先将BFloat16转换为Float32
                    img = images[i]
                    if img.dtype == torch.bfloat16:
                        img = img.float()

                    # 将tensor转换为numpy数组
                    img = img.cpu().detach().numpy()

                    # 将[C, H, W]转换为[H, W, C]
                    img = np.transpose(img, (1, 2, 0))

                    # 确保值范围为0-255
                    if img.max() <= 1.0:
                        img = (img * 255.0).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                    # 确保图像有3个通道
                    if img.shape[2] == 1:  # 如果是单通道，转为3通道
                        img = np.repeat(img, 3, axis=2)
                    elif img.shape[2] > 3:  # 如果通道数超过3，只取前3个通道
                        img = img[:, :, :3]

                    # 检查图像大小是否为空
                    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                        logger.warning(f"遇到空图像，跳过第 {i} 个图像")
                        # 创建一个默认的小图像
                        img = np.zeros((32, 32, 3), dtype=np.uint8)

                    # 转换为PIL图像
                    pil_img = Image.fromarray(img)
                    images_pil.append(pil_img)
                except Exception as e:
                    logger.error(f"转换第 {i} 个图像时出错: {str(e)}")
                    logger.error(f"图像形状: {images[i].shape}")
                    logger.error(f"图像数据类型: {images[i].dtype}")
                    # 创建一个默认的替代图像
                    pil_img = Image.new('RGB', (32, 32), color='black')
                    images_pil.append(pil_img)

            return images_pil
        else:
            # 如果已经是PIL图像列表，直接返回
            return images

    def extract_features_from_vision_blocks(self, x):
        """从视觉Transformer块中提取多级特征"""
        features = []

        # 访问视觉模型的层
        vision_tower = self.model.vision_model.vision_tower

        # 检查我们要提取的层级是否有效
        max_layer = len(vision_tower.blocks) - 1
        valid_layers = [min(layer, max_layer) for layer in self.feature_layers]

        # 通过所有块，提取特定层的特征
        layer_outputs = {}
        x_current = x

        for i, block in enumerate(vision_tower.blocks):
            x_current = block(x_current)
            if i in valid_layers:
                layer_outputs[i] = x_current

        # 按照指定顺序收集特征
        for layer in valid_layers:
            if layer in layer_outputs:
                features.append(layer_outputs[layer])
            else:
                # 如果指定层不存在，使用最后一层
                features.append(x_current)

        return features

    def process_extracted_features(self, features, batch_size, orig_h, orig_w):
        """处理从视觉模型提取的特征，转换为空间特征图"""
        processed_features = []

        for i, feature in enumerate(features):
            try:
                # 确保数据类型是float32
                feature = feature.to(torch.float32)

                # 检查是否需要调整形状
                if len(feature.shape) == 3:  # [B, L, C]
                    B, L, C = feature.shape
                    H = W = int((L - 1) ** 0.5)  # 减去CLS token

                    # 为每个特征级别计算适当的空间尺寸
                    feature_size = min(32, max(16, 2 ** (i + 4)))

                    # 如果特征长度与预期的不匹配，使用自适应池化
                    if H * W + 1 != L:
                        # 使用简单的重塑，忽略正方形约束
                        spatial_feature = feature[:, 1:, :].reshape(B, -1, C)
                        spatial_feature = spatial_feature.permute(0, 2, 1)
                        spatial_feature = spatial_feature.reshape(B, C, -1, 1)

                        # 使用自适应池化创建所需尺寸的特征图
                        spatial_feature = F.adaptive_avg_pool2d(
                            spatial_feature,
                            (feature_size, feature_size)
                        )
                    else:
                        # 正确的重塑：从[B, L, C]到[B, C, H, W]
                        # 首先移除CLS令牌
                        spatial_feature = feature[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)

                        # 调整到所需的特征大小
                        spatial_feature = F.interpolate(
                            spatial_feature,
                            size=(feature_size, feature_size),
                            mode='bilinear',
                            align_corners=True
                        )

                    # 应用特征变换器
                    if i < len(self.image_transformers):
                        transformed = self.image_transformers[i](spatial_feature)
                        processed_features.append(transformed)
                    else:
                        # 如果没有对应的变换器，直接使用1x1卷积
                        conv = nn.Conv2d(C, self.output_dims[i % len(self.output_dims)], kernel_size=1).to(
                            feature.device)
                        transformed = conv(spatial_feature)
                        processed_features.append(transformed)
                else:
                    logger.warning(f"特征形状不符合预期: {feature.shape}")
                    # 创建替代特征
                    dummy_feature = torch.zeros(
                        batch_size, self.output_dims[i % len(self.output_dims)], 16, 16,
                        device=feature.device
                    )
                    processed_features.append(dummy_feature)

            except Exception as e:
                logger.error(f"处理提取的特征时出错 (层 {i}): {str(e)}")
                # 创建替代特征
                dummy_feature = torch.zeros(
                    batch_size, self.output_dims[i % len(self.output_dims)], 16, 16,
                    device=features[0].device if features else self.model.device
                )
                processed_features.append(dummy_feature)

        # 如果没有足够的特征，添加虚拟特征
        while len(processed_features) < 4:
            idx = len(processed_features)
            dummy_feature = torch.zeros(
                batch_size, self.output_dims[idx % len(self.output_dims)], 16, 16,
                device=features[0].device if features else self.model.device
            )
            processed_features.append(dummy_feature)

        # 提取边界信息
        try:
            edge_map = self.edge_detector(processed_features[-1])
        except Exception as e:
            logger.error(f"提取边界信息时出错: {str(e)}")
            edge_map = torch.zeros(batch_size, 1, 16, 16, device=processed_features[-1].device)

        # 确保所有特征都调整到原始图像尺寸的对应分辨率
        final_features = []
        for i, feat in enumerate(processed_features):
            # 针对每个特征级别计算合适的目标尺寸
            scale_factor = 2 ** (3 - i)  # 最深层特征分辨率最低
            target_h = max(8, orig_h // scale_factor)
            target_w = max(8, orig_w // scale_factor)

            # 调整特征大小
            try:
                resized_feat = F.interpolate(
                    feat,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=True
                )
                final_features.append(resized_feat)
            except Exception as e:
                logger.error(f"调整特征大小时出错 (层 {i}): {str(e)}")
                # 使用原始特征
                final_features.append(feat)

        return final_features, edge_map

    def extract_image_features(self, images):
        """提取多层图像特征"""
        batch_size, _, orig_h, orig_w = images.shape

        # 对于较大的批次，拆分处理以避免OOM
        if batch_size > self.max_batch_size:
            # 拆分为更小的批次
            all_features = []
            all_edge_maps = []

            for start_idx in range(0, batch_size, self.max_batch_size):
                end_idx = min(start_idx + self.max_batch_size, batch_size)
                sub_batch = images[start_idx:end_idx]

                # 递归处理子批次
                sub_features, sub_edge_map = self.extract_image_features(sub_batch)

                all_features.append(sub_features)
                all_edge_maps.append(sub_edge_map)

            # 合并结果
            combined_features = []
            for i in range(len(all_features[0])):
                feature_batch = torch.cat([fs[i] for fs in all_features], dim=0)
                combined_features.append(feature_batch)

            combined_edge_map = torch.cat(all_edge_maps, dim=0)

            return combined_features, combined_edge_map

        with torch.no_grad():
            try:
                # 确保图像数据类型正确 - 始终使用float32处理图像
                images = images.float()  # 强制转换为float32

                # 将输入转换为PIL图像列表
                images_pil = self._tensor_to_pil_images(images)

                if not images_pil or len(images_pil) == 0:
                    logger.error("未能转换任何有效图像")
                    # 创建零填充特征和边界图
                    dummy_features = [torch.zeros((batch_size, dim, 16, 16), device=self.model.device)
                                      for dim in self.output_dims]
                    dummy_edge_map = torch.zeros((batch_size, 1, 16, 16), device=self.model.device)
                    return dummy_features, dummy_edge_map

                # 预处理图像
                try:
                    inputs = self.processor.image_processor(images=images_pil, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(self.model.device)

                    # 确保pixel_values是正确的数据类型
                    if hasattr(self.model, 'vision_model'):
                        model_dtype = next(self.model.vision_model.parameters()).dtype
                        pixel_values = pixel_values.to(model_dtype)
                except Exception as e:
                    logger.error(f"图像预处理错误: {str(e)}")
                    # 创建零填充特征和边界图
                    dummy_features = [torch.zeros((batch_size, dim, 16, 16), device=self.model.device)
                                      for dim in self.output_dims]
                    dummy_edge_map = torch.zeros((batch_size, 1, 16, 16), device=self.model.device)
                    return dummy_features, dummy_edge_map

                # 直接使用vision_tower的各个组件处理图像
                vision_tower = self.model.vision_model.vision_tower

                # 提取特征：首先应用patch_embed
                x = vision_tower.patch_embed(pixel_values)

                # 应用位置编码（如果有的话）
                if hasattr(vision_tower, 'pos_embed'):
                    x = x + vision_tower.pos_embed

                # 应用dropout
                x = vision_tower.pos_drop(x)

                # 从vision blocks中提取多级特征
                multi_scale_features = self.extract_features_from_vision_blocks(x)

                # 处理提取的特征
                processed_features, edge_map = self.process_extracted_features(
                    multi_scale_features, batch_size, orig_h, orig_w
                )

                return processed_features, edge_map

            except Exception as e:
                logger.error(f"提取图像特征时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # 创建零填充特征和边界图作为回退
                dummy_features = [torch.zeros((batch_size, dim, 16, 16), device=self.model.device)
                                  for dim in self.output_dims]
                dummy_edge_map = torch.zeros((batch_size, 1, 16, 16), device=self.model.device)
                return dummy_features, dummy_edge_map

    def forward(self, images, text_features=None):
        """前向传播"""
        try:
            # 标准化图像值（如果需要）
            if isinstance(images, torch.Tensor) and images.max() > 1.1:
                images = images / 255.0

            # 提取图像特征和边界信息
            image_features, edge_map = self.extract_image_features(images)

            # 处理文本特征
            processed_text_features = None
            if text_features is not None and self.use_text:
                if isinstance(text_features, str):
                    processed_text_features = self.extract_text_features(text_features)
                else:
                    # 确保文本特征是float32类型
                    processed_text_features = text_features.to(torch.float32)

            return {
                'image_features': image_features,
                'text_features': processed_text_features,
                'edge_map': edge_map
            }

        except Exception as e:
            logger.error(f"JanusProExtractor前向传播出错: {str(e)}")
            # 创建零填充特征和边界图作为回退
            batch_size = images.shape[0]
            dummy_features = [torch.zeros((batch_size, dim, 16, 16), device=self.model.device)
                              for dim in self.output_dims]
            dummy_edge_map = torch.zeros((batch_size, 1, 16, 16), device=self.model.device)

            return {
                'image_features': dummy_features,
                'text_features': None,
                'edge_map': dummy_edge_map
            }