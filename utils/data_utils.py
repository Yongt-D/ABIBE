# utils/data_utils.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BuildingDataset(Dataset):
    def __init__(self, image_dir, label_dir, text_features_dir=None,
                 phase='train', image_size=512, augmentation=False):
        # 使用Path处理路径，增强跨平台兼容性
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.text_features_dir = Path(text_features_dir) if text_features_dir else None
        self.phase = phase
        self.image_size = image_size
        self.augmentation = augmentation and phase == 'train'

        # 检查目录是否存在
        if not self.image_dir.exists():
            logger.error(f"图像目录不存在: {self.image_dir}")
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")

        if not self.label_dir.exists():
            logger.error(f"标签目录不存在: {self.label_dir}")
            raise FileNotFoundError(f"标签目录不存在: {self.label_dir}")

        if self.text_features_dir and not self.text_features_dir.exists():
            logger.warning(f"文本特征目录不存在: {self.text_features_dir}, 将不使用文本特征")
            self.text_features_dir = None

        # 获取图像文件列表
        self.image_files = sorted([f.name for f in self.image_dir.glob('*.tif')] +
                                  [f.name for f in self.image_dir.glob('*.jpg')] +
                                  [f.name for f in self.image_dir.glob('*.png')])

        logger.info(f"找到 {len(self.image_files)} 个图像文件")
        if len(self.image_files) > 0:
            logger.info(f"图像文件示例: {self.image_files[:5]}")

        # 获取标签文件列表
        self.label_files = sorted([f.name for f in self.label_dir.glob('*.*')])

        logger.info(f"找到 {len(self.label_files)} 个标签文件")
        if len(self.label_files) > 0:
            logger.info(f"标签文件示例: {self.label_files[:5]}")

        # 检查标签是否存在
        self.validate_files()

        # 基本转换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 标签转换
        self.label_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        logger.info(
            f"创建{phase}数据集，包含{len(self.image_files)}个样本，数据增强={'启用' if augmentation else '禁用'}")

        # 打印一些样本的标签形状
        if len(self.image_files) > 0:
            sample_img_file = self.image_files[0]
            sample_label_file = self.image_to_label_map[sample_img_file]
            try:
                sample_label_path = self.label_dir / sample_label_file
                sample_mask = Image.open(sample_label_path).convert('L')
                sample_mask_tensor = self.label_transform(sample_mask)
                logger.info(f"样本标签形状: {sample_mask_tensor.shape}")
            except Exception as e:
                logger.error(f"检查样本标签形状时出错: {str(e)}")

    def validate_files(self):
        """验证文件是否存在，处理不同格式的标签文件"""
        valid_images = []
        image_to_label_map = {}

        # 创建基于文件名的标签映射（不考虑扩展名）
        label_basename_map = {}
        for label_file in self.label_files:
            # 获取不带扩展名的文件名
            basename = Path(label_file).stem
            label_basename_map[basename] = label_file

        # 记录找不到对应标签的图像
        missing_labels = []

        for img_file in self.image_files:
            # 获取不带扩展名的文件名
            img_basename = Path(img_file).stem

            # 查找对应的标签文件
            if img_basename in label_basename_map:
                # 找到了匹配的标签文件
                label_file = label_basename_map[img_basename]

                # 检查文本特征（如果需要）
                if self.text_features_dir:
                    text_feature_path = self.text_features_dir / f"{img_basename}.pt"

                    if not text_feature_path.exists():
                        logger.warning(f"找不到文本特征文件: {text_feature_path}，跳过样本 {img_file}")
                        continue

                # 所有检查通过，添加到有效图像列表
                valid_images.append(img_file)
                image_to_label_map[img_file] = label_file
            else:
                # 找不到匹配的标签文件
                missing_labels.append(img_file)

        # 记录找不到标签的图像数量
        if missing_labels:
            logger.warning(f"找不到对应标签的图像数量: {len(missing_labels)}")
            if len(missing_labels) < 10:
                logger.warning(f"找不到标签的图像: {missing_labels}")
            else:
                logger.warning(f"找不到标签的图像示例: {missing_labels[:10]}...")

        # 更新图像列表和标签映射
        self.image_files = valid_images
        self.image_to_label_map = image_to_label_map

        if len(valid_images) == 0:
            logger.error(f"数据集{self.phase}没有有效样本！请检查目录结构和文件名是否匹配")
            # 打印目录中实际存在的文件，帮助调试
            logger.error(f"图像目录 {self.image_dir} 中的文件: {list(self.image_dir.glob('*.*'))[:10]}")
            logger.error(f"标签目录 {self.label_dir} 中的文件: {list(self.label_dir.glob('*.*'))[:10]}")
            if self.text_features_dir:
                logger.error(
                    f"文本特征目录 {self.text_features_dir} 中的文件: {list(self.text_features_dir.glob('*.*'))[:10]}")

    def __len__(self):
        return len(self.image_files)

    def apply_augmentation(self, image, mask):
        """应用数据增强"""
        # 转换为PIL图像以便应用变换
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)

        if not isinstance(mask, Image.Image):
            mask = transforms.ToPILImage()(mask)

        # 随机水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 随机垂直翻转
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # 随机旋转
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # 随机亮度和对比度变化 (只对图像应用)
        if random.random() > 0.5:
            brightness_factor = 1.0 + random.uniform(-0.2, 0.2)
            contrast_factor = 1.0 + random.uniform(-0.2, 0.2)
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)

        return image, mask

    def __getitem__(self, idx):
        """获取一个样本"""
        try:
            img_file = self.image_files[idx]

            # 获取图像路径
            img_path = self.image_dir / img_file

            # 获取标签路径 - 使用映射找到对应的标签文件
            label_file = self.image_to_label_map[img_file]
            label_path = self.label_dir / label_file

            # 读取图像和标签
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('L')  # 转换为灰度图

            # 打印标签的最小值和最大值（用于调试）
            mask_array = np.array(mask)
            min_val = mask_array.min()
            max_val = mask_array.max()

            # 数据增强
            if self.augmentation:
                # 正常增强图像和掩码
                image, mask = self.apply_augmentation(image, mask)

                # 转换
                image = self.transform(image)
                mask = self.label_transform(mask)
                mask = (mask > 0.5).float()
            else:
                # 不进行增强，直接转换
                image = self.transform(image)
                mask = self.label_transform(mask)
                mask = (mask > 0.5).float()

            # 确保mask是[1, H, W]形状
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

            # 读取文本特征（如果有）
            text_features = None
            if self.text_features_dir:
                # 获取不带扩展名的文件名
                img_basename = Path(img_file).stem
                text_feature_path = self.text_features_dir / f"{img_basename}.pt"

                if text_feature_path.exists():
                    try:
                        text_features = torch.load(text_feature_path, map_location='cpu')

                        # 如果特征是字典，获取适当的键
                        if isinstance(text_features, dict) and 'features' in text_features:
                            text_features = text_features['features']

                        # 确保是正确的张量形状
                        if len(text_features.shape) > 1 and text_features.shape[0] > 1:
                            # 如果是批量特征，取第一个
                            text_features = text_features[0]

                    except Exception as e:
                        logger.warning(f"加载文本特征失败 {text_feature_path}: {str(e)}")
                        text_features = None

            return image, text_features, mask

        except Exception as e:
            logger.error(f"加载样本时出错 idx={idx}, file={self.image_files[idx]}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 返回一个空的样本，避免整个数据集出错
            empty_image = torch.zeros(3, self.image_size, self.image_size)
            empty_mask = torch.zeros(1, self.image_size, self.image_size)
            return empty_image, None, empty_mask