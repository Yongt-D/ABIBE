# scripts/test.py
import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import sys
import random
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型和工具
from models.building_extraction_model import BuildingExtractionModel
from utils.visualization import denormalize, save_batch_results, create_overlay_image
from utils.metrics import calculate_metrics

# 创建日志目录
os.makedirs('logs', exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试建筑提取模型')
    parser.add_argument('--config', type=str, default='configs/enhanced_loss_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='测试设备')
    parser.add_argument('--output', type=str, default='results/predictions',
                        help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='预测二值化阈值')
    parser.add_argument('--vis-samples', type=int, default=20,
                        help='可视化样本数量')
    parser.add_argument('--test-train', action='store_true',
                        help='是否测试训练数据')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')

    # 允许在命令行直接设置模型配置
    parser.add_argument('--enable-janus', action='store_true',
                        help='启用JanusPro特征')
    parser.add_argument('--disable-janus', action='store_true',
                        help='禁用JanusPro特征')
    parser.add_argument('--use-text', action='store_true',
                        help='使用文本特征')
    parser.add_argument('--disable-text', action='store_true',
                        help='禁用文本特征')
    parser.add_argument('--force-config', action='store_true',
                        help='强制使用配置文件中的设置，忽略检查点中的设置')

    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {str(e)}")
        raise


def update_config_from_args(config, args):
    """根据命令行参数更新配置"""
    # 更新模型配置
    if args.enable_janus:
        config['model']['enable_janus'] = True
        logger.info("通过命令行参数启用JanusPro特征")
    elif args.disable_janus:
        config['model']['enable_janus'] = False
        logger.info("通过命令行参数禁用JanusPro特征")

    if args.use_text:
        config['model']['use_text'] = True
        logger.info("通过命令行参数启用文本特征")
    elif args.disable_text:
        config['model']['use_text'] = False
        logger.info("通过命令行参数禁用文本特征")

    # 禁用边界感知
    config['model']['enable_boundary_awareness'] = False

    return config


def load_model(config, checkpoint_path, device, args):
    """加载模型"""
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 打印检查点中的轮次信息
        if 'epoch' in checkpoint:
            logger.info(f"检查点轮次: {checkpoint['epoch']}")

        # 使用保存的配置还是强制使用配置文件
        if 'config' in checkpoint and not args.force_config:
            saved_config = checkpoint['config']
            # 合并配置
            for key in saved_config:
                if key not in config:
                    config[key] = saved_config[key]
                elif key == 'model':
                    # 确保模型配置一致性
                    for model_key in saved_config[key]:
                        # 只合并非特性开关的配置
                        if model_key not in ['enable_janus', 'use_text']:
                            config[key][model_key] = saved_config[key][model_key]
        else:
            if args.force_config:
                logger.info("强制使用配置文件中的设置，忽略检查点中的设置")
            else:
                logger.warning("检查点中没有配置信息，使用命令行提供的配置")

        # 输出当前模型配置
        logger.info(f"模型配置: JanusPro特征={'启用' if config['model']['enable_janus'] else '禁用'}, "
                    f"文本特征={'启用' if config['model']['use_text'] else '禁用'}")

        # 创建模型
        model = BuildingExtractionModel(config).to(device)

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            # 检查是否需要忽略某些参数
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']

            # 如果模型设置与加载的检查点不同，可能需要忽略某些参数
            if (config['model']['enable_janus'] != checkpoint.get('config', {}).get('model', {}).get('enable_janus',
                                                                                                     True) or
                    config['model']['use_text'] != checkpoint.get('config', {}).get('model', {}).get('use_text', True)):
                # 过滤需要的参数
                filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                logger.info(f"模型配置不同，加载部分权重: {len(filtered_dict)}/{len(pretrained_dict)}")
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(pretrained_dict)

            # 打印检查点中的详细信息
            logger.info(f"从状态字典加载模型权重，检查点epoch: {checkpoint.get('epoch', 'unknown')}")
            if 'val_loss' in checkpoint:
                logger.info(f"检查点验证损失: {checkpoint['val_loss']:.6f}")
            if 'current_epoch_iou' in checkpoint:
                logger.info(f"检查点当前IoU: {checkpoint['current_epoch_iou']:.4f}")
        else:
            # 尝试直接加载
            model.load_state_dict(checkpoint)
            logger.info("直接从检查点加载模型权重")

        # 记录模型信息
        logger.info(f"模型已加载，设备: {device}")
        if 'best_metric' in checkpoint:
            logger.info(f"检查点最佳IoU: {checkpoint['best_metric']:.4f}")

        # 设置为评估模式
        model.eval()
        return model
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


class SimpleDataset(torch.utils.data.Dataset):
    """简单的数据集类，直接加载图像和标签文件"""

    def __init__(self, image_dir, label_dir, text_features_dir=None, image_size=512):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.text_features_dir = Path(text_features_dir) if text_features_dir else None
        self.image_size = image_size

        # 获取所有图像文件
        self.image_files = []
        for ext in ['.tif', '.jpg', '.png', '.jpeg']:
            self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))

        # 排序文件以确保一致的顺序
        self.image_files = sorted(self.image_files)
        logger.info(f"找到 {len(self.image_files)} 个图像文件")

        if self.text_features_dir and self.text_features_dir.exists():
            logger.info(f"使用文本特征目录: {self.text_features_dir}")
        else:
            logger.info("不使用文本特征")

        # 使用正确的转换库
        from torchvision import transforms

        # 标准转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 标签转换
        self.label_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        """返回数据集的长度"""
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"无法读取图像: {img_path}")
            # 返回空图像
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # 转换图像颜色通道
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 寻找对应的标签文件
        basename = img_path.stem
        label_found = False
        label_path = None

        # 查找不同扩展名的标签文件
        for ext in ['.tif', '.jpg', '.png', '.jpeg']:
            potential_path = self.label_dir / f"{basename}{ext}"
            if potential_path.exists():
                label_path = potential_path
                label_found = True
                break

        # 如果找不到标签文件，使用空标签
        if not label_found:
            logger.warning(f"找不到图像 {img_path} 对应的标签文件")
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            try:
                # 读取标签
                mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    logger.warning(f"标签文件读取失败: {label_path}")
                    mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            except Exception as e:
                logger.error(f"处理标签时出错 {label_path}: {str(e)}")
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # 加载文本特征（如果有）
        text_features = None
        if self.text_features_dir:
            text_feature_path = self.text_features_dir / f"{basename}.pt"
            if text_feature_path.exists():
                try:
                    text_features = torch.load(text_feature_path)
                except Exception as e:
                    logger.error(f"加载文本特征失败 {text_feature_path}: {str(e)}")

        # 使用PIL的Image进行转换
        from PIL import Image
        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray(mask)

        # 应用转换
        img_tensor = self.transform(img_pil)
        mask_tensor = self.label_transform(mask_pil)

        # 确保标签是二值的
        if mask_tensor.max() > 0.5:
            mask_tensor = (mask_tensor > 0.5).float()

        # 对于纯UNet模式，返回一个空张量而不是None
        if text_features is None:
            # 创建一个空张量作为占位符
            text_features = torch.zeros(0)  # 或者使用torch.tensor([])

        return img_tensor, text_features, mask_tensor, basename


def test_model(model, test_dataset, device, output_dir, threshold=0.5, batch_size=4, num_workers=0):
    """测试模型，对所有样本进行处理，包括无建筑物样本"""
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建预测目录
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # 创建可视化目录
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # 禁用pin_memory
    )

    # 设置模型为评估模式
    model.eval()

    # 初始化指标
    metrics = {
        'total_samples': 0,
        'samples_with_buildings': 0,
        'samples_without_buildings': 0,
        'samples_with_predictions': 0,
        'true_positive_pixels': 0,
        'true_negative_pixels': 0,
        'false_positive_pixels': 0,
        'false_negative_pixels': 0,
        'correct_empty_predictions': 0,  # 正确预测无建筑物的样本数
        'false_alarm_samples': 0,  # 错误预测有建筑物的样本数（实际无建筑）
        # 添加Dice指标相关的统计
        'dice_intersection': 0,  # Dice计算中的交集
        'dice_pred_sum': 0,  # Dice计算中的预测像素总数
        'dice_target_sum': 0  # Dice计算中的目标像素总数
    }

    # 存储一些样本用于可视化
    samples_to_visualize = []

    # 处理每个批次
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                # 获取样本
                images, text_features, masks, basenames = batch

                # 移动到设备
                images = images.to(device)
                if text_features is not None and any(x is not None for x in text_features):
                    # 处理文本特征，将None转换为零张量
                    valid_text_features = []
                    for tf in text_features:
                        if tf is not None:
                            valid_text_features.append(tf.to(device))
                        else:
                            # 创建零张量作为占位符
                            valid_text_features.append(torch.zeros(2048, device=device))
                    text_features = torch.stack(valid_text_features)
                else:
                    text_features = None

                # 记录批次大小
                batch_size = images.shape[0]
                metrics['total_samples'] += batch_size

                # 前向传播
                outputs = model(images, text_features)

                # 处理模型输出为元组的情况
                if isinstance(outputs, tuple):
                    main_output = outputs[0]  # 主输出是元组的第一个元素
                    # 如果需要使用edge_map，可以在这里获取: edge_map = outputs[1]
                else:
                    main_output = outputs

                # 应用sigmoid获取概率
                probs = torch.sigmoid(main_output)

                # 二值化预测
                preds = (probs > threshold).float()

                # 逐样本处理
                for i in range(batch_size):
                    # 获取单个样本
                    mask = masks[i:i + 1]
                    pred = preds[i:i + 1]
                    prob = probs[i:i + 1]
                    basename = basenames[i]

                    # 二值化标签
                    binary_mask = (mask > 0.5).float()

                    # 检查是否包含建筑物（真实标签）
                    has_buildings = binary_mask.sum().item() > 0

                    # 检查是否预测有建筑物
                    has_predictions = pred.sum().item() > 0

                    # 处理有建筑物的样本
                    if has_buildings:
                        metrics['samples_with_buildings'] += 1

                        # 移动标签到设备
                        binary_mask = binary_mask.to(device)

                        if has_predictions:
                            metrics['samples_with_predictions'] += 1

                        # 计算像素级指标
                        intersection = (pred * binary_mask).sum().item()
                        metrics['true_positive_pixels'] += intersection

                        false_positive = (pred * (1 - binary_mask)).sum().item()
                        metrics['false_positive_pixels'] += false_positive

                        false_negative = ((1 - pred) * binary_mask).sum().item()
                        metrics['false_negative_pixels'] += false_negative

                        # 计算真阴性像素
                        true_negative = ((1 - pred) * (1 - binary_mask)).sum().item()
                        metrics['true_negative_pixels'] += true_negative

                        # 计算Dice相关统计 - 针对有建筑物的样本
                        metrics['dice_intersection'] += intersection
                        metrics['dice_pred_sum'] += pred.sum().item()
                        metrics['dice_target_sum'] += binary_mask.sum().item()

                    # 处理无建筑物的样本
                    else:
                        metrics['samples_without_buildings'] += 1

                        # 如果预测也没有建筑物，则为正确的空预测
                        if not has_predictions:
                            metrics['correct_empty_predictions'] += 1
                            # 对于无建筑物且无预测的样本，添加到Dice计算中
                            # 这种情况下，交集=0，预测=0，目标=0，Dice应该是1.0
                            # 但为了保持计算的一致性，我们只在有像素的情况下计算Dice
                        else:
                            # 错误地预测有建筑物（误报）
                            metrics['false_alarm_samples'] += 1
                            # 计算误检像素数
                            false_positive_pixels = pred.sum().item()
                            metrics['false_positive_pixels'] += false_positive_pixels

                            # 对于误报样本，添加到Dice计算中
                            # 交集=0，预测>0，目标=0
                            metrics['dice_pred_sum'] += false_positive_pixels
                            # dice_target_sum 保持不变（目标为0）

                    # 保存预测结果
                    pred_image = pred[0, 0].cpu().numpy()
                    prob_image = prob[0, 0].cpu().numpy()

                    # 保存二进制预测
                    pred_path = pred_dir / f"{basename}_pred.png"
                    cv2.imwrite(str(pred_path), (pred_image * 255).astype(np.uint8))

                    # 保存概率预测
                    prob_path = pred_dir / f"{basename}_prob.png"
                    cv2.imwrite(str(prob_path), (prob_image * 255).astype(np.uint8))

                    # 如果需要可视化且样本数量不超过10，且是随机采样
                    if len(samples_to_visualize) < 10 and (has_buildings or has_predictions or random.random() < 0.1):
                        # 存储样本用于稍后可视化
                        samples_to_visualize.append({
                            'image': images[i:i + 1].cpu(),
                            'mask': binary_mask.cpu(),
                            'pred': pred.cpu(),
                            'prob': prob.cpu(),
                            'basename': basename,
                            'has_buildings': has_buildings,
                            'has_predictions': has_predictions
                        })
            except Exception as e:
                logger.error(f"处理批次 {batch_idx} 时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

    # 计算性能指标
    results = {}
    results['total_samples'] = metrics['total_samples']
    results['samples_with_buildings'] = metrics['samples_with_buildings']
    results['samples_without_buildings'] = metrics['samples_without_buildings']
    results['samples_with_predictions'] = metrics['samples_with_predictions']
    results['correct_empty_predictions'] = metrics['correct_empty_predictions']
    results['false_alarm_samples'] = metrics['false_alarm_samples']

    # 计算IoU
    tp = metrics['true_positive_pixels']
    fp = metrics['false_positive_pixels']
    fn = metrics['false_negative_pixels']

    if tp + fp + fn > 0:
        results['iou'] = tp / (tp + fp + fn)
    else:
        results['iou'] = 0.0

    # 计算Dice系数
    dice_intersection = metrics['dice_intersection']
    dice_pred_sum = metrics['dice_pred_sum']
    dice_target_sum = metrics['dice_target_sum']

    if dice_pred_sum + dice_target_sum > 0:
        results['dice'] = (2.0 * dice_intersection) / (dice_pred_sum + dice_target_sum)
    else:
        # 如果预测和目标都为0（即没有任何像素），Dice系数应该是1.0
        results['dice'] = 1.0 if metrics['total_samples'] > 0 else 0.0

    # 计算精确度和召回率
    if tp + fp > 0:
        results['precision'] = tp / (tp + fp)
    else:
        results['precision'] = 0.0

    if tp + fn > 0:
        results['recall'] = tp / (tp + fn)
    else:
        results['recall'] = 0.0

    # 计算F1分数
    if results['precision'] + results['recall'] > 0:
        results['f1'] = (2 * results['precision'] * results['recall']) / (results['precision'] + results['recall'])
    else:
        results['f1'] = 0.0

    # 计算像素准确率（新增）
    total_pixels = tp + fp + fn + metrics['true_negative_pixels']
    if total_pixels > 0:
        results['pixel_accuracy'] = (tp + metrics['true_negative_pixels']) / total_pixels
    else:
        results['pixel_accuracy'] = 0.0

    # 计算样本级别的准确率
    correct_samples = metrics['samples_with_predictions'] + metrics['correct_empty_predictions']
    results['sample_accuracy'] = correct_samples / max(1, metrics['total_samples'])

    # 计算无建筑物样本的准确率
    if metrics['samples_without_buildings'] > 0:
        results['empty_accuracy'] = metrics['correct_empty_predictions'] / metrics['samples_without_buildings']
    else:
        results['empty_accuracy'] = 1.0  # 如果没有无建筑物样本，则设为1.0

    # 输出结果
    logger.info(f"像素级统计 - 真阳性: {tp}, 假阳性: {fp}, 假阴性: {fn}")
    logger.info(f"Dice计算统计 - 交集: {dice_intersection}, 预测总和: {dice_pred_sum}, 目标总和: {dice_target_sum}")
    logger.info(f"总样本数: {results['total_samples']}")
    logger.info(
        f"有建筑物的样本数: {results['samples_with_buildings']} ({results['samples_with_buildings'] / max(1, results['total_samples']) * 100:.2f}%)")
    logger.info(
        f"无建筑物的样本数: {results['samples_without_buildings']} ({results['samples_without_buildings'] / max(1, results['total_samples']) * 100:.2f}%)")
    logger.info(
        f"有预测的样本数: {results['samples_with_predictions']} ({results['samples_with_predictions'] / max(1, results['samples_with_buildings']) * 100:.2f}%)")
    logger.info(
        f"正确预测无建筑物的样本数: {results['correct_empty_predictions']} ({results['correct_empty_predictions'] / max(1, results['samples_without_buildings']) * 100:.2f}%)")
    logger.info(
        f"误报样本数（预测有但实际无）: {results['false_alarm_samples']} ({results['false_alarm_samples'] / max(1, results['samples_without_buildings']) * 100:.2f}%)")
    logger.info(f"样本级准确率: {results['sample_accuracy']:.4f}")
    logger.info(f"无建筑物样本准确率: {results['empty_accuracy']:.4f}")
    logger.info(f"IoU: {results['iou']:.4f}")
    logger.info(f"Dice系数: {results['dice']:.4f}")  # 新增Dice输出
    logger.info(f"像素准确率: {results['pixel_accuracy']:.4f}")  # 新增像素准确率输出
    logger.info(f"精确度: {results['precision']:.4f}")
    logger.info(f"召回率: {results['recall']:.4f}")
    logger.info(f"F1分数: {results['f1']:.4f}")

    # 保存结果
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    # 创建可视化
    if samples_to_visualize:
        create_visualizations(samples_to_visualize, vis_dir)

    return results


def create_visualizations(samples, output_dir):
    """创建可视化图像"""
    for i, sample in enumerate(samples):
        try:
            # 获取样本
            image = sample['image'][0]  # [C, H, W]
            mask = sample['mask'][0, 0]  # [H, W]
            pred = sample['pred'][0, 0]  # [H, W]
            prob = sample['prob'][0, 0]  # [H, W]
            basename = sample['basename']

            # 转换图像
            image_np = denormalize(image).permute(1, 2, 0).numpy()
            # 检查是否需要归一化
            if image_np.max() > 1.0:
                image_np = image_np / 255.0

            mask_np = mask.numpy()
            pred_np = pred.numpy()
            prob_np = prob.numpy()

            # 创建图像
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 显示原始图像
            axes[0, 0].imshow(image_np)
            axes[0, 0].set_title('Input Image')
            axes[0, 0].axis('off')

            # 显示标签
            axes[0, 1].imshow(mask_np, cmap='gray')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')

            # 显示预测
            axes[1, 0].imshow(pred_np, cmap='gray')
            axes[1, 0].set_title('Prediction (Binary)')
            axes[1, 0].axis('off')

            # 显示概率
            im = axes[1, 1].imshow(prob_np, cmap='jet')
            axes[1, 1].set_title('Prediction (Probability)')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])

            # 保存图像
            plt.tight_layout()
            plt.savefig(output_dir / f"{basename}_vis.png", dpi=200)
            plt.close(fig)

            # 创建叠加图像
            overlay_img = image_np.copy()
            # 确保像素值在0-1范围内
            if overlay_img.max() > 1.0:
                overlay_img = overlay_img / 255.0
            # 添加标签轮廓
            mask_contours, _ = cv2.findContours((mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            # 检查overlay_img的范围
            if overlay_img.max() <= 1.0:
                # 如果是0-1范围，需要对颜色值进行归一化
                cv2.drawContours(overlay_img, mask_contours, -1, (0, 1, 0), 2)  # 归一化的颜色值
            else:
                cv2.drawContours(overlay_img, mask_contours, -1, (0, 255, 0), 2)  # 标准的颜色值

            # 添加预测轮廓
            pred_contours, _ = cv2.findContours((pred_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, pred_contours, -1, (255, 0, 0), 2)

            # 保存叠加图像
            plt.figure(figsize=(8, 8))
            plt.imshow(overlay_img.astype(np.uint8))
            plt.title(f'Ground Truth (green) vs Prediction (red)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / f"{basename}_overlay.png", dpi=200)
            plt.close()

        except Exception as e:
            logger.error(f"创建可视化时出错 {basename}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


def check_image_labels(image_dir, label_dir, num_samples=10):
    """直接检查图像和标签"""
    # 获取所有图像
    image_files = []
    for ext in ['.tif', '.jpg', '.png', '.jpeg']:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))

    # 随机选择样本
    if len(image_files) > num_samples:
        samples = random.sample(image_files, num_samples)
    else:
        samples = image_files

    # 检查每个样本
    for img_path in samples:
        # 获取基础文件名
        basename = img_path.stem

        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"无法读取图像: {img_path}")
            continue

        # 查找标签
        label_found = False
        for ext in ['.tif', '.jpg', '.png', '.jpeg']:
            label_path = Path(label_dir) / f"{basename}{ext}"
            if label_path.exists():
                # 读取标签
                mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    logger.info(f"图像: {basename}, 尺寸: {img.shape}, 标签尺寸: {mask.shape}")
                    logger.info(f"  标签范围: [{mask.min()}, {mask.max()}], 非零元素: {np.count_nonzero(mask)}")
                    label_found = True
                    break

        if not label_found:
            logger.warning(f"找不到图像 {basename} 对应的标签文件")

    logger.info(f"检查了 {len(samples)} 个样本")


def get_model_folder_name(config):
    """根据模型配置生成适合的文件夹名称"""
    model_config = config.get('model', {})

    # 基础名称
    base_name = "unet"

    # 添加特性标识
    if model_config.get('enable_janus', False):
        base_name += "_janus"

    if model_config.get('use_text', False):
        base_name += "_text"

    # 添加损失函数类型
    loss_type = config['training'].get('loss', {}).get('type', 'bce')
    base_name += f"_{loss_type}"

    return base_name


def main():
    # 解析命令行参数
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 根据命令行参数更新配置
    config = update_config_from_args(config, args)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 获取模型特定的文件夹名
    model_folder_name = get_model_folder_name(config)

    # 创建输出目录 - 使用模型特定的子目录
    base_output_dir = Path(args.output)
    output_dir = base_output_dir / model_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 添加时间戳到输出目录，确保每次测试结果不会覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"测试结果将保存到: {output_dir}")

    # 获取数据路径
    data_config = config['data']
    data_root = Path(data_config.get('root_dir', './data'))

    # 图像和标签目录
    if args.test_train:
        image_dir = data_root / data_config['train']['image_dir']
        label_dir = data_root / data_config['train']['label_dir']
        text_features_dir = data_root / data_config['train']['text_features_dir'] if config['model'].get('use_text',
                                                                                                         False) else None
        logger.info("测试训练集数据...")
    else:
        image_dir = data_root / data_config['test']['image_dir']
        label_dir = data_root / data_config['test']['label_dir']
        text_features_dir = data_root / data_config['test']['text_features_dir'] if config['model'].get('use_text',
                                                                                                        False) else None
        logger.info("测试测试集数据...")

    logger.info(f"图像目录: {image_dir}")
    logger.info(f"标签目录: {label_dir}")
    if text_features_dir:
        logger.info(f"文本特征目录: {text_features_dir}")

    # 检查图像和标签
    logger.info("检查图像和标签...")
    check_image_labels(image_dir, label_dir, num_samples=5)

    # 加载模型
    model = load_model(config, args.checkpoint, device, args)

    # 创建数据集
    image_size = config['model'].get('image_size', 512)
    test_dataset = SimpleDataset(image_dir, label_dir, text_features_dir, image_size=image_size)

    # 保存模型检查点信息
    checkpoint_info = {
        'path': args.checkpoint,
        'threshold': args.threshold,
        'config_file': args.config,
        'model_config': {
            'enable_janus': config['model'].get('enable_janus', False),
            'use_text': config['model'].get('use_text', False)
        }
    }

    with open(output_dir / "test_info.json", "w") as f:
        json.dump(checkpoint_info, f, indent=4)

    # 测试模型
    batch_size = config.get('testing', {}).get('batch_size', 4)
    results = test_model(model, test_dataset, device, output_dir,
                         threshold=args.threshold, batch_size=batch_size)

    # 打印结果摘要
    logger.info("测试完成!")
    logger.info(f"评估结果:")
    logger.info(f"  总样本数: {results['total_samples']}")
    logger.info(
        f"  有建筑物的样本数: {results['samples_with_buildings']} ({results['samples_with_buildings'] / max(1, results['total_samples']) * 100:.2f}%)")
    logger.info(
        f"  有预测的样本数: {results['samples_with_predictions']} ({results['samples_with_predictions'] / max(1, results['samples_with_buildings']) * 100:.2f}%)")
    logger.info(f"  IoU: {results['iou']:.4f}")
    logger.info(f"  F1分数: {results['f1']:.4f}")

    # 将结果写入文件
    with open(output_dir / "test_summary.txt", "w") as f:
        f.write(f"模型: {args.checkpoint}\n")
        f.write(f"阈值: {args.threshold}\n\n")
        f.write(f"总样本数: {results['total_samples']}\n")
        f.write(
            f"有建筑物的样本数: {results['samples_with_buildings']} ({results['samples_with_buildings'] / max(1, results['total_samples']) * 100:.2f}%)\n")
        f.write(
            f"有预测的样本数: {results['samples_with_predictions']} ({results['samples_with_predictions'] / max(1, results['samples_with_buildings']) * 100:.2f}%)\n\n")
        f.write(f"IoU: {results['iou']:.4f}\n")
        f.write(f"精确度: {results['precision']:.4f}\n")
        f.write(f"召回率: {results['recall']:.4f}\n")
        f.write(f"F1分数: {results['f1']:.4f}\n")

    logger.info(f"结果摘要已保存到 {output_dir / 'test_summary.txt'}")


if __name__ == '__main__':
    main()