# scripts/train.py
import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import datetime
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import math


# 确保项目根目录在Python路径中
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型和工具
from models.building_extraction_model import BuildingExtractionModel
# from models.simplified_abibe_model import SimplifiedABIBE as BuildingExtractionModel
from utils.data_utils import BuildingDataset
from utils.metrics import calculate_metrics

# 导入增强型损失函数
try:
    from utils.enhanced_loss import EnhancedSegmentationLoss, SimplifiedLossBalancer
except ImportError:
    logging.warning("未能导入增强型损失函数，如需使用请确保此模块存在")

# 创建日志目录
os.makedirs('logs', exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/train_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统一的建筑提取模型训练脚本')
    parser.add_argument('--config', type=str, default='configs/enhanced_loss_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='训练设备')
    parser.add_argument('--gpu', type=int, default=None,
                        help='指定GPU ID (例如: 0, 1, 2)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--debug', action='store_true',
                        help='是否开启调试模式')

    # 新增：允许在命令行直接设置模型配置
    parser.add_argument('--enable-janus', action='store_true',
                        help='启用JanusPro特征')
    parser.add_argument('--use-text', action='store_true',
                        help='使用文本特征')

    return parser.parse_args()


def get_loss_function(config):
    """根据配置选择损失函数"""
    loss_type = config['training'].get('loss', {}).get('type', 'bce')

    if loss_type == 'enhanced':
        # 使用增强型损失函数
        loss_config = config['training'].get('loss', {})
        return EnhancedSegmentationLoss(
            bce_weight=loss_config.get('bce_weight', 1.0),
            dice_weight=loss_config.get('dice_weight', 1.0),
            focal_weight=loss_config.get('focal_weight', 0.5),
            tversky_weight=loss_config.get('tversky_weight', 0.5),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            tversky_alpha=loss_config.get('tversky_alpha', 0.6),
            tversky_beta=loss_config.get('tversky_beta', 0.4)
        )
    elif loss_type == 'simplified':
        # 使用简化的损失平衡器
        loss_config = config['training'].get('loss', {})
        return SimplifiedLossBalancer(
            bce_weight=loss_config.get('bce_weight', 1.0),
            dice_weight=loss_config.get('dice_weight', 1.0),
            focal_weight=loss_config.get('focal_weight', 0.5),
            tversky_weight=loss_config.get('tversky_weight', 0.5)
        )
    else:
        # 标准损失函数
        return nn.BCEWithLogitsLoss()


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

    if args.use_text:
        config['model']['use_text'] = True
        logger.info("通过命令行参数启用文本特征")

    return config


def create_dataloaders(config):
    """创建数据加载器"""
    try:
        # 数据集参数
        data_config = config['data']
        train_config = config['training']
        model_config = config['model']

        logger.info("创建训练和验证数据集...")

        # 获取项目根目录的绝对路径
        root_dir = Path.cwd()
        data_root = root_dir / Path(data_config['root_dir'])

        logger.info(f"数据根目录: {data_root}")

        # 创建训练数据集
        train_image_dir = data_root / data_config['train']['image_dir']
        train_label_dir = data_root / data_config['train']['label_dir']
        train_text_dir = data_root / data_config['train']['text_features_dir'] if data_config['train'].get(
            'text_features_dir') else None

        logger.info(f"训练图像目录: {train_image_dir}")
        logger.info(f"训练标签目录: {train_label_dir}")
        logger.info(f"训练文本特征目录: {train_text_dir}")

        train_dataset = BuildingDataset(
            image_dir=train_image_dir,
            label_dir=train_label_dir,
            text_features_dir=train_text_dir,
            phase='train',
            image_size=model_config.get('image_size', 512),
            augmentation=train_config['augmentation']['enabled'],
        )

        # 创建验证数据集
        val_image_dir = data_root / data_config['val']['image_dir']
        val_label_dir = data_root / data_config['val']['label_dir']
        val_text_dir = data_root / data_config['val']['text_features_dir'] if data_config['val'].get(
            'text_features_dir') else None

        logger.info(f"验证图像目录: {val_image_dir}")
        logger.info(f"验证标签目录: {val_label_dir}")
        logger.info(f"验证文本特征目录: {val_text_dir}")

        val_dataset = BuildingDataset(
            image_dir=val_image_dir,
            label_dir=val_label_dir,
            text_features_dir=val_text_dir,
            phase='val',
            image_size=model_config.get('image_size', 512),
            augmentation=False,
        )

        logger.info(f"数据集创建成功，训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=data_config['dataloader']['num_workers'],
            pin_memory=True,
            drop_last=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=data_config['dataloader']['num_workers'],
            pin_memory=True,
            drop_last=False
        )

        logger.info(f"数据加载器创建成功，训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

        return train_loader, val_loader

    except Exception as e:
        logger.error(f"创建数据加载器失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


# 优化学习率调度
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    创建带预热的余弦学习率调度器
    """

    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦衰减阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 修改优化器配置
def create_optimizer_and_scheduler(model, config):
    """创建优化器和学习率调度器"""
    # 提取参数组 - 对不同部分使用不同的学习率
    decay_parameters = []
    no_decay_parameters = []
    janus_parameters = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'janus' in name:
            janus_parameters.append(param)
        elif 'norm' in name or 'bias' in name:
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)

    # 创建参数组
    optimizer_grouped_parameters = [
        {'params': decay_parameters, 'weight_decay': 5e-4, 'lr': 2e-4},  # 提高主体学习率
        {'params': no_decay_parameters, 'weight_decay': 0.0, 'lr': 2e-4},  # 提高主体学习率
        {'params': janus_parameters, 'weight_decay': 1e-6, 'lr': 1e-5}  # 降低JanusPro学习率
    ]

    # 创建优化器
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config['training']['optimizer']['learning_rate'],
        eps=1e-8
    )

    # 创建学习率调度器
    epochs = config['training']['epochs']
    warmup_epochs = int(epochs * 0.1)  # 预热阶段为总epoch的10%
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=epochs
    )

    return optimizer, scheduler


def validate(model, val_loader, criterion, device, epoch, config, debug=False):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    metrics_sum = {
        'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0
    }
    loss_components_sum = {
        'bce': 0.0, 'dice': 0.0, 'focal': 0.0, 'tversky': 0.0, 'total': 0.0
    }
    processed_batches = 0

    # 使用tqdm显示进度条
    pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1} [Val]')

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(pbar):
            try:
                # 解析批次数据
                images, text_features, masks = batch_data

                # 移动数据到设备
                images = images.to(device)
                if text_features is not None:
                    text_features = text_features.to(device)
                masks = masks.to(device)

                # 打印形状（调试用）
                if debug and batch_idx == 0:
                    logger.info(f"验证批次 - 图像形状: {images.shape}, 标签形状: {masks.shape}")

                # 确保标签形状正确
                if len(masks.shape) != 4:
                    if len(masks.shape) == 3:
                        masks = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

                # 前向传播
                outputs = model(images, text_features)

                # 处理输出为元组的情况(主输出和辅助输出)
                if isinstance(outputs, tuple):
                    main_output = outputs[0]  # 主输出是元组的第一个元素

                    # 确保主输出形状与掩码匹配
                    if main_output.shape != masks.shape:
                        main_output = F.interpolate(main_output, size=masks.shape[2:], mode='bilinear',
                                                    align_corners=True)

                    # 仅使用主输出进行损失计算
                    outputs_for_loss = main_output
                else:
                    # 常规单输出处理
                    if outputs.shape != masks.shape:
                        outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=True)
                    outputs_for_loss = outputs

                # 计算损失
                loss_result = criterion(outputs_for_loss, masks)
                if isinstance(loss_result, tuple):
                    loss = loss_result[0]
                    if len(loss_result) > 1:
                        loss_components = loss_result[1] if isinstance(loss_result[1], dict) else {'total': loss.item()}
                    else:
                        loss_components = {'total': loss.item()}
                else:
                    loss = loss_result
                    loss_components = {'total': loss.item()}

                # 计算指标
                # 应用sigmoid获取概率
                preds = torch.sigmoid(outputs_for_loss)
                # 二值化
                preds_binary = (preds > 0.5).float()

                # 常规分割指标
                batch_metrics = calculate_metrics(preds_binary, masks)

                # 累加指标
                for k in metrics_sum:
                    if k in batch_metrics:
                        metrics_sum[k] += batch_metrics[k]

                # 累加损失组件
                for k, v in loss_components.items():
                    loss_components_sum[k] = loss_components_sum.get(k, 0.0) + v

                # 累计损失
                running_loss += loss.item()
                processed_batches += 1

                # 更新进度条
                status = {
                    'loss': f'{loss_components["total"]:.4f}',  # 使用组件中的总损失
                    'iou': f'{batch_metrics["iou"]:.4f}'
                }
                pbar.set_postfix(status)

            except Exception as e:
                logger.error(f"验证批次 {batch_idx} 处理时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    # 计算平均值
    avg_loss = running_loss / max(processed_batches, 1)
    avg_metrics = {k: v / max(processed_batches, 1) for k, v in metrics_sum.items()}
    avg_loss_components = {k: v / max(processed_batches, 1) for k, v in loss_components_sum.items()}

    return avg_loss, avg_metrics, avg_loss_components


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, filename):
    """保存检查点"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 保存模型状态的副本，排除JanusPro提取器
    if hasattr(model, 'janus_extractor') and model.janus_extractor is not None:
        # 临时保存提取器引用
        temp_extractor = model.janus_extractor
        # 设置为None以避免保存
        model.janus_extractor = None

        # 创建检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_metric': best_metric,
            'config': config
        }

        # 恢复提取器引用
        model.janus_extractor = temp_extractor
    else:
        # 正常创建检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_metric': best_metric,
            'config': config
        }

    torch.save(checkpoint, filename)
    logger.info(f"检查点已保存: {filename}")


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


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, scaler, debug=False):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    metrics_sum = {
        'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0
    }
    loss_components_sum = {
        'bce': 0.0, 'dice': 0.0, 'focal': 0.0, 'tversky': 0.0, 'total': 0.0
    }
    processed_batches = 0

    # 获取混合精度配置
    use_amp = config.get('training', {}).get('mixed_precision', True)

    # 移除梯度累积相关代码
    # accumulation_steps = config.get('training', {}).get('accumulation_steps', 1)

    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [Train]')

    for batch_idx, batch_data in enumerate(pbar):
        try:
            # 解析批次数据
            images, text_features, masks = batch_data

            # 移动数据到设备
            images = images.to(device)
            if text_features is not None:
                text_features = text_features.to(device)
            masks = masks.to(device)

            # 打印形状（调试用）
            if debug and batch_idx == 0:
                logger.info(f"训练批次 - 图像形状: {images.shape}, 标签形状: {masks.shape}")
                logger.info(f"标签值范围: {masks.min().item()} ~ {masks.max().item()}")

            # 确保标签形状正确
            if len(masks.shape) != 4:
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

            # 清除梯度
            optimizer.zero_grad()

            # 使用autocast进行前向传播
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 前向传播
                outputs = model(images, text_features)

                # 第一个批次打印输出信息用于调试
                if batch_idx == 0:
                    # 检查模型输出格式
                    logger.info(f"模型输出类型: {type(outputs)}")
                    if isinstance(outputs, tuple):
                        logger.info(f"输出元组长度: {len(outputs)}")
                        logger.info(f"主输出形状: {outputs[0].shape}")
                        logger.info(f"主输出统计: 最小值={outputs[0].min().item():.4f}, 最大值={outputs[0].max().item():.4f}")
                    else:
                        logger.info(f"输出形状: {outputs.shape}")
                        logger.info(f"输出统计: 最小值={outputs.min().item():.4f}, 最大值={outputs.max().item():.4f}")

                    # 检查目标统计
                    logger.info(f"目标统计: 最小值={masks.min().item():.4f}, 最大值={masks.max().item():.4f}")

                # 处理输出为元组的情况(主输出和辅助输出)
                if isinstance(outputs, tuple):
                    main_output = outputs[0]  # 主输出是元组的第一个元素
                    aux_outputs = outputs[1] if len(outputs) > 1 else None  # 辅助输出

                    # 确保主输出形状与掩码匹配
                    if main_output.shape != masks.shape:
                        logger.info(f"调整主输出形状从 {main_output.shape} 到 {masks.shape}")
                        main_output = F.interpolate(main_output, size=masks.shape[2:], mode='bilinear', align_corners=True)

                    # 仅使用主输出进行损失计算
                    outputs_for_loss = main_output
                else:
                    # 常规单输出处理
                    if outputs.shape != masks.shape:
                        logger.info(f"调整输出形状从 {outputs.shape} 到 {masks.shape}")
                        outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=True)
                    outputs_for_loss = outputs

                # 检查输出是否为None
                if outputs_for_loss is None:
                    logger.error(f"模型输出为None，跳过此批次")
                    continue

                # 计算损失
                loss_result = criterion(outputs_for_loss, masks)
                loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result

                # 检查每50个批次的统计数据
                if batch_idx % 50 == 0:
                    with torch.no_grad():
                        # 检查sigmoid输出
                        sigmoid_out = torch.sigmoid(outputs_for_loss)
                        # 计算接近0或1的像素比例
                        near_zero = (sigmoid_out < 0.01).float().sum() / sigmoid_out.numel()
                        near_one = (sigmoid_out > 0.99).float().sum() / sigmoid_out.numel()

                        logger.info(f"Batch {batch_idx} statistics:")
                        logger.info(f"  Pred sigmoid near 0: {near_zero.item() * 100:.2f}%, near 1: {near_one.item() * 100:.2f}%")

                        # 检查单个损失组件
                        components = loss_result[1] if isinstance(loss_result, tuple) and len(loss_result) > 1 else {}
                        if isinstance(components, dict):
                            for k, v in components.items():
                                logger.info(f"  Loss component {k}: {v}")

                # 处理损失结果
                if isinstance(loss_result, tuple):
                    orig_loss = loss_result[0]
                    if len(loss_result) > 1:
                        loss_components = loss_result[1] if isinstance(loss_result[1], dict) else {'total': orig_loss.item()}
                    else:
                        loss_components = {'total': orig_loss.item()}
                else:
                    orig_loss = loss_result
                    loss_components = {'total': orig_loss.item()}

            # 使用scaler进行反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            if config.get('training', {}).get('gradient_clipping', {}).get('enabled', False):
                max_norm = config.get('training', {}).get('gradient_clipping', {}).get('max_norm', 1.0)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            else:
                # 使用默认裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # 计算标准指标 - 使用未缩放的损失和预测
            with torch.no_grad():
                # 应用sigmoid获取概率
                preds = torch.sigmoid(outputs_for_loss)
                # 二值化
                preds_binary = (preds > 0.5).float()

                # 计算常规分割指标
                batch_metrics = calculate_metrics(preds_binary, masks)

                # 累加指标
                for k in metrics_sum:
                    if k in batch_metrics:
                        metrics_sum[k] += batch_metrics[k]

                # 累加损失组件
                for k, v in loss_components.items():
                    loss_components_sum[k] = loss_components_sum.get(k, 0.0) + v

            # 累计损失
            running_loss += orig_loss.item()
            processed_batches += 1

            # 更新进度条
            status = {
                'loss': f'{orig_loss.item():.4f}',
                'iou': f'{batch_metrics["iou"]:.4f}'
            }
            pbar.set_postfix(status)

        except Exception as e:
            logger.error(f"训练批次 {batch_idx} 处理时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # 计算平均值
    avg_loss = running_loss / max(processed_batches, 1)
    avg_metrics = {k: v / max(processed_batches, 1) for k, v in metrics_sum.items()}
    avg_loss_components = {k: v / max(processed_batches, 1) for k, v in loss_components_sum.items()}

    # 打印详细损失组件（用于调试）
    if debug:
        logger.info("损失组件详情:")
        for k, v in avg_loss_components.items():
            logger.info(f"  {k}: {v:.4f}")

    return avg_loss, avg_metrics, avg_loss_components

def setup_logging(debug=False):
    """设置日志记录，避免创建空日志文件"""
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)

    # 生成时间戳和日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/train_{timestamp}.log'

    # 配置日志
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 控制台处理器(立即添加)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 返回日志文件路径和配置好的日志器
    return log_file, logger


def add_file_handler(logger, log_file):
    """训练开始后添加文件处理器，避免创建空日志文件"""
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"日志文件已创建: {log_file}")


def save_training_summary(epoch, train_loss, train_metrics, val_loss, val_metrics, summary_file=None):
    """
    保存训练摘要到日志文件

    Args:
        epoch: 当前轮次
        train_loss: 训练损失
        train_metrics: 训练指标字典
        val_loss: 验证损失
        val_metrics: 验证指标字典
        summary_file: 摘要文件路径，如果为None则自动创建
    """
    # 如果没有指定摘要文件，创建一个新文件
    if summary_file is None:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(log_dir, f'training_summary_{timestamp}.txt')

    # 格式化摘要行
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    train_iou = train_metrics.get('iou', 0.0)
    val_iou = val_metrics.get('iou', 0.0)
    summary_line = f"{current_time} - INFO - Epoch {epoch + 1}/150 - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}\n"

    # 写入摘要文件
    with open(summary_file, 'a') as f:
        f.write(summary_line)

    return summary_file


def cleanup_empty_logs(threshold_size=10):
    """清理空的或几乎为空的日志文件"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        return

    cleaned = 0
    for filename in os.listdir(log_dir):
        if filename.startswith('train_') and filename.endswith('.log'):
            filepath = os.path.join(log_dir, filename)
            # 检查文件大小是否小于阈值(字节)
            if os.path.getsize(filepath) < threshold_size:
                try:
                    os.remove(filepath)
                    cleaned += 1
                except:
                    pass

    if cleaned > 0:
        print(f"已清理 {cleaned} 个空日志文件")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置日志 - 只创建控制台日志，文件日志稍后添加
    log_file, logger = setup_logging(args.debug)

    try:
        # 加载配置
        config = load_config(args.config)

        # 根据命令行参数更新配置
        config = update_config_from_args(config, args)

        # 初始化摘要文件路径
        summary_file = None

        # 此时添加文件处理器 - 确认配置加载成功后
        add_file_handler(logger, log_file)

        # 设置设备
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")

        # 模型配置信息
        model_config = config['model']
        logger.info(f"JanusPro特征: {'启用' if model_config.get('enable_janus', False) else '禁用'}")
        logger.info(f"文本特征: {'启用' if model_config.get('use_text', False) else '禁用'}")

        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(config)

        # 创建模型
        model = BuildingExtractionModel(config).to(device)
        logger.info("模型已创建并移至设备")

        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数量: {total_params:,}")
        logger.info(f"可训练参数量: {trainable_params:,}")

        # 混合精度训练设置
        use_amp = config.get('training', {}).get('mixed_precision', True)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        logger.info(f"混合精度训练: {'启用' if use_amp else '禁用'}")

        # 仅在调试模式下禁用JanusPro特征
        if args.debug:
            logger.info("=== 调试模式 ===")
            logger.info("禁用高级特性并使用基础BCE损失函数")

            # 禁用JanusPro
            if hasattr(model, 'enable_janus'):
                model.enable_janus = False
                model.use_text = False
                # 完全移除JanusPro提取器以避免GPU内存使用
                model.janus_extractor = None
                logger.info("已禁用JanusPro特性")

            # 使用简单的BCE损失
            criterion = nn.BCEWithLogitsLoss()
            logger.info("使用简单的BCEWithLogitsLoss")

            # 缩小最终层权重
            scaled_layers = 0
            for m in model.modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                    with torch.no_grad():
                        original_norm = m.weight.data.norm().item()
                        m.weight.data.mul_(0.01)
                        new_norm = m.weight.data.norm().item()
                        if m.bias is not None:
                            m.bias.data.zero_()
                        scaled_layers += 1
                        logger.info(f"缩小输出层权重: {original_norm:.4f} -> {new_norm:.4f}")

            logger.info(f"共缩小了 {scaled_layers} 个输出层的权重")
        else:
            # 非调试模式 - 使用配置指定的损失函数
            loss_type = config['training'].get('loss', {}).get('type', 'bce')
            criterion = get_loss_function(config)
            logger.info(f"使用配置指定的损失函数: {loss_type}")

            # 在非调试模式下也缩小输出层权重以提高训练稳定性
            scaled_layers = 0
            for m in model.modules():
                if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                    with torch.no_grad():
                        original_norm = m.weight.data.norm().item()
                        m.weight.data.mul_(0.01)
                        new_norm = m.weight.data.norm().item()
                        if m.bias is not None:
                            m.bias.data.zero_()
                        scaled_layers += 1
                        logger.info(f"缩小输出层权重: {original_norm:.4f} -> {new_norm:.4f}")

            logger.info(f"共缩小了 {scaled_layers} 个输出层的权重")

        # 测试模型输出范围
        with torch.no_grad():
            try:
                dummy_input = torch.randn(2, 3, 512, 512).to(device)
                dummy_output = model(dummy_input)
                if isinstance(dummy_output, tuple):
                    dummy_output = dummy_output[0]
                logger.info(
                    f"初始模型输出范围: 最小值={dummy_output.min().item():.4f}, 最大值={dummy_output.max().item():.4f}")
            except Exception as e:
                logger.error(f"测试模型输出时出错: {str(e)}")

        # 定义优化器
        optimizer_config = config['training']['optimizer']
        optimizer_type = optimizer_config.get('type', 'Adam')

        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
            logger.info(
                f"使用AdamW优化器，学习率: {optimizer_config['learning_rate']}, 权重衰减: {optimizer_config['weight_decay']}")
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
            logger.info(
                f"使用Adam优化器，学习率: {optimizer_config['learning_rate']}, 权重衰减: {optimizer_config['weight_decay']}")

        # 定义学习率调度器
        scheduler_config = config['training'].get('lr_scheduler', {})
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            verbose=True,
            min_lr=scheduler_config.get('min_lr', 0.000001)
        )
        metric_key = 'iou'
        logger.info(
            f"使用标准IoU作为学习率调度指标，减小因子: {scheduler_config.get('factor', 0.5)}, 耐心值: {scheduler_config.get('patience', 5)}")

        # 初始化变量
        start_epoch = 0
        best_metric = 0.0
        best_val_loss = float('inf')  # 新增：追踪最佳验证损失

        # 获取模型特定的文件夹名
        model_folder_name = get_model_folder_name(config)

        # 创建检查点目录
        checkpoint_base_dir = config['training']['checkpoint'].get('dir', 'checkpoints')
        checkpoint_dir = os.path.join(checkpoint_base_dir, model_folder_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"模型检查点将保存到: {checkpoint_dir}")

        # 是否恢复训练
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info(f"加载检查点: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=device)
                start_epoch = checkpoint['epoch'] + 1
                best_metric = checkpoint['best_metric']
                # 新增：如果检查点中包含最佳验证损失，则加载它
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']
                    logger.info(f"从检查点加载最佳验证损失: {best_val_loss:.4f}")
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if checkpoint.get('scheduler_state_dict') and scheduler:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # 恢复scaler状态（如果有）
                if use_amp and 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info(f"从epoch {start_epoch}恢复训练，最佳{metric_key}: {best_metric:.4f}")
            else:
                logger.error(f"找不到检查点: {args.resume}")

        # 训练循环
        epochs = config['training']['epochs']
        logger.info(f"开始训练，总epoch数: {epochs}")

        # 记录最后保存的文件名
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        # 新增：记录验证损失最小的模型路径
        best_val_loss_model_path = os.path.join(checkpoint_dir, 'best_val_loss_model.pth')

        # 早停设置
        early_stopping = config['training'].get('checkpoint', {}).get('early_stopping', 0)
        if early_stopping > 0:
            no_improve_epochs = 0
            logger.info(f"启用早停，耐心值: {early_stopping}")

        # 训练循环
        for epoch in range(start_epoch, epochs):
            # 如果使用自适应损失平衡器，更新当前epoch
            if hasattr(criterion, 'update_epoch'):
                criterion.update_epoch(epoch)

            # 训练一个epoch
            train_loss, train_metrics, train_loss_components = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, config, scaler, args.debug
            )

            # 验证
            val_loss, val_metrics, val_loss_components = validate(
                model, val_loader, criterion, device, epoch, config, args.debug
            )

            # 打印指标
            log_msg = f"Epoch {epoch + 1}/{epochs} - "
            log_msg += f"Train Loss: {train_loss:.4f}, Train IoU: {train_metrics['iou']:.4f}, "
            log_msg += f"Val Loss: {val_loss:.4f}, Val IoU: {val_metrics['iou']:.4f}"
            logger.info(log_msg)

            # 保存训练摘要
            summary_file = save_training_summary(epoch, train_loss, train_metrics, val_loss, val_metrics, summary_file)

            # 更新学习率
            current_metric = val_metrics['iou']
            scheduler.step(current_metric)

            # 保存最佳IoU模型
            if current_metric > best_metric:
                best_metric = current_metric
                # 确定保存的文件名
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if use_amp else None,  # 保存scaler状态
                    'best_metric': best_metric,
                    'best_val_loss': best_val_loss,  # 新增：保存最佳验证损失
                    'config': config
                }
                torch.save(checkpoint, best_model_path)
                logger.info(f"保存最佳模型，{metric_key}: {best_metric:.4f}")

                # 重置早停计数器
                no_improve_epochs = 0
            else:
                # 早停检查
                if early_stopping > 0:
                    no_improve_epochs += 1
                    if no_improve_epochs >= early_stopping:
                        logger.info(f"连续 {early_stopping} 个epoch没有改进，提前终止训练")
                        break
                    else:
                        logger.info(f"当前性能未改进，还剩 {early_stopping - no_improve_epochs} 个epoch进行早停检查")

            # 新增：保存验证损失最小的模型
            if val_loss < best_val_loss:
                old_best = best_val_loss
                best_val_loss = val_loss
                # 保存验证损失最小的模型
                checkpoint = {
                    'epoch': epoch + 1,  # 确保这是当前epoch
                    'model_state_dict': model.state_dict(),  # 确保这是当前模型状态
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if use_amp else None,
                    'best_metric': current_metric,  # 使用当前轮次的IoU而不是历史最佳
                    'best_val_loss': best_val_loss,
                    'val_loss': val_loss,  # 添加当前验证损失以便核对
                    'current_epoch_iou': val_metrics['iou'],  # 添加当前轮次的IoU以便核对
                    'config': config
                }

                # 添加详细日志
                logger.info(
                    f"发现新的最佳验证损失: {old_best:.6f} -> {best_val_loss:.6f}，轮次: {epoch + 1}, 当前IoU: {val_metrics['iou']:.4f}")

                torch.save(checkpoint, best_val_loss_model_path)
                logger.info(f"保存验证损失最小的模型，Val Loss: {best_val_loss:.6f}, Epoch: {epoch + 1}")

            # 定期保存检查点
            if (epoch + 1) % config['training'].get('checkpoint', {}).get('save_freq', 10) == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if use_amp else None,  # 保存scaler状态
                    'current_metric': current_metric,
                    'best_val_loss': best_val_loss,  # 新增：保存最佳验证损失
                    'config': config
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"保存周期性检查点 epoch_{epoch + 1}")

        logger.info(f"训练完成，最佳{metric_key}: {best_metric:.4f}，最佳验证损失: {best_val_loss:.4f}")
        return 0

    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # 如果这时还没有文件处理器，添加一个以记录错误
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            add_file_handler(logger, log_file)
        return 1

if __name__ == '__main__':
    exit_code = main()
    # 清理空日志文件
    cleanup_empty_logs()
    sys.exit(exit_code)