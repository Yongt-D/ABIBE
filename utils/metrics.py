# utils/metrics.py
import torch
import numpy as np
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(pred, target, smooth=1e-6, threshold=0.5, verbose=False):
    try:
        # 确保输入是张量
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)

        # 移到CPU进行计算
        pred = pred.detach().cpu().float()  # 确保是float类型
        target = target.detach().cpu().float()

        if verbose:
            logger.info(f"预测形状: {pred.shape}, 标签形状: {target.shape}")
            logger.info(f"预测范围: {pred.min().item()} ~ {pred.max().item()}")
            logger.info(f"标签范围: {target.min().item()} ~ {target.max().item()}")

        # 应用阈值 (如果pred还不是二值的)
        if pred.max() > 1.1 or pred.min() < -0.1:
            # 可能是logits，应用sigmoid
            if verbose:
                logger.info("应用sigmoid到预测")
            pred = torch.sigmoid(pred)

        if threshold is not None:
            pred_binary = (pred > threshold).float()
        else:
            pred_binary = pred.float()

        # 确保维度正确
        if len(pred_binary.shape) == 4 and pred_binary.shape[1] == 1:
            # [B, 1, H, W] -> [B, H, W]
            pred_binary = pred_binary.squeeze(1)

        if len(target.shape) == 4 and target.shape[1] == 1:
            # [B, 1, H, W] -> [B, H, W]
            target = target.squeeze(1)

        # 确保target是二值的
        target_binary = (target > 0.5).float()

        # 确保形状匹配
        if pred_binary.shape != target_binary.shape:
            if verbose:
                logger.warning(f"预测和标签形状不匹配: {pred_binary.shape} vs {target_binary.shape}")

            # 尝试调整大小
            if len(pred_binary.shape) == 3 and len(target_binary.shape) == 3:
                if pred_binary.shape[0] == target_binary.shape[0]:  # 批次大小相同
                    # 调整预测大小以匹配标签
                    pred_binary = F.interpolate(
                        pred_binary.unsqueeze(1),  # [B, H, W] -> [B, 1, H, W]
                        size=target_binary.shape[1:],
                        mode='nearest'
                    ).squeeze(1)  # [B, 1, H, W] -> [B, H, W]

        # 再次检查形状是否匹配
        if pred_binary.shape != target_binary.shape:
            logger.error(f"调整后预测和标签形状仍不匹配: {pred_binary.shape} vs {target_binary.shape}")
            return get_zero_metrics()

        # 计算交集和并集 - 在空间维度上求和
        intersection = (pred_binary * target_binary).sum(dim=(1, 2))
        pred_sum = pred_binary.sum(dim=(1, 2))
        target_sum = target_binary.sum(dim=(1, 2))
        union = pred_sum + target_sum - intersection

        # 计算IoU
        iou_per_sample = (intersection + smooth) / (union + smooth)
        iou = iou_per_sample.mean().item()

        # 计算Dice系数 - 修复这里的计算
        dice_per_sample = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice = dice_per_sample.mean().item()

        # 验证Dice和IoU的关系
        if verbose:
            logger.info(f"IoU: {iou:.4f}, Dice: {dice:.4f}")
            # 理论上 Dice = 2*IoU / (1+IoU)
            expected_dice = 2 * iou / (1 + iou) if iou > 0 else 0
            logger.info(f"期望Dice (基于IoU): {expected_dice:.4f}")

        # 计算精确度和召回率
        true_positive = intersection.sum().item()
        false_positive = (pred_binary * (1 - target_binary)).sum().item()
        false_negative = ((1 - pred_binary) * target_binary).sum().item()
        true_negative = ((1 - pred_binary) * (1 - target_binary)).sum().item()

        precision = true_positive / (true_positive + false_positive + smooth)
        recall = true_positive / (true_positive + false_negative + smooth)

        # F1分数 - 应该与Dice非常接近
        f1 = 2 * precision * recall / (precision + recall + smooth)

        # 像素精确度
        correct_pixels = (pred_binary == target_binary).sum().item()
        total_pixels = target_binary.numel()
        pixel_accuracy = correct_pixels / total_pixels

        # 验证F1和Dice的一致性
        if verbose and abs(f1 - dice) > 0.01:
            logger.warning(f"F1 ({f1:.4f}) 和 Dice ({dice:.4f}) 差异较大，可能存在计算错误")

        metrics = {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'pixel_accuracy': float(pixel_accuracy)
        }

        # 验证所有指标都是有效数值
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                logger.error(f"指标 {key} 计算异常: {value}")
                return get_zero_metrics()

        return metrics

    except Exception as e:
        logger.error(f"计算指标时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return get_zero_metrics()


def get_zero_metrics():
    return {
        'iou': 0.0,
        'dice': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'pixel_accuracy': 0.0
    }


def validate_metrics_consistency(metrics, tolerance=0.01):
    dice = metrics['dice']
    f1 = metrics['f1']
    iou = metrics['iou']

    # 检查Dice和F1的一致性（对于二值分割应该相等）
    if abs(dice - f1) > tolerance:
        logger.warning(f"Dice ({dice:.4f}) 和 F1 ({f1:.4f}) 不一致")
        return False

    # 检查IoU和Dice的关系：Dice = 2*IoU / (1+IoU)
    if iou > 0:
        expected_dice = 2 * iou / (1 + iou)
        if abs(dice - expected_dice) > tolerance:
            logger.warning(f"Dice ({dice:.4f}) 与基于IoU的期望值 ({expected_dice:.4f}) 不一致")
            return False

    return True


# 添加一个测试函数来验证指标计算
def test_metrics_calculation():
    """测试指标计算的正确性"""
    # 创建测试数据
    batch_size = 2
    height, width = 64, 64

    # 完美预测的情况
    pred_perfect = torch.ones(batch_size, 1, height, width)
    target = torch.ones(batch_size, 1, height, width)

    metrics = calculate_metrics(pred_perfect, target, verbose=True)
    print("完美预测指标:", metrics)

    # 部分重叠的情况
    pred_partial = torch.zeros(batch_size, 1, height, width)
    pred_partial[:, :, :height // 2, :] = 1  # 上半部分为1

    metrics = calculate_metrics(pred_partial, target, verbose=True)
    print("部分重叠指标:", metrics)
    print("一致性检查:", validate_metrics_consistency(metrics))


if __name__ == "__main__":
    test_metrics_calculation()