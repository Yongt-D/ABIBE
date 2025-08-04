# utils/enhanced_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class EnhancedSegmentationLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=0.5, tversky_weight=0.5,
                 focal_gamma=2.0, tversky_alpha=0.6, tversky_beta=0.4):
        super(EnhancedSegmentationLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

        # Focal Loss参数
        self.focal_gamma = focal_gamma

        # Tversky Loss参数
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        logger.info(f"初始化增强分割损失: bce_weight={bce_weight}, dice_weight={dice_weight}, "
                    f"focal_weight={focal_weight}, tversky_weight={tversky_weight}, "
                    f"focal_gamma={focal_gamma}, tversky_alpha={tversky_alpha}, tversky_beta={tversky_beta}")

    def binary_focal_loss(self, pred, target, gamma=2.0, alpha=0.25, epsilon=1e-6):
        # 应用sigmoid获取概率
        pred_prob = torch.sigmoid(pred)

        # 二值交叉熵计算
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Focal权重项
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_factor = alpha * target + (1 - alpha) * (1 - target)
        modulating_factor = (1.0 - p_t) ** gamma

        # 应用权重
        focal_loss = alpha_factor * modulating_factor * bce

        return focal_loss.mean()

    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)

        # 展平预测和目标
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # 计算交集和并集
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return 1 - dice

    def tversky_loss(self, pred, target, alpha=0.7, beta=0.3, smooth=1.0):
        pred = torch.sigmoid(pred)

        # 展平预测和目标
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # 计算TP, FP, FN
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        # Tversky系数
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

        return 1 - tversky

    def forward(self, pred, target):
        try:
            # 确保形状一致
            if pred.shape != target.shape:
                pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=True)

            # 限制极端预测以防止数值问题
            pred_clamped = torch.clamp(pred, min=-15.0, max=15.0)

            # 计算各个损失分量
            bce_loss = F.binary_cross_entropy_with_logits(pred_clamped, target, reduction='mean')
            dice = self.dice_loss(pred_clamped, target)
            focal = self.binary_focal_loss(pred_clamped, target, gamma=self.focal_gamma)
            tversky = self.tversky_loss(pred_clamped, target, alpha=self.tversky_alpha, beta=self.tversky_beta)

            # 计算总损失
            total_loss = (self.bce_weight * bce_loss +
                          self.dice_weight * dice +
                          self.focal_weight * focal +
                          self.tversky_weight * tversky)

            # 记录损失分量
            loss_components = {
                'bce': bce_loss.item(),
                'dice': dice.item(),
                'focal': focal.item(),
                'tversky': tversky.item(),
                'total': total_loss.item()
            }

            return total_loss, loss_components

        except Exception as e:
            logger.error(f"计算增强分割损失时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 回退到基础BCE损失
            pred_clamped = torch.clamp(pred, min=-15.0, max=15.0)
            bce = F.binary_cross_entropy_with_logits(pred_clamped, target)
            return bce, {'bce': bce.item(), 'total': bce.item()}


class SimplifiedLossBalancer(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=0.5, tversky_weight=0.5):
        super(SimplifiedLossBalancer, self).__init__()

        # 使用固定权重
        self.weights = {
            'bce': bce_weight,
            'dice': dice_weight,
            'focal': focal_weight,
            'tversky': tversky_weight
        }

        # 创建基础损失函数
        self.base_loss = EnhancedSegmentationLoss(
            bce_weight=self.weights['bce'],
            dice_weight=self.weights['dice'],
            focal_weight=self.weights['focal'],
            tversky_weight=self.weights['tversky']
        )

        logger.info(f"初始化简化损失平衡器: {self.weights}")

    def update_epoch(self, epoch):
        """保留接口兼容性，但不实际改变权重"""
        pass

    def forward(self, pred, target):
        """计算损失"""
        return self.base_loss(pred, target)