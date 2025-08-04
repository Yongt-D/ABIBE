# utils/visualization.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # 克隆张量避免修改原始张量
    tensor = tensor.clone().detach()

    # 确保维度正确
    if len(tensor.shape) == 4:  # 批次的情况
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    else:  # 单个图像的情况
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

    # 裁剪到 [0, 1] 范围
    tensor.clamp_(0, 1)

    return tensor


def save_batch_results(images, masks, predictions, save_path, max_samples=4, denorm=True):
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 确保输入是CPU张量
        images = images.detach().cpu()
        masks = masks.detach().cpu()
        predictions = predictions.detach().cpu()

        # 限制样本数
        batch_size = min(images.shape[0], max_samples)

        # 创建图像网格
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))

        # 单个样本的情况
        if batch_size == 1:
            axes = [axes]

        for i in range(batch_size):
            # 获取单个样本
            img = images[i]
            mask = masks[i]
            pred = predictions[i]

            # 反归一化图像
            if denorm:
                img = denormalize(img)

            # 确保维度正确
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            if len(pred.shape) == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)

            # 将张量转换为NumPy数组
            img_np = img.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()
            pred_np = pred.numpy()

            # 显示原始图像
            axes[i][0].imshow(img_np)
            axes[i][0].set_title('Input Image')
            axes[i][0].axis('off')

            # 显示目标掩码
            axes[i][1].imshow(mask_np, cmap='gray')
            axes[i][1].set_title('Ground Truth')
            axes[i][1].axis('off')

            # 显示预测掩码
            axes[i][2].imshow(pred_np, cmap='gray')
            axes[i][2].set_title('Prediction')
            axes[i][2].axis('off')

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"保存批次可视化结果到 {save_path}")

    except Exception as e:
        logger.error(f"保存批次结果时出错: {str(e)}")


def create_overlay_image(image, mask, color=(0, 255, 0), alpha=0.5):
    try:
        # 将张量转换为NumPy数组
        if isinstance(image, torch.Tensor):
            # 确保范围为 [0, 1]
            if image.max() <= 1.0:
                image = (image * 255).byte()
            image = image.permute(1, 2, 0).cpu().numpy()

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # 确保二值掩码
        binary_mask = (mask > 0.5).astype(np.uint8)

        # 创建彩色掩码
        color_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
        color_mask[binary_mask == 1] = color

        # 创建叠加图像
        image_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(color_mask)

        # 叠加
        overlay = Image.blend(image_pil, mask_pil, alpha)

        return overlay

    except Exception as e:
        logger.error(f"创建叠加图像时出错: {str(e)}")
        # 返回原始图像
        return Image.fromarray(image.astype(np.uint8))


def visualize_edge_map(edge_map, save_path=None):
    try:
        # 确保是CPU张量并转换为NumPy数组
        if isinstance(edge_map, torch.Tensor):
            edge_map = edge_map.detach().cpu()
            if len(edge_map.shape) == 3 and edge_map.shape[0] == 1:
                edge_map = edge_map.squeeze(0)
            edge_map = edge_map.numpy()

        # 创建图像
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(edge_map, cmap='jet')
        fig.colorbar(im, ax=ax)
        ax.set_title('Edge Map')
        ax.axis('off')

        # 保存图像
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            logger.info(f"保存边缘图到 {save_path}")

        return fig

    except Exception as e:
        logger.error(f"可视化边缘图时出错: {str(e)}")
        return None


def create_comparison_grid(images, masks, predictions, save_path=None, max_samples=4):
    try:
        # 确保输入是CPU张量
        images = [img.detach().cpu() for img in images]
        masks = [mask.detach().cpu() for mask in masks]
        predictions = {k: pred.detach().cpu() for k, pred in predictions.items()}

        # 确定样本数和模型数
        batch_size = min(images[0].shape[0], max_samples)
        num_models = len(predictions)

        # 创建图像网格
        fig, axes = plt.subplots(batch_size, num_models + 2, figsize=(5 * (num_models + 2), 5 * batch_size))

        # 单个样本的情况
        if batch_size == 1:
            axes = [axes]

        # 设置标题
        col_titles = ['Input Image', 'Ground Truth'] + list(predictions.keys())
        for j, title in enumerate(col_titles):
            axes[0][j].set_title(title, fontsize=15)

        for i in range(batch_size):
            # 获取单个样本
            img = images[0][i]
            mask = masks[0][i]

            # 反归一化图像
            img = denormalize(img)

            # 确保维度正确
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

            # 将张量转换为NumPy数组
            img_np = img.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()

            # 显示原始图像
            axes[i][0].imshow(img_np)
            axes[i][0].axis('off')

            # 显示目标掩码
            axes[i][1].imshow(mask_np, cmap='gray')
            axes[i][1].axis('off')

            # 显示各模型的预测掩码
            for j, (model_name, preds) in enumerate(predictions.items()):
                pred = preds[i]
                if len(pred.shape) == 3 and pred.shape[0] == 1:
                    pred = pred.squeeze(0)
                pred_np = pred.numpy()

                axes[i][j + 2].imshow(pred_np, cmap='gray')
                axes[i][j + 2].axis('off')

        # 调整布局并保存
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            logger.info(f"保存对比网格到 {save_path}")

        return fig

    except Exception as e:
        logger.error(f"创建对比网格时出错: {str(e)}")
        return None