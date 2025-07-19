# ABIBE: Adaptive Building Information-Based Extraction from Remote Sensing Imagery Using Vision-Language Models

**Good news! Our paper has been officially accepted by *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*.**

**好消息！我们的论文已被 *IEEE Transactions on Geoscience and Remote Sensing (TGRS)* 正式接收。**

We are currently actively organizing and refining the architecture and documentation of the codebase to ensure its usability and to fully align it with the final published version of the paper. **We expect to fully release all code in this repository by July 31, 2025.**

我们目前正在积极整理和完善代码库的架构与文档，以确保代码的易用性，并使其与最终发表的论文版本完全匹配。**我们预计将在2025年7月31日之前，在此仓库中完整公开所有代码。**

[English](#english) | [中文](#chinese)

<a name="english"></a>

## English

ABIBE is an advanced framework for high-precision building extraction from remote sensing imagery, leveraging vision-language models (VLMs) to enhance feature representation and segmentation accuracy.

### Introduction

Building extraction from remote sensing imagery is a critical task with applications in urban planning, population monitoring, disaster response, and environmental assessment. Traditional deep learning approaches often struggle with complex building structures and diverse imaging conditions.

The ABIBE framework addresses these challenges through several key innovations:

1.  **Hierarchical Feature Transfer Mechanism**: Selectively extracts and adapts multi-scale visual representations from the JanusPro vision-language model, effectively leveraging pre-trained knowledge while addressing domain gaps.
2.  **Dynamic Multi-Scale Feature Fusion**: Adaptively integrates vision-language features with spatial information using text-guided attention to focus on relevant building characteristics.
3.  **Enhanced Loss Function**: Combines BCE, Dice, Focal, and Tversky loss components to address class imbalance and improve segmentation quality in challenging regions.

### Performance

#### WHU Building Dataset

| Method | IoU(%) | Precision(%) | Recall(%) | F1(%) |
| :--- | :--- | :--- | :--- | :--- |
| U-Net | 87.17 | 86.61 | 85.23 | 85.42 |
| ABIBE | **91.23** | **96.04** | **94.80** | **95.41** |

#### INRIA Building Dataset

| Method | IoU(%) | Precision(%) | Recall(%) | F1(%) |
| :--- | :--- | :--- | :--- | :--- |
| U-Net | 73.70 | 84.11 | 78.79 | 80.81 |
| ABIBE | **82.07** | **91.99** | **88.39** | **90.15** |

### Installation

> **Note:** Our paper has been accepted by *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*. We are currently organizing the codebase and expect to release the full code before **July 31, 2025**.

#### Requirements

```
Python >= 3.8
PyTorch >= 1.10.0
CUDA >= 11.3 (for GPU support)
```

#### Dependencies

```
pip install -r requirements.txt
```

### Usage Guide

#### Data Preparation

To run ABIBE, organize your dataset with the following structure:

```
data/
├── dataset_name/
│   ├── train/
│   │   ├── image/
│   │   └── label/
│   ├── val/
│   │   ├── image/
│   │   └── label/
│   └── test/
│       ├── image/
│       └── label/
```

#### Training

```bash
# Train base model
python scripts/train.py --config configs/base_config.yaml

# Train full ABIBE model
python scripts/train.py --config configs/abibe_config.yaml
```

#### Testing

```bash
python scripts/test.py --config configs/abibe_config.yaml --checkpoint path/to/checkpoint
```

#### Inference

```bash
python scripts/predict.py --image path/to/image --config configs/abibe_config.yaml --checkpoint path/to/checkpoint
```

### Citation

If you use ABIBE in your research, please cite our paper:

```
@ARTICLE{11077429,
  author={Deng, Yongtao and Lei, Dajiang and Peng, Yidong and Li, Weisheng and Zhang, Liping},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ABIBE: Adaptive Building Information-Based Extraction From Remote Sensing Imagery Using Vision–Language Models}, 
  year={2025},
  volume={63},
  number={},
  pages={1-17},
  keywords={Feature extraction;Buildings;Remote sensing;Visualization;Adaptation models;Data mining;Semantics;Mathematical models;Data models;Architecture;Building extraction;cross-modal fusion;deep learning;hierarchical feature transfer;remote sensing;semantic segmentation;vision–language models (VLMs)},
  doi={10.1109/TGRS.2025.3587692}}
```

### License

This project is licensed under the [MIT License](LICENSE).

---

<a name="chinese"></a>

## 中文

ABIBE是一个利用视觉语言模型（VLM）增强的遥感图像建筑物提取框架，旨在实现高精度的建筑物分割。

### 简介

遥感图像中的建筑物提取是一项关键任务，广泛应用于城市规划、人口监测、灾害响应和环境评估。传统的深度学习方法通常难以处理复杂的建筑结构和多样的成像条件。

ABIBE框架通过以下关键创新解决这些挑战：

1.  **分层特征迁移机制**：从JanusPro视觉语言模型中选择性地提取和适应多尺度视觉表示，有效利用预训练知识同时解决域差异问题。
2.  **动态多尺度特征融合**：通过文本引导的注意力机制，自适应地整合视觉语言特征与空间信息，关注相关的建筑特征。
3.  **增强损失函数**：结合BCE、Dice、Focal和Tversky损失组件，解决类别不平衡问题并提高困难区域的分割质量。

### 性能表现

#### WHU建筑物数据集

| 方法 | IoU(%) | 精确率(%) | 召回率(%) | F1(%) |
| :--- | :--- | :--- | :--- | :--- |
| U-Net | 87.17 | 86.61 | 85.23 | 85.42 |
| ABIBE | **91.23** | **96.04** | **94.80** | **95.41** |

#### INRIA建筑物数据集

| 方法 | IoU(%) | 精确率(%) | 召回率(%) | F1(%) |
| :--- | :--- | :--- | :--- | :--- |
| U-Net | 73.70 | 84.11 | 78.79 | 80.81 |
| ABIBE | **82.07** | **91.99** | **88.39** | **90.15** |

### 安装说明

> **请注意:** 我们的论文已被 *IEEE Transactions on Geoscience and Remote Sensing (TGRS)* 接收。我们正在整理代码库，完整代码预计将于**2025年7月31日**前发布。

#### 环境要求

```
Python >= 3.8
PyTorch >= 1.10.0
CUDA >= 11.3 (GPU支持)
```

#### 依赖包

```
pip install -r requirements.txt
```

### 使用指南

#### 数据准备

运行ABIBE需要按以下结构组织数据集：

```
data/
├── dataset_name/
│   ├── train/
│   │   ├── image/
│   │   └── label/
│   ├── val/
│   │   ├── image/
│   │   └── label/
│   └── test/
│       ├── image/
│       └── label/
```

#### 训练模型

```bash
# 训练基础模型
python scripts/train.py --config configs/base_config.yaml

# 训练完整ABIBE模型
python scripts/train.py --config configs/abibe_config.yaml
```

#### 测试模型

```bash
python scripts/test.py --config configs/abibe_config.yaml --checkpoint path/to/checkpoint
```

#### 预测

```bash
python scripts/predict.py --image path/to/image --config configs/abibe_config.yaml --checkpoint path/to/checkpoint
```

### 引用

如果您在研究中使用了ABIBE，请引用我们的论文：

```
@ARTICLE{11077429,
  author={Deng, Yongtao and Lei, Dajiang and Peng, Yidong and Li, Weisheng and Zhang, Liping},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ABIBE: Adaptive Building Information-Based Extraction From Remote Sensing Imagery Using Vision–Language Models}, 
  year={2025},
  volume={63},
  number={},
  pages={1-17},
  keywords={Feature extraction;Buildings;Remote sensing;Visualization;Adaptation models;Data mining;Semantics;Mathematical models;Data models;Architecture;Building extraction;cross-modal fusion;deep learning;hierarchical feature transfer;remote sensing;semantic segmentation;vision–language models (VLMs)},
  doi={10.1109/TGRS.2025.3587692}}
```

### 许可证

本项目采用 [MIT 许可证](LICENSE)。
