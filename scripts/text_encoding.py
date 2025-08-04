import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime
from janus.models import MultiModalityCausalLM, VLChatProcessor


class MultimodalFeatureExtractor:
    def __init__(self, model_path="../models/janus_pro_1b"):
        """初始化特征提取器"""
        # 设置日志
        self.setup_logging()

        self.model_path = Path(model_path).resolve()
        self.logger.info(f"正在加载模型，路径: {self.model_path}")

        try:
            self.vl_chat_processor = VLChatProcessor.from_pretrained(str(self.model_path), legacy=False)
            self.logger.info("成功加载 VLChatProcessor")

            self.vl_gpt = MultiModalityCausalLM.from_pretrained(str(self.model_path), trust_remote_code=True)
            self.logger.info("成功加载 MultiModalityCausalLM")

            self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
            self.logger.info("模型已成功移至GPU并设置为评估模式")
        except Exception as e:
            self.logger.error(f"模型加载出错: {str(e)}")
            raise

    def setup_logging(self):
        """设置日志记录"""
        self.logger = logging.getLogger('FeatureExtractor')
        self.logger.setLevel(logging.INFO)

        # 创建logs目录（如果不存在）
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # 创建文件处理器，使用当前时间作为文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'logs/feature_extractor_{timestamp}.log', encoding='utf-8')
        fh.setLevel(logging.INFO)

        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 添加处理器到logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def extract_text_features(self, text):
        """提取单个文本的特征"""
        try:
            with torch.no_grad():
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": text,
                        "images": []
                    },
                    {"role": "<|Assistant|>", "content": ""}
                ]

                prepare_inputs = self.vl_chat_processor(
                    conversations=conversation,
                    images=[],
                    force_batchify=True
                ).to(self.vl_gpt.device)

                inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                if isinstance(inputs_embeds, dict):
                    features = inputs_embeds.get('input_embeds',
                                                 inputs_embeds.get('text_embeds',
                                                                   inputs_embeds))
                else:
                    features = inputs_embeds

                # 获取平均池化特征
                features = torch.mean(features, dim=1)  # Shape变为 [1, 2048]
                return features
        except Exception as e:
            self.logger.error(f"特征提取出错: {str(e)}")
            raise

    def check_existing_features(self, text_files, output_dir):
        """检查已存在的特征文件"""
        existing_features = {}
        batch_features_path = os.path.join(output_dir, 'batch_features.pt')

        # 检查是否存在批量特征文件
        if os.path.exists(batch_features_path):
            try:
                batch_data = torch.load(batch_features_path)
                if isinstance(batch_data, dict) and 'file_names' in batch_data:
                    for idx, file_name in enumerate(batch_data['file_names']):
                        feature_path = os.path.join(output_dir, file_name.replace('.txt', '.pt'))
                        if os.path.exists(feature_path):
                            existing_features[file_name] = True
            except Exception as e:
                self.logger.warning(f"读取批量特征文件出错: {str(e)}")

        # 检查单个特征文件
        for text_file in text_files:
            feature_path = os.path.join(output_dir, text_file.replace('.txt', '.pt'))
            if os.path.exists(feature_path):
                existing_features[text_file] = True

        return existing_features

    def batch_process_features(self, text_dir, output_dir):
        """批量处理文本特征并保存"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"输出目录已创建/确认: {output_dir}")

            # 获取所有文本文件
            text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
            self.logger.info(f"找到 {len(text_files)} 个文本文件")

            if len(text_files) == 0:
                self.logger.warning(f"在目录 {text_dir} 中没有找到.txt文件")
                return None

            # 检查已存在的特征
            existing_features = self.check_existing_features(text_files, output_dir)
            unprocessed_files = [f for f in text_files if f not in existing_features]

            self.logger.info(f"其中 {len(text_files) - len(unprocessed_files)} 个文件已处理")
            self.logger.info(f"待处理文件数: {len(unprocessed_files)}")

            if len(unprocessed_files) == 0:
                self.logger.info("所有文件都已处理完成")
                # 返回已有的批量特征
                batch_features_path = os.path.join(output_dir, 'batch_features.pt')
                if os.path.exists(batch_features_path):
                    batch_data = torch.load(batch_features_path)
                    return batch_data['features']
                return None

            all_features = []
            file_names = []
            failed_files = []  # 记录处理失败的文件

            self.logger.info("开始处理文本文件...")
            for text_file in tqdm(unprocessed_files, desc="提取特征"):
                try:
                    with open(os.path.join(text_dir, text_file), 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()

                    features = self.extract_text_features(text_content)

                    # 保存单个特征
                    output_file = os.path.join(output_dir, text_file.replace('.txt', '.pt'))
                    torch.save(features, output_file)
                    self.logger.info(f"已保存特征到: {output_file}")

                    all_features.append(features.cpu())
                    file_names.append(text_file)

                except Exception as e:
                    self.logger.error(f"处理文件 {text_file} 时出错: {str(e)}")
                    failed_files.append((text_file, str(e)))
                    continue

            # 加载和整合已存在的特征
            if existing_features:
                batch_features_path = os.path.join(output_dir, 'batch_features.pt')
                if os.path.exists(batch_features_path):
                    try:
                        existing_data = torch.load(batch_features_path)
                        all_features.extend([existing_data['features'][i].unsqueeze(0)
                                             for i, name in enumerate(existing_data['file_names'])
                                             if name in existing_features])
                        file_names.extend([name for name in existing_data['file_names']
                                           if name in existing_features])
                    except Exception as e:
                        self.logger.error(f"加载已存在的特征时出错: {str(e)}")

            # 保存批量特征
            if all_features:
                batch_features = torch.cat(all_features, dim=0)

                batch_output = {
                    'features': batch_features,
                    'file_names': file_names
                }
                batch_output_path = os.path.join(output_dir, 'batch_features.pt')
                torch.save(batch_output, batch_output_path)
                self.logger.info(f"已保存批量特征到: {batch_output_path}")
                self.logger.info(f"批量特征形状: {batch_features.shape}")
                self.logger.info(f"处理的文件总数: {len(file_names)}")

                # 打印失败文件的详细信息
                if failed_files:
                    self.logger.error("\n处理失败的文件列表:")
                    for file_name, error in failed_files:
                        self.logger.error(f"文件: {file_name} - 错误: {error}")

                return batch_features
            else:
                self.logger.warning("警告: 没有成功处理任何特征")
                return None

        except Exception as e:
            self.logger.error(f"批处理过程出错: {str(e)}")
            raise


def main():
    try:
        # 设置路径
        text_dir = "../data/AerialImageDataset/test/text_descriptions"
        output_dir = "../data/AerialImageDataset/test/text_features"
        model_path = "../models/janus_pro_1b"

        # 创建特征提取器
        extractor = MultimodalFeatureExtractor(model_path)
        extractor.logger.info(f"文本目录: {text_dir}")
        extractor.logger.info(f"输出目录: {output_dir}")
        extractor.logger.info(f"模型路径: {model_path}")

        # 检查输入目录是否存在
        if not os.path.exists(text_dir):
            raise FileNotFoundError(f"文本目录不存在: {text_dir}")

        # 批量处理特征
        features = extractor.batch_process_features(text_dir, output_dir)

        if features is not None:
            extractor.logger.info("\n特征提取完成!")
            extractor.logger.info(f"特征已保存到: {output_dir}")
        else:
            extractor.logger.warning("\n特征提取过程中出现问题，请检查上述错误信息")

    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()