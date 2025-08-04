import os
import torch
import re
import logging
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from janus.models import MultiModalityCausalLM, VLChatProcessor


class TextGenerator:
    def __init__(self, model_path="../models/janus_pro_1b"):
        """初始化模型和处理器"""
        # 设置日志
        self.setup_logging()

        self.model_path = Path(model_path).resolve()
        self.logger.info(f"正在加载模型，路径: {self.model_path}")

        try:
            self.vl_chat_processor = VLChatProcessor.from_pretrained(
                str(self.model_path),
                legacy=False
            )
            self.logger.info("成功加载 VLChatProcessor")

            self.vl_gpt = MultiModalityCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            self.logger.info("成功加载 MultiModalityCausalLM")

            self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
            self.logger.info("模型已成功移至GPU并设置为评估模式")

        except Exception as e:
            self.logger.error(f"模型加载出错: {str(e)}")
            raise

        # 定义标准提示模板
        self.question_template = """You are an expert in the field of remote sensing architecture extraction. For the following image, answer the following questions:
1. How many buildings are visible in the image?
2. What is the shape of the buildings?
3. How are the buildings distributed in the image? 
Please make sure to answer each question separately and clearly. Answer each question with only one sentence."""

    def setup_logging(self):
        """设置日志记录"""
        self.logger = logging.getLogger('TextGenerator')
        self.logger.setLevel(logging.INFO)

        # 创建logs目录（如果不存在）
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # 创建文件处理器，使用当前时间作为文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'logs/text_generator_{timestamp}.log', encoding='utf-8')
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

    def clean_generated_text(self, text):
        """清理生成的文本，使用更灵活的规则"""
        if not text:
            return text

        # 分割成单独的句子
        sentences = text.split('\n')
        cleaned_sentences = []

        # 定义需要删除的模式（更少的限制）
        patterns_to_remove = [
            r'^(The image |This image |In this image )',  # 只删除句子开头的这些短语
            r'^(Question:|Answer:|Image:|Context:|Description:)\s*',  # 删除特定标记
        ]

        # 用于提取数字序号的模式
        number_pattern = r'^(\d+\.)'

        for sentence in sentences:
            # 跳过空句子
            if not sentence.strip():
                continue

            # 清理句子
            cleaned_sentence = sentence.strip()

            # 移除定义的模式
            for pattern in patterns_to_remove:
                cleaned_sentence = re.sub(pattern, '', cleaned_sentence)

            # 移除多余的空格
            cleaned_sentence = ' '.join(cleaned_sentence.split())

            # 提取可能存在的数字序号
            number_match = re.match(number_pattern, cleaned_sentence)

            # 如果没有数字序号但句子有意义，也保留
            if len(cleaned_sentence) > 5:  # 确保句子有足够的长度
                if not number_match and len(cleaned_sentences) < 3:
                    # 添加适当的序号
                    cleaned_sentence = f"{len(cleaned_sentences) + 1}. {cleaned_sentence}"
                cleaned_sentences.append(cleaned_sentence)

        # 确保只保留前三个有效句子
        cleaned_sentences = cleaned_sentences[:3]

        # 如果没有得到三个句子，生成默认回答
        while len(cleaned_sentences) < 3:
            missing_number = len(cleaned_sentences) + 1
            if missing_number == 1:
                cleaned_sentences.append("1. No buildings are visible in the image.")
            elif missing_number == 2:
                cleaned_sentences.append("2. There are no distinct building shapes to describe.")
            elif missing_number == 3:
                cleaned_sentences.append("3. There is no specific building distribution to describe.")

        return '\n'.join(cleaned_sentences)

    def generate_text_description(self, image_path):
        """为单个图像生成文本描述"""
        try:
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")

            # 构建对话
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{self.question_template}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # 加载和处理图像
            try:
                pil_image = Image.open(image_path)
                # 确保图像被正确加载
                pil_image.verify()
                # 重新打开图像（因为verify会关闭文件）
                pil_image = Image.open(image_path)
            except Exception as e:
                self.logger.error(f"图像打开失败 {image_path}: {str(e)}")
                raise

            # 准备模型输入
            try:
                prepare_inputs = self.vl_chat_processor(
                    conversations=conversation,
                    images=[pil_image],
                    force_batchify=True
                ).to(self.vl_gpt.device)
            except Exception as e:
                self.logger.error(f"处理器准备输入失败 {image_path}: {str(e)}")
                raise

            # 生成文本描述
            try:
                inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                outputs = self.vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.vl_chat_processor.tokenizer.eos_token_id
                )
            except Exception as e:
                self.logger.error(f"生成文本失败 {image_path}: {str(e)}")
                raise

            # 解码并清理输出
            try:
                description = self.vl_chat_processor.tokenizer.decode(
                    outputs[0].cpu().tolist(),
                    skip_special_tokens=True
                )
            except Exception as e:
                self.logger.error(f"解码输出失败 {image_path}: {str(e)}")
                raise

            # 清理生成的文本
            cleaned_description = self.clean_generated_text(description.strip())

            # 即使清理后的文本为空，也返回默认描述
            if not cleaned_description:
                self.logger.warning(f"清理后的文本为空 {image_path}，使用默认描述")
                cleaned_description = (
                    "1. No buildings are visible in the image.\n"
                    "2. There are no distinct building shapes to describe.\n"
                    "3. There is no specific building distribution to describe."
                )

            return cleaned_description

        except Exception as e:
            self.logger.error(f"生成描述时出错 {image_path}: {str(e)}")
            return None

    def process_directory(self, image_dir, output_dir):
        """处理整个目录的图像"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"输出目录已创建/确认: {output_dir}")

        # 获取所有图像文件
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.tif', '.jpg', '.png', '.jpeg'))]

        # 获取已处理的文件列表
        processed_files = set(f.replace('.txt', '') for f in os.listdir(output_dir) if f.endswith('.txt'))

        # 过滤出未处理的图像文件
        unprocessed_files = [f for f in image_files if os.path.splitext(f)[0] not in processed_files]

        total_images = len(unprocessed_files)
        self.logger.info(f"找到 {len(image_files)} 个图像文件")
        self.logger.info(f"其中 {len(image_files) - total_images} 个文件已处理")
        self.logger.info(f"待处理文件数: {total_images}")

        if total_images == 0:
            self.logger.info("所有文件都已处理完成")
            return

        # 处理进度统计
        successful = 0
        failed = 0
        failed_files = []  # 记录处理失败的文件

        # 使用tqdm创建进度条
        for image_name in tqdm(unprocessed_files, desc="生成文本描述"):
            image_path = os.path.join(image_dir, image_name)
            output_path = os.path.join(output_dir, image_name.replace('.tif', '.txt')
                                       .replace('.jpg', '.txt')
                                       .replace('.png', '.txt')
                                       .replace('.jpeg', '.txt'))

            # 生成描述
            description = self.generate_text_description(image_path)

            if description:
                # 保存描述
                try:
                    with open(output_path, "w", encoding='utf-8') as f:
                        f.write(description)
                    successful += 1
                    self.logger.info(f"成功处理文件: {image_name}")
                except Exception as e:
                    failed += 1
                    failed_files.append((image_name, f"保存描述文件失败: {str(e)}"))
                    self.logger.error(f"保存描述文件失败 {output_path}: {str(e)}")
            else:
                failed += 1
                failed_files.append((image_name, "生成描述失败"))
                self.logger.error(f"生成描述失败: {image_name}")

        # 打印最终统计信息
        self.logger.info("\n处理完成!")
        self.logger.info(f"本次成功处理: {successful}/{total_images} 个文件")
        self.logger.info(f"本次处理失败: {failed}/{total_images} 个文件")

        # 打印失败文件的详细信息
        if failed_files:
            self.logger.error("\n处理失败的文件列表:")
            for file_name, error in failed_files:
                self.logger.error(f"文件: {file_name} - 错误: {error}")


def main():
    try:
        # 设置路径
        image_dir = "../data/AerialImageDataset/test/image"
        output_dir = "../data/AerialImageDataset/test/text_descriptions"
        model_path = "../models/janus_pro_1b"

        # 创建生成器实例并处理图像
        generator = TextGenerator(model_path)
        generator.logger.info(f"图像目录: {image_dir}")
        generator.logger.info(f"输出目录: {output_dir}")
        generator.logger.info(f"模型路径: {model_path}")

        # 检查输入目录是否存在
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")

        generator.process_directory(image_dir, output_dir)

    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()