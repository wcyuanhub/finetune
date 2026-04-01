#!/usr/bin/env python3
"""
数据预处理脚本
将原始电商评论数据转换为 ms-swift 所需的格式

ms-swift 文本分类数据格式:
{
    "messages": [{"role": "user", "content": "评论文本"}],
    "label": 0  # 0=负面, 1=中性, 2=正面
}
"""

import json
import os
import re
from typing import Dict, List, Any
from collections import Counter

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def clean_text(text: str) -> str:
    """
    清洗文本数据
    - 去除多余空白字符
    - 去除特殊符号
    - 限制文本长度
    """
    if not text:
        return ""

    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)

    # 去除常见HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 去除URL
    text = re.sub(r'http[s]?://\S+', '', text)

    # 限制最大长度 (Qwen2.5-1.5B 推荐 512 tokens ~= 1000 字符)
    max_length = 1000
    if len(text) > max_length:
        text = text[:max_length]

    return text.strip()

def parse_jd_dataset(data: List[Dict]) -> List[Dict]:
    """
    解析京东评论数据集
    根据实际数据结构调整解析逻辑
    """
    processed_data = []

    for item in data:
        # 尝试多种可能的字段名
        text = None
        label = None

        # 常见的评论字段
        text_candidates = ['review', 'text', 'content', 'comment', '整体评论', '评价内容']
        for field in text_candidates:
            if field in item and item[field]:
                text = item[field]
                break

        # 常见的标签字段
        label_candidates = ['label', 'sentiment', '情感', 'rating', 'star']
        for field in label_candidates:
            if field in item:
                label = item[field]
                break

        # 如果没有找到合适的标签，尝试从评论内容推断
        if label is None and text:
            # 简单规则：包含负面词汇 -> 负面
            negative_words = ['差', '坏', '不好', '失望', '退货', '退款', '烂', '垃圾', '坑', '后悔']
            positive_words = ['好', '棒', '赞', '喜欢', '满意', '推荐', '优秀', '超值', '值得']

            text_lower = text.lower()
            neg_count = sum(1 for w in negative_words if w in text_lower)
            pos_count = sum(1 for w in positive_words if w in text_lower)

            if neg_count > pos_count:
                label = 0  # 负面
            elif pos_count > neg_count:
                label = 2  # 正面
            else:
                label = 1  # 中性

        if text and label is not None:
            cleaned_text = clean_text(text)
            if cleaned_text:
                processed_data.append({
                    "messages": [
                        {"role": "user", "content": f"请判断这条电商评论的情感类别（0=负面，1=中性，2=正面）：{cleaned_text}"}
                    ],
                    "label": int(label) if isinstance(label, (int, str)) else 1
                })

    return processed_data

def load_raw_data() -> List[Dict]:
    """
    加载原始数据
    尝试从多个可能的位置加载数据
    """
    possible_paths = [
        os.path.join(DATA_DIR, "DAMO_NLP", "jd", "default", "train.json"),
        os.path.join(DATA_DIR, "train.json"),
        os.path.join(DATA_DIR, "jd_train.json"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到数据文件: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

    # 如果没有找到文件，生成模拟数据进行测试
    print("未找到原始数据文件，生成模拟数据进行测试...")
    return generate_mock_data()

def generate_mock_data() -> List[Dict]:
    """
    生成模拟电商评论数据用于测试流程
    实际使用时应该用真实数据
    """
    mock_data = [
        {"text": "这个商品质量很好，物流很快，很满意！", "label": 2},
        {"text": "东西不错，性价比很高，值得购买。", "label": 2},
        {"text": "一般般，没有描述的那么好。", "label": 1},
        {"text": "太失望了，质量很差，完全不值这个价。", "label": 0},
        {"text": "客服态度很好，商品也很棒，强烈推荐！", "label": 2},
        {"text": "物流太慢了，等了好久。", "label": 0},
        {"text": "还行吧，中规中矩，没有特别出彩的地方。", "label": 1},
        {"text": "太差了，买来就是坏的，要求退货。", "label": 0},
        {"text": "非常满意，会回购的，好评！", "label": 2},
        {"text": "包装破损，产品有划痕，不推荐购买。", "label": 0},
    ]

    # 将模拟数据保存到文件
    mock_file = os.path.join(DATA_DIR, "mock_data.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(mock_file, 'w', encoding='utf-8') as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)

    return mock_data

def split_dataset(data: List[Dict], train_ratio: float = 0.8) -> tuple:
    """
    划分训练集和验证集
    """
    import random
    random.seed(42)
    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return train_data, val_data

def save_dataset(data: List[Dict], output_file: str):
    """
    保存处理后的数据集为 JSONL 格式
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"已保存 {len(data)} 条数据到: {output_file}")

def main():
    print("=" * 50)
    print("开始预处理数据...")
    print("=" * 50)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载原始数据
    raw_data = load_raw_data()
    print(f"原始数据: {len(raw_data)} 条")

    # 解析数据
    processed_data = parse_jd_dataset(raw_data)
    print(f"处理后数据: {len(processed_data)} 条")

    # 统计标签分布
    labels = [item['label'] for item in processed_data]
    label_counts = Counter(labels)
    print(f"\n标签分布:")
    print(f"  负面(0): {label_counts.get(0, 0)} 条")
    print(f"  中性(1): {label_counts.get(1, 0)} 条")
    print(f"  正面(2): {label_counts.get(2, 0)} 条")

    # 划分数据集
    train_data, val_data = split_dataset(processed_data)

    # 保存数据
    train_file = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_file = os.path.join(OUTPUT_DIR, "val.jsonl")

    save_dataset(train_data, train_file)
    save_dataset(val_data, val_file)

    # 保存数据集配置信息
    config_file = os.path.join(OUTPUT_DIR, "dataset_info.json")
    config = {
        "task_type": "seq_cls",
        "num_labels": 3,
        "labels": ["负面", "中性", "正面"],
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "max_length": 1000
    }

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n数据集配置已保存到: {config_file}")
    print("\n数据预处理完成!")
    print(f"\n接下来可以运行 train.sh 开始训练")

if __name__ == "__main__":
    main()
