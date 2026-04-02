#!/usr/bin/env python3
"""
数据预处理脚本
将原始电商评论数据转换为 ms-swift 所需的格式

ms-swift seq_cls 任务格式:
{"text": "评论文本", "label": 0}
"""

import json
import os
import re
from typing import Dict, List
from collections import Counter

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def clean_text(text: str) -> str:
    """清洗文本数据"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    max_length = 500
    if len(text) > max_length:
        text = text[:max_length]
    return text.strip()


def parse_jd_dataset(data: List[Dict]) -> List[Dict]:
    """
    解析京东评论数据集
    使用 ms-swift seq_cls 格式
    """
    processed_data = []

    for item in data:
        text = item.get('sentence') or item.get('review') or item.get('text')
        raw_label = item.get('label')

        if text is None or raw_label is None:
            continue

        # 转换标签: 1.0=正面(2), 0.0=负面(0)
        try:
            label_float = float(raw_label)
            if label_float == 1.0:
                label = 2  # 正面
            elif label_float == 0.0:
                label = 0  # 负面
            else:
                label = 1  # 中性
        except (ValueError, TypeError):
            continue

        cleaned_text = clean_text(str(text))
        if cleaned_text:
            # ms-swift seq_cls 格式: 直接使用 text 和 label
            processed_data.append({
                "text": cleaned_text,
                "label": label
            })

    return processed_data


def load_raw_data() -> List[Dict]:
    """加载原始数据"""
    possible_paths = [
        os.path.join(DATA_DIR, "jd_reviews.json"),
        os.path.join(DATA_DIR, "mock_data.json"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到数据文件: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

    print("未找到原始数据文件，生成模拟数据...")
    return generate_mock_data()


def generate_mock_data() -> List[Dict]:
    """生成模拟数据"""
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

    mock_file = os.path.join(DATA_DIR, "mock_data.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(mock_file, 'w', encoding='utf-8') as f:
        json.dump(mock_data, f, ensure_ascii=False)

    return mock_data


def split_dataset(data: List[Dict], train_ratio: float = 0.8) -> tuple:
    """划分数据集"""
    import random
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def save_dataset(data: List[Dict], output_file: str):
    """保存为 JSONL 格式"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存 {len(data)} 条数据到: {output_file}")


def main():
    print("=" * 50)
    print("开始预处理数据...")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载原始数据
    raw_data = load_raw_data()
    print(f"原始数据: {len(raw_data)} 条")

    # 解析数据
    processed_data = parse_jd_dataset(raw_data)
    print(f"处理后数据: {len(processed_data)} 条")

    # 标签分布
    labels = [item['label'] for item in processed_data]
    label_counts = Counter(labels)
    print(f"\n标签分布:")
    print(f"  负面(0): {label_counts.get(0, 0)} 条")
    print(f"  中性(1): {label_counts.get(1, 0)} 条")
    print(f"  正面(2): {label_counts.get(2, 0)} 条")

    # 划分数据集
    train_data, val_data = split_dataset(processed_data)

    # 保存
    save_dataset(train_data, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_dataset(val_data, os.path.join(OUTPUT_DIR, "val.jsonl"))

    print(f"\n数据预处理完成!")
    print(f"下一步: python train.py")


if __name__ == "__main__":
    main()
