#!/usr/bin/env python3
"""
ms-swift 数据格式转换脚本
将其他格式的数据集转换为 ms-swift 所需的格式

支持的数据格式:
1. 原始格式: {"text": "...", "label": 0}
2. CSV格式: text,label
3. Excel格式: text, label
4. 多标签格式: {"text": "...", "labels": [0, 1]}
"""

import json
import os
import argparse
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# 标签映射配置
DEFAULT_LABEL_MAPPING = {
    # 二分类
    "binary": {
        "负面": 0, "差": 0, "坏": 0, "1": 0, "0": 0, "negative": 0, "neg": 0, "bad": 0,
        "正面": 1, "好": 1, "棒": 1, "赞": 1, "1": 1, "positive": 1, "pos": 1, "good": 1,
    },
    # 三分类 (情感分析)
    "sentiment_3": {
        "负面": 0, "差评": 0, "差": 0, "不满意": 0, "negative": 0, "neg": 0,
        "中性": 1, "一般": 1, "中评": 1, "neutral": 1,
        "正面": 2, "好评": 2, "满意": 2, "positive": 2, "pos": 2,
    },
    # 多分类 (评分)
    "rating_5": {
        "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
    }
}

def parse_label(label, label_type: str = "sentiment_3") -> int:
    """
    解析标签值
    """
    # 如果已经是数字，直接返回
    if isinstance(label, int):
        return label

    # 如果是字符串数字
    if isinstance(label, str):
        label = label.strip()
        if label.isdigit():
            return int(label)

        # 尝试使用映射表
        mapping = DEFAULT_LABEL_MAPPING.get(label_type, {})
        if label in mapping:
            return mapping[label]

        # 尝试忽略空白后匹配
        for key, value in mapping.items():
            if key in label:
                return value

    return 1  # 默认返回中性

def load_json_file(file_path: str) -> List[Dict]:
    """
    加载 JSON 文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

def load_csv_file(file_path: str) -> List[Dict]:
    """
    加载 CSV 文件
    """
    df = pd.read_csv(file_path)
    return df.to_dict('records')

def load_excel_file(file_path: str) -> List[Dict]:
    """
    加载 Excel 文件
    """
    df = pd.read_excel(file_path)
    return df.to_dict('records')

def convert_to_swift_format(
    data: List[Dict],
    text_field: str = "text",
    label_field: str = "label",
    label_type: str = "sentiment_3",
    prompt_template: Optional[str] = None
) -> List[Dict]:
    """
    转换为 ms-swift 格式

    参数:
        data: 原始数据列表
        text_field: 文本字段名
        label_field: 标签字段名
        label_type: 标签类型 (binary, sentiment_3, rating_5)
        prompt_template: 自定义提示模板，留空使用默认模板
    """
    converted = []

    # 默认提示模板
    if prompt_template is None:
        prompt_template = "请判断这条电商评论的情感类别（0=负面，1=中性，2=正面）：{text}"

    for item in data:
        # 获取文本
        text = None
        for field in [text_field, 'text', 'content', 'comment', 'review']:
            if field in item and item[field]:
                text = str(item[field]).strip()
                break

        if not text:
            continue

        # 获取标签
        label = None
        for field in [label_field, 'label', 'sentiment', 'rating', 'star', 'labels']:
            if field in item:
                label = item[field]
                break

        if label is None:
            continue

        # 处理多标签情况
        if isinstance(label, list):
            label = label[0] if label else 1

        # 解析标签值
        parsed_label = parse_label(label, label_type)

        # 构建 ms-swift 格式
        converted_item = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt_template.format(text=text)
                }
            ],
            "label": parsed_label
        }

        converted.append(converted_item)

    return converted

def split_and_save(
    data: List[Dict],
    output_dir: str,
    train_ratio: float = 0.8,
    train_file: str = "train.jsonl",
    val_file: str = "val.jsonl",
    test_file: str = "test.jsonl",
    seed: int = 42
):
    """
    划分数据集并保存
    """
    import random
    random.seed(seed)
    random.shuffle(data)

    # 计算划分点
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * (1 - train_ratio) / 2)

    # 划分数据
    train_data = data[:train_end]
    val_data = data[train_end:val_end] if val_end < n else []
    test_data = data[val_end:] if val_end < n else []

    # 保存数据
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    if train_data:
        train_path = os.path.join(output_dir, train_file)
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        results['train'] = train_path

    if val_data:
        val_path = os.path.join(output_dir, val_file)
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        results['val'] = val_path

    if test_data:
        test_path = os.path.join(output_dir, test_file)
        with open(test_path, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        results['test'] = test_path

    return results

def generate_dataset_info(
    output_dir: str,
    data: List[Dict],
    label_type: str,
    num_labels: int
):
    """
    生成数据集信息文件
    """
    from collections import Counter

    labels = [item['label'] for item in data]
    label_counts = Counter(labels)

    info = {
        "task_type": "seq_cls",
        "num_labels": num_labels,
        "label_type": label_type,
        "total_samples": len(data),
        "label_distribution": dict(label_counts)
    }

    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return info

def main():
    parser = argparse.ArgumentParser(description='转换数据集为 ms-swift 格式')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_DIR, help='输出目录')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'excel', 'jsonl'],
                        default='json', help='输入文件格式')
    parser.add_argument('--text_field', '-t', default='text', help='文本字段名')
    parser.add_argument('--label_field', '-l', default='label', help='标签字段名')
    parser.add_argument('--label_type', choices=['binary', 'sentiment_3', 'rating_5', 'custom'],
                        default='sentiment_3', help='标签类型')
    parser.add_argument('--num_labels', type=int, default=3, help='分类数量')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--prompt', help='自定义提示模板')

    args = parser.parse_args()

    print("=" * 50)
    print("ms-swift 数据格式转换工具")
    print("=" * 50)

    # 加载数据
    print(f"\n加载数据: {args.input}")
    if args.format == 'csv':
        raw_data = load_csv_file(args.input)
    elif args.format == 'excel':
        raw_data = load_excel_file(args.input)
    else:
        raw_data = load_json_file(args.input)

    print(f"原始数据: {len(raw_data)} 条")

    # 转换格式
    print(f"\n转换数据格式...")
    converted_data = convert_to_swift_format(
        raw_data,
        text_field=args.text_field,
        label_field=args.label_field,
        label_type=args.label_type,
        prompt_template=args.prompt
    )

    print(f"转换后数据: {len(converted_data)} 条")

    # 划分并保存
    print(f"\n划分数据集 (训练集比例: {args.train_ratio})...")
    saved_files = split_and_save(
        converted_data,
        args.output,
        train_ratio=args.train_ratio
    )

    print("\n保存的文件:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")

    # 生成数据集信息
    info = generate_dataset_info(args.output, converted_data, args.label_type, args.num_labels)

    print(f"\n数据集信息:")
    print(f"  任务类型: {info['task_type']}")
    print(f"  分类数量: {info['num_labels']}")
    print(f"  总样本数: {info['total_samples']}")
    print(f"  标签分布: {info['label_distribution']}")

    print("\n转换完成!")

if __name__ == "__main__":
    main()
