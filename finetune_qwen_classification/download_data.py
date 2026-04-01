#!/usr/bin/env python3
"""
ModelScope 数据集下载脚本
下载京东电商评论数据集用于文本分类微调
"""

import os
from modelscope.msdatasets import MsDataset

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def download_jd_reviews():
    """
    下载京东电商评论数据集
    数据集来源: DAMO_NLP/jd (ModelScope)
    包含商品评论及对应的情感标签
    """
    print("=" * 50)
    print("开始下载京东电商评论数据集...")
    print("=" * 50)

    # 加载数据集
    # 该数据集包含商品的正面评论、负面评论和整体评论
    dataset = MsDataset.load(
        'DAMO_NLP/jd',
        split='train',  # 使用训练集
        cache_dir=DATA_DIR
    )

    print(f"\n数据集加载成功!")
    print(f"数据集大小: {len(dataset)} 条")

    # 查看数据集结构
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n数据集字段: {list(sample.keys())}")
        print(f"\n样例数据:")
        for key, value in sample.items():
            print(f"  {key}: {str(value)[:200]}...")

    return dataset

def download_alternative_dataset():
    """
    备选数据集: 如果主数据集下载失败，尝试下载替代数据集
    """
    print("\n尝试下载备选数据集...")
    try:
        # 使用中文通用评论数据集
        dataset = MsDataset.load(
            'tyd弯弯/XED_Datasets',
            split='train',
            cache_dir=DATA_DIR
        )
        print(f"备选数据集加载成功，大小: {len(dataset)} 条")
        return dataset
    except Exception as e:
        print(f"备选数据集下载失败: {e}")
        return None

def main():
    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"数据将保存到: {DATA_DIR}")

    try:
        dataset = download_jd_reviews()
        if dataset is None:
            print("主数据集下载失败，尝试备选方案...")
            dataset = download_alternative_dataset()

        if dataset:
            # 保存数据集信息
            info_file = os.path.join(DATA_DIR, "dataset_info.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"数据集名称: DAMO_NLP/jd\n")
                f.write(f"数据条数: {len(dataset)}\n")
                f.write(f"下载时间: 2026-04-01\n")
                if len(dataset) > 0:
                    f.write(f"\n数据集字段: {list(dataset[0].keys())}\n")
            print(f"\n数据集信息已保存到: {info_file}")
            print("数据下载完成!")
        else:
            print("所有数据集下载均失败，请检查网络连接。")

    except Exception as e:
        print(f"下载过程中出错: {e}")
        print("\n备选方案: 手动下载数据集到 data/ 目录")
        print("访问: https://modelscope.cn/datasets/DAMO_NLP/jd")

if __name__ == "__main__":
    main()
