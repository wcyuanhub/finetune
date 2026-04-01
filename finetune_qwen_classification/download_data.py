#!/usr/bin/env python3
"""
ModelScope 数据集下载脚本
下载京东电商评论数据集用于文本分类微调
"""

import os
import json
from modelscope.msdatasets import MsDataset

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "jd_reviews.json")

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
    dataset = MsDataset.load(
        'DAMO_NLP/jd',
        split='train',
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
            print(f"  {key}: {str(value)[:100]}...")

    return dataset

def save_dataset_to_file(dataset):
    """
    将数据集保存为本地 JSON 文件，供后续处理使用
    """
    print(f"\n保存数据集到本地文件...")

    # 转换为列表格式
    data_list = []
    for i, item in enumerate(dataset):
        data_list.append({
            'sentence': item.get('sentence', ''),
            'label': item.get('label', 1.0),
            'dataset': item.get('dataset', 'jd')
        })

    # 保存为 JSON 文件
    with open(RAW_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False)

    print(f"已保存 {len(data_list)} 条数据到: {RAW_DATA_FILE}")
    return RAW_DATA_FILE

def download_alternative_dataset():
    """
    备选数据集: 如果主数据集下载失败，尝试下载替代数据集
    """
    print("\n尝试下载备选数据集...")
    try:
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
            # 保存为本地文件
            save_dataset_to_file(dataset)

            # 保存数据集信息
            info_file = os.path.join(DATA_DIR, "dataset_info.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"数据集名称: DAMO_NLP/jd\n")
                f.write(f"数据条数: {len(dataset)}\n")
                f.write(f"下载时间: 2026-04-01\n")
                f.write(f"本地文件: {RAW_DATA_FILE}\n")
                if len(dataset) > 0:
                    f.write(f"\n数据集字段: {list(dataset[0].keys())}\n")
            print(f"\n数据集信息已保存到: {info_file}")
            print("数据下载完成!")
            print(f"\n接下来运行: python preprocess_data.py")
        else:
            print("所有数据集下载均失败，请检查网络连接。")

    except Exception as e:
        print(f"下载过程中出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n备选方案: 手动下载数据集到 data/ 目录")
        print("访问: https://modelscope.cn/datasets/DAMO_NLP/jd")

if __name__ == "__main__":
    main()
