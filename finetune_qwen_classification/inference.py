#!/usr/bin/env python3
"""
微调模型推理测试脚本
用于测试训练好的模型在电商评论分类任务上的效果
"""

import os
import sys
import json
import torch
from typing import List, Dict

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "checkpoint-xxx")  # 替换为实际路径

# 标签映射
LABEL_MAP = {
    0: "负面",
    1: "中性",
    2: "正面"
}

# 示例评论
SAMPLE_REVIEWS = [
    "这个商品质量很好，物流也很快，很满意！",
    "东西一般，没有描述的那么好，有点失望。",
    "太差了，买来就是坏的，要求退货。",
    "客服态度不错，东西也不错，好评！",
    "等了好久才收到，包装还破了。",
    "性价比很高，值得购买，推荐！",
    "中规中矩吧，没有特别出彩的地方。",
    "非常好用，已经回购好几次了！",
]

def load_model(model_path: str):
    """
    加载微调后的模型
    """
    from swift.llm import get_model_tokenizer, inference

    print(f"加载模型: {model_path}")

    # 获取模型和分词器
    model, tokenizer = get_model_tokenizer(model_path)

    # 如果有 LoRA 权重，合并
    # model = merge_lora(model, lora_path)

    model.eval()

    return model, tokenizer

def predict_single(model, tokenizer, text: str) -> Dict:
    """
    对单条评论进行预测
    """
    # 构建提示
    prompt = f"请判断这条电商评论的情感类别（0=负面，1=中性，2=正面）：{text}"

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False
        )

    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取预测结果 (简化处理)
    try:
        # 尝试从回复中提取数字
        for char in response:
            if char.isdigit() and int(char) in [0, 1, 2]:
                return {
                    "text": text,
                    "prediction": int(char),
                    "label": LABEL_MAP[int(char)],
                    "response": response
                }
    except:
        pass

    return {
        "text": text,
        "prediction": -1,
        "label": "未知",
        "response": response
    }

def batch_predict(model, tokenizer, texts: List[str]) -> List[Dict]:
    """
    批量预测
    """
    results = []
    for text in texts:
        result = predict_single(model, tokenizer, text)
        results.append(result)
    return results

def print_results(results: List[Dict]):
    """
    打印预测结果
    """
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)

    correct = 0
    for i, result in enumerate(results, 1):
        status = "✓" if result["prediction"] != -1 else "?"
        print(f"\n[{i}] {status} {result['text'][:40]}...")
        if result["prediction"] != -1:
            print(f"    预测: {result['label']} ({result['prediction']})")
        else:
            print(f"    预测: 解析失败")
            print(f"    原始回复: {result['response']}")

    print("\n" + "=" * 60)

def main():
    print("=" * 60)
    print("Qwen2.5-1.5B 电商评论分类推理测试")
    print("=" * 60)

    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        print(f"\n警告: 模型路径不存在: {MODEL_PATH}")
        print("请先训练模型，或更新 MODEL_PATH 为正确的路径")
        print("\n使用模拟数据进行演示...")

        # 模拟预测结果
        results = []
        import random
        for text in SAMPLE_REVIEWS:
            pred = random.choice([0, 1, 2])
            results.append({
                "text": text,
                "prediction": pred,
                "label": LABEL_MAP[pred],
                "response": ""
            })

        print_results(results)
        return

    try:
        # 加载模型
        model, tokenizer = load_model(MODEL_PATH)

        # 批量预测
        print("\n正在预测...")
        results = batch_predict(model, tokenizer, SAMPLE_REVIEWS)

        # 打印结果
        print_results(results)

    except Exception as e:
        print(f"\n推理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
