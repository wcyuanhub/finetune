#!/usr/bin/env python3
"""
Qwen2.5-1.5B 电商评论分类推理脚本
用于测试训练好的模型
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig


# 默认配置
DEFAULT_MODEL_PATH = "./output/final_model"
DEFAULT_PROMPT_TEMPLATE = """判断以下电商评论的情感是正面、中性还是负面：

评论：{text}
情感："""


def load_model(model_path: str, use_lora: bool = True):
    """加载模型"""
    print(f"[INFO] 加载模型: {model_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检查是否是 LoRA 模型
    peft_config_path = os.path.join(model_path, "adapter_config.json")

    if use_lora and os.path.exists(peft_config_path):
        print("[INFO] 检测到 LoRA 模型")
        # 加载基础模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            num_labels=3,
        )
        # 加载 LoRA 权重
        model = PeftModel.from_pretrained(base_model, model_path)
        print("[INFO] LoRA 模型加载成功")
    else:
        print("[INFO] 加载完整模型")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


def predict(text: str, model, tokenizer, id2label=None):
    """预测单条文本"""
    if id2label is None:
        id2label = {0: "负面", 1: "中性", 2: "正面"}

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][pred_id].item()

    # Get result
    label = id2label.get(pred_id, "未知")
    all_probs = {id2label[i]: f"{probs[0][i].item():.4f}" for i in range(3)}

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": all_probs,
    }


def interactive_predict(model, tokenizer):
    """交互式预测"""
    print("\n" + "=" * 50)
    print("交互式评论分类")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            text = input("请输入评论: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break

            if not text:
                print("请输入有效内容\n")
                continue

            result = predict(text, model, tokenizer)

            print(f"\n预测结果: {result['label']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"各类别概率: {result['probabilities']}")
            print()

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}\n")


def batch_predict(texts: list, model, tokenizer):
    """批量预测"""
    print(f"[INFO] 批量预测 {len(texts)} 条文本...\n")

    for i, text in enumerate(texts):
        result = predict(text, model, tokenizer)
        print(f"[{i+1}] {result['label']} (置信度: {result['confidence']:.4f})")
        print(f"    原文: {text[:50]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="电商评论分类推理")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径")
    parser.add_argument("--text", type=str, default=None,
                        help="要预测的文本")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式预测模式")
    parser.add_argument("--no_lora", action="store_true",
                        help="不使用 LoRA (完整模型)")
    parser.add_argument("--test", action="store_true",
                        help="运行测试样例")

    args = parser.parse_args()

    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"[ERROR] 模型路径不存在: {args.model_path}")
        print("请先运行训练: python train.py")
        sys.exit(1)

    # 加载模型
    model, tokenizer = load_model(args.model_path, use_lora=not args.no_lora)

    # 测试样例
    if args.test:
        test_texts = [
            "这个商品质量很好，物流也很快，非常满意！",
            "一般般，没有描述的那么好。",
            "太差了，质量很差，完全不值这个价。",
        ]
        batch_predict(test_texts, model, tokenizer)
        return

    # 单条预测
    if args.text:
        result = predict(args.text, model, tokenizer)
        print(f"\n预测结果: {result['label']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"各类别概率: {result['probabilities']}")
        return

    # 交互式模式
    if args.interactive or sys.stdin.isatty():
        interactive_predict(model, tokenizer)
    else:
        # 批量测试
        test_texts = [
            "这个商品质量很好，物流也很快，非常满意！",
            "一般般，没有描述的那么好。",
            "太差了，质量很差，完全不值这个价。",
        ]
        batch_predict(test_texts, model, tokenizer)


if __name__ == "__main__":
    main()
