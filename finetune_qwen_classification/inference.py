#!/usr/bin/env python3
"""
电商评论分类器 - 交互式推理
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def load_model():
    """加载模型"""
    model_path = "./output/final_model"

    if not os.path.exists(model_path):
        print("错误: 模型不存在！请先运行 python train.py")
        return None, None

    print("正在加载模型...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检查是否是 LoRA 模型
    peft_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(peft_config_path):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            num_labels=3,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    print("模型加载完成！\n")
    return model, tokenizer


def predict(text: str, model, tokenizer):
    """预测文本"""
    id2label = {0: "负面", 1: "中性", 2: "正面"}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][pred_id].item()

    label = id2label[pred_id]
    return label, confidence


def show_menu():
    """显示菜单"""
    print("=" * 50)
    print("       电商评论分类器")
    print("=" * 50)
    print()
    print("  1. 测试样例")
    print("  2. 手动输入评论")
    print("  3. 退出")
    print()


def run_examples(model, tokenizer):
    """运行测试样例"""
    examples = [
        ("这个商品质量很好，物流也很快，非常满意！", "预期: 正面"),
        ("东西不错，性价比很高，值得购买。", "预期: 正面"),
        ("一般般，没有描述的那么好。", "预期: 中性"),
        ("还行吧，中规中矩。", "预期: 中性"),
        ("太差了，质量很差，完全不值这个价。", "预期: 负面"),
        ("物流太慢了，等了好久。", "预期: 负面"),
        ("客服态度恶劣，产品有损坏，不推荐。", "预期: 负面"),
    ]

    print()
    print("=" * 50)
    print("       测试样例")
    print("=" * 50)
    print()

    for i, (text, expected) in enumerate(examples, 1):
        label, confidence = predict(text, model, tokenizer)
        status = "正确" if expected.startswith(label) else "错误"
        print(f"[{i}] {label} (置信度: {confidence:.2%}) {status}")
        print(f"    {text}")
        print(f"    {expected}")
        print()


def manual_input(model, tokenizer):
    """手动输入评论"""
    print()
    print("=" * 50)
    print("       手动输入评论")
    print("=" * 50)
    print()
    print("输入评论后按回车预测，输入 'q' 返回菜单")
    print()

    while True:
        text = input("请输入评论: ").strip()

        if text.lower() == 'q':
            break

        if not text:
            continue

        label, confidence = predict(text, model, tokenizer)

        print()
        print("-" * 30)
        print(f"预测结果: {label}")
        print(f"置信度: {confidence:.2%}")
        print("-" * 30)
        print()


def main():
    clear_screen()
    print("正在初始化...")

    result = load_model()
    if result[0] is None:
        return

    model, tokenizer = result

    while True:
        clear_screen()
        show_menu()

        choice = input("请选择 (1-3): ").strip()

        if choice == '1':
            clear_screen()
            run_examples(model, tokenizer)
            input("\n按回车返回菜单...")

        elif choice == '2':
            clear_screen()
            manual_input(model, tokenizer)

        elif choice == '3':
            print("\n再见！")
            break

        else:
            print("\n无效选择，请重新输入...")
            input()


if __name__ == "__main__":
    main()
