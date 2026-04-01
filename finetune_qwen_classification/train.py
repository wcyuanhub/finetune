#!/usr/bin/env python3
"""
Qwen2.5-1.5B 电商评论文本分类微调训练脚本
使用 ms-swift 框架进行 LoRA 微调
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_cache_dir: str = ""

    # 数据配置
    train_data: str = ""
    val_data: str = ""
    max_length: int = 512

    # 任务配置
    task_type: str = "seq_cls"
    num_labels: int = 3

    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # LoRA 参数
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "ALL"

    # 输出配置
    output_dir: str = "./output"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 1

    # 其他配置
    seed: int = 42
    gradient_checkpointing: bool = True
    fp16: bool = True


def setup_environment(config: TrainingConfig):
    """设置环境变量"""
    if config.model_cache_dir:
        os.environ["MODELSCOPE_CACHE"] = config.model_cache_dir
        print(f"模型缓存目录: {config.model_cache_dir}")


def build_dataset(config: TrainingConfig):
    """构建数据集"""
    from swift.llm import dataset_map, get_dataset

    # 读取本地数据集
    train_dataset = None
    val_dataset = None

    if config.train_data and os.path.exists(config.train_data):
        print(f"加载训练数据: {config.train_data}")
        # 读取 JSONL 文件
        train_dataset = get_dataset(
            config.train_data,
            **({"val_dataset": config.val_data} if config.val_data else {})
        )

    if train_dataset is None:
        raise FileNotFoundError(f"训练数据不存在: {config.train_data}")

    return train_dataset


def run_training(config: TrainingConfig):
    """运行训练"""
    from swift.llm import SftArguments, sft_main

    # 构建命令行参数
    cmd_args = [
        "sft",
        "--model", config.model_name,
        "--dataset", config.train_data,
        "--output_dir", config.output_dir,
        "--num_train_epochs", str(config.num_train_epochs),
        "--per_device_train_batch_size", str(config.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(config.per_device_eval_batch_size),
        "--learning_rate", str(config.learning_rate),
        "--max_length", str(config.max_length),
        "--logging_steps", str(config.logging_steps),
        "--eval_steps", str(config.eval_steps),
        "--save_steps", str(config.save_steps),
        "--save_total_limit", str(config.save_total_limit),
        "--gradient_checkpointing", str(config.gradient_checkpointing).lower(),
        "--report_to", "tensorboard",
    ]

    # 添加验证集
    if config.val_data and os.path.exists(config.val_data):
        cmd_args[-2] = config.train_data + ":" + config.val_data

    # 添加 LoRA 参数
    if config.use_lora:
        cmd_args.extend([
            "--sft_type", "lora",
            "--lora_rank", str(config.lora_rank),
            "--lora_alpha", str(config.lora_alpha),
            "--lora_dropout", str(config.lora_dropout),
            "--lora_target_modules", config.lora_target_modules,
        ])

    # 添加其他参数
    if config.seed:
        cmd_args.extend(["--seed", str(config.seed)])

    if config.warmup_ratio:
        cmd_args.extend(["--warmup_ratio", str(config.warmup_ratio)])

    if config.weight_decay:
        cmd_args.extend(["--weight_decay", str(config.weight_decay)])

    if config.gradient_accumulation_steps > 1:
        cmd_args.extend(["--gradient_accumulation_steps", str(config.gradient_accumulation_steps)])

    # 打印命令
    print("=" * 60)
    print("训练命令:")
    print("swift " + " ".join(cmd_args))
    print("=" * 60)

    # 使用 SftArguments 解析参数
    args = SftArguments.parse_args(cmd_args)

    # 运行训练
    print("\n开始训练...")
    result = sft_main(args)

    return result


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B 文本分类微调训练")

    # 添加命令行参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="模型名称或路径")
    parser.add_argument("--train_data", type=str, default="",
                        help="训练数据路径")
    parser.add_argument("--val_data", type=str, default="",
                        help="验证数据路径")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="是否使用 LoRA")
    parser.add_argument("--no_lora", action="store_true",
                        help="不使用 LoRA")

    args = parser.parse_args()

    # 构建配置
    config = TrainingConfig()

    # 设置路径
    config.model_name = args.model_name
    config.train_data = args.train_data or os.path.join(PROJECT_ROOT, "data/processed/train.jsonl")
    config.val_data = args.val_data or os.path.join(PROJECT_ROOT, "data/processed/val.jsonl")
    config.output_dir = args.output_dir

    # 训练参数
    config.num_train_epochs = args.num_epochs
    config.per_device_train_batch_size = args.batch_size
    config.per_device_eval_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.max_length = args.max_length

    # LoRA 参数
    config.use_lora = not args.no_lora
    config.lora_rank = args.lora_rank

    # 设置环境
    setup_environment(config)

    # 检查数据文件
    if not os.path.exists(config.train_data):
        print(f"错误: 训练数据不存在: {config.train_data}")
        print("请先运行:")
        print("  python download_data.py")
        print("  python preprocess_data.py")
        sys.exit(1)

    # 打印配置
    print("\n" + "=" * 60)
    print("训练配置:")
    print(f"  模型: {config.model_name}")
    print(f"  训练数据: {config.train_data}")
    print(f"  验证数据: {config.val_data}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  训练轮数: {config.num_train_epochs}")
    print(f"  批次大小: {config.per_device_train_batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  LoRA: {'是' if config.use_lora else '否'}")
    if config.use_lora:
        print(f"  LoRA rank: {config.lora_rank}")
    print("=" * 60 + "\n")

    # 运行训练
    import time
    start_time = time.time()

    try:
        run_training(config)
        elapsed = int(time.time() - start_time)
        print("\n" + "=" * 60)
        print(f"训练完成!")
        print(f"总耗时: {elapsed // 60} 分钟 {elapsed % 60} 秒")
        print(f"模型保存位置: {config.output_dir}")
        print("=" * 60)
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
