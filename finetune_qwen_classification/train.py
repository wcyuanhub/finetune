#!/usr/bin/env python3
"""
Qwen2.5-1.5B 电商评论文本分类微调训练脚本
使用 ms-swift 框架进行 LoRA 微调
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# ==============================================================================
# 训练配置 - 直接修改这里的参数
# ==============================================================================

CONFIG = {
    # 模型配置
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",  # ModelScope 模型ID
    "model_cache_dir": "./model_cache",  # 模型缓存目录

    # 数据配置
    "train_data": "./data/processed/train.jsonl",  # 训练数据路径
    "val_data": "./data/processed/val.jsonl",  # 验证数据路径
    "max_length": 512,  # 最大序列长度

    # 训练参数
    "num_train_epochs": 3,  # 训练轮数
    "per_device_train_batch_size": 4,  # 训练批次大小 (24GB显存建议4-8)
    "per_device_eval_batch_size": 4,  # 评估批次大小
    "learning_rate": 1e-4,  # 学习率

    # 输出配置
    "output_dir": "./output",  # 输出目录
    "log_dir": "./logs",  # 日志目录
    "logging_steps": 1,  # 每批次都记录日志
    "eval_steps": 100,  # 评估步数
    "save_steps": 500,  # 保存步数
    "save_total_limit": 1,  # 最多保存的checkpoint数量

    # 其他配置
    "seed": 42,  # 随机种子
    "gradient_checkpointing": True,  # 梯度检查点
}


def setup_environment():
    """设置环境"""
    # 设置模型缓存目录
    if CONFIG["model_cache_dir"]:
        os.makedirs(CONFIG["model_cache_dir"], exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = CONFIG["model_cache_dir"]

    # 创建日志目录
    os.makedirs(CONFIG["log_dir"], exist_ok=True)


def check_data():
    """检查数据文件"""
    train_path = CONFIG["train_data"]
    if not os.path.exists(train_path):
        print(f"错误: 训练数据不存在: {train_path}")
        print("\n请先运行以下命令下载和预处理数据:")
        print("  python download_data.py")
        print("  python preprocess_data.py")
        sys.exit(1)

    print(f"训练数据: {train_path}")
    if CONFIG["val_data"] and os.path.exists(CONFIG["val_data"]):
        print(f"验证数据: {CONFIG['val_data']}")
    print()


def build_command():
    """构建 swift sft 命令"""
    cmd = ["swift", "sft"]

    # 必需参数
    cmd.extend(["--model", CONFIG["model_name"]])
    cmd.extend(["--dataset", CONFIG["train_data"]])
    cmd.extend(["--output_dir", CONFIG["output_dir"]])

    # 添加验证集
    if CONFIG["val_data"] and os.path.exists(CONFIG["val_data"]):
        cmd[-1] = CONFIG["train_data"] + ":" + CONFIG["val_data"]

    # 训练参数
    cmd.extend(["--num_train_epochs", str(CONFIG["num_train_epochs"])])
    cmd.extend(["--per_device_train_batch_size", str(CONFIG["per_device_train_batch_size"])])
    cmd.extend(["--per_device_eval_batch_size", str(CONFIG["per_device_eval_batch_size"])])
    cmd.extend(["--learning_rate", str(CONFIG["learning_rate"])])
    cmd.extend(["--max_length", str(CONFIG["max_length"]])

    # 日志配置 - 关键：每批次都记录
    cmd.extend(["--logging_steps", str(CONFIG["logging_steps"])])
    cmd.extend(["--eval_steps", str(CONFIG["eval_steps"])])
    cmd.extend(["--save_steps", str(CONFIG["save_steps"])])
    cmd.extend(["--save_total_limit", str(CONFIG["save_total_limit"])])

    # TensorBoard - 同时记录到 tensorboard 和 terminal
    cmd.extend(["--report_to", "tensorboard"])

    # 梯度检查点
    if CONFIG["gradient_checkpointing"]:
        cmd.append("--gradient_checkpointing")

    # 随机种子
    cmd.extend(["--seed", str(CONFIG["seed"])])

    return cmd


def get_log_file():
    """获取日志文件路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(CONFIG["log_dir"], f"training_{timestamp}.log")
    return log_file


def print_config():
    """打印配置信息"""
    print("=" * 60)
    print("训练配置:")
    print("=" * 60)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Qwen2.5-1.5B 电商评论文本分类微调训练")
    print("=" * 60 + "\n")

    # 设置环境
    setup_environment()

    # 检查数据
    check_data()

    # 打印配置
    print_config()

    # 检测 GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("GPU: 未检测到 (使用 CPU)")
    except:
        pass
    print()

    # 获取日志文件
    log_file = get_log_file()
    print(f"日志文件: {log_file}")
    print()

    # 构建命令
    cmd = build_command()

    # 打印训练命令
    print("训练命令:")
    print("swift " + " ".join(cmd[1:]))
    print()

    # 开始训练
    print("开始训练...")
    print("-" * 60)

    start_time = time.time()
    step_count = 0
    last_loss = "N/A"

    # 打开日志文件
    log_f = open(log_file, 'w', encoding='utf-8')
    log_f.write("=" * 60 + "\n")
    log_f.write(f"训练开始时间: {datetime.now()}\n")
    log_f.write(f"命令: swift {' '.join(cmd[1:])}\n")
    log_f.write("=" * 60 + "\n\n")
    log_f.flush()

    try:
        # 使用 subprocess 运行 swift 命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 实时打印并记录日志
        for line in process.stdout:
            print(line, end='')
            log_f.write(line)
            log_f.flush()

            # 解析 loss 并更新
            if "'loss':" in line and "'epoch':" in line:
                step_count += 1

        # 等待进程结束
        process.wait()

        elapsed = int(time.time() - start_time)

        # 写入结束信息
        log_f.write("\n" + "=" * 60 + "\n")
        log_f.write(f"训练结束时间: {datetime.now()}\n")
        log_f.write(f"总耗时: {elapsed // 60} 分钟 {elapsed % 60} 秒\n")
        log_f.write(f"退出码: {process.returncode}\n")
        log_f.write("=" * 60 + "\n")
        log_f.close()

        if process.returncode == 0:
            print("\n" + "=" * 60)
            print("训练完成!")
            print(f"总耗时: {elapsed // 60} 分钟 {elapsed % 60} 秒")
            print(f"模型保存位置: {CONFIG['output_dir']}")
            print(f"日志文件: {log_file}")
            print("=" * 60)
            print("\n查看训练曲线:")
            print(f"  tensorboard --logdir {CONFIG['output_dir']}")
        else:
            print("\n" + "=" * 60)
            print(f"训练失败，退出码: {process.returncode}")
            print(f"日志文件: {log_file}")
            print("=" * 60)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        log_f.write("\n训练被用户中断\n")
        log_f.close()
        process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\n训练出错: {e}")
        log_f.write(f"\n训练出错: {e}\n")
        log_f.close()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
