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
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_cache_dir": "./model_cache",
    "train_data": "./data/processed/train.jsonl",
    "val_data": "./data/processed/val.jsonl",
    "max_length": 256,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 1e-4,
    "output_dir": "./output",
    "log_dir": "./logs",
    "logging_steps": 10,
    "eval_steps": 500,
    "save_steps": 1000,
    "save_total_limit": 1,
    "seed": 42,
    "gradient_checkpointing": True,
}


def setup_environment():
    if CONFIG["model_cache_dir"]:
        os.makedirs(CONFIG["model_cache_dir"], exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = CONFIG["model_cache_dir"]
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    os.makedirs(CONFIG["output_dir"], exist_ok=True)


def check_data():
    train_path = CONFIG["train_data"]
    if not os.path.exists(train_path):
        print(f"[ERROR] 训练数据不存在: {train_path}")
        print(f"[INFO] 请运行: python download_data.py && python preprocess_data.py")
        sys.exit(1)


def build_command():
    cmd = ["swift", "sft"]
    cmd.extend(["--model", CONFIG["model_name"]])
    cmd.extend(["--dataset", CONFIG["train_data"]])
    cmd.extend(["--output_dir", CONFIG["output_dir"]])

    if CONFIG["val_data"] and os.path.exists(CONFIG["val_data"]):
        cmd[-1] = CONFIG["train_data"] + ":" + CONFIG["val_data"]

    cmd.extend(["--num_train_epochs", str(CONFIG["num_train_epochs"])])
    cmd.extend(["--per_device_train_batch_size", str(CONFIG["per_device_train_batch_size"])])
    cmd.extend(["--per_device_eval_batch_size", str(CONFIG["per_device_eval_batch_size"])])
    cmd.extend(["--learning_rate", str(CONFIG["learning_rate"])])
    cmd.extend(["--max_length", str(CONFIG["max_length"])])
    cmd.extend(["--logging_steps", str(CONFIG["logging_steps"])])
    cmd.extend(["--eval_steps", str(CONFIG["eval_steps"])])
    cmd.extend(["--save_steps", str(CONFIG["save_steps"])])
    cmd.extend(["--save_total_limit", str(CONFIG["save_total_limit"])])
    cmd.extend(["--report_to", "tensorboard"])

    if CONFIG["gradient_checkpointing"]:
        cmd.append("--gradient_checkpointing")

    cmd.extend(["--seed", str(CONFIG["seed"])])
    return cmd


def parse_metrics(line):
    """解析日志行，提取关键指标"""
    info = {}

    if "'loss':" in line:
        try:
            start = line.find("'loss':") + 8
            end = line.find(",", start)
            info['loss'] = line[start:end].strip().strip("'\"")
        except:
            pass

    if "'grad_norm':" in line:
        try:
            start = line.find("'grad_norm':") + 13
            end = line.find(",", start)
            info['grad_norm'] = line[start:end].strip().strip("'\"")
        except:
            pass

    if "'learning_rate':" in line:
        try:
            start = line.find("'learning_rate':") + 16
            end = line.find(",", start)
            lr = line[start:end].strip().strip("'\"")
            info['lr'] = f"{float(lr):.2e}" if lr else "N/A"
        except:
            pass

    if "'epoch':" in line:
        try:
            start = line.find("'epoch':") + 9
            end = line.find(",", start)
            info['epoch'] = line[start:end].strip().strip("'\"")
        except:
            pass

    if "'global_step/max_steps':" in line:
        try:
            start = line.find("'global_step/max_steps':") + 24
            end = line.find(",", start)
            info['step'] = line[start:end].strip().strip("'\"")
        except:
            pass

    if "'elapsed_time':" in line:
        try:
            start = line.find("'elapsed_time':") + 16
            end = line.find(",", start)
            info['elapsed'] = line[start:end].strip().strip("'\"")
        except:
            pass

    if "'remaining_time':" in line:
        try:
            start = line.find("'remaining_time':") + 17
            end = line.find(",", start)
            info['remaining'] = line[start:end].strip().strip("'\"")
        except:
            pass

    if "'memory(GiB)':" in line:
        try:
            start = line.find("'memory(GiB)':") + 14
            end = line.find(",", start)
            info['memory'] = line[start:end].strip().strip("'\"")
        except:
            pass

    return info


def main():
    print("")
    print("=" * 60)
    print("       Qwen2.5-1.5B 文本分类微调训练")
    print("=" * 60)
    print("")

    setup_environment()
    check_data()

    print("┌─ 配置 ─────────────────────────────────────────────┐")
    print(f"│  模型: {CONFIG['model_name']}")
    print(f"│  Epochs: {CONFIG['num_train_epochs']} | Batch: {CONFIG['per_device_train_batch_size']} | LR: {CONFIG['learning_rate']}")
    print(f"│  Max Length: {CONFIG['max_length']}")
    print(f"│  输出: {CONFIG['output_dir']}")
    print("└──────────────────────────────────────────────────┘")
    print("")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    except:
        pass

    print("")
    print("=" * 60)
    print("[START] 开始训练...")
    print("=" * 60)
    print("")

    cmd = build_command()
    print(f"[CMD] swift {' '.join(cmd[1:])}")
    print("")

    start_time = time.time()
    log_file = os.path.join(CONFIG["log_dir"], f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_f = open(log_file, 'w', encoding='utf-8')
    log_f.write(f"训练开始: {datetime.now()}\n\n")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in process.stdout:
            log_f.write(line)
            log_f.flush()

            info = parse_metrics(line)

            if info:
                elapsed = int(time.time() - start_time)
                parts = []

                if info.get('step'):
                    parts.append(f"Step {info['step']}")
                if info.get('loss'):
                    parts.append(f"Loss {float(info['loss']):.4f}")
                if info.get('lr'):
                    parts.append(f"LR {info['lr']}")
                if info.get('remaining'):
                    parts.append(f"Left {info['remaining']}")
                if info.get('memory'):
                    parts.append(f"Mem {info['memory']}GiB")

                print(f"\r[{elapsed//60:02d}:{elapsed%60:02d}] " + " | ".join(parts), end='', flush=True)

            if "INFO" in line or "ERROR" in line or "Traceback" in line:
                print(f"\n{line}", end='')

        process.wait()
        log_f.close()

        elapsed = int(time.time() - start_time)

        print("\n\n" + "=" * 60)
        if process.returncode == 0:
            print("[SUCCESS] 训练完成!")
            print(f"[TIME] 耗时: {elapsed // 60} 分 {elapsed % 60} 秒")
            print(f"[SAVE] 模型: {CONFIG['output_dir']}")
            print(f"[LOG] 日志: {log_file}")
        else:
            print(f"[ERROR] 训练失败: {process.returncode}")
        print("=" * 60)
        print("\n[TIP] TensorBoard: tensorboard --logdir output --port 6006")

    except KeyboardInterrupt:
        print("\n\n[STOP] 训练已中断")
        log_f.close()
        process.terminate()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        log_f.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
