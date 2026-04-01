#!/usr/bin/env python3
"""
训练曲线可视化脚本
从训练日志中提取指标数据并生成可视化图表

支持的可视化:
1. 训练损失曲线
2. 验证损失曲线
3. 学习率曲线
4. 准确率曲线
5. 训练/验证指标对比
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "figures")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# 颜色配置
COLORS = {
    'train_loss': '#e74c3c',      # 红色
    'eval_loss': '#3498db',        # 蓝色
    'train_acc': '#27ae60',        # 绿色
    'eval_acc': '#9b59b6',         # 紫色
    'learning_rate': '#f39c12',     # 橙色
    'grad_norm': '#1abc9c',        # 青色
}


class LogParser:
    """训练日志解析器"""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.data = {
            'step': [],
            'train_loss': [],
            'eval_loss': [],
            'train_accuracy': [],
            'eval_accuracy': [],
            'learning_rate': [],
            'grad_norm': [],
            'epoch': [],
        }

    def parse(self) -> Dict[str, List]:
        """解析日志文件"""
        if not os.path.exists(self.log_file):
            print(f"警告: 日志文件不存在: {self.log_file}")
            print("将生成模拟数据进行演示...")
            return self._generate_mock_data()

        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析训练损失
        train_loss_pattern = r"'loss':\s*([0-9.]+)"
        for match in re.finditer(train_loss_pattern, content):
            self.data['train_loss'].append(float(match.group(1)))

        # 解析验证损失
        eval_loss_pattern = r"'eval_loss':\s*([0-9.]+)"
        for match in re.finditer(eval_loss_pattern, content):
            self.data['eval_loss'].append(float(match.group(1)))

        # 解析训练准确率
        train_acc_pattern = r"'train_accuracy':\s*([0-9.]+)"
        for match in re.finditer(train_acc_pattern, content):
            self.data['train_accuracy'].append(float(match.group(1)))

        # 解析验证准确率
        eval_acc_pattern = r"'eval_accuracy':\s*([0-9.]+)"
        for match in re.finditer(eval_acc_pattern, content):
            self.data['eval_accuracy'].append(float(match.group(1)))

        # 解析学习率
        lr_pattern = r"'learning_rate':\s*([0-9.e-]+)"
        for match in re.finditer(lr_pattern, content):
            self.data['learning_rate'].append(float(match.group(1)))

        # 解析梯度范数
        grad_pattern = r"'grad_norm':\s*([0-9.e-]+)"
        for match in re.finditer(grad_pattern, content):
            self.data['grad_norm'].append(float(match.group(1)))

        # 生成步数
        for i in range(len(self.data['train_loss'])):
            self.data['step'].append(i * 10)  # logging_steps = 10

        # 如果没有数据，生成模拟数据
        if not any(self.data.values()):
            print("未在日志中找到有效数据，生成模拟数据进行演示...")
            return self._generate_mock_data()

        return self.data

    def _generate_mock_data(self) -> Dict[str, List]:
        """生成模拟数据用于演示"""
        steps = list(range(0, 300, 10))
        n = len(steps)

        # 生成模拟损失曲线 (从高到低，逐渐收敛)
        self.data['step'] = steps
        self.data['train_loss'] = [2.5 * np.exp(-0.02 * s) + 0.1 + np.random.normal(0, 0.05)
                                    for s in steps]
        self.data['eval_loss'] = [2.3 * np.exp(-0.015 * s) + 0.15 + np.random.normal(0, 0.05)
                                   for s in steps]

        # 生成模拟准确率曲线 (从低到高，逐渐收敛)
        self.data['train_accuracy'] = [0.3 + 0.5 * (1 - np.exp(-0.02 * s)) + np.random.normal(0, 0.02)
                                       for s in steps]
        self.data['eval_accuracy'] = [0.28 + 0.45 * (1 - np.exp(-0.018 * s)) + np.random.normal(0, 0.02)
                                       for s in steps]

        # 学习率 (cosine 衰减)
        base_lr = 1e-4
        self.data['learning_rate'] = [base_lr * 0.5 * (1 + np.cos(np.pi * s / 300))
                                       for s in steps]

        # 梯度范数
        self.data['grad_norm'] = [2.0 + np.random.normal(0, 0.3) for _ in steps]

        return self.data


def parse_tensorboard_events(runs_dir: str) -> Dict[str, List]:
    """从 TensorBoard events 文件中提取数据"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("请安装 tensorboard: pip install tensorboard")
        return {}

    data = {
        'step': [],
        'train_loss': [],
        'eval_loss': [],
        'train_accuracy': [],
        'eval_accuracy': [],
        'learning_rate': [],
    }

    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_path = os.path.join(root, file)
                try:
                    ea = event_accumulator.EventAccumulator(event_path)
                    ea.Reload()

                    # 获取所有可用的 tags
                    tags = ea.Tags().get('scalars', [])

                    for tag in tags:
                        if 'loss' in tag.lower():
                            events = ea.Scalars(tag)
                            data['step'] = [e.step for e in events]
                            if 'train' in tag.lower() or 'training' in tag.lower():
                                data['train_loss'] = [e.value for e in events]
                            else:
                                data['eval_loss'] = [e.value for e in events]
                        elif 'accuracy' in tag.lower() or 'acc' in tag.lower():
                            events = ea.Scalars(tag)
                            if 'train' in tag.lower() or 'training' in tag.lower():
                                data['train_accuracy'] = [e.value for e in events]
                            else:
                                data['eval_accuracy'] = [e.value for e in events]
                        elif 'learning_rate' in tag.lower() or 'lr' in tag.lower():
                            events = ea.Scalars(tag)
                            data['learning_rate'] = [e.value for e in events]

                except Exception as e:
                    print(f"读取 {event_path} 时出错: {e}")

    return data


def plot_loss_curves(data: Dict[str, List], output_dir: str):
    """绘制损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if data.get('train_loss'):
        ax.plot(data['step'], data['train_loss'],
                label='训练损失 (Train Loss)', color=COLORS['train_loss'],
                linewidth=2, marker='o', markersize=3)

    if data.get('eval_loss'):
        ax.plot(data['step'], data['eval_loss'],
                label='验证损失 (Eval Loss)', color=COLORS['eval_loss'],
                linewidth=2, marker='s', markersize=3)

    ax.set_xlabel('训练步数 (Steps)', fontsize=12)
    ax.set_ylabel('损失值 (Loss)', fontsize=12)
    ax.set_title('训练与验证损失曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"损失曲线已保存: {output_path}")
    plt.close()


def plot_accuracy_curves(data: Dict[str, List], output_dir: str):
    """绘制准确率曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if data.get('train_accuracy'):
        ax.plot(data['step'], data['train_accuracy'],
                label='训练准确率 (Train Acc)', color=COLORS['train_acc'],
                linewidth=2, marker='o', markersize=3)

    if data.get('eval_accuracy'):
        ax.plot(data['step'], data['eval_accuracy'],
                label='验证准确率 (Eval Acc)', color=COLORS['eval_acc'],
                linewidth=2, marker='s', markersize=3)

    ax.set_xlabel('训练步数 (Steps)', fontsize=12)
    ax.set_ylabel('准确率 (Accuracy)', fontsize=12)
    ax.set_title('训练与验证准确率曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"准确率曲线已保存: {output_path}")
    plt.close()


def plot_learning_rate_curve(data: Dict[str, List], output_dir: str):
    """绘制学习率曲线"""
    if not data.get('learning_rate'):
        print("未找到学习率数据")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data['step'], data['learning_rate'],
            label='学习率 (Learning Rate)', color=COLORS['learning_rate'],
            linewidth=2)

    ax.set_xlabel('训练步数 (Steps)', fontsize=12)
    ax.set_ylabel('学习率 (Learning Rate)', fontsize=12)
    ax.set_title('学习率变化曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'learning_rate_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"学习率曲线已保存: {output_path}")
    plt.close()


def plot_combined_metrics(data: Dict[str, List], output_dir: str):
    """绘制综合指标图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 损失曲线
    ax1 = axes[0, 0]
    if data.get('train_loss'):
        ax1.plot(data['step'], data['train_loss'],
                 label='训练损失', color=COLORS['train_loss'], linewidth=2)
    if data.get('eval_loss'):
        ax1.plot(data['step'], data['eval_loss'],
                 label='验证损失', color=COLORS['eval_loss'], linewidth=2)
    ax1.set_xlabel('步数')
    ax1.set_ylabel('损失')
    ax1.set_title('损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2 = axes[0, 1]
    if data.get('train_accuracy'):
        ax2.plot(data['step'], data['train_accuracy'],
                 label='训练准确率', color=COLORS['train_acc'], linewidth=2)
    if data.get('eval_accuracy'):
        ax2.plot(data['step'], data['eval_accuracy'],
                 label='验证准确率', color=COLORS['eval_acc'], linewidth=2)
    ax2.set_xlabel('步数')
    ax2.set_ylabel('准确率')
    ax2.set_title('准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # 学习率曲线
    ax3 = axes[1, 0]
    if data.get('learning_rate'):
        ax3.plot(data['step'], data['learning_rate'],
                 color=COLORS['learning_rate'], linewidth=2)
        ax3.set_xlabel('步数')
        ax3.set_ylabel('学习率')
        ax3.set_title('学习率曲线')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

    # 梯度范数曲线
    ax4 = axes[1, 1]
    if data.get('grad_norm'):
        ax4.plot(data['step'], data['grad_norm'],
                 color=COLORS['grad_norm'], linewidth=2)
        ax4.set_xlabel('步数')
        ax4.set_ylabel('梯度范数')
        ax4.set_title('梯度范数曲线')
        ax4.grid(True, alpha=0.3)

    plt.suptitle('Qwen2.5-1.5B 电商评论分类训练监控', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'combined_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"综合指标图已保存: {output_path}")
    plt.close()


def generate_summary(data: Dict[str, List]):
    """生成训练摘要"""
    summary = []

    summary.append("=" * 50)
    summary.append("训练指标摘要")
    summary.append("=" * 50)

    if data.get('train_loss'):
        final_train_loss = data['train_loss'][-1]
        min_train_loss = min(data['train_loss'])
        summary.append(f"\n训练损失:")
        summary.append(f"  最终值: {final_train_loss:.4f}")
        summary.append(f"  最小值: {min_train_loss:.4f}")

    if data.get('eval_loss'):
        final_eval_loss = data['eval_loss'][-1]
        min_eval_loss = min(data['eval_loss'])
        summary.append(f"\n验证损失:")
        summary.append(f"  最终值: {final_eval_loss:.4f}")
        summary.append(f"  最小值: {min_eval_loss:.4f}")

    if data.get('train_accuracy'):
        final_train_acc = data['train_accuracy'][-1]
        max_train_acc = max(data['train_accuracy'])
        summary.append(f"\n训练准确率:")
        summary.append(f"  最终值: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        summary.append(f"  最大值: {max_train_acc:.4f} ({max_train_acc*100:.2f}%)")

    if data.get('eval_accuracy'):
        final_eval_acc = data['eval_accuracy'][-1]
        max_eval_acc = max(data['eval_accuracy'])
        summary.append(f"\n验证准确率:")
        summary.append(f"  最终值: {final_eval_acc:.4f} ({final_eval_acc*100:.2f}%)")
        summary.append(f"  最大值: {max_eval_acc:.4f} ({max_eval_acc*100:.2f}%)")

    summary.append("\n" + "=" * 50)

    summary_text = "\n".join(summary)
    print(summary_text)
    return summary_text


def main():
    parser = argparse.ArgumentParser(description='训练曲线可视化工具')
    parser.add_argument('--log', '-l', help='训练日志文件路径')
    parser.add_argument('--runs', '-r', default=RUNS_DIR, help='TensorBoard runs 目录')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR, help='输出图片目录')
    parser.add_argument('--tensorboard', '-t', action='store_true', help='从 TensorBoard 读取数据')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print("=" * 50)
    print("训练曲线可视化工具")
    print("=" * 50)

    # 获取数据
    if args.tensorboard:
        print(f"\n从 TensorBoard 读取数据: {args.runs}")
        data = parse_tensorboard_events(args.runs)
    elif args.log:
        print(f"\n从日志文件读取数据: {args.log}")
        parser = LogParser(args.log)
        data = parser.parse()
    else:
        # 尝试从日志文件读取
        log_file = os.path.join(LOG_DIR, "training.log")
        if os.path.exists(log_file):
            parser = LogParser(log_file)
            data = parser.parse()
        else:
            print(f"\n未指定日志文件，尝试从 TensorBoard 读取...")
            data = parse_tensorboard_events(RUNS_DIR)
            if not data:
                print("未找到数据，将生成演示图表...")
                parser = LogParser("")
                data = parser.parse()

    # 生成可视化
    print(f"\n生成可视化图表...")

    plot_loss_curves(data, args.output)
    plot_accuracy_curves(data, args.output)
    plot_learning_rate_curve(data, args.output)
    plot_combined_metrics(data, args.output)

    # 生成摘要
    generate_summary(data)

    print(f"\n所有图表已保存到: {args.output}")
    print("\n提示:")
    print("  1. 查看图片: open figures/")
    print("  2. 启动 TensorBoard: bash visualize.sh")


if __name__ == "__main__":
    main()
