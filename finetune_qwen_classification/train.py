#!/usr/bin/env python3
"""
Qwen2.5-1.5B 电商评论文本分类微调训练脚本
使用 Huggingface Transformers + PEFT (LoRA) 进行微调
所有参数在 Python 代码中配置，无需命令行参数
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


# ==============================================================================
# 训练配置 - 直接修改这里的参数
# ==============================================================================

CONFIG = {
    # 模型配置
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_cache_dir": "./model_cache",  # 模型缓存目录
    "use_huggingface_hub": False,  # 是否使用HuggingFace Hub（False则使用ModelScope）

    # 分类配置
    "num_labels": 3,  # 分类数量（负面=0, 中性=1, 正面=2）
    "id2label": {0: "负面", 1: "中性", 2: "正面"},
    "label2id": {"负面": 0, "中性": 1, "正面": 2},

    # 数据配置
    "train_data": "./data/processed/train.jsonl",
    "val_data": "./data/processed/val.jsonl",
    "max_length": 256,  # 最大序列长度

    # 训练配置
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,  # 梯度累积步数
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,  # 预热比例
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",  # 学习率调度器: linear, cosine, constant

    # LoRA 配置
    "lora_r": 16,  # LoRA 秩
    "lora_alpha": 32,  # LoRA alpha 参数
    "lora_dropout": 0.05,  # LoRA dropout
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # LoRA 目标模块

    # 输出配置
    "output_dir": "./output",
    "logging_dir": "./logs",
    "logging_steps": 10,
    "eval_strategy": "epoch",  # 评估策略: no, steps, epoch
    "save_strategy": "epoch",  # 保存策略: no, steps, epoch
    "save_total_limit": 3,
    "save_only_model": False,

    # 优化配置
    "gradient_checkpointing": True,
    "fp16": True,  # 混合精度训练
    "bf16": False,  # BF16 训练（如果GPU支持）
    "group_by_length": False,

    # 其他配置
    "seed": 42,
    "report_to": ["tensorboard"],  # 报告工具: tensorboard, none
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_accuracy",
    "greater_is_better": True,
    "remove_unused_columns": False,
}


# ==============================================================================
# 工具函数
# ==============================================================================

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_environment():
    """设置环境变量"""
    if CONFIG["model_cache_dir"]:
        os.makedirs(CONFIG["model_cache_dir"], exist_ok=True)
        os.environ["HF_HOME"] = CONFIG["model_cache_dir"]
        os.environ["TRANSFORMERS_CACHE"] = CONFIG["model_cache_dir"]

    os.makedirs(CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(CONFIG["output_dir"], exist_ok=True)


def check_data():
    """检查数据文件是否存在"""
    train_path = CONFIG["train_data"]
    if not os.path.exists(train_path):
        print(f"[ERROR] 训练数据不存在: {train_path}")
        print(f"[INFO] 请运行: python preprocess_data.py")
        sys.exit(1)


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# ==============================================================================
# 数据集类
# ==============================================================================

class TextClassificationDataset(Dataset):
    """文本分类数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        label2id: Optional[Dict[str, int]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {"负面": 0, "中性": 1, "正面": 2}
        self.examples = []

        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """加载数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        print(f"[DATA] 加载 {len(self.examples)} 条数据: {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]

        text = item.get("text", "")
        label = item.get("label", 0)

        # 处理标签
        if isinstance(label, str):
            label = self.label2id.get(label, 0)
        elif isinstance(label, float):
            label = int(label)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label,
        }


# ==============================================================================
# 主训练函数
# ==============================================================================

def main():
    print("")
    print("=" * 60)
    print("       Qwen2.5-1.5B 文本分类微调训练 (Huggingface)")
    print("=" * 60)
    print("")

    # 设置环境
    setup_environment()
    check_data()
    set_seed(CONFIG["seed"])

    # 打印配置
    print("┌─ 配置 ─────────────────────────────────────────────┐")
    print(f"│  模型: {CONFIG['model_name']}")
    print(f"│  Epochs: {CONFIG['num_train_epochs']} | Batch: {CONFIG['per_device_train_batch_size']} | LR: {CONFIG['learning_rate']}")
    print(f"│  LoRA: r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}, dropout={CONFIG['lora_dropout']}")
    print(f"│  Max Length: {CONFIG['max_length']}")
    print(f"│  输出: {CONFIG['output_dir']}")
    print("└──────────────────────────────────────────────────┘")
    print("")

    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
        if CONFIG["bf16"]:
            print("[GPU] 使用 BF16 混合精度")
        elif CONFIG["fp16"]:
            print("[GPU] 使用 FP16 混合精度")
    else:
        print("[CPU] 使用 CPU 训练（不推荐）")
    print("")

    # ======================================================================
    # 加载 tokenizer
    # ======================================================================
    print("[STEP 1/5] 加载 Tokenizer...")

    tokenizer_kwargs = {
        "cache_dir": CONFIG["model_cache_dir"],
        "trust_remote_code": True,
    }

    if CONFIG["use_huggingface_hub"]:
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"],
            **tokenizer_kwargs
        )
    else:
        # 使用 ModelScope 的模型
        from modelscope import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"],
            cache_dir=CONFIG["model_cache_dir"],
            trust_remote_code=True,
        )

    # Qwen 需要 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[TOKENIZER] 设置 pad_token = eos_token")

    print("[STEP 1/5] Tokenizer 加载完成")
    print("")

    # ======================================================================
    # 加载模型
    # ======================================================================
    print("[STEP 2/5] 加载模型...")

    model_kwargs = {
        "cache_dir": CONFIG["model_cache_dir"],
        "trust_remote_code": True,
        "num_labels": CONFIG["num_labels"],
        "id2label": CONFIG["id2label"],
        "label2id": CONFIG["label2id"],
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }

    if CONFIG["use_huggingface_hub"]:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["model_name"],
            **model_kwargs
        )
    else:
        from modelscope import AutoModelForSequenceClassification
        base_model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["model_name"],
            cache_dir=CONFIG["model_cache_dir"],
            trust_remote_code=True,
            num_labels=CONFIG["num_labels"],
            id2label=CONFIG["id2label"],
            label2id=CONFIG["label2id"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    # 设置 pad_token_id
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"[MODEL] 模型参数量: {sum(p.numel() for p in base_model.parameters()) / 1e9:.2f} B")

    # ======================================================================
    # 配置 LoRA
    # ======================================================================
    print("[STEP 2/5] 配置 LoRA...")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print("[STEP 2/5] LoRA 配置完成")
    print("")

    # ======================================================================
    # 加载数据集
    # ======================================================================
    print("[STEP 3/5] 加载数据集...")

    train_dataset = TextClassificationDataset(
        data_path=CONFIG["train_data"],
        tokenizer=tokenizer,
        max_length=CONFIG["max_length"],
        label2id=CONFIG["label2id"],
    )

    val_dataset = None
    if CONFIG["val_data"] and os.path.exists(CONFIG["val_data"]):
        val_dataset = TextClassificationDataset(
            data_path=CONFIG["val_data"],
            tokenizer=tokenizer,
            max_length=CONFIG["max_length"],
            label2id=CONFIG["label2id"],
        )

    print("[STEP 3/5] 数据集加载完成")
    print("")

    # ======================================================================
    # 配置训练参数
    # ======================================================================
    print("[STEP 4/5] 配置训练参数...")

    # 强制使用 FP16，禁用 BF16（避免兼容性问题）
    os.environ["ACCELERATE_DOWNCAST_BF16"] = "false"

    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        logging_dir=CONFIG["logging_dir"],
        logging_steps=CONFIG["logging_steps"],
        eval_strategy=CONFIG["eval_strategy"],
        save_strategy=CONFIG["save_strategy"],
        save_total_limit=CONFIG["save_total_limit"],
        save_only_model=CONFIG["save_only_model"],
        load_best_model_at_end=CONFIG["load_best_model_at_end"],
        metric_for_best_model=CONFIG["metric_for_best_model"],
        greater_is_better=CONFIG["greater_is_better"],
        fp16=True,  # 强制使用 FP16
        bf16=False,  # 禁用 BF16
        gradient_checkpointing=CONFIG["gradient_checkpointing"],
        seed=CONFIG["seed"],
        report_to=CONFIG["report_to"],
        remove_unused_columns=CONFIG["remove_unused_columns"],
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    print("[STEP 4/5] 训练参数配置完成")
    print("")

    # ======================================================================
    # 创建 Trainer
    # ======================================================================
    print("[STEP 5/5] 创建 Trainer...")

    callbacks = []
    if val_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=CONFIG["max_length"],
        ),
        callbacks=callbacks,
    )

    print("[STEP 5/5] Trainer 创建完成")
    print("")

    # ======================================================================
    # 开始训练
    # ======================================================================
    print("=" * 60)
    print("[START] 开始训练...")
    print("=" * 60)
    print("")

    start_time = time.time()

    try:
        train_result = trainer.train()

        # 保存训练结果
        trainer.save_model(os.path.join(CONFIG["output_dir"], "final_model"))
        trainer.save_state()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        elapsed = int(time.time() - start_time)

        print("")
        print("=" * 60)
        print("[SUCCESS] 训练完成!")
        print(f"[TIME] 耗时: {elapsed // 60} 分 {elapsed % 60} 秒")
        print(f"[SAVE] 模型: {os.path.join(CONFIG['output_dir'], 'final_model')}")
        print("=" * 60)

        # 如果有验证集，打印最终评估结果
        if val_dataset is not None:
            print("")
            print("=" * 60)
            print("[EVAL] 最终评估结果:")
            print("=" * 60)
            metrics = trainer.evaluate()
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

        print("")
        print("[TIP] 查看日志: tensorboard --logdir ./logs")
        print("[TIP] 测试推理: python inference.py --model_path ./output/final_model")

    except KeyboardInterrupt:
        print("\n\n[STOP] 训练已中断")
        trainer.save_model(os.path.join(CONFIG["output_dir"], "interrupted_model"))
    except Exception as e:
        print(f"\n[ERROR] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
