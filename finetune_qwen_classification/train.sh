#!/bin/bash
# ==============================================================================
# Qwen2.5-1.5B 电商评论文本分类微调训练脚本
# 使用 ms-swift 框架进行 LoRA 微调
# ==============================================================================

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 路径配置
DATA_DIR="${PROJECT_ROOT}/data/processed"
OUTPUT_DIR="${PROJECT_ROOT}/output"
LOG_DIR="${PROJECT_ROOT}/logs"
RUNS_DIR="${PROJECT_ROOT}/runs"
MODEL_CACHE_DIR="${PROJECT_ROOT}/model_cache"

# 创建必要的目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RUNS_DIR}"
mkdir -p "${MODEL_CACHE_DIR}"

# ==============================================================================
# 训练参数配置
# ==============================================================================

# 模型配置
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

# 数据集配置
TRAIN_DATA="${DATA_DIR}/train.jsonl"
VAL_DATA="${DATA_DIR}/val.jsonl"

# 训练参数
EPOCHS="3"
BATCH_SIZE="4"
LEARNING_RATE="1e-4"
MAX_LENGTH="512"
GRADIENT_ACCUMULATION="4"

# LoRA 配置
LORA_RANK="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

# ==============================================================================
# 环境检测
# ==============================================================================

echo "========================================"
echo "Qwen2.5-1.5B 文本分类微调训练"
echo "========================================"
echo ""
echo "配置信息:"
echo "  模型: ${MODEL_NAME}"
echo "  训练数据: ${TRAIN_DATA}"
echo "  验证数据: ${VAL_DATA}"
echo "  批次大小: ${BATCH_SIZE}"
echo "  训练轮数: ${EPOCHS}"
echo "  学习率: ${LEARNING_RATE}"
echo "  LoRA rank: ${LORA_RANK}"
echo ""

# 检测 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

# 设置模型缓存目录
export MODELSCOPE_CACHE="${MODEL_CACHE_DIR}"

# ==============================================================================
# 检查数据文件
# ==============================================================================

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "错误: 训练数据不存在: ${TRAIN_DATA}"
    echo "请先运行:"
    echo "  python download_data.py"
    echo "  python preprocess_data.py"
    exit 1
fi

# ==============================================================================
# 构建训练命令
# ==============================================================================

SWIFT_CMD="swift sft"

# 模型参数
SWIFT_CMD="${SWIFT_CMD} --model ${MODEL_NAME}"

# 数据集参数
SWIFT_CMD="${SWIFT_CMD} --dataset ${TRAIN_DATA}"
if [ -f "${VAL_DATA}" ]; then
    SWIFT_CMD="${SWIFT_CMD}:${VAL_DATA}"
fi

# 任务配置
SWIFT_CMD="${SWIFT_CMD} --task_type seq_cls --num_labels 3"

# 训练参数
SWIFT_CMD="${SWIFT_CMD} --output_dir ${OUTPUT_DIR}"
SWIFT_CMD="${SWIFT_CMD} --num_train_epochs ${EPOCHS}"
SWIFT_CMD="${SWIFT_CMD} --per_device_train_batch_size ${BATCH_SIZE}"
SWIFT_CMD="${SWIFT_CMD} --per_device_eval_batch_size ${BATCH_SIZE}"
SWIFT_CMD="${SWIFT_CMD} --learning_rate ${LEARNING_RATE}"
SWIFT_CMD="${SWIFT_CMD} --max_length ${MAX_LENGTH}"
SWIFT_CMD="${SWIFT_CMD} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION}"

# LoRA 参数
SWIFT_CMD="${SWIFT_CMD} --sft_type lora"
SWIFT_CMD="${SWIFT_CMD} --lora_rank ${LORA_RANK}"
SWIFT_CMD="${SWIFT_CMD} --lora_alpha ${LORA_ALPHA}"
SWIFT_CMD="${SWIFT_CMD} --lora_dropout ${LORA_DROPOUT}"
SWIFT_CMD="${SWIFT_CMD} --lora_target_modules ALL"

# 日志和保存
SWIFT_CMD="${SWIFT_CMD} --logging_steps 10"
SWIFT_CMD="${SWIFT_CMD} --eval_steps 100"
SWIFT_CMD="${SWIFT_CMD} --save_steps 100"
SWIFT_CMD="${SWIFT_CMD} --save_total_limit 1"

# TensorBoard
SWIFT_CMD="${SWIFT_CMD} --report_to tensorboard"

# 梯度检查点
SWIFT_CMD="${SWIFT_CMD} --gradient_checkpointing true"

# ==============================================================================
# 执行训练
# ==============================================================================

echo "开始训练..."
echo ""
echo "完整命令:"
echo "${SWIFT_CMD}"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 执行训练
${SWIFT_CMD}

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "训练完成!"
echo "总耗时: $((ELAPSED / 60)) 分钟 $((ELAPSED % 60)) 秒"
echo "模型保存位置: ${OUTPUT_DIR}"
echo "========================================"
echo ""
echo "查看训练曲线:"
echo "  tensorboard --logdir ${RUNS_DIR}"
echo "  或启动可视化: bash visualize.sh"
