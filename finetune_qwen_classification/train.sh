#!/bin/bash
# ==============================================================================
# Qwen2.5-1.5B 电商评论文本分类微调训练脚本
# 使用 ms-swift 框架进行 LoRA 微调
# ==============================================================================

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 路径配置
DATA_DIR="${PROJECT_ROOT}/data/processed"
OUTPUT_DIR="${PROJECT_ROOT}/output"
LOG_DIR="${PROJECT_ROOT}/logs"
MODEL_CACHE_DIR="${PROJECT_ROOT}/model_cache"

# 创建必要的目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${MODEL_CACHE_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

# ==============================================================================
# 训练参数配置
# ==============================================================================

# 模型配置
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"  # ModelScope 模型ID
MODEL_PATH=""  # 本地模型路径，留空则从 ModelScope 下载

# 数据集配置
TRAIN_DATA="${DATA_DIR}/train.jsonl"      # 训练数据
VAL_DATA="${DATA_DIR}/val.jsonl"          # 验证数据
DATASET_FORMAT="jsonl"                     # 数据格式

# 任务配置
TASK_TYPE="seq_cls"                        # 文本分类任务
NUM_LABELS="3"                             # 分类类别数 (负面/中性/正面)

# 训练参数
EPOCHS="3"                                 # 训练轮数
BATCH_SIZE="4"                             # 批次大小 (根据显存调整，24GB 建议 4-8)
LEARNING_RATE="1e-4"                        # 学习率
MAX_LENGTH="512"                            # 最大序列长度
GRADIENT_ACCUMULATION="4"                  # 梯度累积步数

# LoRA 配置
LORA_RANK="8"                               # LoRA rank 值
LORA_ALPHA="16"                             # LoRA alpha 值
LORA_DROPOUT="0.05"                         # LoRA dropout
LORA_TARGET="all"                           # 应用到所有层

# 优化器配置
OPTIMIZER_TYPE="adamw_torch"                # 优化器类型
LR_SCHEDULER_TYPE="cosine"                  # 学习率调度器
WARMUP_RATIO="0.05"                         # 预热比例
WEIGHT_DECAY="0.01"                          # 权重衰减

# 其他配置
SEED="42"                                    # 随机种子
NUM_WORKERS="4"                              # 数据加载线程数
USE_flash_attn="false"                       # 是否使用 Flash Attention (A100/H100 设为 true)

# TensorBoard 配置
USE_TENSORBOARD="true"                        # 是否启用 TensorBoard 日志
TENSORBOARD_DIR="${PROJECT_ROOT}/runs"        # TensorBoard 日志目录

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

# ==============================================================================
# 构建训练命令
# ==============================================================================

SWIFT_CMD="swift sft"

# 模型参数
if [ -n "${MODEL_PATH}" ]; then
    SWIFT_CMD="${SWIFT_CMD} --model_type ${MODEL_PATH}"
else
    SWIFT_CMD="${SWIFT_CMD} --model_type ${MODEL_NAME}"
fi

# 数据集参数 (支持本地数据集)
if [ -f "${TRAIN_DATA}" ]; then
    SWIFT_CMD="${SWIFT_CMD} --dataset ${TRAIN_DATA}"
    if [ -f "${VAL_DATA}" ]; then
        SWIFT_CMD="${SWIFT_CMD}:${VAL_DATA}"
    fi
else
    # 使用 ModelScope 内置数据集示例
    SWIFT_CMD="${SWIFT_CMD} --dataset modelscope/course_dataset:default"
fi

# 任务类型
SWIFT_CMD="${SWIFT_CMD} --task_type ${TASK_TYPE} --num_labels ${NUM_LABELS}"

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
SWIFT_CMD="${SWIFT_CMD} --lora_target ${LORA_TARGET}"

# 优化器参数
SWIFT_CMD="${SWIFT_CMD} --optimizer_type ${OPTIMIZER_TYPE}"
SWIFT_CMD="${SWIFT_CMD} --lr_scheduler_type ${LR_SCHEDULER_TYPE}"
SWIFT_CMD="${SWIFT_CMD} --warmup_ratio ${WARMUP_RATIO}"
SWIFT_CMD="${SWIFT_CMD} --weight_decay ${WEIGHT_DECAY}"

# 其他参数
SWIFT_CMD="${SWIFT_CMD} --seed ${SEED}"
SWIFT_CMD="${SWIFT_CMD} --num_workers ${NUM_WORKERS}"
SWIFT_CMD="${SWIFT_CMD} --logging_steps 10"
SWIFT_CMD="${SWIFT_CMD} --eval_steps 100"
SWIFT_CMD="${SWIFT_CMD} --save_steps 100"
SWIFT_CMD="${SWIFT_CMD} --save_only_last_checkpoint true"

# 日志
SWIFT_CMD="${SWIFT_CMD} --log_file ${LOG_DIR}/training.log"

# TensorBoard 日志
if [ "${USE_TENSORBOARD}" = "true" ]; then
    SWIFT_CMD="${SWIFT_CMD} --report_to tensorboard"
    SWIFT_CMD="${SWIFT_CMD} --tensorboard_root_dir ${TENSORBOARD_DIR}"
    echo "  TensorBoard: 已启用"
fi

# 梯度检查点 (节省显存)
SWIFT_CMD="${SWIFT_CMD} --gradient_checkpointing true"

# 模型缓存目录
if [ -n "${MODEL_CACHE_DIR}" ]; then
    export MODELSCOPE_CACHE="${MODEL_CACHE_DIR}"
fi

# ==============================================================================
# 执行训练
# ==============================================================================

echo "开始训练..."
echo "完整命令:"
echo "${SWIFT_CMD}"
echo ""
echo "日志文件: ${LOG_DIR}/training.log"
echo "模型输出: ${OUTPUT_DIR}"
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
