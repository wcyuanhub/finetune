#!/bin/bash
# ==============================================================================
# TensorBoard 可视化启动脚本
# 启动后访问 http://localhost:6006 查看训练曲线
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSORBOARD_DIR="${PROJECT_ROOT}/runs"
PORT="${1:-6006}"

echo "========================================"
echo "启动 TensorBoard 可视化"
echo "========================================"
echo ""
echo "日志目录: ${TENSORBOARD_DIR}"
echo "访问地址: http://localhost:${PORT}"
echo ""

# 检查是否安装了 tensorboard
if ! command -v tensorboard &> /dev/null; then
    echo "TensorBoard 未安装，正在安装..."
    pip install tensorboard
fi

# 启动 TensorBoard
tensorboard --logdir "${TENSORBOARD_DIR}" --port "${PORT}" --bind_all
