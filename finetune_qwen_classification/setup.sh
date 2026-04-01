#!/bin/bash
# 安装依赖 + 下载数据 + 预处理数据

set -e

echo "========================================"
echo "环境准备"
echo "========================================"

# 安装依赖
echo ""
echo "[1/3] 安装依赖..."
pip install torch transformers peft modelscope ms-swift pandas tqdm -q

# 下载数据
echo ""
echo "[2/3] 下载数据..."
python download_data.py

# 预处理数据
echo ""
echo "[3/3] 预处理数据..."
python preprocess_data.py

echo ""
echo "========================================"
echo "环境准备完成!"
echo "========================================"
echo ""
echo "下一步: python train.py"
