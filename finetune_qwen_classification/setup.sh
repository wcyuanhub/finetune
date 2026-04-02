#!/bin/bash
# 安装依赖 + 下载数据 + 预处理数据

set -e

echo "========================================"
echo "环境准备"
echo "========================================"

# 安装依赖
echo ""
echo "[1/4] 修复依赖版本兼容性..."
# 先升级 transformers 到兼容版本
pip install transformers>=4.40.0 --upgrade -q
# 再安装兼容版本的 peft
pip install peft>=0.10.0 --upgrade -q

echo ""
echo "[2/4] 安装核心依赖..."
pip install torch modelscope ms-swift pandas tqdm -q

echo ""
echo "[3/4] 下载数据..."
python download_data.py

echo ""
echo "[4/4] 预处理数据..."
python preprocess_data.py

echo ""
echo "========================================"
echo "环境准备完成!"
echo "========================================"
echo ""
echo "下一步: python train.py"
