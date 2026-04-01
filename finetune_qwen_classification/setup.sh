#!/bin/bash
# ==============================================================================
# 环境安装脚本
# 在算力平台上运行此脚本安装依赖
# ==============================================================================

set -e

echo "========================================"
echo "Qwen2.5-1.5B 微调环境安装"
echo "========================================"

# Python 版本检查
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前 Python 版本: ${python_version}"

# 创建虚拟环境 (可选)
# python3 -m venv venv
# source venv/bin/activate

# 安装 PyTorch (根据 CUDA 版本选择)
echo ""
echo "安装 PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # GPU 版本
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d'.' -f1)
    echo "检测到 NVIDIA GPU，CUDA 版本: ${cuda_version}"

    # 根据 CUDA 版本选择 PyTorch
    if [ "${cuda_version}" -ge 12 ]; then
        pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
    elif [ "${cuda_version}" -ge 11 ]; then
        pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch>=2.0.0
    fi
else
    echo "未检测到 GPU，安装 CPU 版本 PyTorch"
    pip install torch>=2.0.0
fi

# 安装 ModelScope
echo ""
echo "安装 ModelScope..."
pip install modelscope

# 安装 ms-swift
echo ""
echo "安装 ms-swift..."
pip install ms-swift -U

# 安装其他依赖
echo ""
echo "安装其他依赖..."
pip install pandas numpy scikit-learn tqdm pyyaml

# 验证安装
echo ""
echo "========================================"
echo "验证安装..."
echo "========================================"

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

python3 -c "
import modelscope
print(f'ModelScope: {modelscope.__version__}')
"

python3 -c "
import swift
print(f'ms-swift: {swift.__version__}')
"

echo ""
echo "========================================"
echo "环境安装完成!"
echo "========================================"
echo ""
echo "接下来执行以下步骤:"
echo "1. python download_data.py    # 下载数据集"
echo "2. python preprocess_data.py # 预处理数据"
echo "3. bash train.sh             # 开始训练"
