#!/bin/bash
# ==============================================================================
# 环境安装脚本 - 完整版
# 自动检测环境并安装所有依赖
# ==============================================================================

set -e

echo "========================================"
echo "Qwen2.5-1.5B 微调环境安装"
echo "========================================"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ==============================================================================
# 1. 检测环境
# ==============================================================================

echo ""
echo "[1/6] 检测环境..."

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python 版本: ${python_version}"

if command -v nvidia-smi &> /dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | awk '{print $1}')
    echo "  GPU: ${gpu_name}"
    echo "  显存: ${gpu_memory} MB"
    HAS_GPU=1
else
    echo "  GPU: 未检测到 (将安装 CPU 版本)"
    HAS_GPU=0
fi

# ==============================================================================
# 2. 安装 PyTorch
# ==============================================================================

echo ""
echo "[2/6] 安装 PyTorch..."

if [ ${HAS_GPU} -eq 1 ]; then
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d'.' -f1)
    echo "  检测到 CUDA 版本: ${cuda_version}"

    if [ "${cuda_version}" -ge 12 ]; then
        echo "  安装 PyTorch (CUDA 12.x)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [ "${cuda_version}" -ge 11 ]; then
        echo "  安装 PyTorch (CUDA 11.x)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  安装 PyTorch (CUDA)..."
        pip install torch torchvision torchaudio
    fi
else
    echo "  安装 PyTorch (CPU)..."
    pip install torch torchvision torchaudio
fi

# ==============================================================================
# 3. 安装 transformers 和 peft (关键：版本兼容性)
# ==============================================================================

echo ""
echo "[3/6] 安装 transformers 和 peft..."

# 先卸载旧版本，确保干净安装
pip uninstall -y transformers peft accelerate 2>/dev/null || true

# 安装兼容版本
pip install transformers>=4.40.0
pip install peft>=0.10.0
pip install accelerate

# ==============================================================================
# 4. 安装 ms-swift 和 ModelScope
# ==============================================================================

echo ""
echo "[4/6] 安装 ms-swift 和 ModelScope..."

pip install modelscope
pip install ms-swift -U --no-cache-dir

# ==============================================================================
# 5. 安装其他依赖
# ==============================================================================

echo ""
echo "[5/6] 安装其他依赖..."

pip install \
    pandas \
    numpy \
    scikit-learn \
    tqdm \
    pyyaml \
    matplotlib \
    tensorboard \
    ipython

# ==============================================================================
# 6. 验证安装
# ==============================================================================

echo ""
echo "[6/6] 验证安装..."
echo "========================================"

check_package() {
    local name=$1
    local import=$2
    echo -n "  ${name}... "
    if python3 -c "import ${import}; print(${import}.__version__)" 2>/dev/null; then
        return 0
    else
        echo "安装失败"
        return 1
    fi
}

all_ok=true

check_package "PyTorch" "torch" || all_ok=false
check_package "transformers" "transformers" || all_ok=false
check_package "peft" "peft" || all_ok=false
check_package "ModelScope" "modelscope" || all_ok=false
check_package "ms-swift" "swift" || all_ok=false
check_package "pandas" "pandas" || all_ok=false
check_package "numpy" "numpy" || all_ok=false
check_package "matplotlib" "matplotlib" || all_ok=false
check_package "tensorboard" "tensorboard" || all_ok=false

echo "========================================"

if [ "${all_ok}" = true ]; then
    echo ""
    echo "所有依赖安装成功!"
    echo ""
    echo "下一步操作:"
    echo "  1. python download_data.py     # 下载数据集"
    echo "  2. python preprocess_data.py  # 预处理数据"
    echo "  3. bash train.sh               # 开始训练"
else
    echo ""
    echo "部分依赖安装失败，请检查上方错误信息"
    exit 1
fi
