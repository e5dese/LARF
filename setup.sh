#!/bin/bash
# ==============================================================================
# setup.sh — 安装 LARF 安全能力测试所需的全部依赖
#
# 使用方法:
#   cd /opt/tiger/entry/LARF
#   bash setup.sh
#
# 涵盖:
#   1. 系统级工具 (git / wget 等; 仅在缺失时尝试安装)
#   2. requirements.txt 中的训练/推理 Python 依赖
#   3. vLLM (Llama Guard 打分服务需要; 未写入 requirements.txt)
# ==============================================================================

set -e

# ----------- 路径配置 ----------------------------------------------------------
ENTRY_DIR="/opt/tiger/entry"
LARF_DIR="${ENTRY_DIR}/LARF"

echo "=================================================="
echo "[setup] ENTRY_DIR = ${ENTRY_DIR}"
echo "[setup] LARF_DIR  = ${LARF_DIR}"
echo "=================================================="

mkdir -p "${ENTRY_DIR}"
cd "${LARF_DIR}"

# ----------- 1. 系统级工具 (可选, 缺啥装啥) -----------------------------------
if command -v apt-get >/dev/null 2>&1; then
    echo "[setup] 检测到 apt-get, 安装基础工具..."
    apt-get update -y || true
    apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential \
        || true
fi

# ----------- 2. pip 升级 + 中科大镜像加速 (可选) ------------------------------
python -m pip install --upgrade pip setuptools wheel

# 如果是国内环境想加速, 取消下一行注释:
# PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

# ----------- 3. PyTorch (CUDA 12.1) -------------------------------------------
# requirements.txt 指定 torch==2.5.1+cu121, 必须指定 cu121 wheel 源
echo "[setup] 安装 PyTorch 2.5.1 + cu121 ..."
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121

# ----------- 4. 推理 / 训练依赖 (requirements.txt 里的其余包) ------------------
echo "[setup] 安装 requirements.txt 其余依赖 ..."
pip install -i "${PIP_INDEX_URL}" \
    datasets \
    numpy \
    matplotlib \
    openai \
    pandas \
    peft \
    wandb \
    scikit-learn \
    tqdm \
    transformers==4.51.3

# ----------- 5. vLLM (Llama Guard 服务依赖, requirements.txt 没写) -------------
# 与 torch 2.5.1 + cu121 兼容的版本; 如果 vLLM 自动拉取的 torch 把版本顶掉,
# 用 --no-deps 加单独装它需要的依赖, 但绝大多数情况下直接装就行。
echo "[setup] 安装 vLLM (用于启动 Llama Guard OpenAI 兼容服务) ..."
pip install -i "${PIP_INDEX_URL}" "vllm>=0.6.3"

# ----------- 6. HDFS fuse 写入目录预创建 --------------------------------------
HDFS_BASE="/mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt"
if [ -d "/mnt/hdfs" ]; then
    mkdir -p "${HDFS_BASE}" || echo "[setup] WARN: 创建 ${HDFS_BASE} 失败, 请确认 HDFS fuse 已挂载"
else
    echo "[setup] WARN: /mnt/hdfs 不存在, 跑 run.sh 时再确认 HDFS 挂载"
fi

# ----------- 7. 校验 -----------------------------------------------------------
echo "=================================================="
echo "[setup] 安装完成, 校验关键包版本:"
python - <<'PY'
import importlib, sys
for pkg in ["torch", "transformers", "vllm", "openai", "pandas", "tqdm", "peft"]:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "unknown")
        print(f"  - {pkg:14s} {ver}")
    except Exception as e:
        print(f"  - {pkg:14s} IMPORT FAILED: {e}", file=sys.stderr)
PY
echo "=================================================="
echo "[setup] 完成! 接下来:"
echo "  1) 把待测模型 / Llama-Guard 权重下载到 ${ENTRY_DIR} 下"
echo "  2) 编辑 ${LARF_DIR}/run.sh 里的 MODELS 与 LLAMA_GUARD_MODEL"
echo "  3) bash ${LARF_DIR}/run.sh"
echo "=================================================="
