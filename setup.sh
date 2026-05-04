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

# ----------- 1. requirements.txt 其余依赖------------------------------
echo "[setup] 安装 requirements.txt ..."
pip install -r "${LARF_DIR}/requirements.txt" --user