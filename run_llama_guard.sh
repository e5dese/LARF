#!/bin/bash
# ==============================================================================
# Llama Guard 批量打分 — 当前任务: 给 28 个 Llama-3-8B vote-samples8 变体打分
#
# 跑前必须先启动 vLLM Llama Guard 服务, 例如:
  CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
      --model /ssddata/lihao/projects/models/Llama-Guard-4-12B \
      --port 8000 \
      --trust-remote-code \
      --tensor-parallel-size 2 \
      --max-model-len 16384 \
      --gpu-memory-utilization 0.9
#
# 服务地址 (host:port) 在 LARF/llama_guard.py 里硬编码为 http://localhost:8000/v1.
# ==============================================================================

set -e

BASE_DIR="safe_test"
LLAMA_GUARD_URL="${LLAMA_GUARD_URL:-http://localhost:8000}"

# ===== 28 个 Llama-3-8B-Instruct vote-samples8 变体 =====
# 跟 LARF/run_safe_system_prompt_new.sh 里的 MODELS 数组一一对应。
MODELS=(
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep2-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon8-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon8-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon8-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon10-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon10-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
    "Llama-3-8B-Instruct-vector-mode2-top50-horizon10-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
)

echo "=================================================="
echo "Starting Llama Guard Auto-Evaluation Batch Job"
echo "  Models   : ${#MODELS[@]} 个 Llama-3-8B vote-samples8 变体"
echo "  vLLM URL : ${LLAMA_GUARD_URL}"
echo "=================================================="

# --- vLLM health check ---
echo ">>> 检查 vLLM 服务可达性 ..."
if ! curl -sf "${LLAMA_GUARD_URL}/v1/models" >/dev/null; then
    echo "ERROR: 无法连接 vLLM 服务 (${LLAMA_GUARD_URL}/v1/models)"
    echo "       请先启动 Llama-Guard-4-12B vllm openai server (见脚本头部注释)。"
    exit 1
fi
echo "    OK"

# --- 逐个模型打分 ---
for MODEL in "${MODELS[@]}"; do
    TARGET_DIR="${BASE_DIR}/${MODEL}/eval_results"

    if [ -d "${TARGET_DIR}" ]; then
        echo "--------------------------------------------------"
        echo ">>> Evaluating Directory: ${TARGET_DIR}"
        echo "--------------------------------------------------"
        python llama_guard.py --input_dir "${TARGET_DIR}"
    else
        echo "Warning: Directory ${TARGET_DIR} does not exist. Skipping..."
    fi
done

echo "=================================================="
echo "All Llama Guard evaluations completed successfully!"
echo "=================================================="
