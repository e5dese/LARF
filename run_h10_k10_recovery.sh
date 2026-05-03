#!/bin/bash
set +e

MODELS=(
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon10-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon10-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
)

# 4 个 (temperature, run) 任务: temp_0, temp_1, temp_1_run2, temp_1_run3
TASKS=("0:temp_0" "1:temp_1" "1:temp_1_run2" "1:temp_1_run3")

BASE_OUT_DIR="safe_test"
LOG_DIR="/share/project/lihao/projects/Self-Distillation-eval/LARF/eval_log/0501_h10_k10"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "🚀 补测 h10 k10 vote-samples8 (Llama-3.2-3B + Qwen3-4B)"
echo "策略: 每个模型 4 个 temp 任务并发跑 4 张 GPU; 模型间串行"
echo "日志目录: $LOG_DIR"
echo "=================================================="

run_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$(basename "${MODEL_PATH}")
    local SAFE_MODEL_NAME="${MODEL_NAME// /_}"

    echo ">>> 启动模型: ${MODEL_NAME}"

    for j in "${!TASKS[@]}"; do
        local GPU_ID=$j
        IFS=':' read -r TEMP SUBDIR <<< "${TASKS[$j]}"

        local OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/${SUBDIR}"
        local LOG_FILE="${LOG_DIR}/eval_log_${SAFE_MODEL_NAME}_${SUBDIR}.log"

        echo "    [GPU ${GPU_ID}] ${SUBDIR} -> ${LOG_FILE}"

        (
            CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model.py \
                --model_path "${MODEL_PATH}" \
                --output_dir "${OUTPUT_DIR}" \
                --temperature "${TEMP}" \
                --benches direct harm phi harmful_behaviors
        ) > "${LOG_FILE}" 2>&1 &
    done

    wait
    echo ">>> 完成模型: ${MODEL_NAME}"
}

for MODEL_PATH in "${MODELS[@]}"; do
    run_model "${MODEL_PATH}"
done

echo "=================================================="
echo "🎉 h10 k10 补测完成"
echo "下一步: 启动 vLLM Llama-Guard 并对两个新目录打分"
echo "=================================================="
