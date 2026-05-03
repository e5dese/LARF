#!/bin/bash
set +e

# 恢复脚本：继续未完成的 0427_vote54 实验
# 共 8 个模型未完成

MODELS=(
    # horizon4 缺 run3 (5个)
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # horizon6 缺 run3 (3个) + keep12 完全未测
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # horizon8 完全未开始 (3个)
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon8-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon8-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon8-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # horizon10 完全未开始 (2个)
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon10-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon10-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
)

BATCH_SIZE=4

BASE_OUT_DIR="safe_test"

LOG_DIR="/share/project/lihao/projects/Self-Distillation-eval/LARF/eval_log/0427_vote54_recovery"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "恢复运行未完成的 vote54 实验 (共 13 个模型)"
echo "日志目录: $LOG_DIR"
echo "=================================================="

for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    GPU_IDS=$(( i % BATCH_SIZE ))

    MODEL_NAME=$(basename "${MODEL_PATH}")
    SAFE_MODEL_NAME="${MODEL_NAME// /_}"

    LOG_FILE="${LOG_DIR}/eval_log_${SAFE_MODEL_NAME}.log"

    echo ">>> [Task $i] 启动评测: ${MODEL_NAME}"
    echo "    - GPU: ${GPU_IDS}"
    echo "    - 日志: ${LOG_FILE}"

    (
        echo "--- Started Evaluation for ${MODEL_NAME} on GPU ${GPU_IDS} ---"

        # --- 1. Temp = 0 ---
        TEMP=0
        OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}"
        CUDA_VISIBLE_DEVICES=${GPU_IDS} python eval_student_model.py \
            --model_path "${MODEL_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --temperature "${TEMP}" \
            --benches direct harm phi harmful_behaviors

        # --- 2. Temp = 1 (跑 3 次) ---
        TEMP=1
        for RUN in 1 2 3; do
            if [ "$RUN" -eq 1 ]; then
                OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}"
            else
                OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}_run${RUN}"
            fi

            CUDA_VISIBLE_DEVICES=${GPU_IDS} python eval_student_model.py \
                --model_path "${MODEL_PATH}" \
                --output_dir "${OUTPUT_DIR}" \
                --temperature "${TEMP}" \
                --benches direct harm phi harmful_behaviors
        done

        echo "--- Finished Evaluation for ${MODEL_NAME} ---"
    ) > "${LOG_FILE}" 2>&1 &

    if (( (i + 1) % BATCH_SIZE == 0 )); then
        echo ">>> Batch full, waiting..."
        wait
    fi

done

wait

echo "=================================================="
echo "恢复实验完成！"
echo "=================================================="
