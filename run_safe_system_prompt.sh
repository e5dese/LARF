#!/bin/bash
set +e

# 定义要评估的模型路径 (只保留前 4 个 Reverse 模型)
MODELS=(
    # --- Reverse (alpha=1.0) ---
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/baseline/Qwen2.5-7B-Instruct-system prompt-V2-new-reverse"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/baseline/Qwen3-8B-system prompt-V2-reverse"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen2.5-7B-Instruct-refusal-vector-Reverse"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen3-8B-refusal-vector-Reverse"
    
    # --- Forward / Normal (alpha=0.0) 暂时注释掉 ---
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/baseline/Qwen2.5-7B-Instruct-system prompt-V2-new"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/baseline/Qwen3-8B-system prompt-V2"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen2.5-7B-Instruct-refusal-vector"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen3-8B-refusal-vector"
)

BASE_OUT_DIR="safe_test"

echo "=================================================="
echo "Starting Safety Evaluation for System Prompt Models"
echo "=================================================="

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    SAFE_MODEL_NAME="${MODEL_NAME// /_}"
    
    echo -e "\n\n>>> Preparing to evaluate model: ${MODEL_NAME}"
    
    # --- 1. 运行 Temp = 0 ---
    TEMP=0
    OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}"
    echo "--- Running Temp = ${TEMP} ---"
    python eval_student_model.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --temperature "${TEMP}"

    # --- 2. 运行 Temp = 1 (循环跑 3 次) ---
    TEMP=1
    for RUN in 1 2 3; do
        if [ "$RUN" -eq 1 ]; then
            OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}"
        else
            OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}_run${RUN}"
        fi

        echo "--- Running Temp = ${TEMP} (Round ${RUN}/3) ---"
        python eval_student_model.py \
            --model_path "${MODEL_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --temperature "${TEMP}"
    done
done

echo "=================================================="
echo "🎉 4 个 Reverse 模型的生成任务全部完成！"
echo "下一步：请启动 vLLM 并运行 run_llama_guard.sh 进行安全打分。"
echo "=================================================="