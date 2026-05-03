#!/bin/bash

# 遇到错误继续执行
set +e

# 定义要评估的模型路径
MODELS=(
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen2.5-7B-Instruct-system prompt-V1-new"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen2.5-7B-Instruct-system prompt-V2-new"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-4B-system prompt-V1"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-4B-system prompt-V2"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-8B-system prompt-V1-new"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-8B-system prompt-V2"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen2.5-7B-Instruct-system prompt-V1-new-reverse"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen2.5-7B-Instruct-system prompt-V2-new-reverse"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-4B-system prompt-V1-reverse"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-4B-system prompt-V2-reverse"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-8B-system prompt-V1-new-reverse"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-8B-system prompt-V2-reverse"
)

# 基础输出目录
BASE_OUT_DIR="safe_test"

echo "=================================================="
echo "Starting Safety Evaluation for System Prompt Models"
echo "=================================================="

for MODEL_PATH in "${MODELS[@]}"; do
    # 从路径提取模型名称
    MODEL_NAME=$(basename "${MODEL_PATH}")
    
    # 替换空格为下划线，保证路径安全
    SAFE_MODEL_NAME="${MODEL_NAME// /_}"
    
    echo -e "\n\n>>> Preparing to evaluate model: ${MODEL_NAME}"
    
    # --------------------------------------------------
    # 1. 运行 Temp = 0 (只跑 1 次)
    # --------------------------------------------------
    TEMP=0
    OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}"
    echo "--- Running Temp = ${TEMP} ---"
    python eval_student_model.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --temperature "${TEMP}"

    # --------------------------------------------------
    # 2. 运行 Temp = 1 (循环跑 3 次)
    # --------------------------------------------------
    TEMP=1
    for RUN in 1 2 3; do
        # 匹配之前的文件夹命名习惯
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
echo "🎉 所有 6 个 System Prompt 模型的生成任务全部完成！"
echo "下一步：请启动 vLLM 并运行 run_llama_guard.sh 进行安全打分。"
echo "=================================================="