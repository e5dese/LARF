#!/bin/bash

# 遇到错误时退出
set -e

# 定义要评估的模型路径数组
MODELS=(
    "/share/project/huggingface/models/Qwen2.5-7B-Instruct"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/refusal-vector/Qwen2.5-7B-Instruct-refusal-vector"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/refusal-vector/Qwen2.5-7B-Instruct-refusal-vector-Reverse"
    "/share/project/huggingface/models/Qwen3-4B"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/refusal-vector/Qwen3-4B-refusal-vector"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/refusal-vector/Qwen3-4B-refusal-vector-Reverse"
    "/share/project/huggingface/models/Qwen3-8B"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/refusal-vector/Qwen3-8B-refusal-vector"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/refusal-vector/Qwen3-8B-refusal-vector-Reverse"
)

# 固定温度为 1
TEMP=1
# 补充运行的轮次标识（第2次和第3次）
RUNS=(2 3)

# 基础输出目录
BASE_OUT_DIR="safe_test"

echo "=================================================="
echo "Starting Supplementary Evaluation Job for Temp = 1"
echo "Total Models: ${#MODELS[@]}"
echo "Runs: ${RUNS[*]}"
echo "=================================================="

for MODEL_PATH in "${MODELS[@]}"; do
    # 从路径中提取模型名称
    MODEL_NAME=$(basename "${MODEL_PATH}")
    
    echo -e "\n\n>>> Preparing to evaluate model: ${MODEL_NAME}"
    echo ">>> Model Path: ${MODEL_PATH}"
    
    for RUN in "${RUNS[@]}"; do
        # 为第2次和第3次创建独立的文件夹，防止覆盖之前 temp_1 的结果
        OUTPUT_DIR="${BASE_OUT_DIR}/${MODEL_NAME}/eval_results/temp_${TEMP}_run${RUN}"
        
        echo "--------------------------------------------------"
        echo "Running evaluation for ${MODEL_NAME} with Temperature = ${TEMP} (Round ${RUN}/3)"
        echo "Output directory: ${OUTPUT_DIR}"
        echo "--------------------------------------------------"
        
        # 执行 Python 脚本
        python eval_student_model.py \
            --model_path "${MODEL_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --temperature "${TEMP}"
            
        echo "Finished Temp = ${TEMP} (Round ${RUN}/3) for ${MODEL_NAME}"
    done
done

echo "=================================================="
echo "All supplementary evaluations completed successfully!"
echo "You now have 3 runs for Temperature 1 across all models."
echo "=================================================="