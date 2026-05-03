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

# 定义要测试的 temperature
TEMPERATURES=(0 1)

# 基础输出目录
BASE_OUT_DIR="safe_test"

echo "=================================================="
echo "Starting Batch Evaluation Job"
echo "Total Models: ${#MODELS[@]}"
echo "Temperatures: ${TEMPERATURES[*]}"
echo "=================================================="

for MODEL_PATH in "${MODELS[@]}"; do
    # 从路径中提取模型名称 (例如: Qwen2.5-7B-Instruct-refusal-vector)
    # 使用 basename 获取最后一级目录名
    MODEL_NAME=$(basename "${MODEL_PATH}")
    
    echo -e "\n\n>>> Preparing to evaluate model: ${MODEL_NAME}"
    echo ">>> Model Path: ${MODEL_PATH}"
    
    for TEMP in "${TEMPERATURES[@]}"; do
        # 为每个模型和温度组合创建一个专门的输出文件夹
        OUTPUT_DIR="${BASE_OUT_DIR}/${MODEL_NAME}/eval_results/temp_${TEMP}"
        
        echo "--------------------------------------------------"
        echo "Running evaluation for ${MODEL_NAME} with Temperature = ${TEMP}"
        echo "Output directory: ${OUTPUT_DIR}"
        echo "--------------------------------------------------"
        
        # 执行 Python 脚本
        python eval_student_model.py \
            --model_path "${MODEL_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --temperature "${TEMP}"
            
        echo "Finished Temp = ${TEMP} for ${MODEL_NAME}"
    done
done

echo "=================================================="
echo "All evaluations completed successfully!"
echo "Next step: Run llama_guard.py to get safety scores"
echo "=================================================="