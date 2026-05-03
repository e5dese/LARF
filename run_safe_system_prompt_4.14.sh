#!/bin/bash
set +e

# 定义要评估的模型路径 (对应 Alpha=0.0 的 4 个模型)
MODELS=(
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/baseline/Qwen2.5-7B-Instruct-system prompt-V2-new"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/baseline/Qwen3-8B-system prompt-V2"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen2.5-7B-Instruct-refusal-vector"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen3-8B-refusal-vector"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen2.5-7B-Instruct-refusal-vector-Reverse"
    "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen3-8B-refusal-vector-Reverse"
)

# 对应的 GPU 分配 (4 个任务，每任务 2 张卡，刚好打满 8 张卡)
GPU_ALLOCATIONS=("6,7")

BASE_OUT_DIR="safe_test"

# === 日志目录定义与创建 ===
LOG_DIR="/share/project/lihao/projects/Self-Distillation-eval/LARF/eval_log/0415"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "🚀 开始并行安全评测 (4 个模型同时进行，打满 8 卡)"
echo "日志将统一输出至: $LOG_DIR"
echo "=================================================="

for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    GPU_IDS="${GPU_ALLOCATIONS[$i]}"
    
    MODEL_NAME=$(basename "${MODEL_PATH}")
    SAFE_MODEL_NAME="${MODEL_NAME// /_}"
    
    # 为每个模型指定带有绝对路径的独立日志文件
    LOG_FILE="${LOG_DIR}/eval_log_${SAFE_MODEL_NAME}.log"
    
    echo ">>> [Task $i] 正在启动评测: ${MODEL_NAME}"
    echo "    - 分配 GPU: ${GPU_IDS}"
    echo "    - 日志文件: ${LOG_FILE}"

    # 将单个模型的所有生成任务打包到一个子 shell 中，并在后台 (&) 运行
    (
        echo "--- Started Evaluation for ${MODEL_NAME} on GPU ${GPU_IDS} ---"
        
        # --- 1. 运行 Temp = 0 ---
        TEMP=0
        OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}"
        CUDA_VISIBLE_DEVICES=${GPU_IDS} python eval_student_model.py \
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

            CUDA_VISIBLE_DEVICES=${GPU_IDS} python eval_student_model.py \
                --model_path "${MODEL_PATH}" \
                --output_dir "${OUTPUT_DIR}" \
                --temperature "${TEMP}"
        done
        
        echo "--- Finished Evaluation for ${MODEL_NAME} ---"
    ) > "${LOG_FILE}" 2>&1 &  # 重定向输出并放后台
    
done

echo ""
echo "⌛ 所有 4 个模型的评测流水线已放置后台并行运行！等待全部完成..."
# wait 会阻塞当前脚本，直到所有后台的 & 任务都跑完
wait 

echo "=================================================="
echo "🎉 所有 4 个模型的并行生成任务全部完成！"
echo "下一步：请启动 vLLM 并运行 run_llama_guard.sh 进行安全打分。"
echo "=================================================="