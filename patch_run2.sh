#!/bin/bash
set +e

echo "=================================================="
echo "🚀 开始专门补跑缺失的 Temp=1 (Run 2) 数据块..."
echo "=================================================="

# 激活环境
source /share/project/lihao/miniconda3/etc/profile.d/conda.sh
conda activate songzijun
cd /share/project/lihao/projects/Self-Distillation-eval/LARF/

# 确保旧的脏文件夹已经被彻底删除 (防止再次被跳过)
rm -rf "safe_test/Qwen2.5-7B-Instruct-system_prompt-V2-new-reverse/eval_results/temp_1_run2"
rm -rf "safe_test/Qwen3-8B-system_prompt-V2-reverse/eval_results/temp_1_run2"
rm -rf "safe_test/Qwen3-8B-refusal-vector-Reverse/eval_results/temp_1_run2"

# --------------------------------------------------
# 补丁 1: Qwen2.5-7B Baseline (V2-reverse) -> 使用 GPU 0,1
# --------------------------------------------------
echo ">>> [后台任务 1] 正在补跑: Qwen2.5-7B Baseline (GPU: 0,1)"
CUDA_VISIBLE_DEVICES=0,1 python eval_student_model.py \
    --model_path "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen2.5-7B-Instruct-system prompt-V2-new-reverse" \
    --output_dir "safe_test/Qwen2.5-7B-Instruct-system_prompt-V2-new-reverse/eval_results/temp_1_run2" \
    --temperature 1 > patch_log_qwen25.log 2>&1 &

# --------------------------------------------------
# 补丁 2: Qwen3-8B Baseline (V2-reverse) -> 使用 GPU 2,3
# --------------------------------------------------
echo ">>> [后台任务 2] 正在补跑: Qwen3-8B Baseline (GPU: 2,3)"
CUDA_VISIBLE_DEVICES=2,3 python eval_student_model.py \
    --model_path "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-8B-system prompt-V2-reverse" \
    --output_dir "safe_test/Qwen3-8B-system_prompt-V2-reverse/eval_results/temp_1_run2" \
    --temperature 1 > patch_log_qwen3_base.log 2>&1 &

# --------------------------------------------------
# 补丁 3: Qwen3-8B Refusal Vector (Reverse) -> 使用 GPU 4,5
# --------------------------------------------------
echo ">>> [后台任务 3] 正在补跑: Qwen3-8B Refusal Vector (GPU: 4,5)"
CUDA_VISIBLE_DEVICES=4,5 python eval_student_model.py \
    --model_path "/share/project/lihao/projects/Self-Distillation/Final_checkpoints/refusal-vector/Qwen3-8B-refusal-vector-Reverse" \
    --output_dir "safe_test/Qwen3-8B-refusal-vector-Reverse/eval_results/temp_1_run2" \
    --temperature 1 > patch_log_qwen3_refusal.log 2>&1 &


echo "⌛ 3 个补跑任务已放入后台并行执行，请稍作等待..."
wait

echo "=================================================="
echo "🎉 所有缺失的 Run 2 数据块已完美补齐！"
echo "你可以再次运行 Llama Guard 进行最终打分了。"
echo "=================================================="