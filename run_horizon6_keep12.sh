#!/bin/bash
set +e

MODEL_PATH="/share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
MODEL_NAME="Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"
BASE_OUT_DIR="safe_test"

echo "=== 开始并行评测 $MODEL_NAME (使用 GPU 1,2,3) ==="

# GPU1: temp_0
(
echo "--- GPU1: Temp 0 ---"
CUDA_VISIBLE_DEVICES=1 python eval_student_model.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${BASE_OUT_DIR}/${MODEL_NAME}/eval_results/temp_0" \
    --temperature 0 \
    --benches direct harm phi harmful_behaviors
echo "--- GPU1 Done ---"
) &

# GPU2: temp_1
(
echo "--- GPU2: Temp 1 ---"
CUDA_VISIBLE_DEVICES=2 python eval_student_model.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${BASE_OUT_DIR}/${MODEL_NAME}/eval_results/temp_1" \
    --temperature 1 \
    --benches direct harm phi harmful_behaviors
echo "--- GPU2 Done ---"
) &

# GPU3: temp_1_run2
(
echo "--- GPU3: Temp 1 Run2 ---"
CUDA_VISIBLE_DEVICES=3 python eval_student_model.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${BASE_OUT_DIR}/${MODEL_NAME}/eval_results/temp_1_run2" \
    --temperature 1 \
    --benches direct harm phi harmful_behaviors
echo "--- GPU3 Done ---"
) &

wait
echo "=== 第一批完成，开始 temp_1_run3 ==="

# 最后一个用 GPU1
CUDA_VISIBLE_DEVICES=1 python eval_student_model.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${BASE_OUT_DIR}/${MODEL_NAME}/eval_results/temp_1_run3" \
    --temperature 1 \
    --benches direct harm phi harmful_behaviors

echo "=== 全部完成 ==="
