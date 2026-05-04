#!/bin/bash
# ==============================================================================
# run.sh — LARF 安全能力测试
#   阶段 1: 并行跑 eval_student_model.py (temp=0 一次 + temp=1 三次)
#   阶段 2: vLLM 起 Llama-Guard 服务 (port 8000, GPU 0)
#   阶段 3: llama_guard.py 给 JSON 打分
# ==============================================================================
set +e

ENTRY_DIR="/opt/tiger/entry"
LARF_DIR="${ENTRY_DIR}/LARF"
LLAMA_GUARD_MODEL="${ENTRY_DIR}/Llama-Guard-4-12B"

CKPT_ROOT="/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts"
CKPT_GLOB="${CKPT_GLOB:-Llama-3-8B-Instruct-vector-*-PKU_UnSafeRLHF_100}"

HDFS_RUN_DIR="/mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt/larf_eval_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${HDFS_RUN_DIR}/logs"
RESULTS_ROOT="${HDFS_RUN_DIR}/safe_test"
mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"

BATCH_SIZE=8
BENCHES=(direct harm phi harmful_behaviors)

shopt -s nullglob
MODELS=("${CKPT_ROOT}"/${CKPT_GLOB})
shopt -u nullglob
echo "[run] 发现 ${#MODELS[@]} 个模型, 输出 → ${HDFS_RUN_DIR}"

cd "${LARF_DIR}"

# ============ 阶段 1: 推理 ============
for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    GPU_ID=$(( i % BATCH_SIZE ))
    NAME=$(basename "${MODEL_PATH}")

    EVAL_PATH=""; LATEST=-1
    shopt -s nullglob
    for c in "${MODEL_PATH}"/checkpoint-*; do
        [ -d "$c" ] || continue
        step="${c##*/checkpoint-}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        (( step > LATEST )) && LATEST=$step && EVAL_PATH="$c"
    done
    shopt -u nullglob
    [ -z "${EVAL_PATH}" ] && { echo "[run] WARN: ${NAME} 无 checkpoint, 跳过"; continue; }

    LOG="${LOG_DIR}/eval_${NAME}.log"
    echo ">>> [GPU ${GPU_ID}] ${NAME} → $(basename ${EVAL_PATH})"
    (
        OUT_BASE="${RESULTS_ROOT}/${NAME}/eval_results"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model.py \
            --model_path "${EVAL_PATH}" --output_dir "${OUT_BASE}/temp_0" \
            --temperature 0 --benches "${BENCHES[@]}"
        for r in 1 2 3; do
            sub="temp_1"; [ "$r" -gt 1 ] && sub="temp_1_run${r}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model.py \
                --model_path "${EVAL_PATH}" --output_dir "${OUT_BASE}/${sub}" \
                --temperature 1 --benches "${BENCHES[@]}"
        done
    ) > "${LOG}" 2>&1 &

    (( (i + 1) % BATCH_SIZE == 0 )) && wait
done
wait
echo "[run] 阶段 1 完成"

# ============ 阶段 2: vLLM Llama-Guard 服务 (GPU 0, port 8000) ============
VLLM_LOG="${LOG_DIR}/vllm_llama_guard.log"
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model "${LLAMA_GUARD_MODEL}" --port 8000 \
    --tensor-parallel-size 1 --trust-remote-code \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap "kill ${VLLM_PID} 2>/dev/null" EXIT
echo "[run] 等待 vLLM (PID=${VLLM_PID}) ..."
for _ in $(seq 1 120); do
    curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && { echo "[run] vLLM 就绪"; break; }
    kill -0 ${VLLM_PID} 2>/dev/null || { echo "[run] vLLM 退出, 见 ${VLLM_LOG}"; exit 1; }
    sleep 5
done

# ============ 阶段 3: Llama Guard 打分 ============
for MODEL_PATH in "${MODELS[@]}"; do
    NAME=$(basename "${MODEL_PATH}")
    TARGET="${RESULTS_ROOT}/${NAME}/eval_results"
    [ -d "${TARGET}" ] || continue
    echo ">>> 打分: ${TARGET}"
    python llama_guard.py --input_dir "${TARGET}" 2>&1 | tee "${LOG_DIR}/guard_${NAME}.log"
done

echo "[run] 全部完成! ${HDFS_RUN_DIR}"
