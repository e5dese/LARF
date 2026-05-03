#!/bin/bash
# ==============================================================================
# run_pipeline.sh — LARF 安全能力测试一站式脚本
#
# 流程:
#   阶段 1 (推理)   : 8 卡并行调用 eval_student_model.py, 对每个模型分别在
#                     temp=0 (1 次) 与 temp=1 (3 次) 下生成回答。
#   阶段 2 (启服务) : 用 vLLM 启动 Llama Guard 的 OpenAI 兼容服务 (端口 8000),
#                     等待健康检查通过。
#   阶段 3 (打分)   : 调用 llama_guard.py, 把 unsafe 数量写回各 JSON 文件。
#
# 全部中间/最终结果直接写入 HDFS fuse 目录:
#   /mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt/${RUN_NAME}/
#
# 使用方法:
#   1) 先执行过 setup.sh
#   2) 把模型权重下载到 ${ENTRY_DIR} 下面, 然后填到下面的 MODELS 数组
#   3) 把 Llama Guard 模型下载到 ${ENTRY_DIR}, 把路径填到 LLAMA_GUARD_MODEL
#   4) bash /opt/tiger/entry/LARF/run_pipeline.sh
# ==============================================================================

set +e  # 单个模型失败不阻塞后续

# =========================== 用户需要确认/修改的部分 ===========================

# --- 项目路径 ---
ENTRY_DIR="/opt/tiger/entry"
LARF_DIR="${ENTRY_DIR}/LARF"

# --- HDFS 输出根目录 (新开一个子目录, 默认按时间戳命名, 可改成有意义的名字) ---
HDFS_BASE="/mnt/hdfs/tiktok_aiic/user/lihao.612/ckpt"
RUN_NAME="${RUN_NAME:-larf_eval_$(date +%Y%m%d_%H%M%S)}"
HDFS_RUN_DIR="${HDFS_BASE}/${RUN_NAME}"

# --- Llama Guard 模型路径 (下载好后填进来) ---
# 例如: ${ENTRY_DIR}/Llama-Guard-4-12B
LLAMA_GUARD_MODEL="${ENTRY_DIR}/<FILL_LLAMA_GUARD_MODEL_DIR>"

# --- 待测学生模型列表 (绝对路径; 放到 ${ENTRY_DIR} 下, 路径自己填) ---
MODELS=(
    # "${ENTRY_DIR}/<FILL_MODEL_DIR_1>"
    # "${ENTRY_DIR}/<FILL_MODEL_DIR_2>"
    # "${ENTRY_DIR}/<FILL_MODEL_DIR_N>"
)

# --- GPU 并行批大小 (= 同时跑的模型数; 一张卡跑一个模型) ---
BATCH_SIZE="${BATCH_SIZE:-8}"

# --- vLLM 服务参数 ---
VLLM_PORT="${VLLM_PORT:-8000}"
# Llama-Guard-4-12B 单卡放得下, 默认占 GPU 0; 多卡可改 tensor-parallel-size
VLLM_GPU="${VLLM_GPU:-0}"
VLLM_TP="${VLLM_TP:-1}"

# --- 评测的子任务 (对应 LARF/safe_test/ 下的 csv) ---
BENCHES=(direct harm phi harmful_behaviors)

# =============================================================================

# 校验 LARF 目录与必要文件
cd "${LARF_DIR}" || { echo "[run] LARF_DIR 不存在: ${LARF_DIR}"; exit 1; }
for f in eval_student_model.py llama_guard.py safe_test/directHarm4.csv; do
    [ -e "${LARF_DIR}/${f}" ] || { echo "[run] 缺少文件: ${LARF_DIR}/${f}"; exit 1; }
done

# 校验模型列表非空
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "[run] ERROR: MODELS 数组为空, 请在 ${LARF_DIR}/run_pipeline.sh 里填入模型路径。"
    exit 1
fi

# 校验 Llama Guard 模型路径
if [[ "${LLAMA_GUARD_MODEL}" == *"<FILL_"* ]] || [ ! -d "${LLAMA_GUARD_MODEL}" ]; then
    echo "[run] ERROR: LLAMA_GUARD_MODEL 未配置或目录不存在: ${LLAMA_GUARD_MODEL}"
    exit 1
fi

# 准备输出目录
mkdir -p "${HDFS_RUN_DIR}" || { echo "[run] 无法创建 HDFS 输出目录: ${HDFS_RUN_DIR}"; exit 1; }
LOG_DIR="${HDFS_RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

# eval_student_model.py 同时会从 LARF/safe_test/ 读取 csv 数据,
# 所以只把每个模型的输出目录指到 HDFS 下, 数据集文件留在原处。
RESULTS_HDFS_ROOT="${HDFS_RUN_DIR}/safe_test"
mkdir -p "${RESULTS_HDFS_ROOT}"

echo "=================================================="
echo "[run] RUN_NAME           = ${RUN_NAME}"
echo "[run] HDFS_RUN_DIR       = ${HDFS_RUN_DIR}"
echo "[run] LLAMA_GUARD_MODEL  = ${LLAMA_GUARD_MODEL}"
echo "[run] # of MODELS        = ${#MODELS[@]}"
echo "[run] BATCH_SIZE (GPUs)  = ${BATCH_SIZE}"
echo "[run] vLLM port / GPU    = ${VLLM_PORT} / ${VLLM_GPU} (TP=${VLLM_TP})"
echo "[run] LOG_DIR            = ${LOG_DIR}"
echo "=================================================="

# ============================== 阶段 1: 推理 ===================================
echo ""
echo "=================================================="
echo "[run] 阶段 1/3: 8 卡并行生成回答"
echo "=================================================="

for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    GPU_ID=$(( i % BATCH_SIZE ))

    MODEL_NAME=$(basename "${MODEL_PATH}")
    SAFE_MODEL_NAME="${MODEL_NAME// /_}"

    LOG_FILE="${LOG_DIR}/eval_${SAFE_MODEL_NAME}.log"
    echo ">>> [Task $i] ${MODEL_NAME}  (GPU ${GPU_ID})  log=${LOG_FILE}"

    (
        echo "--- Started Evaluation for ${MODEL_NAME} on GPU ${GPU_ID} ---"

        # temp = 0
        TEMP=0
        OUTPUT_DIR="${RESULTS_HDFS_ROOT}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model.py \
            --model_path "${MODEL_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --temperature "${TEMP}" \
            --benches "${BENCHES[@]}"

        # temp = 1, 跑 3 次
        TEMP=1
        for RUN in 1 2 3; do
            if [ "${RUN}" -eq 1 ]; then
                OUTPUT_DIR="${RESULTS_HDFS_ROOT}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}"
            else
                OUTPUT_DIR="${RESULTS_HDFS_ROOT}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}_run${RUN}"
            fi
            CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model.py \
                --model_path "${MODEL_PATH}" \
                --output_dir "${OUTPUT_DIR}" \
                --temperature "${TEMP}" \
                --benches "${BENCHES[@]}"
        done

        echo "--- Finished Evaluation for ${MODEL_NAME} ---"
    ) > "${LOG_FILE}" 2>&1 &

    # 凑满一批就 wait, 避免 GPU 冲突
    if (( (i + 1) % BATCH_SIZE == 0 )); then
        echo ">>> Batch full ($(( (i+1) / BATCH_SIZE ))), waiting..."
        wait
    fi
done
wait
echo "[run] 阶段 1 完成。"

# ============================== 阶段 2: 启 vLLM ================================
echo ""
echo "=================================================="
echo "[run] 阶段 2/3: 启动 vLLM Llama Guard 服务 (port=${VLLM_PORT}, GPU=${VLLM_GPU})"
echo "=================================================="

VLLM_LOG="${LOG_DIR}/vllm_llama_guard.log"

CUDA_VISIBLE_DEVICES=${VLLM_GPU} \
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "${LLAMA_GUARD_MODEL}" \
        --port "${VLLM_PORT}" \
        --tensor-parallel-size "${VLLM_TP}" \
        --trust-remote-code \
        > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[run] vLLM PID = ${VLLM_PID}, log = ${VLLM_LOG}"

# 健康检查: 最多等 10 分钟
echo "[run] 等待 vLLM 服务就绪 ..."
READY=0
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        READY=1
        echo "[run] vLLM 服务就绪 (用时 $((i*5)) 秒)。"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[run] ERROR: vLLM 进程已退出, 看日志 ${VLLM_LOG}"
        exit 1
    fi
    sleep 5
done
if [ "${READY}" -ne 1 ]; then
    echo "[run] ERROR: vLLM 10 分钟未就绪, 终止。"
    kill "${VLLM_PID}" 2>/dev/null || true
    exit 1
fi

# 退出时务必清理 vLLM 进程
cleanup() {
    echo "[run] 清理 vLLM 进程 (PID=${VLLM_PID}) ..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ============================== 阶段 3: Llama Guard 打分 =======================
echo ""
echo "=================================================="
echo "[run] 阶段 3/3: 调用 llama_guard.py 给所有 JSON 打分"
echo "=================================================="

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "${MODEL_PATH}")
    SAFE_MODEL_NAME="${MODEL_NAME// /_}"
    TARGET_DIR="${RESULTS_HDFS_ROOT}/${SAFE_MODEL_NAME}/eval_results"

    if [ -d "${TARGET_DIR}" ]; then
        echo "--------------------------------------------------"
        echo ">>> 打分: ${TARGET_DIR}"
        echo "--------------------------------------------------"
        python llama_guard.py --input_dir "${TARGET_DIR}" \
            2>&1 | tee "${LOG_DIR}/guard_${SAFE_MODEL_NAME}.log"
    else
        echo "[run] WARN: ${TARGET_DIR} 不存在, 跳过。"
    fi
done

echo ""
echo "=================================================="
echo "[run] 全部完成! 结果目录: ${HDFS_RUN_DIR}"
echo "=================================================="
