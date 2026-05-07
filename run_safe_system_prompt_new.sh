#!/bin/bash
set +e

# ===== 模型根目录 (每个模型实际权重在 <name>/checkpoint-25 子目录下) =====
MODEL_ROOT="/ssddata/lihao/projects/SDFT-for-Safe-Alignment/model_weight"
CKPT_SUBDIR="checkpoint-25"

# ===== 模型族标签 (改这一行 / 或调用前 export MODEL_FAMILY=...) =====
# 当前活跃: llama3_8b (重训于 2026-05-06, ckpt mtime May 6 03:12)
# 历史轮次:
#   - llama3_8b 旧权重: 2026-05-04 跑过一轮
#   - qwen3_8b           : 2026-05-05 跑过一轮
MODEL_FAMILY="${MODEL_FAMILY:-llama3_8b}"

# ===== 28 个变体的 (horizon, keep) 模式, 两个模型族共用 =====
HK_PAIRS=(
    "1 0" "1 2" "1 4" "1 6" "1 8" "1 10" "1 12"
    "2 0"       "2 4" "2 6" "2 8" "2 10" "2 12"
    "4 0"             "4 6" "4 8" "4 10" "4 12"
    "6 0"                   "6 8" "6 10" "6 12"
    "8 0"                         "8 10" "8 12"
    "10 0"                        "10 10" "10 12"
)

# 根据 MODEL_FAMILY 选 base 名
case "${MODEL_FAMILY}" in
    llama3_8b)  MODEL_BASE="Llama-3-8B-Instruct" ;;
    qwen3_8b)   MODEL_BASE="Qwen3-8B" ;;
    *)
        echo "ERROR: 未知 MODEL_FAMILY='${MODEL_FAMILY}', 支持: llama3_8b / qwen3_8b" >&2
        exit 1
        ;;
esac

# 拼出 28 个完整路径
MODELS=()
for hk in "${HK_PAIRS[@]}"; do
    h="${hk% *}"
    k="${hk#* }"
    MODELS+=( "${MODEL_ROOT}/${MODEL_BASE}-vector-mode2-top50-horizon${h}-alpha1.0-keep${k}-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100/${CKPT_SUBDIR}" )
done


# 4 GPU 轮转；批大小 = 4，每批跑完再起下一批
BATCH_SIZE=4

# 实际用的物理 GPU id (按位置映射): GPU_LIST[i % BATCH_SIZE]
# 跑前 nvidia-smi 看了下: 0/1/2/3 都被占用, 4-7 空闲, 故指定到这 4 张。
GPU_LIST=(4 5 6 7)

BASE_OUT_DIR="safe_test"

# === 日志目录定义与创建 ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${SCRIPT_DIR}/eval_log/$(date +%m%d)_${MODEL_FAMILY}"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "🚀 开始并行安全评测 (${#GPU_LIST[@]} 卡同时跑, ${#MODELS[@]} 个 ${MODEL_FAMILY} 模型)"
echo "模型根目录: ${MODEL_ROOT}  (子目录: ${CKPT_SUBDIR})"
echo "日志输出至: $LOG_DIR"
echo "=================================================="

# ===== 跑 LARF/datasets/ 里的 7 个新数据集 (Goal 列), 用 usedatasets 版本脚本 =====
# datasets/ 下的 csv: advbench / ALERT / HarmfulQA / JBB-Behaviors / PKU-SafeRLHF-30K
#                     sorry_bench_202503 / harmbench (注意: 这个 harmbench 来自
#                     datasets/, 跟 safe_test/harmbench.csv 不是同一份文件)
DATA_DIR="datasets"
BENCHES=(advbench ALERT HarmfulQA JBB-Behaviors PKU-SafeRLHF-30K sorry_bench_202503 harmbench)

# 两阶段: 先所有 28 个模型跑完 temp=0, 再跑 temp=1 (各一遍即可, 不再重复 3 次)
for TEMP in 0 1; do
    echo ""
    echo "=================================================="
    echo "🚀 Phase TEMP=${TEMP}: 28 个模型 × 7 个 datasets/ 数据集 (4 卡轮转)"
    echo "=================================================="

    for i in "${!MODELS[@]}"; do
        MODEL_PATH="${MODELS[$i]}"
        GPU_IDS="${GPU_LIST[$(( i % BATCH_SIZE ))]}"

        # 用上层目录名作为 SAFE_MODEL_NAME (而不是 "checkpoint-25"),
        # 避免所有结果都被覆盖到同一个 checkpoint-25/ 目录里。
        MODEL_NAME=$(basename "$(dirname "${MODEL_PATH}")")
        SAFE_MODEL_NAME="${MODEL_NAME// /_}"

        LOG_FILE="${LOG_DIR}/eval_log_${SAFE_MODEL_NAME}_temp${TEMP}.log"
        OUTPUT_DIR="${BASE_OUT_DIR}/${SAFE_MODEL_NAME}/eval_results/temp_${TEMP}"

        echo ">>> [Task $i / TEMP=$TEMP] ${MODEL_NAME}  (GPU ${GPU_IDS})"
        echo "    log : ${LOG_FILE}"
        echo "    out : ${OUTPUT_DIR}"

        (
            echo "--- Started ${MODEL_NAME} on GPU ${GPU_IDS}, TEMP=${TEMP} ---"
            CUDA_VISIBLE_DEVICES=${GPU_IDS} python eval_student_model_usedatasets.py \
                --model_path "${MODEL_PATH}" \
                --output_dir "${OUTPUT_DIR}" \
                --data_dir "${DATA_DIR}" \
                --temperature "${TEMP}" \
                --benches "${BENCHES[@]}"
            echo "--- Finished ${MODEL_NAME}, TEMP=${TEMP} ---"
        ) > "${LOG_FILE}" 2>&1 &

        # 每跑满 BATCH_SIZE 个就 wait, 避免 GPU 冲突
        if (( (i + 1) % BATCH_SIZE == 0 )); then
            echo ">>> Batch full ($(( (i+1) / BATCH_SIZE ))), waiting..."
            wait
        fi
    done

    # 等本 TEMP 阶段最后一批 (可能不满 BATCH_SIZE) 跑完, 再开 TEMP=1
    wait
    echo "✅ Phase TEMP=${TEMP} done."
done

echo ""
echo "=================================================="
echo "🎉 所有 ${#MODELS[@]} 个 ${MODEL_FAMILY} 模型 × 2 个温度 × 7 个 datasets 全部完成！"
echo "下一步：请启动 vLLM 并运行 run_llama_guard.sh 进行安全打分。"
echo "=================================================="
