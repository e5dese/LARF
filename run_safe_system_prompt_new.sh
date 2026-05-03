#!/bin/bash
set +e

# 定义要评估的模型路径 (目前是这 4 个 Reverse 模型)
MODELS=(
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/baseline/Qwen2.5-7B-Instruct-system prompt-V2-new-reverse"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/baseline/Qwen3-8B-system prompt-V2-reverse"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen2.5-7B-Instruct-refusal-vector-Reverse"
    # "/share/project/lihao/projects/Self-Distillation/Final_checkpoints-4.13/refusal-vector/Qwen3-8B-refusal-vector-Reverse"
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-mode0-alpha1.0-jailbreakbench
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-mode0-alpha1.0-jailbreakbench
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-mode1-top50-alpha1.0-jailbreakbench
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-mode2-top50-alpha1.0-jailbreakbench
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-mode0-alpha0.0-jailbreakbench
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-mode1-top50-alpha0.0-jailbreakbench
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-mode2-top50-alpha0.0-jailbreakbench
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-mode0-alpha1.0-jailbreakbench
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-mode1-top50-alpha1.0-jailbreakbench
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-mode2-top50-alpha1.0-jailbreakbench
    # alpha0.0 模型目录缺失权重文件，跳过
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode0-alpha0.0-PKU_UnSafeRLHF_500
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode1-top50-alpha0.0-PKU_UnSafeRLHF_500
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha0.0-PKU_UnSafeRLHF_500
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode0-alpha1.0-PKU_UnSafeRLHF_500
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode1-top50-alpha1.0-PKU_UnSafeRLHF_500
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-PKU_UnSafeRLHF_500
    # Qwen3-8B PKU_UnSafeRLHF_100 alpha1.0
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode0-alpha1.0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode1-top50-alpha1.0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon10-alpha1.0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon1-alpha1.0-PKU_UnSafeRLHF_100
    # Llama-3.2-3B-Instruct PKU_UnSafeRLHF_100 alpha1.0
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode0-alpha1.0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode1-top50-alpha1.0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon10-alpha1.0-PKU_UnSafeRLHF_100
    # 24 个 keep0/keep10 变体 PKU_UnSafeRLHF_100 alpha1.0
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode0-alpha1.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode0-alpha1.0-keep10-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode1-top50-alpha1.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode1-top50-alpha1.0-keep10-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon1-alpha1.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon1-alpha1.0-keep10-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon10-alpha1.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon10-alpha1.0-keep10-PKU_UnSafeRLHF_100
    # 重训练后的 Qwen3-4B keep0/keep10 变体（权重已更新，路径未变）
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode0-alpha1.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode0-alpha1.0-keep10-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode1-top50-alpha1.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode1-top50-alpha1.0-keep10-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep10-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon10-alpha1.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon10-alpha1.0-keep10-PKU_UnSafeRLHF_100
    # Llama-3-8B-Instruct keep0/keep10 变体 PKU_UnSafeRLHF_100 alpha0.0
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode0-alpha0.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode0-alpha0.0-keep10-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode1-top50-alpha0.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode1-top50-alpha0.0-keep10-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha0.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha0.0-keep10-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon10-alpha0.0-keep0-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon10-alpha0.0-keep10-PKU_UnSafeRLHF_100
    # 32 个 vote-samples8 horizon{N}-keep{N} 变体 (Llama-3-8B/Llama-3.2-3B/Qwen3-4B/Qwen3-8B × horizon 1..8)
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep1-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep2-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon3-alpha1.0-keep3-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon5-alpha1.0-keep5-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon7-alpha1.0-keep7-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3-8B-Instruct-vector-mode2-top50-horizon8-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep1-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep2-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon3-alpha1.0-keep3-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon5-alpha1.0-keep5-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon7-alpha1.0-keep7-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon8-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep1-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon2-alpha1.0-keep2-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon3-alpha1.0-keep3-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon5-alpha1.0-keep5-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon7-alpha1.0-keep7-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon8-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon1-alpha1.0-keep1-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon2-alpha1.0-keep2-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon3-alpha1.0-keep3-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon4-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon5-alpha1.0-keep5-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon6-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon7-alpha1.0-keep7-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-8B-vector-mode2-top50-horizon8-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    # 54 个 vote-samples8 (h<k 或 k=0) 变体 (Llama-3.2-3B + Qwen3-4B 各 27 个)
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep2-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon1-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon2-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon4-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon6-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon8-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon8-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon8-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon10-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Llama-3.2-3B-Instruct-vector-mode2-top50-horizon10-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep2-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon1-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon2-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon2-alpha1.0-keep4-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon2-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon2-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon2-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon2-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep6-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon4-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep8-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon6-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon8-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon8-alpha1.0-keep10-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon8-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon10-alpha1.0-keep0-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
    /share/project/lihao/projects/Self-Distillation/model_weight/Qwen3-4B-vector-mode2-top50-horizon10-alpha1.0-keep12-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100
)


# 8 GPU 轮转；批大小 = 8，每批跑完再起下一批
BATCH_SIZE=8

BASE_OUT_DIR="safe_test"

# === 日志目录定义与创建 ===
LOG_DIR="/share/project/lihao/projects/Self-Distillation-eval/LARF/eval_log/0427_vote54"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "🚀 开始并行安全评测 (4 个模型同时进行，打满 8 卡)"
echo "日志将统一输出至: $LOG_DIR"
echo "=================================================="

for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    GPU_IDS=$(( i % BATCH_SIZE ))

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
            --temperature "${TEMP}" \
            --benches direct harm phi harmful_behaviors

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
                --temperature "${TEMP}" \
                --benches direct harm phi harmful_behaviors
        done
        
        echo "--- Finished Evaluation for ${MODEL_NAME} ---"
    ) > "${LOG_FILE}" 2>&1 &  # 后台并行运行

    # 每跑满 BATCH_SIZE 个就 wait，避免 GPU 冲突
    if (( (i + 1) % BATCH_SIZE == 0 )); then
        echo ">>> Batch full ($(( (i+1) / BATCH_SIZE ))/3), waiting for finish..."
        wait
    fi

done

wait

echo "=================================================="
echo "🎉 所有 4 个模型的并行生成任务全部完成！"
echo "下一步：请启动 vLLM 并运行 run_llama_guard.sh 进行安全打分。"
echo "=================================================="