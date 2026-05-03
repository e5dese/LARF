1 cd E:\LLMinSafety2026\LARF
2 python eval_student_model.py --model_path ../Self-distillation/checkpoints/你的模型文件夹名

python eval_student_model.py --model_path /share/project/lihao/projects/Self-Distillation/checkpoints-reverse/checkpoint-25

python eval_refusal.py --results_path /share/project/lihao/projects/Self-Distillation-eval/LARF/safe_test/results/checkpoint-25-phi.json

#Qwen2.5-7B-Instruct-SDFT
python eval_student_model.py --model_path /share/project/lihao/projects/Self-Distillation/Qwen25checkpoints/checkpoint-25

python eval_refusal.py --results_path /share/project/lihao/projects/Self-Distillation-eval/LARF/safe_test/results/checkpoint-25-phi.json

#Qwen2.5-7B-Instruct
python eval_student_model.py --model_path /share/project/models/Qwen/Qwen2.5-7B-Instruct --output_dir safe_test/results-base

python eval_refusal.py --results_path /share/project/lihao/projects/Self-Distillation-eval/LARF/safe_test/results-base/Qwen2.5-7B-Instruct-direct.json

##Qwen3-8B-Base-no-thinking
/share/project/huggingface/models/Qwen3-8B
#一共两个json,一个是模型的输出，一个是refusal评判的结果
python eval_student_model.py --model_path /share/project/huggingface/models/Qwen3-8B --output_dir safe_test/Qwen3-8B/eval_results

python eval_refusal.py --results_path /share/project/lihao/projects/Self-Distillation-eval/LARF/safe_test/Qwen3-8B/eval_results/Qwen3-8B-phi.json

##Qwen3-8B-Base-no-thinking
/share/project/huggingface/models/Qwen3-8B
#一共两个json,一个是模型的输出，一个是refusal评判的结果
python eval_student_model.py --model_path /share/project/huggingface/models/Qwen3-8B --output_dir safe_test/Qwen3-8B/eval_results

python eval_refusal.py --results_path /share/project/lihao/projects/Self-Distillation-eval/LARF/safe_test/Qwen3-8B/eval_results/Qwen3-8B-phi.json


##Qwen3-8B-SDFT
/share/project/lihao/projects/Self-Distillation/Checkpoints/Qwen3checkpoints/final
python eval_student_model.py \
    --model_path /share/project/lihao/projects/Self-Distillation/Checkpoints/Qwen3checkpoints/final \
    --output_dir safe_test/Qwen3-8B-SDFT/eval_results

python eval_refusal.py \
     --results_path /share/project/lihao/projects/Self-Distillation-eval/LARF/safe_test/Qwen3-8B-SDFT/eval_results/final-direct.json

# #
# /share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-8B-final_distilled_model
# python eval_student_model.py \
#     --model_path /share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-8B-final_distilled_model \
#     --output_dir safe_test/Qwen3-8B-SDFT/eval_results

# python eval_refusal.py \
#      --results_path /share/project/lihao/projects/Self-Distillation-eval/LARF/safe_test/Qwen3-8B-SDFT/eval_results/Qwen3-8B-SDFT-phi.json

#system prompt V1
/share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-8B-system_prompt_V1_model
python eval_student_model.py \
    --model_path /share/project/lihao/projects/Self-Distillation/Final_checkpoints/baseline/Qwen3-8B-system_prompt_V1_model \
    --output_dir safe_test/Qwen3-8B-system_prompt/eval_results

python eval_refusal.py \
     --results_path /share/project/lihao/projects/Self-Distillation-eval/LARF/safe_test/Qwen3-8B-system_prompt/eval_results/Qwen3-8B-system_prompt_V1_model-direct.json


##4.2 补充一下实验，temp之前只跑了一次
chmod +x run_eval_temp1_completions.sh
./run_eval_temp1_completions.sh | tee eval_completions_log.txt


chmod +x run_llama_guard.sh
./run_llama_guard.sh | tee llama_guard_log_4.11.txt

chmod +x run_llama_guard.sh
./run_llama_guard.sh | tee llama_guard_log_4.13.txt
./run_llama_guard.sh | tee llama_guard_log_4.13_completions.txt

#4.16 

CUDA_VISIBLE_DEVICES=0,1 python eval_student_model.py \
    --model_path /share/project/lihao/projects/Self-Distillation/Checkpoints/Qwen3checkpoints/final \
    --output_dir safe_test/Qwen3-8B-SDFT/eval_results