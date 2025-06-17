#!/bin/bash

models=(
    "allenai/OLMo-2-0425-1B"
    # "allenai/OLMo-2-1124-7B"
    # "allenai/OLMo-2-1124-13B"
    # "allenai/OLMo-2-0325-32B"
    # "Qwen/Qwen2.5-0.5B"
    # "Qwen/Qwen2.5-1.5B"
    # "Qwen/Qwen2.5-3B"
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-14B"
    # "Qwen/Qwen2.5-32B"
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B"
    # "meta-llama/Llama-3.1-8B"
)

export WANDB_PROJECT="trace-eval"

for model in "${models[@]}"; do
    lighteval vllm \
        "model_name=$model,dtype=float16,max_model_length=4096,use_chat_template=False,trust_remote_code=True,seed=42,generation_parameters={temperature:0.0,seed:42}" \
        "community|hotpotqa|0|0" \
        --custom-tasks community_tasks/hotpotqa.py \
        --wandb \
        --save-details
done

# --load-responses-from-details-date-id="last" \

# metric="entailment_exact_hotpotqa"
# task="community|hotpotqa"
# timestamp="latest"
# csv_dir="hotpotqa/metric_results"

# python hotpotqa/print_results.py --results_dir="results/results" --metric="$metric" --task="$task" --timestamp="$timestamp" --csv_dir="$csv_dir"