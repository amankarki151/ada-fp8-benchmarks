#!/usr/bin/env bash
# End-to-end comparison used for the L4 numbers in the README (Oct 10, 2026).
# Two vLLM installs: MAIN_VENV = precompiled main at the base commit (Marlin path),
# PR_VENV = the PR branch built from source with TORCH_CUDA_ARCH_LIST=8.9.
# Usage: MAIN_VENV=/path/.venv-main PR_VENV=/path/.venv bash scripts/e2e_l4.sh
set -euo pipefail
M=Qwen/Qwen3-8B-FP8
OUT=${OUT:-e2e_out}
mkdir -p "$OUT"
for label in main pr; do
    if [ "$label" = main ]; then source "$MAIN_VENV/bin/activate"; else source "$PR_VENV/bin/activate"; fi
    vllm bench throughput --model $M --max-model-len 4096 --input-len 1024 --output-len 128 --num-prompts 300 \
        > "$OUT/e2e_throughput_$label.txt" 2>&1
    for bs in 1 32; do
        vllm bench latency --model $M --max-model-len 4096 --input-len 1024 --output-len 128 --batch-size $bs \
            --num-iters 10 > "$OUT/e2e_latency_bs${bs}_$label.txt" 2>&1
    done
    lm_eval --model vllm --model_args "pretrained=$M,max_model_len=4096" --tasks gsm8k --num_fewshot 5 --limit 250 \
        > "$OUT/e2e_gsm8k_$label.txt" 2>&1
    deactivate
done
echo "done: $OUT"
