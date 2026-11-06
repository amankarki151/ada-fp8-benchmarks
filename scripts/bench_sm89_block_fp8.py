#!/usr/bin/env python3
"""Baseline benchmark for block-FP8 GEMMs on SM89 (Ada), for vllm#58241.

Measures, on real Qwen3-8B-FP8 layer shapes (128x128 weight blocks):
  bf16          torch.mm in bf16 (reference point)
  marlin        weight-only FP8 Marlin, group 128 (what vLLM picks on sm89 today)
  triton_blk    w8a8_triton_block_scaled_mm, with and without activation quant
  cutlass_pt    per-token/per-channel CUTLASS FP8 (not block-scaled: a rough ceiling)
  cutlass_blk   the new SM89 blockwise CUTLASS kernel (only when this vLLM build
                reports cutlass_scaled_mm_supports_block_fp8 = True on the GPU)

Prints a table and writes results_sm89_block_fp8.{csv,md}.

Usage:  python bench_sm89_block_fp8.py [--m 1 16 64 ...] [--quick]
"""

import argparse
import csv
import platform
import sys
import time

import torch

import vllm
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL,
    GPTQ_MARLIN_MIN_THREAD_N,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace,
)
from vllm.scalar_type import scalar_types

BLOCK = 128
FP8 = torch.float8_e4m3fn

# Qwen3-8B: hidden 4096, 32 q heads / 8 kv heads x 128, intermediate 12288.
# (name, N, K). Verify against the model config.json before trusting the labels.
SHAPES = [
    ("qkv_proj", 6144, 4096),
    ("o_proj", 4096, 4096),
    ("gate_up_proj", 24576, 4096),
    ("down_proj", 4096, 12288),
]
DEFAULT_M = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def bench_us(fn, warmup=10, iters=100):
    """Median time in microseconds. Uses a CUDA graph when capture works
    (removes launch overhead, like vLLM's own kernel benchmarks)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    run = fn
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            fn()
        run = g.replay
        run()
        torch.cuda.synchronize()
    except Exception:
        run = fn
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def make_block_weight(N, K, device):
    """Random bf16 weight -> FP8 with one scale per 128x128 block ([N/128, K/128])."""
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.05
    nb, kb = N // BLOCK, K // BLOCK
    blocks = w.view(nb, BLOCK, kb, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-6).float()
    Bs = amax / torch.finfo(FP8).max
    q = (blocks.float() / Bs[:, None, :, None]).clamp(
        torch.finfo(FP8).min, torch.finfo(FP8).max
    )
    return w, q.view(N, K).to(FP8), Bs.contiguous()


def selected_default_kernel(N, K):
    """Which kernel vLLM's selector picks for block-FP8 on this GPU (best effort)."""
    try:
        from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            GroupShape,
            create_fp8_quant_key,
        )

        k = init_fp8_linear_kernel(
            activation_quant_key=create_fp8_quant_key(
                static=False, group_shape=GroupShape(1, BLOCK)
            ),
            weight_quant_key=create_fp8_quant_key(
                static=True, group_shape=GroupShape(BLOCK, BLOCK)
            ),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(N, K),
        )
        return type(k).__name__
    except Exception as e:  # API differs between vLLM versions
        return f"unknown ({type(e).__name__}: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", nargs="+", type=int, default=DEFAULT_M)
    ap.add_argument("--quick", action="store_true", help="fewer iterations")
    args = ap.parse_args()
    iters = 30 if args.quick else 100

    dev = "cuda"
    torch.manual_seed(0)
    cap = torch.cuda.get_device_capability()
    info = {
        "gpu": torch.cuda.get_device_name(),
        "capability": f"sm{cap[0]}{cap[1]}",
        "vllm": getattr(vllm, "__version__", "?"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "cutlass_block_fp8_supported": ops.cutlass_scaled_mm_supports_block_fp8(
            cap[0] * 10 + cap[1]
        ),
        "default_block_fp8_kernel": selected_default_kernel(4096, 4096),
    }
    has_blk = bool(info["cutlass_block_fp8_supported"])
    print("== environment ==")
    for k, v in info.items():
        print(f"  {k}: {v}")

    rows = []
    for name, N, K in SHAPES:
        w_bf16, Bq, Bs = make_block_weight(N, K, dev)

        # Marlin FP8 (weight-only, group 128), same call as benchmarks/kernels/benchmark_marlin.py
        w_half_kn = w_bf16.t().contiguous().to(torch.half)  # [K, N]
        _, marlin_q_w, marlin_s = marlin_quant_fp8_torch(w_half_kn.T, BLOCK)
        ws = MarlinWorkspace(N, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL)

        # Per-channel CUTLASS FP8 weight (ceiling; not block-scaled)
        b_pc, sb_pc = ops.scaled_fp8_quant(w_bf16, use_per_token_if_dynamic=True)
        b_pc_t = b_pc.t()  # [K, N] column-major, as cutlass_scaled_mm expects

        for M in args.m:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            x_half = x.to(torch.half)
            r = {"layer": name, "N": N, "K": K, "M": M}

            r["bf16_us"] = bench_us(lambda: x @ w_bf16.t(), iters=iters)

            r["marlin_us"] = bench_us(
                lambda: ops.marlin_gemm(
                    x_half, None, marlin_q_w, None, marlin_s, None, None, None,
                    ws.scratch, scalar_types.float8_e4m3fn, M, N, K,
                    False, False, False,
                ),
                iters=iters,
            )

            xq, xs = per_token_group_quant_fp8(x, BLOCK)
            r["triton_blk_us"] = bench_us(
                lambda: w8a8_triton_block_scaled_mm(
                    xq, Bq, xs, Bs, [BLOCK, BLOCK], torch.bfloat16
                ),
                iters=iters,
            )

            def triton_with_quant():
                q, s = per_token_group_quant_fp8(x, BLOCK)
                return w8a8_triton_block_scaled_mm(
                    q, Bq, s, Bs, [BLOCK, BLOCK], torch.bfloat16
                )

            r["triton_blk_q_us"] = bench_us(triton_with_quant, iters=iters)

            aq, sa = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            r["cutlass_pt_us"] = bench_us(
                lambda: ops.cutlass_scaled_mm(aq, b_pc_t, sa, sb_pc, torch.bfloat16),
                iters=iters,
            )

            def cutlass_with_quant():
                a, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(a, b_pc_t, s, sb_pc, torch.bfloat16)

            r["cutlass_pt_q_us"] = bench_us(cutlass_with_quant, iters=iters)

            # New SM89 blockwise kernel, called exactly like vLLM's
            # CutlassFp8BlockScaledMMKernel: column-major activation scales,
            # B as [K, N] column-major view, Bs transposed to [K/128, N/128].
            if has_blk:
                xq_cm, xs_cm = per_token_group_quant_fp8(
                    x, BLOCK, column_major_scales=True)
                r["cutlass_blk_us"] = bench_us(
                    lambda: ops.cutlass_scaled_mm(
                        xq_cm, Bq.t(), xs_cm, Bs.t(), torch.bfloat16),
                    iters=iters,
                )

                def cutlass_blk_with_quant():
                    q, sc = per_token_group_quant_fp8(
                        x, BLOCK, column_major_scales=True)
                    return ops.cutlass_scaled_mm(q, Bq.t(), sc, Bs.t(),
                                                 torch.bfloat16)

                r["cutlass_blk_q_us"] = bench_us(cutlass_blk_with_quant,
                                                 iters=iters)
            else:
                r["cutlass_blk_us"] = float("nan")
                r["cutlass_blk_q_us"] = float("nan")

            flops = 2.0 * M * N * K
            for key in list(r):
                if key.endswith("_us"):
                    r[key.replace("_us", "_tflops")] = flops / (r[key] * 1e-6) / 1e12
            rows.append(r)
            print(
                f"{name:13s} M={M:5d}  bf16 {r['bf16_us']:8.1f}  marlin {r['marlin_us']:8.1f}  "
                f"triton_blk {r['triton_blk_us']:8.1f} (+q {r['triton_blk_q_us']:8.1f})  "
                f"cutlass_pt {r['cutlass_pt_us']:8.1f} (+q {r['cutlass_pt_q_us']:8.1f})  "
                f"cutlass_blk {r['cutlass_blk_us']:8.1f} (+q {r['cutlass_blk_q_us']:8.1f})  us",
                flush=True,
            )

        # quick correctness sanity for the Triton block path vs a dequantized reference
        x = torch.randn(64, K, device=dev, dtype=torch.bfloat16)
        xq, xs = per_token_group_quant_fp8(x, BLOCK)
        out = w8a8_triton_block_scaled_mm(xq, Bq, xs, Bs, [BLOCK, BLOCK], torch.bfloat16)
        w_deq = (Bq.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
                 * Bs[:, None, :, None]).view(N, K)
        x_deq = (xq.float().view(64, K // BLOCK, BLOCK) * xs[:, :, None]).view(64, K)
        ref = x_deq @ w_deq.t()
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        print(f"  [check] {name}: triton_blk rel. error vs dequant ref = {rel:.2e}")
        if has_blk:
            for mm in (1, 64, 65, 1026):
                xc = torch.randn(mm, K, device=dev, dtype=torch.bfloat16)
                qc, sc = per_token_group_quant_fp8(xc, BLOCK, column_major_scales=True)
                outc = ops.cutlass_scaled_mm(qc, Bq.t(), sc, Bs.t(), torch.bfloat16)
                xdc = (qc.float().view(mm, K // BLOCK, BLOCK) * sc[:, :, None]).view(mm, K)
                refc = xdc @ w_deq.t()
                relc = ((outc.float() - refc).norm() / refc.norm()).item()
                ok = "OK" if relc < 1e-2 else "FAIL"
                print(f"  [check] {name}: cutlass_blk M={mm} rel. error = {relc:.2e} {ok}")

    cols = list(rows[0].keys())
    with open("results_sm89_block_fp8.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    with open("results_sm89_block_fp8.md", "w") as f:
        f.write("## Environment\n\n")
        for k, v in info.items():
            f.write(f"- {k}: `{v}`\n")
        f.write("\n## Latency (us, median, lower is better)\n\n")
        f.write("| layer | N | K | M | bf16 | marlin | triton_blk | triton_blk +quant "
                "| cutlass per-tensor | cutlass per-tensor +quant "
                "| cutlass blockwise | cutlass blockwise +quant |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['layer']} | {r['N']} | {r['K']} | {r['M']} | {r['bf16_us']:.1f} "
                f"| {r['marlin_us']:.1f} | {r['triton_blk_us']:.1f} | {r['triton_blk_q_us']:.1f} "
                f"| {r['cutlass_pt_us']:.1f} | {r['cutlass_pt_q_us']:.1f} "
                f"| {r['cutlass_blk_us']:.1f} | {r['cutlass_blk_q_us']:.1f} |\n"
            )
    print("\nwrote results_sm89_block_fp8.csv and results_sm89_block_fp8.md")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.0f}s")
