# The issue's own repro (vLLM #56924, by AshrithaG), widened to every M from 16 to 64 and all six shapes,
# plus Triton for comparison. Run it in the stock vLLM venv (before) and the patched build (after).
#   python repro_issue.py | tee before.txt
import statistics
import torch
import torch.nn.functional as F
import vllm
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm

LAYERS = [("q_proj/o_proj", 2048, 2048), ("k_proj/v_proj", 2048, 1024), ("qkv_proj", 2048, 4096),
          ("gate/up_proj", 2048, 6144), ("gate_up_proj", 2048, 12288), ("down_proj", 6144, 2048)]
BATCH = [1, 8, 16, 17, 24, 32, 33, 48, 64, 65, 128]


def replay_us(fn, iters=100, windows=7):
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    times = []
    for _ in range(windows):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); graph.replay(); stop.record(); stop.synchronize()
        times.append(start.elapsed_time(stop) * 1e3 / iters)
    return statistics.median(times)


torch.manual_seed(0)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), vllm.__version__)
print("layer,M,cutlass_us,triton_us,bf16_us,max_rel_err_vs_fp64")
for name, k, n in LAYERS:
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    w_s = w.float().abs().amax(dim=1, keepdim=True) / 127
    w_q = (w.float() / w_s).round().clamp(-127, 127).to(torch.int8).t()
    for m in BATCH:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        x_q, x_s, _ = ops.scaled_int8_quant(x)
        out = ops.cutlass_scaled_mm(x_q, w_q, x_s, w_s, torch.bfloat16)
        ref = (x_q.double() * x_s.double()) @ (w_q.double() * w_s.double().t())
        err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
        c = replay_us(lambda: ops.cutlass_scaled_mm(x_q, w_q, x_s, w_s, torch.bfloat16))
        t = replay_us(lambda: triton_scaled_mm(x_q, w_q, x_s, w_s, torch.bfloat16))
        b = replay_us(lambda: F.linear(x, w))
        print(f"{name},{m},{c:.2f},{t:.2f},{b:.2f},{err:.2e}")
