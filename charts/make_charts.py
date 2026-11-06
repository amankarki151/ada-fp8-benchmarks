#!/usr/bin/env python3
"""Draw the README chart from the raw L4 GEMM benchmark (standard library only).

    python3 charts/make_charts.py

Writes charts/l4_speedup_vs_marlin.svg: Marlin time / blockwise CUTLASS time (activation
quantization included) for each Qwen3-8B layer across M. Above 1.0 = faster than Marlin,
which is what vLLM runs on Ada for block-FP8 checkpoints today.
"""
import csv
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "results/blockwise-fp8-l4-2026-10-10/gemm_bench.csv"
OUT = ROOT / "charts/l4_speedup_vs_marlin.svg"

# Categorical slots 1-4 of the reference palette, light and dark steps (validated as a set).
LAYERS = [("qkv_proj", "#2a78d6", "#3987e5"), ("o_proj", "#eb6834", "#d95926"),
          ("gate_up_proj", "#1baf7a", "#199e70"), ("down_proj", "#eda100", "#c98500")]

W, H = 760, 440
L, R, T, B = 64, 132, 92, 56          # plot margins; the right margin holds the direct labels
Y_MIN, Y_MAX = 0.75, 3.0


def x_of(m):
    return L + (math.log2(m) / 12) * (W - L - R)    # M = 1 .. 4096 on a log2 axis


def y_of(v):
    return T + (Y_MAX - v) / (Y_MAX - Y_MIN) * (H - T - B)


rows = {}
for r in csv.DictReader(open(SRC)):
    rows.setdefault(r["layer"], []).append((int(r["M"]), float(r["marlin_us"]) / float(r["cutlass_blk_q_us"])))

s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
     'aria-labelledby="t d" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">']
s.append('<title id="t">Blockwise FP8 CUTLASS vs Marlin on an NVIDIA L4, Qwen3-8B layers</title>')
s.append('<desc id="d">Speedup over Marlin (Marlin time divided by blockwise CUTLASS time including activation '
         'quantization) for qkv_proj, o_proj, gate_up_proj and down_proj at M from 1 to 4096. '
         'Full numbers in results/blockwise-fp8-l4-2026-10-10/gemm_bench.csv.</desc>')
css = "".join(f".s{i}{{stroke:{lt};fill:{lt}}}" for i, (_, lt, _) in enumerate(LAYERS))
css_dark = "".join(f".s{i}{{stroke:{dk};fill:{dk}}}" for i, (_, _, dk) in enumerate(LAYERS))
s.append("<style>"
         ".bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.mu{fill:#898781}.grid{stroke:#e6e5e0}"
         ".base{stroke:#898781}.ring{stroke:#fcfcfb!important}" + css +
         "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}"
         ".grid{stroke:#2e2e2b}.ring{stroke:#1a1a19!important}" + css_dark + "}"
         "</style>")
s.append(f'<rect class="bg" width="{W}" height="{H}" rx="8"/>')
s.append(f'<text class="t1" x="{L}" y="30" font-size="17" font-weight="600">Faster than Marlin on most layers once M grows</text>')
s.append(f'<text class="t2" x="{L}" y="50" font-size="13">NVIDIA L4 · SM89 blockwise CUTLASS kernel (vLLM PR #61224, open) vs Marlin, '
         'activation quant included</text>')
lx = L
for i, (name, _, _) in enumerate(LAYERS):
    s.append(f'<circle class="s{i}" cx="{lx + 5}" cy="66" r="4"/>')
    s.append(f'<text class="t2" x="{lx + 14}" y="70" font-size="12">{name}</text>')
    lx += 14 + 8 * len(name) + 18

for v in (1.0, 1.5, 2.0, 2.5, 3.0):
    y = y_of(v)
    cls = "base" if v == 1.0 else "grid"
    dash = ' stroke-dasharray="5 4"' if v == 1.0 else ""
    s.append(f'<line class="{cls}" x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke-width="1"{dash}/>')
    s.append(f'<text class="mu" x="{L - 8}" y="{y + 4:.1f}" font-size="12" text-anchor="end">{v:.1f}x</text>')
s.append(f'<text class="mu" x="{W - R}" y="{y_of(1.0) + 15:.1f}" font-size="11" text-anchor="end">'
         'below this line: slower than Marlin</text>')
for m in (1, 4, 16, 64, 256, 1024, 4096):
    s.append(f'<text class="mu" x="{x_of(m):.1f}" y="{H - B + 20}" font-size="12" text-anchor="middle">{m}</text>')
s.append(f'<text class="t2" x="{(L + W - R) / 2:.0f}" y="{H - 12}" font-size="12" text-anchor="middle">'
         'M (tokens in the batch), log scale</text>')

# Ends of the four lines sit close together, so nudge the direct labels apart.
ends = []
for i, (name, _, _) in enumerate(LAYERS):
    pts = sorted(rows[name])
    path = " ".join(f"{'M' if j == 0 else 'L'}{x_of(m):.1f},{y_of(v):.1f}" for j, (m, v) in enumerate(pts))
    s.append(f'<path class="s{i}" d="{path}" fill="none" stroke-width="2" stroke-linejoin="round" style="fill:none"/>')
    for m, v in pts:
        s.append(f'<circle class="s{i} ring" cx="{x_of(m):.1f}" cy="{y_of(v):.1f}" r="4" stroke-width="2" '
                 f'><title>{name}, M={m}: {v:.2f}x vs Marlin</title></circle>')
    ends.append([y_of(pts[-1][1]), i, name, pts[-1][1]])
ends.sort()
for k in range(1, len(ends)):
    ends[k][0] = max(ends[k][0], ends[k - 1][0] + 16)
for y, i, name, v in ends:
    s.append(f'<circle class="s{i}" cx="{W - R + 14}" cy="{y - 4:.1f}" r="4"/>')
    s.append(f'<text class="t2" x="{W - R + 24}" y="{y:.1f}" font-size="12">{name} {v:.2f}x</text>')
s.append("</svg>")
OUT.write_text("\n".join(s) + "\n")
print("wrote", OUT.relative_to(ROOT))
