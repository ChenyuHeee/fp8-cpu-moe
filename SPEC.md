# Block-FP8 CPU MoE Kernel — Specification

## Goal

Write an AVX-512 CPU kernel that computes an **expert GEMV over block-FP8 weights**,
fast enough to make FreeToken's `hybrid` MoE backend viable for DeepSeek-V4-Flash.

Today FreeToken has CPU MoE kernels for `bf16`, `nvfp4`, and `ds_fp4`, but **none for
`fp8_block`**. That single gap forces DeepSeek-V4-Flash (whose experts are block-FP8)
onto the `offload` backend, where every cache miss crosses PCIe at 25 GB/s instead of
being computed on the CPU at ~70 GB/s. 76 idle physical cores go unused.

This repo builds and validates the missing kernel standalone. Upstream integration is
out of scope here.

## Weight format (must match exactly)

DeepSeek-V4-Flash-0731 `config.json`:

```json
"expert_dtype": "fp8",
"quantization_config": {
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "scale_fmt": "ue8m0",
  "weight_block_size": [128, 128]
}
```

- **Weights**: `float8_e4m3fn` (OCP "fn" variant: finite only, no inf, exponent bias 7,
  max normal 448). One byte per weight.
- **Scales**: one scale per **128×128 weight block**. Stored as `ue8m0` — an unsigned
  8-bit *exponent only*, no mantissa, no sign. The effective scale is `2^(s - 127)`.
  Because it is a pure power of two, dequantisation is an **exponent add**, not a
  multiply. Exploit this.
- **Dequant**: `w_real = e4m3_decode(w_byte) * 2^(scale_byte - 127)`
- **Activations**: `float32`, dense, no quantisation. (The `activation_scheme:
  "dynamic"` in the config refers to the GPU path; for this kernel activations arrive
  as fp32.)

Scale indexing for a `[N, K]` weight matrix: the scale for element `(n, k)` lives at
`scales[(n / 128) * ceil(K / 128) + (k / 128)]`. Row-major, and `K` is **not**
guaranteed to be a multiple of 128 — handle the ragged last block.

## Reference semantics (ground truth)

`tests/` must check against a PyTorch reference that dequantises with
`torch.float8_e4m3fn` and computes the GEMV in float64, then compares.

Read these for the exact conventions before writing anything:

- `/home/hcy/tools/FreeToken/python/freetoken/kernel/triton/fp8_blockscale_moe.py`
  — the GPU kernel this must agree with.
- `/home/hcy/tools/FreeToken/python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`
  — existing CPU kernels. `dot_avx512f` (bf16) is the closest structural analogue:
  4 independent accumulators for memory-level parallelism, `_mm_prefetch` nudge.
  `e4m3_decode` at ~line 258 already exists (used for NVFP4 block scales).

## Target hardware (measure on this, not on assumptions)

```
Intel Xeon Platinum 8368 @ 2.40GHz, 2 sockets x 38 physical cores (76 total, 152 HT)
NUMA: 2 nodes, 257 GB each
```

ISA available — **verify with /proc/cpuinfo, do not assume**:

```
✓ avx512f  avx512bw  avx512vl  avx512dq  avx512_vnni  avx512vbmi  avx512_vbmi2
✗ avx512_bf16   ✗ avx512_fp16   ✗ amx_bf16   ✗ amx_int8
```

**There is no AVX512-BF16 on this machine.** FreeToken's bf16 kernel therefore runs its
`avx512f` tier (widen bf16→fp32, then FMA) and measures 69.92 GB/s. An fp8 kernel that
decodes to fp32 and uses the same FMA path is on equal footing — it is not handicapped.

Measured ceilings on this box (from `ft bench bw`):

| | GB/s |
|---|---|
| CPU STREAM read (DRAM ceiling) | 154.86 |
| CPU-MoE bf16 kernel, avx512f tier | 69.92 |
| PCIe linear H2D | 25.28 |

## Acceptance criteria

All three must pass. These are the definition of done.

1. **Numerical parity.** Max relative error vs the float64 PyTorch reference
   `< 1e-6` across at least: K ∈ {128, 256, 2048, 4096, 5120 (non-multiple-of-128
   tail), 8192}, and weight values spanning the full e4m3 range including
   denormals, zeros, negatives, and the max normal 448. Include a test where a whole
   128-block is zero and one where scales differ sharply between adjacent blocks.

2. **Throughput.** ≥ **50.6 GB/s** single-socket on a working set that exceeds LLC
   (this is the 2.0× PCIe threshold that makes `hybrid` worthwhile). Report the number
   honestly; if it lands under, say so and explain where the time goes rather than
   tuning the benchmark to pass.

3. **Correct ISA discipline.** Use per-function `__attribute__((target(...)))` plus a
   runtime dispatch to a scalar fallback — **never** a global `-march`. This mirrors
   how `cpu_moe_ext.cpp` does it, and it is what lets one binary run on a machine
   without AVX-512. A scalar reference implementation must exist and be tested.

## Build environment (non-obvious, will waste hours if missed)

```bash
source /opt/rh/gcc-toolset-15/enable   # REQUIRED: system gcc 11 / binutils 2.35
                                       # cannot assemble the VNNI/AVX-512 encodings
                                       # emitted by function-level target attributes
```

A Python venv with torch is available for the reference implementation:

```bash
source /home/hcy/tools/FreeToken/env.sh   # also sets CUDA_HOME; torch 2.11 cu130
```

Do not modify anything under `/home/hcy/tools/FreeToken/` — it is a working install
serving a live deployment. Read from it freely; write only inside this repo.

## Deliverables

```
src/fp8_moe.cpp        kernel + scalar fallback + runtime dispatch
src/fp8_moe.h
tests/test_parity.py   numerical parity vs float64 PyTorch reference
bench/bench.py         throughput measurement, prints GB/s
Makefile               builds the extension
README.md              what it does, how to build, MEASURED results
```

`README.md` must report **measured** numbers, not projected ones, and must state
plainly whether criterion 2 passed.

## Design notes (suggestions, not requirements)

- e4m3 → fp32 is a bit-manipulation, not necessarily a table lookup: e4m3 is
  `s|eeee|mmm`, fp32 is `s|eeeeeeee|mmmmmmmmmmmmmmmmmmmmmmm`. A shift plus an exponent
  rebias (127−7 = 120) handles normals in a couple of instructions. Denormals and the
  e4m3 NaN encoding (0xFF / 0x7F) need care — decide and document how you handle them,
  and make the tests cover it. A 256-entry LUT is also legitimate if it measures
  faster; justify the choice with a measurement.
- The `ue8m0` scale is a pure exponent. Folding it into the fp32 exponent field costs
  an integer add and avoids a multiply per block.
- The bf16 kernel reaches only 45% of the STREAM ceiling, so there is headroom: a
  decode step that costs some ALU work can likely hide behind the memory stream.
  Verify this rather than assuming it.
- Multiple independent accumulators matter more than clever decoding — the operation
  is memory-bound. See the comment above `dot_avx512f`.

## Out of scope

- Integrating into FreeToken (no edits to that tree)
- GPU kernels
- The `ds_fp4` / `mxfp4` formats
- Multi-threading beyond what the benchmark needs to saturate one socket
