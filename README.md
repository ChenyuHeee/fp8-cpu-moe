# Block-FP8 CPU MoE GEMV

A standalone AVX-512 CPU kernel for the expert GEMV over block-FP8 weights:

```text
y[N] = float8_e4m3fn_weights[N,K] @ float32_activation[K]
```

Weights stay packed at one byte each. One `ue8m0` scale byte covers each 128×128
weight block, laid out as `scales[(n / 128) * ceil(K / 128) + (k / 128)]` with
value `2^(s - 127)`. The output is float32.

## Why this exists

[FreeToken](https://github.com/FlashML-org/FreeToken) has CPU MoE kernels for
`bf16`, `nvfp4`, `ds_fp4` and `mxfp4`, but none for `fp8_block`. That gap pins
DeepSeek-V4-class checkpoints to the `offload` backend, where every expert cache
miss crosses PCIe at ~25 GB/s while the CPU cores sit idle. A CPU kernel makes
the `hybrid` backend viable, which is worth roughly 2× on this hardware.

**The work has since moved upstream.** See
[FreeToken#399](https://github.com/FlashML-org/FreeToken/pull/399), which builds
on [#36](https://github.com/FlashML-org/FreeToken/pull/36) by @gdevenyi. This
repository remains as an independent reference implementation and as the origin
of the float64 parity methodology used there.

Two assumptions in `SPEC.md` turned out to be wrong for upstream integration —
they are documented in that file rather than quietly corrected, because they are
the interesting part of the exercise. In short: FreeToken converts `ue8m0`
scales to bf16 at load time, and its CPU executor feeds bf16 activations, not
fp32. Neither is visible from the HF checkpoint's `config.json`.

## Build and test

GCC Toolset 15 is required — function-level AVX-512 target attributes emit
encodings that binutils 2.35 (the RHEL 9 system default) cannot assemble. The
Makefile rejects an older compiler rather than failing obscurely later, and it
does not use a global `-march`.

```bash
source /opt/rh/gcc-toolset-15/enable
make

source /home/hcy/tools/FreeToken/env.sh   # PyTorch 2.11, for the float64 reference
make test
```

`make test` uses the standard-library runner; pytest is not required.

## Implementation

The binary carries a portable scalar decoder and scalar GEMV. Runtime CPU
detection selects the AVX-512 path only when both AVX-512F and AVX-512BW are
present. SIMD code is isolated behind per-function
`__attribute__((target("avx512f,avx512bw")))` — no AVX-512 BF16, FP16, VNNI or
AMX dependency, and no global `-march`, so one binary runs anywhere.

Normal e4m3 values decode by shifting the sign/exponent/mantissa fields into an
fp32 bit pattern and adding the exponent-bias difference (127 − 7 = 120). The
eight subnormal magnitudes come from a register-resident permutation table, and
the NaN codes are handled by a mask. Because `ue8m0` is a pure exponent,
`VSCALEFPD` applies the block scale as an exponent add rather than a multiply.

Accumulation is fp64 throughout, in eight `__m512d` accumulators for
memory-level parallelism. This is not gold-plating — see the measurements below.
A 1 KiB software-prefetch lead nudges the streaming weight loads.

A persistent worker pool keeps thread-creation cost out of steady-state calls.
Workers pin to the caller's allowed CPU set and split work on whole 128-row
scale groups, which lets the benchmark bind one physical core per worker on one
NUMA node and first-touch the weight bank there.

The C API is in `src/fp8_moe.h`: `fp8_moe_gemv` (dispatched),
`fp8_moe_gemv_scalar` (forced portable path), `fp8_moe_isa_name`.

## Measured results

Host: dual-socket Intel Xeon Platinum 8368 (Ice Lake-SP), 2 × 38 physical cores.
`/proc/cpuinfo` reports AVX-512F/BW/VL/DQ/VNNI/VBMI/VBMI2 and **no** AVX-512
BF16, AVX-512 FP16 or AMX. An `objdump` scan of the built library finds no BF16
or AMX mnemonics, and all 165 AVX-512 instructions are confined to the single
target-attributed function.

### Numerical accuracy

Against a float64 PyTorch reference, with mixed-sign weights drawn from the full
finite e4m3 range and zero-mean (`standard_normal`) activations:

| K | `fp8_moe_gemv` | `fp8_moe_gemv_scalar` |
|---:|---:|---:|
| 128 | 4.330e-08 | 4.330e-08 |
| 256 | 3.664e-08 | 3.664e-08 |
| 2048 | 2.770e-08 | 2.770e-08 |
| 4096 | 3.491e-08 | 3.491e-08 |
| 5137 (ragged) | 3.530e-08 | 3.530e-08 |
| 8192 | 4.839e-08 | 4.839e-08 |

The two paths agree bit-for-bit. Worst case 4.8e-08 ≈ 2⁻²⁴, which is the floor
imposed by returning float32 — 17× inside the 1e-6 requirement.

### Why fp64 accumulation

The first version accumulated in fp32 and passed its tests. It was wrong. Real
MoE weights are zero-mean, and cancellation exposed the narrow accumulator:

| data | fp32 accumulator | fp64 accumulator |
|---|---:|---:|
| positive-only weights and activations | ~1e-07 | ~1e-07 |
| mixed-sign, zero-mean (realistic) | **2.4e-03** | 5.9e-08 |

The tests missed it because they filtered weights to positive codes and drew
activations from `uniform(0.03125, 1.0)`. A cheaper fix — fp32 within each
128-block, fp64 only across blocks — was also measured, and still failed at
4.0e-04: most of the error accumulates *inside* a block, so widening only the
block boundary does not help.

Note this conclusion is specific to fp32 activations. Upstream feeds bf16
activations (8 significant bits), so products of bf16 × e4m3 are ≤12 bits and
exact in fp32 — fp32 accumulation is considerably more defensible there.

### Throughput

```bash
python bench/bench.py --node 0 --threads 38 \
  --working-set-mib 1024 --warmup 5 --iterations 31
```

Three consecutive runs: median **64.40 / 65.90 / 65.92 GB/s** (range across all
samples 56.96 .. 68.56). The bar for making `hybrid` worthwhile is 50.6 GB/s —
2× the measured 25.3 GB/s PCIe gather rate.

The byte rate counts the e4m3 weight bank plus its scale bank; the activation
vector is cache-resident and excluded, consistent with an expert-weight
streaming metric. The 1024 MiB stream is 18× the detected 57 MiB socket LLC, and
`bench.py` refuses to run if the working set would fit in cache. Timing includes
worker wake-up, synchronization, dispatch, computation and output stores.

The benchmark fills weights with a constant by default, which invites the
question of whether that flatters the result. It does not — the SIMD path is
branch-free over weight values. `--random-data` draws from the full finite e4m3
range and measures 64.26 GB/s median against 65.44 for the constant fill (three
runs each, overlapping ranges).

### The fp64 accumulator makes this compute-bound

This is the caveat that matters for deployment, and it is a direct consequence
of the accuracy fix:

| physical cores | GB/s | |
|---:|---:|---|
| 8 | 15.78 | fail |
| 16 | 31.55 | fail |
| 24 | 46.08 | fail |
| 28 | 53.13 | **pass** |
| 32 | 58.22 | pass |
| 38 | 64.80 | pass |

Scaling is near-linear, i.e. the kernel is compute-bound rather than
memory-bound. Clearing the 50.6 GB/s bar needs **at least 28 of the 38 cores on
a socket**. On a shared machine where most of the socket is busy, the hybrid
backend stops paying for itself — the fp32 version was memory-bound and cleared
the bar with far fewer cores, but produced wrong answers.

## Known limits

- Only fp32 activations are supported. Upstream's CPU executor uses bf16.
- Scales are read as single-byte `ue8m0`. Upstream banks store bf16 scales.
- The NaN codes `0x7F`/`0xFF` propagate NaN here. FreeToken deliberately decodes
  them to ±480 (see `kernel/triton/e4m3_compat.py`), since checkpoints never
  store NaN weights. Matching upstream would require changing this.
- Single-socket only; there is no cross-socket work splitting.
