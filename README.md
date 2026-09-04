# Block-FP8 CPU MoE GEMV

This repository implements the standalone CPU kernel specified in `SPEC.md`:

```text
y[N] = float8_e4m3fn_weights[N,K] @ float32_activation[K]
```

Weights remain packed at one byte each. One `ue8m0` scale byte covers each
128×128 weight block, with the exact layout
`scales[(n / 128) * ceil(K / 128) + (k / 128)]` and value `2^(s - 127)`.
The output is float32.

## Build and test

GCC Toolset 15 is required. The Makefile deliberately rejects an older compiler,
and it does not use a global `-march`:

```bash
source /opt/rh/gcc-toolset-15/enable
make

source /home/hcy/tools/FreeToken/env.sh
make test
```

`make test` uses the standard-library test runner plus PyTorch 2.11 for the
float64 ground truth; pytest is not required. It checks:

- K = 128, 256, 2048, 4096, 5120, and 8192;
- a genuinely ragged K = 5137 tail (5120 is exactly 40×128);
- every one of the 254 finite e4m3fn encodings individually, including signed
  zeros, subnormals, negative values, and ±448;
- an all-zero 128-wide block and sharply different adjacent K- and N-block
  scales;
- e4m3fn NaN propagation for codes `0x7f` and `0xff`;
- both the explicit scalar fallback and runtime-dispatched implementation.

## Implementation

The baseline binary contains a portable scalar decoder and scalar GEMV. Runtime
CPU detection selects the AVX-512 path only when both AVX-512F and AVX-512BW are
available. SIMD code is isolated with per-function
`__attribute__((target("avx512f,avx512bw")))`; there is no AVX-512 BF16, FP16,
VNNI, or AMX dependency.

Normal e4m3 values decode by moving the sign/exponent/mantissa fields into an
fp32 bit pattern and adding the exponent-bias difference. The eight subnormal
magnitudes come from a register-resident permutation table. Four independent
accumulators expose memory-level parallelism. At the end of each 128-wide block,
`VSCALEFPS` adjusts the four accumulator exponents by `s-127`, avoiding a scale
multiply for every weight. A 1 KiB software-prefetch lead nudges the streaming
weight loads.

A persistent worker pool removes thread-creation cost from steady-state calls.
Workers are pinned to the caller's allowed CPU set and divide work on complete
128-row scale groups. This lets the benchmark bind one physical core per worker
on one NUMA node and first-touch the weight bank on that node.

The C API is declared in `src/fp8_moe.h`:

- `fp8_moe_gemv`: runtime-dispatched GEMV;
- `fp8_moe_gemv_scalar`: forced portable reference path;
- `fp8_moe_isa_name`: selected implementation name.

## Measured results

Measured on 2026-09-04 on the specified dual-socket Intel Xeon Platinum 8368
host. `/proc/cpuinfo` reported AVX-512F/BW/VL/DQ/VNNI/VBMI/VBMI2 and did not
report AVX-512 BF16, AVX-512 FP16, or AMX. An `objdump` scan of the built shared
library found no BF16 or AMX instruction mnemonics.

Numerical results from the acceptance suite:

| implementation | measured maximum relative error vs float64 PyTorch |
|---|---:|
| scalar | 5.6303e-08 |
| AVX-512F/BW | 1.4098e-07 |

Both pass the required `< 1e-6` criterion.

The throughput measurement used this command:

```bash
source /opt/rh/gcc-toolset-15/enable
make
python bench/bench.py --node 0 --threads 38 \
  --working-set-mib 1024 --warmup 5 --iterations 31
```

Measured output:

```text
ISA:             avx512f+avx512bw
NUMA node:       0
physical cores:  38 (0..37)
shape:           N=131072, K=8192
working set:     1024.06 MiB
detected LLC:    57.00 MiB
samples:         31 after 5 warmups
median latency:  10.327 ms
median:          103.98 GB/s
range:           72.50 .. 127.48 GB/s
criterion:       PASS (>= 50.6 GB/s)
```

The reported byte rate counts the e4m3 weight bank plus its `ue8m0` scale bank;
the activation vector is cache-resident and is not counted, consistent with an
expert-weight streaming bandwidth metric. The 1024.06 MiB stream is 18 times the
detected 57 MiB socket LLC. Timing includes worker wake-up, synchronization,
dispatch, computation, and output stores. The measured **103.98 GB/s passes the
50.6 GB/s throughput criterion**. The full observed range is included because
the host was shared and sample-to-sample system load was visible.

Upstream FreeToken integration is intentionally outside this repository's scope.
