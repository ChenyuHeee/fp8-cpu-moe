from __future__ import annotations

import ctypes
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "build" / "libfp8_moe.so"
FINITE_CODES = np.array([i for i in range(256) if (i & 0x7F) != 0x7F], dtype=np.uint8)
REQUIRED_K = (128, 256, 2048, 4096, 5120, 8192)


def _load_library():
    if not LIBRARY.exists():
        raise FileNotFoundError(f"{LIBRARY} does not exist; run `make` first")
    lib = ctypes.CDLL(str(LIBRARY))
    args = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.fp8_moe_gemv.argtypes = args
    lib.fp8_moe_gemv.restype = ctypes.c_int
    lib.fp8_moe_gemv_scalar.argtypes = args
    lib.fp8_moe_gemv_scalar.restype = ctypes.c_int
    lib.fp8_moe_isa_name.restype = ctypes.c_char_p
    return lib


def _call(lib, name: str, weights: np.ndarray, scales: np.ndarray, x: np.ndarray,
          threads: int = 4) -> np.ndarray:
    weights = np.ascontiguousarray(weights, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.uint8)
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty(weights.shape[0], dtype=np.float32)
    fn = getattr(lib, name)
    rc = fn(
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weights.shape[0],
        weights.shape[1],
        threads,
    )
    assert rc == 0
    return out


def _reference(weights: np.ndarray, scales: np.ndarray, x: np.ndarray) -> torch.Tensor:
    wt_u8 = torch.from_numpy(np.ascontiguousarray(weights))
    wt = wt_u8.view(torch.float8_e4m3fn).to(torch.float64)
    act = torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64)
    n, k = weights.shape
    out = torch.zeros(n, dtype=torch.float64)
    for nb, n0 in enumerate(range(0, n, 128)):
        n1 = min(n, n0 + 128)
        for kb, k0 in enumerate(range(0, k, 128)):
            k1 = min(k, k0 + 128)
            exponent = torch.tensor(int(scales[nb, kb]) - 127,
                                    dtype=torch.int32)
            scale = torch.ldexp(torch.tensor(1.0, dtype=torch.float64), exponent)
            out[n0:n1] += torch.mv(wt[n0:n1, k0:k1], act[k0:k1]) * scale
    return out


def _max_relative(got: np.ndarray, reference: torch.Tensor) -> float:
    got64 = torch.from_numpy(got).to(torch.float64)
    return float(((got64 - reference).abs() / reference.abs().clamp_min(1e-300)).max())


def _case(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    # Positive values dominate so a strict relative metric remains meaningful;
    # explicit negative extrema below still exercise the sign path.
    positive_codes = FINITE_CODES[FINITE_CODES < 0x7F]
    weights = rng.choice(positive_codes, size=(n, k)).astype(np.uint8)
    special = np.array(
        [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88,
         0x38, 0xB8, 0x7E, 0xFE],
        dtype=np.uint8,
    )
    weights.reshape(-1)[: len(special)] = special
    scale_shape = ((n + 127) // 128, (k + 127) // 128)
    scales = rng.integers(124, 131, size=scale_shape, dtype=np.uint8)
    x = rng.uniform(0.03125, 1.0, size=k).astype(np.float32)
    return weights, scales, x


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = _load_library()

    def test_required_k_and_full_finite_range(self):
        for implementation in ("fp8_moe_gemv_scalar", "fp8_moe_gemv"):
            for k in REQUIRED_K:
                with self.subTest(implementation=implementation, k=k):
                    weights, scales, x = _case(5, k, seed=1000 + k)
                    # Put the complete finite grid in every case so regressions
                    # localize cleanly rather than only being covered in aggregate.
                    weights.reshape(-1)[: FINITE_CODES.size] = FINITE_CODES
                    reference = _reference(weights, scales, x)
                    got = _call(self.library, implementation, weights, scales, x)
                    self.assertLess(_max_relative(got, reference), 1e-6)

    def test_every_finite_code_decodes_individually(self):
        weights = FINITE_CODES.reshape(-1, 1)
        scales = np.full(((weights.shape[0] + 127) // 128, 1), 127,
                         dtype=np.uint8)
        x = np.ones(1, dtype=np.float32)
        reference = _reference(weights, scales, x).to(torch.float32).numpy()
        for implementation in ("fp8_moe_gemv_scalar", "fp8_moe_gemv"):
            with self.subTest(implementation=implementation):
                got = _call(self.library, implementation, weights, scales, x,
                            threads=2)
                np.testing.assert_array_equal(got, reference)

    def test_zero_block_and_ragged_k_tail(self):
        for implementation in ("fp8_moe_gemv_scalar", "fp8_moe_gemv"):
            with self.subTest(implementation=implementation):
                # 5137 is intentionally ragged. (The required 5120 case is
                # exactly 40*128 despite the description in SPEC.md.)
                weights, scales, x = _case(7, 5137, seed=22)
                weights[:, 128:256] = 0
                reference = _reference(weights, scales, x)
                got = _call(self.library, implementation, weights, scales, x)
                self.assertLess(_max_relative(got, reference), 1e-6)

    def test_sharp_adjacent_scales_in_k_and_n(self):
        n, k = 129, 385
        weights = np.full((n, k), 0x38, dtype=np.uint8)  # exactly +1
        weights[0, :12] = np.array(
            [0x01, 0x81, 0x07, 0x87, 0x08, 0x88,
             0x7E, 0xFE, 0x00, 0x80, 0x38, 0xB8], dtype=np.uint8)
        scales = np.array([[117, 137, 125, 132],
                           [137, 117, 132, 125]], dtype=np.uint8)
        x = np.linspace(0.125, 1.0, k, dtype=np.float32)
        reference = _reference(weights, scales, x)
        for implementation in ("fp8_moe_gemv_scalar", "fp8_moe_gemv"):
            with self.subTest(implementation=implementation):
                got = _call(self.library, implementation, weights, scales, x,
                            threads=2)
                self.assertLess(_max_relative(got, reference), 1e-6)

    def test_random_ragged_matrix_across_worker_partitions(self):
        weights, scales, x = _case(385, 257, seed=881)
        reference = _reference(weights, scales, x)
        for implementation in ("fp8_moe_gemv_scalar", "fp8_moe_gemv"):
            with self.subTest(implementation=implementation):
                got = _call(self.library, implementation, weights, scales, x,
                            threads=3)
                self.assertLess(_max_relative(got, reference), 1e-6)

    def test_nan_codes_match_torch(self):
        weights = np.array([[0x7F], [0xFF]], dtype=np.uint8)
        scales = np.array([[127]], dtype=np.uint8)
        x = np.ones(1, dtype=np.float32)
        self.assertTrue(torch.isnan(_reference(weights, scales, x)).all())
        scalar = _call(self.library, "fp8_moe_gemv_scalar", weights, scales,
                       x, threads=1)
        dispatched = _call(self.library, "fp8_moe_gemv", weights, scales,
                           x, threads=1)
        self.assertTrue(np.isnan(scalar).all())
        self.assertTrue(np.isnan(dispatched).all())

    def test_runtime_dispatch_reports_supported_tier(self):
        name = self.library.fp8_moe_isa_name().decode("ascii")
        self.assertIn(name, {"scalar", "avx512f+avx512bw"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
