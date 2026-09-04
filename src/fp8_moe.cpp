#include "fp8_moe.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#define FP8_MOE_LINUX 1
#else
#define FP8_MOE_LINUX 0
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define FP8_MOE_X86 1
#else
#define FP8_MOE_X86 0
#endif

namespace {

constexpr int64_t kScaleBlock = 128;
constexpr int kPrefetchBytes = 1024;

inline float bits_to_float(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// OCP e4m3fn: exponent bias 7, finite-only exponent 15, and 0x7f/0xff
// reserved as NaN. Every finite value is exactly representable in float32.
inline float e4m3fn_decode(uint8_t code) {
  const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24;
  const uint32_t magnitude = code & 0x7fu;
  const uint32_t exponent = magnitude >> 3;
  const uint32_t mantissa = magnitude & 7u;
  if (magnitude == 0x7fu) return bits_to_float(sign | 0x7fc00000u);
  if (exponent == 0) {
    if (mantissa == 0) return bits_to_float(sign);
    return std::copysign(std::ldexp(static_cast<float>(mantissa), -9),
                         sign ? -1.0f : 1.0f);
  }
  // Move the e4m3 exponent/mantissa together and add the bias difference.
  return bits_to_float(sign | ((magnitude << 20) + (120u << 23)));
}

using RowKernel = float (*)(const uint8_t*, const uint8_t*, const float*,
                            int64_t);

float gemv_row_scalar(const uint8_t* weights, const uint8_t* scales,
                      const float* x, int64_t k) {
  double total = 0.0;
  const int64_t blocks = (k + kScaleBlock - 1) / kScaleBlock;
  for (int64_t block = 0; block < blocks; ++block) {
    const int64_t begin = block * kScaleBlock;
    const int64_t end = std::min(k, begin + kScaleBlock);
    double block_sum = 0.0;
    for (int64_t i = begin; i < end; ++i) {
      block_sum += static_cast<double>(e4m3fn_decode(weights[i])) *
                   static_cast<double>(x[i]);
    }
    total += std::scalbn(block_sum, static_cast<int>(scales[block]) - 127);
  }
  return static_cast<float>(total);
}

#if FP8_MOE_X86

__attribute__((target("avx512f,avx512bw")))
inline __m512 decode_e4m3fn_16(__m128i packed) {
  const __m512i codes = _mm512_cvtepu8_epi32(packed);
  const __m512i magnitude =
      _mm512_and_si512(codes, _mm512_set1_epi32(0x7f));
  const __m512i sign = _mm512_slli_epi32(
      _mm512_and_si512(codes, _mm512_set1_epi32(0x80)), 24);

  // Normal e4m3 values become fp32 by moving eeee|mmm up twenty bits and
  // adding the 127-7 exponent-bias difference. No floating multiply is used.
  __m512i normal_bits = _mm512_add_epi32(
      _mm512_slli_epi32(magnitude, 20), _mm512_set1_epi32(120 << 23));
  normal_bits = _mm512_or_si512(normal_bits, sign);
  __m512 decoded = _mm512_castsi512_ps(normal_bits);

  // e4m3 subnormals are the tiny eight-entry grid m*2^-9. A register
  // permutation handles them exactly and is used only in masked lanes.
  const __m512 subnormal_lut = _mm512_setr_ps(
      0.0f, 0x1p-9f, 0x1p-8f, 0x1.8p-8f,
      0x1p-7f, 0x1.4p-7f, 0x1.8p-7f, 0x1.cp-7f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  const __m512i mantissa =
      _mm512_and_si512(magnitude, _mm512_set1_epi32(7));
  __m512 subnormal = _mm512_permutexvar_ps(mantissa, subnormal_lut);
  subnormal = _mm512_castsi512_ps(
      _mm512_xor_si512(_mm512_castps_si512(subnormal), sign));
  const __mmask16 is_subnormal = _mm512_cmpeq_epi32_mask(
      _mm512_and_si512(magnitude, _mm512_set1_epi32(0x78)),
      _mm512_setzero_si512());
  decoded = _mm512_mask_mov_ps(decoded, is_subnormal, subnormal);

  // torch.float8_e4m3fn treats the two all-one magnitude codes as NaN.
  const __mmask16 is_nan =
      _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(0x7f));
  const __m512 signed_nan = _mm512_castsi512_ps(
      _mm512_or_si512(sign, _mm512_set1_epi32(0x7fc00000u)));
  return _mm512_mask_mov_ps(decoded, is_nan, signed_nan);
}

__attribute__((target("avx512f,avx512bw")))
float gemv_row_avx512(const uint8_t* weights, const uint8_t* scales,
                      const float* x, int64_t k) {
  __m512 total0 = _mm512_setzero_ps();
  __m512 total1 = _mm512_setzero_ps();
  __m512 total2 = _mm512_setzero_ps();
  __m512 total3 = _mm512_setzero_ps();
  float scalar_tail = 0.0f;

  const int64_t blocks = (k + kScaleBlock - 1) / kScaleBlock;
  for (int64_t block = 0; block < blocks; ++block) {
    const int64_t begin = block * kScaleBlock;
    const int64_t end = std::min(k, begin + kScaleBlock);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    int64_t i = begin;
    for (; i + 64 <= end; i += 64) {
      _mm_prefetch(reinterpret_cast<const char*>(weights + i) +
                       kPrefetchBytes,
                   _MM_HINT_T0);
      const __m512 w0 = decode_e4m3fn_16(_mm_loadu_si128(
          reinterpret_cast<const __m128i*>(weights + i)));
      const __m512 w1 = decode_e4m3fn_16(_mm_loadu_si128(
          reinterpret_cast<const __m128i*>(weights + i + 16)));
      const __m512 w2 = decode_e4m3fn_16(_mm_loadu_si128(
          reinterpret_cast<const __m128i*>(weights + i + 32)));
      const __m512 w3 = decode_e4m3fn_16(_mm_loadu_si128(
          reinterpret_cast<const __m128i*>(weights + i + 48)));
      acc0 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x + i), acc0);
      acc1 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x + i + 16), acc1);
      acc2 = _mm512_fmadd_ps(w2, _mm512_loadu_ps(x + i + 32), acc2);
      acc3 = _mm512_fmadd_ps(w3, _mm512_loadu_ps(x + i + 48), acc3);
    }
    int lane = 0;
    for (; i + 16 <= end; i += 16, ++lane) {
      const __m512 w = decode_e4m3fn_16(_mm_loadu_si128(
          reinterpret_cast<const __m128i*>(weights + i)));
      const __m512 xv = _mm512_loadu_ps(x + i);
      if ((lane & 3) == 0)
        acc0 = _mm512_fmadd_ps(w, xv, acc0);
      else if ((lane & 3) == 1)
        acc1 = _mm512_fmadd_ps(w, xv, acc1);
      else if ((lane & 3) == 2)
        acc2 = _mm512_fmadd_ps(w, xv, acc2);
      else
        acc3 = _mm512_fmadd_ps(w, xv, acc3);
    }

    // VSCALEFPS is an exponent adjustment. The shared ue8m0 scale is applied
    // to four partial sums rather than multiplied into all 128 weights.
    const __m512 scale_exp =
        _mm512_set1_ps(static_cast<float>(static_cast<int>(scales[block]) - 127));
    total0 = _mm512_add_ps(total0, _mm512_scalef_ps(acc0, scale_exp));
    total1 = _mm512_add_ps(total1, _mm512_scalef_ps(acc1, scale_exp));
    total2 = _mm512_add_ps(total2, _mm512_scalef_ps(acc2, scale_exp));
    total3 = _mm512_add_ps(total3, _mm512_scalef_ps(acc3, scale_exp));

    float tail = 0.0f;
    for (; i < end; ++i) tail += e4m3fn_decode(weights[i]) * x[i];
    scalar_tail +=
        std::scalbn(tail, static_cast<int>(scales[block]) - 127);
  }

  const __m512 pair01 = _mm512_add_ps(total0, total1);
  const __m512 pair23 = _mm512_add_ps(total2, total3);
  return _mm512_reduce_add_ps(_mm512_add_ps(pair01, pair23)) + scalar_tail;
}

#endif  // FP8_MOE_X86

struct DispatchChoice {
  RowKernel kernel;
  const char* name;
};

DispatchChoice select_kernel() {
#if FP8_MOE_X86
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512bw")) {
    return {gemv_row_avx512, "avx512f+avx512bw"};
  }
#endif
  return {gemv_row_scalar, "scalar"};
}

const DispatchChoice& dispatch_choice() {
  static const DispatchChoice choice = select_kernel();
  return choice;
}

std::vector<int> allowed_cpus() {
  std::vector<int> cpus;
#if FP8_MOE_LINUX
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &set)) cpus.push_back(cpu);
    }
  }
#endif
  return cpus;
}

void pin_this_thread(int cpu) {
#if FP8_MOE_LINUX
  if (cpu < 0) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cpu;
#endif
}

class WorkerPool {
 public:
  explicit WorkerPool(int threads)
      : threads_(threads), cpus_(allowed_cpus()), workers_() {
    workers_.reserve(static_cast<size_t>(threads_));
    for (int tid = 0; tid < threads_; ++tid) {
      workers_.emplace_back([this, tid] { worker_loop(tid); });
    }
  }

  ~WorkerPool() {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      stop_ = true;
      ++generation_;
    }
    work_cv_.notify_all();
    for (std::thread& worker : workers_) worker.join();
  }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  void run(RowKernel kernel, const uint8_t* weights, const uint8_t* scales,
           const float* x, float* y, int64_t n, int64_t k) {
    std::unique_lock<std::mutex> call_lock(call_mutex_);
    {
      std::lock_guard<std::mutex> state_lock(state_mutex_);
      kernel_ = kernel;
      weights_ = weights;
      scales_ = scales;
      x_ = x;
      y_ = y;
      n_ = n;
      k_ = k;
      finished_ = 0;
      ++generation_;
    }
    work_cv_.notify_all();
    done_cv_.wait(call_lock, [this] {
      return finished_.load(std::memory_order_acquire) == threads_;
    });
  }

 private:
  void worker_loop(int tid) {
    const int cpu = tid < static_cast<int>(cpus_.size()) ? cpus_[tid] : -1;
    pin_this_thread(cpu);
    uint64_t seen_generation = 0;
    for (;;) {
      RowKernel kernel;
      const uint8_t* weights;
      const uint8_t* scales;
      const float* x;
      float* y;
      int64_t n;
      int64_t k;
      uint64_t generation;
      {
        std::unique_lock<std::mutex> lock(state_mutex_);
        work_cv_.wait(lock, [this, seen_generation] {
          return stop_ || generation_ != seen_generation;
        });
        if (stop_) return;
        generation = generation_;
        kernel = kernel_;
        weights = weights_;
        scales = scales_;
        x = x_;
        y = y_;
        n = n_;
        k = k_;
      }

      // Keep a 128-row scale group wholly owned by one worker. Besides avoiding
      // false sharing in y, this makes the scale stream local to that worker.
      const int64_t row_blocks = (n + kScaleBlock - 1) / kScaleBlock;
      const int64_t block_begin = row_blocks * tid / threads_;
      const int64_t block_end = row_blocks * (tid + 1) / threads_;
      const int64_t row_begin = block_begin * kScaleBlock;
      const int64_t row_end = std::min(n, block_end * kScaleBlock);
      const int64_t scale_stride = (k + kScaleBlock - 1) / kScaleBlock;
      for (int64_t row = row_begin; row < row_end; ++row) {
        y[row] = kernel(weights + static_cast<size_t>(row) * k,
                        scales + (row / kScaleBlock) * scale_stride, x, k);
      }

      seen_generation = generation;
      if (finished_.fetch_add(1, std::memory_order_acq_rel) + 1 == threads_) {
        done_cv_.notify_one();
      }
    }
  }

  const int threads_;
  const std::vector<int> cpus_;
  std::vector<std::thread> workers_;
  std::mutex call_mutex_;
  std::mutex state_mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  bool stop_ = false;
  uint64_t generation_ = 0;
  std::atomic<int> finished_{0};

  RowKernel kernel_ = nullptr;
  const uint8_t* weights_ = nullptr;
  const uint8_t* scales_ = nullptr;
  const float* x_ = nullptr;
  float* y_ = nullptr;
  int64_t n_ = 0;
  int64_t k_ = 0;
};

std::mutex pool_mutex;
std::unique_ptr<WorkerPool> pool;
int pool_threads = 0;

int default_thread_count() {
  const unsigned logical = std::thread::hardware_concurrency();
  return logical > 1 ? static_cast<int>(logical / 2) : 1;
}

int run_gemv(RowKernel kernel, const uint8_t* weights, const uint8_t* scales,
             const float* x, float* y, int64_t n, int64_t k,
             int num_threads) {
  if (weights == nullptr || scales == nullptr || x == nullptr || y == nullptr)
    return -1;
  if (n <= 0 || k <= 0) return -2;
  if (num_threads < 0) return -3;
  int threads = num_threads == 0 ? default_thread_count() : num_threads;
  const int64_t row_blocks = (n + kScaleBlock - 1) / kScaleBlock;
  threads = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(threads, row_blocks)));

  if (threads == 1) {
    const int64_t scale_stride = (k + kScaleBlock - 1) / kScaleBlock;
    for (int64_t row = 0; row < n; ++row) {
      y[row] = kernel(weights + static_cast<size_t>(row) * k,
                      scales + (row / kScaleBlock) * scale_stride, x, k);
    }
    return 0;
  }

  std::lock_guard<std::mutex> lock(pool_mutex);
  if (!pool || pool_threads != threads) {
    pool.reset();
    pool = std::make_unique<WorkerPool>(threads);
    pool_threads = threads;
  }
  pool->run(kernel, weights, scales, x, y, n, k);
  return 0;
}

}  // namespace

extern "C" int fp8_moe_gemv(const uint8_t* weights, const uint8_t* scales,
                              const float* x, float* y, int64_t n, int64_t k,
                              int num_threads) {
  return run_gemv(dispatch_choice().kernel, weights, scales, x, y, n, k,
                  num_threads);
}

extern "C" int fp8_moe_gemv_scalar(const uint8_t* weights,
                                     const uint8_t* scales, const float* x,
                                     float* y, int64_t n, int64_t k,
                                     int num_threads) {
  return run_gemv(gemv_row_scalar, weights, scales, x, y, n, k, num_threads);
}

extern "C" const char* fp8_moe_isa_name(void) {
  return dispatch_choice().name;
}
