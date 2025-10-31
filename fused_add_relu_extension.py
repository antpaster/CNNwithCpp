import torch
from torch.utils.cpp_extension import load_inline
import time

src = r"""
#include <torch/extension.h>
#include <vector>
#include <immintrin.h>  // AVX intrinsics
#include <ATen/Parallel.h> // at::parallel_for

// fused op: out = relu(a + b)
//
// Requirements:
// - a, b are float32 tensors, same shape, contiguous
// - returns a new tensor

torch::Tensor fused_add_relu(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cpu(), "a must be CPU tensor");
    TORCH_CHECK(b.device().is_cpu(), "b must be CPU tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "shapes must match");
    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();

    auto out = torch::empty_like(a_contig);

    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* o_ptr = out.data_ptr<float>();
    int64_t N = a_contig.numel();

    // vector size for AVX: 8 floats per 256-bit register
    const int64_t simd_width = 8;
    const int64_t simd_step = simd_width;

    at::parallel_for(0, N, 1, [&](int64_t start, int64_t end) {
        int64_t i = start;
        // Align loop head: process as many simd blocks as possible
        for (; i + simd_step <= end; i += simd_step) {
            // load a and b
            __m256 va = _mm256_loadu_ps(a_ptr + i);
            __m256 vb = _mm256_loadu_ps(b_ptr + i);
            __m256 vsum = _mm256_add_ps(va, vb);
            // compute max(s, 0.0f)
            __m256 vzero = _mm256_setzero_ps();
            __m256 vout = _mm256_max_ps(vsum, vzero);
            _mm256_storeu_ps(o_ptr + i, vout);
        }
        // scalar remainder
        for (; i < end; ++i) {
            float s = a_ptr[i] + b_ptr[i];
            o_ptr[i] = s > 0.0f ? s : 0.0f;
        }
    });

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_relu", &fused_add_relu, "Fused add+relu (AVX2)");
}
"""

# compile with aggressive flags
module = load_inline(
    name="fused_add_relu_ext",
    cpp_sources=src,
    functions=["fused_add_relu"],
    extra_cflags=["-O3", "-march=native", "-mavx2", "-std=c++17"],
    verbose=True,
)

# small test + comparison
def test_and_bench():
    a = torch.randn(2**20, dtype=torch.float32)  # about 1M elements
    b = torch.randn_like(a)
    # warm-up
    ref = torch.nn.functional.relu(a + b)
    out = module.fused_add_relu(a, b)
    assert torch.allclose(out, ref, atol=1e-6)

    # benchmark python/pytorch reference vs native
    def timeit(fn, iterations=10):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(iterations):
            r = fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return (time.time() - t0) / iterations

    t_ref = timeit(lambda: torch.nn.functional.relu(a + b))
    t_ext = timeit(lambda: module.fused_add_relu(a, b))

    print(f"reference avg time: {t_ref*1000:.3f} ms")
    print(f"extension avg time: {t_ext*1000:.3f} ms")
    print("speedup: {:.2f}x".format(t_ref / t_ext))

if __name__ == "__main__":
    test_and_bench()
