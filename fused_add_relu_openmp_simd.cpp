#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cmath>

// Fused add + ReLU with OpenMP and SIMD
torch::Tensor fused_add_relu(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input sizes must match");
    TORCH_CHECK(a.device().is_cpu(), "CPU only extension");

    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();
    auto out = torch::empty_like(a_contig);

    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const auto size = a_contig.numel();

    // Parallelize across elements
    // #pragma omp parallel for simd schedule(static) // issue with MSVC compiler
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        float x = a_ptr[i] + b_ptr[i];
        out_ptr[i] = x > 0.0f ? x : 0.0f;
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_relu", &fused_add_relu, "Fused Add + ReLU (OpenMP)");
}
