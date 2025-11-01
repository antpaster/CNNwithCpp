#include <torch/extension.h>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <tuple>
#include <string>

// helper: read big-endian 32-bit int
static uint32_t read_be_uint32(std::ifstream &f) {
    uint8_t bytes[4];
    if (!f.read(reinterpret_cast<char*>(bytes), 4)) {
        throw std::runtime_error("Failed to read uint32 from file");
    }
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) | (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

// Read images file (IDX - magic 2051)
static at::Tensor load_mnist_images(const std::string &images_path, bool normalize=true, double mean=0.1307, double std=0.3081) {
    std::ifstream f(images_path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open images file: " + images_path);

    uint32_t magic = read_be_uint32(f);
    if (magic != 0x00000803) { // 0x00000803 == 2051
        // Sometimes magic read as 2051 directly; accept both
        if (magic != 2051) {
            throw std::runtime_error("Invalid magic in images file (expected 2051). Got: " + std::to_string(magic));
        }
    }

    uint32_t num_images = read_be_uint32(f);
    uint32_t rows = read_be_uint32(f);
    uint32_t cols = read_be_uint32(f);

    const int64_t N = static_cast<int64_t>(num_images);
    const int64_t H = static_cast<int64_t>(rows);
    const int64_t W = static_cast<int64_t>(cols);
    const int64_t img_size = H * W;

    // allocate tensor [N, 1, H, W], float32
    at::Tensor images = torch::empty({N, 1, H, W}, at::kFloat);

    // temporary buffer for one image
    std::vector<uint8_t> buf(img_size);

    for (int64_t i = 0; i < N; ++i) {
        if (!f.read(reinterpret_cast<char*>(buf.data()), img_size)) {
            throw std::runtime_error("Unexpected EOF while reading image data");
        }
        // Write into tensor row-major: [1, H, W]
        float *dst = images[i].data_ptr<float>(); // pointer to start of the [1,H,W] slice
        // images is contiguous and in float order expected
        for (int64_t p = 0; p < img_size; ++p) {
            dst[p] = static_cast<float>(buf[p]) / 255.0f;
        }
    }

    if (normalize) {
        // apply (x - mean)/std in-place
        images = (images - mean) / std;
    }
    return images;
}

// Read labels file (IDX - magic 2049)
static at::Tensor load_mnist_labels(const std::string &labels_path) {
    std::ifstream f(labels_path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open labels file: " + labels_path);

    uint32_t magic = read_be_uint32(f);
    if (magic != 0x00000801) { // 0x00000801 == 2049
        if (magic != 2049) {
            throw std::runtime_error("Invalid magic in labels file (expected 2049). Got: " + std::to_string(magic));
        }
    }

    uint32_t num_labels = read_be_uint32(f);
    const int64_t N = static_cast<int64_t>(num_labels);

    at::Tensor labels = torch::empty({N}, at::kLong); // int64 labels

    for (int64_t i = 0; i < N; ++i) {
        uint8_t v;
        if (!f.read(reinterpret_cast<char*>(&v), 1)) {
            throw std::runtime_error("Unexpected EOF while reading label data");
        }
        labels[i] = static_cast<int64_t>(v);
    }
    return labels;
}

// Public API: load both and return tuple
std::tuple<at::Tensor, at::Tensor> load_mnist(const std::string &images_path, const std::string &labels_path, bool normalize=true, double mean=0.1307, double std=0.3081) {
    auto imgs = load_mnist_images(images_path, normalize, mean, std);
    auto lbls = load_mnist_labels(labels_path);
    return std::make_tuple(imgs, lbls);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_mnist", &load_mnist, "Load MNIST IDX files into tensors (images, labels)",
          py::arg("images_path"), py::arg("labels_path"), py::arg("normalize")=true, py::arg("mean")=0.1307, py::arg("std")=0.3081);
}
