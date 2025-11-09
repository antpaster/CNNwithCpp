#include <torch/extension.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <omp.h>
#include <random>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace py = pybind11;

// =======================================================
//                MNISTDataset class
// =======================================================

struct MNISTDataset {
    std::string images_path;
    std::string labels_path;
    std::vector<torch::Tensor> images;
    std::vector<int64_t> labels;

    // augmentation params
    bool augment = false;
    float flip_prob = 0.5f;
    int pad = 2;
    bool normalize = false;
    float mean = 0.1307f, stdv = 0.3081f;
    float brightness_max_delta = 0.0f; // if >0, multiply by (1 +/- delta)
    int num_threads = 0; // 0 => default omp

    MNISTDataset(std::string images, std::string labels)
        : images_path(std::move(images)), labels_path(std::move(labels)) {
        load_data();
    }

    void set_num_threads(int t) {
        num_threads = t;
        if (t > 0) omp_set_num_threads(t);
    }

    void enable_augmentation(bool state=true) { augment = state; }
    void set_flip_prob(float p) { flip_prob = p; }
    void set_pad(int p) { pad = p; }
    void set_normalize(bool state=true) { normalize = state; }
    void set_mean_std(float m, float s) { mean = m; stdv = s; }
    void set_brightness_max_delta(float d) { brightness_max_delta = d; }

    // -------------------------------------------------------
    // Helper to read a big-endian 32-bit integer from file
    // -------------------------------------------------------
    static uint32_t read_int(std::ifstream &f) {
        unsigned char bytes[4];
        f.read(reinterpret_cast<char *>(bytes), 4);
        return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
               (uint32_t(bytes[2]) << 8) | (uint32_t(bytes[3]));
    }

    // -------------------------------------------------------
    // Load MNIST .idx3 and .idx1 files into memory
    // -------------------------------------------------------
    void load_data() {
        // --- images ---
        std::ifstream image_file(images_path, std::ios::binary);
        std::ifstream label_file(labels_path, std::ios::binary);
        if (!image_file.is_open() || !label_file.is_open()) {
            throw std::runtime_error("Cannot open MNIST files.");
        }

        uint32_t magic_images = read_int(image_file);
        uint32_t num_images = read_int(image_file);
        uint32_t num_rows = read_int(image_file);
        uint32_t num_cols = read_int(image_file);

        uint32_t magic_labels = read_int(label_file);
        uint32_t num_labels = read_int(label_file);
        if (num_images != num_labels)
            throw std::runtime_error("Image/label count mismatch");

        const size_t image_size = num_rows * num_cols;
        std::vector<unsigned char> img_bytes(num_images * image_size);
        std::vector<unsigned char> label_bytes(num_labels);

        // --- read all bytes at once ---
        image_file.read(reinterpret_cast<char *>(img_bytes.data()), num_images * image_size);
        label_file.read(reinterpret_cast<char *>(label_bytes.data()), num_labels);

        images.resize(num_images);
        labels.resize(num_images);

        // Parallel decode to float tensors
        #pragma omp parallel
        {
            if (num_threads > 0) omp_set_num_threads(num_threads);
        }
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(num_images); ++i) {
            unsigned char *img_ptr = img_bytes.data() + i * image_size;
            torch::Tensor img = torch::from_blob(
                img_ptr,
                {1, (long)num_rows, (long)num_cols},
                torch::kUInt8
            ).clone().to(torch::kFloat32).div_(255.0);

            images[i] = std::move(img);
            labels[i] = static_cast<int64_t>(label_bytes[i]);
        }

        std::cout << "Loaded " << num_images << " images using "
                  << omp_get_max_threads() << " threads.\n";
    }

    // SIMD-normalize & brightness multiply in-place on contiguous float buffer
    static void simd_apply_brightness_normalize(float* data, int64_t len, float brightness, float mean, float stdv) {
#if defined(__AVX2__)
        const int64_t simd_step = 8; // 8 floats per __m256
        __m256 v_brightness = _mm256_set1_ps(brightness);
        __m256 v_mean = _mm256_set1_ps(mean);
        __m256 v_std = _mm256_set1_ps(stdv);

        int64_t i = 0;
        for (; i + simd_step <= len; i += simd_step) {
            __m256 v = _mm256_loadu_ps(data + i);        // load 8 floats
            v = _mm256_mul_ps(v, v_brightness);         // v *= brightness
            v = _mm256_sub_ps(v, v_mean);               // v -= mean
            v = _mm256_div_ps(v, v_std);                // v /= std
            _mm256_storeu_ps(data + i, v);              // store
        }
        // remaining
        for (; i < len; ++i) {
            float v = data[i];
            v = v * brightness;
            v = (v - mean) / stdv;
            data[i] = v;
        }
#else
        // scalar fallback
        for (int64_t i = 0; i < len; ++i) {
            float v = data[i];
            v = v * brightness;
            v = (v - mean) / stdv;
            data[i] = v;
        }
#endif
    }

    // augment_image: do crop/pad/flip and then SIMD-normalize+brightness
    torch::Tensor augment_image(const torch::Tensor &img) const {
        // img shape: [1, H, W], float32 in [0,1]
        auto out = img;
        int H = img.size(1);
        int W = img.size(2);

        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> flip_dist(0.0, 1.0);
        std::uniform_int_distribution<int> shift_dist(-pad, pad);
        std::uniform_real_distribution<float> bright_dist(1.0f - brightness_max_delta, 1.0f + brightness_max_delta);

        // Crop with pad: perform padding (constant 0) then slice (safe torch ops)
        if (augment && pad > 0) {
            out = torch::constant_pad_nd(out, {pad, pad, pad, pad}, 0.0f);
            int dy = shift_dist(gen);
            int dx = shift_dist(gen);
            int y0 = pad + dy;
            int x0 = pad + dx;
            out = out.slice(1, y0, y0 + H).slice(2, x0, x0 + W);
        }

        // Flip
        if (augment && flip_dist(gen) < flip_prob) {
            out = out.flip({2}); // flip width dimension
        }

        // Now apply brightness + normalization on contiguous buffer
        if (!out.is_contiguous()) out = out.contiguous();
        float* ptr = out.data_ptr<float>();
        int64_t len = (int64_t)H * W; // channel=1 omitted

        float brightness = 1.0f;
        if (brightness_max_delta > 0.0f) brightness = bright_dist(gen);

        if (normalize) {
            simd_apply_brightness_normalize(ptr, len, brightness, mean, stdv);
        } else {
#if defined(__AVX2__)
            // only brightness multiply
            const int64_t simd_step = 8;
            __m256 v_b = _mm256_set1_ps(brightness);
            int64_t i = 0;
            for (; i + simd_step <= len; i += simd_step) {
                __m256 v = _mm256_loadu_ps(ptr + i);
                v = _mm256_mul_ps(v, v_b);
                _mm256_storeu_ps(ptr + i, v);
            }
            for (; i < len; ++i) ptr[i] *= brightness;
#else
            for (int64_t i = 0; i < len; ++i) ptr[i] *= brightness;
#endif
        }

        return out;
    }

    torch::Tensor get_image(int64_t index) const { return images[index]; }

    int64_t get_label(int64_t index) const { return labels[index]; }

    py::tuple get(int64_t index) const {
        torch::Tensor img = augment_image(images[index]);
        return py::make_tuple(img, labels[index]);
    }

    int64_t size() const { return (int64_t)images.size(); }
};

// =======================================================
//                Pybind11 module definition
// =======================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MNISTDataset>(m, "MNISTDataset")
        .def(py::init<std::string, std::string>())
        .def("__len__", &MNISTDataset::size)
        .def("__getitem__", &MNISTDataset::get)
        .def("enable_augmentation", &MNISTDataset::enable_augmentation)
        .def("set_flip_prob", &MNISTDataset::set_flip_prob)
        .def("set_pad", &MNISTDataset::set_pad)
        .def("set_normalize", &MNISTDataset::set_normalize)
        .def("set_mean_std", &MNISTDataset::set_mean_std)
        .def("set_brightness_max_delta", &MNISTDataset::set_brightness_max_delta)
        .def("set_num_threads", &MNISTDataset::set_num_threads)

        // ✅ Pickle support for Windows multiprocessing
        .def(py::pickle(
            [](const MNISTDataset &d) { // __getstate__
                return py::make_tuple(d.images_path, d.labels_path);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state for MNISTDataset");
                return MNISTDataset(t[0].cast<std::string>(),
                                    t[1].cast<std::string>());
            }
        ));
}