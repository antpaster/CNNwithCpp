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
    torch::Tensor images; // float32 tensor [N,1,H,W], contiguous
    torch::Tensor labels; // int64 tensor [N]

    // augmentation params
    bool augment = false;
    float flip_prob = 0.5f;
    int pad = 2;
    bool normalize = false;
    float mean = 0.1307f, stdv = 0.3081f;
    float brightness_max_delta = 0.0f; // if >0, multiply by (1 +/- delta)
    int num_threads = 0; // 0 => default omp

    MNISTDataset(const std::string& images_p, const std::string& labels_p)
        : images_path(images_p), labels_path(labels_p) {
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
        if (num_images != num_labels) {
            throw std::runtime_error("Image/label count mismatch");
        }
            
        const int64_t N = (int64_t)num_images;
        const int64_t H = (int64_t)num_rows;
        const int64_t W = (int64_t)num_cols;
        const int64_t img_size = H * W;

        // read raw bytes into buffer (one big block)
        std::vector<unsigned char> raw_imgs;
        raw_imgs.resize((size_t)N * (size_t)img_size);
        if (!image_file.read(reinterpret_cast<char*>(raw_imgs.data()), (std::streamsize)(N * img_size)))
            throw std::runtime_error("Failed to read image bytes");

        std::vector<unsigned char> raw_labels;
        raw_labels.resize((size_t)N);
        if (!label_file.read(reinterpret_cast<char*>(raw_labels.data()), (std::streamsize)N))
            throw std::runtime_error("Failed to read label bytes");

        // allocate big tensors
        // images -> float32 [N,1,H,W]
        images = torch::empty({N, 1, H, W}, torch::kFloat32);
        labels = torch::empty({N}, torch::kLong);

        // fill images in parallel (convert uint8->float /255.0)
        // We'll write to contiguous memory of images tensor
        float* img_data = images.data_ptr<float>();
        int64_t stride_img = img_size; // per-sample float count

        // set threads if requested
        if (num_threads > 0) omp_set_num_threads(num_threads);
        
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            const unsigned char* src = raw_imgs.data() + (size_t)i * (size_t)img_size;
            float* dst = img_data + i * stride_img;
            for (int64_t p = 0; p < img_size; ++p) {
                dst[p] = static_cast<float>(src[p]) / 255.0f;
            }
            // set label
            labels.data_ptr<int64_t>()[i] = static_cast<int64_t>(raw_labels[i]);
        }

        // ensure contiguous & proper device
        images = images.contiguous();
        labels = labels.contiguous();

        std::cout << "MNISTContiguous: loaded " << N << " images (" << H << "x" << W << ") using " << omp_get_max_threads() << " threads\n";
    }

    // SIMD-normalize & brightness
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

    // apply brightness only
    static void simd_apply_brightness(float* data, int64_t len, float brightness) {
#if defined(__AVX2__)
        const int64_t step = 8;
        __m256 v_b = _mm256_set1_ps(brightness);
        int64_t i = 0;
        for (; i + step <= len; i += step) {
            __m256 v = _mm256_loadu_ps(data + i);
            v = _mm256_mul_ps(v, v_b);
            _mm256_storeu_ps(data + i, v);
        }
        for (; i < len; ++i) data[i] *= brightness;
#else
        for (int64_t i = 0; i < len; ++i) data[i] *= brightness;
#endif
    }

    // returns (tensor, label)
    py::tuple get(int64_t index) const {
        const int64_t N = images.size(0);
        if (index < 0 || index >= N) throw std::out_of_range("Index out of range");

        // if no augmentation -> return view (fast, zero-copy)
        if (!augment) {
            // view: [1, H, W] -> we return it (DataLoader will stack into batch)
            torch::Tensor view = images[index]; // this is a view with shape [1,H,W]
            int64_t lbl = labels.data_ptr<int64_t>()[index];
            return py::make_tuple(view, lbl);
        }

        // If augmentation enabled: make a cloned tensor to modify
        torch::Tensor sample = images[index].clone(); // float32 [1,H,W]
        int64_t H = sample.size(1);
        int64_t W = sample.size(2);

        // RNG (thread local)
        static thread_local std::mt19937 gen((std::random_device())());
        std::uniform_real_distribution<float> flip_dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> shift_dist(-pad, pad);
        std::uniform_real_distribution<float> bright_dist(1.0f - brightness_max_delta, 1.0f + brightness_max_delta);

        // pad & random crop
        if (pad > 0) {
            sample = torch::constant_pad_nd(sample, {pad, pad, pad, pad}, 0.0f);
            int dy = shift_dist(gen);
            int dx = shift_dist(gen);
            int y0 = pad + dy;
            int x0 = pad + dx;
            sample = sample.slice(1, y0, y0 + H).slice(2, x0, x0 + W);
        }

        // flip
        if (flip_dist(gen) < flip_prob) sample = sample.flip({2});

        // brightness + normalize (apply to contiguous buffer)
        if (!sample.is_contiguous()) sample = sample.contiguous();
        float* ptr = sample.data_ptr<float>();
        int64_t len = H * W;

        float brightness = 1.0f;
        if (brightness_max_delta > 0.0f) brightness = bright_dist(gen);

        if (normalize) simd_apply_brightness_normalize(ptr, len, brightness, mean, stdv);
        else simd_apply_brightness(ptr, len, brightness);

        int64_t lbl = labels.data_ptr<int64_t>()[index];
        return py::make_tuple(sample, lbl);
    }

    int64_t size() const { return (int64_t)images.size(0); }
};

// =======================================================
//                Pybind11 module definition
// =======================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MNISTDataset>(m, "MNISTDataset")
        .def(py::init<const std::string&, const std::string&>())
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
                if (t.size() != 2) {
                    throw std::runtime_error("Invalid state for MNISTDataset");
                }
                return MNISTDataset(t[0].cast<std::string>(),
                                    t[1].cast<std::string>());
            }
        ));
}