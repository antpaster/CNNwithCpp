#include <torch/extension.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <omp.h>
#include <random>

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
    float mean = 0.1307f, std = 0.3081f;

    MNISTDataset(std::string images, std::string labels)
        : images_path(std::move(images)), labels_path(std::move(labels)) {
        load_data();
    }

    void enable_augmentation(bool state=true) { augment = state; }
    void set_flip_prob(float p) { flip_prob = p; }
    void set_pad(int p) { pad = p; }
    void set_normalize(bool state=true) { normalize = state; }

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

        // --- Parallel decoding ---
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

    // --- Random crop + flip augmentations ---
    torch::Tensor augment_image(const torch::Tensor &img) const {
        auto result = img;
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> flip_dist(0.0, 1.0);
        std::uniform_int_distribution<int> shift_dist(-pad, pad);

        if (augment) {
            // 1. random crop (shift)
            int dx = shift_dist(gen);
            int dy = shift_dist(gen);
            result = torch::constant_pad_nd(result, {pad, pad, pad, pad}, 0);
            result = result.slice(1, pad + dy, pad + dy + 28)
                           .slice(2, pad + dx, pad + dx + 28);

            // 2. random horizontal flip
            if (flip_dist(gen) < flip_prob)
                result = result.flip({2});
        }

        if (normalize)
            result = (result - mean) / std;

        return result;
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