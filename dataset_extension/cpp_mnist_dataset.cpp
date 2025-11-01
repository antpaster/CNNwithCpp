#include "cpp_mnist_dataset.h"
#include <fstream>
#include <stdexcept>

static int32_t read_int(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

MNISTDataset::MNISTDataset(const std::string& images_path, const std::string& labels_path) {
    std::ifstream imgf(images_path, std::ios::binary);
    std::ifstream labf(labels_path, std::ios::binary);
    if (!imgf || !labf)
        throw std::runtime_error("Cannot open MNIST files");

    int32_t magic_img = read_int(imgf);
    int32_t num_images = read_int(imgf);
    int32_t rows = read_int(imgf);
    int32_t cols = read_int(imgf);

    int32_t magic_lab = read_int(labf);
    int32_t num_labels = read_int(labf);

    if (num_images != num_labels)
        throw std::runtime_error("Images and labels count mismatch");

    num_images_ = num_images;
    rows_ = rows;
    cols_ = cols;

    images_.resize(num_images_ * rows_ * cols_);
    labels_.resize(num_images_);

    imgf.read(reinterpret_cast<char*>(images_.data()), images_.size());
    labf.read(reinterpret_cast<char*>(labels_.data()), labels_.size());
}

int64_t MNISTDataset::size() const {
    return num_images_;
}

std::pair<torch::Tensor, int64_t> MNISTDataset::get(size_t index) {
    int64_t offset = index * rows_ * cols_;
    auto tensor = torch::from_blob(
        (void*)(images_.data() + offset),
        {1, rows_, cols_},
        torch::kUInt8).clone().to(torch::kFloat32).div_(255.0);
    int64_t label = labels_[index];
    return {tensor, label};
}
