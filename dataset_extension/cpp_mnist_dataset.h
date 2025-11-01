#pragma once
#include <torch/extension.h>
#include <vector>
#include <string>

class MNISTDataset {
public:
    MNISTDataset(const std::string& images_path, const std::string& labels_path);
    int64_t size() const;
    std::pair<torch::Tensor, int64_t> get(size_t index);

private:
    std::vector<uint8_t> images_;
    std::vector<uint8_t> labels_;
    int64_t num_images_, rows_, cols_;
};
