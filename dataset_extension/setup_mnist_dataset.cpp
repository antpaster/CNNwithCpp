#include <torch/extension.h>
#include "cpp_mnist_dataset.h"

PYBIND11_MODULE(cpp_mnist_dataset_ext, m) {
    py::class_<MNISTDataset>(m, "MNISTDataset")
        .def(py::init<const std::string&, const std::string&>())
        .def("__len__", &MNISTDataset::size)
        .def("__getitem__", &MNISTDataset::get);
}
