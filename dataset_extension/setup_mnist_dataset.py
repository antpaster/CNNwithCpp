# python setup_mnist_dataset.py clean --all
# python setup_mnist_dataset.py build_ext --inplace

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import sys

extra_compile_args = {"cxx": []}
extra_link_args = []

if sys.platform == "win32":
    # MSVC flags
    extra_compile_args["cxx"] = ["/O2", "/openmp", "/std:c++17", "/arch:AVX2"]
else:
    extra_compile_args["cxx"] = ["-O3", "-std=c++17", "-fopenmp", "-mavx2", "-ffast-math"]
    extra_link_args = ["-fopenmp"]

setup(
    name="cpp_mnist_contiguous_ext",
    ext_modules=[
        CppExtension(
            name="cpp_mnist_contiguous_ext",
            sources=["cpp_mnist_dataset.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
