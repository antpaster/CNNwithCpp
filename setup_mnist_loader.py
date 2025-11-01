from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import sys

extra_compile_args = {"cxx": []}
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args["cxx"] = ["/O2", "/std:c++17"]
else:
    extra_compile_args["cxx"] = ["-O3", "-std=c++17"]

setup(
    name="mnist_loader_ext",
    ext_modules=[
        CppExtension(
            "mnist_loader_ext",
            ["mnist_loader.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
