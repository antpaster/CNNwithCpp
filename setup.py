# python setup.py clean --all
# python setup.py build_ext --inplace


from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="fused_add_relu_ext",
    ext_modules=[
        CppExtension(
            "fused_add_relu_ext",
            ["fused_add_relu.cpp"],  # put C++ code in a file
            extra_compile_args=["-O3","-march=native","-mavx2","-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
