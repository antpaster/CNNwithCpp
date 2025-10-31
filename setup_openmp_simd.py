# python setup_openmp_simd.py clean --all
# python setup_openmp_simd.py build_ext --inplace

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='fused_add_relu_ext_openmp_simd',
    ext_modules=[
        CppExtension(
            name='fused_add_relu_ext_openmp_simd',
            sources=['fused_add_relu_openmp_simd.cpp'],
            extra_compile_args={
                'cxx': ['/Ox', '/openmp', '/arch:AVX2', '/fp:fast']
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)