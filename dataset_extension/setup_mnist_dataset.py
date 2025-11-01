from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='cpp_mnist_dataset_ext',
    ext_modules=[
        CppExtension(
            name='cpp_mnist_dataset_ext',
            sources=['cpp_mnist_dataset.cpp', 'setup_mnist_dataset.cpp'],
            extra_compile_args={'cxx': ['/O2']},
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
