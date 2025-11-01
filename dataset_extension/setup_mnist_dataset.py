from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='cpp_mnist_dataset_ext_many_workers',
    ext_modules=[
        CppExtension(
            name='cpp_mnist_dataset_ext_many_workers',
            sources=['cpp_mnist_dataset.cpp'],
            extra_compile_args={'cxx': ['/O2']},
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
