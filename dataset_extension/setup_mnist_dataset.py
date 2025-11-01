from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='cpp_mnist_dataset_ext_many_workers_openmp',
    ext_modules=[
        CppExtension(
            name='cpp_mnist_dataset_ext_many_workers_openmp',
            sources=['cpp_mnist_dataset.cpp'],
            extra_compile_args={
                "cxx": ["/O2", "/openmp"] if hasattr(__import__('sys'), 'getwindowsversion') else ["-O3", "-fopenmp"],
            },
            extra_link_args=["/openmp"] if hasattr(__import__('sys'), 'getwindowsversion') else ["-fopenmp"],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
