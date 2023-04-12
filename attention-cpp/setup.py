from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='attention_cpp',
    ext_modules=[
        CppExtension('attention_cpp', ['attention.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
    )
