from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention_cuda',
    ext_modules=[
        CUDAExtension('attention_cuda', [
            'attention_cuda.cpp',
            'attention_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })