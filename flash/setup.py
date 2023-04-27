from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention_flash',
    ext_modules=[
        CUDAExtension('attention_flash', [
            'attention_flash.cpp',
            'attention_flash_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })