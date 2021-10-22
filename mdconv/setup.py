from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="mdconv",
    version="1.0",
    ext_modules=[
        CUDAExtension('mdconv',[
            'md_conv_cuda.cpp',
            'md_conv_cuda_kernel.cu'
        ])
    ],
    author="wzj",
    cmdclass={
        'build_ext':BuildExtension
    }
)