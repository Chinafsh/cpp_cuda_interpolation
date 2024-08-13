import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')

setup(
    name='cppcuda_tutorial',        # package name for "import" in python
    version=1.0,
    author='fsh',
    author_email='shihao.feng@foxmail.com',
    description='cppcuda example',
    long_description='cppcuda example',
    ext_modules=[
        CUDAExtension(
            name='cppcuda_tutorial',
            sources=sources,
            include_dirs = include_dirs,
            extra_compile_args = {'cxx':['-O2'],
                                  'nvcc':['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)