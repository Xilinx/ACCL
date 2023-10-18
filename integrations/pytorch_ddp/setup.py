#!/usr/bin/env python3

# /*****************************************************************************
#  Copyright (C) 2023 Advanced Micro Devices, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# *****************************************************************************/

import os
import sys
import torch
from setuptools import setup
from torch.utils import cpp_extension
from pathlib import Path

root = Path(__file__).parent.resolve()

if not 'XILINX_XRT' in os.environ:
    print("ERROR: Xilinx XRT required for building ACCL process group")
    exit(1)

hip_enabled = 'USE_ROCM' in os.environ and int(os.environ['USE_ROCM']) != 0
cuda_enabled = 'USE_CUDA' in os.environ and int(os.environ['USE_CUDA']) != 0
accl_debug_enabled = 'ACCL_DEBUG' in os.environ \
    and int(os.environ['ACCL_DEBUG']) != 0

xrt_dir = Path(os.environ['XILINX_XRT'])

driver_dir = root / 'accl' / 'driver'
accl_utils_dir = driver_dir / 'utils' / 'accl_network_utils'
vnx_dir = root / 'accl' / 'test' / 'hardware' / 'xup_vitis_network_example' \
    / 'xrt_host_api'
roce_dir = root / 'accl' / 'test' / 'hardware' / 'HiveNet' \
    / 'network' / 'roce_v2' / 'xrt_utils'

include_dirs = [root / 'include',  driver_dir / 'xrt' / 'include',
                accl_utils_dir / 'include', xrt_dir / 'include',
                root / 'accl' / 'test' / 'model' / 'zmq',
                vnx_dir / 'include', roce_dir / 'include',
                '/usr/include/jsoncpp']
library_dirs = [driver_dir / 'xrt' / 'lib', xrt_dir / 'lib']
libraries = ['accl', 'jsoncpp', 'zmq']
sources = [root / 'src' / 'ProcessGroupACCL.cpp', vnx_dir / 'src' / 'cmac.cpp',
           vnx_dir / 'src' / 'networklayer.cpp', roce_dir / 'src' / 'cmac.cpp',
           roce_dir / 'src' / 'hivenet.cpp',
           accl_utils_dir / 'src' / 'accl_network_utils.cpp']

compile_args = ['-Wno-reorder',
                '-Wno-sign-compare',
                '-Wno-unused-but-set-variable',
                '-DACCL_HARDWARE_SUPPORT',
                '-std=c++17',
                '-g']

if hip_enabled:
    compile_args.append('-DACCL_PROCESS_GROUP_HIP_ENABLED')
if cuda_enabled:
    compile_args.append('-DACCL_PROCESS_GROUP_CUDA_ENABLED')
if accl_debug_enabled:
    compile_args.append('-DACCL_DEBUG')

if hip_enabled or cuda_enabled:
    ext = cpp_extension.CUDAExtension
else:
    ext = cpp_extension.CppExtension

module = ext(
    name="accl_process_group._c.ProcessGroupACCL",
    sources=[str(s) for s in sources],
    include_dirs=[str(i) for i in include_dirs],
    library_dirs=[str(i) for i in library_dirs],
    libraries=libraries,
    extra_compile_args=compile_args)

setup(
    name='accl_process_group',
    version='0.0.1',
    packages=['accl_process_group'],
    package_data={'accl_process_group': ['py.typed', '_c/*.pyi']},
    ext_modules=[module],
    python_requires='>=3.7',
    install_requires=['torch'],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
