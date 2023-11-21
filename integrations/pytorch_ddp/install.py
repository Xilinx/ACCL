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

from shutil import rmtree
import subprocess
import sys
import os
from pathlib import Path

_CURRENT_PYTORCH = '2.1'
CURRENT_PYTORCH_VERSION = f'v{_CURRENT_PYTORCH}'
CURRENT_PYTORCH_BRANCH = f'release/{_CURRENT_PYTORCH}'

CURRENT_ACCL_BRANCH = f'dev'

root = Path(__file__).parent.resolve()
accl_repo = root / 'accl'
accl_driver_path = accl_repo / 'driver' / 'xrt'
accl_driver = accl_driver_path / 'lib' / 'libaccl.so'
torch_dir = root / 'torch'

if sys.executable:
    python = sys.executable
else:
    python = 'python3'


def test_packages():
    packages = {
        'torch': False,
        'accl-process-group': False
    }

    p = subprocess.run([python, '-m', 'pip', 'list'], capture_output=True)

    for line in p.stdout.decode().splitlines():
        package = line.split(' ')[0]
        if package.lower() in packages:
            packages[package] = True

    return packages


def check_torch():
    p = subprocess.run(
        [python, '-c',
         'import torch; exit(0 if torch._C._GLIBCXX_USE_CXX11_ABI else 1)'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.returncode == 0


def clone_pytorch():
    print("Cloning PyTorch...")
    url = 'https://github.com/pytorch/pytorch.git'

    subprocess.run(['git', 'clone', '--depth=1', '--recursive',
                    f'--branch={CURRENT_PYTORCH_BRANCH}',
                    url, 'torch'],
                   check=True, cwd=root)


def clone_accl():
    print("Cloning ACCL...")
    url = 'https://github.com/Xilinx/ACCL.git'

    subprocess.run(['git', 'clone', '--depth=1', '--recursive',
                    f'--branch={CURRENT_ACCL_BRANCH}',
                    url, 'accl'],
                   check=True, cwd=root)


def install_pytorch(rocm: bool = False, cuda: bool = False):
    if not torch_dir.exists():
        clone_pytorch()

    print("Installing requirements...")
    subprocess.run([python, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                   cwd=torch_dir, check=True)

    env = os.environ.copy()
    env['USE_ROCM'] = '1' if rocm else '0'
    env['USE_CUDA'] = '1' if cuda else '0'
    env['USE_CONDA'] = '0'
    env['USE_NUMPY'] = '1'
    env['TORCH_CXX_FLAGS'] = '-D_GLIBCXX_USE_CXX11_ABI=1'

    if rocm:
        print("Hipifying PyTorch...")
        subprocess.run([python, 'tools/amd_build/build_amd.py'],
                       cwd=torch_dir, env=env, check=True)

    print("Installing PyTorch...")
    subprocess.run([python, '-m', 'pip', '-v', 'install', '.'],
                   cwd=torch_dir, env=env, check=True)


def install_accl_driver(accl_driver_path: Path):
    print("Installing accl driver...")
    if 'ACCL_DEBUG' in os.environ:
        extra_args = ['-DACCL_DEBUG=1']
    else:
        extra_args = []
    subprocess.run(['/bin/cmake', '.', *extra_args],
                   cwd=accl_driver_path, check=True)
    subprocess.run(['make'], cwd=accl_driver_path, check=True)


def install_accl_process_group(rocm: bool = False, cuda: bool = False):
    if not accl_driver_path.exists():
        clone_accl()
    if not accl_driver.exists():
        install_accl_driver(accl_driver_path)

    print("Installing ACCL Process Group...")
    env = os.environ.copy()
    env['USE_ROCM'] = '1' if rocm else '0'
    env['USE_CUDA'] = '1' if cuda else '0'
    subprocess.run([python, '-m', 'pip', '-v', 'install', '.'],
                   env=env, cwd=root, check=True)


def main(rocm: bool = False, cuda: bool = False,
         force_accl_process_group: bool = False, force_pytorch: bool = False):
    packages = test_packages()

    if force_pytorch and torch_dir.exists():
        rmtree(torch_dir)

    if not packages['torch'] or force_pytorch:
        print("PyTorch not found, installing...")
        install_pytorch(rocm, cuda)
    elif not check_torch():
        print("Currently installed version of PyTorch does not use CXX11 ABI, "
              "please rerun with the --force-pytorch flag enabled.")
        exit(1)

    if not packages['accl-process-group'] or force_accl_process_group:
        print("ACCL Process Group not found, installing...")
        install_accl_process_group(rocm, cuda)


if __name__ == '__main__':
    import argparse
    from argparse import RawDescriptionHelpFormatter

    parser = argparse.ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description='ACCL Process Group installer\n\nInstalls the ACCL '
        'ProcessGroup in the current virtual environment.\nWill also install '
        'PyTorch if it isn\'t installed already.')
    gpu_support = parser.add_mutually_exclusive_group()
    gpu_support.add_argument('--rocm', action='store_true',
                             help='Installs the Process Group with ROCm '
                             'support.')
    gpu_support.add_argument('--cuda', action='store_true',
                             help='Installs the Process Group with CUDA '
                             'support.')
    parser.add_argument('--force-accl-process-group', action='store_true',
                        help='Force a reinstall of the ACCL Process Group')
    parser.add_argument('--force-pytorch', action='store_true',
                        help='Force a reinstall of PyTorch '
                        f'{CURRENT_PYTORCH_VERSION} with the correct CXX11 ABI'
                        ' settings applied.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Enables both --force-accl-process-group and '
                        '--force-pytorch.')

    args = parser.parse_args()
    if args.force:
        args.force_accl_process_group = True
        args.force_pytorch = True

    try:
        main(args.rocm, args.cuda, args.force_accl_process_group,
             args.force_pytorch)
    except KeyboardInterrupt:
        print("Cancelled installation")
        exit(1)
