# /*****************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
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
from pathlib import Path
import subprocess
import argparse
import sys
import signal

from matplotlib import use

cwd = Path(__file__).parent.resolve()
executable = cwd / 'cclo_sim'
makefile = cwd / 'Makefile'
xsim_path_tail = 'xsim.dir/ccl_offload_behav/xsimk.so'


def gen_makefile():
    p = subprocess.run(['cmake', '.'], cwd=cwd)
    if p.returncode != 0:
        print("cmake failed!")
        sys.exit(1)


def build_executable():
    p = subprocess.run(['make'], cwd=cwd)
    if p.returncode != 0:
        print("make failed!")
        sys.exit(1)


def run_simulator(ranks: int, start_port: int, use_udp: bool):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f"{os.environ['XILINX_VIVADO']}/lib/lnx64.o"
    args = ['mpirun', '-np', str(ranks), '--tag-output', str(executable),
            'udp' if use_udp else 'tcp', str(start_port), xsim_path_tail]
    print(' '.join(args))
    with subprocess.Popen(args, cwd=cwd, env=env) as p:
        try:
            p.wait()
        except KeyboardInterrupt:
            try:
                print("Stopping simulator...")
                p.send_signal(signal.SIGINT)
            except KeyboardInterrupt:
                try:
                    print("Force stopping simulator...")
                    p.kill()
                except KeyboardInterrupt:
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    print("Terminating simulator...")
                    p.terminate()


def main(ranks: int, start_port: int, use_udp: bool, build: bool):
    if not build and not executable.exists():
        print(f"Executable {executable} does not exists!")
        sys.exit(1)
    else:
        if not makefile.exists():
            print("Makefile doesn't exist, running cmake...")
            gen_makefile()
        print("Building executable...")
        build_executable()

    print("Starting simulator...")
    run_simulator(ranks, start_port, use_udp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ACCL simulator')
    parser.add_argument('-n', '--nranks', type=int, default=1,
                        help='How many ranks to use for the simulator')
    parser.add_argument('-s', '--start-port', type=int, default=5500,
                        help='Start port of simulator')
    parser.add_argument('-u', '--udp', action='store_true', default=False,
                        help='Run simulator over UDP instead of TCP')
    parser.add_argument('--no-build', action='store_true', default=False,
                        help="Don't build latest executable")
    args = parser.parse_args()
    main(args.nranks, args.start_port, args.udp, not args.no_build)
