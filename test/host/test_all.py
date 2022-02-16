#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import signal
import sys
import re
import time
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Tests:
    sendrecv = 'sndrcv'
    sendrecv_pl_kernel = 'sndrcv_strm'
    sendrecv_fanin = 'sndrcv_fanin'
    bcast = 'bcast'
    scatter = 'scatter'
    gather = 'gather'
    allgather = 'allgather'
    reduce = 'reduce'


@dataclass(frozen=True)
class Config:
    nranks: int
    type: str
    stacktype: str = 'tcp'
    start_port: int = 5000
    log: Path = Path('log/')

    # internals
    emulation: Path = field(init=False, default=Path('../emulation'))
    simulation: Path = field(init=False, default=Path('../simulation'))
    cclo: Path = field(init=False, default=Path('../../kernels/cclo'))
    timeout: int = field(init=False, default=20)
    tests: list[str] = field(init=False, default_factory=lambda: [
        Tests.sendrecv, Tests.sendrecv_pl_kernel, Tests.sendrecv_fanin,
        Tests.bcast, Tests.scatter, Tests.gather, Tests.allgather,
        Tests.reduce
    ])
    reduce_func: list[int] = field(init=False, default_factory=lambda: [0])


def compile_emulator(cfg: Config):
    process = subprocess.run(['make'], cwd=cfg.emulation, capture_output=True)
    if process.returncode:
        print(f"Error: failed to compile emulator. Exited with error code "
              f"{process.returncode}\nstdout:\n{process.stdout}\n"
              f"stderr:\n{process.stderr}")
        sys.exit(1)


@contextmanager
def run_emulator(cfg: Config, test: str):
    log = cfg.log / 'emulator'
    log.mkdir(parents=True, exist_ok=True)
    with (log / f'log-{test}-{datetime.now():%Y%m%dT%H%M%S}.txt').open('w') \
            as f:
        process = subprocess.Popen(['mpirun', '-np', f'{cfg.nranks}',
                                    '--tag-output', './cclo_emu',
                                    f'{cfg.stacktype}', f'{cfg.start_port}'],
                                cwd=cfg.emulation, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        time.sleep(5)
        try:
            yield process
        finally:
            process.terminate()
            process.wait()
            time.sleep(2)


def compile_simulator(cfg: Config):
    process = subprocess.run(['make'], cwd=cfg.emulation, capture_output=True)
    if process.returncode:
        print(f"Error: failed to compile simulator. Exited with error code "
              f"{process.returncode}\nstdout:\n{process.stdout}\n"
              f"stderr:\n{process.stderr}")
        sys.exit(1)


def compile_cclo(cfg: Config):
    process = subprocess.run(['make', f'STACKTYPE={cfg.stacktype.upper()}',
                              'EN_FANIN=1', 'simdll'],
                             cwd=cfg.cclo, capture_output=True)
    if process.returncode:
        print(f"Error: failed to compile simulator. Exited with error code "
              f"{process.returncode}\nstdout:\n{process.stdout}\n"
              f"stderr:\n{process.stderr}")
        sys.exit(1)


@contextmanager
def run_simulator(cfg: Config, test: str):
    log = cfg.log / 'simulator'
    log.mkdir(parents=True, exist_ok=True)
    vivado = os.environ['XILINX_VIVADO']
    env =  {**dict(os.environ),'LD_LIBRARY_PATH': f'{vivado}/lib/lnx64.o'}
    args = ['mpirun', '-np', f'{cfg.nranks}', '--tag-output', './cclo_sim',
            f'{cfg.stacktype}', f'{cfg.start_port}',
            'xsim.dir/ccl_offload_behav/xsimk.so']
    with (log / f'log-{test}-{datetime.now():%Y%m%dT%H%M%S}.txt').open('w') \
            as f:
        process = subprocess.Popen(args, env=env, cwd=cfg.simulation, stdout=f,
                                   stderr=subprocess.DEVNULL)
        time.sleep(5)
        try:
            yield process
        finally:
            process.terminate()
            process.wait()
            time.sleep(1)


def run_successfull(stdout: str):
    return re.search(r'succeeded', stdout.splitlines()[-2].decode()) is not None


def run_test(test: str, cfg: Config, reduce_func: int = None):
    if test == Tests.reduce:
        test_name = f'{test} ({reduce_func})'
    else:
        test_name = test

    print(f"Starting test {test_name}.")
    args = ['mpirun', '-np', f'{cfg.nranks}',
            'python3', 'test_sim.py', f'--{test}']
    if cfg.stacktype == 'tcp':
        args.append('--tcp')
    process = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.wait(timeout=cfg.timeout)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
        process.wait()
        stdout, stderr = process.communicate()
        print(f"Test {test_name} timed out!\nstdout:\n{stdout.decode()}"
              f"stderr:\n{stderr.decode()}")
        return

    stdout, stderr = process.communicate()

    if process.returncode or not run_successfull(stdout):
        print(f"Test {test_name} failed (errorcode {process.returncode})!\n"
              f"stdout:\n{stdout.decode()}stderr:\n{stderr.decode()}")
    else:
        print(f"Test {test_name} succeeded!")


def start_test(test: str, cfg: Config, reduce_func: int = None):
    if cfg.type == 'emulation':
        with run_emulator(cfg, test):
            run_test(test, cfg, reduce_func)
    else:
        with run_simulator(cfg, test):
            run_test(test, cfg, reduce_func)


def run(cfg: Config):
    if cfg.type == 'emulation':
        exe = cfg.emulation / 'cclo_emu'
        if not exe.exists():
            compile_emulator(cfg)
    elif cfg.type == 'simulation':
        exe = cfg.simulation / 'cclo_sim'
        if not exe.exists():
            compile_emulator(cfg)

        sim = (cfg.simulation / 'xsim.dir').resolve()
        if not sim.exists():
            compile_cclo(cfg)

    for test in cfg.tests:
        if test == Tests.reduce:
            for reduce_func in cfg.reduce_func:
                start_test(test, cfg, reduce_func)
        else:
            start_test(test, cfg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run tests for ACCL')
    parser.add_argument('-n', '--nranks', type=int, required=True,
                        help='How many processes to use for the tests')
    parser.add_argument('-t', '--type', type=str,required=True,
                        help='Type of test, use simulation or emulation')
    parser.add_argument('--udp', action='store_true', default=False,
                        help='Run tests using UDP')
    parser.add_argument('--start_port', type=int, default=5500,
                        help='Start of range of ports usable for sim')

    args = parser.parse_args()
    cfg = Config(args.nranks, args.type,
                 'udp' if args.udp else 'tcp', args.start_port)

    run(cfg)
