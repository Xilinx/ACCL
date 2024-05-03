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

from __future__ import annotations
from typing import Optional
import numpy as np
import os
from mpi4py.MPI import COMM_WORLD as mpi

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
import accl_process_group as accl

rank = 0
size = 0

count = 1024
rxbufsize = 1500 * 4


def test_broadcast():
    if rank == 0:
        x = torch.linspace(0, 10, count)
    else:
        x = torch.zeros(count)

    dist.broadcast(x, 0)

    np.testing.assert_allclose(x, torch.linspace(0, 10, count)
                               .to(torch.float16).to(torch.float32),
                               rtol=1e-3, atol=1e-4)
    print("Test broadcast finished!")


def test_sendrcv():
    x = torch.linspace(rank, rank + 1, count)

    y = torch.empty(count)

    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size

    if rank % 2:
        dist.send(x, next_rank)
        dist.recv(y, prev_rank)
    else:
        dist.recv(y, prev_rank)
        dist.send(x, next_rank)

    np.testing.assert_allclose(y, torch.linspace(prev_rank, prev_rank + 1,
                               count).to(torch.float16).to(torch.float32),
                               rtol=1e-3, atol=1e-4)
    print("Test sendrcv finished!")


def test_scatter():
    if rank == 0:
        x = [torch.linspace(i, i + 1, count) for i in range(size)]
    else:
        x = None
    y = torch.empty(count)

    dist.scatter(y, x, 0)

    np.testing.assert_allclose(y, torch.linspace(rank, rank + 1, count)
                               .to(torch.float16).to(torch.float32),
                               rtol=1e-3, atol=1e-4)
    print("Test scatter finished!")


def test_gather():
    x = torch.linspace(rank, rank + 1, count)

    if rank == 0:
        y = [torch.empty(count) for _ in range(size)]
    else:
        y = None

    dist.gather(x, y, 0)

    if rank == 0:
        for i, c in enumerate(y):
            np.testing.assert_allclose(c, torch.linspace(i, i + 1, count)
                                       .to(torch.float16).to(torch.float32),
                                       rtol=1e-3, atol=1e-4)
    print("Test gather finished!")


def test_allgather():
    x = torch.linspace(rank, rank + 1, count)
    y = [torch.empty(count) for _ in range(size)]

    dist.all_gather(y, x)

    for i, c in enumerate(y):
        np.testing.assert_allclose(c, torch.linspace(i, i + 1, count)
                                   .to(torch.float16).to(torch.float32),
                                   rtol=1e-3, atol=1e-4)
    print("Test allgather finished!")


def test_reduce():
    x = torch.linspace(0, 10, count)

    dist.reduce(x, 0, dist.ReduceOp.SUM)

    if rank == 0:
        np.testing.assert_allclose(x, torch.linspace(0, 10 * size, count)
                                   .to(torch.float16).to(torch.float32),
                                   rtol=5e-3, atol=5e-4)
    print("Test reduce finished!")


def test_allreduce():
    x = torch.linspace(0, 10, count)

    dist.all_reduce(x, dist.ReduceOp.SUM)

    np.testing.assert_allclose(x, torch.linspace(0, 10 * size, count)
                               .to(torch.float16).to(torch.float32),
                               rtol=5e-3, atol=5e-4)
    print("Test allreduce finished!")


def start_test(simulator: bool, *,
               xclbin: Optional[str] = None,
               device_index: Optional[int] = None):
    global rank, size
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '30500'
    rank = mpi.Get_rank()
    size = mpi.Get_size()
    ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize)
             for i in range(size)]

    compression = {accl.DataType.float32: accl.DataType.float16}

    if simulator:
        accl.create_simulate_process_group(ranks, bufsize=rxbufsize,
                                           compression=compression)
    else:
        accl.create_process_group(ranks, xclbin, device_index,
                                  accl.ACCLDesign.axis3x,
                                  bufsize=rxbufsize,
                                  compression=compression)
    dist.init_process_group("ACCL", rank=rank, world_size=size)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True) as prof:
        mpi.Barrier()
        test_broadcast()
        mpi.Barrier()
        test_sendrcv()
        mpi.Barrier()
        test_scatter()
        mpi.Barrier()
        test_gather()
        mpi.Barrier()
        test_allgather()
        mpi.Barrier()
        test_reduce()
        mpi.Barrier()
        test_allreduce()

    print(prof.key_averages(group_by_input_shape=True)
          .table(sort_by="cpu_time_total", row_limit=15))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tests for ACCL ProcessGroup')
    parser.add_argument('-s', '--simulation', action='store_true',
                        default=False, help='Use simulation instead of '
                                            'hardware')
    parser.add_argument('--xclbin', type=str, default='',
                        help='Path to xclbin')
    parser.add_argument('--device-index', type=int, default=0,
                        help='Device index of fpga')

    args = parser.parse_args()

    start_test(
        args.simulation,
        xclbin=args.xclbin,
        device_index=args.device_index)
