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
        x = torch.ones(count)
    else:
        x = torch.zeros(count)

    dist.broadcast(x, 0)

    np.testing.assert_allclose(x, torch.ones(count))
    print("Test broadcast finished!")


def test_sendrcv():
    x = torch.full((count,), float(rank))

    y = torch.empty(count)

    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size

    if rank % 2:
        dist.send(x, next_rank)
        dist.recv(y, prev_rank)
    else:
        dist.recv(y, prev_rank)
        dist.send(x, next_rank)

    np.testing.assert_allclose(y, torch.full((count,), prev_rank))
    print("Test sendrcv finished!")


def test_scatter():
    if rank == 0:
        x = [torch.full((count,), float(i)) for i in range(size)]
    else:
        x = None
    y = torch.empty(count)

    dist.scatter(y, x, 0)

    np.testing.assert_allclose(y, torch.full((count,), float(rank)))
    print("Test scatter finished!")


def test_gather():
    x = torch.full((count,), float(rank))

    if rank == 0:
        y = [torch.empty(count) for _ in range(size)]
    else:
        y = None

    dist.gather(x, y, 0)

    if rank == 0:
        for i, c in enumerate(y):
            np.testing.assert_allclose(c, torch.full((count,), float(i)))
    print("Test gather finished!")


def test_allgather():
    x = torch.full((count,), float(rank))
    y = [torch.empty(count) for _ in range(size)]

    dist.all_gather(y, x)

    for i, c in enumerate(y):
        np.testing.assert_allclose(c, torch.full((count,), float(i)))
    print("Test allgather finished!")


def test_reduce():
    x = torch.ones(count)

    dist.reduce(x, 0, dist.ReduceOp.SUM)

    if rank == 0:
        np.testing.assert_allclose(x, [size for _ in range(count)])
    print("Test reduce finished!")


def test_allreduce():
    x = torch.ones(count)

    dist.all_reduce(x, dist.ReduceOp.SUM)

    np.testing.assert_allclose(x, [size for _ in range(count)])
    print("Test allreduce finished!")


def exchange_qp(first_rank, second_rank, rank, ranks):
    if rank == first_rank:
        mpi.send(accl.get_local_qp(second_rank), dest=second_rank, tag=23)
    elif rank == second_rank:
        accl.set_remote_qp(first_rank, mpi.recv(source=first_rank, tag=23))

    mpi.barrier()

    if rank == second_rank:
        mpi.send(accl.get_local_qp(first_rank), dest=first_rank, tag=24)
    elif rank == first_rank:
        accl.set_remote_qp(second_rank, mpi.recv(source=second_rank, tag=24))

    mpi.barrier()


def configure_cyt_rdma(ranks):
    global rank, size
    for first_rank in range(0, size):
        for second_rank in range(first_rank + 1, size):
            exchange_qp(first_rank, second_rank, rank, ranks)
    accl.initialize()
    mpi.barrier()



def start_test(comms: str, simulator: bool):
    global rank, size
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '30500'
    rank = mpi.Get_rank()
    size = mpi.Get_size()
    ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize)
             for i in range(size)]

    accl.create_process_group(ranks, accl.ACCLDesign.cyt_rdma, bufsize=rxbufsize, initialize=True)
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
    parser = argparse.ArgumentParser(description='Coyote tests for ACCL ProcessGroup')
    parser.add_argument('-s', '--simulation', action='store_true',
                        default=False, help='Use simulation instead of '
                                            'hardware')
    parser.add_argument('-c', '--comms', choices=['udp', 'tcp', 'cyt_rdma'], default='tcp',
                        help='Run tests over specied communication backend')
    args = parser.parse_args()

    #if args.comms != 'cyt_rdma' or not args.simulation:
    if args.comms != 'cyt_rdma':
        sys.exit('Currently only supports -c cyt_rdma and -s flags')
    start_test(args.comms, args.simulation)
