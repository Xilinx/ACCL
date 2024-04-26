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
import sys
import logging
from mpi4py.MPI import COMM_WORLD as mpi

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
import accl_process_group as accl

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

#Configure logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

if "ACCL_DEBUG" in os.environ and os.environ["ACCL_DEBUG"]=="1":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)
    
rank = 0
size = 0

count = 1024
#As in test.cpp defaults
rxbufsize = 4096 * 1024


def test_broadcast():
    if rank == 0:
        x = torch.ones(count)
    else:
        x = torch.zeros(count)

    dist.broadcast(x, 0)

    logger.debug('Tensor after broadcast: ' + str(x))
    print('Tensor after broadcast: ' + str(x))
    
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


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(10), torch.rand(5)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
    

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )    
    
def demo_basic(rank: int):
    model = ToyModel()
    ddp_model = DDP(model)

    train_set = MyTrainDataset(2048)  # load your dataset
    batch_size=64
    train_data = prepare_dataloader(train_set, batch_size)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    max_epochs = 10
    for epoch in range(max_epochs):
        batch_size = len(next(iter(train_data))[0])
        train_data.sampler.set_epoch(epoch)
        for x, y in train_data:
            
            optimizer.zero_grad()
            outputs = ddp_model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}: Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(train_data)} | Loss: {loss}")
        

    print("finished training")
    dist.destroy_process_group()

def start_test(comms: str, simulator: bool, host_file: str=None, fpga_file: str=None, ma: str="localhost", mp: str="30505"):
    global rank, size
    if ma==None:
        ma = "localhost"
    if mp==None:
        mp = "30505"
    os.environ['MASTER_ADDR'] = ma
    os.environ['MASTER_PORT'] = mp
    rank = mpi.Get_rank()
    size = mpi.Get_size()
    start_port = 5005
    print(f"Starting tests with the following parameters:\n\
Simulation: {simulator}, Communication Backend: {comms}\n\
Rank: {rank}, World size: {size}\n\
Host file: {host_file}, FPGA file: {fpga_file}\n\
Master address: {ma}:{mp}, Start port for FPGA: {start_port}")
    

    if not simulator:
        if host_file==None or fpga_file==None: sys.exit('Host and FPGA file need to be specified in hardware mode')
            
        with open(host_file, 'r') as hf:
            host_ips = hf.readlines()
            
        with open(fpga_file, 'r') as ff:
            fpga_ips = ff.readlines()

        if comms == "cyt_rdma":
            ranks = [accl.Rank(a, start_port, i, rxbufsize) for i, a in enumerate(fpga_ips)]
        else:
            ranks = [accl.Rank(a, start_port + i, 0, rxbufsize) for i, a in enumerate(fpga_ips)]            
    else:
        ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize) for i in range(size)]

    logger.debug(f'Ranks: {ranks}')

    if comms == 'udp':
        design = accl.ACCLDesign.udp
    elif comms == 'tcp':
        design = accl.ACCLDesign.tcp
    elif comms == 'cyt_rdma' and not simulator:
        design = accl.ACCLDesign.cyt_rdma
    else:
        if simulator:
            sys.exit('Design "' + comms + '" currently not supported in simulator mode')
        else:
            sys.exit('Design "' + comms + '" currently not supported in hardware mode')

    
    accl.create_process_group(ranks, design, bufsize=rxbufsize, initialize=True, simulation=simulator)
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
        # test_reduce()
        # mpi.Barrier()
        # test_allreduce()
        # mpi.Barrier()
        # demo_basic(rank)
        # mpi.Barrier()
        
    print(prof.key_averages(group_by_input_shape=True)
          .table(sort_by="cpu_time_total", row_limit=15))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Coyote tests for ACCL ProcessGroup')
    parser.add_argument('-s', '--simulation', action='store_true',
                        default=False, help='Use simulation instead of '
                                            'hardware')
    parser.add_argument('-c', '--comms', choices=['udp', 'tcp', 'cyt_rdma'], default='tcp',
                        help='Run tests over specified communication backend')
    parser.add_argument('-i', '--host-file', type=str, help='Specify the file, where the host IPs are listed')
    parser.add_argument('-f', '--fpga-file', type=str, help='Specify the file, where the FPGA IPs are listed')
    parser.add_argument('-a','--master-address', type=str)
    parser.add_argument('-p','--master-port', type=str)
    args = parser.parse_args()

    #if args.comms != 'cyt_rdma' or not args.simulation:
    #if args.comms != 'cyt_rdma':
    #    sys.exit('Currently only supports -c cyt_rdma and -s flags')
    start_test(args.comms, args.simulation, args.host_file, args.fpga_file, args.master_address, args.master_port)
