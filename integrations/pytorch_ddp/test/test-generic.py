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
import time
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

x = 28939
y = 1

count = x * y
num_el = x * y
shape = (x , y)
#As in test.cpp defaults
rxbufsize = 4096 * 1024


def test_broadcast_segment():
    global num_errors
    shape_segment = (1024 * 1,)
    if rank == 0:
        x = torch.ones(shape_segment, dtype=torch.float)
    else:
        x = torch.zeros(shape_segment, dtype=torch.float)

    with torch.profiler.record_function("test bcast segmented"):
            
        dist.broadcast(x, 0)

        mpi.Barrier()            
        # logger.debug('Tensor after broadcast: ' + str(x))
        # print('Tensor after broadcast: ' + str(x))
    try:
        np.testing.assert_allclose(x, torch.ones(shape_segment, dtype=torch.float))
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Broadcast failed")
        logger.debug(str(e))
    else:
        logger.debug("Test broadcast finished!")

def test_broadcast():
    shape = (256,)
    
    global num_errors

        
    # for i in range(10):
    if True:

        if rank == 0:
            x = torch.ones(shape)
        else:
            x = torch.zeros(shape)

        
        with torch.profiler.record_function("test bcast "):

            start_time = time.perf_counter()

            dist.broadcast(x, 0)

            end_time = time.perf_counter()
            
            measured_time = (end_time - start_time) * 1000000
            
            logger.debug("Directly measured time us 1:" + str(measured_time))
            
            mpi.Barrier()

            end_time = time.perf_counter()

            measured_time = (end_time - start_time) * 1000000

            logger.debug("Directly measured time us 2:" + str(measured_time))

    # logger.debug('Tensor after broadcast: ' + str(x))
    # print('Tensor after broadcast: ' + str(x))
    try:
        np.testing.assert_allclose(x, torch.ones(shape))
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Broadcast failed")
        logger.debug(str(e))
    else:
        logger.debug("Test broadcast finished!")

def test_broadcast_2():
    test_type = torch.float
    shape_2 = (1048576,)
    global num_errors
    if rank == 0:
        x = torch.ones(shape_2, dtype=test_type)
    else:
        x = torch.zeros(shape_2, dtype=test_type)

    with torch.profiler.record_function("test bcast float prec"):
        dist.broadcast(x, 0)
        mpi.Barrier()            

    # logger.debug('Tensor after broadcast: ' + str(x))
    # print('Tensor after broadcast: ' + str(x))
    try:
        np.testing.assert_allclose(x, torch.ones(shape_2, dtype=test_type))
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Broadcast failed")
        logger.debug(str(e))
    else:
        logger.debug("Test broadcast finished!")

        
def test_sendrcv():
    global num_errors
    x = torch.full(shape, float(rank))

    y = torch.empty(shape)

    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size


    with torch.profiler.record_function("test_sendrcv"):

        if rank % 2:
            dist.send(x, next_rank)
            dist.recv(y, prev_rank)
        else:
            dist.recv(y, prev_rank)
            dist.send(x, next_rank)
        mpi.Barrier()
    try:
        np.testing.assert_allclose(y, torch.full(shape, prev_rank))
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Sendrcv failed")
        logger.debug(str(e))
    else:
        logger.debug("Test Sendrcv finished!")


def test_scatter():
    global num_errors
    if rank == 0:
        x = [torch.full(shape, float(i+1)) for i in range(size)]
    else:
        x = None
    y = torch.full(shape, float(0))

    with torch.profiler.record_function("test_scatter"):
        
        dist.scatter(y, x, 0)
        mpi.Barrier()
    try:
        np.testing.assert_allclose(y, torch.full(shape, float(rank+1)))
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Scatter failed")
        logger.debug(str(e))
    else:
        logger.debug("Test Scatter finished!")
    


def test_gather():
    global num_errors
    x = torch.full(shape, float(rank))

    if rank == 0:
        y = [torch.empty(shape) for _ in range(size)]
    else:
        y = None

    with torch.profiler.record_function("test_gather"):
            
        dist.gather(x, y, 0)
        mpi.Barrier()
    if rank == 0:
        for i, c in enumerate(y):
            try:
                np.testing.assert_allclose(c, torch.full(shape, float(i)))
            except AssertionError as e:
                num_errors = num_errors + 1
                logger.debug("Test Gather failed")
                logger.debug(str(e))
            else:
                logger.debug("Test Gather finished!")

            
def test_allgather():
    global num_errors
    shape_gather = (2,)
    x = torch.full(shape_gather, float(rank), dtype=torch.float)
    y = [torch.empty(shape_gather, dtype=torch.float) for _ in range(size)]

    with torch.profiler.record_function("test_allgather"):
        
        
        dist.all_gather(y, x)
        mpi.Barrier()
    for i, c in enumerate(y):
        try:
            np.testing.assert_allclose(c, torch.full(shape_gather, float(i), dtype=torch.float))
        except AssertionError as e:
            num_errors = num_errors + 1
            logger.debug("Test AllGather failed")
            logger.debug(str(e))
        else:
            logger.debug("Test AllGather finished!")
        


def test_reduce():
    global num_errors
    x = torch.ones(shape)

    with torch.profiler.record_function("test_reduce"):

        dist.reduce(x, 0, dist.ReduceOp.SUM)
        mpi.Barrier()            
    if rank == 0:
        try:
            np.testing.assert_allclose(x, torch.full(shape, float(size)))
        except AssertionError as e:
            num_errors = num_errors + 1
            logger.debug("Test Reduce failed")
            logger.debug(str(e))
        else:
            logger.debug("Test Reduce finished!")
        

def test_allreduce():

    # for i in range(10):
    if True:
    
        shape = (256,)
        # shape = (320001,)
        global num_errors
        x = torch.ones(shape)

        with torch.profiler.record_function("test_allreduce"):

            dist.all_reduce(x, dist.ReduceOp.SUM)
            mpi.Barrier()

        try:
            np.testing.assert_allclose(x, torch.full(shape, float(size)))
        except AssertionError as e:
            num_errors = num_errors + 1
            logger.debug("Test AllReduce failed")
            logger.debug(str(e))
        else:
            logger.debug("Test AllReduce finished!")
        
    
def test_alltoall():
    global num_errors

    num_el = 26624
    
    shape = (num_el,)

    input = torch.arange(num_el, dtype=torch.float) + float(rank) * num_el

    input_shaped = input.reshape(shape)

    output = torch.ones(num_el)

    output_shaped = output.reshape(shape)

    with torch.profiler.record_function("test_alltoall"):
        
        dist.all_to_all_single(output_shaped, input_shaped)

        mpi.Barrier()

    test = torch.zeros(num_el)

    section_size = int(num_el/size)

    for section in range(size):
        for el in range(section_size):
            test[section * section_size + el] = float(rank) * section_size + section * num_el + el

    test_shaped = test.reshape(shape)
    try:
        np.testing.assert_allclose(output_shaped, test_shaped)
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test AlltoAll failed")
        logger.debug(str(e))
    else:
        logger.debug("Test AlltoAll finished!")
        
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

        self.data = []
        for i in range(size):
            in_feature = torch.zeros(10)
            out_feature = torch.zeros(5)
            for j in range(10):
                in_feature[j] = float((i^2  + j) % 5)
                # try to learn a linear function of the input, to make sure it's parameterizable
                out_feature[j//2] = out_feature[j//2] + float(((i^2 + j) % 5) * 3 * ( -1 ** (j % 2)))
            self.data.append((in_feature, out_feature))
                
                
        

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

    with torch.profiler.record_function("basic 2 Layer NN"):
        model = ToyModel()
        ddp_model = DDP(model, bucket_cap_mb=4)
        # ddp_model = DDP(model, bucket_cap_mb=4, broadcast_buffers=False)
        
        train_set = MyTrainDataset(2048)  # load your dataset
        batch_size=64
        train_data = prepare_dataloader(train_set, batch_size)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(ddp_model.parameters(), lr=0.005)

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
        mpi.Barrier()
    # print("final params:")
    # print(ddp_model)
    # dist.destroy_process_group()

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
    logger.debug(f"Starting tests with the following parameters:\n\
Simulation: {simulator}, Communication Backend: {comms}\n\
Rank: {rank}, World size: {size}\n\
Host file: {host_file}, FPGA file: {fpga_file}\n\
Master address: {ma}:{mp}, Start port for FPGA: {start_port}")
    

    if not simulator:
        #default from test.cpp
        rxbufsize = 4096 * 1024
        if host_file==None or fpga_file==None: sys.exit('Host and FPGA file need to be specified in hardware mode')
            
        with open(host_file, 'r') as hf:
            host_ips = hf.read().splitlines()
            
        with open(fpga_file, 'r') as ff:
            fpga_ips = ff.read().splitlines()

        if comms == "cyt_rdma":
            ranks = [accl.Rank(a, start_port, i, rxbufsize) for i, a in enumerate(fpga_ips)]
        else:
            ranks = [accl.Rank(a, start_port + i, 0, rxbufsize) for i, a in enumerate(fpga_ips)]
    else:
        # Somehow the simulator gets stuck if I use the same rxbufsize
        rxbufsize = 4096 # * 1024
        ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize) for i in range(size)]

    logger.debug(f'Ranks: {ranks}')

    if comms == 'udp':
        design = accl.ACCLDesign.udp
    elif comms == 'tcp':
        design = accl.ACCLDesign.tcp
    elif comms == 'cyt_rdma': # and not simulator:
        design = accl.ACCLDesign.cyt_rdma
    # else:
        # if simulator:
            # sys.exit('Design "' + comms + '" currently not supported in simulator mode')
        # else:
            # sys.exit('Design "' + comms + '" currently not supported in hardware mode')

    # Sometimes ACCL gets stuck on the mpi import statement, so this is to avoid issues:
    mpi.Barrier()            


    # dist.init_process_group("mpi", rank=rank, world_size=size)

    
    accl.create_process_group(ranks, design, bufsize=rxbufsize, initialize=True, simulation=simulator)
    dist.init_process_group("ACCL", rank=rank, world_size=size)
    
    global num_errors
    num_errors = 0
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:

    # for i in range(10):
    if True:

        # test_broadcast_2()
        test_broadcast()
        test_allreduce()
        # test_allgather()
        # test_broadcast_segment()
        # test_broadcast()
        # test_broadcast()
        # test_broadcast()
        # test_broadcast()
        # test_broadcast()
        # test_sendrcv()
        # test_scatter()
        # test_gather()
        # test_allgather()
        # test_alltoall()
        # test_allreduce()
        # test_allgather()
        # test_allreduce()
        # test_allreduce()

        # test_reduce()


        # demo_basic(rank)


    mpi.Barrier()

    if num_errors == 0:
        print("======== Successfully Finished testing======")
        logger.debug("======== Successfully Finished testing======")
    else:
        print(f"!!!!!!!! - {num_errors} Errors found - !!!!!!!!!")
        logger.debug(f"!!!!!!!! - {num_errors} Errors found - !!!!!!!!!")        
    # print(prof.key_averages(group_by_input_shape=True)
          # .table(sort_by="cpu_time_total", row_limit=15))

    logger.debug('Destroying ACCL Process Group')
    dist.destroy_process_group()

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
