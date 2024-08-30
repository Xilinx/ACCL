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

x = 1024
y = 1

seed = 48
torch.manual_seed(seed)

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

def test_broadcast(numel, testtype):
    shape = (numel,)

    # testtype = torch.float32
    global num_errors

    if testtype == torch.int64 or testtype == torch.int32:
        rand_torch = torch.randint(torch.iinfo(testtype).min, torch.iinfo(testtype).max,shape, dtype=testtype)
        # rand_torch = torch.ones(shape, dtype=testtype)
    else:
        rand_torch = torch.rand(shape, dtype=testtype)
    
    # for i in range(10):
    if True:

        if rank == 0:
            x = rand_torch.clone()
        else:
            x = torch.ones(shape, dtype=testtype)

        mpi.Barrier()            
        
        with torch.profiler.record_function("test bcast "):

            start_time = time.perf_counter()

            dist.broadcast(x, 0)

            end_time = time.perf_counter()
            
        measured_time = (end_time - start_time) * 1000000

        print(str(rank) + "_pytorch_Broadcast_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        
        logger.debug("Directly measured time us 1:" + str(measured_time))
            
        mpi.Barrier()

        end_time = time.perf_counter()

        measured_time = (end_time - start_time) * 1000000

        logger.debug("Directly measured time us 2:" + str(measured_time))

    try:
        np.testing.assert_allclose(x, rand_torch)
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

        
def test_sendrcv(numel):
    global num_errors

    shape = (numel,)
    x = torch.full(shape, float(rank))

    y = torch.empty(shape)

    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size


    with torch.profiler.record_function("test_sendrcv"):
        if rank % 2:
            mpi.Barrier()            
            start_time = time.perf_counter()
            dist.send(x, next_rank)
            end_time = time.perf_counter()
            measured_time = (end_time - start_time) * 1000000
            print(str(rank) + "_pytorch_Send_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)

            mpi.Barrier()            
            start_time = time.perf_counter()
            dist.recv(y, prev_rank)
            end_time = time.perf_counter()
            measured_time = (end_time - start_time) * 1000000
            print(str(rank) + "_pytorch_Recv_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        else:
            mpi.Barrier()            
            start_time = time.perf_counter()
            dist.recv(y, prev_rank)
            end_time = time.perf_counter()
            measured_time = (end_time - start_time) * 1000000
            print(str(rank) + "_pytorch_Recv_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)

            mpi.Barrier()            
            start_time = time.perf_counter()
            dist.send(x, next_rank)
            end_time = time.perf_counter()
            measured_time = (end_time - start_time) * 1000000
            print(str(rank) + "_pytorch_Send_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        mpi.Barrier()
    try:
        np.testing.assert_allclose(y, torch.full(shape, prev_rank))
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Sendrcv failed")
        logger.debug(str(e))
    else:
        logger.debug("Test Sendrcv finished!")


def test_scatter(numel):
    global num_errors

    shape = (numel,)
    if rank == 0:
        x = [torch.full(shape, float(i+1)) for i in range(size)]
    else:
        x = None
    y = torch.full(shape, float(0))

    mpi.Barrier()            
    start_time = time.perf_counter()
    
    with torch.profiler.record_function("test_scatter"):
        
        dist.scatter(y, x, 0)

    end_time = time.perf_counter()
    measured_time = (end_time - start_time) * 1000000
    print(str(rank) + "_pytorch_Scatter_" + str(y.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
    
    try:
        np.testing.assert_allclose(y, torch.full(shape, float(rank+1)))
    except AssertionError as e:
        num_errors = num_errors + 1
        logger.debug("Test Scatter failed")
        logger.debug(str(e))
    else:
        logger.debug("Test Scatter finished!")
    


def test_gather(numel):
    global num_errors

    shape = (numel,)
    x = torch.full(shape, float(rank))

    if rank == 0:
        y = [torch.empty(shape) for _ in range(size)]
    else:
        y = None

    mpi.Barrier()            
    start_time = time.perf_counter()
        
    with torch.profiler.record_function("test_gather"):
            
        dist.gather(x, y, 0)

    end_time = time.perf_counter()
    measured_time = (end_time - start_time) * 1000000
    print(str(rank) + "_pytorch_Gather_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
    
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

            
def test_allgather(numel, testtype):
    global num_errors

    shape = (numel,)
    if testtype == torch.int64 or testtype == torch.int32:
        rand_torch = torch.randint(torch.iinfo(testtype).min, torch.iinfo(testtype).max,shape, dtype=testtype)
    else:
        rand_torch = torch.rand(shape, dtype=testtype)
    x = rand_torch.clone()
    y = [torch.full(shape, 0, dtype=testtype) for _ in range(size)]

    mpi.Barrier()            
    start_time = time.perf_counter()

    print('len y:' + str(len(y)))
    
    with torch.profiler.record_function("test_allgather"):
        dist.all_gather(y, x)

    end_time = time.perf_counter()
    measured_time = (end_time - start_time) * 1000000
    print(str(rank) + "_pytorch_Allgather_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        
    mpi.Barrier()

        
    for i, c in enumerate(y):
        try:
            np.testing.assert_allclose(c, rand_torch)
        except AssertionError as e:
            num_errors = num_errors + 1
            logger.debug("Test AllGather failed")
            logger.debug(str(e))
        else:
            logger.debug("Test AllGather finished!")
        


def test_reduce(numel):
    global num_errors


    shape = (numel,)
    x = torch.ones(shape)

    mpi.Barrier()            
    start_time = time.perf_counter()
    with torch.profiler.record_function("test_reduce"):

        dist.reduce(x, 0, dist.ReduceOp.SUM)
        mpi.Barrier()

    end_time = time.perf_counter()
    measured_time = (end_time - start_time) * 1000000
    print(str(rank) + "_pytorch_Reduce_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
    
    if rank == 0:
        try:
            np.testing.assert_allclose(x, torch.full(shape, float(size)))
        except AssertionError as e:
            num_errors = num_errors + 1
            logger.debug("Test Reduce failed")
            logger.debug(str(e))
        else:
            logger.debug("Test Reduce finished!")
        

def test_allreduce(numel, testtype):

    global num_errors

    shape = (numel,)

    
    if testtype == torch.int64 or testtype == torch.int32:
        rand_torch = torch.randint(torch.iinfo(testtype).min//size, torch.iinfo(testtype).max//size,shape, dtype=testtype)
    else:
        rand_torch = torch.rand(shape, dtype=testtype)
    
    # for i in range(10):
    if True:
    
        # shape = (320001,)
        x = rand_torch.clone()

        mpi.Barrier()            
        
        start_time = time.perf_counter()

        
        with torch.profiler.record_function("test_allreduce"):

            dist.all_reduce(x, dist.ReduceOp.SUM)

        end_time = time.perf_counter()
        measured_time = (end_time - start_time) * 1000000
        print(str(rank) + "_pytorch_Allreduce_" + str(x.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        
        logger.debug("Directly measured time us 1:" + str(measured_time))            
        
        mpi.Barrier()

        try:
            np.testing.assert_allclose(x, rand_torch * size)
        except AssertionError as e:
            num_errors = num_errors + 1
            logger.debug("Test AllReduce failed")
            logger.debug(str(e))
        else:
            logger.debug("Test AllReduce finished!")
        
    
def test_alltoall(numel):
    global num_errors

    # num_el = 26624
    
    shape = (numel,)

    input = torch.arange(numel, dtype=torch.float) + float(rank) * numel

    input_shaped = input.reshape(shape)

    output = torch.ones(numel)

    output_shaped = output.reshape(shape)

    start_time = time.perf_counter()
    
    with torch.profiler.record_function("test_alltoall"):
        
        dist.all_to_all_single(output_shaped, input_shaped)

    end_time = time.perf_counter()

    measured_time = (end_time - start_time) * 1000000
    
    print(str(rank) + "_pytorch_AlltoAll_" + str(input.nbytes) + " durationUs: " + str(measured_time), file=sys.stderr)
        
    test = torch.zeros(numel)

    section_size = int(numel/size)

    for section in range(size):
        for el in range(section_size):
            test[section * section_size + el] = float(rank) * section_size + section * numel + el

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

    test_allreduce(256, torch.float32)
    test_broadcast(256, torch.float32)

    schedule = torch.profiler.schedule(
        wait=1,
        warmup=2,
        active=5,
    )
    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, schedule=schedule, record_shapes=True) as prof:

    n = 19
    
    if True:
        for i in range(40):
            num = 2**n * 3
            test_broadcast(num, torch.float32)
            test_allreduce(num, torch.float32)
            test_alltoall(num)
            test_allgather(num, torch.float32)
            test_sendrcv(num)
            test_scatter(num)
            test_gather(num)
            test_reduce(num)
            
            # prof.step()
    
    # for i in range(10):
    if False:
        # test_allreduce(256, torch.int32)
        # test_allreduce(256, torch.int64)
        # test_broadcast(256, torch.float32)
        
        # test_allgather()

        # test_broadcast_2()
        test_broadcast(1024, torch.float32)
        # test_broadcast(25610152, torch.float32)
        # test_broadcast(53, torch.int64)
        # test_broadcast(53120, torch.float32)
        # test_broadcast(53, torch.int64)
        test_allreduce(1024, torch.float32)
        # test_broadcast(162, torch.int32)
        # test_broadcast(25, torch.int32)
        # test_broadcast(53120, torch.float32)
        # test_broadcast(53, torch.int64)
        # test_allreduce(2049000, torch.float32)
        # test_allreduce()
        # test_broadcast_segment()
        # test_broadcast()
        # test_broadcast()
        # test_broadcast()
        # test_broadcast()
        # test_broadcast()
        test_alltoall()
        # test_allreduce(1000, torch.float32)
        # test_allreduce(2052096, torch.float32)
        # test_allreduce(1049600, torch.float32)
        # test_broadcast(256 * 1024, torch.float32)
        # test_allreduce(256 * 1024, torch.float32)        
        # test_broadcast(53, torch.int64)
        # test_broadcast(53120, torch.float32)
        # test_broadcast(53, torch.int64)
        # test_broadcast(162, torch.int32)
        # test_broadcast(25, torch.int32)
        # test_allreduce(8196000, torch.float32)
        # test_allreduce()
        # test_allreduce()



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
