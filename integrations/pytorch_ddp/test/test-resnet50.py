import torch
import torchvision
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.distributed as dist
import accl_process_group as accl

from mpi4py.MPI import COMM_WORLD as mpi
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import argparse
import numpy as np
import os
import sys
import logging
import time

seed = 43
torch.manual_seed(seed)

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

if "ACCL_DEBUG" in os.environ and os.environ["ACCL_DEBUG"]=="1":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)

# Run via ACCL

def train(num_epochs, model, loaders, criterion):

    start_time_train = time.perf_counter()
    
    model.train()

    total_step = len(loaders['train'])

    optimizer = optim.Adam(model.parameters(), lr = 0.001)   

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loaders['train']):
            start_time = time.perf_counter()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i+1) % 100 == 0:
                break
            if True:
                end_time = time.perf_counter()
                measured_time = (end_time - start_time) * 1000000
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time(us): {}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), measured_time))

    end_time_train = time.perf_counter()
    measured_time_train = (end_time_train - start_time_train) * 1000000

    print('Total train time: ' + str(measured_time_train))
        

def test(num_epochs, model, loaders, criterion):
    # Test the model
    start_time_test = time.perf_counter()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0
        for i, (inputs, labels) in enumerate(loaders['test']):
            test_output = model(inputs)
            loss = criterion(test_output, labels)
            val_loss += loss.item()

            _, predicted = torch.max(test_output, 1)
            correct_current = (predicted == labels).sum().item()
            total += labels.size(0)
            correct += correct_current
            
            print(f'Test Batch accuracy: {correct_current}/{labels.size(0)} {correct_current/float(labels.size(0))}')


    end_time_test = time.perf_counter()
    measured_time_test = (end_time_test - start_time_test) * 1000000

    print('Total test time: ' + str(measured_time_test))            
    print(f'Total accuracy: {correct}/{total} {correct/float(total)}')


def test_allreduce(numel, testtype):

    shape = (numel,)

    
    if testtype == torch.int64 or testtype == torch.int32:
        rand_torch = torch.randint(torch.iinfo(testtype).min/size, torch.iinfo(testtype).max/size,shape, dtype=testtype)
    else:
        rand_torch = torch.rand(shape, dtype=testtype)
    
    # for i in range(10):
    if True:
    
        # shape = (320001,)
        x = rand_torch.clone()

        dist.all_reduce(x, dist.ReduceOp.SUM)
        mpi.Barrier()

        try:
            np.testing.assert_allclose(x, rand_torch * size)
        except AssertionError as e:
            logger.debug("Test AllReduce failed")
            logger.debug(str(e))
        else:
            logger.debug("Test AllReduce finished!")

def test_broadcast(numel, testtype):
    shape = (numel,)

    # testtype = torch.float32
    if testtype == torch.int64 or testtype == torch.int32:
        rand_torch = torch.randint(torch.iinfo(testtype).min, torch.iinfo(testtype).max,shape, dtype=testtype)
        # rand_torch = torch.ones(shape, dtype=testtype)
    else:
        rand_torch = torch.rand(shape, dtype=testtype)
    
    # for i in range(10):
    if True:

        if rank == 1:
            x = rand_torch.clone()
        else:
            x = torch.ones(shape, dtype=testtype)
        
        dist.broadcast(x, 1)

        mpi.Barrier()

    # logger.debug('Tensor after broadcast: ' + str(x))
    # print('Tensor after broadcast: ' + str(x))
    try:
        np.testing.assert_allclose(x, rand_torch)
    except AssertionError as e:
        logger.debug("Test Broadcast failed")
        logger.debug(str(e))
    else:
        logger.debug("Test broadcast finished!")
            
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("-d", type=bool, default=None)


    parser.add_argument('-s', '--simulator', action='store_true',
                        default=False, help='Use simulation instead of '
                                            'hardware')
    parser.add_argument('-c', '--comms', choices=['udp', 'tcp', 'cyt_rdma'], default='tcp',
                        help='Run tests over specified communication backend')
    parser.add_argument('-i', '--host-file', type=str, help='Specify the file, where the host IPs are listed')
    parser.add_argument('-f', '--fpga-file', type=str, help='Specify the file, where the FPGA IPs are listed')
    parser.add_argument('-a','--master-address', type=str)
    parser.add_argument('-p','--master-port', type=str)


    args = parser.parse_args()

    if args.n == 1 and args.d == None :
        print("only one machine specified. Assuming Non distributed setup")
        args.d = False
    elif args.n > 1 and args.d == None:
        print("Assuming DDP setup")
        args.d = True


    host_file = args.host_file
    fpga_file = args.fpga_file
    comms = args.comms
    start_port = 5005
    
    global rank, size
    if args.master_address==None:
        args.master_address = "localhost"
    if args.master_port==None:
        args.master_port = "30505"
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    rank = mpi.Get_rank()
    size = mpi.Get_size()

    rxbufsize = 4096 * 1024

    if args.d:
        if not args.simulator:
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
            rxbufsize = 4096 * 1024
            ranks = [accl.Rank("127.0.0.1", 5500 + i, i, rxbufsize) for i in range(size)]

        logger.debug(f'Ranks: {ranks}')

        if args.comms == 'udp':
            design = accl.ACCLDesign.udp
        elif args.comms == 'tcp':
            design = accl.ACCLDesign.tcp
        elif args.comms == 'cyt_rdma': # and not simulator:
            design = accl.ACCLDesign.cyt_rdma
    

        mpi.Barrier()            
    
        accl.create_process_group(ranks, design, bufsize=rxbufsize, initialize=True, simulation=args.simulator)
        dist.init_process_group("ACCL", rank=rank, world_size=size)

    # dist.init_process_group("mpi", rank=rank, world_size=size)
        

    test_allreduce(256, torch.float32)
    test_broadcast(256, torch.float32)

    test_broadcast(162, torch.int32)
    # if args.d : dist.destroy_process_group()

    # sys.exit(0)
    
    device = 'cpu'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.CIFAR10(root='cifar10_data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='cifar10_data', train=False, download=True, transform=transform)

    if args.d : sampler = DistributedSampler
    else : sampler = lambda x : None
    
    loaders = {
        'train' : torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=32, 
                                              shuffle=False,
                                              num_workers=4,
                                              sampler=sampler(train_dataset)),
        'test'  : torch.utils.data.DataLoader(val_dataset, 
                                              batch_size=32, 
                                              shuffle=False,
                                              num_workers=4,
                                              sampler=sampler(val_dataset)),
    }

    model = models.resnet50(pretrained=True)
    
    if args.d : model = DDP(model, bucket_cap_mb=2, broadcast_buffers=True, find_unused_parameters=True)

    loss_func = nn.CrossEntropyLoss()   

    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 1

    mpi.Barrier()

    print("starting training")

    schedule = torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=10,
        repeat=3
    )

    
    # with torch.profiler.profile(
            # activities=[torch.profiler.ProfilerActivity.CPU],
            # schedule=schedule,
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./accl_log/profiler_log'),
            # record_shapes=True,
    # ) as p:

    if True:
    
        
        train(num_epochs, model, loaders, criterion)

        # test(num_epochs, model, loaders, criterion)

    # p.stop()

    # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))

    if args.d : dist.destroy_process_group()
