#!/bin/bash/mpiexec --n 2 --python test_n.py --device_index 0 2 --xclbin xccl_offload.xclbin 
# /*******************************************************************************
#  Copyright (C) 2021 Xilinx, Inc
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
# *******************************************************************************/

import sys
import numpy as np
sys.path.append('../../driver/pynq/') #append path
from cclo import *
import argparse
import random
from mpi4py import MPI
from queue import Queue
import time 

def configure_xccl(xclbin, board_idx, nbufs=16, bufsize=1024*1024):
    comm = MPI.COMM_WORLD
    rank_id = comm.Get_rank()
    size = comm.Get_size()

    local_alveo = pynq.Device.devices[board_idx]
    print("local_alveo: {}".format(local_alveo.name))
    ol=pynq.Overlay(xclbin, device=local_alveo)

    for i in ol.ip_dict:
        print(i)

    # print(ol.ip_dict)


    print("Allocating 1MB scratchpad memory")
    if local_alveo.name == 'xilinx_u250_gen3x16_xdma_shell_3_1':
        devicemem = ol.bank1
        rxbufmem = ol.bank1
        networkmem = ol.bank1
    elif local_alveo.name == 'xilinx_u250_xdma_201830_2':
        devicemem = ol.bank0
    elif local_alveo.name == 'xilinx_u280_xdma_201920_3':
        devicemem = ol.HBM0
        rxbufmem = [ol.HBM0, ol.HBM1, ol.HBM2, ol.HBM3, ol.HBM4, ol.HBM5 ]
        networkmem = ol.HBM6

    cclo           = ol.ccl_offload_0
    network_kernel = ol.network_krnl_0

    print("CCLO {} HWID: {} at {}".format(rank_id, hex(cclo.get_hwid()), hex(cclo.mmio.base_addr)))

    ip_network_str = []
    ip_network = []
    port = []
    ranks = []

    arp_addr = []
    for i in range (size):
        ip_network_str.append("10.1.212.{}".format(151+i))
        port.append(5001+i)
        ranks.append({"ip": ip_network_str[i], "port": port[i]})
        ip_network.append(int(ipaddress.IPv4Address(ip_network_str[i])))

        arp_addr.append(ip_network[i])
        

    print(f"CCLO {rank_id}: Configuring RX Buffers")
    cclo.setup_rx_buffers(nbufs, bufsize, rxbufmem)
    print(f"CCLO {rank_id}: Configuring a communicator")
    cclo.configure_communicator(ranks, rank_id)
    print(f"CCLO {rank_id}: Configuring network stack")

    #assign 64 MB network tx and rx buffer
    tx_buf_network = pynq.allocate((64*1024*1024,), dtype=np.int8, target=networkmem)
    rx_buf_network = pynq.allocate((64*1024*1024,), dtype=np.int8, target=networkmem)
    
    tx_buf_network.sync_to_device()
    rx_buf_network.sync_to_device()


    print(f"CCLO {rank_id}: Launch network kernel, ip {hex(ip_network[rank_id])}, board number {rank_id}, arp {hex(arp_addr[rank_id])}")
    network_kernel.call(ip_network[rank_id], rank_id, arp_addr[rank_id], tx_buf_network, rx_buf_network)


    #to synchronize the processes
    comm.barrier()

    # pdb.set_trace()
    print(f"CCLO {rank_id}: open port")
    cclo.open_port(0)

    #to synchronize the processes
    comm.barrier()

    print(f"CCLO {rank_id}: open connection")
    cclo.open_con(0)

    #to synchronize the processes
    comm.barrier()
    print(f"CCLO {rank_id}: Accelerator ready!")

    cclo.dump_communicator()

    return ol, cclo, devicemem



def test_sendrecv(bsize, to_from_fpga=True):
    print("========================================")
    print("SendRecv")
    print("========================================")
    naccel = 2
    # test sending from each cclo 
    tx_buf[:]=np.arange(0,bsize*1024)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    # to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank     == 0:
            for i in range(num_message):
                cclo.recv(0, rx_buf, 1, tag=i, to_fpga=to_from_fpga)
        elif rank   == 1 :
            for i in range(num_message):
                cclo.send(0, tx_buf, 0, tag=i, from_fpga=to_from_fpga)
                # cclo.dump_rx_buffers(nbufs=16)
                # if not to_from_fpga:
                #     if not (rx_buf == tx_buf).all():
                #         print("Message {} Send/Recv {} -> {} failed".format(i, 0, 1))
                #     else:
                #         print("Message {} Send/Recv {} -> {} succeeded".format(i, 0, 1))
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))

def test_bcast(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Broadcast")
    print("========================================")
    tx_buf[:]=np.arange(0,bsize*1024)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    # to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.bcast(0, tx_buf, 0, sw=False, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # print("rank 0 finishes")
        else :
            for i in range(num_message):
                cclo.bcast(0, rx_buf, 0, sw=False, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # cclo.dump_rx_buffers(nbufs=16)
                # print(f"rank {rank} finishes")
                # if not to_from_fpga:
                #     if not (rx_buf == tx_buf).all():
                #         print("Rank {} Message {} Bcast failed".format(rank, i))
                #     else:
                #         print("Rank {} Message {} Bcast succeeded".format(rank, i))
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = (naccel-1)*num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))

def test_ring_reduce(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Ring-reduce")
    print("========================================")
    tx_buf[0]=0
    rx_buf[0]=0
    # to_from_fpga = True
    count = tx_buf.nbytes
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank     == 0:
            for i in range(num_message):
                tx_buf[0]=i+1
                cclo.reduce(0, tx_buf, rx_buf, count, 0, func=2, sw=False, shift=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # cclo.dump_rx_buffers_spares(nbufs=16)
                # if rx_buf[0]== (tx_buf[0]*naccel):
                #     print("Message {} Ring-reduce succeeded".format(i))
                # else:
                #     print("Message {} Ring-reduce failed, expected {} got {}".format(i, (tx_buf[0]*naccel), rx_buf[0]))
        else :
            for i in range(num_message):
                # cclo.dump_rx_buffers(nbufs=16)
                tx_buf[0]=i+1
                cclo.reduce(0, tx_buf, rx_buf, count, 0, func=2, sw=False, shift=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # print(f"Finishes rank {rank}")
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = (naccel)*num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))

def test_ring_all_reduce(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Ring_All_Reduce")
    print("========================================")
    tx_buf.fill(5)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    result=tx_buf*naccel
    # print(tx_buf)
    # print(result)
    count = tx_buf.nbytes
    # to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.allreduce(0, tx_buf, rx_buf, count, func=2, fused=True, sw=False, ring=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # cclo.dump_rx_buffers_spares(nbufs=16)
                # print("rank 0 finishes")
        else :
            for i in range(num_message):
                cclo.allreduce(0, tx_buf, rx_buf, count, func=2, fused=True, sw=False, ring=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # cclo.dump_rx_buffers(nbufs=16)
                # print(f"rank {rank} finishes")
                # if not to_from_fpga:
                #     if not (rx_buf == result).all():
                #         print("Rank {} Message {} Ring_All_Reduce failed".format(rank, i))
                #     else:
                #         print("Rank {} Message {} Ring_All_Reduce succeeded".format(rank, i))
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = (naccel)*num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))

def test_scatter(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Scatter")
    print("========================================")
    tx_buf[:]=np.arange(0,bsize*1024)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    count = tx_buf.nbytes//naccel
    print(f"Scatter Send count:{count}")
    # to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.scatter(0, tx_buf, rx_buf, count, 0,  sw=False, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # print("rank 0 finishes")
        else :
            for i in range(num_message):
                cclo.scatter(0, tx_buf, rx_buf, count, 0,  sw=False, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # cclo.dump_rx_buffers(nbufs=16)
                # print(f"rank {rank} finishes")
                # if not to_from_fpga:
                #     if not (rx_buf[0:count-1] == tx_buf[(count*rank):(count*(rank+1)-1)]).all():
                #         print("Rank {} Message {} Scatter failed".format(rank, i))
                #     else:
                #         print("Rank {} Message {} Scatter succeeded".format(rank, i))
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))



def test_gather(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Gather")
    print("========================================")
    tx_buf[:]=np.arange(0,bsize*1024)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    count = tx_buf.nbytes//naccel
    print(f"Gather Send count:{count}")
    # to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.gather(0, tx_buf, rx_buf, count, 0, sw=False, shift=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # print("rank 0 finishes")
                # if not to_from_fpga:
                #     if not (tx_buf[0:count-1] == rx_buf[(count*rank):(count*(rank+1)-1)]).all():
                #         print("Rank {} Message {} Gather failed".format(rank, i))
                #     else:
                #         print("Rank {} Message {} Gather succeeded".format(rank, i))
        else :
            for i in range(num_message):
                cclo.gather(0, tx_buf, rx_buf, count, 0, sw=False, shift=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # cclo.dump_rx_buffers(nbufs=16)
                # print(f"rank {rank} finishes")
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))

def test_allgather(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("All_Gather")
    print("========================================")
    tx_buf[:]=np.arange(0,bsize*1024)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    count = tx_buf.nbytes//naccel
    print(f"All_Gather Send count:{count}")
    # to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.allgather(0, tx_buf, rx_buf, count, fused=True, sw=False, ring=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # print("rank 0 finishes")
                # if not to_from_fpga:
                #     if not (tx_buf[0:count-1] == rx_buf[(count*rank):(count*(rank+1)-1)]).all():
                #         print("Rank {} Message {} All_Gather failed".format(rank, i))
                #     else:
                #         print("Rank {} Message {} All_Gather succeeded".format(rank, i))
        else :
            for i in range(num_message):
                cclo.allgather(0, tx_buf, rx_buf, count, fused=True, sw=False, ring=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # cclo.dump_rx_buffers(nbufs=16)
                # print(f"rank {rank} finishes")
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = (naccel)*num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))


parser = argparse.ArgumentParser(description='Tests for MPI collectives offload with UDP (VNx) backend')
parser.add_argument('--xclbin',         type=str, default=None,             help='Accelerator image file (xclbin)', required=True)
parser.add_argument('--device_index', type=int, default=1, help='Card index')
parser.add_argument('--nruns',          type=int, default=1,                help='How many times to run each test')
parser.add_argument('--bsize', type=int, default=1024, help='How many KB per buffer')
parser.add_argument('--sendrecv', action='store_true', default=False, help='Run send/recv test')
parser.add_argument('--bcast', action='store_true', default=False, help='Run bcast test')
parser.add_argument('--scatter', action='store_true', default=False, help='Run scatter test')
parser.add_argument('--gather', action='store_true', default=False, help='Run gather test')
parser.add_argument('--allgather', action='store_true', default=False, help='Run allgather test')
parser.add_argument('--reduce', action='store_true', default=False, help='Run reduce test')
parser.add_argument('--allreduce', action='store_true', default=False, help='Run allreduce test')
parser.add_argument('--sum', action='store_true', default=False, help='Run fp/dp/i32/i64 test')
parser.add_argument('--fused', action='store_true', default=False, help='For all-* collectives, run the fused implementation')
parser.add_argument('--dump_rx_regs', type=int, default=0, help='Print RX regs of specified ')

args = parser.parse_args()

comm         = MPI.COMM_WORLD
args.naccel  = MPI.COMM_WORLD.Get_size()
rank         = comm.Get_rank()
# assert(args.naccel == len(args.device_index))
# pdb.set_trace()

if __name__ == "__main__":    
    try:
        #configure FPGA and CCLO cores with the default 16 RX buffers of 16KB each
        ol, cclo, devicemem = configure_xccl(args.xclbin, args.device_index)

        tx_buf = pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem)
        rx_buf = pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem)
        print(f"message size {args.bsize} KB")

        print(f"CCLO {rank}: rx_buf {hex(rx_buf.device_address)}")
        print(f"CCLO {rank}: tx_buf {hex(tx_buf.device_address)}")

        if rank == 0:
            # if args.dump_rx_regs >= 0:
            cclo.dump_rx_buffers_spares(nbufs=16)

        cclo.set_timeout(10_000_000)

        #to synchronize the processes
        comm.barrier()

        if args.sendrecv:
            for i in range(args.nruns):
                test_sendrecv(args.bsize)
                test_sendrecv(args.bsize, False)

        if args.bcast:
            for i in range(args.nruns):
                test_bcast(args.bsize, args.naccel)
                test_bcast(args.bsize, args.naccel, False)

        if args.reduce:
            for i in range(args.nruns):
                test_ring_reduce(args.bsize, args.naccel)
                test_ring_reduce(args.bsize, args.naccel, False)

        if args.scatter:
            for i in range(args.nruns):
                test_scatter(args.bsize, args.naccel)
                test_scatter(args.bsize, args.naccel, False)

        if args.gather:
            for i in range(args.nruns):
                test_gather(args.bsize, args.naccel)
                test_gather(args.bsize, args.naccel, False)

        if args.allreduce:
            for i in range(args.nruns):
                test_ring_all_reduce(args.bsize, args.naccel)
                test_ring_all_reduce(args.bsize, args.naccel,False)

        if args.allgather:
            for i in range(args.nruns):
                test_allgather(args.bsize, args.naccel)
                test_allgather(args.bsize, args.naccel,False)

    except KeyboardInterrupt:
        print("CTR^C")
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)
    cclo.deinit()
   