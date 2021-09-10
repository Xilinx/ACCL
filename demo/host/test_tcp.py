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
sys.path.append('../../driver/pynq/') #append path
from cclo import *
import numpy as np
import argparse
import random
from mpi4py import MPI
from queue import Queue
import time 
import ipaddress

def configure_xccl(xclbin, board_idx, nbufs=16, bufsize=1024*1024):
    comm = MPI.COMM_WORLD
    rank_id = comm.Get_rank()
    size = comm.Get_size()

    local_alveo = pynq.Device.devices[board_idx]
    print("local_alveo: {}".format(local_alveo.name))
    if   local_alveo.name == 'xilinx_u250_gen3x16_xdma_shell_3_1':
        xclbin = "../build/tcp_u250/ccl_offload.xclbin"
    elif local_alveo.name == 'xilinx_u280_xdma_201920_3':
        xclbin = "../../../ccl_offload.xclbin"
    ol=pynq.Overlay(xclbin, device=local_alveo)

    global args
    args.board_instance = local_alveo.name

    print("Allocating scratchpad memory")
    if local_alveo.name == 'xilinx_u250_gen3x16_xdma_shell_3_1':
        devicemem   = ol.bank1
        rxbufmem    = ol.bank1
        networkmem  = ol.bank1
    elif local_alveo.name == 'xilinx_u250_xdma_201830_2':
        devicemem   = ol.bank0
    elif local_alveo.name == 'xilinx_u280_xdma_201920_3':
        devicemem   = ol.HBM0
        rxbufmem    = [ol.__getattr__(f"HBM{j}") for j in range(1, args.num_banks) ] 
        if args.use_tcp:
            networkmem  = ol.HBM6

    cclo            = ol.ccl_offload_0


    print("CCLO {} HWID: {} at {}".format(rank_id, hex(cclo.get_hwid()), hex(cclo.mmio.base_addr)))
    

    ranks       = []
    ip_network  = []
    for i in range (size):
        ip_network_str  = "10.1.212.{}".format(151+i)
        port            = 5001+i
        ranks.append({"ip": ip_network_str, "port": port})
        ip_network.append(int(ipaddress.IPv4Address(ip_network_str)))


    print("set transport protocol")
    if args.use_tcp :
        cclo.use_tcp()
    else:
        cclo.use_udp()

    print(f"CCLO {rank_id}: Configuring RX Buffers")
    cclo.setup_rx_buffers(nbufs, bufsize, rxbufmem)
    print(f"CCLO {rank_id}: Configuring a communicator")
    cclo.configure_communicator(ranks, rank_id)
    print(f"CCLO {rank_id}: Configuring network stack")

    if args.use_tcp:
        network_kernel  = ol.network_krnl_0
        #assign 64 MB network tx and rx buffer
        tx_buf_network = pynq.allocate((64*1024*1024,), dtype=np.int8, target=networkmem)
        rx_buf_network = pynq.allocate((64*1024*1024,), dtype=np.int8, target=networkmem)
        
        tx_buf_network.sync_to_device()
        rx_buf_network.sync_to_device()


        print(f"CCLO {rank_id}: Launch network kernel, ip {hex(ip_network[rank_id])}, board number {rank_id}")
        network_kernel.start_sw(ip_network[rank_id], rank_id, ip_network[rank_id], tx_buf_network, rx_buf_network)
    else:
        import setup_vnx
        print(f"CCLO {rank_id}: about to configure network kernel")
        setup_vnx.configure_vnx(ol, rank_id, ranks)
        print(f"CCLO {rank_id}: configured network kernel")

    #to synchronize the processes
    comm.barrier()
    if args.use_tcp :
        # pdb.set_trace()
        print(f"CCLO {rank_id}: open port")
        cclo.open_port(0)

        #to synchronize the processes
        comm.barrier()
        #flag = True
        #while(flag):
        #    try:
        #        print(f"CCLO {rank_id}: open connection")
        cclo.open_con(0)
        #        flag = False
        #    except Exception as e:
        #        print(e)
    comm.barrier()

    #to synchronize the processes
    if(rank_id ==0 and args.debug):
        input("Hit any button to continue..")
    comm.barrier()
    print(f"CCLO {rank_id}: Accelerator ready!")

    if local_alveo.name == 'xilinx_u250_gen3x16_xdma_shell_3_1':
        cclo.set_timeout(500_000)
    elif local_alveo.name == 'xilinx_u280_xdma_201920_3':
        cclo.set_timeout(10_000_000)

    cclo.dump_communicator()
    
    return ol, cclo, devicemem



def test_sendrecv(bsize,  to_from_fpga=True):
    print("========================================")
    print("SendRecv 1 -> 0")
    print("========================================")
    naccel = 2
    # test sending from each cclo 
    tx_buf[:bsize*1024]=np.arange(0,bsize*1024)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    # to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank     == 0:
            for i in range(num_message):
                cclo.recv(0, rx_buf[:bsize*1024], src=1, tag=i, to_fpga=to_from_fpga)
        elif rank   == 1 :
            for i in range(num_message):
                cclo.send(0, tx_buf[:bsize*1024], dst=0, tag=i, from_fpga=to_from_fpga)
                if args.debug and to_from_fpga:
                    cclo.dump_rx_buffers_spares()
                    if not to_from_fpga:
                        if not (rx_buf[:bsize*1024] == tx_buf[:bsize*1024]).all():
                            print("Message {} Send/Recv {} -> {} failed".format(i, 0, 1))
                        else:
                            print("Message {} Send/Recv {} -> {} succeeded".format(i, 0, 1))
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    return duration_us, throughput_gbps

def test_bcast(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Broadcast (0)")
    print("========================================")
    tx_buf[:bsize*1024]=np.arange(0,bsize*1024)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    # to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.bcast(0, tx_buf[:bsize*1024], root=0, sw=False, rr=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                # print("rank 0 finishes")
        else :
            for i in range(num_message):
                cclo.bcast(0, rx_buf[:bsize*1024], root=0, sw=False, rr=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    cclo.dump_rx_buffers_spares()
                    print(f"rank {rank} finishes")
                    if not to_from_fpga:
                        if not (rx_buf[:bsize*1024] == tx_buf[:bsize*1024]).all():
                            print("Rank {} Message {} Bcast failed".format(rank, i))
                        else:
                            print("Rank {} Message {} Bcast succeeded".format(rank, i))
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = (naccel-1)*num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    return duration_us, throughput_gbps


def test_ring_reduce(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Ring-reduce (0)")
    print("========================================")
    tx_buf[0]=0
    rx_buf[0]=0
    # to_from_fpga = True
    count = tx_buf[:bsize*1024].nbytes
    niter = 1
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank     == 0:
            for i in range(num_message):
                tx_buf[0]=i+1
                cclo.reduce(0, tx_buf[:bsize*1024], rx_buf[:bsize*1024], count, root=0, func=2, sw=False, shift=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)

                if args.debug and to_from_fpga:
                    cclo.dump_rx_buffers_spares()
                    if rx_buf[0]== (tx_buf[0]*naccel):
                        print("Message {} Ring-reduce succeeded".format(i))
                    else:
                        print("Message {} Ring-reduce failed, expected {} got {}".format(i, (tx_buf[0]*naccel), rx_buf[0]))
        else :
            for i in range(num_message):
                # cclo.dump_rx_buffers_spares(nbufs=16)
                tx_buf[0]=i+1
                cclo.reduce(0, tx_buf, rx_buf, count, root=0, func=2, sw=False, shift=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    print(f"Finishes rank {rank}")
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = (naccel)*num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    return duration_us, throughput_gbps

def test_ring_all_reduce(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Ring_All_Reduce")
    print("========================================")
    tx_buf.fill(5)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    result=tx_buf*naccel
    # print(tx_buf)
    # print(result)
    count = tx_buf[:bsize*1024].nbytes
    # to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.allreduce(0, tx_buf, rx_buf, count, func=2, fused=True, sw=False, ring=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    cclo.dump_rx_buffers_spares()
                    print("rank 0 finishes")
        else :
            for i in range(num_message):
                cclo.allreduce(0, tx_buf, rx_buf, count, func=2, fused=True, sw=False, ring=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    cclo.dump_rx_buffers_spares()
                    print(f"rank {rank} finishes")
                    if not to_from_fpga:
                        if not (rx_buf == result).all():
                            print("Rank {} Message {} Ring_All_Reduce failed".format(rank, i))
                        else:
                            print("Rank {} Message {} Ring_All_Reduce succeeded".format(rank, i))
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = (naccel)*num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    return duration_us, throughput_gbps
    

def test_scatter(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Scatter (0)")
    print("========================================")
    tx_buf[:bsize*1024]=np.arange(0,bsize*1024)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    count = tx_buf[:bsize*1024].nbytes//naccel
    print(f"Scatter Send count:{count}")
    #to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.scatter(0, tx_buf, rx_buf, count, root=0,  sw=False, rr=False, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    print("rank 0 finishes")
        else :
            for i in range(num_message):
                cclo.scatter(0, tx_buf, rx_buf, count, root=0,  sw=False, rr=False, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    cclo.dump_rx_buffers_spares()
                    print(f"rank {rank} finishes")
                    if not to_from_fpga:
                        if not (rx_buf[0:count-1] == tx_buf[(count*rank):(count*(rank+1)-1)]).all():
                            print("Rank {} Message {} Scatter failed".format(rank, i))
                        else:
                            print("Rank {} Message {} Scatter succeeded".format(rank, i))
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    return duration_us, throughput_gbps



def test_gather(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("Gather (0)")
    print("========================================")
    tx_buf[:bsize*1024]=np.arange(0,bsize*1024)
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    count = tx_buf[:bsize*1024].nbytes//naccel
    print(f"Gather Send count:{count}")
    #to_from_fpga = True
    niter = 10
    num_message=1
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.gather(0, tx_buf, rx_buf, count, root=0, sw=False, shift=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    print("rank 0 finishes")
                    if not to_from_fpga:
                        if not (tx_buf[0:count-1] == rx_buf[(count*rank):(count*(rank+1)-1)]).all():
                            print("Rank {} Message {} Gather failed".format(rank, i))
                        else:
                            print("Rank {} Message {} Gather succeeded".format(rank, i))
        else :
            for i in range(num_message):
                cclo.gather(0, tx_buf, rx_buf, count, root=0, sw=False, shift=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    cclo.dump_rx_buffers_spares()
                    print(f"rank {rank} finishes")
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    return duration_us, throughput_gbps


def test_allgather(bsize, naccel, to_from_fpga=True):
    print("========================================")
    print("All_Gather")
    print("========================================")
    rx_buf[:]=np.zeros(rx_buf.shape)# clear rx buffers
    count = tx_buf[:bsize*1024].nbytes//naccel
    tx_buf[:count]=np.arange(count*(rank-1),count*rank)
    print(f"All_Gather Send count:{count}")
    #to_from_fpga = True
    niter = 10
    num_message=1
    tx_buf[:bsize*1024].sync_to_device()
    comm.barrier()
    start = time.perf_counter()
    for j in range (niter):
        if rank == 0:
            for i in range(num_message):
                cclo.allgather(0, tx_buf, rx_buf, count, fused=True, sw=False, ring=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    print("rank 0 finishes")
                    if not to_from_fpga:
                        if not (tx_buf[0:count-1] == rx_buf[(count*rank):(count*(rank+1)-1)]).all():
                            print("Rank {} Message {} All_Gather failed".format(rank, i))
                        else:
                            print("Rank {} Message {} All_Gather succeeded".format(rank, i))
        else :
            for i in range(num_message):
                cclo.allgather(0, tx_buf, rx_buf, count, fused=True, sw=False, ring=True, from_fpga=to_from_fpga, to_fpga=to_from_fpga, run_async=False)
                if args.debug and to_from_fpga:
                    cclo.dump_rx_buffers_spares()
                    print(f"rank {rank} finishes")
        #to synchronize the processes
        comm.barrier()
    end = time.perf_counter()        
    duration_us = ((end - start)/niter)*1000000
    throughput_gbps = (naccel)*num_message*bsize*1024*8/(duration_us*1000)
    if to_from_fpga:
        print("Size[KB],{},Num device,{},*Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    else:
        print("FullPath,Size[KB],{},Num device,{},&Duration[us],{},throughput[gbps],{}".format(bsize,naccel, duration_us, throughput_gbps))
    return duration_us, throughput_gbps

parser = argparse.ArgumentParser(description='Tests for MPI collectives offload with UDP (VNx) backend')
parser.add_argument('--xclbin',         type=str, default=None,                             help='Accelerator image file (xclbin)', required=True)
parser.add_argument('--device_index',   type=int, default=1,                                help='Card index')
parser.add_argument('--nruns',          type=int, default=1,                                help='How many times to run each test')
parser.add_argument('--nbufs',          type=int, default=16,                               help='How many times to run each test')
parser.add_argument('--bsize',          type=int, default=1024,             nargs="+" ,    help='How many KB per buffer')
parser.add_argument('--segment_size',   type=int, default=1024,             nargs="+" ,    help='How many KB per spare_buffer')
parser.add_argument('--dump_rx_regs',   type=int, default=0,    	                        help='Print RX regs of specified ')
parser.add_argument('--num_banks',      type=int, default=6,                                help='for U280 specifies how many memory banks to use per CCL_Offload instance')
parser.add_argument('--sendrecv',       action='store_true', default=False ,                help='Run send/recv test')
parser.add_argument('--bcast',          action='store_true', default=False ,                help='Run bcast test')
parser.add_argument('--scatter',        action='store_true', default=False ,                help='Run scatter test')
parser.add_argument('--gather',         action='store_true', default=False ,                help='Run gather test')
parser.add_argument('--allgather',      action='store_true', default=False ,                help='Run allgather test')
parser.add_argument('--reduce',         action='store_true', default=False ,                help='Run reduce test')
parser.add_argument('--allreduce',      action='store_true', default=False ,                help='Run allreduce test')
parser.add_argument('--sum',            action='store_true', default=False ,                help='Run fp/dp/i32/i64 test')
parser.add_argument('--fused',          action='store_true', default=False ,                help='For all-* collectives, run the fused implementation')
parser.add_argument('--debug',          action='store_true', default=False ,                help='activate tests')
parser.add_argument('--use_tcp',        action='store_true', default=False ,                help='use tcp stack')
parser.add_argument('--experiment',     type=str,            default="test",                help='experiment meaningful name')
args = parser.parse_args()

comm         = MPI.COMM_WORLD
args.naccel  = MPI.COMM_WORLD.Get_size()
rank         = comm.Get_rank()
# assert(args.naccel == len(args.device_index))
# pdb.set_trace()

if __name__ == "__main__":    
    try:
        #configure FPGA and CCLO cores with the default 16 RX buffers of 16KB each
        ol, cclo, devicemem = configure_xccl(args.xclbin, args.device_index, nbufs=args.nbufs ,bufsize=max(args.segment_size)*1024 )

        tx_buf = pynq.allocate((max(args.bsize)*1024,), dtype=np.int8, target=devicemem)
        rx_buf = pynq.allocate((max(args.bsize)*1024,), dtype=np.int8, target=devicemem)

        print(f"CCLO {rank}: rx_buf {hex(rx_buf.device_address)}")
        print(f"CCLO {rank}: tx_buf {hex(tx_buf.device_address)}")

        if rank == 0:
            # if args.dump_rx_regs >= 0:
            cclo.dump_rx_buffers_spares(nbufs=32)
        
        csv_file = open(f"../measurements/accl/{args.experiment}_rank{rank}.csv", "a+", newline="") 
        import csv
        csv_writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["experiment", "board_instance", "number of nodes", "rank id", "number of banks", "buffer size[KB]", "segment_size[KB]", "collective name", "execution_time[us]", "throughput[Gbps]","execution_time_fullpath[us]", "throughput_fullpath[Gbps]"])

        
        cclo.set_max_dma_transaction_flight(18)
        cclo.set_delay(0)
        for segment_size in args.segment_size:
            #change dma_transaction size
            cclo.set_dma_transaction_size(segment_size*1024)
            print(f"segment size size {segment_size} KB")
            for bsize in args.bsize:
                print(f"message size {bsize} KB")
                #to synchronize the processes
                comm.barrier()
                if args.sendrecv:
                    for i in range(args.nruns):
                        duration_us, throughput_gbps        = test_sendrecv(bsize,)
                        duration_us_fp, throughput_gbps_fp  = test_sendrecv(bsize, False)

                        csv_writer.writerow([args.experiment, args.board_instance, args.naccel, rank, args.num_banks, bsize, segment_size, "Send/recv", duration_us, throughput_gbps, duration_us_fp, throughput_gbps_fp])

                if args.bcast:
                    for i in range(args.nruns):
                        duration_us, throughput_gbps        = test_bcast(bsize, args.naccel)
                        duration_us_fp, throughput_gbps_fp  = test_bcast(bsize, args.naccel, False)

                        csv_writer.writerow([args.experiment, args.board_instance, args.naccel, rank, args.num_banks, bsize, segment_size, "Broadcast", duration_us, throughput_gbps, duration_us_fp, throughput_gbps_fp])

                if args.reduce:
                    for i in range(args.nruns):
                        duration_us, throughput_gbps        = test_ring_reduce(bsize, args.naccel)
                        duration_us_fp, throughput_gbps_fp  = test_ring_reduce(bsize, args.naccel, False)

                        csv_writer.writerow([args.experiment, args.board_instance, args.naccel, rank, args.num_banks, bsize, segment_size, "Reduce", duration_us, throughput_gbps, duration_us_fp, throughput_gbps_fp])
                        #if i % 5 == 0:
                        #    from time import sleep
                        #    sleep(rank*0.1)
                        #    cclo.dump_rx_buffers_spares()
                if args.scatter:
                    for i in range(args.nruns):
                        duration_us, throughput_gbps        = test_scatter(bsize, args.naccel)
                        duration_us_fp, throughput_gbps_fp  = test_scatter(bsize, args.naccel, False)

                        csv_writer.writerow([args.experiment, args.board_instance, args.naccel, rank, args.num_banks, bsize, segment_size, "Scatter", duration_us, throughput_gbps, duration_us_fp, throughput_gbps_fp])

                if args.gather:
                    for i in range(args.nruns):
                        duration_us, throughput_gbps        = test_gather(bsize, args.naccel)
                        duration_us_fp, throughput_gbps_fp  = test_gather(bsize, args.naccel, False)

                        csv_writer.writerow([args.experiment, args.board_instance, args.naccel, rank, args.num_banks, bsize, segment_size, "Gather", duration_us, throughput_gbps, duration_us_fp, throughput_gbps_fp])

                if args.allreduce:
                    for i in range(args.nruns):
                        duration_us, throughput_gbps        = test_ring_all_reduce(bsize, args.naccel)
                        duration_us_fp, throughput_gbps_fp  = test_ring_all_reduce(bsize, args.naccel, False)

                        csv_writer.writerow([args.experiment, args.board_instance, args.naccel, rank, args.num_banks, bsize, segment_size, "Allreduce", duration_us, throughput_gbps, duration_us_fp, throughput_gbps_fp])


                if args.allgather:
                    for i in range(args.nruns):
                        duration_us, throughput_gbps        = test_allgather(bsize, args.naccel)
                        duration_us_fp, throughput_gbps_fp  = test_allgather(bsize, args.naccel, False)

                        csv_writer.writerow([args.experiment, args.board_instance, args.naccel, rank, args.num_banks, bsize, segment_size, "Allgather", duration_us, throughput_gbps, duration_us_fp, throughput_gbps_fp])
                csv_file.flush()
    except KeyboardInterrupt:
        print("CTR^C")
        print("Rank", rank)
        from time import sleep
        sleep(rank)
        cclo.dump_rx_buffers_spares()
        exit(1)
    except Exception as e:
        print("Rank", rank)
        print(e)
        import traceback
        from time import sleep
        sleep(rank)
        traceback.print_tb(e.__traceback__)
        cclo.dump_rx_buffers_spares()
        exit(1)
    cclo.deinit()
   