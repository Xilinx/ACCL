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
sys.path.append('../../driver/pynq/')
from cclo import *
import json 
import argparse
import random
from queue import Queue
import threading
import time

def configure_accl(xclbin, board_idx, nbufs=16, bufsize=16*1024):
    global ext_arithm  
    local_alveo = pynq.Device.devices[board_idx]
    ol=pynq.Overlay(xclbin, device=local_alveo)

    print("Allocating 1MB scratchpad memory")
    if local_alveo.name == 'xilinx_u250_xdma_201830_2':
        devicemem = [ol.__getattr__(f"bank{i}") for i in range(args.naccel)]
    elif local_alveo.name == 'xilinx_u280_xdma_201920_3':
        devicemem = [ol.HBM0                    for _ in range(args.naccel)]

    cclo = [ol.__getattr__(f"ccl_offload_{i}")   for i in range(args.naccel)]
    ext_arithm = [ol.__getattr__(f"external_reduce_arith_{i}")   for i in range(args.naccel)]
    for i in range(len(cclo)):
        print("CCLO {} HWID: {} at {}".format(i, hex(cclo[i].get_hwid()), hex(cclo[i].mmio.base_addr)))

    ranks = [{"ip": "127.0.0.1", "port": i} for i in range(args.naccel)]

    for i in range(args.naccel):
        print("CCLO ",i)
        print("Configuring RX Buffers")
        cclo[i].setup_rx_buffers(nbufs, bufsize, devicemem[i])
        print("Configuring a communicator")
        cclo[i].configure_communicator(ranks, i)

    print("Accelerator ready!")

    return ol, cclo, devicemem

def test_self_sendrecv():
    print("========================================")
    print("Self Send/Recv ")
    print("========================================")
    for j in range(args.naccel):
        src_rank = j
        dst_rank = j
        tag      = 5+10*j
        senddata = np.random.randint(100, size=tx_buf[src_rank].shape)
        tx_buf[src_rank][:]=senddata
        cclo_inst[src_rank].send(0, tx_buf[src_rank], dst_rank, tag)

        cclo_inst[dst_rank].recv(0, rx_buf[dst_rank], src_rank, tag)

        recvdata = rx_buf[dst_rank]
        if (recvdata == senddata).all():
            print("Send/Recv {} -> {} succeeded".format(src_rank, dst_rank))
        else:
            diff = (recvdata != senddata)
            firstdiff = np.argmax(diff)
            ndiffs = diff.sum()
            print("Send/Recv {} -> {} failed, {} bytes different starting at {}".format(src_rank, dst_rank, ndiffs, firstdiff))
            print(f"Senddata: {senddata} != {recvdata} Recvdata ")
            cclo_inst[dst_rank].dump_communicator()
            cclo_inst[src_rank].dump_communicator()
            cclo_inst[dst_rank].dump_rx_buffers_spares()
            import pdb; pdb.set_trace()


def test_sendrecv():
    # test sending from each cclo_inst to each other cclo_inst
    queues = [[Queue() for i in range(args.naccel)] for j in range(args.naccel)]
    src_rank = 0
    dst_rank = 1
    #print(f"{hex(tag)} tag sent to {dst_rank}")
    senddata = np.random.randint(100, size=tx_buf[src_rank].shape)
    queues[src_rank][dst_rank].put(senddata)
    tx_buf[src_rank][:]=senddata
    cclo_inst[src_rank].send(0, tx_buf[src_rank], dst_rank)
    cclo_inst[dst_rank].recv(0, rx_buf[dst_rank], src_rank)
    exp_recvdata = queues[src_rank][dst_rank].get()
    recvdata = rx_buf[dst_rank]
    if (recvdata == exp_recvdata).all():
        print("Send/Recv {} -> {} succeeded".format(src_rank, dst_rank))
    else:
        diff        = np.where(recvdata != exp_recvdata)
        firstdiff   = np.min(diff)
        ndiffs      = diff[0].size 
        print("Send/Recv {} -> {} failed, {} bytes different starting at {}".format(src_rank, dst_rank, ndiffs, firstdiff))
        print(f"Senddata: {senddata} != {recvdata} Recvdata ")
        cclo_inst[dst_rank].dump_communicator()
        cclo_inst[src_rank].dump_communicator()
        cclo_inst[dst_rank].dump_rx_buffers_spares()
        import pdb; pdb.set_trace()
    # scenario 2: send to everyone and recv 
    print("========================================")
    print("Send/Recv Scenario 1")
    print("========================================")
    from time import sleep
    for i in range(args.naccel):
        for j in range(args.naccel):
            src_rank = i
            dst_rank = (i+j+1)%args.naccel
            tag = i+5+10*j
            #print(f"{hex(tag)} tag sent to {dst_rank}")
            senddata = np.random.randint(100, size=tx_buf[src_rank].shape)
            queues[i][j].put(senddata)
            tx_buf[src_rank][:]=senddata
            cclo_inst[src_rank].send(0, tx_buf[src_rank], dst_rank, tag)
            cclo_inst[dst_rank].recv(0, rx_buf[dst_rank], src_rank, tag)
            exp_recvdata = queues[i][j].get()
            recvdata = rx_buf[dst_rank]
            if (recvdata == exp_recvdata).all():
                print("Send/Recv {} -> {} succeeded".format(src_rank, dst_rank))
            else:
                diff        = np.where(recvdata != exp_recvdata)
                firstdiff   = np.min(diff)
                ndiffs      = diff[0].size 
                print("Send/Recv {} -> {} failed, {} bytes different starting at {}".format(src_rank, dst_rank, ndiffs, firstdiff))
                print(f"Senddata: {senddata} != {recvdata} Recvdata ")
                cclo_inst[dst_rank].dump_communicator()
                cclo_inst[src_rank].dump_communicator()
                cclo_inst[dst_rank].dump_rx_buffers_spares()
                import pdb; pdb.set_trace()
    # scenario 1: for each instance, send multiple, then recv multiple at the other instances
    print("========================================")
    print("Send/Recv Scenario 2")
    print("========================================")
    for i in range(args.naccel):
        for j in range(args.naccel):
            src_rank = i
            dst_rank = (i+j+1)%args.naccel
            tag = i+5+10*j
            senddata = np.random.randint(100, size=tx_buf[src_rank].shape)
            tx_buf[src_rank][:]=senddata
            cclo_inst[src_rank].send(0, tx_buf[src_rank], dst_rank, tag)
            queues[i][j].put(senddata)
            
        for j in range(args.naccel):
            src_rank = i
            dst_rank = (i+j+1)%args.naccel
            tag = i+5+10*j
            exp_recvdata = queues[i][j].get()

            cclo_inst[dst_rank].recv(0, rx_buf[dst_rank], src_rank, tag)
        
            recvdata = rx_buf[dst_rank]
            if (recvdata == exp_recvdata).all():
                print("Send/Recv {} -> {} succeeded".format(src_rank, dst_rank))
            else:
                diff        = np.where(recvdata != exp_recvdata)
                firstdiff   = np.min(diff)
                ndiffs      = diff[0].size 
                print("Send/Recv {} -> {} failed, {} bytes different starting at {}".format(src_rank, dst_rank, ndiffs, firstdiff))
                print(f"Senddata: {exp_recvdata} != {recvdata} Recvdata ")
                cclo_inst[dst_rank].dump_communicator()
                cclo_inst[src_rank].dump_communicator()
                cclo_inst[dst_rank].dump_rx_buffers_spares()

                import pdb; pdb.set_trace()
    # scenario 3: send everything, recv everything
    print("========================================")
    print("Send/Recv Scenario 2")
    print("========================================")
    for i in range(args.naccel):
        for j in range(args.naccel):
            src_rank = i
            dst_rank = (i+j+1)%args.naccel
            tag = i+5+10*j
            senddata = np.random.randint(100, size=tx_buf[src_rank].shape)
            queues[i][j].put(senddata)
            tx_buf[src_rank][:]=senddata
            cclo_inst[src_rank].send(0, tx_buf[src_rank], dst_rank, tag)
            #print(f"sent {src_rank}")
            #cclo_inst[src_rank].dump_communicator()


    for i in range(args.naccel):
        for j in range(args.naccel):
            src_rank = i
            dst_rank = (i+j+1)%args.naccel
            tag = i+5+10*j
            exp_recvdata = queues[i][j].get()

            #print(f"before recv {dst_rank}")
            #cclo_inst[src_rank].dump_communicator()
            ##cclo_inst[dst_rank].dump_communicator()
            ##cclo_inst[dst_rank].dump_rx_buffers_spares()
            cclo_inst[dst_rank].recv(0, rx_buf[dst_rank], src_rank, tag)
            recvdata = rx_buf[dst_rank]
            if (recvdata == exp_recvdata).all():
                print("Send/Recv {} -> {} succeeded".format(src_rank, dst_rank))
            else:
                diff        = np.where(recvdata != exp_recvdata)
                firstdiff   = np.min(diff)
                ndiffs      = diff[0].size 
                print("Send/Recv {} -> {} failed, {} bytes different starting at {}".format(src_rank, dst_rank, ndiffs, firstdiff))
                print(f"Senddata: {exp_recvdata} != {recvdata} Recvdata ")
                cclo_inst[dst_rank].dump_communicator()
                cclo_inst[src_rank].dump_communicator()
                cclo_inst[dst_rank].dump_rx_buffers_spares()
                import pdb; pdb.set_trace()
            
def test_sendrecv_unaligned():
    # test sending from each cclo_inst to each other cclo_inst
    print("========================================")
    print("Send/Recv Scenario unaligned")
    print("========================================")

    tag         = 0
    max_offset  = 4
    for src_rank in range(args.naccel):
        for dst_rank in range(args.naccel):
            for offset in range(max_offset):

                if (tx_buf[src_rank][offset:].size < 1):
                    print("buffer too small. to use. skip")
                    continue
                tag +=1
                tx_buf[src_rank][:]= np.random.randint(100, size=tx_buf[src_rank].shape)
                
                cclo_inst[src_rank].send(0, tx_buf[src_rank][offset:], dst_rank, tag=tag)

                cclo_inst[dst_rank].recv(0, rx_buf[dst_rank][offset:], src_rank, tag=tag)

                if ( rx_buf[dst_rank][offset:] == tx_buf[src_rank][offset:]).all():
                    print("Send/Recv {} -> {} succeeded".format(src_rank, dst_rank))
                else:
                    diff        = np.where(recvdata != exp_recvdata)
                    firstdiff   = np.min(diff)
                    ndiffs      = diff[0].size 
                    print("Send/Recv {} -> {} failed, {} bytes different starting at {}".format(src_rank, dst_rank, ndiffs, firstdiff))

def test_bcast(sw=True):
    # test broadcast from each rank
     # test broadcast from each rank
    print("========================================")
    print(f"Broadcast","sw" if sw else "hw","Synch")
    print("========================================")
    for src_rank in range(args.naccel):
        err_count = 0
        tx_buf[src_rank][:]=np.random.randint(100, size=tx_buf[src_rank].shape)
        threads = []
        for j in range(args.naccel):
            buf = tx_buf[src_rank] if (j==src_rank) else rx_buf[j]
            threads.append(threading.Thread(target=cclo_inst[j].bcast, args=(0, buf, src_rank, sw, False, True)))
        
        for j in range(args.naccel):
            threads[j].start()
       
        for j in range(args.naccel):
            threads[j].join()
            
        for j in range(args.naccel):
            if (j==src_rank) :
                buf = tx_buf[j] 
            else:
                buf = rx_buf[j]
                buf.sync_from_device()

            if not (buf == tx_buf[src_rank]).all():
                err_count += 1
                print(f"Bcast {src_rank} -> {j} failed")
                print(f", expected {tx_buf[src_rank]}\n  got: {buf}")

        if err_count == 0:
            print(f"Bcast {src_rank} -> all succeeded")

def test_bcast_rnd(sw=True, repetitions=1):
    # test broadcast from random rank
    import numpy as np
    rng            = np.random.default_rng(12345)
    random_ranks   = rng.integers(low=0, high=args.naccel, size=repetitions)
    print("========================================")
    print(f"Broadcast","sw" if sw else "hw","Synch rnd")
    print("========================================")
    
    for src_rank in random_ranks:
        err_count = 0
        tx_buf[src_rank][:]=np.random.randint(100, size=tx_buf[src_rank].shape)
        threads = []
        for j in range(args.naccel):
            buf = tx_buf[src_rank] if (j==src_rank) else rx_buf[j]
            threads.append(threading.Thread(target=cclo_inst[j].bcast, args=(0, buf, src_rank, sw, False, True)))
        
        for j in range(args.naccel):
            threads[j].start()
       
        for j in range(args.naccel):
            threads[j].join()
            
        for j in range(args.naccel):
            if (j==src_rank) :
                buf = tx_buf[j] 
            else:
                buf = rx_buf[j]
                buf.sync_from_device()

            if not (buf == tx_buf[src_rank]).all():
                err_count += 1
                print(f"Bcast {src_rank} -> {j} failed")
                print(f", expected {tx_buf[src_rank]}\n  got: {buf}")

        if err_count == 0:
            print(f"Bcast {src_rank} -> all succeeded")

def test_bcast_async(sw=True):
    # test broadcast from each rank
    print("========================================")
    print(f"Broadcast","sw" if sw else "hw"," Asynch")
    print("========================================")
    for src_rank in range(args.naccel):
        err_count = 0
        tx_buf[src_rank][:]=np.random.randint(100, size=tx_buf[src_rank].shape)
        calls = []
        for j in range(args.naccel):
            buf = tx_buf[src_rank] if (j==src_rank) else rx_buf[j]
            calls.append(cclo_inst[j].bcast(0, buf, src_rank,sw=sw, from_fpga=False, to_fpga=True, run_async=True))
        for j in range(args.naccel):
            calls[j].wait()
        for j in range(args.naccel):
            if (j==src_rank) :
                buf = tx_buf[j] 
            else:
                buf = rx_buf[j]
                buf.sync_from_device()

            if not (buf == tx_buf[src_rank]).all():
                err_count += 1
                print(f"Bcast {src_rank} -> {j} failed")
                print(f", expected {tx_buf[src_rank]}\n  got: {buf}")
        if err_count == 0:
            print(f"Bcast {src_rank} -> all succeeded")

def test_scatter(sw=True):
    # test scatter from each rank
    print("========================================")
    print("Scatter","sw" if sw else "hw")
    print("========================================")
    for i in range(args.naccel):
        err_count = 0
        count = len(tx_buf[i])//args.naccel
        if count < 1: 
            print("skipped bsize too small")
            continue #avoid improper call with len 0 e.g. with bsize 3 count would be 0
        tx_buf[i][:]=np.random.randint(1000, size=tx_buf[i].shape)
        threads = []
        for j in range(args.naccel):
            rx_buf[j][:]=np.zeros(rx_buf[j].shape)# clear rx buffers
            threads.append(threading.Thread(target=cclo_inst[j].scatter, args=(0, tx_buf[j], rx_buf[j], count, i, sw)))
            threads[-1].start()
        for j in range(args.naccel):
            threads[j].join()
        for j in range(args.naccel):
            if not (rx_buf[j][0:count]==tx_buf[i][count*j:count*(j+1)]).all():
                err_count += 1
                print(f"Scatter {i} -> {j} failed")
        if err_count == 0:
            print(f"Scatter {i} -> all succeeded".format(i))

def test_scatter_async(sw=True):
    # test scatter from each rank
    print("========================================")
    print("Scatter Async","sw" if sw else "hw")
    print("========================================")
    for i in range(args.naccel):
        err_count = 0
        count = len(tx_buf[i])//args.naccel
        if count < 1: 
            print("skipped bsize too small")
            continue #avoid improper call with len 0 e.g. with bsize 3 count would be 0
        tx_buf[i][:]=np.random.randint(1000, size=tx_buf[i].shape)
        calls = []
        for j in range(args.naccel):
            rx_buf[j][:]=np.zeros(rx_buf[j].shape)# clear rx buffers
        for j in range(args.naccel):
            calls.append(cclo_inst[j].scatter(0, tx_buf[j], rx_buf[j], count, i, sw=sw, from_fpga=False, to_fpga=True, run_async=True))
        for j in range(args.naccel):
            calls[j].wait()
        for j in range(args.naccel):
            rx_buf[j].sync_from_device()
            if not (rx_buf[j][0:count]==tx_buf[i][count*j:count*(j+1)]).all():
                err_count += 1
                print(f"Scatter {i} -> {j} failed")
                print(np.where(rx_buf[j][0:count]!=tx_buf[i][count*j:count*(j+1)]))
        if err_count == 0:
            print("Scatter {} -> all succeeded".format(i))

def test_gather(sw=True, ring=True):
    # test gather from each rank
    print("========================================")
    print("Gather","sw" if sw else "hw","ring" if ring else "non-ring")
    print("========================================")
    for i in range(args.naccel):
        err_count = 0
        count = len(tx_buf[i])//args.naccel
        if count < 1: 
            print("skipped bsize too small")
            continue #avoid improper call with len 0 e.g. with bsize 3 count would be 0
        rx_buf[i][:]=np.zeros(rx_buf[i].shape)# clear rx buffer
        threads = []
        for j in range(args.naccel):
            tx_buf[j][:]=np.random.randint(1000, size=tx_buf[j].shape)#init tx buffers
            threads.append(threading.Thread(target=cclo_inst[j].gather, args=(0, tx_buf[j], rx_buf[j], count, i, sw , ring)))
            threads[-1].start()
        for j in range(args.naccel):
            threads[j].join()
        for j in range(args.naccel):
            if not (rx_buf[i][count*j:count*(j+1)]==tx_buf[j][0:count]).all():
                err_count += 1
                print("Gather {} <- {} failed".format(i, j))
        if err_count == 0:
            print("Gather {} <- all succeeded".format(i))

def test_gather_async(sw=True, ring=True):
    # test gather from each rank
    print("========================================")
    print("Gather Async","sw" if sw else "hw","ring" if ring else "non-ring")
    print("========================================")
    for i in range(args.naccel):
        err_count = 0
        count = len(tx_buf[i])//args.naccel
        if count < 1: 
            print("skipped bsize too small")
            continue #avoid improper call with len 0 e.g. with bsize 3 count would be 0
        rx_buf[i][:]=np.zeros(rx_buf[i].shape)# clear rx buffer
        calls = []
        for j in range(args.naccel):
            tx_buf[j][:]=np.random.randint(1000, size=tx_buf[j].shape)#init tx buffers
        for j in range(args.naccel):
            calls.append(cclo_inst[j].gather(0, tx_buf[j], rx_buf[j], count, i, sw=sw, shift=ring, from_fpga=False, to_fpga=True, run_async=True))
        for j in range(args.naccel):
            calls[j].wait()
            rx_buf[i].sync_from_device()
        for j in range(args.naccel):
            rx_buf[i].sync_from_device()
            if not (rx_buf[i][count*j:count*(j+1)]==tx_buf[j][0:count]).all():
                err_count += 1
                print("Gather {} <- {} failed".format(i, j))
        if err_count == 0:
            print("Gather {} <- all succeeded".format(i))

def test_allgather(sw=True, ring=True, fused=False):
    # test gather from each rank
    print("========================================")
    print("AllGather","sw" if sw else "hw","ring" if ring else "non-ring","Fused" if fused else "Non-Fused")
    print("========================================")
    err_count = 0
    count = len(tx_buf[0])//args.naccel
    if count < 1: 
            print("skipped bsize too small")
            return #avoid improper call with len 0 e.g. with bsize 3 count would be 0
    for i in range(args.naccel):
        rx_buf[i][:]=np.zeros(rx_buf[i].shape)# clear rx buffer
        tx_buf[i][:]=np.random.randint(1000, size=tx_buf[i].shape)#init tx buffers
    threads = []
    for j in range(args.naccel):
        threads.append(threading.Thread(target=cclo_inst[j].allgather, args=(0, tx_buf[j], rx_buf[j], count, fused, sw , ring)))
        threads[-1].start()
    for j in range(args.naccel):
        threads[j].join()
    for i in range(args.naccel):
        for j in range(args.naccel):
            if not (rx_buf[i][count*j:count*(j+1)]==tx_buf[j][0:count]).all():
                err_count += 1
                print("AllGather failed on {} block {}".format(i, j))
    if err_count == 0:
        print("AllGather succeeded")

def test_allgather_async(sw=True, ring=True, fused=False):
    # test gather from each rank
    print("========================================")
    print("AllGather Async","sw" if sw else "hw","ring" if ring else "non-ring","Fused" if fused else "Non-Fused")
    print("========================================")
    err_count = 0
    count = len(tx_buf[0])//args.naccel
    if count < 1: 
            print("skipped bsize too small")
            return #avoid improper call with len 0 e.g. with bsize 3 count would be 0
    for i in range(args.naccel):
        rx_buf[i][:]=np.zeros(rx_buf[i].shape)# clear rx buffer
        tx_buf[i][:]=np.random.randint(1000, size=tx_buf[i].shape)#init tx buffers
    calls = []
    for j in range(args.naccel):
        calls.append(cclo_inst[j].allgather(0, tx_buf[j], rx_buf[j], count, fused=fused, sw=sw, ring=ring, from_fpga=False, to_fpga=True, run_async=True))
    for j in range(args.naccel):
        calls[j].wait()
    for i in range(args.naccel):
        rx_buf[i].sync_from_device()
        for j in range(args.naccel):
            if not (rx_buf[i][count*j:count*(j+1)]==tx_buf[j][0:count]).all():
                err_count += 1
                print("AllGather failed on {} block {}".format(i, j))
    if err_count == 0:
        print("AllGather succeeded")

def test_reduce(sw=True, shift=False):
    # test reduce from each rank
    print("========================================")
    print("Reduce","sw" if sw else "hw","shift" if shift else "non-shift")
    print("========================================")
    for np_type in [np.float32, np.float64, np.int32, np.int64]:
        for root in np.random.default_rng().permutation(range(args.naccel)):
            err_count = 0
            count     = tx_buf_fp[root].nbytes
            rx_buf_fp[root][:]=np.zeros(rx_buf_fp[root].shape, dtype=np.int8)# clear rx buffer
            rx_buf_fp[root].sync_to_device()
            threads = []
            #fill vector
            for j in range(args.naccel):
                #tx_buf_fp[j][:]=np.random.random_sample(size=tx_buf_fp[j].shape)#init tx buffers
                tx_buf_fp[j][:]=np.random.randint(127, size=tx_buf_fp[j].size, dtype=np.int8)#init tx buffers
                #tx_buf_fp[j][:]=np.ones(                   rx_buf_fp[j].shape, dtype=np_type)# init tx buffer
                
            #compute expected
            expected=np.zeros(rx_buf_fp[root].shape, dtype=np.int8).view(np_type)
            for j in range(args.naccel):
                expected += tx_buf_fp[j].view(np_type)
            #create run
            for j in range(args.naccel):
                threads.append(threading.Thread(target=cclo_inst[j].reduce, args=(0, tx_buf_fp[j], rx_buf_fp[j], count, root, np_type_2_cclo_type(np_type), sw, shift)))
                threads[-1].start()
            #wait for others to finish
            for j in range(args.naccel):
                threads[j].join()
                
            
            if not np.allclose(                   rx_buf_fp[root].view(np_type),  expected):
                print(f"Reduce {np_type}: {root} <- all failed")
                if args.debug:

                    print(f"result {                  rx_buf_fp[root].view(np_type)}")
                    print(f"distance: {np.linalg.norm(rx_buf_fp[root].view(np_type) - expected)} ({np.absolute(rx_buf_fp[root].view(np_type) - expected)})")
                    diff        = np.where(rx_buf_fp[root].view(np_type)!= expected)
                    firstdiff   = np.min(diff)
                    ndiffs      = diff[0].size 
                    print(f"Reduce failed, {ndiffs} results different starting at {firstdiff} {diff}")
                    for jj in range(args.naccel):
                        print(f"{jj} buffer: {tx_buf_fp[jj].view(np.uint8)}")
                        print(f"{jj} buffer: {tx_buf_fp[jj].view(np_type)}")
                        cclo_inst[jj].dump_rx_buffers_spares()
                    import pdb; pdb.set_trace()
            else:
                print(f"Reduce {np_type}: {root} <- all succeeded")

def test_reduce_async(sw=True, shift=False):
    # test reduce from each rank
    print("========================================")
    print("Reduce Async","sw" if sw else "hw","shift" if shift else "non-shift")
    print("========================================")
    for np_type in [np.float32, np.float64, np.int32, np.int64]:
        for root in np.random.default_rng().permutation(range(args.naccel)):
            err_count = 0
            count = tx_buf_fp[root].nbytes
            rx_buf_fp[root][:]=np.zeros(rx_buf_fp[root].shape, dtype=np.int8)# clear rx buffer
            rx_buf_fp[root].sync_to_device()
            threads = []
            #fill vector
            for j in range(args.naccel):
                #tx_buf_fp[j][:]=np.random.random_sample(size=tx_buf_fp[j].shape)#init tx buffers
                tx_buf_fp[j][:]=np.random.randint(127, size=tx_buf_fp[j].size, dtype=np.int8)#init tx buffers
                #tx_buf_fp[j][:]=np.ones(                   rx_buf_fp[j].shape, dtype=np_type)# init tx buffer
            #compute expected
            expected=np.zeros(rx_buf_fp[root].shape, dtype=np.int8).view(np_type)
            for j in range(args.naccel):
                expected += tx_buf_fp[j].view(np_type)
            #create run
            handles = []
            for j in range(args.naccel):
                handles += [cclo_inst[j].reduce(0, tx_buf_fp[j], rx_buf_fp[j], count, root, np_type_2_cclo_type(np_type), sw=sw, shift=shift, run_async=True, to_fpga=True)]
            #wait for others to finish
            for a_handle in handles:
                a_handle.wait()
            rx_buf_fp[root].sync_from_device()
                
            if not np.allclose(                   rx_buf_fp[root].view(np_type),  expected):
                print(f"Reduce {np_type}: {root} <- all failed")
                if args.debug:

                    print(f"result {                  rx_buf_fp[root].view(np_type)}")
                    print(f"distance: {np.linalg.norm(rx_buf_fp[root].view(np_type) - expected)} ({np.absolute(rx_buf_fp[root].view(np_type) - expected)})")
                    diff        = np.where(rx_buf_fp[root].view(np_type)!= expected)
                    firstdiff   = np.min(diff)
                    ndiffs      = diff[0].size 
                    print(f"Reduce failed, {ndiffs} results different starting at {firstdiff} {diff}")
                    for jj in range(args.naccel):
                        print(f"{jj} buffer: {tx_buf_fp[jj].view(np.uint8)}")
                        print(f"{jj} buffer: {tx_buf_fp[jj].view(np_type)}")
                        cclo_inst[jj].dump_rx_buffers_spares()
                    import pdb; pdb.set_trace()
            else:
                print(f"Reduce {np_type}: {root} <- all succeeded")


def test_allreduce(fused=False, sw=False):
    # test reduce from each rank
    print("========================================")
    print("AllReduce ","Fused" if fused else "Non-Fused","sw" if sw else "hw")
    print("========================================")
    #for each type
    for np_type in [np.float32, np.float64, np.int32, np.int64]:

        #initialize buffers with random data. 
        for j in range(args.naccel):
            # fill buffers with random bytes that will be interpreted differently depending on the type
            err_count = 0
            count = tx_buf_fp[j].nbytes
            rx_buf_fp[j][:]=np.zeros(                  rx_buf_fp[j].shape, dtype=np.int8)# clear rx buffer
            tx_buf_fp[j][:]=np.random.randint(127, size=tx_buf_fp[j].shape, dtype=np.int8)#init tx buffers
            #tx_buf_fp[j][:]=np.ones(                   rx_buf_fp[j].shape, dtype=np_type)# init tx buffer
        
        threads = []
        for j in range(args.naccel):
            threads.append(threading.Thread(target=cclo_inst[j].allreduce, args=(0, tx_buf_fp[j], rx_buf_fp[j], count,  np_type_2_cclo_type(np_type), fused, sw)))
        for j in range(args.naccel):
            threads[j].start()
        #wait for completion
        for j in range(args.naccel):
            threads[j].join()
        #compute expected
        expected=np.zeros(rx_buf_fp[0].shape, dtype=np.int8).view(np_type)
        for j in range(args.naccel):
            expected += tx_buf_fp[j].view(np_type)
        #check results
        for j in range(args.naccel):
            if not np.allclose(rx_buf_fp[j].view(np_type), expected):
                print(f"AllReduce {np_type}: {j} <- all failed")
               
                if args.debug:
                    print(f"expected result {np_type}: {                  expected.view(np_type)}")
                    print(f"expected result {np.uint8}: {                  expected.view(np.uint8)}")
                    print(f"obtained result {np_type}: {                  rx_buf_fp[j].view(np_type)}")
                    print(f"obtained result {np.uint8}: {                  rx_buf_fp[j].view(np.uint8)}")
                    print(f"distance: {np.linalg.norm(rx_buf_fp[j].view(np_type) - expected)} ({np.absolute(rx_buf_fp[j].view(np_type) - expected)})")
                    diff        = np.where(rx_buf_fp[j].view(np_type)!= expected)
                    firstdiff   = np.min(diff)
                    ndiffs      = diff[0].size 
                    for jj in range(args.naccel):
                        print(f"{jj}tx buffer {np.uint8}: {tx_buf_fp[jj].view(np.uint8)}")
                        print(f"{jj}tx buffer {np_type}:  {tx_buf_fp[jj].view(np_type)}")
                        print(f"{jj}rx buffer {np.uint8}: {rx_buf_fp[jj].view(np.uint8)}")
                        print(f"{jj}rx buffer {np_type}:  {rx_buf_fp[jj].view(np_type)}")
                        cclo_inst[jj].dump_communicator()
                        cclo_inst[jj].dump_rx_buffers_spares()
                    print(f"Reduce failed, {ndiffs} results different starting at {firstdiff} {diff}")
                    import pdb; pdb.set_trace()
            else:
                print(f"AllReduce {np_type}: {j} <- all succeeded")

def test_allreduce_async(fused=False, sw=False):
    # test reduce from each rank
    print("========================================")
    print("AllReduce Asynch","Fused" if fused else "Non-Fused","sw" if sw else "hw")
    print("========================================")
    #for each type
    for np_type in [np.float32, np.float64, np.int32, np.int64]:

        #initialize buffers with random data. 
        for j in range(args.naccel):
            # fill buffers with random bytes that will be interpreted differently depending on the type
            err_count = 0
            count = tx_buf_fp[j].nbytes
            rx_buf_fp[j][:]=np.zeros(                   rx_buf_fp[j].shape, dtype=np.int8)# clear rx buffer
            tx_buf_fp[j][:]=np.random.randint(127, size=tx_buf_fp[j].shape, dtype=np.int8)#init tx buffers
            #tx_buf_fp[j][:]=np.ones(                   rx_buf_fp[j].shape, dtype=np_type)# init tx buffer
        handles = []
        for j in range(args.naccel):
            handles += [cclo_inst[j].allreduce(0, tx_buf_fp[j], rx_buf_fp[j], count,  np_type_2_cclo_type(np_type), fused, sw, run_async=True, to_fpga=True)]
        
        #wait for completion
        for a_handle in handles:
            a_handle.wait()
        #compute expected
        expected=np.zeros(rx_buf_fp[0].shape, dtype=np.int8).view(np_type)
        for j in range(args.naccel):
            expected += tx_buf_fp[j].view(np_type)
        #check results
        for j in range(args.naccel):
            rx_buf_fp[j].sync_from_device()
            if not np.allclose(rx_buf_fp[j].view(np_type), expected):
                print(f"AllReduce {np_type}: {j} <- all failed")
               
                if args.debug:
                    print(f"expected result {np_type}: {                  expected.view(np_type)}")
                    print(f"expected result {np.uint8}: {                  expected.view(np.uint8)}")
                    print(f"obtained result {np_type}: {                  rx_buf_fp[j].view(np_type)}")
                    print(f"obtained result {np.uint8}: {                  rx_buf_fp[j].view(np.uint8)}")
                    print(f"distance: {np.linalg.norm(rx_buf_fp[j].view(np_type) - expected)} ({np.absolute(rx_buf_fp[j].view(np_type) - expected)})")
                    diff        = np.where(rx_buf_fp[j].view(np_type)!= expected)
                    firstdiff   = np.min(diff)
                    ndiffs      = diff[0].size 
                    for jj in range(args.naccel):
                        print(f"{jj}tx buffer {np.uint8}: {tx_buf_fp[jj].view(np.uint8)}")
                        print(f"{jj}tx buffer {np_type}:  {tx_buf_fp[jj].view(np_type)}")
                        print(f"{jj}rx buffer {np.uint8}: {rx_buf_fp[jj].view(np.uint8)}")
                        print(f"{jj}rx buffer {np_type}:  {rx_buf_fp[jj].view(np_type)}")
                        cclo_inst[jj].dump_communicator()
                        cclo_inst[jj].dump_rx_buffers_spares()
                    print(f"Reduce failed, {ndiffs} results different starting at {firstdiff} {diff}")
                    import pdb; pdb.set_trace()
            else:
                print(f"AllReduce {np_type}: {j} <- all succeeded")

def test_copy():
    err_count = 0
    for i in range(args.naccel):
        tx_buf[i][:] = np.random.randint(128, size=tx_buf[i].size, dtype=np.int8)
        rx_buf[i][:] = np.random.randint(128, size=rx_buf[i].size, dtype=np.int8)
        cclo_inst[i].copy(tx_buf[i], rx_buf[i])
        if not (rx_buf[i]==tx_buf[i]).all():
            err_count += 1
            print("Copy failed on CCLO {}".format(i))
    if err_count == 0:
        print("Copy succeeded")

def test_external_stream():
    err_count = 0
    for i in range(args.naccel):
        tx_buf[i][:] = np.random.randint(128, size=tx_buf[i].size, dtype=np.int8)
        rx_buf[i][:] = np.random.randint(128, size=rx_buf[i].size, dtype=np.int8)
        #first 512 bits are considered header
        # in particular first 32 indicates bytes to transfer
        non_header_bytes = max(0, tx_buf[i].nbytes-64)
        tx_buf[i][0]=(non_header_bytes)>>24 &0xff
        tx_buf[i][1]=(non_header_bytes)>>16 &0xff
        tx_buf[i][2]=(non_header_bytes)>>8  &0xff
        tx_buf[i][3]=(non_header_bytes)     &0xff

        print("before rx",rx_buf[i] )
        print("before tx",tx_buf[i] )
        cclo_inst[i].external_stream_kernel(tx_buf[i], rx_buf[i])
        if not (rx_buf[i]==tx_buf[i]).all():
            err_count += 1
            print("external copy failed on CCLO {}".format(i))
            print("after rx",rx_buf[i] )
            print("after tx",tx_buf[i] )
            diff        = np.where(rx_buf[i] != tx_buf[i])
            firstdiff   = np.min(diff)
            ndiffs      = diff[0].size 
            print(f"external failed, {ndiffs} results different starting at {firstdiff} {diff}")
                    
    if err_count == 0:
        print("external copy succeeded")


def test_external_reduce():
    global ext_arithm
    def setup_ext_arith(ext_arithm, buf_size,function):
        #//get number of 512b/64B transfers corresponding to byte_count
        #unsigned int ntransfers = (byte_count+63)/64;
        #Xil_Out32(ARITH_BASEADDR+0x10, ntransfers);
        #Xil_Out32(ARITH_BASEADDR+0x18, function);
        #SET(ARITH_BASEADDR, CONTROL_START_MASK);
        CONTROL_START_MASK = 0x0000_0001
        ntransfers = (buf_size+63) // 64
        ext_arithm.mmio.write(0x10, ntransfers)
        ext_arithm.mmio.write(0x18, function.value)
        ext_arithm.mmio.write(0x0, CONTROL_START_MASK)
    #compute rx = tx + rx
    for i in range(args.naccel):
        tx_buf[i][:] = np.random.randint(128, size=tx_buf[i].size, dtype=np.int8)
        rx_buf[i][:] = np.random.randint(128, size=rx_buf[i].size, dtype=np.int8)

        buf0_fp = tx_buf[i].view(np.float32)
        buf1_fp = rx_buf[i].view(np.float32)
        sum_fp = buf0_fp + buf1_fp
        setup_ext_arith(ext_arithm[i], buf0_fp.nbytes, CCLOReduceFunc.fp)
        cclo_inst[i].external_reduce(buf0_fp, buf1_fp,  buf1_fp)
        assert np.allclose(buf1_fp, sum_fp)
        print("FP EXT Sum for CCLO {} succeeded".format(i))
   

def test_acc():
    for i in range(args.naccel):
        tx_buf[i][:] = np.random.randint(128, size=tx_buf[i].size, dtype=np.int8)
        rx_buf[i][:] = np.random.randint(128, size=rx_buf[i].size, dtype=np.int8)

        buf0_fp = tx_buf[i].view(np.float32)
        buf1_fp = rx_buf[i].view(np.float32)
        sum_fp = buf0_fp + buf1_fp

        cclo_inst[i].accumulate(CCLOReduceFunc.fp, buf0_fp, buf1_fp)
        assert np.allclose(buf1_fp, sum_fp)
        print("FP Sum for CCLO {} succeeded".format(i))

        buf0_dp = tx_buf[i].view(np.float64)
        buf1_dp = rx_buf[i].view(np.float64)
        sum_dp = buf0_dp + buf1_dp

        cclo_inst[i].accumulate(CCLOReduceFunc.dp, buf0_dp, buf1_dp)
        assert np.allclose(buf1_dp, sum_dp)
        print("DP Sum for CCLO {} succeeded".format(i))

        buf0_i32 = tx_buf[i].view(np.int32)
        buf1_i32 = rx_buf[i].view(np.int32)
        sum_i32 = buf0_i32 + buf1_i32

        cclo_inst[i].accumulate(CCLOReduceFunc.i32, buf0_i32, buf1_i32)
        assert np.array_equal(buf1_i32, sum_i32)
        print("INT32 Sum for CCLO {} succeeded".format(i))

        buf0_i64 = tx_buf[i].view(np.int64)
        buf1_i64 = rx_buf[i].view(np.int64)
        sum_i64 = buf0_i64 + buf1_i64

        cclo_inst[i].accumulate(CCLOReduceFunc.i64, buf0_i64, buf1_i64)
        assert np.array_equal(buf1_i64, sum_i64)
        print("INT64 Sum for CCLO {} succeeded".format(i))

def test_timeout():
    global cclo_inst
    for i in range(args.naccel):
        cclo_inst[i].set_timeout(100)
        rand_rank = np.random.randint(args.naccel,  size=1)[0]
        rand_tag  = np.random.randint(256,          size=1)[0]
        try:
            cclo_inst[i].recv(0, rx_buf[i], rand_rank , rand_tag)
            #if arrived here error
            assert False
        except Exception as e:
            print("correctly got:",e)
            pass
        #check return code
        assert( cclo_inst[i].get_retcode() == ErrorCode.RECEIVE_TIMEOUT_ERROR )
    print("timeout passed")
    #after that reset
    for i in range(args.naccel):
        cclo_inst[i].deinit()

    ol, cclo_inst, devicemem = configure_accl(args.xclbin, args.device_index, nbufs=args.nbufs, bufsize=max(16*1024, args.bsize))

def print_timing_us(label,duration_us):
    print(f"execution time {label:.<44} us :{duration_us:>6.2f} ")
    global csv_writer
    csv_writer.writerow([label,duration_us])


def benchmark(niter):
    csv_file = open(f"bench_naccel{args.naccel}_niter{niter}_bsize{args.bsize}_sw{args.sw}.csv", "w", newline="") 
    global csv_writer
    import csv
    csv_writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["", f"bsize{args.bsize} sw{args.sw}"])
    csv_writer.writerow(["collective","execution time [us]"])
    print(f"{'':=>50}")
    print(f"Benchmarks ({niter} iterations, bsize {args.bsize}, naccel {args.naccel})" )
    print(f"{'':=>50}")
    #nop warmup
    for j in range(args.naccel):
        for _ in range(niter):
            cclo_inst[j].nop(run_async=False)
    seconds_to_us = 1_000_000
    length_string = 30
    # nop
    if args.nop:
        start = time.perf_counter()
        prevcall = []
        for i in range(niter):
            prevcall = [cclo_inst[0].nop(run_async=True, waitfor=prevcall)]
        prevcall[0].wait()
        end = time.perf_counter()
        for j in range(args.naccel):
            cclo_inst[j].check_return_value()
        duration_us = ((end - start)/niter) * seconds_to_us
        print_timing_us('NOP',duration_us)

    # send
    if args.sendrecv:

        prevcall_send = []
        prevcall_recv = []
        src_rank=0
        dst_rank=1
        
        for c in cclo_inst:
            c.start_profiling()
        start = time.perf_counter()
        for i in range(niter):
            prevcall_send = [cclo_inst[src_rank].send(0, tx_buf[src_rank], dst_rank, from_fpga=True , run_async=True, waitfor=prevcall_recv)]
            prevcall_recv = [cclo_inst[dst_rank].recv(0, rx_buf[dst_rank], src_rank, to_fpga=True   , run_async=True, waitfor=prevcall_recv)]
        
        for handle in [*prevcall_send, *prevcall_recv]:
            handle.wait()

        end = time.perf_counter()
        for j in range(args.naccel):
            cclo_inst[j].check_return_value()
        for c in cclo_inst:
            c.end_profiling()

        duration_us = ((end - start)/niter) * seconds_to_us
        print_timing_us('Send and recv',duration_us)
    # copy
    if args.copy:
        for c in cclo_inst:
            c.start_profiling()
        start = time.perf_counter()
        prevcall = []
        for i in range(niter):
            prevcall = [cclo_inst[0].copy(tx_buf[0], rx_buf[0], from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)]
        prevcall[0].wait()
        end = time.perf_counter()
        for j in range(args.naccel):
            cclo_inst[j].check_return_value()
        for c in cclo_inst:
            c.end_profiling()
        duration_us = ((end - start)/niter) * seconds_to_us
        print_timing_us('Copy',duration_us)

    
    # accumulation on kernel 0
    if args.accumulate:
        prevcall = []
        for c in cclo_inst:
            c.start_profiling()
        start = time.perf_counter()
        for i in range(niter):
            prevcall = [cclo_inst[0].accumulate(CCLOReduceFunc.fp, tx_buf[0], rx_buf[0], val_from_fpga=True, acc_from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)]
        prevcall[0].wait()
        end = time.perf_counter()
        for j in range(args.naccel):
            cclo_inst[j].check_return_value()
        for c in cclo_inst:
            c.end_profiling()
        duration_us = ((end - start)/niter) * seconds_to_us
        print_timing_us('Accumulate',duration_us)

    def bench_base(call):
        global args, cclo_inst, tx_buf, rx_buf, tx_buf_fp, rx_buf_fp
        prevcall = []
        prevcall_next = []
        for c in cclo_inst:
            c.start_profiling()
        ## start measuring
        start = time.perf_counter()
        for i in range(args.nruns):
            for j in range(args.naccel):
                prevcall_next += [call(cclo_inst[j], j , prevcall)]
                #print(prevcall_next)
            prevcall = prevcall_next
            prevcall_next = []  
        for handle in prevcall:
            handle.wait()
        end = time.perf_counter()
        ##end measuring
        for c in cclo_inst:
            c.check_return_value()
            c.end_profiling()
        duration_us = ((end - start)/niter) * seconds_to_us
        return duration_us
        
    #bcast
    if args.bcast:
        if args.sw:
            #bcast sw
            def f(cclo_inst, j,  prevcall):
                return cclo_inst.bcast(0, tx_buf[j], root=0, sw=True, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
            print_timing_us('Bcast sw',duration_us)

        
        else:
            #bcast hw
            def f(cclo_inst,j , prevcall):
                return cclo_inst.bcast(0, tx_buf[j], root=0, sw=False, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
            print_timing_us('Bcast hw',duration_us)
    
    # Scatter
    if args.scatter:
        if args.sw:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.scatter(0, tx_buf[j], rx_buf[j], tx_buf[j].size//args.naccel, root=0,sw=True, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)

            print_timing_us('Scatter sw',duration_us)
        else:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.scatter(0, tx_buf[j], rx_buf[j], tx_buf[j].size//args.naccel, root=0,sw=False, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)

            print_timing_us('Scatter hw',duration_us)

    if args.gather:
        if args.sw:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.gather(0, tx_buf[j], rx_buf[j], tx_buf[j].size//args.naccel, root=1, sw=True, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)

            print_timing_us('Gather sw',duration_us)
        else:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.gather(0, tx_buf[j], rx_buf[j], tx_buf[j].size//args.naccel, root=1, sw=False, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
            
            print_timing_us('Gather hw',duration_us)
    # Allgather
    if args.allgather:
        if args.sw:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.allgather(0, tx_buf[j], rx_buf[j], tx_buf[j].size//args.naccel, sw=True, ring=True, fused=True, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)

            print_timing_us('Allgather fused sw',duration_us)

            def f(cclo_inst,j , prevcall):
                return cclo_inst.allgather(0, tx_buf[j], rx_buf[j], tx_buf[j].size//args.naccel, sw=True, ring=True, fused=False, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)

            print_timing_us('Allgather non fused sw',duration_us)
        else:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.allgather(0, tx_buf[j], rx_buf[j], tx_buf[j].size//args.naccel, sw=False, ring=True, fused=True, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
            
            print_timing_us('Allgather fused hw',duration_us)

            def f(cclo_inst,j , prevcall):
                return cclo_inst.allgather(0, tx_buf[j], rx_buf[j], tx_buf[j].size//args.naccel, sw=False, ring=True, fused=False, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
            
            print_timing_us('Allgather non fused hw',duration_us)
    # Reduce
    
    threads = []
    #fill vector
    for j in range(args.naccel):
        tx_buf_fp[j][:]=np.ones( tx_buf_fp[j].shape, dtype=np.float32)# init tx buffer
        rx_buf_fp[j][:]=np.zeros(rx_buf_fp[j].shape, dtype=np.float32)# clear rx buffer
        tx_buf_fp[j].sync_to_device()
        rx_buf_fp[j].sync_to_device()
        #tx_buf_fp[j][:]=np.random.random_sample(size=tx_buf_fp[j].shape)#init tx buffers
        #tx_buf_fp[j][:]=np.random.randint(127, size=tx_buf_fp[j].size, dtype=np.int8)#init tx buffers
    if args.reduce:
        if args.sw:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.reduce(0, tx_buf[j].view(np.float32), rx_buf[j].view(np.float32), tx_buf[j].view(np.uint8).size, root=1, func=CCLOReduceFunc.fp, sw=True, shift=True, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
        
            print_timing_us('Reduce sw ring',duration_us)
        else:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.reduce(0, tx_buf[j].view(np.float32), rx_buf[j].view(np.float32), tx_buf[j].view(np.uint8).size, root=1, func=CCLOReduceFunc.fp, sw=False, shift=True, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
        
            print_timing_us('Reduce hw ring',duration_us)
    # Allreduce
    if args.allreduce:
        if args.sw:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.allreduce(0, tx_buf[j].view(np.float32), rx_buf[j].view(np.float32), tx_buf[j].view(np.uint8).size, CCLOReduceFunc.fp, sw=True, ring=True, fused=True, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
        
            print_timing_us('Allreduce fused sw ring',duration_us)

            def f(cclo_inst,j , prevcall):
                return cclo_inst.allreduce(0, tx_buf[j].view(np.float32), rx_buf[j].view(np.float32), tx_buf[j].view(np.uint8).size, CCLOReduceFunc.fp, sw=True, ring=True, fused=False, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
        
            print_timing_us('Allreduce non fused sw ring',duration_us)
        else:
            def f(cclo_inst,j , prevcall):
                return cclo_inst.allreduce(0, tx_buf[j].view(np.float32), rx_buf[j].view(np.float32), tx_buf[j].view(np.uint8).size, CCLOReduceFunc.fp, sw=False, ring=True, fused=True, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
            
            print_timing_us('Allreduce fused hw ring',duration_us)

            def f(cclo_inst,j , prevcall):
                return cclo_inst.allreduce(0, tx_buf[j].view(np.float32), rx_buf[j].view(np.float32), tx_buf[j].view(np.uint8).size, CCLOReduceFunc.fp, sw=False, ring=True, fused=False, from_fpga=True, to_fpga=True, run_async=True, waitfor=prevcall)
            duration_us = bench_base(f)
        
            print_timing_us('Allreduce non fused hw ring',duration_us)
    csv_file.close()

def test_spare():
    #try to use all the spare buffers
    from time import sleep
    src_rank = 0
    dst_rank = 1
    for i in range(args.nbufs):
        tx_buf[src_rank][0] = i
        cclo_inst[src_rank].send(0, tx_buf[src_rank], dst_rank, tag=i)
        sleep(1)
        cclo_inst[dst_rank].dump_rx_buffers_spares()
        
    for i in range(args.nbufs):
        cclo_inst[dst_rank].dump_rx_buffers_spares()
        cclo_inst[dst_rank].recv(0, tx_buf[dst_rank], src_rank, tag=i) 
        assert(tx_buf[dst_rank][0] == i)

def deinit_system():
    for i in range(args.naccel):
        cclo_inst[i].deinit()

def reinit():
    global cclo_inst
    for j in range(args.naccel):
        cclo_inst[j].deinit()
    ol, cclo_inst, devicemem = configure_accl(args.xclbin, args.device_index, nbufs=args.nbufs, bufsize=max(16*1024, args.bsize))

def allocate_buffers(n, bsize, devicemem):
    tx_buf = []
    rx_buf = []
    for i in range(n):
        tx_buf.append(pynq.allocate((bsize,), dtype=np.int8, target=devicemem[i]))
        rx_buf.append(pynq.allocate((bsize,), dtype=np.int8, target=devicemem[i]))

    for i, buf in enumerate(rx_buf):
        print(f'rx_buf {i}',hex(buf.device_address))
    for i, buf in enumerate(tx_buf):
        print(f'tx_buf {i}',hex(buf.device_address))

    tx_buf_fp = []
    rx_buf_fp = []
    for i in range(n):
        tx_buf_fp.append(pynq.allocate((bsize,), dtype=np.int8, target=devicemem[i]))
        rx_buf_fp.append(pynq.allocate((bsize,), dtype=np.int8, target=devicemem[i]))

    return tx_buf, rx_buf, tx_buf_fp, rx_buf_fp

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Tests for MPI collectives offload with UDP (VNx) backend')
    parser.add_argument('--xclbin',         type=str, default=None,             help='Accelerator image file (xclbin)', required=True)
    parser.add_argument('--device_index',   type=int, default=1,                help='Card index')
    parser.add_argument('--nruns',          type=int, default=1,                help='How many times to run each test')
    parser.add_argument('--nbufs',          type=int, default=16,               help='number of spare buffers to configure each ccl_offload')
    parser.add_argument('--naccel',         type=int, default=4,                help='number of ccl_offload to test ')
    parser.add_argument('--bsize',          type=int, default=1024,             help='How many KB per buffer')
    parser.add_argument('--dump_rx_regs',   type=int, default=-1,               help='Print RX regs of specified ')
    parser.add_argument('--debug',          action='store_true', default=False, help='enable debug mode')
    parser.add_argument('--all',            action='store_true', default=False, help='Select all collectives')
    parser.add_argument('--nop',            action='store_true', default=False, help='Run nop test')
    parser.add_argument('--sendrecv',       action='store_true', default=False, help='Run send/recv test')
    parser.add_argument('--bcast',          action='store_true', default=False, help='Run bcast test')
    parser.add_argument('--scatter',        action='store_true', default=False, help='Run scatter test')
    parser.add_argument('--gather',         action='store_true', default=False, help='Run gather test')
    parser.add_argument('--allgather',      action='store_true', default=False, help='Run allgather test')
    parser.add_argument('--reduce',         action='store_true', default=False, help='Run reduce test')
    parser.add_argument('--allreduce',      action='store_true', default=False, help='Run allreduce test')
    parser.add_argument('--accumulate',     action='store_true', default=False, help='Run fp/dp/i32/i64 test')
    parser.add_argument('--copy',           action='store_true', default=False, help='Run copy test')
    parser.add_argument('--external_stream',action='store_true', default=False, help='Run external_stream test')
    parser.add_argument('--external_reduce',action='store_true', default=False, help='Run external_reduce test')
    parser.add_argument('--timeout',        action='store_true', default=False, help='Run timeout test')
    parser.add_argument('--spare',          action='store_true', default=False, help='Run tests that include running out of spare buffers')
    parser.add_argument('--regression',     action='store_true', default=False, help='Run all tests with various message sizes')
    parser.add_argument('--benchmark',      action='store_true', default=False, help='Measure performance')
    parser.add_argument('--sw',        action='store_true', default=False, help='Run benchmarks only for sw collectives')

    args = parser.parse_args()
    if args.all:
        args.nop        = True
        args.sendrecv   = True
        args.bcast      = True
        args.scatter    = True
        args.gather     = True
        args.allgather  = True
        args.reduce     = True
        args.allreduce  = True
        args.accumulate = True
        args.copy       = True
        args.external_stream = True
        args.external_reduce = True
        
    #configure FPGA and CCLO cores with the default 16 RX buffers of bsize KB each
    ol, cclo_inst, devicemem = configure_accl(args.xclbin, args.device_index, nbufs=args.nbufs, bufsize=max(16*1024, args.bsize))

    tx_buf, rx_buf, tx_buf_fp, rx_buf_fp = allocate_buffers(args.naccel, args.bsize, devicemem)

    if args.dump_rx_regs >= 0 :
        for cclo_i in cclo_inst:
            cclo_i.dump_rx_buffers_spares()

    try:
    
        #set a random seed to make it reproducible
        np.random.seed(2021)
        for i in range(args.naccel):
            cclo_inst[i].set_timeout(1_000_00)
        
        if not args.benchmark and args.spare:
            for i in range(args.nruns):
                test_spare()

        if not args.benchmark and args.sendrecv:
            for i in range(args.nruns):
                test_sendrecv()
                test_self_sendrecv()
                test_sendrecv_unaligned()

        if not args.benchmark and args.bcast:
            for i in range(args.nruns):
                test_bcast(         sw=True)
                test_bcast_async(   sw=True)
                test_bcast(         sw=False)
                test_bcast_async(   sw=False)
            test_bcast_rnd(     sw=True , repetitions=5)
            test_bcast_rnd(     sw=False, repetitions=5)

        if not args.benchmark and args.scatter:
            for i in range(args.nruns):
                test_scatter(    sw=True)
                test_scatter(    sw=False)
                test_scatter_async(sw=True)
                test_scatter_async(sw=False)
                

        if not args.benchmark and args.gather:
            for i in range(args.nruns):
                ##test_gather(       sw=True , ring=False) non-shift sw gather not implemented
                test_gather(        sw=True , ring=True )
                #test_gather(        sw=False, ring=False) not safe now
                test_gather(        sw=False, ring=True )
                ##test_gather_async( sw=True , ring=False) non-shift sw gather not implemented
                test_gather_async(  sw=True , ring=True )
                #test_gather_async(  sw=False, ring=False) not safe now
                test_gather_async(  sw=False, ring=True ) 
                

        if not args.benchmark and args.allgather:
            for i in range(args.nruns):
                test_allgather(         sw=True , ring=True , fused=True  )
                test_allgather(         sw=True , ring=True , fused=False )
                #test_allgather(         sw=True , ring=False, fused=False ) not safe now
                #test_allgather(         sw=True , ring=False, fused=True  ) not implemented
                test_allgather(         sw=False , ring=True , fused=True  )
                #test_allgather(         sw=False , ring=True , fused=False ) not implemented
                #test_allgather(         sw=False , ring=False, fused=False ) not safe now
                #test_allgather(         sw=False , ring=False, fused=True  ) not implemented
                test_allgather_async(   sw=True , ring=True , fused=True  )
                test_allgather_async(   sw=True , ring=True , fused=False )
                #test_allgather_async(   sw=True , ring=False, fused=False ) not safe now
                #test_allgather_async(   sw=True , ring=False, fused=True  ) not implemented
                test_allgather_async(   sw=False , ring=True , fused=True  )
                #test_allgather_async(   sw=False , ring=True , fused=False ) not implemented
                #test_allgather_async(   sw=False , ring=False, fused=False ) not safe now
                #test_allgather_async(   sw=False , ring=False, fused=True  ) not implemented


        if not args.benchmark and args.reduce:
            for i in range(args.nruns):
                #test_reduce(sw=True , shift=False) no-shift not implemented
                test_reduce(sw=True , shift=True)
                #test_reduce(sw=False, shift=False) not safe now
                test_reduce(sw=False, shift=True)
                #test_reduce_async(sw=True , shift=False) no-shift not implemented
                test_reduce_async(sw=True , shift=True)
                #test_reduce_async(sw=False, shift=False) not safe now
                test_reduce_async(sw=False, shift=True)

        if not args.benchmark and args.allreduce:
            for i in range(args.nruns):
                test_allreduce(fused=False  , sw=True )
                test_allreduce(fused=True   , sw=True )             
                test_allreduce(fused=False  , sw=False)
                test_allreduce(fused=True   , sw=False)
                test_allreduce_async(fused=False  , sw=True )
                test_allreduce_async(fused=True   , sw=True )
                test_allreduce_async(fused=False  , sw=False)
                test_allreduce_async(fused=True   , sw=False)


        if not args.benchmark and args.accumulate:
            for i in range(args.nruns):
                test_acc()

        if not args.benchmark and args.copy:
            for i in range(args.nruns):
                test_copy()

        if not args.benchmark and args.timeout:
            for i in range(args.nruns):
                test_timeout()

        if not args.benchmark and args.external_stream:
            for i in range(args.nruns):
                test_external_stream()
        
        if not args.benchmark and args.external_reduce:
            for i in range(args.nruns):
                test_external_reduce()
        
        if args.benchmark:
            benchmark(args.nruns)

        if args.regression:
            for size in [1, 8, 16, 50, 128, 200, 256, 2_500, 25_000, 1_000_000, 7_000_000]:
                if size > args.bsize:
                    continue #this ensures that spare buffer are large enough to store the intermediate results
                #but since most of the test rely on buffer size we have to reallocate the buffers
                for buf in [*tx_buf, *rx_buf, *tx_buf_fp, *rx_buf_fp]:
                    buf.freebuffer()
                del tx_buf, rx_buf, tx_buf_fp, rx_buf_fp
                tx_buf, rx_buf, tx_buf_fp, rx_buf_fp = allocate_buffers(args.naccel, args.bsize, devicemem)
                print(f"Regression for bsize {size}")
                for i in range(args.nruns):

                    test_self_sendrecv()
                    test_sendrecv()
                    test_sendrecv_unaligned()
                    test_bcast(         sw=True)
                    test_bcast_async(   sw=True)
                    test_bcast(         sw=False)
                    test_bcast_async(   sw=False)
                    test_bcast_rnd(     sw=True , repetitions=5)
                    test_bcast_rnd(     sw=False, repetitions=5)
                    test_scatter(       sw=True)
                    test_scatter(       sw=False)
                    test_scatter_async( sw=True)
                    test_scatter_async( sw=False)
                    ##test_gather(       sw=True , ring=False) non-shift sw gather not implemented
                    test_gather(        sw=True , ring=True )
                    #test_gather(        sw=False, ring=False) not safe now
                    test_gather(        sw=False, ring=True )
                    ##test_gather_async( sw=True , ring=False) non-shift sw gather not implemented
                    test_gather_async(  sw=True , ring=True )
                    #test_gather_async(  sw=False, ring=False) not safe now
                    test_gather_async(  sw=False, ring=True ) 
                    test_allgather(         sw=True , ring=True , fused=True  )
                    test_allgather(         sw=True , ring=True , fused=False )
                    #test_allgather(         sw=True , ring=False, fused=False ) not safe now
                    #test_allgather(         sw=True , ring=False, fused=True  ) not implemented
                    test_allgather(         sw=False , ring=True , fused=True  )
                    #test_allgather(         sw=False , ring=True , fused=False ) not implemented
                    #test_allgather(         sw=False , ring=False, fused=False ) not safe now
                    #test_allgather(         sw=False , ring=False, fused=True  ) not implemented
                    test_allgather_async(   sw=True , ring=True , fused=True  )
                    test_allgather_async(   sw=True , ring=True , fused=False )
                    #test_allgather_async(   sw=True , ring=False, fused=False ) not safe now
                    #test_allgather_async(   sw=True , ring=False, fused=True  ) not implemented
                    test_allgather_async(   sw=False , ring=True , fused=True  )
                    #test_allgather_async(   sw=False , ring=True , fused=False ) not implemented
                    #test_allgather_async(   sw=False , ring=False, fused=False ) not safe now
                    #test_allgather_async(   sw=False , ring=False, fused=True  ) not implemented
                    test_copy()
                    # do not allow the arithmetic test to progress
                    # if sizes aren't divisible by sizeof(fp32) or sizeof(fp64)
                    if size % 8 == 0:
                        test_acc()
                        #test_reduce(sw=True , shift=False) no-shift not implemented
                        test_reduce(sw=True , shift=True)
                        #test_reduce(sw=False, shift=False) not safe now
                        test_reduce(sw=False, shift=True)
                        #test_reduce_async(sw=True , shift=False) no-shift not implemented
                        test_reduce_async(sw=True , shift=True)
                        #test_reduce_async(sw=False, shift=False) not safe now
                        test_reduce_async(sw=False, shift=True)
                        test_allreduce( sw=True , fused=False   )
                        test_allreduce( sw=True , fused=True    )
                        test_allreduce( sw=False, fused=False   )
                        test_allreduce( sw=False, fused=True    )
                        test_allreduce_async(fused=False  , sw=True )
                        test_allreduce_async(fused=True   , sw=True )
                        test_allreduce_async(fused=False  , sw=False)
                        test_allreduce_async(fused=True   , sw=False)

    except KeyboardInterrupt:
        print("CTR^C")
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)

    deinit_system()
