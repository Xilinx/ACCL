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
from cclo import *
import json 
import argparse
import random
from queue import Queue
import threading
import pynq
from _thread import *
import socket
from vnx_utils import *

def configure_xccl(xclbin, board_idx, ranks, init_rx=True, nbufs=16, bufsize=16*1024, vnx=False):

    local_alveo = pynq.Device.devices[board_idx]
    ol=pynq.Overlay(xclbin, device=local_alveo)

    print("Allocating 1MB scratchpad memory")
    if local_alveo.name == 'xilinx_u250_xdma_201830_2':
        devicemem = ol.DDR2
    elif local_alveo.name == 'xilinx_u280_xdma_201920_3':
        devicemem = ol.HBM0

    cclo = [ol.ccl_offload_inst]

    if init_rx:
        for i in range(len(cclo)):
            print("CCLO ",i)
            print("Configuring RX Buffers")
            cclo[i].setup_rx_buffers(nbufs, bufsize, devicemem)
            print("Configuring a communicator")
            cclo[i].configure_communicator(ranks, i, vnx=vnx)
    else:
        print("Skipping RX/Comm init on user request")

    print("Accelerator ready!")

    return ol, cclo, devicemem


def configure_vnx_ip(overlay, our_ip):
    print("Link interface 1 {}".format(ol.cmac_inst.linkStatus()))
    print(ol.networklayer_inst.updateIPAddress(our_ip, debug=True))

def configure_vnx_socket(overlay, their_rank, our_port, their_ip, their_port):
    # populate socket table with tuples of remote ip, remote port, local port 
    # up to 16 entries possible in VNx
    ol.networklayer_inst.sockets[their_rank] = (their_ip, their_port, our_port, True)
    ol.networklayer_inst.populateSocketTable(debug=True)

def configure_vnx(overlay, localrank, ranks):
    assert len(ranks) <= 16, "Too many ranks. VNX supports up to 16 sockets"
    for i in range(len(ranks)):
        if i == localrank:
            configure_vnx_ip(overlay, ranks[i]["ip"])
        else:
            configure_vnx_socket(overlay, i, ranks[localrank]["port"], ranks[i]["ip"], ranks[i]["port"])


parser = argparse.ArgumentParser(description='Tests for MPI collectives offload with UDP (VNx) backend')
parser.add_argument('--xclbin', type=str, default=None, help='Accelerator image file (xclbin)', required=True)
parser.add_argument('--device_index', type=int, default=1, help='Card index')
parser.add_argument('--nruns', type=int, default=1, help='How many times to run each test')
parser.add_argument('--bsize', type=int, default=1, help='How many KB per buffer')
parser.add_argument('--local_rank', type=int, help='Index of local rank', required=True)
parser.add_argument('--use_vnx', action='store_true', default=False, help='Configure and use the UDP stacl')

args = parser.parse_args()

#two ranks, same port on each side
ranks = []
ranks.append({"ip": "192.168.30.1", "port": 12345})
ranks.append({"ip": "192.168.30.2", "port": 12345})

#configure FPGA and CCLO cores with the default 16 RX buffers of 16KB each
ol, cclo, devicemem = configure_xccl(args.xclbin, args.device_index, ranks, vnx=args.use_vnx)

import pdb; pdb.set_trace()

if args.use_vnx:
    #configure VNx UDP stack
    configure_vnx(ol, args.local_rank, ranks)

import pdb; pdb.set_trace()

# run ARP discovery; this has to happen after both sides have configured their IPs
ol.networklayer_inst.arpDiscovery()
ol.networklayer_inst.readARPTable()

import pdb; pdb.set_trace()

tx_buf = [pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem)]
tx_buf.append(pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem))
tx_buf.append(pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem))
tx_buf.append(pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem))

rx_buf = [pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem)]
rx_buf.append(pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem))
rx_buf.append(pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem))
rx_buf.append(pynq.allocate((args.bsize*1024,), dtype=np.int8, target=devicemem))

if args.dump_rx_regs >= 0 and args.dump_rx_regs < 4:
    cclo[args.dump_rx_regs].dump_rx_buffers(nbufs=16)

def test_sendrecv():
    # test sending from each cclo to each other cclo
    # scenario 1: send immediately followed by recv
    queues = [[Queue() for i in range(4)] for j in range(4)]
    print("========================================")
    print("Send/Recv Scenario 1")
    print("========================================")
    for i in range(4):
        for j in range(3):
            senddata = random.randint(10,100)
            queues[i][j].put(senddata)
            src_rank = i
            dst_rank = (i+j+1)%4
            tag = i+5+10*j
            tx_buf[src_rank][0]=senddata
            cclo[src_rank].send(0, tx_buf[src_rank], dst_rank, tag)
            exp_recvdata = queues[i][j].get()
            cclo[dst_rank].recv(0, rx_buf[dst_rank], src_rank, tag)
            recvdata = rx_buf[dst_rank][0]
            if recvdata == exp_recvdata:
                print("Send/Recv {} -> {} succeeded".format(src_rank, dst_rank))
            else:
                print("Send/Recv {} -> {} failed, expected {} got {}".format(src_rank, dst_rank, exp_recvdata, recvdata))
    # scenario 2: for each instance, send multiple, then recv multiple at the other instances
    print("========================================")
    print("Send/Recv Scenario 2")
    print("========================================")
    for i in range(4):
        for j in range(3):
            senddata = random.randint(10,100)
            queues[i][j].put(senddata)
            src_rank = i
            dst_rank = (i+j+1)%4
            tag = i+5+10*j
            tx_buf[src_rank][0]=senddata
            cclo[src_rank].send(0, tx_buf[src_rank], dst_rank, tag)
        for j in range(3):
            src_rank = i
            dst_rank = (i+j+1)%4
            tag = i+5+10*j
            exp_recvdata = queues[i][j].get()
            cclo[dst_rank].recv(0, rx_buf[dst_rank], src_rank, tag)
            recvdata = rx_buf[dst_rank][0]
            if recvdata == exp_recvdata:
                print("Send/Recv {} -> {} succeeded".format(src_rank, dst_rank))
            else:
                print("Send/Recv {} -> {} failed, expected {} got {}".format(src_rank, dst_rank, exp_recvdata, recvdata))
    # scenario 2: send everything, recv everything
    print("========================================")
    print("Send/Recv Scenario 3")
    print("========================================")
    for i in range(4):
        for j in range(3):
            senddata = random.randint(10,100)
            queues[i][j].put(senddata)
            src_rank = i
            dst_rank = (i+j+1)%4
            tag = i+5+10*j
            tx_buf[src_rank][0]=senddata
            cclo[src_rank].send(0, tx_buf[src_rank], dst_rank, tag)
    for i in range(4):
        for j in range(3):
            src_rank = i
            dst_rank = (i+j+1)%4
            tag = i+5+10*j
            exp_recvdata = queues[i][j].get()
            cclo[dst_rank].recv(0, rx_buf[dst_rank], src_rank, tag)
            recvdata = rx_buf[dst_rank][0]
            if recvdata == exp_recvdata:
                print("Send/Recv {} -> {} succeeded".format(src_rank, dst_rank))
            else:
                print("Send/Recv {} -> {} failed, expected {} got {}".format(src_rank, dst_rank, exp_recvdata, recvdata))

def test_bcast():
    # test broadcast from each rank
    print("========================================")
    print("Broadcast")
    print("========================================")
    queues = [[Queue() for i in range(4)] for j in range(4)]
    for i in range(4):
        err_count = 0
        tx_buf[i][0]=random.randint(10,100)
        threads = []
        for j in range(4):
            buf = tx_buf if (j==i) else rx_buf
            threads.append(threading.Thread(target=cclo[j].bcast, args=(0, buf[j], i)))
            threads[-1].start()
        for j in range(4):
            threads[j].join()
            buf = tx_buf if (j==i) else rx_buf
            if buf[j][0] != tx_buf[i][0]:
                err_count += 1
                print("Bcast {} -> {} failed, expected {} got {}".format(i, j, tx_buf[i][0], buf[j][0]))
        if err_count == 0:
            print("Bcast {} -> all succeeded".format(i))

def test_scatter():
    # test scatter from each rank
    print("========================================")
    print("Scatter")
    print("========================================")
    queues = [[Queue() for i in range(4)] for j in range(4)]
    for i in range(4):
        err_count = 0
        count = len(tx_buf[i])//4
        tx_buf[i][:]=np.random.randint(1000, size=tx_buf[i].shape)
        threads = []
        for j in range(4):
            rx_buf[j][:]=np.zeros(rx_buf[j].shape)# clear rx buffers
            threads.append(threading.Thread(target=cclo[j].scatter, args=(0, tx_buf[j], rx_buf[j], count, i)))
            threads[-1].start()
        for j in range(4):
            threads[j].join()
        for j in range(4):
            if not (rx_buf[j][0:count]==tx_buf[i][count*j:count*(j+1)]).all():
                err_count += 1
                print("Scatter {} -> {} failed")
                import pdb; pdb.set_trace()
        if err_count == 0:
            print("Scatter {} -> all succeeded".format(i))

def test_gather():
    # test gather from each rank
    print("========================================")
    print("Gather")
    print("========================================")
    queues = [[Queue() for i in range(4)] for j in range(4)]
    for i in range(4):
        err_count = 0
        count = len(tx_buf[i])//4
        rx_buf[i][:]=np.zeros(rx_buf[i].shape)# clear rx buffer
        threads = []
        for j in range(4):
            tx_buf[j][:]=np.random.randint(1000, size=tx_buf[j].shape)#init tx buffers
            threads.append(threading.Thread(target=cclo[j].gather, args=(0, tx_buf[j], rx_buf[j], count, i)))
            threads[-1].start()
        for j in range(4):
            threads[j].join()
        for j in range(4):
            if not (rx_buf[i][count*j:count*(j+1)]==tx_buf[j][0:count]).all():
                err_count += 1
                print("Gather {} <- {} failed".format(i, j))
                import pdb; pdb.set_trace()
        if err_count == 0:
            print("Gather {} <- all succeeded".format(i))

def test_allgather():
    # test gather from each rank
    print("========================================")
    print("AllGather")
    print("========================================")
    queues = [[Queue() for i in range(4)] for j in range(4)]

    err_count = 0
    count = len(tx_buf[i])//4
    rx_buf[i][:]=np.zeros(rx_buf[i].shape)# clear rx buffer
    threads = []
    for j in range(4):
        tx_buf[j][:]=np.random.randint(1000, size=tx_buf[j].shape)#init tx buffers
        threads.append(threading.Thread(target=cclo[j].allgather, args=(0, tx_buf[j], rx_buf[j], count)))
        threads[-1].start()
    for j in range(4):
        threads[j].join()
    for j in range(4):
        if not (rx_buf[i][count*j:count*(j+1)]==tx_buf[j][0:count]).all():
            err_count += 1
            print("AllGather failed on {}".format(j))
            import pdb; pdb.set_trace()
    if err_count == 0:
        print("AllGather succeeded")

if args.sendrecv:
    for i in range(args.nruns):
        test_sendrecv()

if args.bcast:
    for i in range(args.nruns):
        test_bcast()
        
if args.scatter:
    for i in range(args.nruns):
        test_scatter()

if args.gather:
    for i in range(args.nruns):
        test_gather()
