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
import time
sys.path.append('../../driver/pynq/')
from accl import accl, ACCLReduceFunctions, ACCLStreamFlags
from accl import SimBuffer
import argparse
import itertools
import math
from mpi4py import MPI

def get_buffers(count, op0_dt, op1_dt, res_dt, accl_inst):
    op0_buf = SimBuffer(np.zeros((count,), dtype=op0_dt), accl_inst.cclo.socket)
    op1_buf = SimBuffer(np.zeros((count,), dtype=op1_dt), accl_inst.cclo.socket)
    res_buf = SimBuffer(np.zeros((count,), dtype=res_dt), accl_inst.cclo.socket)
    op0_buf.sync_to_device()
    op1_buf.sync_to_device()
    res_buf.sync_to_device()
    op0_buf.buf[:] = np.random.randn(count).astype(op0_dt)
    op1_buf.buf[:] = np.random.randn(count).astype(op1_dt)
    return op0_buf, op1_buf, res_buf

def test_copy(cclo_inst, count):
    err_count = 0
    dt = [np.float32]#[np.float32, np.half, np.float64]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        cclo_inst.copy(op_buf, res_buf, count)
        if not np.isclose(op_buf.buf, res_buf.buf).all():
            err_count += 1
            print("Copy failed on pair ", op_dt, res_dt)
        else:
            print("Copy succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Copy succeeded")

def test_combine(cclo_inst, count):
    err_count = 0
    dt = [np.float32]#[np.float32, np.half]
    for op0_dt, op1_dt, res_dt in itertools.product(dt, repeat=3):
        op0_buf, op1_buf, res_buf = get_buffers(count, op0_dt, op1_dt, res_dt, cclo_inst)
        cclo_inst.combine(count, ACCLReduceFunctions.SUM, op0_buf, op1_buf, res_buf)
        if not np.isclose(op0_buf.buf+op1_buf.buf, res_buf.buf).all():
            err_count += 1
            print("Combine failed on pair ", op0_dt, op1_dt, res_dt)
        else:
            print("Combine succeeded on pair ", op0_dt, op1_dt, res_dt)
    if err_count == 0:
        print("Combine succeeded")

def test_sendrecv(cclo_inst, world_size, local_rank, count):
    err_count = 0
    dt = [np.float32]#[np.float32, np.half]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        # send to next rank; receive from previous rank; send back data to previous rank; receive from next rank; compare
        next_rank = (local_rank+1)%world_size
        prev_rank = (local_rank+world_size-1)%world_size
        print("Sending on ",local_rank," to ",next_rank)
        cclo_inst.send(0, op_buf, count, next_rank, tag=0)
        print("Receiving on ",local_rank," from ",prev_rank)
        cclo_inst.recv(0, res_buf, count, prev_rank, tag=0)
        print("Sending on ",local_rank," to ",prev_rank)
        cclo_inst.send(0, res_buf, count, prev_rank, tag=1)
        print("Receiving on ",local_rank," from ",next_rank)
        cclo_inst.recv(0, res_buf, count, next_rank, tag=1)
        if not np.isclose(op_buf.buf, res_buf.buf).all():
            err_count += 1
            print("Send/recv failed on pair ", op_dt, res_dt)
        else:
            print("Send/recv succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Send/recv succeeded")

def test_sendrecv_plkernel(cclo_inst, world_size, local_rank, count):
    #NOTE: this requires loopback on the external stream interface
    err_count = 0
    dt = [np.float32]#[np.float32, np.half]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        # send to next rank; receive from previous rank; send back data to previous rank; receive from next rank; compare
        next_rank = (local_rank+1)%world_size
        prev_rank = (local_rank+world_size-1)%world_size
        print("Sending from memory on ",local_rank," to stream on ",next_rank)
        cclo_inst.send(0, op_buf, count, next_rank, stream_flags=ACCLStreamFlags.RES_STREAM, tag=0)
        # recv is direct, no call required
        print("Sending from stream on ",local_rank," to memory on ",prev_rank)
        cclo_inst.send(0, res_buf, count, prev_rank, stream_flags=ACCLStreamFlags.OP0_STREAM, tag=5)
        print("Receiving in memory on ",local_rank," from stream on ",next_rank)
        cclo_inst.recv(0, res_buf, count, next_rank, tag=5)
        if not np.isclose(op_buf.buf, res_buf.buf).all():
            err_count += 1
            print("Send/recv failed on pair ", op_dt, res_dt)
        else:
            print("Send/recv succeeded on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Send/recv succeeded")

def test_sendrecv_fanin(cclo_inst, world_size, local_rank, count):
    err_count = 0
    dt = [np.float32]#[np.float32, np.half]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        # send to next rank; receive from previous rank; send back data to previous rank; receive from next rank; compare
        if local_rank != 0:
            for i in range(len(op_buf.buf)):
                op_buf.buf[i] = i+local_rank
            print("Sending on ", local_rank, " to 0")
            cclo_inst.send(0, op_buf, count, 0, tag=0)
        else:
            for i in range(world_size):
                if i == local_rank:
                    continue
                print("Receiving on 0 from ", i)
                cclo_inst.recv(0, res_buf, count, i, tag=0)
                for j in range(len(op_buf.buf)):
                    op_buf.buf[j] = j+i
                if not np.isclose(op_buf.buf, res_buf.buf).all():
                    err_count += 1
                    print("Fan-in send/recv failed for sender rank", i)
                    print(op_buf.buf)
                    print(res_buf.buf)
                else:
                    print("Fan-in send/recv succeeded for sender rank ", i)
    if err_count == 0:
        print("Fan-in send/recv succeeded")

def test_bcast(cclo_inst, local_rank, root, count):
    err_count = 0
    dt = [np.float32]#[np.float32, np.half]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count, op_dt, op_dt, res_dt, cclo_inst)
        op_buf.buf[:] = [42+i for i in range(len(op_buf.buf))]
        cclo_inst.bcast(0, op_buf if root == local_rank else res_buf, count, root=root)

        if local_rank == root:
            print("Bcast succeeded on pair ", op_dt, res_dt)
        else:
            if not np.isclose(op_buf.buf, res_buf.buf).all():
                err_count += 1
                print("Bcast failed on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Bcast succeeded")

def test_scatter(cclo_inst, world_size, local_rank, root, count):
    err_count = 0
    dt = [np.float32]#[np.float32, np.half]
    for op_dt, res_dt in itertools.product(dt, repeat=2):
        op_buf, _, res_buf = get_buffers(count*world_size, op_dt, op_dt, res_dt, cclo_inst)
        op_buf[:] = [1.0*i for i in range(op_buf.size)]
        cclo_inst.scatter(0, op_buf, res_buf, count, root=root)

        if not np.isclose(op_buf.buf[local_rank*count:(local_rank+1)*count], res_buf.buf[0:count]).all():
            err_count += 1
            print("Scatter failed on pair ", op_dt, res_dt)
    if err_count == 0:
        print("Scatter succeeded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests for ACCL (emulation mode)')
    parser.add_argument('--nruns',      type=int,            default=1,     help='How many times to run each test')
    parser.add_argument('--start_port', type=int,            default=5500,  help='Start of range of ports usable for sim')
    parser.add_argument('--count',      type=int,            default=16,    help='How many B per buffer')
    parser.add_argument('--rxbuf_size', type=int,            default=1,     help='How many KB per RX buffer')
    parser.add_argument('--debug',      action='store_true', default=False, help='enable debug mode')
    parser.add_argument('--all',        action='store_true', default=False, help='Select all collectives')
    parser.add_argument('--nop',        action='store_true', default=False, help='Run nop test')
    parser.add_argument('--combine',    action='store_true', default=False, help='Run fp/dp/i32/i64 test')
    parser.add_argument('--copy',       action='store_true', default=False, help='Run copy test')
    parser.add_argument('--sndrcv',     action='store_true', default=False, help='Run send/receive test')
    parser.add_argument('--sndrcv_strm', action='store_true', default=False, help='Run send/receive stream test')
    parser.add_argument('--sndrcv_fanin', action='store_true', default=False, help='Run send/receive fan-in test')
    parser.add_argument('--bcast',      action='store_true', default=False, help='Run bcast test')
    parser.add_argument('--scatter',    action='store_true', default=False, help='Run scatter test')
    parser.add_argument('--tcp',        action='store_true', default=False, help='Run test using TCP')

    args = parser.parse_args()
    args.rxbuf_size = 1024*args.rxbuf_size #convert from KB to B
    if args.all:
        args.sndrcv  = True
        args.combine = True
        args.copy    = True
        args.bcast   = True
        args.scatter = True
        args.sndrcv_strm = True

    # get communicator size and our local rank in it
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    local_rank = comm.Get_rank()

    #set a random seed to make it reproducible
    np.random.seed(2021+local_rank)

    ranks = []
    for i in range(world_size):
        ranks.append({"ip": "127.0.0.1", "port": args.start_port+world_size+i, "session_id":i, "max_segment_size": args.rxbuf_size})

    #configure FPGA and CCLO cores with the default 16 RX buffers of size given by args.rxbuf_size
    cclo_inst = accl(ranks, local_rank, bufsize=args.rxbuf_size, protocol=("TCP" if args.tcp else "UDP"), sim_sock="tcp://localhost:"+str(args.start_port+local_rank))
    cclo_inst.set_timeout(10**8)
    #barrier here to make sure all the devices are configured before testing
    comm.barrier()

    try:
        for i in range(args.nruns):
            if args.nop:
                cclo_inst.nop()
            if args.combine:
                test_combine(cclo_inst, args.count)
            if args.copy:
                test_copy(cclo_inst, args.count)
            if args.sndrcv:
                test_sendrecv(cclo_inst, world_size, local_rank, args.count)
            if args.sndrcv_strm:
                test_sendrecv_plkernel(cclo_inst, world_size, local_rank, args.count)
            if args.sndrcv_fanin:
                test_sendrecv_fanin(cclo_inst, world_size, local_rank, args.count)
            if args.bcast:
                test_bcast(cclo_inst, local_rank, i, args.count)
            if args.scatter:
                test_scatter(cclo_inst, world_size, local_rank, i, args.count)

    except KeyboardInterrupt:
        print("CTR^C")
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)
        cclo_inst.dump_rx_buffers()

    cclo_inst.deinit()
